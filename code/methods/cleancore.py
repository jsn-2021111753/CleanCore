"""CleanCore: stage-wise cleaning and coreset training with one continuous MLP."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import fields
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from common.interfaces import MethodOutput
from common.model import MLP
from common.seed import set_seed
from common.training import ArrayDataset, TrainingConfig, predict_classifier
from methods.base import MethodContext
from methods.common import bounded_classwise_candidates, one_hot, subset_size_from_fraction
from methods.torch_utils import input_gradients


def _training_config_from_context(ctx: MethodContext) -> TrainingConfig:
    names = {f.name for f in fields(TrainingConfig)}
    data = {k: v for k, v in ctx.training_config.items() if k in names}
    cfg = TrainingConfig(**data)
    cfg.save_best_model = bool(ctx.training_config.get("save_best_model", False))
    cfg.save_last_model = bool(ctx.training_config.get("save_last_model", True))
    return cfg


def _build_model(ctx: MethodContext) -> MLP:
    set_seed(ctx.seed + 800)
    return MLP(
        input_dim=ctx.input_dim,
        num_classes=ctx.num_classes,
        hidden_dims=tuple(ctx.model_config.get("hidden_dims", (256, 128))),
        dropout=float(ctx.model_config.get("dropout", 0.2)),
        batch_norm=bool(ctx.model_config.get("batch_norm", True)),
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _copy_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state_dict_cpu(model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict(state, strict=True)
    model.to(device)


def _make_loader(
    X: np.ndarray,
    y_hard: np.ndarray,
    y_prob: np.ndarray,
    sample_weights: np.ndarray,
    cfg: TrainingConfig,
    seed: int,
) -> DataLoader:
    dataset = ArrayDataset(
        np.asarray(X, dtype=np.float32),
        np.asarray(y_hard, dtype=np.int64),
        sample_weights=np.asarray(sample_weights, dtype=np.float32),
        soft_targets=np.asarray(y_prob, dtype=np.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        generator=generator,
    )


def _train_one_epoch_weighted_soft(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    total_samples = 0

    dataset = loader.dataset
    sample_weights = dataset.sample_weights.to(device) if dataset.sample_weights is not None else None
    soft_targets = dataset.soft_targets.to(device) if dataset.soft_targets is not None else None
    if sample_weights is None or soft_targets is None:
        raise ValueError("CleanCore training requires sample weights and soft targets.")

    for x, _y, local_idx in loader:
        x = x.to(device)
        local_idx = local_idx.to(device)
        q = soft_targets[local_idx]
        w = sample_weights[local_idx]

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss_vec = -(q * torch.log_softmax(logits, dim=1)).sum(dim=1)
        loss = (loss_vec * w).sum() / w.sum().clamp_min(1e-12)
        loss.backward()
        optimizer.step()

        total_loss += float((loss_vec.detach() * w.detach()).sum().cpu())
        total_weight += float(w.detach().sum().cpu())
        total_samples += int(len(x))

    return total_loss / max(total_weight, 1e-12), total_samples


@torch.no_grad()
def _per_sample_soft_losses(
    model: nn.Module,
    X: np.ndarray,
    y_prob: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    losses: list[np.ndarray] = []
    for start in range(0, len(X), int(batch_size)):
        end = min(len(X), start + int(batch_size))
        xb = torch.tensor(np.asarray(X[start:end], dtype=np.float32), device=device)
        qb = torch.tensor(np.asarray(y_prob[start:end], dtype=np.float32), device=device)
        logits = model(xb)
        loss_vec = -(qb * torch.log_softmax(logits, dim=1)).sum(dim=1)
        losses.append(loss_vec.cpu().numpy().astype(np.float32))
    return np.concatenate(losses, axis=0) if losses else np.zeros(0, dtype=np.float32)


def _loss_mean_std_over_snapshots(
    model: nn.Module,
    snapshots: List[Dict[str, torch.Tensor]],
    X: np.ndarray,
    y_prob: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if not snapshots:
        snapshots = [_copy_state_dict_cpu(model)]
    current = _copy_state_dict_cpu(model)
    all_losses = []
    try:
        for state in snapshots:
            _load_state_dict_cpu(model, state, device)
            all_losses.append(_per_sample_soft_losses(model, X, y_prob, batch_size=batch_size, device=device))
    finally:
        _load_state_dict_cpu(model, current, device)
    stacked = np.stack(all_losses, axis=1).astype(np.float32)
    return stacked.mean(axis=1), stacked.std(axis=1)


@torch.no_grad()
def _loss_by_class_over_snapshots(
    model: nn.Module,
    snapshots: List[Dict[str, torch.Tensor]],
    X: np.ndarray,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0, int(num_classes)), dtype=np.float32)
    if not snapshots:
        snapshots = [_copy_state_dict_cpu(model)]
    current = _copy_state_dict_cpu(model)
    out = np.zeros((len(X), int(num_classes)), dtype=np.float64)
    try:
        for state in snapshots:
            _load_state_dict_cpu(model, state, device)
            model.eval()
            rows = []
            for start in range(0, len(X), int(batch_size)):
                xb = torch.tensor(np.asarray(X[start : start + int(batch_size)], dtype=np.float32), device=device)
                logits = model(xb)
                rows.append((-torch.log_softmax(logits, dim=1)).cpu().numpy())
            out += np.concatenate(rows, axis=0)
    finally:
        _load_state_dict_cpu(model, current, device)
    return (out / float(len(snapshots))).astype(np.float32)


@torch.no_grad()
def _base_grad_representation(
    model: nn.Module,
    X: np.ndarray,
    y_prob: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    reps: list[np.ndarray] = []
    for start in range(0, len(X), int(batch_size)):
        end = min(len(X), start + int(batch_size))
        xb = torch.tensor(np.asarray(X[start:end], dtype=np.float32), device=device)
        qb = np.asarray(y_prob[start:end], dtype=np.float32)
        probs = torch.softmax(model(xb), dim=1).cpu().numpy().astype(np.float32)
        probs = np.nan_to_num(probs, nan=1.0 / max(1, y_prob.shape[1]), posinf=1.0, neginf=0.0)
        reps.append(np.clip(probs - qb, -1.0, 1.0).astype(np.float32))
    return np.concatenate(reps, axis=0) if reps else np.zeros((0, y_prob.shape[1]), dtype=np.float32)


def _finite_gradient_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.nan_to_num(np.asarray(vectors, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(vectors, -1.0, 1.0).astype(np.float32)


def _weighted_candidates_from_fraction(active_idx: np.ndarray, weights: np.ndarray, fraction: float, n_samples: int) -> np.ndarray:
    top_fraction = min(1.0, max(0.0, float(fraction)))
    n_top = max(1, int(np.ceil(len(active_idx) * top_fraction)))
    n_top = min(n_top, len(active_idx))
    order = np.argsort(-weights[active_idx], kind="mergesort")[:n_top]
    mask = np.zeros(int(n_samples), dtype=bool)
    mask[active_idx[order]] = True
    return mask


class _LabelState:
    __slots__ = ("p", "dom_hist")

    def __init__(self, num_classes: int, window: int):
        self.p = np.zeros((int(num_classes),), dtype=np.float32)
        self.dom_hist: Deque[int] = deque(maxlen=max(1, int(window)))


def _label_state_is_stable(
    state: _LabelState,
    dom: int,
    min_window: int,
    consistency_thresh: float,
) -> bool:
    hist = list(state.dom_hist)
    min_window = max(1, int(min_window))
    if len(hist) < min_window:
        return False
    freq = float(sum(1 for x in hist if int(x) == int(dom)) / max(1, len(hist)))
    return freq >= float(consistency_thresh)


def _feature_repair_coefficients(
    grad_core: np.ndarray,
    prev_scores: np.ndarray,
    prev_sign: Optional[np.ndarray],
    smooth: float,
    coef_min: float,
    coef_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    grad_core = np.asarray(grad_core, dtype=np.float32)
    prev_scores = np.asarray(prev_scores, dtype=np.float32)
    abs_g = np.abs(grad_core)
    rel = abs_g / max(float(abs_g.sum()), 1e-12)
    sign = np.sign(grad_core).astype(np.int8)
    if prev_sign is None:
        stable = np.ones_like(rel, dtype=np.float32)
    else:
        stable = (sign == np.asarray(prev_sign, dtype=np.int8)).astype(np.float32)
    smooth = float(np.clip(smooth, 0.0, 1.0))
    scores = smooth * prev_scores + (1.0 - smooth) * (rel * stable)
    scores = np.clip(scores, 0.0, 1.0).astype(np.float32)
    coeffs = float(coef_min) + (float(coef_max) - float(coef_min)) * scores
    return coeffs.astype(np.float32), scores


class _GreedyKDNode:
    __slots__ = ("positions", "split_dim", "split_value", "left", "right")

    def __init__(
        self,
        positions: Optional[np.ndarray] = None,
        split_dim: int = -1,
        split_value: float = 0.0,
        left: Optional["_GreedyKDNode"] = None,
        right: Optional["_GreedyKDNode"] = None,
    ):
        self.positions = positions
        self.split_dim = int(split_dim)
        self.split_value = float(split_value)
        self.left = left
        self.right = right


def _build_greedy_kd_tree(centers: np.ndarray, positions: np.ndarray, leaf_size: int) -> _GreedyKDNode:
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) <= max(1, int(leaf_size)):
        return _GreedyKDNode(positions=positions.astype(np.int64))

    pts = centers[positions]
    var = np.var(pts, axis=0)
    split_dim = int(np.argmax(var))
    if float(var[split_dim]) <= 1e-12:
        return _GreedyKDNode(positions=positions.astype(np.int64))

    order = np.argsort(pts[:, split_dim], kind="mergesort")
    mid = len(order) // 2
    if mid <= 0 or mid >= len(order):
        return _GreedyKDNode(positions=positions.astype(np.int64))

    left_pos = positions[order[:mid]]
    right_pos = positions[order[mid:]]
    split_value = float(centers[right_pos[0], split_dim])
    return _GreedyKDNode(
        positions=None,
        split_dim=split_dim,
        split_value=split_value,
        left=_build_greedy_kd_tree(centers, left_pos, leaf_size),
        right=_build_greedy_kd_tree(centers, right_pos, leaf_size),
    )


def _query_greedy_kd_tree(
    node: _GreedyKDNode,
    centers: np.ndarray,
    queries: np.ndarray,
    query_pos: np.ndarray,
    out: np.ndarray,
) -> None:
    if len(query_pos) == 0:
        return
    if node.positions is not None:
        leaf_pos = node.positions
        leaf_centers = centers[leaf_pos]
        block = queries[query_pos]
        block_norm = np.sum(block * block, axis=1, keepdims=True)
        center_norm = np.sum(leaf_centers * leaf_centers, axis=1).reshape(1, -1)
        dots = np.einsum("bd,kd->bk", block, leaf_centers, optimize=True)
        dist2 = block_norm + center_norm - 2.0 * dots
        dist2 = np.nan_to_num(dist2, nan=np.inf, posinf=np.inf, neginf=0.0)
        out[query_pos] = leaf_pos[np.argmin(dist2, axis=1)]
        return

    split_values = queries[query_pos, node.split_dim]
    left_mask = split_values <= node.split_value
    if node.left is not None:
        _query_greedy_kd_tree(node.left, centers, queries, query_pos[left_mask], out)
    if node.right is not None:
        _query_greedy_kd_tree(node.right, centers, queries, query_pos[~left_mask], out)


def _assign_to_subset(
    vectors: np.ndarray,
    subset_idx: np.ndarray,
    alive_mask: np.ndarray,
    chunk_size: int = 5000,
    backend: str = "brute_force",
    leaf_size: int = 64,
) -> np.ndarray:
    vectors = _finite_gradient_vectors(vectors)
    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    alive_mask = np.asarray(alive_mask, dtype=bool)
    mapping = np.full((len(vectors),), -1, dtype=np.int64)
    if len(subset_idx) == 0:
        return mapping
    backend = str(backend).lower()
    if backend == "brute":
        backend = "brute_force"
    if backend not in {"brute_force", "kd_tree", "ball_tree", "greedy_kd_tree"}:
        raise ValueError(f"Unknown subset_mapping_backend: {backend}")

    centers = vectors[subset_idx]
    centers64 = centers.astype(np.float64)
    alive_idx = np.where(alive_mask)[0].astype(np.int64)
    if len(alive_idx) == 0:
        return mapping

    if backend in {"kd_tree", "ball_tree"}:
        try:
            from sklearn.neighbors import BallTree, KDTree
        except Exception as exc:  # pragma: no cover - depends on optional environment
            raise ImportError(f"subset_mapping_backend={backend} requires scikit-learn.") from exc

        tree_cls = KDTree if backend == "kd_tree" else BallTree
        tree = tree_cls(centers64, leaf_size=max(1, int(leaf_size)), metric="euclidean")
        for start in range(0, len(alive_idx), int(chunk_size)):
            idx = alive_idx[start : start + int(chunk_size)]
            block = vectors[idx].astype(np.float64)
            _dist, nearest = tree.query(block, k=1)
            mapping[idx] = nearest.reshape(-1).astype(np.int64)
        return mapping

    if backend == "greedy_kd_tree":
        root = _build_greedy_kd_tree(
            centers64,
            np.arange(len(subset_idx), dtype=np.int64),
            leaf_size=max(1, int(leaf_size)),
        )
        for start in range(0, len(alive_idx), int(chunk_size)):
            idx = alive_idx[start : start + int(chunk_size)]
            block = vectors[idx].astype(np.float64)
            nearest = np.full((len(block),), -1, dtype=np.int64)
            _query_greedy_kd_tree(root, centers64, block, np.arange(len(block), dtype=np.int64), nearest)
            mapping[idx] = nearest.astype(np.int64)
        return mapping

    center_norm = np.sum(centers64 * centers64, axis=1)
    for start in range(0, len(alive_idx), int(chunk_size)):
        idx = alive_idx[start : start + int(chunk_size)]
        block = vectors[idx].astype(np.float64)
        block_norm = np.sum(block * block, axis=1, keepdims=True)
        dots = np.einsum("bd,kd->bk", block, centers64, optimize=True)
        dist2 = block_norm + center_norm.reshape(1, -1) - 2.0 * dots
        dist2 = np.nan_to_num(dist2, nan=np.inf, posinf=np.inf, neginf=0.0)
        mapping[idx] = np.argmin(dist2, axis=1).astype(np.int64)
    return mapping


def _compute_subset_weights_from_mapping(
    weights: np.ndarray,
    subset_idx: np.ndarray,
    mapping: np.ndarray,
    alive_mask: np.ndarray,
) -> np.ndarray:
    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    alpha = np.zeros((len(subset_idx),), dtype=np.float32)
    alive_idx = np.where(np.asarray(alive_mask, dtype=bool))[0]
    for gi in alive_idx.tolist():
        j = int(mapping[gi])
        if 0 <= j < len(alpha):
            alpha[j] += float(weights[gi])
    return alpha


def _compute_subset_gradient_error(
    base_grad: np.ndarray,
    weights: np.ndarray,
    subset_idx: np.ndarray,
    subset_alpha: np.ndarray,
    alive_mask: np.ndarray,
) -> tuple[float, float]:
    base_grad = _finite_gradient_vectors(base_grad)
    weights = np.asarray(weights, dtype=np.float32)
    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    subset_alpha = np.asarray(subset_alpha, dtype=np.float32)
    alive_idx = np.where(np.asarray(alive_mask, dtype=bool))[0]
    if len(alive_idx) == 0:
        return 0.0, 0.0
    full_grad = (weights[alive_idx, None] * base_grad[alive_idx]).sum(axis=0).astype(np.float64)
    subset_grad = np.zeros_like(full_grad)
    for pos, gi in enumerate(subset_idx.tolist()):
        if pos < len(subset_alpha):
            subset_grad += float(subset_alpha[pos]) * base_grad[gi].astype(np.float64)
    grad_error = float(np.linalg.norm(full_grad - subset_grad) / (np.linalg.norm(full_grad) + 1e-12))
    full_weight = float(weights[alive_idx].sum())
    subset_weight = float(subset_alpha.sum())
    weight_error = float(abs(full_weight - subset_weight) / (abs(full_weight) + 1e-12))
    return grad_error, weight_error


def _build_subset_weighted_coverage(
    base_grad: np.ndarray,
    weights: np.ndarray,
    alive_mask: np.ndarray,
    candidate_mask: np.ndarray,
    subset_size: int,
    seed: int,
    candidate_chunk_size: int = 512,
) -> np.ndarray:
    base_grad = _finite_gradient_vectors(base_grad)
    weights = np.asarray(weights, dtype=np.float32)
    alive_mask = np.asarray(alive_mask, dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool) & alive_mask
    alive_idx = np.where(alive_mask)[0].astype(np.int64)
    candidate_idx = np.where(candidate_mask)[0].astype(np.int64)
    if len(alive_idx) == 0:
        return np.array([], dtype=np.int64)
    if len(candidate_idx) == 0:
        candidate_idx = alive_idx.copy()
    subset_size = max(1, min(int(subset_size), len(candidate_idx)))
    if subset_size >= len(candidate_idx):
        return candidate_idx.astype(np.int64)

    # The paper objective is weighted gradient coverage: each greedy step picks
    # the candidate that most reduces weighted nearest-representative distance.
    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(0.0, 1e-12, size=len(candidate_idx))
    alive_vec = base_grad[alive_idx].astype(np.float64)
    alive_w = np.clip(weights[alive_idx].astype(np.float64), 0.0, None)
    selected_positions: list[int] = []
    available = np.ones((len(candidate_idx),), dtype=bool)
    nearest_d2: Optional[np.ndarray] = None
    chunk_size = max(1, int(candidate_chunk_size))

    for _ in range(subset_size):
        best_pos = -1
        best_score = -np.inf
        available_pos = np.where(available)[0]
        for start in range(0, len(available_pos), chunk_size):
            pos = available_pos[start : start + chunk_size]
            cand_vec = base_grad[candidate_idx[pos]].astype(np.float64)
            av_norm = np.sum(alive_vec * alive_vec, axis=1, keepdims=True)
            cand_norm = np.sum(cand_vec * cand_vec, axis=1).reshape(1, -1)
            dots = np.einsum("bd,kd->bk", alive_vec, cand_vec, optimize=True)
            dist2 = av_norm + cand_norm - 2.0 * dots
            dist2 = np.maximum(dist2.astype(np.float64), 0.0)
            if nearest_d2 is None:
                scores = -(alive_w[:, None] * dist2).sum(axis=0)
            else:
                gain = np.maximum(nearest_d2[:, None] - dist2, 0.0)
                scores = (alive_w[:, None] * gain).sum(axis=0)
            scores = scores + jitter[pos]
            local_best = int(np.argmax(scores))
            if float(scores[local_best]) > best_score:
                best_score = float(scores[local_best])
                best_pos = int(pos[local_best])
        if best_pos < 0:
            break
        selected_positions.append(best_pos)
        available[best_pos] = False
        chosen_vec = base_grad[candidate_idx[best_pos]].reshape(1, -1)
        d2 = np.sum((alive_vec - chosen_vec) ** 2, axis=1).astype(np.float64)
        nearest_d2 = d2 if nearest_d2 is None else np.minimum(nearest_d2, d2)

    return candidate_idx[np.array(selected_positions, dtype=np.int64)].astype(np.int64)


def _nearest_to_block(
    base_grad: np.ndarray,
    alive_idx: np.ndarray,
    block_idx: np.ndarray,
    chunk_size: int = 5000,
    backend: str = "brute_force",
    leaf_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    base_grad = _finite_gradient_vectors(base_grad)
    block_idx = np.asarray(block_idx, dtype=np.int64)
    best_d2 = np.full((len(alive_idx),), np.inf, dtype=np.float64)
    best_local = np.zeros((len(alive_idx),), dtype=np.int64)
    if len(block_idx) == 0 or len(alive_idx) == 0:
        return best_d2, best_local

    centers64 = base_grad[block_idx].astype(np.float64)
    backend = str(backend).lower()
    if backend == "brute":
        backend = "brute_force"
    if backend not in {"brute_force", "kd_tree", "ball_tree", "greedy_kd_tree"}:
        raise ValueError(f"Unknown subset_mapping_backend: {backend}")

    if backend in {"kd_tree", "ball_tree"}:
        try:
            from sklearn.neighbors import BallTree, KDTree
        except Exception as exc:  # pragma: no cover - depends on optional environment
            raise ImportError(f"subset_mapping_backend={backend} requires scikit-learn.") from exc

        tree_cls = KDTree if backend == "kd_tree" else BallTree
        tree = tree_cls(centers64, leaf_size=max(1, int(leaf_size)), metric="euclidean")
        for start in range(0, len(alive_idx), int(chunk_size)):
            pos = slice(start, min(len(alive_idx), start + int(chunk_size)))
            idx = alive_idx[pos]
            block = base_grad[idx].astype(np.float64)
            dist, local = tree.query(block, k=1)
            best_local[pos] = local.reshape(-1).astype(np.int64)
            best_d2[pos] = np.square(dist.reshape(-1).astype(np.float64))
        return best_d2, best_local

    if backend == "greedy_kd_tree":
        root = _build_greedy_kd_tree(
            centers64,
            np.arange(len(block_idx), dtype=np.int64),
            leaf_size=max(1, int(leaf_size)),
        )
        for start in range(0, len(alive_idx), int(chunk_size)):
            pos = slice(start, min(len(alive_idx), start + int(chunk_size)))
            idx = alive_idx[pos]
            query = base_grad[idx].astype(np.float64)
            local = np.full((len(query),), -1, dtype=np.int64)
            _query_greedy_kd_tree(root, centers64, query, np.arange(len(query), dtype=np.int64), local)
            chosen = centers64[local]
            best_local[pos] = local.astype(np.int64)
            best_d2[pos] = np.sum((query - chosen) ** 2, axis=1).astype(np.float64)
        return best_d2, best_local

    center_norm = np.sum(centers64 * centers64, axis=1)
    for start in range(0, len(alive_idx), int(chunk_size)):
        pos = slice(start, min(len(alive_idx), start + int(chunk_size)))
        idx = alive_idx[pos]
        block = base_grad[idx].astype(np.float64)
        block_norm = np.sum(block * block, axis=1, keepdims=True)
        dots = np.einsum("bd,kd->bk", block, centers64, optimize=True)
        dist2 = block_norm + center_norm.reshape(1, -1) - 2.0 * dots
        dist2 = np.nan_to_num(dist2, nan=np.inf, posinf=np.inf, neginf=0.0)
        local = np.argmin(dist2, axis=1).astype(np.int64)
        best_local[pos] = local
        best_d2[pos] = dist2[np.arange(len(local)), local]
    return best_d2, best_local


def _build_subset_incremental_random_blocks(
    base_grad: np.ndarray,
    weights: np.ndarray,
    alive_mask: np.ndarray,
    candidate_mask: np.ndarray,
    subset_size: int,
    seed: int,
    random_trials: int,
    block_size: int = 0,
    mapping_backend: str = "brute_force",
    mapping_leaf_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, float, int]:
    base_grad = _finite_gradient_vectors(base_grad)
    weights = np.asarray(weights, dtype=np.float32)
    alive_mask = np.asarray(alive_mask, dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool) & alive_mask
    alive_idx = np.where(alive_mask)[0].astype(np.int64)
    candidate_idx = np.where(candidate_mask)[0].astype(np.int64)
    mapping = np.full((len(base_grad),), -1, dtype=np.int64)
    if len(alive_idx) == 0:
        return np.array([], dtype=np.int64), mapping, np.array([], dtype=np.float32), 0.0, 0.0, 0, 0.0, 0
    if len(candidate_idx) == 0:
        candidate_idx = alive_idx.copy()

    subset_size = max(1, min(int(subset_size), len(candidate_idx)))
    effective_block_size = int(block_size)
    if effective_block_size <= 0:
        effective_block_size = subset_size
    effective_block_size = max(1, min(effective_block_size, subset_size))
    trials = max(1, int(random_trials))

    if effective_block_size >= subset_size:
        subset, trial_mapping, alpha, eg, ew, trials_run, score = _build_subset_random_trial_mapped_gradient(
            base_grad,
            weights,
            alive_mask,
            candidate_mask,
            subset_size=subset_size,
            seed=seed,
            random_trials=trials,
            mapping_backend=mapping_backend,
            mapping_leaf_size=mapping_leaf_size,
        )
        blocks_run = 1 if len(subset) > 0 else 0
        return subset, trial_mapping, alpha, eg, ew, trials_run, score, blocks_run

    rng = np.random.default_rng(int(seed))

    full_grad = (weights[alive_idx, None].astype(np.float64) * base_grad[alive_idx].astype(np.float64)).sum(axis=0)
    full_norm = float(np.linalg.norm(full_grad) + 1e-12)
    full_weight = float(weights[alive_idx].sum())
    alive_weights64 = weights[alive_idx].astype(np.float64)

    selected: list[np.ndarray] = []
    selected_idx = np.array([], dtype=np.int64)
    selected_alpha = np.zeros((0,), dtype=np.float32)
    selected_grad = np.zeros_like(full_grad)
    nearest_d2 = np.full((len(alive_idx),), np.inf, dtype=np.float64)
    mapping_alive = np.full((len(alive_idx),), -1, dtype=np.int64)
    available_mask = np.zeros((len(base_grad),), dtype=bool)
    available_mask[candidate_idx] = True
    trials_run = 0
    blocks_run = 0
    best_score = float("inf")
    best_weight_error = float("inf")

    while len(selected_idx) < subset_size:
        available_idx = np.where(available_mask)[0].astype(np.int64)
        if len(available_idx) == 0:
            break
        remaining = subset_size - len(selected_idx)
        step_size = min(effective_block_size, remaining, len(available_idx))
        if step_size <= 0:
            break
        if step_size >= len(available_idx):
            trial_blocks = [available_idx.astype(np.int64)]
        else:
            trial_blocks = [
                np.sort(rng.choice(available_idx, size=step_size, replace=False).astype(np.int64))
                for _ in range(trials)
            ]

        round_best: Optional[tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        for block_idx in trial_blocks:
            block_idx = np.asarray(block_idx, dtype=np.int64)
            block_d2, block_local = _nearest_to_block(
                base_grad,
                alive_idx,
                block_idx,
                backend=mapping_backend,
                leaf_size=mapping_leaf_size,
            )
            improved = block_d2 < nearest_d2
            if len(selected_idx) == 0:
                improved = np.ones_like(improved, dtype=bool)

            trial_alpha = np.concatenate([selected_alpha.astype(np.float64), np.zeros((len(block_idx),), dtype=np.float64)])
            trial_grad = selected_grad.copy()
            if np.any(improved):
                old_pos = mapping_alive[improved]
                moved_w = alive_weights64[improved]
                old_valid = old_pos >= 0
                if np.any(old_valid):
                    np.add.at(trial_alpha, old_pos[old_valid], -moved_w[old_valid])
                    old_centers = base_grad[selected_idx[old_pos[old_valid]]].astype(np.float64)
                    trial_grad -= (moved_w[old_valid, None] * old_centers).sum(axis=0)

                new_local = block_local[improved]
                new_pos = len(selected_idx) + new_local
                np.add.at(trial_alpha, new_pos, moved_w)
                new_centers = base_grad[block_idx[new_local]].astype(np.float64)
                trial_grad += (moved_w[:, None] * new_centers).sum(axis=0)

            grad_error = float(np.linalg.norm(full_grad - trial_grad) / full_norm)
            subset_weight = float(trial_alpha.sum())
            weight_error = float(abs(full_weight - subset_weight) / (abs(full_weight) + 1e-12))
            trials_run += 1
            key = (grad_error, weight_error)
            if round_best is None or key < (round_best[0], round_best[1]):
                round_best = (
                    grad_error,
                    weight_error,
                    block_idx,
                    block_d2,
                    block_local,
                    improved,
                    trial_alpha,
                    trial_grad,
                )

        if round_best is None:
            break

        eg, ew, block_idx, block_d2, block_local, improved, trial_alpha, trial_grad = round_best
        start_pos = len(selected_idx)
        selected.append(block_idx.astype(np.int64))
        selected_idx = np.concatenate(selected).astype(np.int64)
        selected_alpha = np.maximum(trial_alpha, 0.0).astype(np.float32)
        selected_grad = trial_grad.astype(np.float64)
        if np.any(improved):
            nearest_d2[improved] = block_d2[improved]
            mapping_alive[improved] = start_pos + block_local[improved]
        available_mask[block_idx] = False
        blocks_run += 1
        best_score = float(eg)
        best_weight_error = float(ew)

    if len(selected_idx) == 0:
        return np.array([], dtype=np.int64), mapping, np.array([], dtype=np.float32), float("inf"), float("inf"), trials_run, float("inf"), blocks_run

    mapping[alive_idx] = mapping_alive.astype(np.int64)
    selected_alpha = _compute_subset_weights_from_mapping(weights, selected_idx, mapping, alive_mask)
    eg, ew = _compute_subset_gradient_error(base_grad, weights, selected_idx, selected_alpha, alive_mask)
    return (
        selected_idx.astype(np.int64),
        mapping,
        selected_alpha.astype(np.float32),
        float(eg),
        float(ew),
        int(trials_run),
        float(best_score if np.isfinite(best_score) else eg),
        int(blocks_run),
    )


def _build_subset_random_trial_mapped_gradient(
    base_grad: np.ndarray,
    weights: np.ndarray,
    alive_mask: np.ndarray,
    candidate_mask: np.ndarray,
    subset_size: int,
    seed: int,
    random_trials: int,
    mapping_backend: str = "brute_force",
    mapping_leaf_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, float]:
    base_grad = _finite_gradient_vectors(base_grad)
    weights = np.asarray(weights, dtype=np.float32)
    alive_mask = np.asarray(alive_mask, dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool) & alive_mask
    alive_idx = np.where(alive_mask)[0].astype(np.int64)
    candidate_idx = np.where(candidate_mask)[0].astype(np.int64)
    empty_mapping = np.full((len(base_grad),), -1, dtype=np.int64)
    if len(alive_idx) == 0:
        return np.array([], dtype=np.int64), empty_mapping, np.array([], dtype=np.float32), 0.0, 0.0, 0, 0.0
    if len(candidate_idx) == 0:
        candidate_idx = alive_idx.copy()
    subset_size = max(1, min(int(subset_size), len(candidate_idx)))
    trials = max(1, int(random_trials))
    rng = np.random.default_rng(int(seed))

    best_subset = np.array([], dtype=np.int64)
    best_mapping = empty_mapping
    best_alpha = np.array([], dtype=np.float32)
    best_grad_error = float("inf")
    best_weight_error = float("inf")
    best_score = float("inf")
    trials_run = 0

    if subset_size >= len(candidate_idx):
        trial_subsets = [candidate_idx.astype(np.int64)]
    else:
        trial_subsets = [
            np.sort(rng.choice(candidate_idx, size=subset_size, replace=False).astype(np.int64))
            for _ in range(trials)
        ]

    for subset_idx in trial_subsets:
        mapping = _assign_to_subset(
            base_grad,
            subset_idx,
            alive_mask,
            backend=mapping_backend,
            leaf_size=mapping_leaf_size,
        )
        alpha = _compute_subset_weights_from_mapping(weights, subset_idx, mapping, alive_mask)
        grad_error, weight_error = _compute_subset_gradient_error(base_grad, weights, subset_idx, alpha, alive_mask)
        score = float(grad_error)
        trials_run += 1
        if (score, float(weight_error)) < (best_score, best_weight_error):
            best_subset = subset_idx.astype(np.int64)
            best_mapping = mapping
            best_alpha = alpha.astype(np.float32)
            best_grad_error = float(grad_error)
            best_weight_error = float(weight_error)
            best_score = score

    return best_subset, best_mapping, best_alpha, best_grad_error, best_weight_error, trials_run, best_score


def _input_gradients_over_snapshots(
    model: nn.Module,
    snapshots: List[Dict[str, torch.Tensor]],
    X: np.ndarray,
    y_prob: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if len(X) == 0:
        return np.zeros_like(np.asarray(X, dtype=np.float32))
    if not snapshots:
        snapshots = [_copy_state_dict_cpu(model)]
    current = _copy_state_dict_cpu(model)
    grads = np.zeros_like(np.asarray(X, dtype=np.float32), dtype=np.float64)
    try:
        for state in snapshots:
            _load_state_dict_cpu(model, state, device)
            grads += input_gradients(
                model,
                np.asarray(X, dtype=np.float32),
                np.asarray(y_prob, dtype=np.float32),
                batch_size=max(1, int(batch_size)),
                device=str(device),
            ).astype(np.float64)
    finally:
        _load_state_dict_cpu(model, current, device)
    return (grads / max(1, len(snapshots))).astype(np.float32)


def run(ctx: MethodContext) -> MethodOutput:
    cfg = _training_config_from_context(ctx)
    device = torch.device(cfg.device)
    batch_size = int(cfg.batch_size)

    enable_error_handling = _as_bool(ctx.param("enable_error_handling", True))
    enable_feature_repair = enable_error_handling and _as_bool(ctx.param("enable_feature_repair", True))
    enable_sample_reliability_weight = enable_error_handling and _as_bool(
        ctx.param(
            "enable_sample_reliability_weight",
            ctx.param("enable_sample_weight", ctx.param("use_sample_reliability_weight", True)),
        )
    )

    pretrain_epochs = int(ctx.param("pretrain_epochs", 5))
    stage_update_interval = max(1, int(ctx.param("stage_update_interval", ctx.param("stage_epochs", 2))))
    snapshot_l = max(1, int(ctx.param("snapshot_L", 3)))
    subset_fraction = float(ctx.param("subset_fraction", 0.10))
    subset_size = int(ctx.param("subset_size", 0))
    subset_min_size = int(ctx.param("subset_min_size", ctx.num_classes))
    max_candidate_samples = int(ctx.param("max_candidate_samples", 0))
    max_dirty_samples_per_stage = int(ctx.param("max_dirty_samples_per_stage", 0))
    coreset_weighting = str(ctx.param("coreset_weighting", "nearest")).lower()

    dirty_mean_quantile = float(ctx.param("dirty_mean_quantile", 0.80))
    dirty_std_quantile = float(ctx.param("dirty_std_quantile", 0.50))
    clean_loss_quantile = float(ctx.param("clean_loss_quantile", 0.20))

    label_abs_factor = float(ctx.param("label_abs_factor", 1.8))
    label_rel_improve = float(ctx.param("label_rel_improve", 0.20))
    soft_label_temperature = float(ctx.param("soft_label_temperature", 2.0))
    soft_label_smooth_alpha = float(ctx.param("soft_label_smooth_alpha", 0.7))
    label_init_weight = float(ctx.param("label_init_weight", 0.5))
    label_hard_confidence = float(ctx.param("label_hard_confidence", ctx.param("label_p_hard", 0.90)))
    label_soft_confidence = float(ctx.param("label_soft_confidence", ctx.param("label_p_soft_low", 0.55)))
    label_stable_window = max(1, int(ctx.param("label_stable_window", 3)))
    label_consistency_thresh = float(ctx.param("label_consistency_thresh", 0.67))
    label_judge_interval = max(1, int(ctx.param("label_judge_interval", 3)))
    label_loss_hard_factor = float(ctx.param("label_loss_hard_factor", 1.5))
    label_loss_soft_factor = float(ctx.param("label_loss_soft_factor", 2.2))
    label_loss_delete_factor = float(ctx.param("label_loss_delete_factor", 3.0))
    label_uncertain_rho_max_thresh = float(ctx.param("label_uncertain_rho_max_thresh", 0.50))
    label_uncertain_p_upper = float(ctx.param("label_uncertain_p_upper", 0.75))
    label_decay_factor = float(ctx.param("label_decay_factor", 0.70))
    label_weight_floor = float(ctx.param("label_weight_floor", 0.30))

    grad_amp_quantile = float(ctx.param("grad_amp_quantile", 0.10))
    grad_amp_min = float(ctx.param("grad_amp_min", 1e-8))
    grad_concentration = float(ctx.param("grad_concentration", 0.80))
    virtual_step = float(ctx.param("virtual_step", ctx.param("feature_step", 0.05)))
    virtual_improve_ratio = float(ctx.param("virtual_improve_ratio", ctx.param("feature_improve_ratio", 0.01)))
    max_fix_features = int(ctx.param("max_fix_features", 3))
    max_feature_repairs_per_stage = int(ctx.param("max_feature_repairs_per_stage", 200))
    repair_max_steps = max(1, int(ctx.param("repair_max_steps", 10)))
    repair_recent_window = max(1, int(ctx.param("repair_recent_window", 3)))
    repair_tiny_drop_ratio = float(ctx.param("repair_tiny_drop_ratio", 0.001))
    repair_improve_ratio = float(ctx.param("repair_improve_ratio", 0.10))
    repair_target_drop_ratio = float(ctx.param("repair_target_drop_ratio", 0.50))
    repair_base_scale = float(ctx.param("repair_base_scale", 1.0))
    repair_step_max = float(ctx.param("repair_step_max", 0.05))
    repair_step_min = float(ctx.param("repair_step_min", 1e-4))
    repair_eta_up = float(ctx.param("repair_eta_up", 1.1))
    repair_eta_down = float(ctx.param("repair_eta_down", 0.5))
    repair_drop_high_ratio = float(ctx.param("repair_drop_high_ratio", 0.02))
    repair_drop_low_ratio = float(ctx.param("repair_drop_low_ratio", 0.001))
    repair_weight_smooth = float(ctx.param("repair_weight_smooth", 0.5))
    feature_score_smooth = float(ctx.param("feature_score_smooth", 0.90))
    feature_coef_min = float(ctx.param("feature_coef_min", 0.20))
    feature_coef_max = float(ctx.param("feature_coef_max", 1.00))
    feature_up_step = float(ctx.param("feature_up_step", ctx.param("clean_up_step", 0.05)))
    feature_down_factor = float(ctx.param("feature_down_factor", 0.70))
    mix_down_factor = float(ctx.param("mix_down_factor", 0.5))
    clean_up_step = float(ctx.param("clean_up_step", 0.05))
    weight_min_remove = float(ctx.param("weight_min_remove", 0.05))
    representative_weight_min = float(ctx.param("representative_weight_min", 0.2))
    representative_top_fraction = float(ctx.param("representative_top_fraction", 0.0))
    subset_eps_grad = float(ctx.param("subset_eps_grad", 0.20))
    subset_rebuild_eps_grad = float(ctx.param("subset_rebuild_eps_grad", 0.50))
    subset_eps_weight = float(ctx.param("subset_eps_weight", 0.05))
    subset_drift_thresh = float(ctx.param("subset_drift_thresh", 0.30))
    subset_rep_drift_thresh = float(ctx.param("subset_rep_drift_thresh", 0.40))
    subset_builder = str(ctx.param("subset_builder", "weighted_coverage")).lower()
    subset_random_trials = max(1, int(ctx.param("subset_random_trials", 5)))
    subset_block_size = int(ctx.param("subset_block_size", 0))
    subset_mapping_backend = str(ctx.param("subset_mapping_backend", "brute_force")).lower()
    if subset_mapping_backend == "brute":
        subset_mapping_backend = "brute_force"
    if subset_mapping_backend not in {"brute_force", "kd_tree", "ball_tree", "greedy_kd_tree"}:
        raise ValueError(f"Unknown subset_mapping_backend: {subset_mapping_backend}")
    subset_mapping_leaf_size = max(1, int(ctx.param("subset_mapping_leaf_size", 64)))
    subset_local_update_mapping = str(ctx.param("subset_local_update_mapping", "full_remap")).lower()
    if subset_local_update_mapping in {"inherit", "cluster_inherit"}:
        subset_local_update_mapping = "inherit_cluster"
    if subset_local_update_mapping not in {"full_remap", "inherit_cluster"}:
        raise ValueError(f"Unknown subset_local_update_mapping: {subset_local_update_mapping}")
    subset_rebuild_policy = str(ctx.param("subset_rebuild_policy", "auto")).lower()
    if subset_rebuild_policy not in {"auto", "initial_only"}:
        raise ValueError(f"Unknown subset_rebuild_policy: {subset_rebuild_policy}")
    coverage_candidate_chunk_size = int(ctx.param("coverage_candidate_chunk_size", 512))

    set_seed(ctx.seed)
    model = _build_model(ctx).to(device)
    if cfg.optimizer.lower() != "adamw":
        raise ValueError("Only AdamW is currently supported.")
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    X_work = ctx.X_train.copy()
    y_prob = one_hot(ctx.y_train, ctx.num_classes)
    weights = np.ones(ctx.n_samples, dtype=np.float32)
    alive = np.ones(ctx.n_samples, dtype=bool)
    noisy_mask = np.zeros(ctx.n_samples, dtype=bool)
    label_states: Dict[int, _LabelState] = {}

    subset_idx = np.array([], dtype=np.int64)
    subset_weights = np.array([], dtype=np.float32)
    subset_mapping = np.full((ctx.n_samples,), -1, dtype=np.int64)
    prev_weights = weights.copy()
    prev_alive = alive.copy()
    snapshots: Deque[Dict[str, torch.Tensor]] = deque(maxlen=snapshot_l)
    history: list[dict[str, float]] = []
    stage_reports: list[dict[str, object]] = []
    stage_timing_keys = (
        "stage_update_time_sec",
        "snapshot_loss_time_sec",
        "dirty_selection_time_sec",
        "label_candidate_time_sec",
        "feature_gradient_time_sec",
        "feature_virtual_eval_time_sec",
        "label_update_time_sec",
        "feature_repair_time_sec",
        "mixed_weight_time_sec",
        "clean_weight_time_sec",
        "subset_time_sec",
    )
    stage_timing_totals = {key: 0.0 for key in stage_timing_keys}

    train_time_sec = 0.0
    method_update_time_sec = 0.0
    best_train_loss = float("inf")
    bad_epochs = 0
    best_epoch: Optional[int] = None
    epochs_ran = 0

    def set_reliability_weight(index: int, value: float) -> None:
        if enable_sample_reliability_weight:
            weights[int(index)] = float(value)

    def scale_reliability_weight(index: int, factor: float) -> None:
        if enable_sample_reliability_weight:
            weights[int(index)] = float(weights[int(index)] * float(factor))

    def remove_if_weight_too_small(index: int) -> bool:
        if enable_sample_reliability_weight and weights[int(index)] < weight_min_remove:
            alive[int(index)] = False
            weights[int(index)] = 0.0
            return True
        return False

    output_dir = Path(ctx.output_dir) if ctx.output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    last_model_path = str(output_dir / "last_model.pt") if output_dir is not None and cfg.save_last_model else None
    best_model_path = str(output_dir / "best_model.pt") if output_dir is not None and cfg.save_best_model else None

    def train_epoch(indices: np.ndarray, alpha: np.ndarray, stage_id: int) -> bool:
        nonlocal train_time_sec, best_train_loss, bad_epochs, best_epoch, epochs_ran
        if len(indices) == 0:
            indices = np.where(alive)[0].astype(np.int64)
            alpha = weights[indices]
        loader = _make_loader(
            X_work[indices],
            ctx.y_train[indices],
            y_prob[indices],
            alpha,
            cfg=cfg,
            seed=ctx.seed + int(epochs_ran),
        )
        started = time.perf_counter()
        train_loss, samples_seen = _train_one_epoch_weighted_soft(model, loader, optimizer, device)
        epoch_time = time.perf_counter() - started
        train_time_sec += epoch_time
        epochs_ran += 1
        snapshots.append(_copy_state_dict_cpu(model))

        improved = train_loss < (best_train_loss - float(cfg.early_stop_min_delta))
        if improved:
            best_train_loss = float(train_loss)
            best_epoch = int(epochs_ran)
            bad_epochs = 0
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)
        else:
            bad_epochs += 1
        history.append(
            {
                "epoch": float(epochs_ran),
                "stage": float(stage_id),
                "train_loss": float(train_loss),
                "train_time_sec": float(epoch_time),
                "samples_seen": float(samples_seen),
                "alive_samples": float(alive.sum()),
                "subset_size": float(len(subset_idx)),
                "best_train_loss_so_far": float(best_train_loss),
            }
        )
        return bool(cfg.early_stop and bad_epochs >= int(cfg.early_stop_patience))

    def compute_base_grad_full() -> np.ndarray:
        active_idx = np.where(alive)[0].astype(np.int64)
        base_grad_full = np.zeros((ctx.n_samples, ctx.num_classes), dtype=np.float32)
        if len(active_idx) == 0:
            return base_grad_full
        base_grad = _base_grad_representation(model, X_work[active_idx], y_prob[active_idx], batch_size, device)
        base_grad_full[active_idx] = base_grad
        return base_grad_full

    def assign_to_current_subset(base_grad_full: np.ndarray, reps: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return _assign_to_subset(
            base_grad_full,
            reps,
            mask,
            backend=subset_mapping_backend,
            leaf_size=subset_mapping_leaf_size,
        )

    def subset_report_common(block_size_value: int = 0, blocks_run: int = 0) -> dict[str, object]:
        return {
            "subset_block_size": int(block_size_value),
            "subset_blocks_run": int(blocks_run),
            "subset_mapping_backend": subset_mapping_backend,
            "subset_mapping_leaf_size": int(subset_mapping_leaf_size),
            "subset_local_update_mapping": subset_local_update_mapping,
            "subset_rebuild_policy": subset_rebuild_policy,
        }

    def representative_candidate_mask(stage_id: int) -> np.ndarray:
        active_idx = np.where(alive)[0].astype(np.int64)
        if len(active_idx) == 0:
            return np.zeros(ctx.n_samples, dtype=bool)
        if np.isfinite(representative_top_fraction) and representative_top_fraction > 0.0:
            candidate_mask = _weighted_candidates_from_fraction(active_idx, weights, representative_top_fraction, ctx.n_samples)
        else:
            candidate_mask = alive & (weights >= representative_weight_min)
            if not np.any(candidate_mask):
                candidate_mask = alive.copy()

        candidate_indices = np.where(candidate_mask)[0].astype(np.int64)
        if max_candidate_samples > 0 and len(candidate_indices) > max_candidate_samples:
            local = bounded_classwise_candidates(
                ctx.y_train[candidate_indices],
                ctx.num_classes,
                max_candidates=max_candidate_samples,
                seed=ctx.seed + 910 + int(stage_id),
                scores=weights[candidate_indices],
                largest=True,
            )
            candidate_mask = np.zeros(ctx.n_samples, dtype=bool)
            candidate_mask[candidate_indices[local]] = True
        return candidate_mask & alive

    def requested_core_size(candidate_count: int) -> int:
        if candidate_count <= 0:
            return 0
        if subset_size > 0:
            k = min(int(subset_size), int(candidate_count))
        else:
            k = subset_size_from_fraction(int(alive.sum()), subset_fraction, min_size=max(1, subset_min_size))
        return min(max(1, int(k)), int(candidate_count))

    def rebuild_subset(stage_id: int, base_grad_full: Optional[np.ndarray] = None, reason: str = "rebuild") -> dict[str, object]:
        nonlocal subset_idx, subset_weights, subset_mapping
        active_idx = np.where(alive)[0].astype(np.int64)
        if len(active_idx) == 0:
            subset_idx = np.array([], dtype=np.int64)
            subset_weights = np.array([], dtype=np.float32)
            subset_mapping[:] = -1
            return {"subset_update_mode": "empty", "subset_error_grad": 0.0, "subset_error_weight": 0.0}

        base_grad_full = compute_base_grad_full() if base_grad_full is None else base_grad_full
        candidate_mask = representative_candidate_mask(stage_id)
        candidate_count = int(candidate_mask.sum())
        k = requested_core_size(candidate_count)
        builder_name = subset_builder
        builder_trials_run = 0
        builder_score: Optional[float] = None
        builder_blocks_run = 0
        effective_block_size = int(subset_block_size if subset_block_size > 0 else k)
        if k <= 0:
            subset_idx = active_idx.copy()
            subset_mapping = assign_to_current_subset(base_grad_full, subset_idx, alive)
            subset_weights = _compute_subset_weights_from_mapping(weights, subset_idx, subset_mapping, alive)
            eg, ew = _compute_subset_gradient_error(base_grad_full, weights, subset_idx, subset_weights, alive)
        elif builder_name in {"random_trial_mapped_gradient", "random_trial"}:
            (
                subset_idx,
                subset_mapping,
                subset_weights,
                eg,
                ew,
                builder_trials_run,
                builder_score,
            ) = _build_subset_random_trial_mapped_gradient(
                base_grad_full,
                weights,
                alive,
                candidate_mask,
                subset_size=k,
                seed=ctx.seed + 900 + int(stage_id),
                random_trials=subset_random_trials,
                mapping_backend=subset_mapping_backend,
                mapping_leaf_size=subset_mapping_leaf_size,
            )
            builder_name = "random_trial_mapped_gradient"
        elif builder_name in {"incremental_random_blocks", "incremental_blocks"}:
            (
                subset_idx,
                subset_mapping,
                subset_weights,
                eg,
                ew,
                builder_trials_run,
                builder_score,
                builder_blocks_run,
            ) = _build_subset_incremental_random_blocks(
                base_grad_full,
                weights,
                alive,
                candidate_mask,
                subset_size=k,
                seed=ctx.seed + 900 + int(stage_id),
                random_trials=subset_random_trials,
                block_size=effective_block_size,
                mapping_backend=subset_mapping_backend,
                mapping_leaf_size=subset_mapping_leaf_size,
            )
            builder_name = "incremental_random_blocks"
        elif builder_name in {"weighted_coverage", "coverage"}:
            builder_name = "weighted_coverage"
            subset_idx = _build_subset_weighted_coverage(
                base_grad_full,
                weights,
                alive,
                candidate_mask,
                subset_size=k,
                seed=ctx.seed + 900 + int(stage_id),
                candidate_chunk_size=coverage_candidate_chunk_size,
            )
            subset_mapping = assign_to_current_subset(base_grad_full, subset_idx, alive)
            subset_weights = _compute_subset_weights_from_mapping(weights, subset_idx, subset_mapping, alive)
            eg, ew = _compute_subset_gradient_error(base_grad_full, weights, subset_idx, subset_weights, alive)
        else:
            raise ValueError(f"Unknown CleanCore subset_builder: {subset_builder}")
        return {
            "subset_update_mode": reason,
            "subset_builder": builder_name,
            "subset_builder_score": float(eg if builder_score is None else builder_score),
            "subset_random_trials": int(subset_random_trials),
            "subset_random_trials_run": int(builder_trials_run),
            "subset_error_grad": float(eg),
            "subset_error_weight": float(ew),
            "subset_local_replacements": 0,
            "subset_remapped": int(alive.sum()),
            "candidate_samples": candidate_count,
            **subset_report_common(effective_block_size, builder_blocks_run),
        }

    def best_replacement_for_members(
        base_grad_full: np.ndarray,
        members: np.ndarray,
        candidate_mask: np.ndarray,
        used_reps: set[int],
    ) -> Optional[int]:
        members = np.asarray(members, dtype=np.int64)
        if len(members) == 0:
            return None
        local_candidates = [int(i) for i in members.tolist() if candidate_mask[int(i)] and int(i) not in used_reps]
        if not local_candidates:
            local_candidates = [
                int(i)
                for i in np.where(candidate_mask)[0].astype(np.int64).tolist()
                if int(i) not in used_reps
            ]
        if not local_candidates:
            return None
        cand = np.array(local_candidates, dtype=np.int64)
        member_vec = base_grad_full[members]
        member_w = weights[members].astype(np.float64)
        best = None
        best_score = np.inf
        for start in range(0, len(cand), max(1, coverage_candidate_chunk_size)):
            block = cand[start : start + max(1, coverage_candidate_chunk_size)]
            diff = member_vec[:, None, :] - base_grad_full[block][None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            score = (member_w[:, None] * dist2).sum(axis=0)
            pos = int(np.argmin(score))
            if float(score[pos]) < best_score:
                best_score = float(score[pos])
                best = int(block[pos])
        return best

    def update_or_rebuild_subset(stage_id: int, affected_idx: np.ndarray) -> dict[str, object]:
        nonlocal subset_idx, subset_weights, subset_mapping, prev_weights, prev_alive
        base_grad_full = compute_base_grad_full()
        candidate_mask = representative_candidate_mask(stage_id)
        candidate_count = int(candidate_mask.sum())
        update_only_after_initial = subset_rebuild_policy == "initial_only" and len(subset_idx) > 0
        if len(subset_idx) == 0:
            info = rebuild_subset(stage_id, base_grad_full=base_grad_full, reason="initial_rebuild")
            prev_weights = weights.copy()
            prev_alive = alive.copy()
            return info

        valid_reps = alive[subset_idx] if update_only_after_initial else (alive[subset_idx] & candidate_mask[subset_idx])
        subset_idx = subset_idx[valid_reps]
        if len(subset_idx) == 0:
            if update_only_after_initial:
                prev_weights = weights.copy()
                prev_alive = alive.copy()
                return {
                    "subset_update_mode": "rebuild_suppressed_empty_reps",
                    "subset_builder": subset_builder,
                    "subset_builder_score": float("inf"),
                    "subset_random_trials": int(subset_random_trials),
                    "subset_random_trials_run": 0,
                    "subset_error_grad": float("inf"),
                    "subset_error_weight": float("inf"),
                    "subset_local_replacements": 0,
                    "subset_remapped": 0,
                    "candidate_samples": candidate_count,
                    **subset_report_common(),
                }
            info = rebuild_subset(stage_id, base_grad_full=base_grad_full, reason="rebuild_invalid_reps")
            prev_weights = weights.copy()
            prev_alive = alive.copy()
            return info

        if len(subset_mapping) != ctx.n_samples or np.any(subset_mapping[alive] < 0):
            subset_mapping = assign_to_current_subset(base_grad_full, subset_idx, alive)
        subset_weights = _compute_subset_weights_from_mapping(weights, subset_idx, subset_mapping, alive)
        eg, ew = _compute_subset_gradient_error(base_grad_full, weights, subset_idx, subset_weights, alive)

        if eg > subset_rebuild_eps_grad:
            if update_only_after_initial:
                mode = "local_update_rebuild_suppressed_error"
            else:
                info = rebuild_subset(stage_id, base_grad_full=base_grad_full, reason="rebuild_error")
                prev_weights = weights.copy()
                prev_alive = alive.copy()
                return info
        else:
            mode = "keep_update"

        local_replacements = 0
        remapped = 0
        old_mapping = subset_mapping.copy()
        if (
            eg > subset_eps_grad
            or ew > subset_eps_weight
            or len(affected_idx) > 0
            or mode.startswith("local_update_rebuild_suppressed")
        ):
            if mode == "keep_update":
                mode = "local_update"
            reps_set = set(int(i) for i in subset_idx.tolist())
            clusters = [[] for _ in range(len(subset_idx))]
            for gi in np.where(alive)[0].astype(np.int64).tolist():
                j = int(subset_mapping[gi])
                if 0 <= j < len(clusters):
                    clusters[j].append(int(gi))

            for j, rep in enumerate(subset_idx.tolist()):
                members = np.array(clusters[j], dtype=np.int64)
                if len(members) <= 1:
                    continue
                rep_vec = base_grad_full[int(rep)]
                diff = base_grad_full[members] - rep_vec.reshape(1, -1)
                avg_dev = float((weights[members] * np.linalg.norm(diff, axis=1)).sum() / max(float(weights[members].sum()), 1e-12))
                rel_dev = avg_dev / max(float(np.linalg.norm(rep_vec)), 1e-12)
                if rel_dev > subset_rep_drift_thresh:
                    reps_set.discard(int(rep))
                    replacement = best_replacement_for_members(base_grad_full, members, candidate_mask, reps_set)
                    if replacement is not None and replacement != int(rep):
                        subset_idx[j] = int(replacement)
                        reps_set.add(int(replacement))
                        local_replacements += 1
                    else:
                        reps_set.add(int(rep))

            if len(subset_idx) == 0:
                if update_only_after_initial:
                    prev_weights = weights.copy()
                    prev_alive = alive.copy()
                    return {
                        "subset_update_mode": "rebuild_suppressed_empty_after_local",
                        "subset_builder": subset_builder,
                        "subset_builder_score": float("inf"),
                        "subset_random_trials": int(subset_random_trials),
                        "subset_random_trials_run": 0,
                        "subset_error_grad": float("inf"),
                        "subset_error_weight": float("inf"),
                        "subset_local_replacements": int(local_replacements),
                        "subset_remapped": int(remapped),
                        "candidate_samples": candidate_count,
                        **subset_report_common(),
                    }
                info = rebuild_subset(stage_id, base_grad_full=base_grad_full, reason="rebuild_empty_after_local")
                prev_weights = weights.copy()
                prev_alive = alive.copy()
                return info

            if len(affected_idx) > 0:
                affected_idx = np.asarray([int(i) for i in affected_idx.tolist() if alive[int(i)]], dtype=np.int64)
                if len(affected_idx) > 0:
                    centers = base_grad_full[subset_idx]
                    centers64 = centers.astype(np.float64)
                    center_norm = np.sum(centers64 * centers64, axis=1)
                    block = base_grad_full[affected_idx].astype(np.float64)
                    block_norm = np.sum(block * block, axis=1, keepdims=True)
                    dots = np.einsum("bd,kd->bk", block, centers64, optimize=True)
                    dist2 = block_norm + center_norm.reshape(1, -1) - 2.0 * dots
                    dist2 = np.nan_to_num(dist2, nan=np.inf, posinf=np.inf, neginf=0.0)
                    new_j = np.argmin(dist2, axis=1).astype(np.int64)
                    old_j = subset_mapping[affected_idx]
                    subset_mapping[affected_idx] = new_j
                    remapped += int(np.sum(new_j != old_j))

            if subset_local_update_mapping == "full_remap":
                subset_mapping = assign_to_current_subset(base_grad_full, subset_idx, alive)
                remapped = max(remapped, int(np.sum(subset_mapping != old_mapping)))
            else:
                remapped = max(remapped, int(np.sum(subset_mapping != old_mapping)))
            subset_weights = _compute_subset_weights_from_mapping(weights, subset_idx, subset_mapping, alive)
            eg, ew = _compute_subset_gradient_error(base_grad_full, weights, subset_idx, subset_weights, alive)
            if eg > subset_rebuild_eps_grad or ew > max(subset_eps_weight, 1e-12) * 2.0:
                if update_only_after_initial:
                    mode = "local_update_rebuild_suppressed_after_local"
                else:
                    info = rebuild_subset(stage_id, base_grad_full=base_grad_full, reason="rebuild_after_local")
                    prev_weights = weights.copy()
                    prev_alive = alive.copy()
                    return info
        else:
            if subset_local_update_mapping == "full_remap":
                subset_mapping = assign_to_current_subset(base_grad_full, subset_idx, alive)
                subset_weights = _compute_subset_weights_from_mapping(weights, subset_idx, subset_mapping, alive)
                eg, ew = _compute_subset_gradient_error(base_grad_full, weights, subset_idx, subset_weights, alive)

        prev_weights = weights.copy()
        prev_alive = alive.copy()
        return {
            "subset_update_mode": mode,
            "subset_builder": subset_builder,
            "subset_builder_score": float(eg),
            "subset_random_trials": int(subset_random_trials),
            "subset_random_trials_run": 0,
            "subset_error_grad": float(eg),
            "subset_error_weight": float(ew),
            "subset_local_replacements": int(local_replacements),
            "subset_remapped": int(remapped),
            "candidate_samples": candidate_count,
            **subset_report_common(),
        }

    def stage_boundary_update(stage_id: int) -> None:
        nonlocal method_update_time_sec
        started = time.perf_counter()
        timing = {key: 0.0 for key in stage_timing_keys}
        active_idx = np.where(alive)[0].astype(np.int64)
        if len(active_idx) == 0:
            return

        if not enable_error_handling:
            subset_started = time.perf_counter()
            subset_info = update_or_rebuild_subset(stage_id, np.array([], dtype=np.int64))
            timing["subset_time_sec"] = time.perf_counter() - subset_started
            stage_elapsed = time.perf_counter() - started
            timing["stage_update_time_sec"] = stage_elapsed
            for key in stage_timing_keys:
                stage_timing_totals[key] += float(timing[key])
            method_update_time_sec += stage_elapsed
            stage_reports.append(
                {
                    "stage": int(stage_id),
                    "n_dirty": 0,
                    "n_dirty_handled": 0,
                    "n_label": 0,
                    "n_label_hard": 0,
                    "n_label_soft": 0,
                    "n_feature": 0,
                    "n_feature_repaired": 0,
                    "feature_repair_steps_mean": 0.0,
                    "feature_repair_steps_max": 0,
                    "n_mixed": 0,
                    "n_removed_now": 0,
                    "n_stable_clean_upweighted": 0,
                    "n_alive": int(alive.sum()),
                    "subset_size": int(len(subset_idx)),
                    "clean_typical_loss": 0.0,
                    "mean_threshold": 0.0,
                    "std_threshold": 0.0,
                    "stage_timing": {key: float(timing[key]) for key in stage_timing_keys},
                    **{key: float(timing[key]) for key in stage_timing_keys},
                    **subset_info,
                }
            )
            return

        snapshot_started = time.perf_counter()
        snap_list = list(snapshots) or [_copy_state_dict_cpu(model)]
        mean_loss, std_loss = _loss_mean_std_over_snapshots(
            model,
            snap_list,
            X_work[active_idx],
            y_prob[active_idx],
            batch_size=batch_size,
            device=device,
        )
        timing["snapshot_loss_time_sec"] = time.perf_counter() - snapshot_started

        dirty_started = time.perf_counter()
        mean_loss_full = np.zeros(ctx.n_samples, dtype=np.float32)
        mean_loss_full[active_idx] = mean_loss
        mean_threshold = float(np.quantile(mean_loss, dirty_mean_quantile))
        std_threshold = float(np.quantile(std_loss, dirty_std_quantile))
        dirty_local = (mean_loss >= mean_threshold) & (std_loss <= std_threshold)
        dirty_idx_all = active_idx[dirty_local]
        noisy_mask[dirty_idx_all] = True
        dirty_idx = dirty_idx_all
        if max_dirty_samples_per_stage > 0 and len(dirty_idx_all) > max_dirty_samples_per_stage:
            dirty_positions = np.where(dirty_local)[0]
            order = np.argsort(-mean_loss[dirty_positions], kind="mergesort")[:max_dirty_samples_per_stage]
            dirty_idx = active_idx[dirty_positions[order]]
        timing["dirty_selection_time_sec"] = time.perf_counter() - dirty_started

        clean_typical_loss = max(float(np.quantile(mean_loss, clean_loss_quantile)), 1e-8)
        label_idx: list[int] = []
        label_hard_idx: list[int] = []
        label_soft_idx: list[int] = []
        feature_detect_idx: list[int] = []
        feature_repaired_idx: list[int] = []
        feature_repair_steps: list[int] = []
        mixed_idx: list[int] = []
        affected: set[int] = set()
        label_candidates: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        dirty_pos = {int(gi): pos for pos, gi in enumerate(dirty_idx.tolist())}

        label_candidate_started = time.perf_counter()
        class_losses = np.zeros((len(dirty_idx), ctx.num_classes), dtype=np.float32)
        if len(dirty_idx) > 0:
            class_losses = _loss_by_class_over_snapshots(
                model,
                snap_list,
                X_work[dirty_idx],
                ctx.num_classes,
                batch_size=batch_size,
                device=device,
            )
            current = np.argmax(y_prob[dirty_idx], axis=1).astype(np.int64)
            current_loss = class_losses[np.arange(len(dirty_idx)), current]
            for pos, gi in enumerate(dirty_idx.tolist()):
                cur = float(max(current_loss[pos], 1e-8))
                abs_ok = class_losses[pos] <= (label_abs_factor * clean_typical_loss)
                rel_ok = class_losses[pos] <= ((1.0 - label_rel_improve) * cur)
                candidates = np.where(abs_ok & rel_ok)[0].astype(np.int64)
                candidates = candidates[candidates != current[pos]]
                if len(candidates) == 0:
                    continue
                label_idx.append(int(gi))
                label_candidates[int(gi)] = (
                    candidates.astype(np.int64),
                    class_losses[pos, candidates].astype(np.float32),
                    class_losses[pos].astype(np.float32),
                )
        timing["label_candidate_time_sec"] = time.perf_counter() - label_candidate_started

        remaining_dirty = np.array(
            sorted(set(int(i) for i in dirty_idx.tolist()) - set(label_idx)),
            dtype=np.int64,
        )
        feat_core: Dict[int, np.ndarray] = {}
        feat_virtual_drop: Dict[int, float] = {}
        feat_grad_norm2: Dict[int, float] = {}
        virtual_rows: list[np.ndarray] = []
        virtual_meta: list[tuple[int, np.ndarray, float]] = []
        feature_gradient_started = time.perf_counter()
        if len(remaining_dirty) > 0:
            avg_grads = _input_gradients_over_snapshots(
                model,
                snap_list,
                X_work[remaining_dirty],
                y_prob[remaining_dirty],
                batch_size=max(1, min(batch_size, 256)),
                device=device,
            )
            abs_grads = np.abs(avg_grads)
            grad_amp = abs_grads.sum(axis=1)
            amp_threshold = max(float(grad_amp_min), float(np.quantile(grad_amp, grad_amp_quantile)))
            for pos, gi in enumerate(remaining_dirty.tolist()):
                if float(grad_amp[pos]) <= amp_threshold:
                    mixed_idx.append(int(gi))
                    continue
                abs_g = abs_grads[pos]
                total_amp = float(abs_g.sum())
                if total_amp <= 1e-12:
                    mixed_idx.append(int(gi))
                    continue
                order = np.argsort(-abs_g, kind="mergesort")
                cumulative = np.cumsum(abs_g[order]) / max(total_amp, 1e-12)
                within = np.where(cumulative[: max(1, min(max_fix_features, len(order)))] >= grad_concentration)[0]
                if len(within) == 0:
                    mixed_idx.append(int(gi))
                    continue
                k_core = int(within[0] + 1)
                core = order[:k_core].astype(np.int64)
                x_virtual = X_work[gi].copy()
                x_virtual[core] = x_virtual[core] - float(virtual_step) * np.sign(avg_grads[pos, core])
                virtual_rows.append(x_virtual.astype(np.float32))
                gn2 = float(np.sum(avg_grads[pos, core].astype(np.float64) ** 2))
                virtual_meta.append((int(gi), core, max(gn2, 1e-12)))
        timing["feature_gradient_time_sec"] = time.perf_counter() - feature_gradient_started

        virtual_started = time.perf_counter()
        if virtual_rows:
            virtual_mean, _ = _loss_mean_std_over_snapshots(
                model,
                snap_list,
                np.stack(virtual_rows, axis=0).astype(np.float32),
                y_prob[[m[0] for m in virtual_meta]],
                batch_size=batch_size,
                device=device,
            )
            for pos, (gi, core, gn2) in enumerate(virtual_meta):
                old = float(max(mean_loss_full[gi], 1e-8))
                new = float(virtual_mean[pos])
                drop = old - new
                if drop / max(old, 1e-8) >= virtual_improve_ratio:
                    feature_detect_idx.append(int(gi))
                    feat_core[int(gi)] = core
                    feat_virtual_drop[int(gi)] = max(drop, 0.0)
                    feat_grad_norm2[int(gi)] = float(gn2)
                else:
                    mixed_idx.append(int(gi))
        timing["feature_virtual_eval_time_sec"] = time.perf_counter() - virtual_started

        label_update_started = time.perf_counter()
        loss_hard_th = float(label_loss_hard_factor * clean_typical_loss)
        loss_soft_th = float(label_loss_soft_factor * clean_typical_loss)
        loss_delete_th = float(label_loss_delete_factor * clean_typical_loss)
        do_label_judge = (int(stage_id) % label_judge_interval == 0)

        for gi in label_idx:
            cand_labels, cand_losses, all_class_losses = label_candidates[int(gi)]
            shifted = cand_losses.astype(np.float64) - float(np.min(cand_losses))
            scores = np.exp(-soft_label_temperature * shifted)
            probs = scores / max(float(scores.sum()), 1e-12)
            p_inst = np.zeros((ctx.num_classes,), dtype=np.float32)
            p_inst[cand_labels] = probs.astype(np.float32)
            is_new = int(gi) not in label_states
            if is_new:
                label_states[int(gi)] = _LabelState(ctx.num_classes, label_stable_window)
            state = label_states[int(gi)]
            alpha = float(np.clip(soft_label_smooth_alpha, 0.0, 1.0))
            if is_new or float(state.p.sum()) <= 1e-12:
                p_new = p_inst
            else:
                p_new = alpha * state.p + (1.0 - alpha) * p_inst
            p_new = p_new / max(float(p_new.sum()), 1e-12)
            state.p = p_new.astype(np.float32)
            dom = int(np.argmax(state.p))
            p_max = float(state.p[dom])
            state.dom_hist.append(dom)
            stable = _label_state_is_stable(
                state,
                dom=dom,
                min_window=label_stable_window,
                consistency_thresh=label_consistency_thresh,
            )
            hist = list(state.dom_hist)
            rho_max = float(np.bincount(np.asarray(hist, dtype=np.int64), minlength=ctx.num_classes).max() / max(1, len(hist)))
            dom_loss = float(all_class_losses[dom])
            if do_label_judge and stable and p_max >= label_hard_confidence and dom_loss <= loss_hard_th:
                y_prob[int(gi)] = one_hot(np.array([dom], dtype=np.int64), ctx.num_classes)[0]
                set_reliability_weight(int(gi), 1.0)
                del label_states[int(gi)]
                label_hard_idx.append(int(gi))
            elif p_max >= label_soft_confidence and dom_loss <= loss_soft_th:
                y_prob[int(gi)] = state.p.copy()
                denom = max(1e-12, 1.0 - label_soft_confidence)
                w_new = label_init_weight + (1.0 - label_init_weight) * (p_max - label_soft_confidence) / denom
                set_reliability_weight(int(gi), float(np.clip(w_new, label_weight_floor, 1.0)))
                label_soft_idx.append(int(gi))
            elif rho_max <= label_uncertain_rho_max_thresh and p_max <= label_uncertain_p_upper and dom_loss >= loss_delete_th:
                y_prob[int(gi)] = state.p.copy()
                scale_reliability_weight(int(gi), label_decay_factor)
                if remove_if_weight_too_small(int(gi)):
                    label_states.pop(int(gi), None)
            else:
                y_prob[int(gi)] = state.p.copy()
                set_reliability_weight(
                    int(gi),
                    float(np.clip(min(float(weights[int(gi)]), label_init_weight), label_weight_floor, 1.0)),
                )
                label_soft_idx.append(int(gi))
            affected.add(int(gi))
        timing["label_update_time_sec"] = time.perf_counter() - label_update_started

        feature_repair_started = time.perf_counter()
        feat_err_sorted = sorted(feature_detect_idx, key=lambda i: float(weights[i]), reverse=True)
        if max_feature_repairs_per_stage > 0:
            feat_err_sorted = feat_err_sorted[:max_feature_repairs_per_stage]
        if enable_feature_repair:
            for gi in feat_err_sorted:
                if not alive[int(gi)]:
                    continue
                core = feat_core.get(int(gi))
                if core is None or len(core) == 0:
                    mixed_idx.append(int(gi))
                    continue
                x_orig = X_work[int(gi)].copy()
                x = x_orig.copy()
                l0 = float(_per_sample_soft_losses(model, x.reshape(1, -1), y_prob[int(gi) : int(gi) + 1], 1, device)[0])
                vdrop = max(float(feat_virtual_drop.get(int(gi), 0.0)), 0.0)
                gn2 = max(float(feat_grad_norm2.get(int(gi), 1.0)), 1e-12)
                eta = repair_base_scale * repair_target_drop_ratio * vdrop / gn2
                eta = float(np.clip(eta, repair_step_min, repair_step_max))
                best_x = x.copy()
                best_loss = l0
                prev_sign = None
                scores = np.zeros((len(core),), dtype=np.float32)
                recent_drops: Deque[float] = deque(maxlen=repair_recent_window)
                steps_used = 0
                for step in range(1, repair_max_steps + 1):
                    xb = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=device, requires_grad=True)
                    yb = torch.tensor(y_prob[int(gi)].reshape(1, -1), dtype=torch.float32, device=device)
                    model.zero_grad(set_to_none=True)
                    loss = -(yb * torch.log_softmax(model(xb), dim=1)).sum()
                    loss.backward()
                    grad = xb.grad.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    grad_core = grad[core]
                    coeffs, scores = _feature_repair_coefficients(
                        grad_core,
                        scores,
                        prev_sign,
                        smooth=feature_score_smooth,
                        coef_min=feature_coef_min,
                        coef_max=feature_coef_max,
                    )
                    prev_sign = np.sign(grad_core).astype(np.int8)
                    x_new = x.copy()
                    x_new[core] = x_new[core] - float(eta) * coeffs * grad_core
                    l_new = float(_per_sample_soft_losses(model, x_new.reshape(1, -1), y_prob[int(gi) : int(gi) + 1], 1, device)[0])
                    drop_abs = float(loss.item() - l_new)
                    recent_drops.append(drop_abs)
                    if l_new < best_loss:
                        best_loss = l_new
                        best_x = x_new.copy()
                    high = repair_drop_high_ratio * max(float(loss.item()), 1e-12)
                    low = repair_drop_low_ratio * max(float(loss.item()), 1e-12)
                    if drop_abs >= high:
                        eta = min(repair_step_max, eta * repair_eta_up)
                    elif drop_abs <= low:
                        eta = max(repair_step_min, eta * repair_eta_down)
                    x = x_new
                    steps_used = step
                    improve_ratio = (l0 - best_loss) / max(l0, 1e-12)
                    if improve_ratio >= repair_improve_ratio and best_loss <= clean_typical_loss:
                        break
                    if len(recent_drops) >= repair_recent_window and all(
                        d <= repair_tiny_drop_ratio * max(l0, 1e-12) for d in recent_drops
                    ):
                        break

                gain = max(0.0, l0 - best_loss)
                confidence = float(np.clip(gain / max(vdrop, 1e-12), 0.0, 1.0))
                if gain > 0.0 and best_loss < l0:
                    X_work[int(gi)] = best_x.astype(np.float32)
                    set_reliability_weight(
                        int(gi),
                        float(np.clip((1.0 - repair_weight_smooth) * weights[int(gi)] + repair_weight_smooth * confidence, label_weight_floor, 1.0)),
                    )
                    if confidence >= repair_improve_ratio and enable_sample_reliability_weight:
                        weights[int(gi)] = min(1.0, float(weights[int(gi)]) + feature_up_step)
                    feature_repaired_idx.append(int(gi))
                else:
                    X_work[int(gi)] = x_orig.astype(np.float32)
                    scale_reliability_weight(int(gi), feature_down_factor)
                    mixed_idx.append(int(gi))
                affected.add(int(gi))
                feature_repair_steps.append(int(steps_used))
        else:
            mixed_idx.extend(int(gi) for gi in feat_err_sorted if alive[int(gi)])
            affected.update(int(gi) for gi in feat_err_sorted if alive[int(gi)])
        timing["feature_repair_time_sec"] = time.perf_counter() - feature_repair_started

        mixed_weight_started = time.perf_counter()
        mixed_idx = sorted(set(int(i) for i in mixed_idx if alive[int(i)] and int(i) not in set(label_idx)))
        removed_now = 0
        for gi in mixed_idx:
            scale_reliability_weight(int(gi), mix_down_factor)
            affected.add(int(gi))
            if remove_if_weight_too_small(int(gi)):
                removed_now += 1
        timing["mixed_weight_time_sec"] = time.perf_counter() - mixed_weight_started

        clean_weight_started = time.perf_counter()
        stable_clean_local = (~dirty_local) & (mean_loss <= np.quantile(mean_loss, 0.50)) & (std_loss <= std_threshold)
        stable_clean_idx = active_idx[stable_clean_local]
        for gi in stable_clean_idx.tolist():
            if alive[int(gi)]:
                new_w = min(1.0, float(weights[int(gi)]) + clean_up_step)
                if enable_sample_reliability_weight and abs(new_w - float(weights[int(gi)])) > 1e-12:
                    affected.add(int(gi))
                set_reliability_weight(int(gi), new_w)
        if enable_sample_reliability_weight:
            weights[~alive] = 0.0
        timing["clean_weight_time_sec"] = time.perf_counter() - clean_weight_started

        subset_started = time.perf_counter()
        subset_info = update_or_rebuild_subset(stage_id, np.array(sorted(affected), dtype=np.int64))
        timing["subset_time_sec"] = time.perf_counter() - subset_started
        stage_elapsed = time.perf_counter() - started
        timing["stage_update_time_sec"] = stage_elapsed
        for key in stage_timing_keys:
            stage_timing_totals[key] += float(timing[key])
        method_update_time_sec += stage_elapsed
        stage_reports.append(
            {
                "stage": int(stage_id),
                "n_dirty": int(len(dirty_idx_all)),
                "n_dirty_handled": int(len(dirty_idx)),
                "n_label": int(len(label_idx)),
                "n_label_hard": int(len(label_hard_idx)),
                "n_label_soft": int(len(label_soft_idx)),
                "n_feature": int(len(feature_detect_idx)),
                "n_feature_repaired": int(len(feature_repaired_idx)),
                "feature_repair_steps_mean": float(np.mean(feature_repair_steps)) if feature_repair_steps else 0.0,
                "feature_repair_steps_max": int(max(feature_repair_steps)) if feature_repair_steps else 0,
                "n_mixed": int(len(mixed_idx)),
                "n_removed_now": int(removed_now),
                "n_stable_clean_upweighted": int(len(stable_clean_idx)),
                "n_alive": int(alive.sum()),
                "subset_size": int(len(subset_idx)),
                "clean_typical_loss": float(clean_typical_loss),
                "mean_threshold": float(mean_threshold),
                "std_threshold": float(std_threshold),
                "stage_timing": {key: float(timing[key]) for key in stage_timing_keys},
                **{key: float(timing[key]) for key in stage_timing_keys},
                **subset_info,
            }
        )

    method_max_epochs = ctx.method_config.get("max_epochs")
    max_epochs_source = "method" if method_max_epochs is not None else "training"
    max_epochs = max(1, int(method_max_epochs if method_max_epochs is not None else cfg.max_epochs))
    stopped = False
    for _ in range(min(max(0, pretrain_epochs), max_epochs)):
        active = np.where(alive)[0].astype(np.int64)
        stopped = train_epoch(active, weights[active], stage_id=0)
        if stopped:
            break

    updates_done = 0
    if not stopped and epochs_ran < max_epochs:
        updates_done = 1
        stage_boundary_update(updates_done)

    while not stopped and epochs_ran < max_epochs:
        stage_id = max(1, updates_done)
        for _ in range(stage_update_interval):
            if stopped or epochs_ran >= max_epochs:
                break
            if len(subset_idx) > 0:
                train_idx = subset_idx
                alpha = subset_weights
            else:
                train_idx = np.where(alive)[0].astype(np.int64)
                alpha = weights[train_idx]
            stopped = train_epoch(train_idx, alpha, stage_id=stage_id)
        if stopped or epochs_ran >= max_epochs:
            break
        updates_done += 1
        stage_boundary_update(updates_done)

    if last_model_path is not None:
        torch.save(model.state_dict(), last_model_path)

    final_predictions = None
    if ctx.X_test is not None:
        final_predictions = predict_classifier(model, ctx.X_test, batch_size=batch_size, device=str(device))

    if len(subset_idx) == 0:
        subset_idx = np.where(alive)[0].astype(np.int64)
        subset_weights = weights[subset_idx].astype(np.float32)

    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=subset_idx,
        sample_weights=subset_weights,
        corrected_features=X_work,
        soft_targets=y_prob,
        predicted_noisy_mask=noisy_mask,
        final_predictions=final_predictions,
        training_history=history,
        metadata={
            "paper": "CleanCore Sections 3-4: continuous stage-wise cleaning-aware coreset training",
            "paper_mechanisms": [
                "training_signal_error_typing",
                "stable_soft_label_handling",
                "iterative_feature_repair",
                "reliability_weighted_gradient_coverage",
                "adaptive_dynamic_coreset",
            ],
            "method_type": "training_coupled",
            "continuous_training": True,
            "final_model_predictions": final_predictions is not None,
            "error_handling_enabled": bool(enable_error_handling),
            "feature_repair_enabled": bool(enable_feature_repair),
            "sample_reliability_weight_enabled": bool(enable_sample_reliability_weight),
            "sample_weight_mode": "reliability" if enable_sample_reliability_weight else "uniform_raw_mapping_count",
            "pretrain_epochs": pretrain_epochs,
            "max_epochs": int(max_epochs),
            "max_epochs_source": max_epochs_source,
            "stage_update_interval": stage_update_interval,
            "stage_updates": int(updates_done),
            "snapshot_L": snapshot_l,
            "subset_fraction": subset_fraction,
            "subset_size": int(len(subset_idx)),
            "requested_subset_size": subset_size,
            "candidate_samples": int(len(subset_idx)),
            "max_candidate_samples": max_candidate_samples,
            "max_dirty_samples_per_stage": max_dirty_samples_per_stage,
            "coreset_weighting": coreset_weighting,
            "representative_weight_min": representative_weight_min,
            "representative_top_fraction": representative_top_fraction,
            "label_stable_window": label_stable_window,
            "label_consistency_thresh": label_consistency_thresh,
            "label_judge_interval": label_judge_interval,
            "repair_max_steps": repair_max_steps,
            "max_feature_repairs_per_stage": max_feature_repairs_per_stage,
            "feature_coef_min": feature_coef_min,
            "feature_coef_max": feature_coef_max,
            "subset_eps_grad": subset_eps_grad,
            "subset_rebuild_eps_grad": subset_rebuild_eps_grad,
            "subset_eps_weight": subset_eps_weight,
            "subset_drift_thresh": subset_drift_thresh,
            "subset_rep_drift_thresh": subset_rep_drift_thresh,
            "subset_builder": subset_builder,
            "subset_random_trials": int(subset_random_trials),
            "subset_block_size": int(subset_block_size),
            "subset_mapping_backend": subset_mapping_backend,
            "subset_mapping_leaf_size": int(subset_mapping_leaf_size),
            "subset_local_update_mapping": subset_local_update_mapping,
            "subset_rebuild_policy": subset_rebuild_policy,
            "n_alive": int(alive.sum()),
            "mean_weight": float(weights[alive].mean()) if np.any(alive) else 0.0,
            "epochs_ran": int(epochs_ran),
            "best_epoch": best_epoch,
            "best_train_loss": None if best_epoch is None else float(best_train_loss),
            "train_time_sec": float(train_time_sec),
            "method_update_time_sec": float(method_update_time_sec),
            "stage_timing_totals": {key: float(stage_timing_totals[key]) for key in stage_timing_keys},
            "stage_timing_means": {
                key: float(stage_timing_totals[key] / max(1, len(stage_reports)))
                for key in stage_timing_keys
            },
            "last_model_path": last_model_path,
            "best_model_path": best_model_path,
            "stage_reports": stage_reports,
        },
    )
