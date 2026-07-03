"""Deem baseline: dynamic soft-label subset training over mislabeled data."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from common.interfaces import MethodOutput
from common.training import predict_classifier
from methods.base import MethodContext
from methods.common import bounded_classwise_candidates, one_hot
from methods.coupled import (
    build_model,
    build_optimizer,
    make_loader,
    model_output_paths,
    train_weighted_epoch,
    training_config_from_context,
)
from methods.torch_utils import (
    grad_embeddings_last_layer,
    matching_pursuit_nonnegative,
    per_sample_losses,
    predict_proba,
)


def build_soft_labels(y: np.ndarray, probs: np.ndarray, losses: np.ndarray, alpha_max: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    hard = one_hot(y, probs.shape[1])
    if len(losses) == 0:
        return hard, np.zeros(len(y), dtype=np.float32)
    lo = float(np.min(losses))
    hi = float(np.max(losses))
    norm = (losses - lo) / max(hi - lo, 1e-12)
    alpha = np.clip(float(alpha_max) * (norm ** float(gamma)), 0.0, float(alpha_max)).astype(np.float32)
    soft = (1.0 - alpha[:, None]) * hard + alpha[:, None] * probs.astype(np.float32)
    soft = soft / np.clip(soft.sum(axis=1, keepdims=True), 1e-12, None)
    return soft.astype(np.float32), alpha


def _minmax01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def build_paper_soft_labels(y: np.ndarray, prob_snapshots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Deem-style soft labels from recent class-loss mean and std."""

    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(prob_snapshots, dtype=np.float32)
    if probs.ndim == 2:
        probs = probs[None, :, :]
    probs = np.clip(probs, 1e-12, 1.0)
    class_losses = -np.log(probs)
    avg_loss = class_losses.mean(axis=0)
    std_loss = class_losses.std(axis=0)
    h = 0.5 * (_minmax01(avg_loss) + _minmax01(std_loss))
    label_score = np.clip(1.0 - h, 1e-6, None)
    soft = label_score / np.clip(label_score.sum(axis=1, keepdims=True), 1e-12, None)
    observed_prob = soft[np.arange(len(y)), y].astype(np.float32)
    return soft.astype(np.float32), observed_prob


def _batch_size(value: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    if parsed <= 0:
        parsed = int(default)
    return max(1, int(parsed))


def _squared_distances_block(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    row_norms = np.sum(X * X, axis=1, dtype=np.float32)[:, None]
    center_norms = np.sum(C * C, axis=1, dtype=np.float32)[None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        d2 = row_norms + center_norms - (2.0 * (X @ C.T))
    d2 = np.nan_to_num(d2, nan=np.inf, posinf=np.inf, neginf=0.0)
    return np.maximum(d2, 0.0).astype(np.float32, copy=False)


def _nearest_center_distances_for_rows(
    G: np.ndarray,
    selected_local: np.ndarray,
    row_local: np.ndarray,
    row_batch_size: int,
    center_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_local = np.asarray(selected_local, dtype=np.int64)
    row_local = np.asarray(row_local, dtype=np.int64)
    n_rows = int(len(row_local))
    if len(selected_local) == 0:
        return np.full(n_rows, -1, dtype=np.int64), np.full(n_rows, np.inf, dtype=np.float32)

    row_batch_size = _batch_size(row_batch_size, 4096)
    center_batch_size = _batch_size(center_batch_size, 64)
    mapping = np.full(n_rows, -1, dtype=np.int64)
    best = np.full(n_rows, np.inf, dtype=np.float32)

    for row_start in range(0, n_rows, row_batch_size):
        row_end = min(n_rows, row_start + row_batch_size)
        rows = G[row_local[row_start:row_end]]
        best_block = best[row_start:row_end]
        mapping_block = mapping[row_start:row_end]
        for center_start in range(0, len(selected_local), center_batch_size):
            center_end = min(len(selected_local), center_start + center_batch_size)
            centers = G[selected_local[center_start:center_end]]
            d2 = _squared_distances_block(rows, centers)
            local_argmin = np.argmin(d2, axis=1).astype(np.int64)
            local_best = d2[np.arange(row_end - row_start), local_argmin]
            improved = local_best < best_block
            best_block[improved] = local_best[improved]
            mapping_block[improved] = center_start + local_argmin[improved]
    return mapping, best


def _nearest_center_distances(
    G: np.ndarray,
    selected_local: np.ndarray,
    row_batch_size: int,
    center_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _nearest_center_distances_for_rows(
        G,
        selected_local,
        np.arange(len(G), dtype=np.int64),
        row_batch_size=row_batch_size,
        center_batch_size=center_batch_size,
    )


def _gradient_mapping_weights(
    G: np.ndarray,
    selected_local: np.ndarray,
    row_batch_size: int = 4096,
    center_batch_size: int = 64,
    mapping_sample_size: int = 0,
    seed: int = 0,
    mapping_backend: str = "chunked_matmul",
) -> tuple[np.ndarray, np.ndarray, float]:
    G = np.asarray(G, dtype=np.float32)
    selected_local = np.asarray(selected_local, dtype=np.int64)
    if len(selected_local) == 0:
        return np.full(len(G), -1, dtype=np.int64), np.array([], dtype=np.float32), float("inf")
    n = int(len(G))
    backend = str(mapping_backend).lower()
    if mapping_sample_size and 0 < int(mapping_sample_size) < n:
        rng = np.random.default_rng(int(seed))
        row_local = rng.choice(np.arange(n, dtype=np.int64), size=int(mapping_sample_size), replace=False).astype(np.int64)
        if backend in {"kd_tree", "ball_tree"}:
            sample_mapping, best = _tree_nearest_center_distances(G, selected_local, row_local, backend)
        else:
            sample_mapping, best = _nearest_center_distances_for_rows(
                G,
                selected_local,
                row_local,
                row_batch_size=row_batch_size,
                center_batch_size=center_batch_size,
            )
        mapping = np.full(n, -1, dtype=np.int64)
        mapping[row_local] = sample_mapping
        weights = np.bincount(sample_mapping, minlength=len(selected_local)).astype(np.float32)
        scale = float(n) / max(1.0, float(len(row_local)))
        weights *= scale
        error = float(np.sum(best, dtype=np.float64) * scale)
    else:
        if backend in {"kd_tree", "ball_tree"}:
            mapping, best = _tree_nearest_center_distances(
                G,
                selected_local,
                np.arange(n, dtype=np.int64),
                backend,
            )
        else:
            mapping, best = _nearest_center_distances(
                G,
                selected_local,
                row_batch_size=row_batch_size,
                center_batch_size=center_batch_size,
            )
        weights = np.bincount(mapping, minlength=len(selected_local)).astype(np.float32)
        error = float(np.sum(best, dtype=np.float64))
    weights[weights <= 0.0] = 1.0
    return mapping, weights, error


def _tree_nearest_center_distances(
    G: np.ndarray,
    selected_local: np.ndarray,
    row_local: np.ndarray,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import BallTree, KDTree

    tree_cls = KDTree if str(backend).lower() == "kd_tree" else BallTree
    centers = np.asarray(G[np.asarray(selected_local, dtype=np.int64)], dtype=np.float32)
    rows = np.asarray(G[np.asarray(row_local, dtype=np.int64)], dtype=np.float32)
    tree = tree_cls(centers, metric="euclidean")
    distances, mapping = tree.query(rows, k=1, return_distance=True)
    return mapping[:, 0].astype(np.int64), np.square(distances[:, 0]).astype(np.float32)


def _candidate_improvements(
    G: np.ndarray,
    best_dist: np.ndarray,
    cand: np.ndarray,
    row_batch_size: int,
    candidate_batch_size: int,
    eval_local: Optional[np.ndarray] = None,
) -> np.ndarray:
    rows_idx = np.arange(len(G), dtype=np.int64) if eval_local is None else np.asarray(eval_local, dtype=np.int64)
    scores = np.zeros(len(cand), dtype=np.float64)
    row_batch_size = _batch_size(row_batch_size, 4096)
    candidate_batch_size = _batch_size(candidate_batch_size, 64)
    for cand_start in range(0, len(cand), candidate_batch_size):
        cand_end = min(len(cand), cand_start + candidate_batch_size)
        centers = G[cand[cand_start:cand_end]]
        score_block = np.zeros(cand_end - cand_start, dtype=np.float64)
        for row_start in range(0, len(rows_idx), row_batch_size):
            row_end = min(len(rows_idx), row_start + row_batch_size)
            idx = rows_idx[row_start:row_end]
            d2 = _squared_distances_block(G[idx], centers)
            gain = np.maximum(0.0, best_dist[idx, None] - d2)
            score_block += gain.sum(axis=0, dtype=np.float64)
        scores[cand_start:cand_end] = score_block
    return scores


def _distances_to_one_center(G: np.ndarray, center: np.ndarray, row_batch_size: int) -> np.ndarray:
    row_batch_size = _batch_size(row_batch_size, 4096)
    out = np.empty(len(G), dtype=np.float32)
    center_2d = center.reshape(1, -1)
    for row_start in range(0, len(G), row_batch_size):
        row_end = min(len(G), row_start + row_batch_size)
        out[row_start:row_end] = _squared_distances_block(G[row_start:row_end], center_2d)[:, 0]
    return out


def _paper_greedy_gradient_subset(
    G: np.ndarray,
    global_idx: np.ndarray,
    k: int,
    sample_size: int,
    sample_ratio: float,
    seed: int,
    initial_selected_global: Optional[np.ndarray] = None,
    row_batch_size: int = 4096,
    candidate_batch_size: int = 64,
    eval_sample_size: int = 0,
    greedy_steps_cap: int = 0,
    mapping_sample_size: int = 0,
    greedy_batch_size: int = 1,
    mapping_backend: str = "chunked_matmul",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, object]]:
    """Greedy sampled reduction of gradient representation error."""

    G = np.asarray(G, dtype=np.float32)
    n = int(len(global_idx))
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), np.array([], dtype=np.int64), 0.0, {}
    k = min(max(1, int(k)), n)
    row_batch_size = _batch_size(row_batch_size, 4096)
    candidate_batch_size = _batch_size(candidate_batch_size, 64)
    eval_sample_size = max(0, int(eval_sample_size))
    greedy_steps_cap = max(0, int(greedy_steps_cap))
    mapping_sample_size = max(0, int(mapping_sample_size))
    greedy_batch_size = max(1, int(greedy_batch_size))
    rng = np.random.default_rng(int(seed))
    eval_rng = np.random.default_rng(int(seed) + 1000003)
    eval_local = None
    if eval_sample_size > 0 and eval_sample_size < n:
        eval_local = eval_rng.choice(np.arange(n, dtype=np.int64), size=eval_sample_size, replace=False).astype(np.int64)
    selected_local: list[int] = []
    remaining = np.ones(n, dtype=bool)
    global_to_local = {int(g): i for i, g in enumerate(global_idx.tolist())}
    if initial_selected_global is not None:
        for g in np.asarray(initial_selected_global, dtype=np.int64).tolist():
            loc = global_to_local.get(int(g))
            if loc is None or not remaining[loc]:
                continue
            selected_local.append(int(loc))
            remaining[loc] = False
            if len(selected_local) >= k:
                break

    if selected_local:
        _, best_dist = _nearest_center_distances(
            G,
            np.array(selected_local, dtype=np.int64),
            row_batch_size=row_batch_size,
            center_batch_size=candidate_batch_size,
        )
    else:
        best_dist = np.full(n, np.inf, dtype=np.float32)
    full_mean = G.mean(axis=0)
    rounds_added = 0
    samples_added = 0
    greedy_started = time.perf_counter()
    while len(selected_local) < k and np.any(remaining) and (greedy_steps_cap <= 0 or rounds_added < greedy_steps_cap):
        pool = np.where(remaining)[0].astype(np.int64)
        if sample_size > 0:
            h = min(int(sample_size), len(pool))
        elif sample_ratio > 0.0:
            h = min(max(1, int(round(float(sample_ratio) * len(pool)))), len(pool))
        else:
            h = len(pool)
        cand = pool if h >= len(pool) else rng.choice(pool, size=h, replace=False).astype(np.int64)
        take = min(greedy_batch_size, k - len(selected_local), len(cand))
        if selected_local:
            improvement = _candidate_improvements(
                G,
                best_dist,
                cand,
                row_batch_size=row_batch_size,
                candidate_batch_size=candidate_batch_size,
                eval_local=eval_local,
            )
            chosen = cand[np.argsort(-improvement, kind="stable")[:take]].astype(np.int64)
        else:
            dist_to_mean = np.sum((G[cand] - full_mean[None, :]) ** 2, axis=1)
            chosen = cand[np.argsort(dist_to_mean, kind="stable")[:take]].astype(np.int64)
        selected_local.extend(int(value) for value in chosen.tolist())
        remaining[chosen] = False
        for row_start in range(0, len(G), row_batch_size):
            row_end = min(len(G), row_start + row_batch_size)
            d2 = _squared_distances_block(G[row_start:row_end], G[chosen])
            best_dist[row_start:row_end] = np.minimum(best_dist[row_start:row_end], np.min(d2, axis=1))
        rounds_added += 1
        samples_added += int(len(chosen))

    filled_samples = 0
    if len(selected_local) < k and np.any(remaining):
        pool = np.where(remaining)[0].astype(np.int64)
        fill_n = min(k - len(selected_local), len(pool))
        if fill_n > 0:
            fill = pool if fill_n >= len(pool) else rng.choice(pool, size=fill_n, replace=False).astype(np.int64)
            selected_local.extend(int(x) for x in fill.tolist())
            remaining[fill] = False
            filled_samples = int(len(fill))

    selected_local_arr = np.array(selected_local, dtype=np.int64)
    greedy_time_sec = time.perf_counter() - greedy_started
    mapping_started = time.perf_counter()
    mapping, weights, error = _gradient_mapping_weights(
        G,
        selected_local_arr,
        row_batch_size=row_batch_size,
        center_batch_size=candidate_batch_size,
        mapping_sample_size=mapping_sample_size,
        seed=int(seed) + 2000003,
        mapping_backend=mapping_backend,
    )
    mapping_time_sec = time.perf_counter() - mapping_started
    selected = global_idx[selected_local_arr].astype(np.int64)
    report = {
        "candidate_samples": int(n),
        "selected_samples": int(len(selected)),
        "greedy_steps_added": int(rounds_added),
        "greedy_rounds_added": int(rounds_added),
        "greedy_samples_added": int(samples_added),
        "greedy_batch_size": int(greedy_batch_size),
        "greedy_steps_cap": int(greedy_steps_cap),
        "filled_samples": int(filled_samples),
        "greedy_sample_size": int(sample_size),
        "greedy_sample_ratio": float(sample_ratio),
        "greedy_eval_sample_size": int(eval_sample_size),
        "greedy_effective_eval_samples": int(n if eval_local is None else len(eval_local)),
        "mapping_sample_size": int(mapping_sample_size),
        "mapping_samples": int(n if not mapping_sample_size or mapping_sample_size >= n else mapping_sample_size),
        "distance_backend": str(mapping_backend).lower(),
        "distance_row_batch_size": int(row_batch_size),
        "distance_candidate_batch_size": int(candidate_batch_size),
        "gradient_error": float(error),
        "greedy_time_sec": float(greedy_time_sec),
        "mapping_time_sec": float(mapping_time_sec),
    }
    return selected, weights, mapping, error, report


def _select_deem_subset(
    ctx: MethodContext,
    model: torch.nn.Module,
    soft_targets: np.ndarray,
    losses: np.ndarray,
    subset_fraction: float,
    subset_size: int,
    selector: str,
    steps_cap: int,
    max_candidate_samples: int,
    batch_size: int,
    device: str,
    update_id: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    candidate_idx = bounded_classwise_candidates(
        ctx.y_train,
        ctx.num_classes,
        max_candidates=max_candidate_samples,
        seed=ctx.seed + 710 + int(update_id),
        scores=losses,
        largest=True,
    )
    G = grad_embeddings_last_layer(
        model,
        ctx.X_train[candidate_idx],
        soft_targets[candidate_idx],
        ctx.num_classes,
        batch_size=batch_size,
        device=device,
        soft=True,
    )
    k = subset_size if subset_size > 0 else max(1, int(round(subset_fraction * len(candidate_idx))))
    k = min(max(1, int(k)), len(candidate_idx))

    if selector == "topk":
        g_full = G.mean(axis=0)
        dots = G @ g_full
        score = np.abs(dots) / (np.linalg.norm(G, axis=1) * (np.linalg.norm(g_full) + 1e-12) + 1e-12)
        selected_local = np.argsort(-score)[:k].astype(np.int64)
        selected = candidate_idx[selected_local]
        weights = np.maximum(
            0.0,
            (G[selected_local] @ g_full) / (np.sum(G[selected_local] * G[selected_local], axis=1) + 1e-12),
        ).astype(np.float32)
        if float(weights.sum()) > 0:
            weights = weights / float(weights.sum()) * float(len(weights))
        else:
            weights = np.ones(len(selected), dtype=np.float32)
    elif selector == "omp":
        selected_local, weights = matching_pursuit_nonnegative(G, k=k, steps_cap=steps_cap)
        selected = candidate_idx[selected_local]
    else:
        raise ValueError("Deem selector must be omp or topk.")

    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    weights[weights <= 0.0] = 1.0
    return selected.astype(np.int64), weights, {
        "candidate_samples": int(len(candidate_idx)),
        "selected_samples": int(len(selected)),
    }


def run(ctx: MethodContext) -> MethodOutput:
    cfg = training_config_from_context(ctx)
    device = torch.device(cfg.device)
    batch_size = int(cfg.batch_size)

    deem_mode = str(ctx.param("deem_mode", "legacy")).lower()
    if deem_mode not in {"legacy", "optimized"}:
        raise ValueError("deem_mode must be legacy or optimized.")
    warmup_epochs = max(0, int(ctx.param("warmup_epochs", ctx.param("score_epochs", 5))))
    selection_interval = max(1, int(ctx.param("selection_interval", ctx.param("update_interval", max(1, warmup_epochs)))))
    subset_fraction = float(ctx.param("subset_fraction", 0.10))
    subset_size = int(ctx.param("subset_size", 0))
    alpha_max = float(ctx.param("alpha_max", 0.8))
    gamma = float(ctx.param("gamma", 1.0))
    selector = str(ctx.param("selector", "omp")).lower()
    soft_label_mode = str(ctx.param("soft_label_mode", "loss_blend")).lower()
    soft_label_window = max(1, int(ctx.param("soft_label_window", 3)))
    mislabel_remove_ratio = float(ctx.param("mislabel_remove_ratio", ctx.param("q", 0.0)))
    min_remaining = int(ctx.param("min_remaining", max(2 * ctx.num_classes, 20)))
    subset_update_threshold = float(ctx.param("subset_update_threshold", ctx.param("tau", 0.30)))
    greedy_sample_size = int(ctx.param("greedy_sample_size", ctx.param("sample_size", 0)))
    greedy_sample_ratio = float(ctx.param("greedy_sample_ratio", 0.0))
    distance_row_batch_size = _batch_size(int(ctx.param("distance_row_batch_size", ctx.param("distance_batch_size", 4096))), 4096)
    distance_candidate_batch_size = _batch_size(
        int(ctx.param("distance_candidate_batch_size", ctx.param("candidate_batch_size", max(1, min(64, greedy_sample_size or 64))))),
        64,
    )
    greedy_eval_sample_size = max(0, int(ctx.param("greedy_eval_sample_size", 0)))
    greedy_steps_cap = max(0, int(ctx.param("greedy_steps_cap", 0)))
    mapping_sample_size = max(0, int(ctx.param("mapping_sample_size", 0)))
    greedy_batch_size = max(1, int(ctx.param("greedy_batch_size", 1)))
    mapping_backend = str(ctx.param("mapping_backend", "chunked_matmul")).lower()
    update_strategy = str(ctx.param("update_strategy", "legacy")).lower()
    if deem_mode == "legacy":
        greedy_batch_size = 1
        mapping_backend = "chunked_matmul"
        update_strategy = "legacy"
    elif mapping_backend not in {"chunked_matmul", "kd_tree", "ball_tree"}:
        raise ValueError("mapping_backend must be chunked_matmul, kd_tree, or ball_tree.")
    steps_cap = int(ctx.param("mp_steps_cap", 0))
    max_candidate_samples = int(ctx.param("max_candidate_samples", 0))

    model = build_model(ctx, seed_offset=700).to(device)
    optimizer = build_optimizer(model, cfg)
    last_model_path, best_model_path = model_output_paths(ctx, cfg)

    y_soft = one_hot(ctx.y_train, ctx.num_classes)
    selected_idx = np.arange(ctx.n_samples, dtype=np.int64)
    sample_weights = np.ones(ctx.n_samples, dtype=np.float32)
    alpha = np.zeros(ctx.n_samples, dtype=np.float32)
    active = np.ones(ctx.n_samples, dtype=bool)
    predicted_noisy = np.zeros(ctx.n_samples, dtype=bool)
    observed_label_prob = np.ones(ctx.n_samples, dtype=np.float32)
    prob_window: list[np.ndarray] = []
    history: list[dict[str, float]] = []
    update_reports: list[dict[str, object]] = []
    train_time_sec = 0.0
    selection_time_sec = 0.0
    best_train_loss = float("inf")
    best_epoch: Optional[int] = None
    bad_epochs = 0
    epochs_ran = 0
    updates_done = 0
    rebuild_error: Optional[float] = None
    rebuilds = 0
    local_updates = 0

    def train_epoch(indices: np.ndarray, weights: np.ndarray, soft_targets: np.ndarray, phase: str) -> bool:
        nonlocal train_time_sec, best_train_loss, best_epoch, bad_epochs, epochs_ran
        loader = make_loader(
            ctx.X_train[indices],
            ctx.y_train[indices],
            sample_weights=weights,
            soft_targets=soft_targets,
            cfg=cfg,
            seed=ctx.seed + int(epochs_ran),
        )
        started = time.perf_counter()
        with ctx.timed_phase("deem.training_epoch"):
            train_loss, samples_seen = train_weighted_epoch(model, loader, optimizer, device)
        epoch_time = time.perf_counter() - started
        train_time_sec += epoch_time
        epochs_ran += 1

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
                "phase": phase,
                "train_loss": float(train_loss),
                "train_time_sec": float(epoch_time),
                "samples_seen": float(samples_seen),
                "subset_size": float(len(indices)),
                "mean_soft_alpha": float(alpha.mean()),
                "best_train_loss_so_far": float(best_train_loss),
            }
        )
        return bool(cfg.early_stop and bad_epochs >= int(cfg.early_stop_patience))

    def update_subset() -> None:
        nonlocal selected_idx, sample_weights, y_soft, alpha, updates_done, selection_time_sec
        nonlocal active, predicted_noisy, observed_label_prob, rebuild_error, rebuilds, local_updates
        updates_done += 1
        started = time.perf_counter()
        with ctx.timed_phase("deem.full_prediction"):
            probs = predict_proba(model, ctx.X_train, batch_size=batch_size, device=str(device))
        prob_window.append(probs.astype(np.float32))
        del prob_window[:-soft_label_window]
        with ctx.timed_phase("deem.loss_scoring"):
            losses = per_sample_losses(model, ctx.X_train, ctx.y_train, batch_size=batch_size, device=str(device))
        removed_this_update = 0
        update_mode = "recompute"
        with ctx.timed_phase("deem.soft_label_update"):
            if soft_label_mode in {"paper", "paper_loss_std", "loss_std"}:
                y_soft, observed_label_prob = build_paper_soft_labels(ctx.y_train, np.stack(prob_window, axis=0))
                alpha = (1.0 - observed_label_prob).astype(np.float32)
            else:
                y_soft, alpha = build_soft_labels(ctx.y_train, probs, losses, alpha_max=alpha_max, gamma=gamma)
                observed_label_prob = y_soft[np.arange(ctx.n_samples), ctx.y_train].astype(np.float32)

        if selector == "paper_greedy" and mislabel_remove_ratio > 0.0:
            with ctx.timed_phase("deem.mislabel_filtering"):
                active_idx = np.where(active)[0].astype(np.int64)
                remove_k = max(1, int(round(float(mislabel_remove_ratio) * len(active_idx))))
                remove_k = min(remove_k, max(0, len(active_idx) - max(min_remaining, ctx.num_classes)))
                if remove_k > 0:
                    order = active_idx[np.argsort(observed_label_prob[active_idx])[:remove_k]]
                    active[order] = False
                    predicted_noisy[order] = True
                    removed_this_update = int(len(order))

        if selector == "paper_greedy":
            active_idx = np.where(active)[0].astype(np.int64)
            if len(active_idx) == 0:
                active[:] = True
                active_idx = np.where(active)[0].astype(np.int64)
            requested_k = subset_size if subset_size > 0 else max(1, int(round(subset_fraction * len(active_idx))))
            requested_k = min(max(1, int(requested_k)), len(active_idx))
            previous_selected = np.array([], dtype=np.int64)
            if updates_done > 1 and len(selected_idx) <= requested_k:
                previous_selected = selected_idx[np.isin(selected_idx, active_idx)].astype(np.int64)

            gradient_idx = active_idx
            if deem_mode == "optimized" and max_candidate_samples > 0 and len(active_idx) > max_candidate_samples:
                effective_cap = min(len(active_idx), max(requested_k, int(max_candidate_samples)))
                local = bounded_classwise_candidates(
                    ctx.y_train[active_idx],
                    ctx.num_classes,
                    max_candidates=effective_cap,
                    seed=ctx.seed + 720 + int(updates_done),
                )
                sampled = active_idx[local].astype(np.int64)
                if update_strategy == "inherit_cluster" and len(previous_selected) > 0:
                    inherited = previous_selected[:effective_cap]
                    inherited_set = set(int(value) for value in inherited.tolist())
                    extras = np.array([value for value in sampled if int(value) not in inherited_set], dtype=np.int64)
                    gradient_idx = np.concatenate([inherited, extras[: max(0, effective_cap - len(inherited))]])
                else:
                    gradient_idx = sampled

            with ctx.timed_phase("deem.gradient_embedding"):
                G_active = grad_embeddings_last_layer(
                    model,
                    ctx.X_train[gradient_idx],
                    y_soft[gradient_idx],
                    ctx.num_classes,
                    batch_size=batch_size,
                    device=str(device),
                    soft=True,
                )
            k = min(requested_k, len(gradient_idx))
            previous_selected = previous_selected[np.isin(previous_selected, gradient_idx)]
            if len(previous_selected) > 0:
                global_to_local = {int(value): idx for idx, value in enumerate(gradient_idx.tolist())}
                prev_local = np.array([global_to_local[int(value)] for value in previous_selected], dtype=np.int64)
                with ctx.timed_phase("deem.current_subset_mapping"):
                    _, _, current_error = _gradient_mapping_weights(
                        G_active,
                        prev_local,
                        row_batch_size=distance_row_batch_size,
                        center_batch_size=distance_candidate_batch_size,
                        mapping_sample_size=mapping_sample_size,
                        seed=ctx.seed + 740 + int(updates_done),
                        mapping_backend=mapping_backend,
                    )
            else:
                current_error = float("inf")
            should_rebuild = len(previous_selected) == 0 or rebuild_error is None or not np.isfinite(current_error)
            if deem_mode == "optimized" and update_strategy == "inherit_cluster" and len(previous_selected) > 0:
                should_rebuild = False
            elif not should_rebuild and current_error > 0.0:
                should_rebuild = (float(rebuild_error) / float(current_error)) < subset_update_threshold
            if should_rebuild:
                with ctx.timed_phase("deem.subset_rebuild"):
                    selected_idx, sample_weights, _mapping, subset_error, report = _paper_greedy_gradient_subset(
                        G_active,
                        gradient_idx,
                        k=k,
                        sample_size=greedy_sample_size,
                        sample_ratio=greedy_sample_ratio,
                        seed=ctx.seed + 760 + int(updates_done),
                        row_batch_size=distance_row_batch_size,
                        candidate_batch_size=distance_candidate_batch_size,
                        eval_sample_size=greedy_eval_sample_size,
                        greedy_steps_cap=greedy_steps_cap,
                        mapping_sample_size=mapping_sample_size,
                        greedy_batch_size=greedy_batch_size,
                        mapping_backend=mapping_backend,
                    )
                rebuild_error = float(subset_error)
                rebuilds += 1
                update_mode = "rebuild"
            else:
                with ctx.timed_phase("deem.subset_local_update"):
                    selected_idx, sample_weights, _mapping, subset_error, report = _paper_greedy_gradient_subset(
                        G_active,
                        gradient_idx,
                        k=k,
                        sample_size=greedy_sample_size,
                        sample_ratio=greedy_sample_ratio,
                        seed=ctx.seed + 780 + int(updates_done),
                        initial_selected_global=previous_selected,
                        row_batch_size=distance_row_batch_size,
                        candidate_batch_size=distance_candidate_batch_size,
                        eval_sample_size=greedy_eval_sample_size,
                        greedy_steps_cap=greedy_steps_cap,
                        mapping_sample_size=mapping_sample_size,
                        greedy_batch_size=greedy_batch_size,
                        mapping_backend=mapping_backend,
                    )
                local_updates += 1
                update_mode = "local_update"
            if deem_mode == "optimized" and len(gradient_idx) < len(active_idx) and len(sample_weights) > 0:
                sample_weights = sample_weights * np.float32(float(len(active_idx)) / float(len(gradient_idx)))
            report["candidate_samples"] = int(len(gradient_idx))
            report["active_samples"] = int(len(active_idx))
            if ctx.timing is not None:
                ctx.timing.add_duration("deem.greedy_selection", float(report.get("greedy_time_sec", 0.0)))
                ctx.timing.add_duration("deem.mapping_and_weighting", float(report.get("mapping_time_sec", 0.0)))
        else:
            selected_idx, sample_weights, report = _select_deem_subset(
                ctx,
                model,
                y_soft,
                losses,
                subset_fraction=subset_fraction,
                subset_size=subset_size,
                selector=selector,
                steps_cap=steps_cap,
                max_candidate_samples=max_candidate_samples,
                batch_size=batch_size,
                device=str(device),
                update_id=updates_done,
            )
        elapsed = time.perf_counter() - started
        selection_time_sec += elapsed
        update_reports.append(
            {
                "update": int(updates_done),
                "elapsed_sec": float(elapsed),
                "mean_soft_alpha": float(alpha.mean()),
                "removed": int(removed_this_update),
                "active_samples": int(active.sum()),
                "update_mode": update_mode,
                **report,
            }
        )

    max_epochs = max(1, int(cfg.max_epochs))
    stopped = False
    for _ in range(min(warmup_epochs, max_epochs)):
        all_idx = np.arange(ctx.n_samples, dtype=np.int64)
        all_weights = np.ones(ctx.n_samples, dtype=np.float32)
        stopped = train_epoch(all_idx, all_weights, y_soft, phase="warmup")
        if stopped:
            break

    if updates_done == 0:
        update_subset()

    while not stopped and epochs_ran < max_epochs:
        for _ in range(selection_interval):
            if stopped or epochs_ran >= max_epochs:
                break
            stopped = train_epoch(selected_idx, sample_weights, y_soft[selected_idx], phase="subset")
        if not stopped and epochs_ran < max_epochs:
            update_subset()

    if last_model_path is not None:
        torch.save(model.state_dict(), last_model_path)

    final_predictions = None
    if ctx.X_test is not None:
        with ctx.timed_phase("deem.final_prediction"):
            final_predictions = predict_classifier(model, ctx.X_test, batch_size=batch_size, device=str(device))

    metadata = {
        "paper": "Two Birds with One Stone: Efficient Deep Learning over Mislabeled Data through Subset Selection",
        "method_type": "training_coupled",
        "deem_mode": deem_mode,
        "continuous_training": True,
        "final_model_predictions": final_predictions is not None,
        "warmup_epochs": warmup_epochs,
        "selection_interval": selection_interval,
        "subset_fraction": subset_fraction,
        "requested_subset_size": subset_size,
        "subset_size": int(len(selected_idx)),
        "selector": selector,
        "soft_label_mode": soft_label_mode,
        "soft_label_window": soft_label_window,
        "mislabel_remove_ratio": mislabel_remove_ratio,
        "subset_update_threshold": subset_update_threshold,
        "greedy_sample_size": greedy_sample_size,
        "greedy_sample_ratio": greedy_sample_ratio,
        "greedy_eval_sample_size": greedy_eval_sample_size,
        "greedy_steps_cap": greedy_steps_cap,
        "mapping_sample_size": mapping_sample_size,
        "greedy_batch_size": greedy_batch_size,
        "mapping_backend": mapping_backend,
        "update_strategy": update_strategy,
        "distance_backend": mapping_backend if selector == "paper_greedy" else "gradient_selector",
        "distance_row_batch_size": distance_row_batch_size,
        "distance_candidate_batch_size": distance_candidate_batch_size,
        "mean_soft_alpha": float(alpha.mean()),
        "mean_observed_label_prob": float(observed_label_prob.mean()),
        "num_removed": int(predicted_noisy.sum()),
        "active_samples": int(active.sum()),
        "rebuilds": int(rebuilds),
        "local_updates": int(local_updates),
        "candidate_samples": int(update_reports[-1]["candidate_samples"]) if update_reports else int(ctx.n_samples),
        "updates": int(updates_done),
        "update_reports": update_reports,
        "epochs_ran": int(epochs_ran),
        "best_epoch": best_epoch,
        "best_train_loss": None if best_epoch is None else float(best_train_loss),
        "train_time_sec": float(train_time_sec),
        "selection_time_sec": float(selection_time_sec),
        "last_model_path": last_model_path,
        "best_model_path": best_model_path,
    }
    if soft_label_mode not in {"paper", "paper_loss_std", "loss_std"}:
        metadata["alpha_max"] = alpha_max
        metadata["gamma"] = gamma
    if selector != "paper_greedy" or deem_mode == "optimized":
        metadata["max_candidate_samples"] = max_candidate_samples

    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=selected_idx,
        sample_weights=sample_weights,
        soft_targets=y_soft,
        final_predictions=final_predictions,
        training_history=history,
        predicted_noisy_mask=predicted_noisy,
        metadata=metadata,
    )
