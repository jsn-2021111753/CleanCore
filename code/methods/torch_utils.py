"""PyTorch utilities used by method-specific scoring steps."""

from __future__ import annotations

from dataclasses import fields
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from common.model import MLP
from common.seed import set_seed
from common.training import ArrayDataset, TrainingConfig, train_classifier
from methods.base import MethodContext


def training_config_from_context(ctx: MethodContext, max_epochs: Optional[int] = None) -> TrainingConfig:
    names = {f.name for f in fields(TrainingConfig)}
    data = {k: v for k, v in ctx.training_config.items() if k in names}
    cfg = TrainingConfig(**data)
    if max_epochs is not None:
        cfg.max_epochs = int(max_epochs)
    cfg.save_last_model = False
    cfg.save_best_model = False
    return cfg


def build_model(ctx: MethodContext, seed_offset: int = 0) -> MLP:
    set_seed(ctx.seed + int(seed_offset))
    hidden_dims = ctx.model_config.get("hidden_dims", (256, 128))
    dropout = float(ctx.model_config.get("dropout", 0.2))
    batch_norm = bool(ctx.model_config.get("batch_norm", True))
    return MLP(
        input_dim=ctx.input_dim,
        num_classes=ctx.num_classes,
        hidden_dims=tuple(hidden_dims),
        dropout=dropout,
        batch_norm=batch_norm,
    )


def train_aux_model(
    ctx: MethodContext,
    X: np.ndarray,
    y: np.ndarray,
    max_epochs: int,
    seed_offset: int = 0,
    sample_weights: Optional[np.ndarray] = None,
    soft_targets: Optional[np.ndarray] = None,
) -> MLP:
    model = build_model(ctx, seed_offset=seed_offset)
    cfg = training_config_from_context(ctx, max_epochs=max_epochs)
    train_classifier(
        model=model,
        X_train=np.asarray(X, dtype=np.float32),
        y_train=np.asarray(y, dtype=np.int64),
        cfg=cfg,
        output_dir=None,
        sample_weights=sample_weights,
        soft_targets=soft_targets,
        seed=ctx.seed + int(seed_offset),
    )
    return model


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 4096, device: str = "cpu") -> np.ndarray:
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)
    probs: list[np.ndarray] = []
    for start in range(0, len(X), int(batch_size)):
        xb = torch.tensor(np.asarray(X[start : start + int(batch_size)], dtype=np.float32), device=torch_device)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def per_sample_losses(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 4096,
    device: str = "cpu",
    soft_targets: Optional[np.ndarray] = None,
) -> np.ndarray:
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)
    ce = nn.CrossEntropyLoss(reduction="none")
    losses: list[np.ndarray] = []
    for start in range(0, len(X), int(batch_size)):
        xb = torch.tensor(np.asarray(X[start : start + int(batch_size)], dtype=np.float32), device=torch_device)
        logits = model(xb)
        if soft_targets is None:
            yb = torch.tensor(np.asarray(y[start : start + int(batch_size)], dtype=np.int64), device=torch_device)
            lv = ce(logits, yb)
        else:
            qb = torch.tensor(np.asarray(soft_targets[start : start + int(batch_size)], dtype=np.float32), device=torch_device)
            lv = -(qb * torch.log_softmax(logits, dim=1)).sum(dim=1)
        losses.append(lv.cpu().numpy().astype(np.float32))
    return np.concatenate(losses, axis=0)


def train_loss_trajectories(
    ctx: MethodContext,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    seed_offset: int = 0,
) -> Tuple[MLP, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    model = build_model(ctx, seed_offset=seed_offset)
    cfg = training_config_from_context(ctx, max_epochs=epochs)
    device = torch.device(cfg.device)
    model.to(device)
    dataset = ArrayDataset(X, y)
    generator = torch.Generator()
    generator.manual_seed(ctx.seed + int(seed_offset))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        generator=generator,
    )
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    ce = nn.CrossEntropyLoss(reduction="none")
    curves = np.zeros((len(y), int(epochs)), dtype=np.float32)
    for epoch in range(int(epochs)):
        model.train()
        for xb, yb, _idx in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = ce(model(xb), yb).mean()
            loss.backward()
            optimizer.step()
        curves[:, epoch] = per_sample_losses(model, X, y, batch_size=int(cfg.batch_size), device=cfg.device)
    return model, curves


def grad_embeddings_last_layer(
    model: MLP,
    X: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    batch_size: int,
    device: str = "cpu",
    soft: bool = False,
) -> np.ndarray:
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    with torch.no_grad():
        h0 = model.forward_features(torch.tensor(X[:1], dtype=torch.float32, device=torch_device))
    hidden_dim = int(h0.shape[1])
    out = np.zeros((len(X), int(num_classes) + int(num_classes) * hidden_dim), dtype=np.float32)
    for start in range(0, len(X), int(batch_size)):
        end = min(len(X), start + int(batch_size))
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=torch_device)
        with torch.no_grad():
            h = model.forward_features(xb)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)
            if soft:
                q = torch.tensor(np.asarray(targets[start:end], dtype=np.float32), device=torch_device)
            else:
                yb = torch.tensor(np.asarray(targets[start:end], dtype=np.int64), device=torch_device)
                q = torch.zeros_like(p)
                q.scatter_(1, yb.view(-1, 1), 1.0)
            delta = p - q
            outer = delta.unsqueeze(2) * h.unsqueeze(1)
            emb = torch.cat([delta, outer.reshape(outer.shape[0], -1)], dim=1)
        out[start:end] = emb.cpu().numpy().astype(np.float32)
    return out


def input_gradients(
    model: nn.Module,
    X: np.ndarray,
    soft_targets: np.ndarray,
    batch_size: int,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)
    grads = np.zeros_like(np.asarray(X, dtype=np.float32))
    for start in range(0, len(X), int(batch_size)):
        end = min(len(X), start + int(batch_size))
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=torch_device, requires_grad=True)
        qb = torch.tensor(soft_targets[start:end], dtype=torch.float32, device=torch_device)
        logits = model(xb)
        loss = -(qb * torch.log_softmax(logits, dim=1)).sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads[start:end] = xb.grad.detach().cpu().numpy().astype(np.float32)
    return grads


def matching_pursuit_nonnegative(G: np.ndarray, k: int, eps: float = 1e-8, steps_cap: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    G = np.asarray(G, dtype=np.float64)
    G = np.nan_to_num(G, nan=0.0, posinf=1e12, neginf=-1e12)
    scale = float(np.max(np.abs(G))) if G.size else 1.0
    if np.isfinite(scale) and scale > 1.0:
        G = G / scale
    n = int(G.shape[0])
    if k <= 0 or n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    if k >= n:
        return np.arange(n, dtype=np.int64), np.ones(n, dtype=np.float32)
    residual = G.sum(axis=0).astype(np.float64)
    norms2 = np.sum(G * G, axis=1).astype(np.float64) + float(eps)
    selected: list[int] = []
    weights: dict[int, float] = {}
    chosen = np.zeros(n, dtype=bool)
    steps = min(int(k), int(steps_cap)) if steps_cap and steps_cap > 0 else int(k)
    for _ in range(steps):
        raw_scores = np.einsum("ij,j->i", G, residual, optimize=False)
        scores = np.nan_to_num(raw_scores, nan=-np.inf, posinf=np.finfo(np.float64).max, neginf=-np.inf)
        scores[chosen] = -np.inf
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]):
            break
        alpha = max(0.0, float(scores[j] / norms2[j]))
        if not np.isfinite(alpha):
            alpha = 1e6
        alpha = min(alpha, 1e6)
        if alpha == 0.0:
            alpha = 1e-6
        chosen[j] = True
        selected.append(j)
        weights[j] = weights.get(j, 0.0) + alpha
        residual = np.nan_to_num(residual - alpha * G[j], nan=0.0, posinf=1e12, neginf=-1e12)
        residual_scale = float(np.max(np.abs(residual))) if residual.size else 1.0
        if np.isfinite(residual_scale) and residual_scale > 1e6:
            residual = residual / residual_scale
    if len(selected) < k:
        remaining = np.where(~chosen)[0]
        fill = remaining[np.argsort(-norms2[remaining])[: int(k) - len(selected)]]
        for j in fill.tolist():
            selected.append(int(j))
            weights[int(j)] = weights.get(int(j), 0.0) + 1e-6
    idx = np.array(selected[: int(k)], dtype=np.int64)
    w = np.array([weights[int(i)] for i in idx], dtype=np.float32)
    if float(w.sum()) > 0:
        w = w / float(w.sum()) * float(len(w))
    return idx, w.astype(np.float32)


def cords_orthogonal_mp_reg_nonnegative(
    G: np.ndarray,
    k: int,
    eps: float = 1e-4,
    lam: float = 0.0,
    steps_cap: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """CORDS-style positive OrthogonalMP_REG on per-sample gradients.

    CORDS calls ``OrthogonalMP_REG(X, Y, positive=True)`` with ``X`` shaped
    ``[gradient_dim, num_examples]`` and ``Y`` as the full-gradient target.  The
    implementation below keeps the same greedy selection and nonnegative
    least-squares update while adding numerical guards for tabular benchmarks.
    """

    G = np.asarray(G, dtype=np.float64)
    G = np.nan_to_num(G, nan=0.0, posinf=1e12, neginf=-1e12)
    scale = float(np.max(np.abs(G))) if G.size else 1.0
    if np.isfinite(scale) and scale > 1.0:
        G = G / scale
    n = int(G.shape[0])
    if k <= 0 or n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    if k >= n:
        return np.arange(n, dtype=np.int64), np.ones(n, dtype=np.float32)

    A = G.T
    b = G.sum(axis=0).astype(np.float64)
    normb = float(np.linalg.norm(b))
    if not np.isfinite(normb) or normb <= 0.0:
        return matching_pursuit_nonnegative(G, k=k, eps=max(float(eps), 1e-12))

    AT = A.T
    residual = b.copy()
    selected: list[int] = []
    coeffs = np.zeros(0, dtype=np.float64)
    tol = max(float(eps), 1e-12)
    ridge = max(float(lam), 0.0)
    greedy_steps = min(int(k), int(steps_cap)) if int(steps_cap) > 0 else int(k)
    for _ in range(greedy_steps):
        if float(np.linalg.norm(residual)) / normb < tol:
            break
        projections = AT.dot(residual)
        projections = np.nan_to_num(projections, nan=-np.inf, posinf=np.finfo(np.float64).max, neginf=-np.inf)
        projections[selected] = -np.inf
        index = int(np.argmax(projections))
        if not np.isfinite(projections[index]) or index in selected:
            break
        selected.append(index)
        A_i = A[:, selected].T
        gram = A_i.dot(A_i.T)
        if ridge > 0.0:
            gram = gram + ridge * np.eye(gram.shape[0], dtype=np.float64)
        rhs = A_i.dot(b)
        try:
            coeffs = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        while len(selected) > 0 and np.any(coeffs < 0.0):
            drop = int(np.argmin(coeffs))
            selected.pop(drop)
            if not selected:
                coeffs = np.zeros(0, dtype=np.float64)
                break
            A_i = A[:, selected].T
            gram = A_i.dot(A_i.T)
            if ridge > 0.0:
                gram = gram + ridge * np.eye(gram.shape[0], dtype=np.float64)
            rhs = A_i.dot(b)
            try:
                coeffs = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        if len(selected) == 0:
            break
        residual = np.nan_to_num(b - A_i.T.dot(coeffs), nan=0.0, posinf=1e12, neginf=-1e12)

    if len(selected) < k:
        remaining = np.array([i for i in range(n) if i not in set(selected)], dtype=np.int64)
        if len(remaining) > 0:
            norms = np.sum(G[remaining] * G[remaining], axis=1)
            fill = remaining[np.argsort(-norms)[: int(k) - len(selected)]]
            selected.extend(int(i) for i in fill.tolist())
            coeffs = np.concatenate([coeffs, np.ones(len(fill), dtype=np.float64)])

    idx = np.array(selected[: int(k)], dtype=np.int64)
    weights = np.array(coeffs[: len(idx)], dtype=np.float64)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    weights[weights <= 0.0] = 1.0
    return idx, weights.astype(np.float32)
