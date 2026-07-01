"""HERDING subset selection baseline.

Based on Welling (ICML 2009): greedily select samples whose empirical feature
moments match the full class-conditional feature means.
"""

from __future__ import annotations

import numpy as np

from common.interfaces import MethodOutput
from methods.base import MethodContext
from methods.common import allocate_classwise, dedupe_selected


def _transform_features(X: np.ndarray, mode: str) -> np.ndarray:
    mode_key = str(mode or "raw").lower()
    if mode_key in {"raw", "none"}:
        return X
    X64 = np.asarray(X, dtype=np.float64)
    if mode_key in {"standardize", "zscore"}:
        mean = X64.mean(axis=0, keepdims=True)
        std = X64.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        return (X64 - mean) / std
    if mode_key in {"center", "centered"}:
        return X64 - X64.mean(axis=0, keepdims=True)
    if mode_key in {"l2", "normalize", "row_normalize"}:
        norm = np.linalg.norm(X64, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        return X64 / norm
    raise ValueError(f"Unknown herding feature_transform: {mode}")


def _candidate_pool(X: np.ndarray, m: int, pool_size: int, strategy: str, seed: int) -> np.ndarray:
    n = int(len(X))
    if pool_size <= 0 or pool_size >= n:
        return np.arange(n, dtype=np.int64)
    k = max(int(m), min(int(pool_size), n))
    strategy_key = str(strategy or "random").lower()
    rng = np.random.default_rng(int(seed))
    if strategy_key == "random":
        return np.sort(rng.choice(np.arange(n, dtype=np.int64), size=k, replace=False)).astype(np.int64)

    X64 = np.asarray(X, dtype=np.float64)
    mean = X64.mean(axis=0, keepdims=True)
    dist2 = np.sum((X64 - mean) ** 2, axis=1)
    if strategy_key in {"nearest_mean", "near_mean", "central"}:
        return np.argsort(dist2)[:k].astype(np.int64)
    if strategy_key in {"farthest_mean", "far_mean", "boundary"}:
        return np.argsort(-dist2)[:k].astype(np.int64)
    if strategy_key in {"mixed_mean", "mixed"}:
        near = k // 2
        far = k - near
        picked = np.concatenate([np.argsort(dist2)[:near], np.argsort(-dist2)[:far]])
        return np.unique(picked).astype(np.int64)[:k]
    raise ValueError(f"Unknown herding candidate_strategy: {strategy}")


def herding_sequence(X: np.ndarray, m: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = int(len(X))
    if m <= 0:
        return np.array([], dtype=np.int64)
    if m >= n:
        return np.arange(n, dtype=np.int64)
    mu = X.mean(axis=0).astype(np.float64)
    w = mu.copy()
    selected = np.empty(int(m), dtype=np.int64)
    available = np.ones(n, dtype=bool)
    for t in range(int(m)):
        scores = np.einsum("ij,j->i", X, w, optimize=False)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float64).max, neginf=-np.inf)
        scores[~available] = -np.inf
        j = int(np.argmax(scores))
        selected[t] = j
        available[j] = False
        w = w + mu - X[j]
        w = np.clip(w, -1e6, 1e6)
    return selected


def run(ctx: MethodContext) -> MethodOutput:
    subset_fraction = float(ctx.param("subset_fraction", 0.10))
    subset_size = int(ctx.param("subset_size", 0))
    min_per_class = int(ctx.param("min_per_class", 1))
    feature_transform = str(ctx.param("feature_transform", "raw"))
    candidate_pool_size = int(ctx.param("candidate_pool_size", 0))
    candidate_strategy = str(ctx.param("candidate_strategy", "random"))
    tie_break_seed_offset = int(ctx.param("tie_break_seed_offset", 0))
    with ctx.timed_phase("herding.budget_allocation"):
        alloc = allocate_classwise(ctx.y_train, ctx.num_classes, subset_fraction, subset_size, min_per_class)

    with ctx.timed_phase("herding.greedy_selection"):
        selected: list[int] = []
        weights: list[float] = []
        per_class = {}
        for c in range(ctx.num_classes):
            idx_c = np.where(ctx.y_train == c)[0].astype(np.int64)
            m_c = int(alloc.get(c, 0))
            if len(idx_c) == 0 or m_c == 0:
                per_class[str(c)] = {"n_c": int(len(idx_c)), "selected": 0}
                continue
            X_c = ctx.X_train[idx_c]
            # Preserve the historical exact path unless a new optional knob is
            # explicitly enabled in a config.
            if feature_transform.lower() in {"raw", "none"} and candidate_pool_size <= 0:
                local = herding_sequence(X_c, m_c)
                chosen = idx_c[local]
                pool_size_used = int(len(idx_c))
            else:
                X_work = _transform_features(X_c, feature_transform)
                pool = _candidate_pool(
                    X_work,
                    m=m_c,
                    pool_size=candidate_pool_size,
                    strategy=candidate_strategy,
                    seed=ctx.seed + tie_break_seed_offset + int(c),
                )
                local = herding_sequence(X_work[pool], m_c)
                chosen = idx_c[pool[local]]
                pool_size_used = int(len(pool))
            weight_each = float(len(idx_c)) / float(max(1, len(chosen)))
            selected.extend(int(i) for i in chosen)
            weights.extend([weight_each] * len(chosen))
            per_class[str(c)] = {
                "n_c": int(len(idx_c)),
                "selected": int(len(chosen)),
                "weight_each": weight_each,
                "candidate_pool_size": pool_size_used,
            }

    with ctx.timed_phase("herding.weight_aggregation"):
        selected_idx, sample_weights = dedupe_selected(selected, weights)
    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=selected_idx,
        sample_weights=sample_weights,
        metadata={
            "paper": "Herding Dynamical Weights to Learn",
            "method_type": "data_processing",
            "subset_fraction": subset_fraction,
            "subset_size": subset_size,
            "feature_transform": feature_transform,
            "candidate_pool_size": candidate_pool_size,
            "candidate_strategy": candidate_strategy,
            "per_class": per_class,
        },
    )
