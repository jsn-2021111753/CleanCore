"""GoodCore baseline for coreset selection over incomplete data.

The implementation follows the paper idea of expected-distance coreset
selection over possible repairs. Numeric NaNs are modeled by column means and
missing-value variance terms; if no NaNs are present, the same objective reduces
to distance-based coreset selection in the standardized feature space.
"""

from __future__ import annotations

import numpy as np

from common.interfaces import MethodOutput
from methods.base import MethodContext
from methods.common import allocate_classwise


def expected_distance_components(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_raw = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X_raw, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    std = np.nanstd(X_raw, axis=0)
    std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    X_std = (X_raw - mu) / std
    miss = np.isnan(X_std)
    col_mu = np.nanmean(X_std, axis=0)
    col_mu = np.where(np.isfinite(col_mu), col_mu, 0.0)
    col_var = np.nanvar(X_std, axis=0)
    col_var = np.where(np.isfinite(col_var) & (col_var >= 0.0), col_var, 1.0)
    X_imp = X_std.copy()
    if miss.any():
        X_imp[miss] = np.take(col_mu, np.where(miss)[1])
    row_sq = np.sum(X_imp * X_imp, axis=1)
    miss_var_sum = np.sum(miss * col_var.reshape(1, -1), axis=1)
    return X_imp.astype(np.float32), row_sq.astype(np.float64), miss_var_sum.astype(np.float64)


def greedy_expected_subset(
    X_imp: np.ndarray,
    row_sq: np.ndarray,
    miss_var_sum: np.ndarray,
    k: int,
    h_sample_size: int,
    seed: int,
    batch_size: int = 1,
    utility_eval_size: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = int(len(X_imp))
    if k >= n:
        return np.arange(n, dtype=np.int64)
    candidates_left = np.arange(n, dtype=np.int64)
    selected: list[int] = []
    eval_cap = int(utility_eval_size)
    if eval_cap > 0 and eval_cap < n:
        eval_idx = rng.choice(np.arange(n, dtype=np.int64), size=eval_cap, replace=False).astype(np.int64)
    else:
        eval_idx = np.arange(n, dtype=np.int64)
    X_eval = X_imp[eval_idx]
    row_eval = row_sq[eval_idx]
    miss_eval = miss_var_sum[eval_idx]
    best_dist = np.full(len(eval_idx), np.inf, dtype=np.float64)
    h = max(1, int(h_sample_size))
    batch = max(1, int(batch_size))
    while len(selected) < int(k) and len(candidates_left) > 0:
        candidate_pool_size = max(h, min(batch, int(k) - len(selected)))
        cand = (
            candidates_left
            if len(candidates_left) <= candidate_pool_size
            else rng.choice(candidates_left, size=candidate_pool_size, replace=False)
        )
        V = X_imp[cand]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            dist2 = row_eval[:, None] - 2.0 * (X_eval @ V.T) + row_sq[cand][None, :] + miss_eval[:, None] + miss_var_sum[cand][None, :]
        dist2 = np.nan_to_num(dist2, nan=np.inf, posinf=np.inf, neginf=0.0)
        improvement = np.maximum(0.0, best_dist[:, None] - dist2).sum(axis=0)
        take = min(batch, int(k) - len(selected), len(cand))
        chosen_pos = np.argsort(-improvement)[:take].astype(np.int64)
        chosen = cand[chosen_pos].astype(np.int64)
        selected.extend(int(i) for i in chosen.tolist())
        best_dist = np.minimum(best_dist, np.min(dist2[:, chosen_pos], axis=1))
        chosen_set = set(int(i) for i in chosen.tolist())
        candidates_left = np.array([i for i in candidates_left if int(i) not in chosen_set], dtype=np.int64)
    return np.array(selected, dtype=np.int64)


def assign_expected_weights(
    X_imp: np.ndarray,
    row_sq: np.ndarray,
    miss_var_sum: np.ndarray,
    selected: np.ndarray,
    row_batch_size: int = 4096,
    center_batch_size: int = 1024,
) -> np.ndarray:
    if len(selected) == 0:
        return np.array([], dtype=np.float32)
    centers = X_imp[selected]
    c_sq = row_sq[selected]
    c_miss = miss_var_sum[selected]
    assigned = np.zeros(len(selected), dtype=np.float32)
    rows = max(1, int(row_batch_size))
    centers_per_block = max(1, int(center_batch_size))
    for start in range(0, len(X_imp), rows):
        xb = X_imp[start : start + rows]
        best = np.full(len(xb), np.inf, dtype=np.float64)
        nearest = np.zeros(len(xb), dtype=np.int64)
        for center_start in range(0, len(centers), centers_per_block):
            center_end = min(len(centers), center_start + centers_per_block)
            center_block = centers[center_start:center_end]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                dist2 = (
                    row_sq[start : start + len(xb), None]
                    - 2.0 * (xb @ center_block.T)
                    + c_sq[center_start:center_end][None, :]
                    + miss_var_sum[start : start + len(xb), None]
                    + c_miss[center_start:center_end][None, :]
                )
            dist2 = np.nan_to_num(dist2, nan=np.inf, posinf=np.inf, neginf=0.0)
            local = np.argmin(dist2, axis=1).astype(np.int64)
            local_best = dist2[np.arange(len(xb)), local]
            improved = local_best < best
            best[improved] = local_best[improved]
            nearest[improved] = center_start + local[improved]
        for j in nearest:
            assigned[int(j)] += 1.0
    assigned[assigned <= 0.0] = 1.0
    return assigned.astype(np.float32)


def run(ctx: MethodContext) -> MethodOutput:
    subset_fraction = float(ctx.param("subset_fraction", ctx.param("coreset_fraction", 0.10)))
    subset_size = int(ctx.param("subset_size", ctx.param("coreset_size", 0)))
    min_per_class = int(ctx.param("min_per_class", 1))
    h_sample_size = int(ctx.param("h_sample_size", 200))
    goodcore_batch_size = max(1, int(ctx.param("goodcore_batch_size", ctx.param("batch_greedy_size", 1))))
    utility_eval_size = int(ctx.param("utility_eval_size", 0))
    per_class = bool(ctx.param("per_class", True))
    weighting = str(ctx.param("weighting", "nearest")).lower()
    weight_row_batch_size = int(ctx.param("weight_row_batch_size", 4096))
    weight_center_batch_size = int(ctx.param("weight_center_batch_size", 1024))

    with ctx.timed_phase("goodcore.expected_distance_preprocessing"):
        X_imp, row_sq, miss_var_sum = expected_distance_components(ctx.X_train)
    if subset_size <= 0:
        subset_size = max(1, int(round(subset_fraction * ctx.n_samples)))
    subset_size = min(subset_size, ctx.n_samples)

    with ctx.timed_phase("goodcore.greedy_selection"):
        if per_class:
            alloc = allocate_classwise(ctx.y_train, ctx.num_classes, subset_fraction, subset_size, min_per_class)
            selected_all: list[int] = []
            for c in range(ctx.num_classes):
                idx_c = np.where(ctx.y_train == c)[0].astype(np.int64)
                k_c = int(alloc.get(c, 0))
                if len(idx_c) == 0 or k_c <= 0:
                    continue
                local = greedy_expected_subset(
                    X_imp[idx_c],
                    row_sq[idx_c],
                    miss_var_sum[idx_c],
                    k=k_c,
                    h_sample_size=h_sample_size,
                    seed=ctx.seed + 1000 + c,
                    batch_size=goodcore_batch_size,
                    utility_eval_size=utility_eval_size,
                )
                selected_all.extend(idx_c[local].astype(np.int64)[:k_c].tolist())
            selected = np.array(list(dict.fromkeys(selected_all)), dtype=np.int64)
        else:
            selected = greedy_expected_subset(
                X_imp,
                row_sq,
                miss_var_sum,
                k=subset_size,
                h_sample_size=h_sample_size,
                seed=ctx.seed + 1000,
                batch_size=goodcore_batch_size,
                utility_eval_size=utility_eval_size,
            )
    with ctx.timed_phase("goodcore.weight_assignment"):
        if weighting == "nearest":
            weights = assign_expected_weights(
                X_imp,
                row_sq,
                miss_var_sum,
                selected,
                row_batch_size=weight_row_batch_size,
                center_batch_size=weight_center_batch_size,
            )
        elif weighting == "unit":
            weights = np.ones(len(selected), dtype=np.float32)
        else:
            raise ValueError(f"Unknown GoodCore weighting: {weighting}")
    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=selected,
        sample_weights=weights,
        metadata={
            "paper": "GoodCore: Data-effective and Data-efficient Machine Learning through Coreset Selection over Incomplete Data",
            "method_type": "data_processing",
            "subset_fraction": subset_fraction,
            "subset_size": int(len(selected)),
            "per_class": per_class,
            "weighting": weighting,
            "h_sample_size": h_sample_size,
            "goodcore_batch_size": goodcore_batch_size,
            "batch_greedy_size": goodcore_batch_size,
            "weight_row_batch_size": weight_row_batch_size,
            "weight_center_batch_size": weight_center_batch_size,
            "utility_eval_size": utility_eval_size,
        },
    )
