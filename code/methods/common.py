"""Shared numerical helpers for paper method implementations."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.zeros((len(y), int(num_classes)), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def subset_size_from_fraction(n: int, fraction: float, min_size: int = 1) -> int:
    if n <= 0:
        return 0
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return int(min(n, max(int(min_size), int(round(fraction * n)))))


def allocate_classwise(
    y: np.ndarray,
    num_classes: int,
    subset_fraction: float,
    subset_size: int = 0,
    min_per_class: int = 1,
) -> Dict[int, int]:
    y = np.asarray(y, dtype=np.int64)
    counts = {c: int(np.sum(y == c)) for c in range(int(num_classes))}
    n = int(len(y))
    if n == 0:
        return {c: 0 for c in range(int(num_classes))}

    if subset_size and subset_size > 0:
        total = int(min(max(1, subset_size), n))
    else:
        total = subset_size_from_fraction(n, subset_fraction, min_size=min_per_class)

    alloc: Dict[int, int] = {}
    for c in range(int(num_classes)):
        n_c = counts[c]
        if n_c == 0:
            alloc[c] = 0
            continue
        m_c = int(np.floor(total * (n_c / n)))
        m_c = min(n_c, max(int(min_per_class), m_c))
        alloc[c] = m_c

    current = sum(alloc.values())
    order = sorted(range(int(num_classes)), key=lambda c: counts[c], reverse=True)
    while current < total:
        changed = False
        for c in order:
            if current >= total:
                break
            if alloc[c] < counts[c]:
                alloc[c] += 1
                current += 1
                changed = True
        if not changed:
            break
    while current > total:
        changed = False
        for c in reversed(order):
            if current <= total:
                break
            floor = int(min_per_class) if counts[c] > 0 else 0
            if alloc[c] > floor:
                alloc[c] -= 1
                current -= 1
                changed = True
        if not changed:
            break
    return alloc


def dedupe_selected(indices: Iterable[int], weights: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    totals: Dict[int, float] = {}
    order: list[int] = []
    for idx, weight in zip(indices, weights):
        i = int(idx)
        if i not in totals:
            order.append(i)
            totals[i] = 0.0
        totals[i] += float(weight)
    selected = np.array(order, dtype=np.int64)
    sample_weights = np.array([totals[int(i)] for i in selected], dtype=np.float32)
    return selected, sample_weights


def bounded_classwise_candidates(
    y: np.ndarray,
    num_classes: int,
    max_candidates: int,
    seed: int,
    scores: np.ndarray | None = None,
    largest: bool = True,
) -> np.ndarray:
    """Return a class-aware candidate pool capped by max_candidates.

    When scores are supplied, candidates are chosen by score within each class;
    otherwise they are sampled without replacement.
    """

    y = np.asarray(y, dtype=np.int64)
    n = int(len(y))
    cap = int(max_candidates)
    if cap <= 0 or cap >= n:
        return np.arange(n, dtype=np.int64)
    alloc = allocate_classwise(y, num_classes, subset_fraction=1.0, subset_size=cap, min_per_class=1)
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    used: set[int] = set()
    score_arr = None if scores is None else np.asarray(scores, dtype=np.float64)
    for c in range(int(num_classes)):
        idx_c = np.where(y == c)[0].astype(np.int64)
        k_c = min(int(alloc.get(c, 0)), len(idx_c))
        if k_c <= 0:
            continue
        if score_arr is None:
            picked = rng.choice(idx_c, size=k_c, replace=False)
        else:
            local_scores = score_arr[idx_c]
            order = np.argsort(local_scores)
            if largest:
                order = order[::-1]
            picked = idx_c[order[:k_c]]
        for i in picked.tolist():
            used.add(int(i))
            selected.append(int(i))

    if len(selected) < cap:
        remaining = np.array([i for i in range(n) if i not in used], dtype=np.int64)
        if len(remaining) > 0:
            need = min(cap - len(selected), len(remaining))
            if score_arr is None:
                fill = rng.choice(remaining, size=need, replace=False)
            else:
                order = np.argsort(score_arr[remaining])
                if largest:
                    order = order[::-1]
                fill = remaining[order[:need]]
            selected.extend(int(i) for i in fill.tolist())
    return np.array(selected[:cap], dtype=np.int64)


def classwise_topk(
    y: np.ndarray,
    scores: np.ndarray,
    num_classes: int,
    subset_fraction: float,
    subset_size: int = 0,
    min_per_class: int = 1,
    largest: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    alloc = allocate_classwise(y, num_classes, subset_fraction, subset_size, min_per_class)
    selected: list[int] = []
    weights: list[float] = []
    per_class: Dict[str, object] = {}
    for c in range(int(num_classes)):
        idx_c = np.where(y == c)[0].astype(np.int64)
        k_c = int(alloc.get(c, 0))
        if len(idx_c) == 0 or k_c <= 0:
            per_class[str(c)] = {"n_c": int(len(idx_c)), "selected": 0}
            continue
        sc = scores[idx_c]
        valid = np.isfinite(sc)
        idx_valid = idx_c[valid]
        sc_valid = sc[valid]
        if len(idx_valid) == 0:
            idx_valid = idx_c
            sc_valid = np.zeros(len(idx_c), dtype=np.float64)
        order = np.argsort(sc_valid)
        if largest:
            order = order[::-1]
        picked = idx_valid[order[: min(k_c, len(idx_valid))]]
        selected.extend(int(i) for i in picked)
        weight_each = float(len(idx_c)) / float(max(1, len(picked)))
        weights.extend([weight_each] * len(picked))
        per_class[str(c)] = {"n_c": int(len(idx_c)), "selected": int(len(picked)), "weight_each": weight_each}
    if not selected:
        selected = [int(np.argmax(scores))]
        weights = [1.0]
    idx, w = dedupe_selected(selected, weights)
    return idx, w, {"per_class": per_class}


def nearest_center_weights(X: np.ndarray, selected: np.ndarray, sample_weights: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) == 0:
        return np.array([], dtype=np.float32)
    weights = np.ones(len(X), dtype=np.float32) if sample_weights is None else np.asarray(sample_weights, dtype=np.float32)
    centers = X[selected]
    center_norm = np.sum(centers * centers, axis=1)
    assigned = np.zeros(len(selected), dtype=np.float32)
    for start in range(0, len(X), 4096):
        xb = X[start : start + 4096]
        xb_norm = np.sum(xb * xb, axis=1, keepdims=True)
        dots = np.einsum("ij,kj->ik", xb, centers, optimize=False)
        dist2 = xb_norm + center_norm.reshape(1, -1) - 2.0 * dots
        nearest = np.argmin(dist2, axis=1)
        for j, w in zip(nearest, weights[start : start + len(xb)]):
            assigned[int(j)] += float(w)
    assigned[assigned <= 0.0] = 1.0
    return assigned.astype(np.float32)


def farthest_first(rep_vec: np.ndarray, candidate_mask: np.ndarray, subset_size: int, seed: int) -> np.ndarray:
    rep_vec = np.asarray(rep_vec, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    candidates = np.where(candidate_mask)[0].astype(np.int64)
    if len(candidates) == 0:
        raise RuntimeError("No candidates available for farthest-first selection.")
    k = min(max(1, int(subset_size)), len(candidates))
    rng = np.random.default_rng(int(seed))
    cand_vec = rep_vec[candidates]
    norms = np.linalg.norm(cand_vec, axis=1)
    first_pos = int(np.argmax(norms)) if float(norms.max()) > 1e-12 else int(rng.integers(0, len(candidates)))
    selected_pos = [first_pos]
    dist2 = np.sum((cand_vec - cand_vec[first_pos]) ** 2, axis=1)
    dist2[first_pos] = -1.0
    for _ in range(1, k):
        j_pos = int(np.argmax(dist2))
        if dist2[j_pos] <= 1e-18:
            remaining = np.array([i for i in range(len(candidates)) if i not in set(selected_pos)], dtype=np.int64)
            if len(remaining) == 0:
                break
            j_pos = int(rng.choice(remaining))
        selected_pos.append(j_pos)
        new_d2 = np.sum((cand_vec - cand_vec[j_pos]) ** 2, axis=1)
        dist2 = np.minimum(dist2, new_d2)
        dist2[np.array(selected_pos, dtype=np.int64)] = -1.0
    return candidates[np.array(selected_pos, dtype=np.int64)]
