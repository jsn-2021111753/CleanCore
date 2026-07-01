"""Shared utilities for controlled label and feature corruptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


ERROR_CLEAN = 0
ERROR_LABEL_ONLY = 1
ERROR_FEATURE_ONLY = 2
ERROR_MIXED = 3


@dataclass
class NoiseResult:
    X: np.ndarray
    y: np.ndarray
    is_noisy: np.ndarray
    is_label_noisy: np.ndarray
    is_feature_noisy: np.ndarray
    error_type: np.ndarray
    corrupted_feature_mask: np.ndarray


def assign_error_types(n_samples: int, noise_rate: float, seed: int) -> np.ndarray:
    """Assign clean/label-only/feature-only/mixed types with a 2:2:1 ratio."""
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError("noise_rate must be in [0, 1].")
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")

    rng = np.random.default_rng(seed)
    error_type = np.zeros(n_samples, dtype=np.int8)
    n_noisy = int(round(n_samples * noise_rate))
    if n_noisy == 0:
        return error_type

    noisy_idx = rng.choice(n_samples, size=n_noisy, replace=False)
    n_label = int(round(n_noisy * 2 / 5))
    n_feature = int(round(n_noisy * 2 / 5))
    n_mixed = n_noisy - n_label - n_feature

    error_type[noisy_idx[:n_label]] = ERROR_LABEL_ONLY
    error_type[noisy_idx[n_label : n_label + n_feature]] = ERROR_FEATURE_ONLY
    error_type[noisy_idx[n_label + n_feature : n_label + n_feature + n_mixed]] = ERROR_MIXED
    return error_type


def feature_bounds(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mins = X.min(axis=0).astype(np.float32)
    maxs = X.max(axis=0).astype(np.float32)
    stds = X.std(axis=0).astype(np.float32)
    mutable = maxs > mins
    return mins, maxs, stds, mutable


def binary_feature_mask(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return (mins == 0) & (maxs == 1)


def force_different_in_bounds(
    current: np.float32,
    lower: np.float32,
    upper: np.float32,
    candidate: float,
) -> np.float32:
    current32 = np.float32(current)
    lower32 = np.float32(lower)
    upper32 = np.float32(upper)
    candidate32 = np.float32(np.clip(candidate, lower32, upper32))
    if candidate32 != current32:
        return candidate32
    if current32 < upper32:
        return np.float32(np.nextafter(current32, upper32))
    if current32 > lower32:
        return np.float32(np.nextafter(current32, lower32))
    return current32


def max_corrupted_features(n_features: int, max_corrupted_feature_frac: float) -> int:
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if not 0.0 < max_corrupted_feature_frac <= 1.0:
        raise ValueError("max_corrupted_feature_frac must be in (0, 1].")
    return max(1, int(np.floor(max_corrupted_feature_frac * n_features)))


def select_feature_indices(
    rng: np.random.Generator,
    mutable_features: np.ndarray,
    max_count: int,
) -> np.ndarray:
    candidates = np.where(mutable_features)[0]
    if len(candidates) == 0:
        return np.array([], dtype=np.int64)
    count = int(rng.integers(1, min(max_count, len(candidates)) + 1))
    return rng.choice(candidates, size=count, replace=False).astype(np.int64)


def choose_different_labels(y: np.ndarray, num_classes: int, rng: np.random.Generator) -> np.ndarray:
    if num_classes < 2:
        raise ValueError("Label corruption requires at least two classes.")
    offsets = rng.integers(1, num_classes, size=len(y), dtype=np.int64)
    return ((y.astype(np.int64) + offsets) % num_classes).astype(np.int64)


def corrupted_feature_counts(mask: np.ndarray) -> np.ndarray:
    return mask.sum(axis=1).astype(np.int64)


def make_noise_result(
    X: np.ndarray,
    y: np.ndarray,
    error_type: np.ndarray,
    corrupted_feature_mask: np.ndarray,
) -> NoiseResult:
    is_label_noisy = (error_type == ERROR_LABEL_ONLY) | (error_type == ERROR_MIXED)
    is_feature_noisy = (error_type == ERROR_FEATURE_ONLY) | (error_type == ERROR_MIXED)
    return NoiseResult(
        X=X.astype(np.float32, copy=False),
        y=y.astype(np.int64, copy=False),
        is_noisy=(error_type > 0),
        is_label_noisy=is_label_noisy,
        is_feature_noisy=is_feature_noisy,
        error_type=error_type.astype(np.int8, copy=False),
        corrupted_feature_mask=corrupted_feature_mask.astype(bool, copy=False),
    )
