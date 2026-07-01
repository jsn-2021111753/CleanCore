"""Systematic label and feature shift corruptions within legal domains."""

from __future__ import annotations

import numpy as np

from noise.injector import (
    ERROR_FEATURE_ONLY,
    ERROR_LABEL_ONLY,
    ERROR_MIXED,
    NoiseResult,
    binary_feature_mask,
    feature_bounds,
    force_different_in_bounds,
    make_noise_result,
    max_corrupted_features,
    select_feature_indices,
)


def apply_shift_noise(
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    error_type: np.ndarray,
    seed: int,
    max_corrupted_feature_frac: float,
    shift_scale: float,
) -> NoiseResult:
    X_noisy = X_clean.astype(np.float32, copy=True)
    y_noisy = y_clean.astype(np.int64, copy=True)
    rng = np.random.default_rng(seed)

    num_classes = int(y_clean.max()) + 1
    label_idx = np.where((error_type == ERROR_LABEL_ONLY) | (error_type == ERROR_MIXED))[0]
    if len(label_idx) > 0:
        y_noisy[label_idx] = ((y_clean[label_idx].astype(np.int64) + 1) % num_classes).astype(np.int64)

    mins, maxs, stds, mutable = feature_bounds(X_clean)
    binary_mask = binary_feature_mask(mins, maxs)
    max_count = max_corrupted_features(X_clean.shape[1], max_corrupted_feature_frac)
    directions = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=X_clean.shape[1])
    corrupted_feature_mask = np.zeros(X_clean.shape, dtype=bool)

    feature_rows = np.where((error_type == ERROR_FEATURE_ONLY) | (error_type == ERROR_MIXED))[0]
    for row in feature_rows:
        cols = select_feature_indices(rng, mutable, max_count)
        if len(cols) == 0:
            continue
        for col in cols:
            if binary_mask[col]:
                X_noisy[row, col] = 1.0 - X_noisy[row, col]
            else:
                delta = directions[col] * np.float32(shift_scale) * stds[col]
                X_noisy[row, col] = force_different_in_bounds(
                    X_noisy[row, col],
                    mins[col],
                    maxs[col],
                    float(X_noisy[row, col] + delta),
                )
        corrupted_feature_mask[row, cols] = True

    np.clip(X_noisy, mins, maxs, out=X_noisy)
    return make_noise_result(X_noisy, y_noisy, error_type, corrupted_feature_mask)
