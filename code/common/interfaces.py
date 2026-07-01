"""Shared method input/output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MethodOutput:
    selected_indices: np.ndarray
    sample_weights: np.ndarray
    corrected_labels: Optional[np.ndarray] = None
    corrected_features: Optional[np.ndarray] = None
    soft_targets: Optional[np.ndarray] = None
    predicted_noisy_mask: Optional[np.ndarray] = None
    final_predictions: Optional[np.ndarray] = None
    training_history: List[Dict[str, object]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        n_samples: int,
        selected_indices: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        corrected_labels: Optional[np.ndarray] = None,
        corrected_features: Optional[np.ndarray] = None,
        soft_targets: Optional[np.ndarray] = None,
        predicted_noisy_mask: Optional[np.ndarray] = None,
        final_predictions: Optional[np.ndarray] = None,
        training_history: Optional[List[Dict[str, object]]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> "MethodOutput":
        if selected_indices is None:
            selected_indices = np.arange(n_samples, dtype=np.int64)
        else:
            selected_indices = np.asarray(selected_indices, dtype=np.int64)

        if sample_weights is None:
            sample_weights = np.ones(len(selected_indices), dtype=np.float32)
        else:
            sample_weights = np.asarray(sample_weights, dtype=np.float32)

        if len(sample_weights) != len(selected_indices):
            raise ValueError("sample_weights length must match selected_indices length.")

        if corrected_labels is not None:
            corrected_labels = np.asarray(corrected_labels, dtype=np.int64)
            if len(corrected_labels) != n_samples:
                raise ValueError("corrected_labels must have length n_samples.")
        if corrected_features is not None:
            corrected_features = np.asarray(corrected_features, dtype=np.float32)
            if corrected_features.ndim != 2 or len(corrected_features) != n_samples:
                raise ValueError("corrected_features must be a 2D array with n_samples rows.")
        if soft_targets is not None:
            soft_targets = np.asarray(soft_targets, dtype=np.float32)
            if soft_targets.ndim != 2 or len(soft_targets) != n_samples:
                raise ValueError("soft_targets must be a 2D array with n_samples rows.")
            row_sums = soft_targets.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            soft_targets = soft_targets / row_sums
        if predicted_noisy_mask is not None:
            predicted_noisy_mask = np.asarray(predicted_noisy_mask, dtype=bool)
            if len(predicted_noisy_mask) != n_samples:
                raise ValueError("predicted_noisy_mask must have length n_samples.")
        if final_predictions is not None:
            final_predictions = np.asarray(final_predictions, dtype=np.int64)
            if final_predictions.ndim != 1:
                raise ValueError("final_predictions must be a 1D array.")

        return cls(
            selected_indices=selected_indices,
            sample_weights=sample_weights,
            corrected_labels=corrected_labels,
            corrected_features=corrected_features,
            soft_targets=soft_targets,
            predicted_noisy_mask=predicted_noisy_mask,
            final_predictions=final_predictions,
            training_history=[] if training_history is None else list(training_history),
            metadata={} if metadata is None else dict(metadata),
        )
