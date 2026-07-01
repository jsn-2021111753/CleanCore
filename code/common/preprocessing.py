"""Common preprocessing for tabular NPZ arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardizePreprocessor:
    """Standardize features using train-only mean and standard deviation."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "StandardizePreprocessor":
        X = np.asarray(X_train, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X_train must be 2D.")
        mean = X.mean(axis=0).astype(np.float32)
        scale = X.std(axis=0).astype(np.float32)
        scale[scale == 0] = 1.0
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardizePreprocessor is not fitted.")
        X_arr = np.asarray(X, dtype=np.float32)
        return ((X_arr - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.fit(X_train)
        return self.transform(X_train), self.transform(X_test)

