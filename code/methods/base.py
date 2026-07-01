"""Shared context passed to all method implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class MethodContext:
    X_train: np.ndarray
    y_train: np.ndarray
    num_classes: int
    seed: int
    X_test: Optional[np.ndarray] = None
    output_dir: Optional[Path] = None
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    method_config: Dict[str, Any] = field(default_factory=dict)
    timing: Optional[Any] = None

    def __post_init__(self) -> None:
        self.X_train = np.asarray(self.X_train, dtype=np.float32)
        self.y_train = np.asarray(self.y_train, dtype=np.int64)
        if self.X_train.ndim != 2:
            raise ValueError("X_train must be 2D.")
        if self.X_test is not None:
            self.X_test = np.asarray(self.X_test, dtype=np.float32)
            if self.X_test.ndim != 2:
                raise ValueError("X_test must be 2D.")
            if self.X_test.shape[1] != self.X_train.shape[1]:
                raise ValueError("X_test must have the same feature dimension as X_train.")
        if self.y_train.ndim != 1:
            raise ValueError("y_train must be 1D.")
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_train/y_train length mismatch.")
        self.num_classes = int(self.num_classes)
        self.seed = int(self.seed)

    @property
    def n_samples(self) -> int:
        return int(len(self.y_train))

    @property
    def input_dim(self) -> int:
        return int(self.X_train.shape[1])

    def param(self, name: str, default: Any) -> Any:
        return self.method_config.get(name, default)

    def timed_phase(self, name: str):
        if self.timing is None:
            return nullcontext()
        return self.timing.phase(str(name))

    def timing_totals(self) -> Dict[str, float]:
        if self.timing is None:
            return {}
        return self.timing.totals()

    def timing_counts(self) -> Dict[str, int]:
        if self.timing is None:
            return {}
        return self.timing.counts()
