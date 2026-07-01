"""Shared evaluation metrics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }

