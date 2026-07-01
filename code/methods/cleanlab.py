"""Cleanlab/confident-learning baseline using the official Cleanlab API.

The benchmark still trains its own out-of-fold auxiliary models so all methods
share the same model family and training controls. Label issue detection itself
is delegated to ``cleanlab.filter.find_label_issues``.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from common.interfaces import MethodOutput
from methods.base import MethodContext
from methods.torch_utils import predict_proba, train_aux_model


def _oof_probabilities(ctx: MethodContext, cv_folds: int, cv_epochs: int) -> np.ndarray:
    y = ctx.y_train
    min_count = int(min(np.sum(y == c) for c in np.unique(y)))
    folds = int(min(max(2, cv_folds), max(2, min_count)))
    if min_count < 2:
        model = train_aux_model(ctx, ctx.X_train, y, max_epochs=cv_epochs, seed_offset=201)
        return predict_proba(model, ctx.X_train, batch_size=int(ctx.training_config.get("batch_size", 1024)))

    oof = np.zeros((ctx.n_samples, ctx.num_classes), dtype=np.float32)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=ctx.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(ctx.n_samples), y), start=1):
        model = train_aux_model(
            ctx,
            ctx.X_train[train_idx],
            y[train_idx],
            max_epochs=cv_epochs,
            seed_offset=200 + fold,
        )
        oof[val_idx] = predict_proba(
            model,
            ctx.X_train[val_idx],
            batch_size=int(ctx.training_config.get("batch_size", 1024)),
        )
    return oof


def find_label_issues(
    y: np.ndarray,
    probs: np.ndarray,
    frac_noise: float = 1.0,
    filter_by: str = "prune_by_noise_rate",
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    predicted = probs.argmax(axis=1).astype(np.int64)
    try:
        from cleanlab.filter import find_label_issues as official_find_label_issues
    except ImportError as exc:
        raise ImportError(
            "The official cleanlab package is required for the cleanlab baseline. "
            "Install it with `python -m pip install cleanlab`."
        ) from exc

    noisy = official_find_label_issues(
        labels=y,
        pred_probs=probs,
        filter_by=str(filter_by or "prune_by_noise_rate"),
        frac_noise=float(frac_noise),
    )
    return np.asarray(noisy, dtype=bool), predicted


def run(ctx: MethodContext) -> MethodOutput:
    cv_folds = int(ctx.param("cv_folds", ctx.param("cv_n_folds", 5)))
    cv_epochs = int(ctx.param("cv_epochs", ctx.param("score_epochs", 10)))
    action = str(ctx.param("action", "remove")).lower()
    frac_noise = float(ctx.param("issue_fraction", ctx.param("frac_noise", 0.0)))
    filter_by = str(ctx.param("filter_by", "confident_learning"))
    downweight_value = float(ctx.param("downweight_value", 0.1))

    with ctx.timed_phase("cleanlab.oof_training_and_prediction"):
        oof = _oof_probabilities(ctx, cv_folds=cv_folds, cv_epochs=max(1, cv_epochs))
    with ctx.timed_phase("cleanlab.label_issue_detection"):
        noisy_mask, predicted = find_label_issues(ctx.y_train, oof, frac_noise=frac_noise, filter_by=filter_by)

    with ctx.timed_phase("cleanlab.action_application"):
        corrected_labels = None
        if action == "remove":
            selected = np.where(~noisy_mask)[0].astype(np.int64)
            weights = np.ones(len(selected), dtype=np.float32)
        elif action == "relabel":
            selected = np.arange(ctx.n_samples, dtype=np.int64)
            weights = np.ones(ctx.n_samples, dtype=np.float32)
            corrected_labels = ctx.y_train.copy()
            corrected_labels[noisy_mask] = predicted[noisy_mask]
        elif action == "downweight":
            selected = np.arange(ctx.n_samples, dtype=np.int64)
            weights = np.ones(ctx.n_samples, dtype=np.float32)
            weights[noisy_mask] = np.float32(downweight_value)
        else:
            raise ValueError("Cleanlab action must be remove, relabel, or downweight.")
        if len(selected) == 0:
            selected = np.arange(ctx.n_samples, dtype=np.int64)
            weights = np.ones(ctx.n_samples, dtype=np.float32)

    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=selected,
        sample_weights=weights,
        corrected_labels=corrected_labels,
        predicted_noisy_mask=noisy_mask,
        metadata={
            "paper": "Confident Learning: Estimating Uncertainty in Dataset Labels",
            "method_type": "training_assisted_data_processing",
            "cv_folds": cv_folds,
            "cv_n_folds": cv_folds,
            "cv_epochs": cv_epochs,
            "action": action,
            "issue_fraction": frac_noise,
            "frac_noise": frac_noise,
            "filter_by": filter_by,
            "num_predicted_noisy": int(noisy_mask.sum()),
        },
    )
