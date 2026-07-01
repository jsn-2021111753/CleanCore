"""Pipeline baseline: MisDetect cleaning followed by CoreTab selection."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from common.interfaces import MethodOutput
from methods import coretab, misdetect
from methods.base import MethodContext


def _stage_config(ctx: MethodContext, stage: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        key: value
        for key, value in ctx.method_config.items()
        if key not in {"misdetect", "coretab"} and not isinstance(value, dict)
    }
    nested = ctx.method_config.get(stage, {})
    if isinstance(nested, dict):
        cfg.update(nested)
    return cfg


def _stage_context(
    ctx: MethodContext,
    X: np.ndarray,
    y: np.ndarray,
    stage: str,
    seed_offset: int,
) -> MethodContext:
    y_arr = _contiguous_labels(np.asarray(y, dtype=np.int64), ctx.num_classes)
    training_config = dict(ctx.training_config)
    model_config = dict(ctx.model_config)
    batch_size = max(1, int(training_config.get("batch_size", 1024)))
    n_rows = int(len(y_arr))
    if n_rows < 2:
        raise ValueError(f"{stage} stage needs at least two samples after the previous pipeline step.")
    if n_rows < batch_size or n_rows % batch_size == 1:
        model_config["batch_norm"] = False
    return MethodContext(
        X_train=np.asarray(X, dtype=np.float32),
        y_train=y_arr,
        X_test=ctx.X_test,
        num_classes=max(ctx.num_classes, int(y_arr.max()) + 1 if len(y_arr) else ctx.num_classes),
        seed=ctx.seed + int(seed_offset),
        output_dir=ctx.output_dir,
        model_config=model_config,
        training_config=training_config,
        method_config=_stage_config(ctx, stage),
        timing=ctx.timing,
    )


def _selected_stage_data(
    X: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(selected, dtype=np.int64)
    return np.asarray(X[idx], dtype=np.float32), np.asarray(y[idx], dtype=np.int64)


def _full_mask_from_local(local_mask: np.ndarray | None, local_to_global: np.ndarray, n_samples: int) -> np.ndarray:
    full = np.zeros(int(n_samples), dtype=bool)
    if local_mask is None:
        return full
    local_mask = np.asarray(local_mask, dtype=bool)
    full[np.asarray(local_to_global, dtype=np.int64)[local_mask]] = True
    return full


def _contiguous_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if len(y) == 0:
        return y
    unique = np.unique(y)
    if unique.min() >= 0 and unique.max() < int(num_classes):
        return y
    mapping = {int(label): i for i, label in enumerate(unique.tolist())}
    return np.asarray([mapping[int(label)] for label in y], dtype=np.int64)


def run(ctx: MethodContext) -> MethodOutput:
    with ctx.timed_phase("pipeline_combo.misdetect"):
        mis_ctx = _stage_context(ctx, ctx.X_train, ctx.y_train, "misdetect", seed_offset=710)
        mis_out = misdetect.run(mis_ctx)

    clean_global = np.asarray(mis_out.selected_indices, dtype=np.int64)
    X_clean, y_clean = _selected_stage_data(ctx.X_train, ctx.y_train, clean_global)

    with ctx.timed_phase("pipeline_combo.coretab"):
        core_ctx = _stage_context(ctx, X_clean, y_clean, "coretab", seed_offset=720)
        core_out = coretab.run(core_ctx)

    core_local = np.asarray(core_out.selected_indices, dtype=np.int64)
    selected = clean_global[core_local]
    sample_weights = np.asarray(core_out.sample_weights, dtype=np.float32)

    predicted_noisy = np.zeros(ctx.n_samples, dtype=bool)
    if mis_out.predicted_noisy_mask is not None:
        predicted_noisy |= np.asarray(mis_out.predicted_noisy_mask, dtype=bool)

    metadata = {
        "paper": "Sequential baseline: MisDetect followed by CoreTab",
        "method_type": "clean_then_coreset_pipeline",
        "pipeline": ["misdetect", "coretab"],
        "misdetect_selected_before_coretab": int(len(clean_global)),
        "final_selected_samples": int(len(selected)),
        "misdetect_metadata": dict(mis_out.metadata),
        "coretab_metadata": dict(core_out.metadata),
    }
    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=selected,
        sample_weights=sample_weights,
        predicted_noisy_mask=predicted_noisy,
        metadata=metadata,
    )
