"""Pipeline baseline: CoreTab selection followed by MisDetect cleaning."""

from __future__ import annotations

import numpy as np

from common.interfaces import MethodOutput
from methods import coretab, misdetect
from methods.base import MethodContext
from methods.misdetect_coretab import _full_mask_from_local, _selected_stage_data, _stage_context


def run(ctx: MethodContext) -> MethodOutput:
    with ctx.timed_phase("pipeline_combo.coretab"):
        core_ctx = _stage_context(ctx, ctx.X_train, ctx.y_train, "coretab", seed_offset=730)
        core_out = coretab.run(core_ctx)

    core_global = np.asarray(core_out.selected_indices, dtype=np.int64)
    X_core, y_core = _selected_stage_data(ctx.X_train, ctx.y_train, core_global)

    with ctx.timed_phase("pipeline_combo.misdetect"):
        mis_ctx = _stage_context(ctx, X_core, y_core, "misdetect", seed_offset=740)
        mis_out = misdetect.run(mis_ctx)

    kept_local = np.asarray(mis_out.selected_indices, dtype=np.int64)
    selected = core_global[kept_local]
    core_weights = np.asarray(core_out.sample_weights, dtype=np.float32)
    sample_weights = core_weights[kept_local]
    predicted_noisy = _full_mask_from_local(mis_out.predicted_noisy_mask, core_global, ctx.n_samples)

    metadata = {
        "paper": "Sequential baseline: CoreTab followed by MisDetect",
        "method_type": "coreset_then_clean_pipeline",
        "pipeline": ["coretab", "misdetect"],
        "coretab_selected_before_misdetect": int(len(core_global)),
        "final_selected_samples": int(len(selected)),
        "coretab_metadata": dict(core_out.metadata),
        "misdetect_metadata": dict(mis_out.metadata),
    }
    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=selected,
        sample_weights=sample_weights,
        predicted_noisy_mask=predicted_noisy,
        metadata=metadata,
    )
