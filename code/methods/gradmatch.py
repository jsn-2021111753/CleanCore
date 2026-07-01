"""GradMatch baseline: adaptive gradient-matching subset training."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from common.interfaces import MethodOutput
from common.training import predict_classifier
from methods.base import MethodContext
from methods.common import allocate_classwise, bounded_classwise_candidates, dedupe_selected
from methods.coupled import (
    build_model,
    build_optimizer,
    make_loader,
    model_output_paths,
    train_weighted_epoch,
    training_config_from_context,
)
from methods.torch_utils import cords_orthogonal_mp_reg_nonnegative, grad_embeddings_last_layer


def _select_gradmatch_subset(
    ctx: MethodContext,
    model: torch.nn.Module,
    subset_fraction: float,
    subset_size: int,
    min_per_class: int,
    class_balanced: bool,
    selection_type: str,
    steps_cap: int,
    mp_eps: float,
    lam: float,
    max_candidate_samples: int,
    batch_size: int,
    device: str,
    update_id: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with ctx.timed_phase("gradmatch.candidate_sampling"):
        candidate_idx = bounded_classwise_candidates(
            ctx.y_train,
            ctx.num_classes,
            max_candidates=max_candidate_samples,
            seed=ctx.seed + 610 + int(update_id),
        )
        y_candidate = ctx.y_train[candidate_idx]
    with ctx.timed_phase("gradmatch.gradient_embedding"):
        G = grad_embeddings_last_layer(
            model,
            ctx.X_train[candidate_idx],
            y_candidate,
            ctx.num_classes,
            batch_size=batch_size,
            device=device,
        )

    selected: list[int] = []
    weights: list[float] = []
    per_class: dict[str, object] = {}
    budget = subset_size if subset_size > 0 else max(1, int(round(subset_fraction * len(candidate_idx))))
    budget = min(max(1, int(budget)), len(candidate_idx))
    selection_type_key = str(selection_type).lower()
    with ctx.timed_phase("gradmatch.omp_selection"):
        if class_balanced or selection_type_key in {"perclass", "per_class", "perclasspergradient", "per_class_per_gradient"}:
            alloc = allocate_classwise(y_candidate, ctx.num_classes, subset_fraction, subset_size, min_per_class)
            for c in range(ctx.num_classes):
                idx_c = np.where(y_candidate == c)[0].astype(np.int64)
                k_c = int(np.ceil(budget * len(idx_c) / max(1, len(candidate_idx))))
                k_c = min(len(idx_c), max(int(min_per_class), k_c)) if len(idx_c) else 0
                if k_c <= 0:
                    k_c = int(alloc.get(c, 0))
                if len(idx_c) == 0 or k_c <= 0:
                    per_class[str(c)] = {"n_c": int(len(idx_c)), "selected": 0}
                    continue
                local_gradients = G[idx_c]
                if selection_type_key in {"perclasspergradient", "per_class_per_gradient"}:
                    emb_dim = max(0, int((local_gradients.shape[1] - ctx.num_classes) // max(1, ctx.num_classes)))
                    start = ctx.num_classes + emb_dim * c
                    end = ctx.num_classes + emb_dim * (c + 1)
                    local_gradients = np.concatenate([local_gradients[:, c : c + 1], local_gradients[:, start:end]], axis=1)
                local, w_local = cords_orthogonal_mp_reg_nonnegative(
                    local_gradients,
                    k=k_c,
                    eps=mp_eps,
                    lam=lam,
                    steps_cap=steps_cap,
                )
                chosen = candidate_idx[idx_c[local]]
                selected.extend(int(i) for i in chosen)
                weights.extend(float(w) for w in w_local)
                per_class[str(c)] = {"n_c": int(len(idx_c)), "selected": int(len(chosen))}
        else:
            k = budget
            local, w_local = cords_orthogonal_mp_reg_nonnegative(
                G,
                k=k,
                eps=mp_eps,
                lam=lam,
                steps_cap=steps_cap,
            )
            selected.extend(int(i) for i in candidate_idx[local])
            weights.extend(float(w) for w in w_local)

    with ctx.timed_phase("gradmatch.weight_postprocessing"):
        selected_idx, sample_weights = dedupe_selected(selected, weights)
        if len(selected_idx) == 0:
            selected_idx = candidate_idx[:1].astype(np.int64)
            sample_weights = np.ones(1, dtype=np.float32)
        sample_weights = np.nan_to_num(sample_weights, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
        sample_weights[sample_weights <= 0.0] = 1.0
        if len(selected_idx) > budget:
            keep = np.argsort(-sample_weights)[:budget]
            selected_idx = selected_idx[keep]
            sample_weights = sample_weights[keep]
        elif len(selected_idx) < budget:
            chosen = set(int(i) for i in selected_idx.tolist())
            remaining = np.array([i for i in candidate_idx if int(i) not in chosen], dtype=np.int64)
            if len(remaining) > 0:
                fill = remaining[: min(int(budget) - len(selected_idx), len(remaining))]
                selected_idx = np.concatenate([selected_idx, fill.astype(np.int64)])
                sample_weights = np.concatenate([sample_weights, np.ones(len(fill), dtype=np.float32)])
    return selected_idx, sample_weights, {
        "candidate_samples": int(len(candidate_idx)),
        "selected_samples": int(len(selected_idx)),
        "per_class": per_class,
        "omp_solver": "cords_orthogonal_mp_reg_positive",
    }


def run(ctx: MethodContext) -> MethodOutput:
    cfg = training_config_from_context(ctx)
    device = torch.device(cfg.device)
    batch_size = int(cfg.batch_size)

    warmup_epochs = max(0, int(ctx.param("warmup_epochs", ctx.param("score_epochs", 5))))
    selection_interval = max(1, int(ctx.param("selection_interval", ctx.param("select_every", ctx.param("update_interval", max(1, warmup_epochs))))))
    subset_fraction = float(ctx.param("subset_fraction", ctx.param("fraction", 0.10)))
    subset_size = int(ctx.param("subset_size", 0))
    min_per_class = int(ctx.param("min_per_class", 1))
    selection_type = str(ctx.param("selection_type", "PerClass")).lower()
    class_balanced = bool(ctx.param("class_balanced", selection_type in {"perclass", "per_class"}))
    steps_cap = int(ctx.param("mp_steps_cap", 0))
    mp_eps = float(ctx.param("mp_eps", ctx.param("eps", 1e-8)))
    max_candidate_samples = int(ctx.param("max_candidate_samples", 0))
    linear_layer = bool(ctx.param("linear_layer", True))
    valid = bool(ctx.param("valid", False))
    v1 = bool(ctx.param("v1", True))
    lam = float(ctx.param("lam", 0.0))

    model = build_model(ctx, seed_offset=600).to(device)
    optimizer = build_optimizer(model, cfg)
    last_model_path, best_model_path = model_output_paths(ctx, cfg)

    history: list[dict[str, float]] = []
    update_reports: list[dict[str, object]] = []
    train_time_sec = 0.0
    selection_time_sec = 0.0
    best_train_loss = float("inf")
    best_epoch: Optional[int] = None
    bad_epochs = 0
    epochs_ran = 0
    updates_done = 0
    selected_idx = np.arange(ctx.n_samples, dtype=np.int64)
    sample_weights = np.ones(ctx.n_samples, dtype=np.float32)

    def train_epoch(indices: np.ndarray, weights: np.ndarray, phase: str) -> bool:
        nonlocal train_time_sec, best_train_loss, best_epoch, bad_epochs, epochs_ran
        loader = make_loader(
            ctx.X_train[indices],
            ctx.y_train[indices],
            sample_weights=weights,
            soft_targets=None,
            cfg=cfg,
            seed=ctx.seed + int(epochs_ran),
        )
        started = time.perf_counter()
        with ctx.timed_phase("gradmatch.training_epoch"):
            train_loss, samples_seen = train_weighted_epoch(model, loader, optimizer, device)
        epoch_time = time.perf_counter() - started
        train_time_sec += epoch_time
        epochs_ran += 1

        improved = train_loss < (best_train_loss - float(cfg.early_stop_min_delta))
        if improved:
            best_train_loss = float(train_loss)
            best_epoch = int(epochs_ran)
            bad_epochs = 0
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)
        else:
            bad_epochs += 1
        history.append(
            {
                "epoch": float(epochs_ran),
                "phase": phase,
                "train_loss": float(train_loss),
                "train_time_sec": float(epoch_time),
                "samples_seen": float(samples_seen),
                "subset_size": float(len(indices)),
                "best_train_loss_so_far": float(best_train_loss),
            }
        )
        return bool(cfg.early_stop and bad_epochs >= int(cfg.early_stop_patience))

    def update_subset() -> None:
        nonlocal selected_idx, sample_weights, updates_done, selection_time_sec
        updates_done += 1
        started = time.perf_counter()
        with ctx.timed_phase("gradmatch.subset_update"):
            selected_idx, sample_weights, report = _select_gradmatch_subset(
                ctx,
                model,
                subset_fraction=subset_fraction,
                subset_size=subset_size,
                min_per_class=min_per_class,
                class_balanced=class_balanced,
                selection_type=selection_type,
                steps_cap=steps_cap,
                mp_eps=mp_eps,
                lam=lam,
                max_candidate_samples=max_candidate_samples,
                batch_size=batch_size,
                device=str(device),
                update_id=updates_done,
            )
        elapsed = time.perf_counter() - started
        selection_time_sec += elapsed
        update_reports.append({"update": int(updates_done), "elapsed_sec": float(elapsed), **report})

    max_epochs = max(1, int(cfg.max_epochs))
    stopped = False
    for _ in range(min(warmup_epochs, max_epochs)):
        all_idx = np.arange(ctx.n_samples, dtype=np.int64)
        all_weights = np.ones(ctx.n_samples, dtype=np.float32)
        stopped = train_epoch(all_idx, all_weights, phase="warmup")
        if stopped:
            break

    if updates_done == 0:
        update_subset()

    while not stopped and epochs_ran < max_epochs:
        for _ in range(selection_interval):
            if stopped or epochs_ran >= max_epochs:
                break
            stopped = train_epoch(selected_idx, sample_weights, phase="subset")
        if not stopped and epochs_ran < max_epochs:
            update_subset()

    if last_model_path is not None:
        torch.save(model.state_dict(), last_model_path)

    final_predictions = None
    if ctx.X_test is not None:
        with ctx.timed_phase("gradmatch.final_prediction"):
            final_predictions = predict_classifier(model, ctx.X_test, batch_size=batch_size, device=str(device))

    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=selected_idx,
        sample_weights=sample_weights,
        final_predictions=final_predictions,
        training_history=history,
        metadata={
            "paper": "GradMatch: Gradient Matching Based Data Subset Selection for Efficient Deep Model Training",
            "method_type": "training_coupled",
            "continuous_training": True,
            "final_model_predictions": final_predictions is not None,
            "warmup_epochs": warmup_epochs,
            "selection_interval": selection_interval,
            "subset_fraction": subset_fraction,
            "fraction": subset_fraction,
            "requested_subset_size": subset_size,
            "subset_size": int(len(selected_idx)),
            "selection_type": selection_type,
            "class_balanced": class_balanced,
            "select_every": selection_interval,
            "linear_layer": linear_layer,
            "valid": valid,
            "v1": v1,
            "lam": lam,
            "eps": mp_eps,
            "mp_steps_cap": steps_cap,
            "candidate_samples": int(update_reports[-1]["candidate_samples"]) if update_reports else int(ctx.n_samples),
            "max_candidate_samples": max_candidate_samples,
            "updates": int(updates_done),
            "update_reports": update_reports,
            "epochs_ran": int(epochs_ran),
            "best_epoch": best_epoch,
            "best_train_loss": None if best_epoch is None else float(best_train_loss),
            "train_time_sec": float(train_time_sec),
            "selection_time_sec": float(selection_time_sec),
            "last_model_path": last_model_path,
            "best_model_path": best_model_path,
        },
    )
