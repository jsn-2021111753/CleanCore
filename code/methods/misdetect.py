"""MisDetect baseline: iterative mislabel detection using early loss."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from common.interfaces import MethodOutput
from methods.base import MethodContext
from methods.torch_utils import per_sample_losses, train_aux_model


def _knn_verify_candidates(
    X: np.ndarray,
    y: np.ndarray,
    candidate_mask: np.ndarray,
    clean_pool_mask: np.ndarray,
    k: int,
    disagreement_threshold: float,
) -> np.ndarray:
    """Keep candidates whose neighbors disagree with their observed labels."""

    candidates = np.where(candidate_mask)[0].astype(np.int64)
    if len(candidates) == 0:
        return candidate_mask
    reference = np.where(clean_pool_mask)[0].astype(np.int64)
    if len(reference) < max(2, int(k)):
        reference = np.arange(len(y), dtype=np.int64)
    n_neighbors = min(max(1, int(k)), len(reference))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X[reference])
    verified = np.zeros(len(y), dtype=bool)
    _, local = nn.kneighbors(X[candidates], return_distance=True)
    for row, idx in enumerate(candidates.tolist()):
        neighbors = reference[local[row]]
        counts = np.bincount(y[neighbors], minlength=int(np.max(y)) + 1)
        same_ratio = float(counts[int(y[idx])] / max(1, len(neighbors)))
        if (1.0 - same_ratio) >= float(disagreement_threshold):
            verified[int(idx)] = True
    return verified


def run(ctx: MethodContext) -> MethodOutput:
    rounds = int(ctx.param("rounds", 5))
    epochs_per_round = int(ctx.param("epochs_per_round", 5))
    threshold_strategy = str(ctx.param("threshold_strategy", "top_ratio")).lower()
    remove_ratio = float(ctx.param("remove_ratio", 0.05))
    loss_std_factor = float(ctx.param("loss_std_factor", 1.0))
    clean_std_factor = float(ctx.param("clean_std_factor", 1.0))
    default_max_remove_ratio = 0.0 if threshold_strategy in {"mean_std", "sigma", "paper"} else remove_ratio
    max_remove_ratio = float(ctx.param("max_remove_ratio", default_max_remove_ratio))
    enable_knn_verification = bool(ctx.param("enable_knn_verification", False))
    knn_k = int(ctx.param("knn_k", 20))
    knn_disagreement_threshold = float(ctx.param("knn_disagreement_threshold", 0.5))
    min_remaining = int(ctx.param("min_remaining", max(2 * ctx.num_classes, 20)))

    active = np.ones(ctx.n_samples, dtype=bool)
    clean_pool = np.zeros(ctx.n_samples, dtype=bool)
    removed_score = np.zeros(ctx.n_samples, dtype=np.float32)
    removed_round = np.zeros(ctx.n_samples, dtype=np.int32)
    round_reports: list[dict[str, object]] = []

    for r in range(1, max(1, rounds) + 1):
        active_idx = np.where(active)[0].astype(np.int64)
        if len(active_idx) <= max(min_remaining, ctx.num_classes):
            break
        with ctx.timed_phase("misdetect.auxiliary_training"):
            model = train_aux_model(
                ctx,
                ctx.X_train[active_idx],
                ctx.y_train[active_idx],
                max_epochs=max(1, epochs_per_round),
                seed_offset=300 + r,
            )
        with ctx.timed_phase("misdetect.loss_scoring"):
            losses = per_sample_losses(
                model,
                ctx.X_train[active_idx],
                ctx.y_train[active_idx],
                batch_size=int(ctx.training_config.get("batch_size", 1024)),
                device=str(ctx.training_config.get("device", "cpu")),
            )
        with ctx.timed_phase("misdetect.thresholding"):
            if threshold_strategy in {"mean_std", "sigma", "paper"}:
                mu = float(np.mean(losses))
                sigma = float(np.std(losses))
                local_remove = np.where(losses > mu + loss_std_factor * sigma)[0].astype(np.int64)
                local_clean = np.where(losses < mu - clean_std_factor * sigma)[0].astype(np.int64)
                clean_pool[active_idx[local_clean]] = True
                if max_remove_ratio and max_remove_ratio > 0:
                    cap = max(1, int(round(float(max_remove_ratio) * len(active_idx))))
                    if len(local_remove) > cap:
                        ranked = local_remove[np.argsort(-losses[local_remove])[:cap]]
                        local_remove = ranked.astype(np.int64)
            else:
                remove_k = max(1, int(round(float(remove_ratio) * len(active_idx))))
                remove_k = min(remove_k, max(0, len(active_idx) - max(min_remaining, ctx.num_classes)))
                if remove_k <= 0:
                    break
                local_remove = np.argsort(-losses)[:remove_k]
                clean_pool[active_idx[np.argsort(losses)[:remove_k]]] = True
        remove_k = min(len(local_remove), max(0, len(active_idx) - max(min_remaining, ctx.num_classes)))
        if remove_k <= 0:
            break
        local_remove = local_remove[:remove_k]
        global_remove = active_idx[local_remove]
        active[global_remove] = False
        removed_score[global_remove] = losses[local_remove]
        removed_round[global_remove] = r
        round_reports.append(
            {
                "round": int(r),
                "active_before": int(len(active_idx)),
                "removed": int(len(global_remove)),
                "clean_pool_size": int(clean_pool.sum()),
                "loss_mean": float(np.mean(losses)),
                "loss_std": float(np.std(losses)),
            }
        )

    predicted_noisy = ~active
    if enable_knn_verification:
        with ctx.timed_phase("misdetect.knn_verification"):
            predicted_noisy = _knn_verify_candidates(
                ctx.X_train,
                ctx.y_train,
                predicted_noisy,
                clean_pool,
                k=knn_k,
                disagreement_threshold=knn_disagreement_threshold,
            )
            active = ~predicted_noisy

    selected = np.where(active)[0].astype(np.int64)
    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=selected,
        sample_weights=np.ones(len(selected), dtype=np.float32),
        predicted_noisy_mask=predicted_noisy,
        metadata={
            "paper": "MisDetect: Iterative Mislabel Detection using Early Loss",
            "method_type": "training_assisted_data_processing",
            "rounds": rounds,
            "epochs_per_round": epochs_per_round,
            "remove_ratio": remove_ratio,
            "threshold_strategy": threshold_strategy,
            "loss_std_factor": loss_std_factor,
            "clean_std_factor": clean_std_factor,
            "max_remove_ratio": max_remove_ratio,
            "enable_knn_verification": enable_knn_verification,
            "knn_k": knn_k,
            "knn_disagreement_threshold": knn_disagreement_threshold,
            "num_removed": int(predicted_noisy.sum()),
            "clean_pool_size": int(clean_pool.sum()),
            "round_reports": round_reports,
            "removed_round": removed_round.tolist(),
            "removed_score_sum": float(removed_score.sum()),
        },
    )
