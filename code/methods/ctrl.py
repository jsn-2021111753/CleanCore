"""CTRL baseline: clustering training loss curves for label error detection."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from common.interfaces import MethodOutput
from methods.base import MethodContext
from methods.torch_utils import train_loss_trajectories


def _moving_average(curves: np.ndarray, window: int) -> np.ndarray:
    """CTRL official trailing moving average along the epoch axis."""

    curves = np.asarray(curves, dtype=np.float32)
    window = max(1, int(window))
    if window <= 1 or curves.shape[1] <= 1:
        return curves.astype(np.float32)
    window = min(window, curves.shape[1])
    cumsum = np.cumsum(curves, axis=1, dtype=np.float64)
    cumsum = np.concatenate([np.zeros((curves.shape[0], 1), dtype=np.float64), cumsum], axis=1)
    ma = (cumsum[:, window:] - cumsum[:, :-window]) / float(window)
    return np.concatenate([ma[:, :1] * np.ones((1, window - 1)), ma], axis=1).astype(np.float32)


def _ctrl_noisy_mask(
    y: np.ndarray,
    curves: np.ndarray,
    num_classes: int,
    seed: int,
    n_clusters: int,
    noisy_clusters: int,
    num_windows: int,
    window_threshold: float,
) -> tuple[np.ndarray, dict[str, object]]:
    n, epochs = curves.shape
    if n < 2:
        return np.zeros(n, dtype=bool), {"windows": 0, "votes_required": 0}
    windows = max(1, min(int(num_windows), int(epochs)))
    clusters = max(2, int(n_clusters))
    select_clusters = max(1, min(int(noisy_clusters), clusters))
    clean_votes = np.ones((n, windows), dtype=np.int32)
    window_edges = np.linspace(0, epochs, windows + 1, dtype=int)
    used_windows = 0
    for wi in range(windows):
        start, end = int(window_edges[wi]), int(window_edges[wi + 1])
        if end <= start:
            continue
        used_windows += 1
        for c in range(int(num_classes)):
            idx_c = np.where(y == c)[0].astype(np.int64)
            if len(idx_c) < 2:
                continue
            k = min(clusters, len(idx_c))
            features = curves[idx_c, start:end]
            labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(features)
            areas = np.array([features[labels == j].sum(axis=1).mean() if np.any(labels == j) else -np.inf for j in range(k)])
            noisy_cluster_ids = set(np.argsort(-areas)[: min(select_clusters, k)].tolist())
            local_noisy = np.array([label in noisy_cluster_ids for label in labels], dtype=bool)
            clean_votes[idx_c[local_noisy], wi] = 0
    required = int(np.ceil(float(window_threshold))) if window_threshold <= 1.0 else int(window_threshold)
    required = max(1, min(required, max(1, used_windows)))
    clean_mask = clean_votes[:, : max(1, used_windows)].sum(axis=1) >= required
    return ~clean_mask, {
        "windows": int(used_windows),
        "votes_required": int(required),
        "n_clusters": clusters,
        "noisy_clusters": select_clusters,
        "official_mask_semantics": "clean_votes_ge_threshold",
    }


def run(ctx: MethodContext) -> MethodOutput:
    loss_epochs = int(ctx.param("loss_epochs", ctx.param("score_epochs", 10)))
    moving_average_size = int(ctx.param("moving_average_size", ctx.param("smooth_window", 5)))
    n_clusters = int(ctx.param("n_clusters", ctx.param("k", 2)))
    noisy_clusters = int(ctx.param("noisy_clusters", ctx.param("s", 1)))
    num_windows = int(ctx.param("num_windows", ctx.param("w", 1)))
    window_threshold = float(ctx.param("window_threshold", ctx.param("t", 1.0)))
    clamp_losses = bool(ctx.param("clamp_losses", True))
    loss_thresh_factor = float(ctx.param("loss_thresh_factor", 2.0))
    action = str(ctx.param("action", "remove")).lower()
    downweight_value = float(ctx.param("downweight_value", 0.1))

    with ctx.timed_phase("ctrl.loss_trajectory_training"):
        _, curves = train_loss_trajectories(ctx, ctx.X_train, ctx.y_train, epochs=max(2, loss_epochs), seed_offset=400)
    with ctx.timed_phase("ctrl.curve_preprocessing"):
        loss_thresh = float(ctx.param("loss_thresh", loss_thresh_factor * np.log(max(2, ctx.num_classes))))
        if clamp_losses:
            curves = np.minimum(curves, np.float32(loss_thresh))
        curve_features = _moving_average(curves, moving_average_size)
    with ctx.timed_phase("ctrl.clustering"):
        noisy_mask, ctrl_meta = _ctrl_noisy_mask(
            ctx.y_train,
            curve_features,
            ctx.num_classes,
            seed=ctx.seed,
            n_clusters=n_clusters,
            noisy_clusters=noisy_clusters,
            num_windows=num_windows,
            window_threshold=window_threshold,
        )

    with ctx.timed_phase("ctrl.action_application"):
        if action == "remove":
            selected = np.where(~noisy_mask)[0].astype(np.int64)
            weights = np.ones(len(selected), dtype=np.float32)
        elif action == "downweight":
            selected = np.arange(ctx.n_samples, dtype=np.int64)
            weights = np.ones(ctx.n_samples, dtype=np.float32)
            weights[noisy_mask] = np.float32(downweight_value)
        else:
            raise ValueError("CTRL action must be remove or downweight.")
        if len(selected) == 0:
            selected = np.arange(ctx.n_samples, dtype=np.int64)
            weights = np.ones(ctx.n_samples, dtype=np.float32)

    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=selected,
        sample_weights=weights,
        predicted_noisy_mask=noisy_mask,
        metadata={
            "paper": "CTRL: Clustering Training Losses for Label Error Detection",
            "method_type": "training_assisted_data_processing",
            "implementation_reference": "chang-yue/ctrl clustering.compute_mask",
            "loss_epochs": loss_epochs,
            "moving_average_size": moving_average_size,
            "clamp_losses": clamp_losses,
            "loss_thresh": loss_thresh,
            "action": action,
            "num_predicted_noisy": int(noisy_mask.sum()),
            **ctrl_meta,
        },
    )
