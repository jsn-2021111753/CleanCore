"""CoreTab baseline: datamap-driven tabular coreset selection."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from common.interfaces import MethodOutput
from methods.base import MethodContext
from methods.common import allocate_classwise, bounded_classwise_candidates, classwise_topk, dedupe_selected


@dataclass
class _FilteredGroup:
    key: str
    label: int
    size: int
    group: list[tuple[int, int]]


class _OfficialCoreTabXGB:
    """Small adapter for the official CoreTabXGB create_coreset path."""

    def __init__(
        self,
        trees_number: int = 30,
        sample_percent: float = 0.03,
        examples_to_keep: int = 10000,
        threshold: int = 1,
        params: dict[str, object] | None = None,
        index_name: str = "index",
        n_jobs: int = 1,
        seed: int = 0,
    ) -> None:
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "CoreTab official_xgboost backend requires xgboost. "
                "Install it with `python -m pip install xgboost` on the experiment environment."
            ) from exc
        self.xgb = xgb
        self.trees_number = int(trees_number)
        self.sample_percent = float(sample_percent)
        self.examples_to_keep = int(examples_to_keep)
        self.threshold = int(threshold)
        self.params = dict(params or {})
        self.params.update({"n_jobs": int(n_jobs), "seed": int(seed)})
        self.index_name = str(index_name)
        self.seed = int(seed)
        self.target_col = "target_col"
        self.hom_groups: dict[str, int] = {}
        self.hom_groups_candidates: list[_FilteredGroup] = []
        self.groups: dict[str, list[tuple[int, int]]] | None = None
        self.X_leaves: pd.DataFrame | None = None
        self.model = None

    def get_dmatrix(self, X: pd.DataFrame, y: pd.Series | None = None):
        if y is not None:
            return self.xgb.DMatrix(X, label=y)
        return self.xgb.DMatrix(X)

    def filter_groups(self, i: int) -> list[int]:
        assert self.X_leaves is not None
        new_groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
        if self.groups is None:
            groups_first_leaf = self.X_leaves.reset_index().loc[:, ["index", "leaf_0", self.target_col]].groupby("leaf_0")
            new_groups = {
                str(leaf): list(zip(list(group["index"].values), list(group[self.target_col].values)))
                for leaf, group in groups_first_leaf
            }
        else:
            index_to_leaf = self.X_leaves[f"leaf_{i}"].to_dict()
            for key, group in self.groups.items():
                if len(group) <= self.threshold:
                    continue
                for item in group:
                    new_groups[f"{key}_{index_to_leaf[item[0]]}"].append(item)

        indexes_to_drop: list[int] = []
        groups_to_drop: list[str] = []
        for key, group in new_groups.items():
            group_length = len(group)
            if group_length <= self.threshold:
                continue
            labels = [int(item[1]) for item in group]
            if len(set(labels)) == 1:
                self.hom_groups_candidates.append(
                    _FilteredGroup(key=key, label=labels[0], size=group_length, group=group)
                )
                indexes_to_drop.extend([int(item[0]) for item in group])
                groups_to_drop.append(key)

        for key in groups_to_drop:
            new_groups.pop(key, None)
        self.groups = dict(new_groups)
        self.X_leaves = self.X_leaves[~self.X_leaves.index.isin(indexes_to_drop)]
        return indexes_to_drop

    def choose_groups(self, y_train: pd.Series) -> list[int]:
        rng = random.Random(self.seed)
        sorted_candidates = sorted(self.hom_groups_candidates, key=lambda g: g.size, reverse=True)
        label_counter: dict[int, int] = defaultdict(int)
        label_amount = {int(k): int(v) for k, v in dict(y_train.value_counts()).items()}
        indexes_to_filter: list[int] = []
        for group in sorted_candidates:
            label_counter[group.label] += group.size
            if label_amount.get(group.label, 0) - label_counter[group.label] < self.examples_to_keep:
                continue
            self.hom_groups[group.key] = group.label
            candidate_indexes = [int(item[0]) for item in group.group]
            k = int(math.floor((1.0 - self.sample_percent) * len(candidate_indexes)))
            if k > 0:
                indexes_to_filter.extend(rng.sample(candidate_indexes, k=min(k, len(candidate_indexes))))
        return indexes_to_filter

    def create_coreset(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        self.X_leaves = pd.DataFrame(y_train.values, columns=[self.target_col], index=y_train.index)
        self.hom_groups = {}
        self.hom_groups_candidates = []
        self.groups = None

        params = dict(self.params)
        num_classes = int(y_train.nunique())
        if "objective" not in params:
            if num_classes > 2:
                params.update({"objective": "multi:softprob", "num_class": num_classes})
            else:
                params.update({"objective": "binary:logistic"})
        dtrain = self.get_dmatrix(X_train, y_train)
        self.model = self.xgb.train(params, num_boost_round=self.trees_number, dtrain=dtrain)
        pred_leaves = self.model[: self.trees_number].predict(dtrain, pred_leaf=True)
        if pred_leaves.ndim == 1:
            pred_leaves = pred_leaves.reshape(-1, 1)
        self.X_leaves = self.X_leaves.assign(
            **{f"leaf_{i}": pred_leaves[:, i] for i in range(min(self.trees_number, pred_leaves.shape[1]))}
        )
        for i in range(min(self.trees_number, pred_leaves.shape[1])):
            self.filter_groups(i)
        indexes_to_filter = self.choose_groups(y_train)
        indexes_to_keep = sorted(set(X_train.index).difference(set(indexes_to_filter)))
        return X_train.loc[indexes_to_keep], y_train.loc[indexes_to_keep]


def datamap_scores(ctx: MethodContext) -> tuple[np.ndarray, dict[str, object]]:
    n_estimators = int(ctx.param("gbdt_estimators", ctx.param("n_estimators", ctx.param("num_trees", 30))))
    max_depth = int(ctx.param("gbdt_max_depth", 3))
    min_region_size = int(ctx.param("min_region_size", ctx.param("tau", 5)))
    homogeneity_threshold = float(ctx.param("homogeneity_threshold", ctx.param("psi", 1.0)))
    max_fit_samples = int(ctx.param("max_fit_samples", 0))
    fit_idx = bounded_classwise_candidates(
        ctx.y_train,
        ctx.num_classes,
        max_candidates=max_fit_samples,
        seed=ctx.seed + 430,
    )

    clf = GradientBoostingClassifier(
        n_estimators=max(1, n_estimators),
        max_depth=max(1, max_depth),
        random_state=ctx.seed,
    )
    clf.fit(ctx.X_train[fit_idx], ctx.y_train[fit_idx])
    leaves = clf.apply(ctx.X_train)
    leaves = leaves.reshape(ctx.n_samples, -1)

    easy = np.zeros(ctx.n_samples, dtype=np.float32)
    hard = np.zeros(ctx.n_samples, dtype=np.float32)
    ambiguous = np.zeros(ctx.n_samples, dtype=np.float32)
    for t in range(leaves.shape[1]):
        leaf_ids = leaves[:, t]
        for leaf in np.unique(leaf_ids):
            idx = np.where(leaf_ids == leaf)[0]
            if len(idx) == 0:
                continue
            labels = ctx.y_train[idx]
            counts = np.bincount(labels, minlength=ctx.num_classes)
            homogeneity = float(counts.max() / max(1, len(idx)))
            if homogeneity >= homogeneity_threshold and len(idx) >= min_region_size:
                easy[idx] += 1.0
            elif homogeneity >= homogeneity_threshold:
                ambiguous[idx] += 1.0
            else:
                hard[idx] += 1.0

    total = np.maximum(1.0, easy + hard + ambiguous)
    hard_score = hard / total
    ambiguous_score = ambiguous / total
    easy_score = easy / total
    # CoreTab keeps hard regions and representatives from easy regions; this
    # score prioritizes hard/ambiguous examples while still allowing easy reps.
    score = 2.0 * hard_score + ambiguous_score + 0.25 * easy_score
    return score.astype(np.float32), {
        "gbdt_estimators": n_estimators,
        "n_estimators": n_estimators,
        "num_trees": n_estimators,
        "gbdt_max_depth": max_depth,
        "fit_samples": int(len(fit_idx)),
        "max_fit_samples": max_fit_samples,
        "min_region_size": min_region_size,
        "tau": min_region_size,
        "homogeneity_threshold": homogeneity_threshold,
        "psi": homogeneity_threshold,
        "easy_mean": float(easy_score.mean()),
        "hard_mean": float(hard_score.mean()),
        "ambiguous_mean": float(ambiguous_score.mean()),
    }


def run(ctx: MethodContext) -> MethodOutput:
    backend = str(ctx.param("backend", "sklearn_datamap")).lower()
    subset_fraction = float(ctx.param("subset_fraction", 0.10))
    subset_size = int(ctx.param("subset_size", 0))
    min_per_class = int(ctx.param("min_per_class", 1))
    if backend in {"official_xgboost", "xgboost", "coretab_xgb"}:
        with ctx.timed_phase("coretab.xgboost_coreset"):
            selected, weights, meta = official_xgboost_coreset(ctx, subset_fraction, subset_size, min_per_class)
        return MethodOutput.from_arrays(
            n_samples=ctx.n_samples,
            selected_indices=selected,
            sample_weights=weights,
            metadata={
                "paper": "Datamap-Driven Tabular Coreset Selection for Classifier Training",
                "method_type": "training_assisted_data_processing",
                "implementation_reference": "CoreTab official CoreTabXGB create_coreset",
                "backend": "official_xgboost",
                "subset_fraction": subset_fraction,
                **meta,
            },
        )
    with ctx.timed_phase("coretab.datamap_training_and_scoring"):
        scores, meta = datamap_scores(ctx)
    with ctx.timed_phase("coretab.subset_selection"):
        selected, weights, report = classwise_topk(
            ctx.y_train,
            scores,
            ctx.num_classes,
            subset_fraction=subset_fraction,
            subset_size=subset_size,
            min_per_class=min_per_class,
            largest=True,
        )
    return MethodOutput.from_arrays(
        n_samples=ctx.n_samples,
        selected_indices=selected,
        sample_weights=weights,
        metadata={
            "paper": "Datamap-Driven Tabular Coreset Selection for Classifier Training",
            "method_type": "training_assisted_data_processing",
            "implementation_reference": "sklearn_datamap_compatibility_backend",
            "backend": "sklearn_datamap",
            "subset_fraction": subset_fraction,
            **meta,
            **report,
        },
    )


def _align_selected_to_budget(
    selected: np.ndarray,
    y: np.ndarray,
    subset_fraction: float,
    subset_size: int,
    min_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    y = np.asarray(y, dtype=np.int64)
    selected = np.asarray(selected, dtype=np.int64)
    selected_set = set(int(i) for i in selected.tolist())
    alloc = allocate_classwise(y, int(y.max()) + 1 if len(y) else 0, subset_fraction, subset_size, min_per_class)
    rng = np.random.default_rng(int(seed))
    final: list[int] = []
    for c, k_c in alloc.items():
        idx_selected_c = np.array([i for i in selected if int(y[int(i)]) == int(c)], dtype=np.int64)
        if len(idx_selected_c) >= int(k_c):
            picked = rng.choice(idx_selected_c, size=int(k_c), replace=False) if int(k_c) > 0 else np.array([], dtype=np.int64)
        else:
            picked = idx_selected_c
            need = int(k_c) - len(picked)
            pool = np.array([i for i in np.where(y == int(c))[0] if int(i) not in selected_set], dtype=np.int64)
            if need > 0 and len(pool) > 0:
                fill = rng.choice(pool, size=min(need, len(pool)), replace=False)
                picked = np.concatenate([picked, fill.astype(np.int64)])
        final.extend(int(i) for i in picked.tolist())
    if not final and len(selected) > 0:
        final = [int(selected[0])]
    idx, weights = dedupe_selected(final, [1.0] * len(final))
    return idx, weights, {
        "budget_aligned": True,
        "official_selected_before_budget": int(len(selected)),
        "selected_samples": int(len(idx)),
    }


def official_xgboost_coreset(
    ctx: MethodContext,
    subset_fraction: float,
    subset_size: int,
    min_per_class: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    trees_number = int(ctx.param("trees_number", ctx.param("num_trees", ctx.param("gbdt_estimators", 30))))
    sample_percent = float(ctx.param("sample_percent", 0.03))
    examples_to_keep = int(ctx.param("examples_to_keep", 10000))
    threshold = int(ctx.param("threshold", ctx.param("tau", 1)))
    n_jobs = int(ctx.param("n_jobs", 1))
    X_df = pd.DataFrame(np.asarray(ctx.X_train, dtype=np.float32), index=np.arange(ctx.n_samples))
    y_raw = np.asarray(ctx.y_train, dtype=np.int64)
    unique_labels = np.unique(y_raw)
    label_to_local = {int(label): i for i, label in enumerate(unique_labels.tolist())}
    y_local = np.asarray([label_to_local[int(label)] for label in y_raw], dtype=np.int64)
    y_series = pd.Series(y_local, index=np.arange(ctx.n_samples))
    coretab = _OfficialCoreTabXGB(
        trees_number=trees_number,
        sample_percent=sample_percent,
        examples_to_keep=examples_to_keep,
        threshold=threshold,
        n_jobs=n_jobs,
        seed=ctx.seed,
    )
    X_filtered, _ = coretab.create_coreset(X_df, y_series)
    official_selected = X_filtered.index.to_numpy(dtype=np.int64)
    selected, weights, budget_meta = _align_selected_to_budget(
        official_selected,
        ctx.y_train,
        subset_fraction=subset_fraction,
        subset_size=subset_size,
        min_per_class=min_per_class,
        seed=ctx.seed + 431,
    )
    return selected, weights, {
        "trees_number": trees_number,
        "gbdt_estimators": trees_number,
        "num_trees": trees_number,
        "sample_percent": sample_percent,
        "examples_to_keep": examples_to_keep,
        "threshold": threshold,
        "tau": int(ctx.param("tau", 5)),
        "psi": float(ctx.param("psi", 1.0)),
        "label_remapped_for_xgboost": bool(not np.array_equal(unique_labels, np.arange(len(unique_labels)))),
        "xgboost_num_classes": int(len(unique_labels)),
        "homogeneous_groups": int(len(coretab.hom_groups)),
        "homogeneous_group_candidates": int(len(coretab.hom_groups_candidates)),
        **budget_meta,
    }
