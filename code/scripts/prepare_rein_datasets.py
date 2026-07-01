"""Prepare selected REIN dirty datasets as aligned NPZ inputs.

The generated noisy split uses rule-normalized dirty training data and clean
test data. Explicit dirty values such as NaNs and out-of-range feature values
are replaced with fixed in-domain values computed from the clean training
split, so downstream methods receive valid numeric arrays.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_REIN_ROOT = PROJECT_ROOT / "rein-datasets"
DEFAULT_CLEAN_ROOT = CODE_ROOT / "data" / "processed" / "npz_clean"
DEFAULT_NOISY_ROOT = CODE_ROOT / "data" / "processed" / "npz_noisy"
DEFAULT_NOISE_TYPE = "rein_dirty"
DEFAULT_NOISE_RATE = 0.0
DEFAULT_SEED = 42
FEATURE_NORMALIZATION_CHOICES = ("rule", "missing_only")

DATASETS: Dict[str, Dict[str, object]] = {
    "smartfactory": {
        "target": "labels",
        "drop_columns": [],
    },
    "har": {
        "target": "gt",
        "drop_columns": ["Index"],
    },
}


def rate_dir_name(rate: float) -> str:
    return f"rate_{float(rate):.2f}"


def _normalize_label_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _encode_labels(
    values: Iterable[object],
    label_to_id: Dict[str, int],
    fallback_id: int | None = None,
) -> tuple[np.ndarray, int, int]:
    encoded: list[int] = []
    missing = 0
    illegal = 0
    mode_id = _fallback_label_id(label_to_id) if fallback_id is None else int(fallback_id)
    for value in values:
        key = _normalize_label_value(value)
        if key is None:
            encoded.append(mode_id)
            missing += 1
        elif key not in label_to_id:
            encoded.append(mode_id)
            illegal += 1
        else:
            encoded.append(int(label_to_id[key]))
    return np.asarray(encoded, dtype=np.int64), missing, illegal


def _fallback_label_id(label_to_id: Dict[str, int]) -> int:
    if not label_to_id:
        raise ValueError("No legal labels available.")
    return int(next(iter(label_to_id.values())))


def _mode_id(y: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(y, dtype=np.int64), return_counts=True)
    return int(values[np.argmax(counts)])


def _stratified_split(y: np.ndarray, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(y):
        idx = np.where(y == int(label))[0].astype(np.int64)
        rng.shuffle(idx)
        n_train = int(np.floor(float(train_fraction) * len(idx)))
        if len(idx) > 1:
            n_train = min(max(1, n_train), len(idx) - 1)
        train_parts.append(idx[:n_train])
        test_parts.append(idx[n_train:])
    train_idx = np.concatenate(train_parts).astype(np.int64)
    test_idx = np.concatenate(test_parts).astype(np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _label_mapping(labels: pd.Series) -> Dict[str, int]:
    normalized = [_normalize_label_value(v) for v in labels.tolist()]
    legal = sorted({v for v in normalized if v is not None})
    return {label: i for i, label in enumerate(legal)}


def _feature_frame(df: pd.DataFrame, target: str, drop_columns: Iterable[str]) -> pd.DataFrame:
    drop_set = {str(c) for c in drop_columns}
    cols = [c for c in df.columns if c != target and c not in drop_set]
    X = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    return X


def _normalize_train_features(
    dirty_train: pd.DataFrame,
    clean_train: pd.DataFrame,
    mode: str,
) -> tuple[np.ndarray, Dict[str, object]]:
    mode = str(mode).lower()
    if mode not in FEATURE_NORMALIZATION_CHOICES:
        raise ValueError(f"Unknown feature normalization mode: {mode}")
    clean_numeric = clean_train.astype(np.float64)
    dirty_numeric = dirty_train.astype(np.float64).copy()
    med = clean_numeric.median(axis=0, skipna=True)
    min_v = clean_numeric.min(axis=0, skipna=True)
    max_v = clean_numeric.max(axis=0, skipna=True)
    med = med.fillna(0.0)
    min_v = min_v.fillna(med)
    max_v = max_v.fillna(med)

    missing_counts: Dict[str, int] = {}
    out_of_range_counts: Dict[str, int] = {}
    for col in dirty_numeric.columns:
        missing = dirty_numeric[col].isna()
        below = dirty_numeric[col] < float(min_v[col])
        above = dirty_numeric[col] > float(max_v[col])
        out_of_range = (below | above).fillna(False)
        replace_mask = missing if mode == "missing_only" else (missing | out_of_range)
        dirty_numeric.loc[replace_mask, col] = float(med[col])
        missing_counts[col] = int(missing.sum())
        out_of_range_counts[col] = int(out_of_range.sum())

    feature_imputation = (
        "dirty_train NaN values replaced by clean-train column median; out-of-clean-train-range values kept"
        if mode == "missing_only"
        else "dirty_train NaN and out-of-clean-train-range values replaced by clean-train column median"
    )
    return dirty_numeric.to_numpy(dtype=np.float32, copy=True), {
        "feature_normalization_mode": mode,
        "feature_imputation": feature_imputation,
        "missing_feature_cells_replaced": missing_counts,
        "out_of_range_feature_cells_observed": out_of_range_counts,
        "out_of_range_feature_cells_replaced": {c: 0 if mode == "missing_only" else v for c, v in out_of_range_counts.items()},
        "clean_train_median": {c: float(med[c]) for c in dirty_numeric.columns},
        "clean_train_min": {c: float(min_v[c]) for c in dirty_numeric.columns},
        "clean_train_max": {c: float(max_v[c]) for c in dirty_numeric.columns},
    }


def _normalize_clean_features(
    clean_part: pd.DataFrame,
    clean_train: pd.DataFrame,
) -> tuple[np.ndarray, Dict[str, int]]:
    clean_numeric = clean_part.astype(np.float64).copy()
    med = clean_train.astype(np.float64).median(axis=0, skipna=True).fillna(0.0)
    filled: Dict[str, int] = {}
    for col in clean_numeric.columns:
        missing = clean_numeric[col].isna()
        filled[col] = int(missing.sum())
        if filled[col] > 0:
            clean_numeric.loc[missing, col] = float(med[col])
    return clean_numeric.to_numpy(dtype=np.float32, copy=True), filled


def _write_npz(path: Path, X: np.ndarray, y: np.ndarray, row_id: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X.astype(np.float32, copy=False), y=y.astype(np.int64, copy=False), row_id=row_id.astype(np.int64))


def prepare_dataset(
    dataset: str,
    rein_root: Path,
    clean_root: Path,
    noisy_root: Path,
    noise_type: str,
    noise_rate: float,
    seed: int,
    overwrite: bool,
    feature_normalization: str,
) -> Dict[str, object]:
    if dataset not in DATASETS:
        raise KeyError(f"Unsupported REIN dataset: {dataset}")
    spec = DATASETS[dataset]
    target = str(spec["target"])
    drop_columns = [str(c) for c in spec.get("drop_columns", [])]

    source_dir = Path(rein_root) / dataset
    clean_csv = source_dir / "clean.csv"
    dirty_csv = source_dir / "dirty.csv"
    clean = pd.read_csv(clean_csv)
    dirty = pd.read_csv(dirty_csv)
    dirty = dirty.loc[:, clean.columns]

    clean_labels_raw = clean[target]
    valid_mask = clean_labels_raw.map(_normalize_label_value).notna().to_numpy()
    removed_clean_label_missing = int((~valid_mask).sum())
    clean = clean.loc[valid_mask].reset_index(drop=False).rename(columns={"index": "source_row_id"})
    dirty = dirty.loc[valid_mask].reset_index(drop=True)

    label_to_id = _label_mapping(clean[target])
    y_clean_all, clean_missing, clean_illegal = _encode_labels(clean[target], label_to_id)
    if clean_missing or clean_illegal:
        raise ValueError(f"{dataset}: clean labels still contain missing/illegal values after filtering.")
    train_idx, test_idx = _stratified_split(y_clean_all, train_fraction=0.8, seed=seed)

    clean_X_all = _feature_frame(clean, target=target, drop_columns=["source_row_id", *drop_columns])
    dirty_X_all = _feature_frame(dirty, target=target, drop_columns=drop_columns)
    clean_X_train = clean_X_all.iloc[train_idx].reset_index(drop=True)
    clean_X_test = clean_X_all.iloc[test_idx].reset_index(drop=True)
    dirty_X_train = dirty_X_all.iloc[train_idx].reset_index(drop=True)

    X_train, train_feature_report = _normalize_train_features(
        dirty_X_train,
        clean_X_train,
        mode=feature_normalization,
    )
    X_test, clean_test_filled = _normalize_clean_features(clean_X_test, clean_X_train)

    y_clean_train = y_clean_all[train_idx].astype(np.int64, copy=False)
    clean_train_mode = _mode_id(y_clean_train)
    y_train_dirty, dirty_label_missing, dirty_label_illegal = _encode_labels(
        dirty[target].iloc[train_idx],
        label_to_id,
        fallback_id=clean_train_mode,
    )
    y_test = y_clean_all[test_idx].astype(np.int64, copy=False)
    dirty_legal_label_flips = int((y_train_dirty != y_clean_train).sum())

    row_id_all = clean["source_row_id"].to_numpy(dtype=np.int64, copy=False)
    train_row_id = row_id_all[train_idx]
    test_row_id = row_id_all[test_idx]

    clean_out_dir = Path(clean_root) / dataset
    noisy_out_dir = Path(noisy_root) / noise_type / rate_dir_name(noise_rate) / dataset
    for out_dir in (clean_out_dir, noisy_out_dir):
        if out_dir.exists() and overwrite:
            shutil.rmtree(out_dir)
        elif out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"{out_dir} already exists. Use --overwrite to replace it.")

    X_clean_train, clean_train_filled = _normalize_clean_features(clean_X_train, clean_X_train)
    _write_npz(clean_out_dir / "train.npz", X_clean_train, y_clean_train, train_row_id)
    _write_npz(clean_out_dir / "test.npz", X_test, y_test, test_row_id)
    _write_npz(noisy_out_dir / "train.npz", X_train, y_train_dirty, train_row_id)
    _write_npz(noisy_out_dir / "test.npz", X_test, y_test, test_row_id)

    train_noisy_mask = y_train_dirty != y_clean_train
    np.savez(
        noisy_out_dir / "noise_info.npz",
        X_clean=X_clean_train,
        y_clean=y_clean_train,
        train_row_id=train_row_id,
        test_row_id=test_row_id,
        is_noisy=train_noisy_mask.astype(bool),
        is_label_noisy=train_noisy_mask.astype(bool),
        is_feature_noisy=np.ones(len(y_train_dirty), dtype=bool),
        error_type=np.where(train_noisy_mask, 1, 2).astype(np.int64),
    )

    metadata: Dict[str, object] = {
        "dataset": dataset,
        "source": "REIN benchmark",
        "source_clean_csv": str(clean_csv),
        "source_dirty_csv": str(dirty_csv),
        "format": "npz",
        "noise_type": noise_type,
        "noise_rate": float(noise_rate),
        "seed": int(seed),
        "split": "class-stratified 80/20 split by clean labels",
        "train_source": "dirty.csv train split after explicit-error normalization",
        "test_source": "clean.csv test split",
        "target_column": target,
        "dropped_columns": drop_columns,
        "feature_columns": list(clean_X_all.columns),
        "label_to_id": label_to_id,
        "n_original": int(len(valid_mask)),
        "n_after_clean_label_filter": int(len(clean)),
        "n_removed_clean_label_missing": removed_clean_label_missing,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(len(label_to_id)),
        "train_row_id_minmax": [int(train_row_id.min()), int(train_row_id.max())],
        "test_row_id_minmax": [int(test_row_id.min()), int(test_row_id.max())],
        "train_test_overlap": int(len(set(train_row_id.tolist()).intersection(set(test_row_id.tolist())))),
        "dirty_label_missing_replaced_by_clean_train_mode": int(dirty_label_missing),
        "dirty_label_illegal_replaced_by_clean_train_mode": int(dirty_label_illegal),
        "dirty_legal_label_flips_kept": dirty_legal_label_flips,
        "clean_train_feature_missing_filled": clean_train_filled,
        "clean_test_feature_missing_filled": clean_test_filled,
        **train_feature_report,
        "files": {
            "clean_train": str(clean_out_dir / "train.npz"),
            "clean_test": str(clean_out_dir / "test.npz"),
            "noisy_train": str(noisy_out_dir / "train.npz"),
            "noisy_test": str(noisy_out_dir / "test.npz"),
            "noise_info": str(noisy_out_dir / "noise_info.npz"),
        },
    }
    clean_out_dir.mkdir(parents=True, exist_ok=True)
    noisy_out_dir.mkdir(parents=True, exist_ok=True)
    for out_dir in (clean_out_dir, noisy_out_dir):
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (noisy_out_dir / "noise_info.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": float(noise_rate),
                "rule": "REIN dirty train split with explicit errors normalized to fixed legal values",
                "dirty_legal_label_flips_kept": dirty_legal_label_flips,
                "dirty_label_missing_replaced": int(dirty_label_missing),
                "dirty_label_illegal_replaced": int(dirty_label_illegal),
                "feature_normalization": train_feature_report["feature_imputation"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Smart Factory and HAR from REIN clean/dirty CSV files.")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name or 'all'.")
    parser.add_argument("--rein-root", type=Path, default=DEFAULT_REIN_ROOT)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--noisy-root", type=Path, default=DEFAULT_NOISY_ROOT)
    parser.add_argument("--noise-type", type=str, default=DEFAULT_NOISE_TYPE)
    parser.add_argument("--noise-rate", type=float, default=DEFAULT_NOISE_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--feature-normalization",
        type=str,
        choices=FEATURE_NORMALIZATION_CHOICES,
        default="rule",
        help="How to make dirty training features numeric and model-ready.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        meta = prepare_dataset(
            dataset=dataset,
            rein_root=args.rein_root,
            clean_root=args.clean_root,
            noisy_root=args.noisy_root,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
            seed=args.seed,
            overwrite=bool(args.overwrite),
            feature_normalization=str(args.feature_normalization),
        )
        print(
            f"[rein] {dataset}: train={meta['n_train']} test={meta['n_test']} "
            f"features={meta['n_features']} classes={meta['n_classes']} "
            f"label_flips={meta['dirty_legal_label_flips_kept']}"
        )
    print("[done] REIN NPZ datasets prepared.")


if __name__ == "__main__":
    main()
