"""Prepare selected REIN clean/dirty CSV files as aligned NPZ inputs.

The generated split keeps the REIN dirty training rows and clean test rows in
the numeric layout consumed by the experiment runners.
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
DATA_ALIGNMENT_CHOICES = ("rule", "csv")

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


def _repo_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _prepare_train_features(
    dirty_train: pd.DataFrame,
    clean_train: pd.DataFrame,
    alignment: str,
) -> tuple[np.ndarray, Dict[str, object]]:
    alignment = str(alignment).lower()
    if alignment not in DATA_ALIGNMENT_CHOICES:
        raise ValueError(f"Unknown input alignment profile: {alignment}")
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
        align_mask = missing if alignment == "csv" else (missing | out_of_range)
        dirty_numeric.loc[align_mask, col] = float(med[col])
        missing_counts[col] = int(missing.sum())
        out_of_range_counts[col] = int(out_of_range.sum())

    return dirty_numeric.to_numpy(dtype=np.float32, copy=True), {
        "input_alignment": alignment,
        "input_alignment_note": "REIN dirty training features aligned to the numeric NPZ schema",
        "dirty_train_feature_na_count": missing_counts,
        "dirty_train_feature_out_of_reference_range_count": out_of_range_counts,
        "dirty_train_feature_out_of_reference_range_aligned_count": {
            c: 0 if alignment == "csv" else v for c, v in out_of_range_counts.items()
        },
        "clean_train_median": {c: float(med[c]) for c in dirty_numeric.columns},
        "clean_train_min": {c: float(min_v[c]) for c in dirty_numeric.columns},
        "clean_train_max": {c: float(max_v[c]) for c in dirty_numeric.columns},
    }


def _prepare_clean_features(
    clean_part: pd.DataFrame,
    clean_train: pd.DataFrame,
) -> tuple[np.ndarray, Dict[str, int]]:
    clean_numeric = clean_part.astype(np.float64).copy()
    med = clean_train.astype(np.float64).median(axis=0, skipna=True).fillna(0.0)
    na_counts: Dict[str, int] = {}
    for col in clean_numeric.columns:
        missing = clean_numeric[col].isna()
        na_counts[col] = int(missing.sum())
        if na_counts[col] > 0:
            clean_numeric.loc[missing, col] = float(med[col])
    return clean_numeric.to_numpy(dtype=np.float32, copy=True), na_counts


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
    input_alignment: str,
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

    X_train, train_feature_report = _prepare_train_features(
        dirty_X_train,
        clean_X_train,
        alignment=input_alignment,
    )
    X_test, clean_test_na_count = _prepare_clean_features(clean_X_test, clean_X_train)

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

    X_clean_train, clean_train_na_count = _prepare_clean_features(clean_X_train, clean_X_train)
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
        "source_clean_csv": _repo_path(clean_csv),
        "source_dirty_csv": _repo_path(dirty_csv),
        "format": "npz",
        "noise_type": noise_type,
        "noise_rate": float(noise_rate),
        "seed": int(seed),
        "split": "class-stratified 80/20 split by clean labels",
        "train_source": "dirty.csv train split aligned to the numeric NPZ schema",
        "test_source": "clean.csv test split",
        "target_column": target,
        "dropped_columns": drop_columns,
        "feature_columns": list(clean_X_all.columns),
        "label_to_id": label_to_id,
        "n_original": int(len(valid_mask)),
        "n_after_clean_label_filter": int(len(clean)),
        "n_removed_clean_label_unavailable": removed_clean_label_missing,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(len(label_to_id)),
        "train_row_id_minmax": [int(train_row_id.min()), int(train_row_id.max())],
        "test_row_id_minmax": [int(test_row_id.min()), int(test_row_id.max())],
        "train_test_overlap": int(len(set(train_row_id.tolist()).intersection(set(test_row_id.tolist())))),
        "dirty_label_unavailable_count": int(dirty_label_missing),
        "dirty_label_out_of_vocabulary_count": int(dirty_label_illegal),
        "dirty_legal_label_flips_kept": dirty_legal_label_flips,
        "clean_train_feature_na_count": clean_train_na_count,
        "clean_test_feature_na_count": clean_test_na_count,
        **train_feature_report,
        "files": {
            "clean_train": _repo_path(clean_out_dir / "train.npz"),
            "clean_test": _repo_path(clean_out_dir / "test.npz"),
            "noisy_train": _repo_path(noisy_out_dir / "train.npz"),
            "noisy_test": _repo_path(noisy_out_dir / "test.npz"),
            "noise_info": _repo_path(noisy_out_dir / "noise_info.npz"),
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
                "rule": "REIN dirty train split aligned to legal label values",
                "dirty_legal_label_flips_kept": dirty_legal_label_flips,
                "dirty_label_unavailable_count": int(dirty_label_missing),
                "dirty_label_out_of_vocabulary_count": int(dirty_label_illegal),
                "input_alignment": train_feature_report["input_alignment_note"],
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
        "--input-alignment",
        type=str,
        choices=DATA_ALIGNMENT_CHOICES,
        default="rule",
        help="CSV-to-NPZ alignment profile.",
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
            input_alignment=str(args.input_alignment),
        )
        print(
            f"[rein] {dataset}: train={meta['n_train']} test={meta['n_test']} "
            f"features={meta['n_features']} classes={meta['n_classes']} "
            f"label_flips={meta['dirty_legal_label_flips_kept']}"
        )
    print("[done] REIN NPZ datasets prepared.")


if __name__ == "__main__":
    main()
