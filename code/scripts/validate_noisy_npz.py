"""Validate generated noisy NPZ datasets against clean NPZ datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_ROOT = CODE_ROOT / "data" / "processed" / "npz_clean"
DEFAULT_NOISY_ROOT = CODE_ROOT / "data" / "processed" / "npz_noisy"
DEFAULT_NOISE_RATE = 0.20
DEFAULT_MAX_CORRUPTED_FEATURE_FRAC = 0.30
DEFAULT_CHUNK_ROWS = 100_000

ERROR_CLEAN = 0
ERROR_LABEL_ONLY = 1
ERROR_FEATURE_ONLY = 2
ERROR_MIXED = 3


def rate_dir_name(noise_rate: float) -> str:
    return f"rate_{noise_rate:.2f}"


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def expected_error_counts(n_train: int, noise_rate: float) -> Tuple[int, int, int, int]:
    n_noisy = int(round(n_train * noise_rate))
    n_label_only = int(round(n_noisy * 2 / 5))
    n_feature_only = int(round(n_noisy * 2 / 5))
    n_mixed = n_noisy - n_label_only - n_feature_only
    return n_noisy, n_label_only, n_feature_only, n_mixed


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def iter_chunks(n_rows: int, chunk_rows: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_rows, chunk_rows):
        yield start, min(start + chunk_rows, n_rows)


def validate_changed_features(
    X_noisy: np.ndarray,
    X_clean: np.ndarray,
    feature_mask: np.ndarray,
    chunk_rows: int,
) -> None:
    for start, end in iter_chunks(X_clean.shape[0], chunk_rows):
        changed = X_noisy[start:end] != X_clean[start:end]
        expected = feature_mask[start:end]
        check(
            bool(np.array_equal(changed, expected)),
            f"feature mask does not match actual changes at rows {start}:{end}",
        )


def validate_dataset(
    dataset: str,
    noise_type: str,
    clean_root: Path,
    noisy_root: Path,
    noise_rate: float,
    max_corrupted_feature_frac: float,
    chunk_rows: int,
) -> None:
    clean_dir = clean_root / dataset
    noisy_dir = noisy_root / noise_type / rate_dir_name(noise_rate) / dataset
    clean_train = load_npz(clean_dir / "train.npz")
    clean_test = load_npz(clean_dir / "test.npz")
    train = load_npz(noisy_dir / "train.npz")
    test = load_npz(noisy_dir / "test.npz")
    info = load_npz(noisy_dir / "noise_info.npz")

    check(set(train) == {"X", "y"}, f"{noisy_dir}/train.npz must contain only X and y")
    check(set(test) == {"X", "y"}, f"{noisy_dir}/test.npz must contain only X and y")
    check(
        {"X_clean", "y_clean", "is_noisy", "is_label_noisy", "is_feature_noisy", "error_type", "corrupted_feature_mask"}
        <= set(info),
        f"{noisy_dir}/noise_info.npz misses required arrays",
    )

    X_clean = clean_train["X"]
    y_clean = clean_train["y"]
    X_noisy = train["X"]
    y_noisy = train["y"]
    X_info_clean = info["X_clean"]
    y_info_clean = info["y_clean"]
    error_type = info["error_type"]
    is_noisy = info["is_noisy"]
    is_label_noisy = info["is_label_noisy"]
    is_feature_noisy = info["is_feature_noisy"]
    feature_mask = info["corrupted_feature_mask"]

    check(X_noisy.dtype == np.float32, f"{dataset}/{noise_type} train X must be float32")
    check(y_noisy.dtype == np.int64, f"{dataset}/{noise_type} train y must be int64")
    check(test["X"].dtype == np.float32, f"{dataset}/{noise_type} test X must be float32")
    check(test["y"].dtype == np.int64, f"{dataset}/{noise_type} test y must be int64")
    check(X_noisy.shape == X_clean.shape == X_info_clean.shape, f"{dataset}/{noise_type} train X shape mismatch")
    check(y_noisy.shape == y_clean.shape == y_info_clean.shape, f"{dataset}/{noise_type} train y shape mismatch")
    check(feature_mask.shape == X_clean.shape, f"{dataset}/{noise_type} corrupted_feature_mask shape mismatch")
    check(bool(np.array_equal(X_info_clean, X_clean)), f"{dataset}/{noise_type} X_clean mismatch")
    check(bool(np.array_equal(y_info_clean, y_clean)), f"{dataset}/{noise_type} y_clean mismatch")
    check(bool(np.array_equal(test["X"], clean_test["X"])), f"{dataset}/{noise_type} test X is not clean")
    check(bool(np.array_equal(test["y"], clean_test["y"])), f"{dataset}/{noise_type} test y is not clean")

    n_noisy, n_label_only, n_feature_only, n_mixed = expected_error_counts(len(y_clean), noise_rate)
    check(int((error_type > 0).sum()) == n_noisy, f"{dataset}/{noise_type} noisy count mismatch")
    check(int((error_type == ERROR_LABEL_ONLY).sum()) == n_label_only, f"{dataset}/{noise_type} label-only count mismatch")
    check(int((error_type == ERROR_FEATURE_ONLY).sum()) == n_feature_only, f"{dataset}/{noise_type} feature-only count mismatch")
    check(int((error_type == ERROR_MIXED).sum()) == n_mixed, f"{dataset}/{noise_type} mixed count mismatch")
    check(bool(np.array_equal(is_noisy, error_type > ERROR_CLEAN)), f"{dataset}/{noise_type} is_noisy mismatch")
    check(
        bool(np.array_equal(is_label_noisy, (error_type == ERROR_LABEL_ONLY) | (error_type == ERROR_MIXED))),
        f"{dataset}/{noise_type} is_label_noisy mismatch",
    )
    check(
        bool(np.array_equal(is_feature_noisy, (error_type == ERROR_FEATURE_ONLY) | (error_type == ERROR_MIXED))),
        f"{dataset}/{noise_type} is_feature_noisy mismatch",
    )

    n_classes = int(y_clean.max()) + 1
    check(int(y_noisy.min()) >= 0 and int(y_noisy.max()) < n_classes, f"{dataset}/{noise_type} noisy labels out of range")
    check(bool(np.all(y_noisy[is_label_noisy] != y_clean[is_label_noisy])), f"{dataset}/{noise_type} label noise has unchanged labels")
    check(bool(np.array_equal(y_noisy[~is_label_noisy], y_clean[~is_label_noisy])), f"{dataset}/{noise_type} clean labels changed")
    if noise_type == "shift":
        expected_shift_labels = (y_clean[is_label_noisy].astype(np.int64) + 1) % n_classes
        check(bool(np.array_equal(y_noisy[is_label_noisy], expected_shift_labels)), f"{dataset}/shift label rule mismatch")

    mins = X_clean.min(axis=0)
    maxs = X_clean.max(axis=0)
    check(bool(np.all(X_noisy >= mins)), f"{dataset}/{noise_type} noisy features below clean range")
    check(bool(np.all(X_noisy <= maxs)), f"{dataset}/{noise_type} noisy features above clean range")
    max_features = max(1, int(np.floor(max_corrupted_feature_frac * X_clean.shape[1])))
    feature_counts = feature_mask.sum(axis=1)
    check(bool(np.all(feature_counts[is_feature_noisy] >= 1)), f"{dataset}/{noise_type} feature-noisy rows without feature changes")
    check(bool(np.all(feature_counts[is_feature_noisy] <= max_features)), f"{dataset}/{noise_type} too many corrupted features")
    check(bool(np.all(feature_counts[~is_feature_noisy] == 0)), f"{dataset}/{noise_type} clean/label-only rows have feature changes")
    validate_changed_features(X_noisy, X_clean, feature_mask, chunk_rows)

    metadata_path = noisy_dir / "metadata.json"
    noise_info_path = noisy_dir / "noise_info.json"
    check(metadata_path.exists(), f"{metadata_path} missing")
    check(noise_info_path.exists(), f"{noise_info_path} missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    check(metadata["n_noisy"] == n_noisy, f"{dataset}/{noise_type} metadata n_noisy mismatch")
    check(metadata["n_label_only"] == n_label_only, f"{dataset}/{noise_type} metadata n_label_only mismatch")
    check(metadata["n_feature_only"] == n_feature_only, f"{dataset}/{noise_type} metadata n_feature_only mismatch")
    check(metadata["n_mixed"] == n_mixed, f"{dataset}/{noise_type} metadata n_mixed mismatch")

    print(
        f"[ok] {noise_type}/{dataset}: train={len(y_clean)} noisy={n_noisy} "
        f"label={n_label_only} feature={n_feature_only} mixed={n_mixed}"
    )


def discover_datasets(clean_root: Path) -> list[str]:
    return sorted(path.name for path in clean_root.iterdir() if path.is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated noisy NPZ datasets.")
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'.")
    parser.add_argument("--noise-type", choices=["random", "shift", "all"], default="all")
    parser.add_argument("--noise-rate", type=float, default=DEFAULT_NOISE_RATE)
    parser.add_argument("--max-corrupted-feature-frac", type=float, default=DEFAULT_MAX_CORRUPTED_FEATURE_FRAC)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--noisy-root", type=Path, default=DEFAULT_NOISY_ROOT)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = discover_datasets(args.clean_root) if args.dataset == "all" else [args.dataset]
    noise_types = ["random", "shift"] if args.noise_type == "all" else [args.noise_type]

    for noise_type in noise_types:
        for dataset in datasets:
            validate_dataset(
                dataset=dataset,
                noise_type=noise_type,
                clean_root=args.clean_root,
                noisy_root=args.noisy_root,
                noise_rate=args.noise_rate,
                max_corrupted_feature_frac=args.max_corrupted_feature_frac,
                chunk_rows=args.chunk_rows,
            )
    print("[done] noisy NPZ validation passed")


if __name__ == "__main__":
    main()
