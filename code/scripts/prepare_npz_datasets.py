"""Convert clean CSV datasets to aligned NPZ datasets.

This script reads ``code/data/processed/csv_clean/<dataset>/train.csv`` and
``test.csv`` and writes aligned ``train.npz`` and ``test.npz`` files under
``code/data/processed/npz_clean/<dataset>``. It never re-splits the data, so the
NPZ rows correspond exactly to the CSV rows in the same order.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_ROOT = CODE_ROOT / "data" / "processed" / "csv_clean"
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "data" / "processed" / "npz_clean"


def validate_csv_header(csv_path: Path) -> List[str]:
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if not columns:
        raise ValueError(f"{csv_path} has no columns.")
    if columns[-1] != "label":
        raise ValueError(f"{csv_path} must have label as the last column.")
    expected_features = [f"feature_{i}" for i in range(len(columns) - 1)]
    if columns[:-1] != expected_features:
        raise ValueError(
            f"{csv_path} feature columns must be {expected_features[:3]}...feature_{len(columns)-2}."
        )
    return columns


def convert_csv_split_to_npz(csv_path: Path, npz_path: Path) -> Dict[str, object]:
    columns = validate_csv_header(csv_path)
    df = pd.read_csv(csv_path)
    if df.isna().any().any():
        raise ValueError(f"{csv_path} contains missing values.")

    feature_cols = columns[:-1]
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df["label"].to_numpy(dtype=np.int64, copy=True)
    row_id = np.arange(len(df), dtype=np.int64)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, X=X, y=y, row_id=row_id)

    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "labels": sorted(int(v) for v in np.unique(y).tolist()),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
        "row_id_dtype": str(row_id.dtype),
    }


def write_npz_metadata(
    csv_metadata: Dict[str, object],
    out_dir: Path,
    csv_dataset_dir: Path,
    summaries: Dict[str, Dict[str, object]],
) -> None:
    metadata = dict(csv_metadata)
    metadata["format"] = "npz"
    metadata["source_format"] = "csv"
    metadata["source_csv_dir"] = str(csv_dataset_dir)
    metadata["files"] = {
        "train": "train.npz",
        "test": "test.npz",
    }
    metadata["arrays"] = {
        "X": "float32",
        "y": "int64",
        "row_id": "int64",
    }
    metadata["npz_summaries"] = summaries

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def convert_dataset(dataset: str, csv_root: Path, output_root: Path, overwrite: bool) -> None:
    csv_dataset_dir = csv_root / dataset
    if not csv_dataset_dir.exists():
        raise FileNotFoundError(f"CSV dataset directory not found: {csv_dataset_dir}")

    out_dir = output_root / dataset
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already exists. Use --overwrite to replace existing files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_metadata = json.loads((csv_dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    summaries = {
        "train": convert_csv_split_to_npz(csv_dataset_dir / "train.csv", out_dir / "train.npz"),
        "test": convert_csv_split_to_npz(csv_dataset_dir / "test.csv", out_dir / "test.npz"),
    }

    if summaries["train"]["n_features"] != summaries["test"]["n_features"]:
        raise ValueError(f"{dataset} train/test feature count mismatch.")
    if summaries["train"]["n_features"] != int(csv_metadata["n_features"]):
        raise ValueError(f"{dataset} feature count does not match metadata.")
    if summaries["train"]["n_samples"] != int(csv_metadata["n_train"]):
        raise ValueError(f"{dataset} train row count does not match metadata.")
    if summaries["test"]["n_samples"] != int(csv_metadata["n_test"]):
        raise ValueError(f"{dataset} test row count does not match metadata.")

    write_npz_metadata(csv_metadata, out_dir, csv_dataset_dir, summaries)


def discover_datasets(csv_root: Path) -> List[str]:
    return sorted(path.name for path in csv_root.iterdir() if path.is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert clean CSV datasets to aligned NPZ files.")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name or 'all'.")
    parser.add_argument("--csv-root", type=Path, default=DEFAULT_CSV_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing NPZ directories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = discover_datasets(args.csv_root) if args.dataset == "all" else [args.dataset]
    args.output_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        print(f"[prepare-npz] {dataset}")
        convert_dataset(dataset, args.csv_root, args.output_root, overwrite=args.overwrite)

    print(f"[done] processed NPZs saved to {args.output_root}")


if __name__ == "__main__":
    main()
