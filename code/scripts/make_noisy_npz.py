"""Generate fixed noisy NPZ datasets from clean NPZ datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from noise.injector import (
    ERROR_FEATURE_ONLY,
    ERROR_LABEL_ONLY,
    ERROR_MIXED,
    assign_error_types,
)
from noise.random_noise import apply_random_noise
from noise.shift_noise import apply_shift_noise


DEFAULT_CLEAN_ROOT = CODE_ROOT / "data" / "processed" / "npz_clean"
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "data" / "processed" / "npz_noisy"
DEFAULT_NOISE_RATE = 0.20
DEFAULT_SEED = 42
DEFAULT_MAX_CORRUPTED_FEATURE_FRAC = 0.30
DEFAULT_SHIFT_SCALE = 1.0


def rate_dir_name(noise_rate: float) -> str:
    return f"rate_{noise_rate:.2f}"


def discover_datasets(clean_root: Path) -> List[str]:
    return sorted(path.name for path in clean_root.iterdir() if path.is_dir())


def load_clean_split(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def write_noisy_dataset(
    dataset: str,
    clean_dataset_dir: Path,
    out_dir: Path,
    noise_type: str,
    noise_rate: float,
    seed: int,
    max_corrupted_feature_frac: float,
    shift_scale: float,
) -> None:
    train_clean = load_clean_split(clean_dataset_dir / "train.npz")
    test_clean = load_clean_split(clean_dataset_dir / "test.npz")
    X_clean = train_clean["X"].astype(np.float32, copy=False)
    y_clean = train_clean["y"].astype(np.int64, copy=False)

    error_type = assign_error_types(len(y_clean), noise_rate=noise_rate, seed=seed)
    if noise_type == "random":
        noisy = apply_random_noise(
            X_clean,
            y_clean,
            error_type,
            seed=seed + 1009,
            max_corrupted_feature_frac=max_corrupted_feature_frac,
        )
    elif noise_type == "shift":
        noisy = apply_shift_noise(
            X_clean,
            y_clean,
            error_type,
            seed=seed + 2003,
            max_corrupted_feature_frac=max_corrupted_feature_frac,
            shift_scale=shift_scale,
        )
    else:
        raise ValueError("noise_type must be 'random' or 'shift'.")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "train.npz", X=noisy.X, y=noisy.y)
    np.savez(
        out_dir / "test.npz",
        X=test_clean["X"].astype(np.float32, copy=False),
        y=test_clean["y"].astype(np.int64, copy=False),
    )
    np.savez(
        out_dir / "noise_info.npz",
        X_clean=X_clean,
        y_clean=y_clean,
        is_noisy=noisy.is_noisy,
        is_label_noisy=noisy.is_label_noisy,
        is_feature_noisy=noisy.is_feature_noisy,
        error_type=noisy.error_type,
        corrupted_feature_mask=noisy.corrupted_feature_mask,
    )

    clean_metadata_path = clean_dataset_dir / "metadata.json"
    clean_metadata: Dict[str, object] = {}
    if clean_metadata_path.exists():
        clean_metadata = json.loads(clean_metadata_path.read_text(encoding="utf-8"))

    n_label_only = int((error_type == ERROR_LABEL_ONLY).sum())
    n_feature_only = int((error_type == ERROR_FEATURE_ONLY).sum())
    n_mixed = int((error_type == ERROR_MIXED).sum())
    n_noisy = int((error_type > 0).sum())
    feature_counts = noisy.corrupted_feature_mask.sum(axis=1)

    metadata = {
        "dataset": dataset,
        "format": "npz",
        "source_clean_dir": str(clean_dataset_dir),
        "noise_type": noise_type,
        "noise_rate": float(noise_rate),
        "seed": int(seed),
        "error_type_ratio": {"label_only": 2, "feature_only": 2, "mixed": 1},
        "max_corrupted_feature_frac": float(max_corrupted_feature_frac),
        "shift_scale": float(shift_scale) if noise_type == "shift" else None,
        "n_train": int(len(y_clean)),
        "n_test": int(len(test_clean["y"])),
        "n_features": int(X_clean.shape[1]),
        "n_classes": int(y_clean.max()) + 1,
        "n_noisy": n_noisy,
        "n_label_only": n_label_only,
        "n_feature_only": n_feature_only,
        "n_mixed": n_mixed,
        "max_corrupted_features_observed": int(feature_counts.max()) if len(feature_counts) else 0,
        "files": {
            "train": "train.npz",
            "test": "test.npz",
            "noise_info": "noise_info.npz",
        },
        "train_arrays": {
            "X": "float32 noisy features",
            "y": "int64 noisy labels",
        },
        "clean_metadata": clean_metadata,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    noise_info = {
        "dataset": dataset,
        "noise_type": noise_type,
        "noise_rate": float(noise_rate),
        "seed": int(seed),
        "label_error_rule": "random different legal class" if noise_type == "random" else "(label + 1) % n_classes",
        "feature_error_rule": (
            "random replacement within clean-train feature range"
            if noise_type == "random"
            else "directional std shift clipped to clean-train feature range"
        ),
        "binary_feature_rule": "binary features are flipped between 0 and 1",
        "n_noisy": n_noisy,
        "n_label_only": n_label_only,
        "n_feature_only": n_feature_only,
        "n_mixed": n_mixed,
    }
    (out_dir / "noise_info.json").write_text(
        json.dumps(noise_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_dataset(
    dataset: str,
    clean_root: Path,
    output_root: Path,
    noise_type: str,
    noise_rate: float,
    seed: int,
    max_corrupted_feature_frac: float,
    shift_scale: float,
    overwrite: bool,
) -> None:
    clean_dataset_dir = clean_root / dataset
    if not clean_dataset_dir.exists():
        raise FileNotFoundError(f"Clean dataset directory not found: {clean_dataset_dir}")

    out_dir = output_root / noise_type / rate_dir_name(noise_rate) / dataset
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already exists. Use --overwrite to replace it.")

    write_noisy_dataset(
        dataset=dataset,
        clean_dataset_dir=clean_dataset_dir,
        out_dir=out_dir,
        noise_type=noise_type,
        noise_rate=noise_rate,
        seed=seed,
        max_corrupted_feature_frac=max_corrupted_feature_frac,
        shift_scale=shift_scale,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed noisy NPZ datasets from clean NPZ datasets.")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name or 'all'.")
    parser.add_argument("--noise-type", type=str, choices=["random", "shift", "all"], default="all")
    parser.add_argument("--noise-rate", type=float, default=DEFAULT_NOISE_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-corrupted-feature-frac", type=float, default=DEFAULT_MAX_CORRUPTED_FEATURE_FRAC)
    parser.add_argument("--shift-scale", type=float, default=DEFAULT_SHIFT_SCALE)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = discover_datasets(args.clean_root) if args.dataset == "all" else [args.dataset]
    noise_types = ["random", "shift"] if args.noise_type == "all" else [args.noise_type]

    for noise_type in noise_types:
        for dataset in datasets:
            print(f"[make-noisy] type={noise_type} rate={args.noise_rate:.2f} dataset={dataset}")
            prepare_dataset(
                dataset=dataset,
                clean_root=args.clean_root,
                output_root=args.output_root,
                noise_type=noise_type,
                noise_rate=args.noise_rate,
                seed=args.seed,
                max_corrupted_feature_frac=args.max_corrupted_feature_frac,
                shift_scale=args.shift_scale,
                overwrite=args.overwrite,
            )

    print(f"[done] noisy NPZs saved to {args.output_root}")


if __name__ == "__main__":
    main()
