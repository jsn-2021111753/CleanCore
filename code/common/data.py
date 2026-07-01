"""Dataset loading helpers for clean and noisy NPZ files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from common.paths import NPZ_CLEAN_ROOT, NPZ_NOISY_ROOT, clean_dataset_dir, noisy_dataset_dir


@dataclass
class TabularData:
    dataset: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_path: Path
    test_path: Path
    metadata: Dict[str, object]
    noise_type: Optional[str] = None
    noise_rate: Optional[float] = None

    @property
    def input_dim(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def num_classes(self) -> int:
        return int(max(self.y_train.max(), self.y_test.max()) + 1)


def _load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        if "X" not in data.files or "y" not in data.files:
            raise ValueError(f"{path} must contain X and y arrays.")
        X = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int64, copy=False)
    if X.ndim != 2:
        raise ValueError(f"{path}: X must be 2D.")
    if y.ndim != 1:
        raise ValueError(f"{path}: y must be 1D.")
    if len(X) != len(y):
        raise ValueError(f"{path}: X/y length mismatch.")
    return X, y


def load_noisy_dataset(
    dataset: str,
    noise_type: str,
    noise_rate: float,
    noisy_root: Path = NPZ_NOISY_ROOT,
) -> TabularData:
    dataset_dir = noisy_dataset_dir(dataset, noise_type, noise_rate, noisy_root=noisy_root)
    train_path = dataset_dir / "train.npz"
    test_path = dataset_dir / "test.npz"
    X_train, y_train = load_npz_xy(train_path)
    X_test, y_test = load_npz_xy(test_path)
    metadata = _load_json_if_exists(dataset_dir / "metadata.json")
    return TabularData(
        dataset=dataset,
        noise_type=noise_type,
        noise_rate=float(noise_rate),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_path=train_path,
        test_path=test_path,
        metadata=metadata,
    )


def load_clean_dataset(dataset: str, clean_root: Path = NPZ_CLEAN_ROOT) -> TabularData:
    dataset_dir = clean_dataset_dir(dataset, clean_root=clean_root)
    train_path = dataset_dir / "train.npz"
    test_path = dataset_dir / "test.npz"
    X_train, y_train = load_npz_xy(train_path)
    X_test, y_test = load_npz_xy(test_path)
    metadata = _load_json_if_exists(dataset_dir / "metadata.json")
    return TabularData(
        dataset=dataset,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_path=train_path,
        test_path=test_path,
        metadata=metadata,
    )


def load_noise_info(
    dataset: str,
    noise_type: str,
    noise_rate: float,
    noisy_root: Path = NPZ_NOISY_ROOT,
) -> Dict[str, np.ndarray]:
    info_path = noisy_dataset_dir(dataset, noise_type, noise_rate, noisy_root=noisy_root) / "noise_info.npz"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    with np.load(info_path) as data:
        return {name: data[name] for name in data.files}

