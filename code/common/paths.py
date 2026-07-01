"""Centralized project paths."""

from __future__ import annotations

from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DATA_ROOT = CODE_ROOT / "data"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
NPZ_CLEAN_ROOT = PROCESSED_DATA_ROOT / "npz_clean"
NPZ_NOISY_ROOT = PROCESSED_DATA_ROOT / "npz_noisy"
CONFIG_ROOT = CODE_ROOT / "configs"
RESULTS_ROOT = CODE_ROOT / "results"


def rate_dir_name(rate: float) -> str:
    return f"rate_{float(rate):.2f}"


def noisy_dataset_dir(
    dataset: str,
    noise_type: str,
    noise_rate: float,
    noisy_root: Path = NPZ_NOISY_ROOT,
) -> Path:
    return Path(noisy_root) / noise_type / rate_dir_name(noise_rate) / dataset


def clean_dataset_dir(dataset: str, clean_root: Path = NPZ_CLEAN_ROOT) -> Path:
    return Path(clean_root) / dataset


def result_dir(
    dataset: str,
    method: str,
    noise_type: str,
    noise_rate: float,
    seed: int,
    results_root: Path = RESULTS_ROOT,
) -> Path:
    run_name = f"{noise_type}_{rate_dir_name(noise_rate)}_seed{int(seed)}"
    return Path(results_root) / dataset / method / run_name

