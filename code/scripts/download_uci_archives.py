"""Download the UCI archives used by the CleanCore experiments."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = CODE_ROOT / "data" / "raw"

ARCHIVES = {
    "wdbc": (
        "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        "wdbc/breast_cancer_wisconsin_diagnostic.zip",
    ),
    "banknote": (
        "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
        "banknote/banknote_authentication.zip",
    ),
    "pendigits": (
        "https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip",
        "pendigits/pen_based_recognition_of_handwritten_digits.zip",
    ),
    "magic": (
        "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
        "magic/magic_gamma_telescope.zip",
    ),
    "sensorless": (
        "https://archive.ics.uci.edu/static/public/325/dataset+for+sensorless+drive+diagnosis.zip",
        "sensorless/sensorless_drive_diagnosis.zip",
    ),
    "miniboone": (
        "https://archive.ics.uci.edu/static/public/199/miniboone+particle+identification.zip",
        "miniboone/miniboone_particle_identification.zip",
    ),
    "skin": (
        "https://archive.ics.uci.edu/static/public/229/skin+segmentation.zip",
        "skin/skin_segmentation.zip",
    ),
    "covertype": (
        "https://archive.ics.uci.edu/static/public/31/covertype.zip",
        "covertype/covertype.zip",
    ),
    "susy": (
        "https://archive.ics.uci.edu/static/public/279/susy.zip",
        "susy/susy.zip",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw UCI zip archives.")
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset name, comma-separated names, or 'all'.",
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_datasets(value: str) -> list[str]:
    value = value.strip().lower()
    if value == "all":
        return list(ARCHIVES)
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in ARCHIVES]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}")
    return names


def download_one(dataset: str, raw_root: Path, overwrite: bool) -> None:
    url, rel_path = ARCHIVES[dataset]
    dest = raw_root / rel_path
    if dest.exists() and not overwrite:
        print(f"[skip] {dataset}: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"[download] {dataset}: {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(dest)
    print(f"[saved] {dest}")


def main() -> None:
    args = parse_args()
    for dataset in selected_datasets(args.dataset):
        download_one(dataset, raw_root=args.raw_root, overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
