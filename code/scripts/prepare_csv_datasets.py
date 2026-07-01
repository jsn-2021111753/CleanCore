"""Prepare clean CSV datasets from raw UCI archives.

Outputs one directory per dataset under ``code/data/processed/csv_clean``:

    <dataset>/
      all.csv      # omitted for SUSY by default because it is very large
      train.csv
      test.csv
      metadata.json

CSV files always use headers, feature columns first, and ``label`` as the last
column. Labels are mapped to contiguous integers starting at 0.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = CODE_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "data" / "processed" / "csv_clean"
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8


@dataclass
class DatasetBundle:
    name: str
    all_df: pd.DataFrame | None
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    metadata: Dict[str, object]
    write_all: bool = True


def feature_columns(n_features: int) -> List[str]:
    return [f"feature_{i}" for i in range(n_features)]


def ensure_label_last(df: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in df.columns if col != "label"] + ["label"]
    return df.loc[:, columns]


def map_labels_to_zero_based(labels: Iterable[object]) -> Tuple[pd.Series, Dict[str, int]]:
    series = pd.Series(labels)
    unique = sorted(series.dropna().unique().tolist())
    mapping = {str(label): idx for idx, label in enumerate(unique)}
    mapped = series.map(lambda value: mapping[str(value)]).astype("int64")
    return mapped, mapping


def build_dataframe(
    raw_df: pd.DataFrame,
    label_col: int | str,
    label_mapping: Dict[object, int] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    label_series = raw_df[label_col]
    feature_df = raw_df.drop(columns=[label_col])
    feature_df = feature_df.apply(pd.to_numeric)

    if label_mapping is None:
        y, mapping = map_labels_to_zero_based(label_series)
    else:
        mapping = {str(key): int(value) for key, value in label_mapping.items()}
        y = label_series.map(lambda value: mapping[str(value)]).astype("int64")

    out = feature_df.copy()
    out.columns = feature_columns(out.shape[1])
    out["label"] = y.to_numpy()
    return ensure_label_last(out), mapping


def stratified_split(
    df: pd.DataFrame,
    seed: int = DEFAULT_SPLIT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, group in df.groupby("label", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed)
        n_train = int(len(shuffled) * train_ratio)
        if len(shuffled) > 1:
            n_train = min(max(n_train, 1), len(shuffled) - 1)
        train_parts.append(shuffled.iloc[:n_train])
        test_parts.append(shuffled.iloc[n_train:])

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return ensure_label_last(train_df), ensure_label_last(test_df)


def base_metadata(
    dataset: str,
    uci_id: int,
    raw_file: str,
    split: str,
    label_mapping: Dict[str, int],
    n_features: int,
    n_classes: int,
    n_total: int,
    n_train: int,
    n_test: int,
    split_seed: int | None = None,
) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "dataset": dataset,
        "source": "UCI Machine Learning Repository",
        "uci_id": uci_id,
        "raw_file": raw_file,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_total": n_total,
        "n_train": n_train,
        "n_test": n_test,
        "feature_columns": feature_columns(n_features),
        "label_column": "label",
        "label_mapping": label_mapping,
        "split": split,
    }
    if split_seed is not None:
        metadata["split_seed"] = split_seed
        metadata["train_ratio"] = DEFAULT_TRAIN_RATIO
        metadata["test_ratio"] = 1.0 - DEFAULT_TRAIN_RATIO
    return metadata


def write_dataset_csvs(bundle: DatasetBundle, output_root: Path) -> None:
    out_dir = output_root / bundle.name
    out_dir.mkdir(parents=True, exist_ok=True)

    files: Dict[str, str] = {
        "train": "train.csv",
        "test": "test.csv",
    }

    bundle.train_df = ensure_label_last(bundle.train_df)
    bundle.test_df = ensure_label_last(bundle.test_df)
    bundle.train_df.to_csv(out_dir / "train.csv", index=False)
    bundle.test_df.to_csv(out_dir / "test.csv", index=False)

    if bundle.write_all:
        if bundle.all_df is None:
            raise ValueError("write_all=True requires all_df.")
        bundle.all_df = ensure_label_last(bundle.all_df)
        bundle.all_df.to_csv(out_dir / "all.csv", index=False)
        files["all"] = "all.csv"

    metadata = dict(bundle.metadata)
    metadata["files"] = files
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_zip_text(raw_root: Path, dataset: str, zip_name: str, inner_name: str) -> io.BytesIO:
    zip_path = raw_root / dataset / zip_name
    with zipfile.ZipFile(zip_path) as zf:
        return io.BytesIO(zf.read(inner_name))


def load_wdbc(raw_root: Path, seed: int) -> DatasetBundle:
    data = read_zip_text(raw_root, "wdbc", "breast_cancer_wisconsin_diagnostic.zip", "wdbc.data")
    raw_df = pd.read_csv(data, header=None)
    raw_df = raw_df.drop(columns=[0])
    all_df, mapping = build_dataframe(raw_df, label_col=1, label_mapping={"B": 0, "M": 1})
    train_df, test_df = stratified_split(all_df, seed=seed)
    metadata = base_metadata(
        "wdbc", 17, "wdbc/breast_cancer_wisconsin_diagnostic.zip", "stratified_80_20_by_label",
        mapping, 30, 2, len(all_df), len(train_df), len(test_df), seed,
    )
    return DatasetBundle("wdbc", all_df, train_df, test_df, metadata)


def load_banknote(raw_root: Path, seed: int) -> DatasetBundle:
    data = read_zip_text(raw_root, "banknote", "banknote_authentication.zip", "data_banknote_authentication.txt")
    raw_df = pd.read_csv(data, header=None)
    all_df, mapping = build_dataframe(raw_df, label_col=4)
    train_df, test_df = stratified_split(all_df, seed=seed)
    metadata = base_metadata(
        "banknote", 267, "banknote/banknote_authentication.zip", "stratified_80_20_by_label",
        mapping, 4, 2, len(all_df), len(train_df), len(test_df), seed,
    )
    return DatasetBundle("banknote", all_df, train_df, test_df, metadata)


def load_pendigits(raw_root: Path, seed: int) -> DatasetBundle:
    train_data = read_zip_text(raw_root, "pendigits", "pen_based_recognition_of_handwritten_digits.zip", "pendigits.tra")
    test_data = read_zip_text(raw_root, "pendigits", "pen_based_recognition_of_handwritten_digits.zip", "pendigits.tes")
    train_raw = pd.read_csv(train_data, header=None, skipinitialspace=True)
    test_raw = pd.read_csv(test_data, header=None, skipinitialspace=True)
    all_raw = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)
    all_df, mapping = build_dataframe(all_raw, label_col=16)
    train_df, _ = build_dataframe(train_raw, label_col=16, label_mapping=mapping)
    test_df, _ = build_dataframe(test_raw, label_col=16, label_mapping=mapping)
    metadata = base_metadata(
        "pendigits", 81, "pendigits/pen_based_recognition_of_handwritten_digits.zip", "official_train_test",
        mapping, 16, 10, len(all_df), len(train_df), len(test_df), None,
    )
    metadata["source_train_file"] = "pendigits.tra"
    metadata["source_test_file"] = "pendigits.tes"
    return DatasetBundle("pendigits", all_df, train_df, test_df, metadata)


def load_magic(raw_root: Path, seed: int) -> DatasetBundle:
    data = read_zip_text(raw_root, "magic", "magic_gamma_telescope.zip", "magic04.data")
    raw_df = pd.read_csv(data, header=None)
    all_df, mapping = build_dataframe(raw_df, label_col=10)
    train_df, test_df = stratified_split(all_df, seed=seed)
    metadata = base_metadata(
        "magic", 159, "magic/magic_gamma_telescope.zip", "stratified_80_20_by_label",
        mapping, 10, 2, len(all_df), len(train_df), len(test_df), seed,
    )
    return DatasetBundle("magic", all_df, train_df, test_df, metadata)


def load_sensorless(raw_root: Path, seed: int) -> DatasetBundle:
    data = read_zip_text(raw_root, "sensorless", "sensorless_drive_diagnosis.zip", "Sensorless_drive_diagnosis.txt")
    raw_df = pd.read_csv(data, header=None, sep=r"\s+")
    all_df, mapping = build_dataframe(raw_df, label_col=48)
    train_df, test_df = stratified_split(all_df, seed=seed)
    metadata = base_metadata(
        "sensorless", 325, "sensorless/sensorless_drive_diagnosis.zip", "stratified_80_20_by_label",
        mapping, 48, 11, len(all_df), len(train_df), len(test_df), seed,
    )
    return DatasetBundle("sensorless", all_df, train_df, test_df, metadata)


def load_miniboone(raw_root: Path, seed: int) -> DatasetBundle:
    data = read_zip_text(raw_root, "miniboone", "miniboone_particle_identification.zip", "MiniBooNE_PID.txt")
    first_line = data.readline().decode("utf-8").strip()
    n_signal, n_background = [int(value) for value in first_line.split()]
    raw_df = pd.read_csv(data, header=None, sep=r"\s+")
    labels = [1] * n_signal + [0] * n_background
    if len(raw_df) != len(labels):
        raise ValueError(f"MiniBooNE row count mismatch: {len(raw_df)} rows vs {len(labels)} labels.")
    raw_df["label_raw"] = labels
    all_df, mapping = build_dataframe(raw_df, label_col="label_raw", label_mapping={0: 0, 1: 1})
    train_df, test_df = stratified_split(all_df, seed=seed)
    metadata = base_metadata(
        "miniboone", 199, "miniboone/miniboone_particle_identification.zip", "stratified_80_20_by_label",
        mapping, 50, 2, len(all_df), len(train_df), len(test_df), seed,
    )
    metadata["raw_class_counts"] = {"signal": n_signal, "background": n_background}
    metadata["label_meaning"] = {"0": "background", "1": "signal"}
    return DatasetBundle("miniboone", all_df, train_df, test_df, metadata)


def load_skin(raw_root: Path, seed: int) -> DatasetBundle:
    data = read_zip_text(raw_root, "skin", "skin_segmentation.zip", "Skin_NonSkin.txt")
    raw_df = pd.read_csv(data, header=None, sep=r"\s+")
    all_df, mapping = build_dataframe(raw_df, label_col=3)
    train_df, test_df = stratified_split(all_df, seed=seed)
    metadata = base_metadata(
        "skin", 229, "skin/skin_segmentation.zip", "stratified_80_20_by_label",
        mapping, 3, 2, len(all_df), len(train_df), len(test_df), seed,
    )
    return DatasetBundle("skin", all_df, train_df, test_df, metadata)


def load_covertype(raw_root: Path, seed: int) -> DatasetBundle:
    raw_zip = raw_root / "covertype" / "covertype.zip"
    with zipfile.ZipFile(raw_zip) as zf:
        with gzip.open(io.BytesIO(zf.read("covtype.data.gz")), "rt") as gz:
            raw_df = pd.read_csv(gz, header=None)
    all_df, mapping = build_dataframe(raw_df, label_col=54)
    train_df, test_df = stratified_split(all_df, seed=seed)
    metadata = base_metadata(
        "covertype", 31, "covertype/covertype.zip", "stratified_80_20_by_label",
        mapping, 54, 7, len(all_df), len(train_df), len(test_df), seed,
    )
    return DatasetBundle("covertype", all_df, train_df, test_df, metadata)


def write_susy_csvs(raw_root: Path, output_root: Path, overwrite: bool) -> None:
    dataset = "susy"
    out_dir = output_root / dataset
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} already exists. Use --overwrite to replace existing files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    header = feature_columns(18) + ["label"]
    raw_zip = raw_root / "susy" / "susy.zip"
    n_train = 4_500_000
    n_test = 500_000
    n_total = n_train + n_test

    with zipfile.ZipFile(raw_zip) as zf:
        with gzip.open(io.BytesIO(zf.read("SUSY.csv.gz")), "rt", newline="") as gz, train_path.open(
            "w", newline="", encoding="utf-8"
        ) as train_file, test_path.open("w", newline="", encoding="utf-8") as test_file:
            train_writer = csv.writer(train_file)
            test_writer = csv.writer(test_file)
            train_writer.writerow(header)
            test_writer.writerow(header)

            for row_idx, line in enumerate(gz):
                parts = line.rstrip("\n").split(",")
                if len(parts) != 19:
                    raise ValueError(f"SUSY row {row_idx} has {len(parts)} columns, expected 19.")
                label = str(int(float(parts[0])))
                output_row = parts[1:] + [label]
                if row_idx < n_train:
                    train_writer.writerow(output_row)
                else:
                    test_writer.writerow(output_row)

            if row_idx + 1 != n_total:
                raise ValueError(f"SUSY row count mismatch: {row_idx + 1} rows, expected {n_total}.")

    metadata = base_metadata(
        "susy",
        279,
        "susy/susy.zip",
        "official_last_500k_test",
        {"0": 0, "1": 1},
        18,
        2,
        n_total,
        n_train,
        n_test,
        None,
    )
    metadata["source_file"] = "SUSY.csv.gz"
    metadata["files"] = {"train": "train.csv", "test": "test.csv"}
    metadata["notes"] = "UCI metadata states that the last 500,000 examples are used as a test set."
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


LOADERS: Dict[str, Callable[[Path, int], DatasetBundle]] = {
    "wdbc": load_wdbc,
    "banknote": load_banknote,
    "pendigits": load_pendigits,
    "magic": load_magic,
    "sensorless": load_sensorless,
    "miniboone": load_miniboone,
    "skin": load_skin,
    "covertype": load_covertype,
}

ALL_DATASETS = list(LOADERS.keys()) + ["susy"]


def prepare_dataset(dataset: str, raw_root: Path, output_root: Path, seed: int, overwrite: bool) -> None:
    if dataset == "susy":
        write_susy_csvs(raw_root, output_root, overwrite=overwrite)
        return

    if dataset not in LOADERS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {', '.join(ALL_DATASETS)}")

    out_dir = output_root / dataset
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already exists. Use --overwrite to replace existing files.")

    bundle = LOADERS[dataset](raw_root, seed)
    write_dataset_csvs(bundle, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw UCI archives to clean CSV train/test splits.")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name or 'all'.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing processed CSV directories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    args.output_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        print(f"[prepare] {dataset}")
        prepare_dataset(
            dataset=dataset,
            raw_root=args.raw_root,
            output_root=args.output_root,
            seed=args.split_seed,
            overwrite=args.overwrite,
        )

    print(f"[done] processed CSVs saved to {args.output_root}")


if __name__ == "__main__":
    main()
