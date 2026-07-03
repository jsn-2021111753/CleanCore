"""Unified experiment entry point for CleanCore and baselines."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from common.config import deep_update, load_default_config, load_yaml_config
from common.data import load_clean_dataset, load_noisy_dataset
from common.interfaces import MethodOutput
from common.metrics import classification_metrics
from common.model import MLP
from common.paths import CODE_ROOT, CONFIG_ROOT, result_dir
from common.preprocessing import StandardizePreprocessor
from common.results import save_config, save_history_csv, save_json
from common.seed import set_seed
from common.training import TrainingConfig, predict_classifier, train_classifier
from common.timing import IncrementalTimingRecorder
from methods.base import MethodContext
from methods.registry import get_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CleanCore experiment.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. wdbc.")
    parser.add_argument("--method", type=str, required=True, help="Method name, e.g. cleancore.")
    parser.add_argument("--noise", type=str, default="random", help="Noise type: clean, random, or shift.")
    parser.add_argument("--noise_rate", type=float, default=0.20, help="Noise rate used by the experiment.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Default config path.")
    parser.add_argument("--method_config", type=str, default="", help="Optional method config path.")
    parser.add_argument(
        "--method_config_group",
        type=str,
        default="",
        help="Optional method config group, e.g. lab1. Loads configs/methods/<group>/<method>.yaml and configs/methods/<group>/<dataset>/<method>.yaml.",
    )
    parser.add_argument("--output_dir", type=str, default="", help="Optional explicit output directory.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override final training max epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override shared training batch size.")
    parser.add_argument("--subset_fraction", type=float, default=None, help="Override method subset fraction when applicable.")
    return parser.parse_args()


def _resolve_config_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return CODE_ROOT / p


def load_method_config_layers(
    method: str,
    dataset: str,
    method_config_group: str = "",
    explicit_method_config: str = "",
) -> tuple[Dict[str, Any], list[str]]:
    """Load method config layers from broad defaults to the requested run."""

    method_name = method.lower()
    paths: list[Path] = [CONFIG_ROOT / "methods" / "defaults" / f"{method_name}.yaml"]
    group = str(method_config_group or "").strip()
    if group:
        paths.append(CONFIG_ROOT / "methods" / group / "defaults" / f"{method_name}.yaml")
        paths.append(CONFIG_ROOT / "methods" / group / dataset / f"{method_name}.yaml")
    if explicit_method_config:
        paths.append(_resolve_config_path(explicit_method_config))

    merged: Dict[str, Any] = {}
    sources: list[str] = []
    for path in paths:
        if path.exists():
            merged = deep_update(merged, load_yaml_config(path))
            sources.append(str(path))
    return merged, sources


def load_experiment_config(args: argparse.Namespace) -> Dict[str, Any]:
    default_path = _resolve_config_path(args.config)
    cfg = load_yaml_config(default_path)
    method_cfg, method_sources = load_method_config_layers(
        method=args.method,
        dataset=args.dataset,
        method_config_group=getattr(args, "method_config_group", ""),
        explicit_method_config=getattr(args, "method_config", ""),
    )
    cfg = deep_update(cfg, {"method": method_cfg})
    overrides: Dict[str, Any] = {}
    if getattr(args, "max_epochs", None) is not None:
        overrides = deep_update(overrides, {"training": {"max_epochs": int(args.max_epochs)}})
    if getattr(args, "batch_size", None) is not None:
        overrides = deep_update(overrides, {"training": {"batch_size": int(args.batch_size)}})
    if getattr(args, "subset_fraction", None) is not None:
        overrides = deep_update(overrides, {"method": {"subset_fraction": float(args.subset_fraction)}})
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg = deep_update(
        cfg,
        {
            "run": {
                "dataset": args.dataset,
                "method": args.method.lower(),
                "noise": args.noise,
                "noise_rate": float(args.noise_rate),
                "seed": int(args.seed),
                "method_config_group": str(getattr(args, "method_config_group", "") or ""),
                "method_config_sources": method_sources,
            }
        },
    )
    return cfg


def training_config_from_dict(data: Dict[str, Any]) -> TrainingConfig:
    names = {f.name for f in fields(TrainingConfig)}
    return TrainingConfig(**{k: v for k, v in data.items() if k in names})


def prepare_training_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method_output: MethodOutput,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    selected = np.asarray(method_output.selected_indices, dtype=np.int64)
    X_base = method_output.corrected_features if method_output.corrected_features is not None else X_train
    y_base = method_output.corrected_labels if method_output.corrected_labels is not None else y_train
    X_selected = np.asarray(X_base[selected], dtype=np.float32)
    y_selected = np.asarray(y_base[selected], dtype=np.int64)
    weights = np.asarray(method_output.sample_weights, dtype=np.float32)
    soft_targets = None
    if method_output.soft_targets is not None:
        soft_targets = np.asarray(method_output.soft_targets[selected], dtype=np.float32)
    if len(weights) != len(selected):
        raise ValueError("Method sample_weights must align with selected_indices.")
    return X_selected, y_selected, weights, soft_targets


def save_method_output(path: Path, out: MethodOutput, save_npz: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, Any] = {
        "selected_indices": out.selected_indices,
        "sample_weights": out.sample_weights,
    }
    if out.corrected_labels is not None:
        arrays["corrected_labels"] = out.corrected_labels
    if out.corrected_features is not None:
        arrays["corrected_features"] = out.corrected_features
    if out.soft_targets is not None:
        arrays["soft_targets"] = out.soft_targets
    if out.predicted_noisy_mask is not None:
        arrays["predicted_noisy_mask"] = out.predicted_noisy_mask.astype(np.uint8)
    if out.final_predictions is not None:
        arrays["final_predictions"] = out.final_predictions
    if bool(save_npz):
        np.savez(path / "method_output.npz", **arrays)
    (path / "method_metadata.json").write_text(
        json.dumps(out.metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    started = time.perf_counter()
    cfg = load_experiment_config(args)
    set_seed(int(args.seed))

    out_dir = Path(args.output_dir) if args.output_dir else result_dir(
        args.dataset,
        args.method.lower(),
        args.noise,
        float(args.noise_rate),
        int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = IncrementalTimingRecorder(out_dir / "timing_progress.json")

    try:
        return _run_experiment_with_timing(args, cfg, out_dir, timing, started)
    except BaseException as exc:
        timing.fail(exc)
        raise


def _run_experiment_with_timing(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    out_dir: Path,
    timing: IncrementalTimingRecorder,
    started: float,
) -> Dict[str, float]:

    with timing.phase("pipeline.data_loading"):
        if args.noise == "clean":
            data = load_clean_dataset(args.dataset)
        else:
            data = load_noisy_dataset(args.dataset, args.noise, float(args.noise_rate))

    X_train = data.X_train
    X_test = data.X_test
    with timing.phase("pipeline.preprocessing"):
        if bool(cfg.get("preprocessing", {}).get("standardize", True)):
            preprocessor = StandardizePreprocessor()
            X_train, X_test = preprocessor.fit_transform(X_train, X_test)

    method = get_method(args.method)
    method_started = time.perf_counter()
    ctx = MethodContext(
        X_train=X_train,
        y_train=data.y_train,
        X_test=X_test,
        num_classes=data.num_classes,
        seed=int(args.seed),
        output_dir=out_dir,
        model_config=dict(cfg.get("model", {})),
        training_config=dict(cfg.get("training", {})),
        method_config=dict(cfg.get("method", {})),
        timing=timing,
    )
    with timing.phase("pipeline.method"):
        method_output = method.run(ctx)
    method_time = time.perf_counter() - method_started
    method_output.metadata.setdefault("stage_timing_totals", timing.totals())
    method_output.metadata.setdefault("stage_timing_counts", timing.counts())
    with timing.phase("pipeline.artifact_save"):
        save_method_output_npz = bool(cfg.get("experiment", {}).get("save_method_output_npz", True))
        save_method_output(out_dir, method_output, save_npz=save_method_output_npz)
        save_config(out_dir / "config.yaml", cfg)

    if method_output.final_predictions is not None:
        y_pred = np.asarray(method_output.final_predictions, dtype=np.int64)
        if len(y_pred) != len(data.y_test):
            raise ValueError("Method final_predictions must align with the test set.")
        with timing.phase("pipeline.evaluation"):
            metrics = classification_metrics(data.y_test, y_pred, labels=list(range(data.num_classes)))
        total_time = time.perf_counter() - started
        train_time = float(method_output.metadata.get("train_time_sec", method_time))
        final = {
            **metrics,
            "total_time_sec": float(total_time),
            "method_time_sec": float(method_time),
            "train_time_sec": train_time,
            "epochs_ran": float(method_output.metadata.get("epochs_ran", len(method_output.training_history))),
            "n_train_input": float(len(data.y_train)),
            "n_train_selected": float(len(method_output.selected_indices)),
        }
        with timing.phase("pipeline.final_save"):
            save_json(out_dir / "final_metrics.json", final)
            save_json(
                out_dir / "runtime.json",
                {
                    "total_time_sec": float(total_time),
                    "method_time_sec": float(method_time),
                    "train_time_sec": train_time,
                    "stage_timing_totals": timing.totals(),
                },
            )
            save_history_csv(out_dir / "history.csv", method_output.training_history)
        timing.complete()
        return final

    X_method, y_method, sample_weights, soft_targets = prepare_training_arrays(X_train, data.y_train, method_output)

    train_cfg = training_config_from_dict(dict(cfg.get("training", {})))
    model_cfg = dict(cfg.get("model", {}))
    model = MLP(
        input_dim=int(X_method.shape[1]),
        num_classes=data.num_classes,
        hidden_dims=tuple(model_cfg.get("hidden_dims", (256, 128))),
        dropout=float(model_cfg.get("dropout", 0.2)),
        batch_norm=bool(model_cfg.get("batch_norm", True)),
    )
    with timing.phase("pipeline.final_training"):
        train_result = train_classifier(
            model=model,
            X_train=X_method,
            y_train=y_method,
            cfg=train_cfg,
            output_dir=out_dir,
            sample_weights=sample_weights,
            soft_targets=soft_targets,
            seed=int(args.seed),
        )

    with timing.phase("pipeline.evaluation"):
        y_pred = predict_classifier(
            model,
            X_test,
            batch_size=int(train_cfg.batch_size),
            device=str(train_cfg.device),
        )
        metrics = classification_metrics(data.y_test, y_pred, labels=list(range(data.num_classes)))
    total_time = time.perf_counter() - started
    final = {
        **metrics,
        "total_time_sec": float(total_time),
        "method_time_sec": float(method_time),
        "train_time_sec": float(train_result.train_time_sec),
        "epochs_ran": float(train_result.epochs_ran),
        "n_train_input": float(len(data.y_train)),
        "n_train_selected": float(len(X_method)),
    }
    with timing.phase("pipeline.final_save"):
        save_json(out_dir / "final_metrics.json", final)
        save_json(
            out_dir / "runtime.json",
            {
                "total_time_sec": float(total_time),
                "method_time_sec": float(method_time),
                "train_time_sec": float(train_result.train_time_sec),
                "stage_timing_totals": timing.totals(),
            },
        )
        save_history_csv(out_dir / "history.csv", train_result.history)
    timing.complete()
    return final


def main() -> None:
    args = parse_args()
    final = run_experiment(args)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
