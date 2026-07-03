"""Run the Lab1-Lab6 reproduction jobs with exact output layouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]

DATASET_ORDER = [
    "wdbc",
    "banknote",
    "pendigits",
    "magic",
    "sensorless",
    "miniboone",
    "skin",
    "covertype",
    "susy",
    "smartfactory",
]

SCHEDULE = {
    "wdbc": {"threads_per_run": 2, "max_parallel": 11, "max_epochs": 800, "batch_size": 64},
    "banknote": {"threads_per_run": 2, "max_parallel": 11, "max_epochs": 800, "batch_size": 64},
    "pendigits": {"threads_per_run": 4, "max_parallel": 8, "max_epochs": 500, "batch_size": 256},
    "magic": {"threads_per_run": 4, "max_parallel": 8, "max_epochs": 400, "batch_size": 512},
    "sensorless": {"threads_per_run": 4, "max_parallel": 8, "max_epochs": 300, "batch_size": 1024},
    "miniboone": {"threads_per_run": 4, "max_parallel": 8, "max_epochs": 250, "batch_size": 2048},
    "skin": {"threads_per_run": 4, "max_parallel": 8, "max_epochs": 200, "batch_size": 2048},
    "covertype": {"threads_per_run": 4, "max_parallel": 6, "max_epochs": 180, "batch_size": 4096},
    "susy": {"threads_per_run": 4, "max_parallel": 6, "max_epochs": 120, "batch_size": 8192},
    "smartfactory": {"threads_per_run": 4, "max_parallel": 8, "max_epochs": 300, "batch_size": 1024},
}

FINAL_CONFIG = "configs/default_no_artifacts.yaml"


@dataclass(frozen=True)
class Job:
    lab: str
    dataset: str
    method: str
    config_path: Path
    output_dir: Path
    noise: str
    noise_rate: float
    seed: int = 42
    label: str = ""

    @property
    def schedule(self) -> dict[str, int]:
        return SCHEDULE[self.dataset]


def rate_dir_name(rate: float) -> str:
    return f"rate_{float(rate):.2f}"


def run_name(noise: str, noise_rate: float, seed: int = 42) -> str:
    return f"{noise}_{rate_dir_name(noise_rate)}_seed{int(seed)}"


def lab1_noise(dataset: str) -> tuple[str, float]:
    if dataset == "smartfactory":
        return "rein_smartfactory", 0.0
    return "random", 0.20


def ordered_existing_dirs(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    names = {p.name: p for p in path.iterdir() if p.is_dir()}
    ordered = [names[name] for name in DATASET_ORDER if name in names]
    ordered.extend(names[name] for name in sorted(names) if name not in DATASET_ORDER)
    return ordered


def method_dirs(dataset_dir: Path) -> list[Path]:
    methods = {p.name: p for p in dataset_dir.iterdir() if p.is_dir()}
    preferred = [
        "cleanlab",
        "misdetect",
        "ctrl",
        "herding",
        "deepfool",
        "gradmatch",
        "coretab",
        "coretab_misdetect",
        "goodcore",
        "deem",
        "misdetect_coretab",
        "cleancore",
    ]
    ordered = [methods[name] for name in preferred if name in methods]
    ordered.extend(methods[name] for name in sorted(methods) if name not in preferred)
    return ordered


def lab1_jobs() -> list[Job]:
    root = CODE_ROOT / "configs/methods/lab1"
    jobs: list[Job] = []
    for dataset_dir in ordered_existing_dirs(root):
        noise, noise_rate = lab1_noise(dataset_dir.name)
        for method_dir in method_dirs(dataset_dir):
            method = method_dir.name
            cfg = method_dir / f"{method}.yaml"
            if not cfg.exists():
                raise FileNotFoundError(cfg)
            jobs.append(
                Job(
                    lab="lab1",
                    dataset=dataset_dir.name,
                    method=method,
                    config_path=cfg,
                    output_dir=CODE_ROOT / "results/lab1" / dataset_dir.name / run_name(noise, noise_rate) / method,
                    noise=noise,
                    noise_rate=noise_rate,
                    label=method,
                )
            )
    return jobs


def lab2_jobs() -> list[Job]:
    root = CODE_ROOT / "configs/methods/lab2"
    jobs: list[Job] = []
    for dataset_dir in ordered_existing_dirs(root):
        for noise in ["random", "shift"]:
            for rate in [0.06, 0.12, 0.18, 0.24, 0.30]:
                for method_dir in method_dirs(dataset_dir):
                    method = method_dir.name
                    cfg = method_dir / f"{method}.yaml"
                    if not cfg.exists():
                        raise FileNotFoundError(cfg)
                    jobs.append(
                        Job(
                            lab="lab2",
                            dataset=dataset_dir.name,
                            method=method,
                            config_path=cfg,
                            output_dir=CODE_ROOT
                            / "results/lab2"
                            / dataset_dir.name
                            / run_name(noise, rate)
                            / method,
                            noise=noise,
                            noise_rate=rate,
                            label=f"{noise}_{rate:.2f}/{method}",
                        )
                    )
    return jobs


def lab3_jobs() -> list[Job]:
    root = CODE_ROOT / "configs/methods/lab3"
    jobs: list[Job] = []
    for dataset_dir in ordered_existing_dirs(root):
        method_dir = dataset_dir / "cleancore"
        for variant in ["full", "no_weight", "no_handle"]:
            cfg = method_dir / f"{variant}.yaml"
            if not cfg.exists():
                raise FileNotFoundError(cfg)
            jobs.append(
                Job(
                    lab="lab3",
                    dataset=dataset_dir.name,
                    method="cleancore",
                    config_path=cfg,
                    output_dir=CODE_ROOT / "results/lab3" / dataset_dir.name / variant,
                    noise="random",
                    noise_rate=0.40,
                    label=variant,
                )
            )
    return jobs


def lab4_jobs() -> list[Job]:
    root = CODE_ROOT / "configs/methods/lab4"
    jobs: list[Job] = []
    for dataset_dir in ordered_existing_dirs(root):
        for pretrain in ["04", "08", "12", "16", "20"]:
            for method_dir in method_dirs(dataset_dir):
                method = method_dir.name
                cfg = method_dir / f"{method}_pretrain{pretrain}.yaml"
                if not cfg.exists():
                    raise FileNotFoundError(cfg)
                jobs.append(
                    Job(
                        lab="lab4",
                        dataset=dataset_dir.name,
                        method=method,
                        config_path=cfg,
                        output_dir=CODE_ROOT / "results/lab4" / dataset_dir.name / f"pretrain{pretrain}" / method,
                        noise="random",
                        noise_rate=0.20,
                        label=f"pretrain{pretrain}/{method}",
                    )
                )
    return jobs


def lab5_jobs() -> list[Job]:
    root = CODE_ROOT / "configs/methods/lab5"
    jobs: list[Job] = []
    for dataset_dir in ordered_existing_dirs(root):
        method_dir = dataset_dir / "cleancore"
        for cfg in sorted(method_dir.glob("cleancore_frac*.yaml")):
            suffix = cfg.stem.replace("cleancore_frac", "")
            fraction = int(suffix) / 100.0
            fraction_label = f"fraction_{fraction:.2f}"
            jobs.append(
                Job(
                    lab="lab5",
                    dataset=dataset_dir.name,
                    method="cleancore",
                    config_path=cfg,
                    output_dir=CODE_ROOT / "results/lab5" / dataset_dir.name / fraction_label / "cleancore",
                    noise="random",
                    noise_rate=0.20,
                    label=fraction_label,
                )
            )
    return jobs


def lab6_jobs() -> list[Job]:
    root = CODE_ROOT / "configs/methods/lab6"
    jobs: list[Job] = []
    for dataset_dir in ordered_existing_dirs(root):
        method_dir = dataset_dir / "cleancore"
        for cfg in sorted(method_dir.glob("cleancore_L*.yaml")):
            suffix = cfg.stem.replace("cleancore_L", "")
            window_label = f"L_{int(suffix)}"
            jobs.append(
                Job(
                    lab="lab6",
                    dataset=dataset_dir.name,
                    method="cleancore",
                    config_path=cfg,
                    output_dir=CODE_ROOT / "results/lab6" / dataset_dir.name / window_label / "cleancore",
                    noise="random",
                    noise_rate=0.20,
                    label=window_label,
                )
            )
    return jobs


JOB_BUILDERS = {
    "lab1": lab1_jobs,
    "lab2": lab2_jobs,
    "lab3": lab3_jobs,
    "lab4": lab4_jobs,
    "lab5": lab5_jobs,
    "lab6": lab6_jobs,
}


def build_command(job: Job, python_executable: str) -> list[str]:
    schedule = job.schedule
    return [
        python_executable,
        str(CODE_ROOT / "run.py"),
        "--dataset",
        job.dataset,
        "--method",
        job.method,
        "--noise",
        job.noise,
        "--noise_rate",
        str(job.noise_rate),
        "--seed",
        str(job.seed),
        "--config",
        FINAL_CONFIG,
        "--method_config",
        str(job.config_path),
        "--output_dir",
        str(job.output_dir),
        "--max_epochs",
        str(schedule["max_epochs"]),
        "--batch_size",
        str(schedule["batch_size"]),
    ]


def child_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    value = str(int(threads))
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ]:
        env[key] = value
    return env


def keep_final_metrics_only(output_dir: Path) -> None:
    final_metrics = output_dir / "final_metrics.json"
    if not final_metrics.exists():
        return
    for path in output_dir.iterdir():
        if path.name == "final_metrics.json":
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            path.rmdir()
        else:
            path.unlink()


def run_one(job: Job, python_executable: str, force: bool) -> dict[str, object]:
    final_metrics = job.output_dir / "final_metrics.json"
    if final_metrics.exists() and not force:
        return {"job": job, "status": "skipped"}
    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.output_dir / "run.log"
    command = build_command(job, python_executable)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("command: " + json.dumps(command, ensure_ascii=False) + "\n")
        log_file.write("\n")
        log_file.flush()
        proc = subprocess.run(
            command,
            cwd=str(CODE_ROOT),
            env=child_env(job.schedule["threads_per_run"]),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = "ok" if proc.returncode == 0 else "failed"
    if status == "ok":
        keep_final_metrics_only(job.output_dir)
    return {"job": job, "status": status, "returncode": proc.returncode}


def group_by_dataset(jobs: list[Job]) -> list[tuple[str, list[Job]]]:
    grouped: list[tuple[str, list[Job]]] = []
    for dataset in DATASET_ORDER:
        dataset_jobs = [job for job in jobs if job.dataset == dataset]
        if dataset_jobs:
            grouped.append((dataset, dataset_jobs))
    extra = sorted({job.dataset for job in jobs} - set(DATASET_ORDER))
    for dataset in extra:
        grouped.append((dataset, [job for job in jobs if job.dataset == dataset]))
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lab1-Lab6 reproduction jobs.")
    parser.add_argument("--lab", choices=sorted(JOB_BUILDERS), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true", help="Re-run jobs even when final_metrics.json exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved jobs without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = JOB_BUILDERS[args.lab]()
    print(f"[{args.lab}] jobs={len(jobs)} config={FINAL_CONFIG}")
    if args.dry_run:
        for job in jobs:
            print(
                json.dumps(
                    {
                        "lab": job.lab,
                        "dataset": job.dataset,
                        "method": job.method,
                        "label": job.label,
                        "noise": job.noise,
                        "noise_rate": job.noise_rate,
                        "method_config": str(job.config_path),
                        "output_dir": str(job.output_dir),
                    },
                    ensure_ascii=False,
                )
            )
        return

    failures = 0
    for dataset, dataset_jobs in group_by_dataset(jobs):
        schedule = SCHEDULE[dataset]
        print(f"[{args.lab}/{dataset}] jobs={len(dataset_jobs)}")
        with ThreadPoolExecutor(max_workers=schedule["max_parallel"]) as pool:
            futures = [pool.submit(run_one, job, args.python, bool(args.force)) for job in dataset_jobs]
            for future in as_completed(futures):
                row = future.result()
                job = row["job"]
                print(f"  {job.method}/{job.label or job.method} seed={job.seed} {row['status']}")
                if row["status"] == "failed":
                    failures += 1
    if failures:
        raise SystemExit(f"{failures} job(s) failed.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
