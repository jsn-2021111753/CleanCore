# CleanCore

CleanCore is the artifact for the paper **"CleanCore: Reliable Coresets for Robust and Efficient Learning over Dirty Data"**.  The repository contains the experiment runner, method implementations, final configuration files, compact result summaries, and small bundled data for quick verification.

Large regenerated data files are intentionally not committed.  The full processed data directory is about 18 GB locally, so this public artifact keeps only two small NPZ smoke-test datasets and provides scripts to download raw UCI archives and regenerate the remaining data.

## Repository Layout

```text
code/
  run.py                  # single experiment entry point
  common/                 # shared data loading, model, training, metrics, results
  methods/                # CleanCore and baseline implementations
  noise/                  # random and distribution-shift corruption generators
  configs/                # final method and sweep configurations
  experiments/            # Lab1-Lab6 experiment matrices
  scripts/                # data preparation and batch runners
  data/                   # small bundled smoke-test NPZs plus data-source notes
  results/                # compact final metrics and summary CSVs
rein-datasets/
  smartfactory/           # small REIN clean/dirty CSVs used by the real dirty-data setting
```

## Environment

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The experiments were designed for CPU execution.  Large sweeps can take many hours because they reproduce multiple baselines over large tabular datasets.

## Quick Smoke Test

The repository includes preprocessed WDBC and Banknote NPZ files so reviewers can run the code immediately after installing dependencies.

```bash
cd code
python run.py \
  --dataset wdbc \
  --method cleancore \
  --noise random \
  --noise_rate 0.20 \
  --seed 42 \
  --config configs/default_no_artifacts.yaml \
  --method_config configs/methods/lab1/wdbc/cleancore/cleancore.yaml \
  --output_dir results/smoke/wdbc_cleancore \
  --max_epochs 5 \
  --batch_size 64
```

Expected output files:

```text
code/results/smoke/wdbc_cleancore/final_metrics.json
code/results/smoke/wdbc_cleancore/config.yaml
code/results/smoke/wdbc_cleancore/history.csv
```

## Full Data Preparation

From the repository root:

```bash
cd code
python scripts/download_uci_archives.py --dataset all
python scripts/prepare_csv_datasets.py --dataset all --overwrite
python scripts/prepare_npz_datasets.py --dataset all --overwrite
python scripts/make_noisy_npz.py --dataset all --noise-type all --noise-rate 0.20 --seed 42 --overwrite
```

Lab2-Lab6 use additional noise rates.  Generate them as needed:

```bash
for rate in 0.06 0.12 0.18 0.24 0.30 0.40; do
  python scripts/make_noisy_npz.py --dataset all --noise-type all --noise-rate "$rate" --seed 42 --overwrite
done
```

SmartFactory uses the bundled REIN clean/dirty CSV files:

```bash
python scripts/prepare_rein_datasets.py \
  --dataset smartfactory \
  --noise-type rein_missing_only \
  --noise-rate 0.0 \
  --feature-normalization missing_only \
  --overwrite
```

## Running Experiments

Dry-run Lab1 to inspect the resolved commands:

```bash
cd code
bash scripts/run_lab1.sh --dry-run
```

Run individual labs:

```bash
bash scripts/run_lab1.sh
bash scripts/run_lab2.sh
bash scripts/run_lab3.sh
bash scripts/run_lab4.sh
bash scripts/run_lab5.sh
bash scripts/run_lab6.sh
```

Run the full suite:

```bash
bash scripts/run_all_labs.sh
```

Use `PYTHON_BIN=/path/to/python` if the desired interpreter is not `python3`.

## Included Results

`code/results/` contains compact `final_metrics.json` files and summary CSVs from the paper experiments.  Heavy artifacts such as model checkpoints, per-sample method outputs, logs, and regenerated data are excluded from git.  New runs write their outputs under `code/results/`.
