# CleanCore

This repository contains the artifact for the paper **"CleanCore: Data-Quality-Aware Coreset Selection for Learning over Dirty Tabular Data"**. It includes the implementation, public-data preparation scripts, experiment runners, final experiment configurations, and compact result files used to reproduce the paper's empirical results.

## Repository Layout

```text
code/
  run.py                  # single experiment entry point
  common/                 # shared data loading, model, training, metrics, and result utilities
  methods/                # CleanCore and baseline implementations
  noise/                  # controlled dirty-data generators
  configs/                # configurations consumed by the lab runners
  experiments/            # Lab1-Lab6 experiment matrices
  scripts/                # data preparation and batch runners
  data/                   # bundled validation data and data-source notes
  results/                # compact final metrics and summaries
rein-datasets/
  smartfactory/           # REIN SmartFactory clean/dirty CSVs
```

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the dependency set used to validate this artifact, install `requirements-tested.txt` instead of `requirements.txt`.

## Data Preparation

The repository includes WDBC and Banknote inputs for immediate execution checks. The remaining public datasets can be prepared from their listed sources.

```bash
cd code
python scripts/download_uci_archives.py --dataset all
python scripts/prepare_csv_datasets.py --dataset all --overwrite
python scripts/prepare_npz_datasets.py --dataset all --overwrite
```

Prepare the controlled dirty-data inputs used by the labs:

```bash
python scripts/make_noisy_npz.py --dataset all --noise-type all --noise-rate 0.20 --seed 42 --overwrite
python scripts/make_noisy_npz.py --dataset all --noise-type all --noise-rate 0.06 --seed 42 --overwrite
python scripts/make_noisy_npz.py --dataset all --noise-type all --noise-rate 0.12 --seed 42 --overwrite
python scripts/make_noisy_npz.py --dataset all --noise-type all --noise-rate 0.18 --seed 42 --overwrite
python scripts/make_noisy_npz.py --dataset all --noise-type all --noise-rate 0.24 --seed 42 --overwrite
python scripts/make_noisy_npz.py --dataset all --noise-type all --noise-rate 0.30 --seed 42 --overwrite
python scripts/make_noisy_npz.py --dataset all --noise-type random --noise-rate 0.40 --seed 42 --overwrite
```

The SmartFactory clean/dirty CSV files used by Lab1 are included under
`rein-datasets/smartfactory/`. The following command prepares them for the
experiment runners.

```bash
bash scripts/prepare_smartfactory.sh
```

## Lab Mapping

| Lab | Paper experiment |
| --- | --- |
| Lab1 | Overall Evaluation: main comparison in Table `tab:fixed_noise` |
| Lab2 | Overall Evaluation: random/shift dirty-data comparison in Figure `fig:random-errors`, Figure `fig:shift-errors`, and Table `tab:skin_covertype` |
| Lab3 | Ablation Study: Figure `fig:ablation` |
| Lab4 | Sensitivity analysis: initial training epochs in Figure `fig:initial-epochs` |
| Lab5 | Sensitivity analysis: coreset budget in Figure `fig:sensitivity` |
| Lab6 | Sensitivity analysis: sliding-window length in Figure `fig:sensitivity` |

## Running the Labs

Run commands from `code/`.

```bash
cd code
bash scripts/run_lab1.sh
bash scripts/run_lab2.sh
bash scripts/run_lab3.sh
bash scripts/run_lab4.sh
bash scripts/run_lab5.sh
bash scripts/run_lab6.sh
```

To run the full sequence:

```bash
bash scripts/run_all_labs.sh
```

The runners write final metrics under `code/results/lab*/`. Existing compact result files are kept in the same layout for direct comparison with the paper experiments listed above.

## Single-Run Check

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
  --output_dir results/checks/wdbc_cleancore
```

The run writes:

```text
code/results/checks/wdbc_cleancore/final_metrics.json
```
