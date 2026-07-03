# Scripts

This directory contains the data preparation scripts and Lab1-Lab6 runners.

## Data Preparation

```bash
python scripts/download_uci_archives.py --dataset all
python scripts/prepare_csv_datasets.py --dataset all --overwrite
python scripts/prepare_npz_datasets.py --dataset all --overwrite
python scripts/make_noisy_npz.py --dataset all --noise-type all --noise-rate 0.20 --seed 42 --overwrite
bash scripts/prepare_smartfactory.sh
```

Additional dirty-data inputs for Lab2 and Lab3 can be generated with the rates listed in the top-level README.

## Lab Runners

```bash
bash scripts/run_lab1.sh
bash scripts/run_lab2.sh
bash scripts/run_lab3.sh
bash scripts/run_lab4.sh
bash scripts/run_lab5.sh
bash scripts/run_lab6.sh
bash scripts/run_all_labs.sh
```

Each successful run writes `final_metrics.json` under `results/lab*/`.
