# Scripts

- `download_uci_archives.py`: download official UCI zip archives into `data/raw/`.
- `prepare_csv_datasets.py`: convert raw UCI archives to unified clean CSV train/test splits.
- `prepare_npz_datasets.py`: convert clean CSV files to aligned clean NPZ files.
- `make_noisy_npz.py`: generate fixed random or distribution-shift noisy NPZ files from clean NPZ files.
- `prepare_rein_datasets.py`: prepare SmartFactory and HAR style REIN clean/dirty CSVs as NPZ inputs.
- `validate_noisy_npz.py`: check noisy NPZ schema and corruption metadata.
- `run_final_lab.py`: build and execute the final Lab1-Lab6 job sets.
- `run_lab1.sh` ... `run_lab6.sh`: run individual labs.
- `run_all_labs.sh`: run all labs in order.

Common commands:

```bash
python scripts/download_uci_archives.py --dataset all
python scripts/prepare_csv_datasets.py --dataset all --overwrite
python scripts/prepare_npz_datasets.py --dataset all --overwrite
python scripts/make_noisy_npz.py --dataset all --noise-type all --noise-rate 0.20 --seed 42 --overwrite
python scripts/validate_noisy_npz.py --dataset all --noise-type random --noise-rate 0.20
```

Batch experiments:

```bash
bash scripts/run_lab1.sh --dry-run
bash scripts/run_lab1.sh
bash scripts/run_all_labs.sh
```

Successful jobs write `final_metrics.json`; failed jobs retain `run.log` in their output directory.
