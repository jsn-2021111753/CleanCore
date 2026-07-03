# CleanCore Experiment Code

This directory is the executable root for the CleanCore artifact.

## Main Components

- `run.py`: runs one dataset, method, dirty-data setting, and seed.
- `common/`: shared data loading, preprocessing, model, training, metrics, and result utilities.
- `methods/`: CleanCore and baseline implementations.
- `noise/`: controlled dirty-data generators.
- `configs/`: configuration files consumed by `run.py` and the lab runners.
- `experiments/`: Lab1-Lab6 experiment matrices.
- `scripts/`: data preparation and batch runners.
- `data/`: bundled validation data and data-source notes.
- `results/`: compact final metrics and summaries.

## Lab Commands

```bash
bash scripts/run_lab1.sh
bash scripts/run_lab2.sh
bash scripts/run_lab3.sh
bash scripts/run_lab4.sh
bash scripts/run_lab5.sh
bash scripts/run_lab6.sh
```

Run all labs in order:

```bash
bash scripts/run_all_labs.sh
```

Use `PYTHON_BIN=/path/to/python` before a command when a specific interpreter should be used.
