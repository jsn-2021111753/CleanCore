# CleanCore Experiment Code

This directory is the executable experiment root.  Run commands from here unless a command explicitly says it should be run from the repository root.

## Main Components

- `run.py`: unified entry point for one dataset, one method, one noise setting, and one seed.
- `common/`: shared tabular data loading, MLP training, preprocessing, metrics, timing, and result writing.
- `methods/`: CleanCore and baselines used in the paper.
- `noise/`: synthetic random and distribution-shift corruption generators.
- `configs/`: shared defaults and method-specific final configurations.
- `experiments/`: Lab1-Lab6 experiment matrices.
- `scripts/`: data preparation, validation, and batch runners.
- `data/`: bundled smoke-test NPZ files plus source notes.
- `results/`: compact final metrics and summaries.

## One-Run Example

```bash
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

## Batch Examples

```bash
bash scripts/run_lab1.sh --dry-run
bash scripts/run_lab1.sh
bash scripts/run_all_labs.sh
```

The shell wrappers default to `python3`.  Override with:

```bash
PYTHON_BIN=/path/to/python bash scripts/run_lab1.sh
```
