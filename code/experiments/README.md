# Experiments

This directory stores the Lab1-Lab6 experiment matrices used by the runners in `../scripts/`.

| File | Paper experiment |
| --- | --- |
| `lab1.yaml` | Overall Evaluation: main comparison in Table `tab:fixed_noise` |
| `lab2.yaml` | Overall Evaluation: random/shift dirty-data comparison in Figure `fig:random-errors`, Figure `fig:shift-errors`, and Table `tab:skin_covertype` |
| `lab3.yaml` | Ablation Study: Figure `fig:ablation` |
| `lab4.yaml` | Sensitivity analysis: initial training epochs in Figure `fig:initial-epochs` |
| `lab5.yaml` | Sensitivity analysis: coreset budget in Figure `fig:sensitivity` |
| `lab6.yaml` | Sensitivity analysis: sliding-window length in Figure `fig:sensitivity` |

Run labs from the `code/` directory with `bash scripts/run_lab*.sh`.
