# Lab3 Ablation Results

This folder contains the final Lab3 CleanCore ablation configs for random 40% errors.

Variants:
- `full`: complete CleanCore.
- `no_weight`: disables sample reliability weighting while keeping error handling and feature repair enabled.
- `no_handle`: disables error handling, feature repair, and sample reliability weighting.

Selected datasets:
- banknote
- pendigits
- magic
- sensorless

CSV files:
- `lab3_ablation_summary.csv`: long-form results, one row per dataset and variant.
- `lab3_ablation_plot.csv`: wide-form results for plotting grouped bars.

Only lightweight final metrics are kept here; large model and method output artifacts are intentionally omitted.
