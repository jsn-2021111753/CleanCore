# Data

This directory contains bundled validation inputs and data-source notes. WDBC and Banknote inputs are included so the artifact can be executed immediately after dependency installation.

```text
processed/npz_clean/wdbc/
processed/npz_clean/banknote/
processed/npz_noisy/random/rate_0.20/wdbc/
processed/npz_noisy/random/rate_0.20/banknote/
```

The remaining public datasets can be prepared with the scripts in `../scripts/`. Raw data sources are listed in `raw/SOURCES.md`.

Expected processed layout:

```text
processed/
  csv_clean/<dataset>/{all.csv,train.csv,test.csv,metadata.json}
  npz_clean/<dataset>/{train.npz,test.npz,metadata.json}
  npz_noisy/<noise_type>/rate_<rate>/<dataset>/
    train.npz
    test.npz
    noise_info.npz
    metadata.json
    noise_info.json
```

`run.py` reads `train.npz` and `test.npz` for method execution.
