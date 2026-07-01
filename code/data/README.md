# Data

This public repository includes only compact data needed for immediate smoke testing:

```text
processed/npz_clean/wdbc/
processed/npz_clean/banknote/
processed/npz_noisy/random/rate_0.20/wdbc/
processed/npz_noisy/random/rate_0.20/banknote/
```

The full local processed data directory is much larger, so regenerated CSV/NPZ files for the remaining datasets are excluded from git.  Use the scripts in `../scripts/` to rebuild them.

## Expected Structure

```text
raw/
  <dataset>/<official UCI zip>
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

`train.npz` and `test.npz` contain `X` and `y`.  Clean NPZ files also contain `row_id`.  `noise_info.npz` contains corruption ground truth for validation and debugging; it is not read by `run.py` during method training.

Raw UCI sources are listed in `raw/SOURCES.md`.
