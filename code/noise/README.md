# Noise

This directory contains the controlled dirty-data generators used by the data preparation scripts.

- `injector.py`: shared helpers for assigning dirty records and applying valid feature changes.
- `random_noise.py`: random dirty-data pattern.
- `shift_noise.py`: distribution-shift dirty-data pattern.

Prepared dirty datasets are written under:

```text
data/processed/npz_noisy/<noise_type>/rate_<rate>/<dataset>/
```
