#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" scripts/prepare_rein_datasets.py \
  --dataset smartfactory \
  --noise-type rein_smartfactory \
  --noise-rate 0.0 \
  --input-alignment csv \
  --overwrite \
  "$@"
