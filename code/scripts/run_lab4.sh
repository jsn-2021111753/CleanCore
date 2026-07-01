#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

cmd=(
  "${PYTHON_BIN}" -u scripts/run_final_lab.py
  --lab lab4
  --python "${PYTHON_BIN}"
  "$@"
)

"${cmd[@]}"
