#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/run_lab1.sh "$@"
bash scripts/run_lab2.sh "$@"
bash scripts/run_lab3.sh "$@"
bash scripts/run_lab4.sh "$@"
bash scripts/run_lab5.sh "$@"
bash scripts/run_lab6.sh "$@"
