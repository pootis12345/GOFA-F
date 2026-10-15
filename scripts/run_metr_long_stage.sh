#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
exec .venv-native/bin/python -u -m gofa_f.long_stage \
  --config configs/stage6_metr_long.yaml --output "${1:-outputs/stage6-metr-long-v1}" "${@:2}"
