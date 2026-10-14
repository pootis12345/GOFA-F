#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
output="${1:-outputs/stage3-late-context-v1}"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
# CPU supervisor: CUDA is admitted only after three consecutive idle observations.
exec .venv-native/bin/python -u -m gofa_f.idle_launcher --output "$output" --total-seconds 14400 -- \
  .venv-native/bin/python -u -m gofa_f.adaptation_stage --config configs/stage3_late_context.yaml --output "$output"
