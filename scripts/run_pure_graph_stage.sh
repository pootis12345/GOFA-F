#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
output="${1:-outputs/stage4-pure-graph-v1}"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export DGLBACKEND=pytorch
export PYTHONDONTWRITEBYTECODE=1
exec .venv-native/bin/python -u -m gofa_f.idle_launcher --output "$output" --unlimited-active \
  --memory-cap-gib 40 --leave-free-gib 16 \
  --wall-deadline '2026-10-15T11:16:00+08:00' -- \
  .venv-native/bin/python -u -m gofa_f.pure_graph_stage \
  --config configs/stage4_pure_graph.yaml --output "$output"
