#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
output="${1:?Specify a NEW output directory, or --resume as second argument}"
mkdir -p -- "$(dirname -- "$output")"
exec 9>"${output}.lock"
flock -n 9 || { echo 'Native stage is already running.' >&2; exit 1; }
exec 8>outputs/native-stage.lock
flock -n 8 || { echo 'Another native stage is already running.' >&2; exit 1; }
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
if [[ "${2:-}" == "--resume" ]]; then
  exec .venv-native/bin/python -u -m gofa_f.native_stage --output "$output" --resume
else
  test ! -e "$output" || { echo 'Output exists; inspect it and explicitly resume.' >&2; exit 1; }
  exec .venv-native/bin/python -u -m gofa_f.native_stage --output "$output"
fi
