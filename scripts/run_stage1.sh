#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
output="${1:-outputs/stage1-v1}"
mkdir -p -- "$(dirname -- "$output")"
exec 9>"${output}.lock"
flock -n 9 || { echo 'This stage is already running.' >&2; exit 1; }
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
printf 'running\n' > "${output}.status"
trap 'code=$?; if [ "$code" -eq 0 ]; then printf "complete\n"; else printf "failed (%s)\n" "$code"; fi > "${output}.status"' EXIT
git rev-parse HEAD > "${output}.commit"
.venv/bin/python -m gofa_f.benchmark --seeds 42 --horizons 12 --output "$output"
.venv/bin/python -m gofa_f.report "$output" --output "${output}.report.md"
