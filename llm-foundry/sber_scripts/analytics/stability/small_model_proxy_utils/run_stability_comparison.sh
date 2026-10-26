#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR="${1:-/home/jovyan/vmamedov/tensorboard_papaya_o5/papayaseek-ultra-3_5-lr_explosion}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/artifacts}"
RESULT_NAME="${3:-stability_exps_res}"

python "${SCRIPT_DIR}/run_stability_comparison.py" \
  "${INPUT_DIR}" \
  "${OUTPUT_DIR}" \
  "${RESULT_NAME}" \
  --min-full-steps 16000 \
  --max-vio-threshold-share 0.2 \
  --max-vio-drop-points 10 \
  --max-vio-top-k 40 \
  --workers 64 \
  --use_cache
