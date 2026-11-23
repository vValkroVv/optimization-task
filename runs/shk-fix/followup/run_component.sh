#!/usr/bin/env bash
set -eo pipefail

checkout=$1
output=$2
gpu=$3
program=$4
shift 4
trap 'printf "%s\n" "$?" > "$output/exit"' EXIT
source /etc/profile.d/wm.sh
set -u

cd "$checkout"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES
test -z "$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)"
export PYTHONPATH="$checkout/gift:$checkout/gift/contrib/flash-linear-attention${PYTHONPATH:+:$PYTHONPATH}"
export DG_JIT_CACHE_DIR="$output/deepgemm" DG_CACHE_FORCE_SKIP=1
export TORCHINDUCTOR_CACHE_DIR="$output/inductor" TRITON_CACHE_DIR="$output/triton"
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum "$program" > "$output/runner.sha256"
printf 'GPU index %s, UUID %s\n' "$gpu" "$CUDA_VISIBLE_DEVICES"
timeout --signal=TERM --kill-after=30 900 /opt/torchtitan-venv/bin/python3.12 -B -u "$program" "$@"
