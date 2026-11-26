#!/usr/bin/env bash
set -euo pipefail

root=$1
case_name=$2
gpu=$3
shift 3
case "$gpu" in
    0|1|2) ;;
    *) echo "GPU must be 0, 1, or 2" >&2; exit 2 ;;
esac
cd "$root"
source /etc/profile.d/wm.sh
nvidia-smi -i "$gpu" --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
if [[ -n $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo "GPU $gpu is occupied" >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
export PYTHONPATH="$root/integrated/gift:$root"
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR="$root/cache/head-inductor" TRITON_CACHE_DIR="$root/cache/head-triton"
export TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_CUDAGRAPHS=0
exec /opt/torchtitan-venv/bin/python3.12 -Bu -m tuning.lt.lt_bench "$case_name" --root "$root" --out "$root/tuning/lt/results" "$@"
