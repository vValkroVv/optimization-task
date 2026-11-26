#!/usr/bin/env bash
set -eo pipefail
root=$1
gpu=$2
model=$3
shift 3
case "$gpu" in
    0|1|2) ;;
    *) echo "GPU must be 0, 1, or 2" >&2; exit 2 ;;
esac
source /etc/profile.d/wm.sh
set -u
cd "$root"
work="$root/length-grid/region-tuning/dw"
mkdir -p "$work/results"
nvidia-smi -i "$gpu" --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
if [[ -n $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo "GPU $gpu is occupied" >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
export PYTHONPATH="$root/integrated/gift:$root:$root/tuning"
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
for seq in "$@"; do
    case_name="$model-c8-b1-s$seq"
    deadline=1200
    timeout "$deadline" /opt/torchtitan-venv/bin/python3.12 -Bu "$work/search.py" "$model" "$seq" \
        --root "$root" --out "$work/results" > "$work/results/$case_name.log" 2>&1 || {
        status=$?
        printf '%s failed with exit status %s\n' "$case_name" "$status" >&2
        exit "$status"
    }
done
