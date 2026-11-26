#!/usr/bin/env bash
set -eo pipefail
root=$1
gpu=$2
shift 2
case "$gpu" in
    0|1|2|3) ;;
    *) echo "GPU must be 0, 1, 2, or 3" >&2; exit 2 ;;
esac
source /etc/profile.d/wm.sh
set -u
cd "$root"
work="$root/length-grid/final-grid/dw"
mkdir -p "$work/results"
nvidia-smi -i "$gpu" --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu --format=csv
if [[ -n $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo "GPU $gpu is occupied" >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
export PYTHONPATH="$root/integrated/gift:$root:$root/tuning"
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
for cell in "$@"; do
    IFS=: read -r model chunks seq <<< "$cell"
    case_name="$model-c$chunks-b1-s$seq"
    status=0
    timeout 1200 /opt/torchtitan-venv/bin/python3.12 -Bu "$work/calibrate.py" "$model" "$chunks" "$seq" \
        --root "$root" --out "$work/results" > "$work/results/$case_name.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$work/results/$case_name.exit"
    printf '%s exit=%s\n' "$case_name" "$status"
done
