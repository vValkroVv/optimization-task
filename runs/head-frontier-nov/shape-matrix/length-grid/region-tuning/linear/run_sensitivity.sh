#!/usr/bin/env bash
set -eo pipefail
lane=$1
shift
case "$lane" in
    6|7) ;;
    *) exit 2 ;;
esac
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
task=$(cd "$root/../../.." && pwd)
cd "$root"
source /etc/profile.d/wm.sh
mkdir -p "$here/cache/gpu-$lane/"{inductor,triton,tmp,xdg} "$here/sensitivity"
nvidia-smi -i "$lane" --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,driver_version \
    --format=csv,noheader,nounits > "$here/sensitivity/preflight-$lane.csv"
nvidia-smi -i "$lane" --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits > "$here/sensitivity/preflight-$lane-apps.csv"
gpu_uuid=$(awk -F ', *' '$4 < 256 && $6 == 0 {print $2}' "$here/sensitivity/preflight-$lane.csv")
if [[ -z "$gpu_uuid" || -s "$here/sensitivity/preflight-$lane-apps.csv" ]]; then
    printf 'GPU %s preflight failed\n' "$lane" >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export PYTHONPATH="$root:$root/integrated/gift:$task/.worktrees/final-all-fix/gift:$task/../lm10-c1/gift"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CUDAGRAPHS=0
export CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_CACHE_DIR="$here/cache/gpu-$lane/inductor"
export TRITON_CACHE_DIR="$here/cache/gpu-$lane/triton"
export TMPDIR="$here/cache/gpu-$lane/tmp"
export XDG_CACHE_HOME="$here/cache/gpu-$lane/xdg"
status=0
for cell in "$@"; do
    case "$cell" in
        700b-s32768-projection|100b-s65536-dhidden|700b-s65536-projection) ;;
        *) exit 2 ;;
    esac
    code=0
    timeout --signal=TERM --kill-after=30s 4m /opt/torchtitan-venv/bin/python3.12 -Bu \
        "$here/confirm_sensitivity.py" "$here/results/$cell/confirm.json" "$here/sensitivity/$cell.json" \
        > "$here/sensitivity/$cell.log" 2>&1 || code=$?
    printf '%s\n' "$code" > "$here/sensitivity/$cell.exit"
    if [[ "$code" != 0 ]]; then status=1; fi
done
exit "$status"
