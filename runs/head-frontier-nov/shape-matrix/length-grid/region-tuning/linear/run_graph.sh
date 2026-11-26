#!/usr/bin/env bash
set -eo pipefail
lane=$1
cell=$2
case "$lane" in
    3|4|5) ;;
    *) exit 2 ;;
esac
case "$cell" in
    700b-s262144-projection|100b-s262144-projection) ;;
    *) exit 2 ;;
esac
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
task=$(cd "$root/../../.." && pwd)
cd "$root"
source /etc/profile.d/wm.sh
nvidia-smi -i "$lane" --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,driver_version \
    --format=csv,noheader,nounits > "$here/preflight-graph-$lane.csv"
nvidia-smi -i "$lane" --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits > "$here/preflight-graph-$lane-apps.csv"
gpu_uuid=$(awk -F ', *' '$4 < 256 && $6 == 0 {print $2}' "$here/preflight-graph-$lane.csv")
if [[ -z "$gpu_uuid" || -s "$here/preflight-graph-$lane-apps.csv" ]]; then
    printf 'GPU %s preflight failed\n' "$lane" >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export PYTHONPATH="$root:$root/integrated/gift:$task/.worktrees/final-all-fix/gift:$task/../lm10-c1/gift"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
timeout --signal=TERM --kill-after=30s 4m /opt/torchtitan-venv/bin/python3.12 -Bu \
    "$here/recover_graph.py" "$here/results/$cell/confirm.json"
