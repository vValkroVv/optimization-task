#!/usr/bin/env bash
# Execute after the eight-rank head campaign releases GPUs 5, 6 and 7.
set -eo pipefail
source /etc/profile.d/wm.sh
cd /workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD:$PWD/integrated/gift"
test -z "$(nvidia-smi -i 5,6,7 --query-compute-apps=pid --format=csv,noheader)"
output=tuning/proj/recheck
mkdir -p "$output"
nvidia-smi -i 5,6,7 --query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version --format=csv > "$output/preflight.csv"
sha256sum tuning/proj/{lt_ops.py,lt_bridge.cpp,lt_bridge.so,gemm_bench.py,gemm_confirm.py} > "$output/source.sha256"

measure() {
    local gpu=$1 case=$2
    shift 2
    CUDA_VISIBLE_DEVICES=$gpu /opt/torchtitan-venv/bin/python3.12 -Bu tuning/proj/gemm_confirm.py \
      --screen "tuning/proj/$case-p0.json" --parity-receipt "tuning/proj/$case-confirm.json" \
      --output "$output/$case-confirm.json" "$@" > "$output/$case.log" 2>&1
}

measure 5 700b-c8 --profile & a=$!
measure 6 100b-c8 & b=$!
measure 7 100b-c2 & c=$!
status=0
wait "$a" || status=$?
wait "$b" || status=$?
wait "$c" || status=$?
exit "$status"
