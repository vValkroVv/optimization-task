#!/bin/bash
set -eu
source /etc/profile.d/wm.sh
cd /workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125
export PYTHONPATH="$PWD:$PWD/integrated/gift:$PWD/src/liger/src:$PWD/src/fla"
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/ce-stock-inductor" TRITON_CACHE_DIR="$PWD/cache/ce-stock-triton"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p output/ce-stock
run_lane() {
    local gpu=$1 model=$2 chunks=$3
    shift 3
    local arm stem
    for arm in "$@"; do
        stem="output/ce-stock/$model-c$chunks-$arm"
        echo "start $(date -Is) $stem"
        if CUDA_VISIBLE_DEVICES=$gpu /opt/torchtitan-venv/bin/python3.12 -Bu ce_stock_chunks.py --case "$model" --chunks "$chunks" --variant "$arm" --fixture "inputs/$model.pt" --exact-so native-ce/centered-folded-v1.so --exact-sha256 17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8 --output "$stem.json" > "$stem.log" 2>&1; then
            echo "ok $(date -Is) $stem"
        else
            echo "failed $(date -Is) $stem"
        fi
    done
}
run_lane 0 100b 2 liger cute fla baseline cuda13 & a=$!
run_lane 1 100b 8 cute fla baseline cuda13 liger & b=$!
run_lane 2 700b 8 fla baseline cuda13 liger cute & c=$!
wait "$a"
wait "$b"
wait "$c"
echo "finished $(date -Is)"
