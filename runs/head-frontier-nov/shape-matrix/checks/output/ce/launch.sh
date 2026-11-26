#!/bin/bash
source /etc/profile.d/wm.sh || exit 1
cd /workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125 || exit 1
export PYTHONPATH="$PWD:$PWD/integrated/gift:$PWD/src/liger/src:$PWD/src/fla:$PWD/src/apple"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=2
python=/opt/torchtitan-venv/bin/python3.12
run_cell() {
    local gpu=$1 variant=$2 model=$3 chunks=$4
    local output="output/ce/$model-c$chunks-$variant"
    echo "start $(date -Is) gpu=$gpu $output"
    CUDA_VISIBLE_DEVICES=$gpu "$python" -u ce_bench.py --case "$model" --chunks "$chunks" --part "$((chunks - 1))" \
      --variant "$variant" --fixture "inputs/$model.pt" --exact-so native-ce/centered-folded-v1.so \
      --exact-sha256 17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8 --graph --output "$output.json" \
      > "$output.log" 2>&1
    local status=$?
    echo "finish $(date -Is) gpu=$gpu $output exit=$status"
}
run_lane() {
    local gpu=$1
    shift
    local variant
    for variant in "$@"; do
        run_cell "$gpu" "$variant" 100b 2
        run_cell "$gpu" "$variant" 100b 8
        run_cell "$gpu" "$variant" 700b 8
    done
}
run_lane 0 baseline &
run_lane 1 folded &
run_lane 2 cuda13 &
run_lane 3 liger liger-fp32 &
run_lane 4 cute cute-fp32 &
run_lane 5 fla fla-centered &
wait
echo "all CE lanes returned $(date -Is)"
