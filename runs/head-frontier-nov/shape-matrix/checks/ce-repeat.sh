#!/bin/bash
set -e
source /etc/profile.d/wm.sh
cd /workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
export PYTHONPATH="$PWD:$PWD/integrated/gift:$PWD/src/liger/src:$PWD/src/fla:$PWD/src/apple"
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2
mkdir -p output/ce-repeat
run() {
 CUDA_VISIBLE_DEVICES=$1 /opt/torchtitan-venv/bin/python3.12 -Bu ce_repeat.py --case "$2" --chunks "$3" --fixture "inputs/$2.pt" --exact-so native-ce/centered-folded-v1.so --exact-sha256 17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8 --output "output/ce-repeat/$2-c$3.json" > "output/ce-repeat/$2-c$3.log" 2>&1
}
run 0 100b 2 & a=$!
run 1 100b 8 & b=$!
run 2 700b 8 & c=$!
wait "$a"
wait "$b"
wait "$c"
