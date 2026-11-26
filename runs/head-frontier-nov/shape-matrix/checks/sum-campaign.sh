#!/bin/bash
set -e
source /etc/profile.d/wm.sh
cd /workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125
for i in $(seq 1 100); do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then break; fi
  sleep 3
done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
export PYTHONPATH="$PWD:$PWD/integrated/gift:$PWD/src/liger/src:$PWD/src/fla:$PWD/src/apple"
export PYTORCH_ALLOC_CONF="expandable_segments:True" CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2
p=/opt/torchtitan-venv/bin/python3.12
ce() {
 CUDA_VISIBLE_DEVICES=$1 "$p" -Bu ce_extra.py --case "$2" --chunks "$3" --part "$(($3-1))" --variant liger-early --fixture "inputs/$2.pt" --graph --output "output/ce/$2-c$3-liger-early.json" >"output/ce/$2-c$3-liger-early.log" 2>&1
}
ce 0 100b 2 &
ce 1 100b 8 &
ce 2 700b 8 &
CUDA_VISIBLE_DEVICES=7 "$p" -m pytest -q unit > output/unit-tests.log 2>&1 &
wait
"$p" -Bu head_matrix_extra.py --series head-sum --oracle --arms baseline cuda13 apple-direct standalone-liger-early folded folded-lt standalone-fla standalone-liger standalone-cute standalone-liger-fp32 standalone-cute-fp32 fla-linear-lt apple liger-linear fla-linear liger-linear-default fla-linear-default cuda13-tuned --wait 1200
"$p" -Bu head_matrix_extra.py --series head-repeat --oracle --arms standalone-fla cuda13 --samples 20 --wait 1200
