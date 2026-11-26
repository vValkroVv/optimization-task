#!/bin/bash
set -e
cd /workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125
for i in $(seq 1 50); do
 if test -f output/head-repeat/700b-c8-cuda13/exit-code; then break; fi
 sleep 45
done
test "$(cat output/head-repeat/700b-c8-cuda13/exit-code)" = 0
source /etc/profile.d/wm.sh
p=/opt/torchtitan-venv/bin/python3.12
"$p" -Bu head_matrix_extra.py --series head-finalists --oracle --arms cuda13-tuned folded-lt cuda13 --samples 20 --wait 300
export PYTHONPATH="$PWD/integrated/gift:$PWD"
export GIFT_EXACT_FOLDED_CE_LIBRARY="$PWD/native-ce/centered-folded-v1.so" GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256=17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/head-inductor" TRITON_CACHE_DIR="$PWD/cache/head-triton" PYTORCH_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1
"$p" -Bu -m torch.distributed.run --standalone --nproc-per-node=8 multi_mtp_smoke.py > output/multi-mtp-smoke.log 2>&1
echo "Smoke passed"
