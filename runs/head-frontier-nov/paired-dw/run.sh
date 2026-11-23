#!/usr/bin/env bash
set -eo pipefail
source /etc/profile.d/wm.sh
set -u
cd /workspace-SR008.nfs2/vkropotin/head-frontier-nov/paired-dw
gpu=$1
chunks=$2
shift 2
[[ $gpu == 6 || $gpu == 7 ]]
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
export REPLAY_DIR=/workspace-SR008.nfs2/vkropotin/optimization-task/.worktrees/lm-head-mtp-fix/runs/lm-head-mtp
export PYTHONPATH="$REPLAY_DIR:${PYTHONPATH:-}"
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$PWD/cache-$gpu/triton TORCHINDUCTOR_CACHE_DIR=$PWD/cache-$gpu/inductor
python=/opt/torchtitan-venv/bin/python3.12
if [[ ${1:-} == tune ]]; then
  exec "$python" -Bu screen.py "$chunks" "tune-c$chunks.json" --first 6
fi
if [[ $# == 0 ]]; then
  exec "$python" -Bu screen.py "$chunks" "screen-c$chunks.json"
fi
for arm in "$@"; do
  "$python" -Bu integrated.py "$chunks" "$arm" > "c$chunks-$arm.log" 2>&1
done
