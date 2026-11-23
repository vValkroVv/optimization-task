#!/usr/bin/env bash
# Source from the experiment directory and pass the assigned physical GPU.
set -Eeo pipefail
source /etc/profile.d/wm.sh
set -u
gpu=${1:?physical GPU index}
[[ $gpu == 4 || $gpu == 5 ]]
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
test -n "$CUDA_VISIBLE_DEVICES"
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
export HARNESS=/workspace-SR008.nfs2/vkropotin/optimization-task/.worktrees/lm-head-mtp-fix/runs/lm-head-mtp
export GIFT_CURRENT=/workspace-SR008.nfs2/vkropotin/lm10-c1/gift
export PYTHONPATH="$PWD:$HARNESS:$GIFT_CURRENT"
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_CACHE_DIR=$PWD/cache/inductor-$gpu TRITON_CACHE_DIR=$PWD/cache/triton-$gpu
export PYTHONPYCACHEPREFIX=$PWD/cache/pycache-$gpu
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
python=/opt/torchtitan-venv/bin/python3.12
