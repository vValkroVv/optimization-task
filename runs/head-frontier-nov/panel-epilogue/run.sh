#!/usr/bin/env bash
set -eo pipefail
source /etc/profile.d/wm.sh
set -u
cd /workspace-SR008.nfs2/vkropotin/head-frontier-nov/panel-epilogue
gpu=$1
chunks=$2
shift 2
[[ $gpu == 6 || $gpu == 7 ]]
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
export PYTHONPATH="/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/panels:/workspace-SR008.nfs2/vkropotin/optimization-task/.worktrees/lm-head-mtp-fix/runs/lm-head-mtp"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
export TRITON_CACHE_DIR=$PWD/cache-$gpu/triton TORCHINDUCTOR_CACHE_DIR=$PWD/cache-$gpu/inductor
unset PYTORCH_CUDA_ALLOC_CONF
python=/opt/torchtitan-venv/bin/python3.12
if [[ $# == 0 ]]; then
  exec "$python" -Bu mixed.py "$chunks"
fi
quack=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/quack-lmhead/runtime
export PYTHONPATH="$quack/patched/quack-bb0e6b2-colvec-v:$PYTHONPATH"
export QUACK_CACHE_DIR=$PWD/cache-$gpu/quack
python=$quack/venv/bin/python
for a in "$@"; do
  "$python" -Bu trial.py "$chunks" "$a" >"c$chunks-$a.log" 2>&1
done
