#!/usr/bin/env bash
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /etc/profile.d/wm.sh
set -u
gpu=${1:?GPU index}
chunks=${2:?outer chunks}
series=${3:?numeric capture series}
[[ $gpu == 0 || $gpu == 1 ]]
[[ $chunks == 2 || $chunks == 8 ]]
out=$PWD/$series/chunks-$chunks
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
export PYTHONPATH=/workspace-SR008.nfs2/vkropotin/optimization-task/.worktrees/lm-head-mtp-fix/runs/lm-head-mtp
export OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$out/triton PYTHONPYCACHEPREFIX=$out/pycache
unset PYTORCH_CUDA_ALLOC_CONF
timeout 240 /opt/torchtitan-venv/bin/python3.12 -Bu panel_numerics.py "$chunks" "$out/verify.json" \
    "$out/A.pt" "65536:$out/B.pt" "32768:$out/C.pt" > "$out/verify.log" 2>&1
