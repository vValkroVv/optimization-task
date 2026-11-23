#!/usr/bin/env bash
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /etc/profile.d/wm.sh
set -u
gpu=${1:?GPU index}
chunks=${2:?outer chunks}
series=${3:?new series}
shift 3
[[ $gpu == 0 || $gpu == 1 ]]
[[ $chunks == 2 || $chunks == 8 ]]
out=$PWD/$series/chunks-$chunks
mkdir -p "$PWD/$series"
mkdir "$out"
trap 'echo "$?" > "$out/exit-code"' EXIT
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
export PYTHONPATH=/workspace-SR008.nfs2/vkropotin/optimization-task/.worktrees/lm-head-mtp-fix/runs/lm-head-mtp
export PANEL_GIFT_SOURCE=/workspace-SR008.nfs2/vkropotin/optimization-task/.worktrees/lm-head-mtp-fix
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_CACHE_DIR=$out/inductor TRITON_CACHE_DIR=$out/triton PYTHONPYCACHEPREFIX=$out/pycache
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTORCH_CUDA_ALLOC_CONF
python=/opt/torchtitan-venv/bin/python3.12
for setting in "$@"; do
    read -r -a arguments <<< "$setting"
    arm=${arguments[0]}
    label=${arguments[1]}
    printf 'GPU %s chunks %s %s %s\n' "$gpu" "$chunks" "$arm" "$label"
    timeout 240 "$python" -Bu panel_replay.py "$arm" "$chunks" "$out/$label.json" "${arguments[@]:2}" > "$out/$label.log" 2>&1
done
