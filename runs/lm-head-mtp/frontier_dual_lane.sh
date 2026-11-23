#!/usr/bin/env bash
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source /etc/profile.d/wm.sh
set -u
series=${1:?series}
shift
[[ $series =~ ^frontier-dual-[a-z0-9-]+$ ]]
out=$PWD/runs/lm-head-mtp/artifacts/$series
test -d "$out"
trap 'echo "$?" > "$out/exit-code"' EXIT
mapfile -t devices < <(nvidia-smi -i 1,2 --query-gpu=uuid --format=csv,noheader)
[[ ${#devices[@]} == 2 ]]
[[ ${devices[0]} == GPU-88f1d954-f0f0-bae5-4051-7162e07d726f ]]
[[ ${devices[1]} == GPU-7a35d02e-1b06-4a5b-7f2c-76b40a15e03f ]]
export CUDA_VISIBLE_DEVICES=${devices[0]},${devices[1]}
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_CACHE_DIR=$out/inductor TRITON_CACHE_DIR=$out/triton PYTHONPYCACHEPREFIX=$out/pycache
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
python=/opt/torchtitan-venv/bin/python3.12
git rev-parse HEAD
nvidia-smi topo -m
for case in "$@"; do
    tag=$out/$case
    arm=${case#*-}
    [[ $arm == stack || $arm == alias ]]
    active=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
    if [[ -n $active ]]; then
        printf 'Selected GPUs have active PIDs: %s\n' "$active" >&2
        exit 1
    fi
    printf '%s GPUs 1/2 chunks 8 %s\n' "$(date -u +%FT%TZ)" "$case"
    timeout 300 "$python" -Bu -m torch.distributed.run --standalone --nproc-per-node=2 \
        runs/lm-head-mtp/frontier_dual.py "$arm" "$tag" > "$tag.log" 2>&1
done
