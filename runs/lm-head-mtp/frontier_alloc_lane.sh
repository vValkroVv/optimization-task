#!/usr/bin/env bash
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source /etc/profile.d/wm.sh
set -u
gpu=${1:?physical GPU}
chunks=${2:?loss chunk count}
shift 2
[[ $gpu =~ ^[0-5]$ ]]
out=$PWD/runs/lm-head-mtp/artifacts/frontier-allocator-final-002/chunks-$chunks
mkdir "$out"
trap 'echo "$?" > "$out/exit-code"' EXIT
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_CACHE_DIR=$out/inductor TRITON_CACHE_DIR=$out/triton PYTHONPYCACHEPREFIX=$out/pycache
unset PYTORCH_CUDA_ALLOC_CONF
python=/opt/torchtitan-venv/bin/python3.12
i=0
for setting in "$@"; do
  echo "GPU $gpu chunks $chunks PYTORCH_ALLOC_CONF=expandable_segments:$setting"
  PYTORCH_ALLOC_CONF=expandable_segments:$setting timeout 300 "$python" -Bu runs/lm-head-mtp/frontier_stack.py stack "$chunks" "$out/$i-$setting.json" > "$out/$i-$setting.log" 2>&1
  ((i+=1))
done
