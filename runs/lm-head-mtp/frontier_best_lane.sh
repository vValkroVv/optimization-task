#!/usr/bin/env bash
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source /etc/profile.d/wm.sh
set -u
gpu=${1:?physical GPU}
series=${2:?series}
chunks=${3:?loss chunks}
shift 3
[[ $gpu =~ ^[0-5]$ ]]
out=$PWD/runs/lm-head-mtp/artifacts/$series/chunks-$chunks
mkdir "$out"
trap 'echo "$?" > "$out/exit-code"' EXIT
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_CACHE_DIR=$out/inductor TRITON_CACHE_DIR=$out/triton PYTHONPYCACHEPREFIX=$out/pycache
unset PYTORCH_CUDA_ALLOC_CONF
python=/opt/torchtitan-venv/bin/python3.12
i=0
for arm in "$@"; do
  setting=False
  [[ $arm != memory ]] || setting=True
  echo "GPU $gpu chunks $chunks $arm expandable_segments:$setting"
  PYTORCH_ALLOC_CONF=expandable_segments:$setting timeout 250 "$python" -Bu runs/lm-head-mtp/frontier_best.py "$arm" "$chunks" "$out/$i-$arm.json" > "$out/$i-$arm.log" 2>&1
  ((i+=1))
done
