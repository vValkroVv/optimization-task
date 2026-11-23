#!/usr/bin/env bash
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source /etc/profile.d/wm.sh
set -u
gpu=${1:?physical GPU index}
chunks=${2:?loss chunks}
series=${3:?series ID}
shift 3
[[ $gpu =~ ^[0-5]$ ]]
out=$PWD/runs/lm-head-mtp/artifacts/$series/chunks-$chunks
mkdir "$out"
trap 'echo "$?" > "$out/exit-code"' EXIT
echo "$$" > "$out/pid"
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
test -z "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=$uuid CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2
export TRITON_CACHE_DIR=$out/triton TORCHINDUCTOR_CACHE_DIR=$out/inductor
export PYTHONPYCACHEPREFIX=$out/pycache
python=/opt/torchtitan-venv/bin/python3.12
i=0
for arm in "$@"; do
  tag=$out/$i-$arm
  printf '%s GPU %s chunk count %s %s\n' "$(date -u +%FT%TZ)" "$gpu" "$chunks" "$arm"
  timeout 300 "$python" -Bu runs/lm-head-mtp/frontier_stack.py "$arm" "$chunks" "$tag.json" > "$tag.log" 2>&1
  ((i+=1))
done
