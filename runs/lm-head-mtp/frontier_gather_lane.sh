#!/usr/bin/env bash
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source /etc/profile.d/wm.sh
set -u
series=${1:?series ID}
shift
[[ $series =~ ^frontier-gather-[a-z0-9-]+$ ]]
out=$PWD/runs/lm-head-mtp/artifacts/$series
mkdir "$out"
trap 'echo "$?" > "$out/exit-code"' EXIT
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 4 --query-gpu=uuid --format=csv,noheader)
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_CACHE_DIR=$out/inductor TRITON_CACHE_DIR=$out/triton PYTHONPYCACHEPREFIX=$out/pycache
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
python=/opt/torchtitan-venv/bin/python3.12
i=0
for case in "$@"; do
  chunks=${case%:*}
  arm=${case#*:}
  tag=$out/$i-n$chunks-$arm
  printf '%s GPU 4 chunk count %s %s\n' "$(date -u +%FT%TZ)" "$chunks" "$arm"
  timeout 300 "$python" -Bu runs/lm-head-mtp/frontier_gather.py "$arm" "$chunks" "$tag.json" > "$tag.log" 2>&1
  ((i+=1))
done
