#!/usr/bin/env bash
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source /etc/profile.d/wm.sh
set -u
gpu=${1:?physical GPU index}
series=${2:?series}
shift 2
out=$PWD/runs/lm-head-mtp/frontier-ce/$series
mkdir -p "$out"
trap 'code=$?; echo "$code" > "$out/exit-code"' EXIT
echo "$$" > "$out/pid"
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
test -z "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=$uuid CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2
export TRITON_CACHE_DIR=$out/triton TORCHINDUCTOR_CACHE_DIR=$out/inductor
export PYTHONPYCACHEPREFIX=$out/pycache
python=/opt/torchtitan-venv/bin/python3.12
i=0
status=0
for arm in "$@"; do
  IFS=: read -r mode chunks tile warps warmup samples dump <<< "$arm"
  tag=$out/$i-$mode-$chunks
  extra=()
  if [[ ${dump:-} == yes ]]; then
    extra=(--dump "$tag.pt")
  fi
  printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$arm" "$uuid"
  FRONTIER_CE_MODE=$mode FRONTIER_CE_BLOCK=${tile:-8192} FRONTIER_CE_WARPS=${warps:-4} \
    timeout 300 "$python" -Bu runs/lm-head-mtp/frontier_ce.py current "$tag.json" \
    --sum-reduction --chunks "$chunks" --writer cublas --warmup "${warmup:-10}" \
    --samples "${samples:-12}" "${extra[@]}" > "$tag.log" 2>&1 || status=1
  ((i+=1))
done
exit "$status"
