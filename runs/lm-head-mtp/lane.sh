#!/usr/bin/env bash
# Each lane owns one GPU and launches fresh processes for matched arms.
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source /etc/profile.d/wm.sh
set -u
gpu=${1:?physical GPU index}
series=${2:?fresh series ID}
shift 2
out=$PWD/runs/lm-head-mtp/artifacts/$series
mkdir "$out"
trap 'code=$?; echo "$code" > "$out/exit-code"' EXIT
echo "$$" > "$out/pid"
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
test -z "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=$uuid CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2
export TRITON_CACHE_DIR=$out/triton TORCHINDUCTOR_CACHE_DIR=$out/inductor
export PYTHONPYCACHEPREFIX=$out/pycache
repo=$(realpath ../..)
quack=$repo/runs/quack-lmhead/runtime
export PYTHONPATH=$quack/patched/quack-bb0e6b2-colvec-v${PYTHONPATH:+:$PYTHONPATH}
export QUACK_CACHE_DIR=$out/quack
python=/opt/torchtitan-venv/bin/python3.12
i=0
status=0
for arm in "$@"; do
  IFS=: read -r variant chunks writer warmup samples storage <<< "$arm"
  chosen=$python
  if [[ $writer == tma ]]; then
    chosen=$quack/venv/bin/python
  fi
  tag=$out/${i}-${variant}-${chunks}-${writer}-${storage:-native}
  printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$arm" "$chosen"
  timeout 480 "$chosen" -Bu runs/lm-head-mtp/replay.py "$variant" "$tag.json" \
    --chunks "$chunks" --writer "$writer" --storage "${storage:-native}" --warmup "${warmup:-10}" --samples "${samples:-10}" \
    > "$tag.log" 2>&1 || status=1
  ((i+=1))
done
exit "$status"
