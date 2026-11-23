#!/usr/bin/env bash
set -eo pipefail
source /etc/profile.d/wm.sh
set -u
cd /workspace-SR008.nfs2/vkropotin/head-frontier-nov/multi-panel
gpu=$1
chunks=$2
shift 2
[[ $gpu == 6 || $gpu == 7 ]]
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
export PYTHONPATH="/workspace-SR008.nfs2/vkropotin/optimization-task/.worktrees/lm-head-mtp-fix/runs/lm-head-mtp"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
export TRITON_CACHE_DIR=$PWD/cache-$gpu/triton TORCHINDUCTOR_CACHE_DIR=$PWD/cache-$gpu/inductor
unset PYTORCH_CUDA_ALLOC_CONF
python=/opt/torchtitan-venv/bin/python3.12
for arm in "$@"; do
  options=()
  case $arm in
    *-alias) options+=(--gather-alias);;
  esac
  case $arm in
    A*) mode=control;;
    B*) mode=panel; options+=(--split 32768);;
    D*) mode=panel; options+=(--split 16384);;
    oracle)
      old=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/panels/numerics-001/chunks-$chunks
      "$python" -Bu multi_numerics.py "$chunks" "c$chunks-oracle.json" "$old/A.pt" "32768:$PWD/c$chunks-B0.pt" "16384:$PWD/c$chunks-D0.pt" >"c$chunks-oracle.log" 2>&1
      continue;;
    *) exit 2;;
  esac
  case $arm in
    B0|D0) options+=(--dump "$PWD/c$chunks-$arm.pt");;
  esac
  "$python" -Bu multi_replay.py "$mode" "$chunks" "$PWD/c$chunks-$arm.json" "${options[@]}" >"c$chunks-$arm.log" 2>&1
done
