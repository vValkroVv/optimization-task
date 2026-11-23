#!/usr/bin/env bash
set -Eeo pipefail
cd /workspace-SR008.nfs2/vkropotin/optimization-task/.worktrees/lm-head-mtp-fix
source /etc/profile.d/wm.sh
set -u

gpu=$1
chunks=$2
series=$3
shift 3

[[ $gpu == 0 ]]
[[ $chunks == 8 || $chunks == 16 ]]
[[ $series =~ ^[A-Za-z0-9._-]+$ ]]
[[ $# -ge 2 ]]
out=$PWD/artifacts/frontier-workspace/$series/chunks-$chunks
mkdir -p "$out"
trap 'echo "$?" > "$out/exit-code"' EXIT

uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
check_gpu_idle() {
  local apps
  apps=$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)
  test -z "$apps"
}
check_gpu_idle

printf 'source_revision=%s\n' "$(git rev-parse HEAD)" > "$out/manifest.txt"
printf 'physical_gpu=%s\n' "$gpu" >> "$out/manifest.txt"
printf 'gpu_uuid=%s\n' "$uuid" >> "$out/manifest.txt"
printf 'chunks=%s\n' "$chunks" >> "$out/manifest.txt"
printf 'torch_python=%s\n' /opt/torchtitan-venv/bin/python3.12 >> "$out/manifest.txt"
nvidia-smi -i "$uuid" --query-gpu=name,driver_version,memory.total --format=csv,noheader >> "$out/manifest.txt"

export CUDA_VISIBLE_DEVICES=$uuid CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2
export TRITON_CACHE_DIR=$out/triton TORCHINDUCTOR_CACHE_DIR=$out/inductor
export PYTHONPYCACHEPREFIX=$out/pycache
python=/opt/torchtitan-venv/bin/python3.12
i=0
for arm in "$@"; do
  check_gpu_idle
  tag=$out/$i-$arm
  printf '%s GPU %s chunk count %s %s\n' "$(date -u +%FT%TZ)" "$gpu" "$chunks" "$arm" | tee -a "$out/manifest.txt"
  case "$arm" in
    control)
      timeout 300 "$python" -Bu runs/lm-head-mtp/frontier_stack.py stack "$chunks" "$tag.json" > "$tag.log" 2>&1
      ;;
    workspace)
      timeout 300 "$python" -Bu runs/lm-head-mtp/frontier_workspace.py "$chunks" "$tag.json" > "$tag.log" 2>&1
      ;;
    *)
      echo "unknown arm: $arm" >&2
      exit 2
      ;;
  esac
  ((i+=1))
done
