#!/usr/bin/env bash
set -eo pipefail
out="$(cd -- "$(dirname -- "$0")" && pwd)"
task=/workspace-SR008.nfs2/vkropotin/optimization-task
source /etc/profile.d/wm.sh
cd "$out/source"
export FLA_DISABLE_BACKEND_DISPATCH=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${1:?Pass the checked H100 GPU UUID}"
export PYTHONPATH="$PWD/gift/contrib/flash-linear-attention:$PWD/gift"
export TORCHINDUCTOR_CACHE_DIR="$out/inductor" TRITON_CACHE_DIR="$out/triton"
export TORCHINDUCTOR_COMPILE_THREADS=2
trap 'printf "%s\n" "$?" > "$out/exit-code"' EXIT
out_name="${2:-measured}"
shift
if (( $# )); then shift; fi
timeout 1200s /opt/torchtitan-venv/bin/python3.12 -Bu \
  runs/linear-fixes/gdn_output_norm_backward_bench.py \
  --fixture "$task/runs/kernel-fusion/artifacts/raw/followup-fixtures-001/component-inputs/gdn-qkv-rank0.pt" \
  --helpers "$task/runs/fusion-followup/gdn_component.py" \
  --resolved-config /workspace-SR008.nfs2/vkropotin/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2/native/config/resolved_config.yaml \
  --output "$out/$out_name" "$@"
