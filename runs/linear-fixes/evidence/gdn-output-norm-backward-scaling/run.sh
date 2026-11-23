#!/usr/bin/env bash
set -eo pipefail
out="$(cd -- "$(dirname -- "$0")" && pwd)"
shape="${1:?Pass a benchmark shape name}"
device="${2:?Pass the checked H100 GPU UUID}"
case_dir="$out/$shape"
task=/workspace-SR008.nfs2/vkropotin/optimization-task
source /etc/profile.d/wm.sh
source_dir="$task/artifacts/linear-fixes/results/gdn-output-norm-backward-001/source"
cd "$source_dir"

[[ -d "$case_dir" ]]
nvidia-smi --id="$device" --query-gpu=uuid,name,memory.used --format=csv,noheader
test -z "$(nvidia-smi --id="$device" --query-compute-apps=pid --format=csv,noheader)"
export FLA_DISABLE_BACKEND_DISPATCH=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$device"
export PYTHONPATH="$PWD/gift/contrib/flash-linear-attention:$PWD/gift"
export TORCHINDUCTOR_CACHE_DIR="$case_dir/inductor" TRITON_CACHE_DIR="$out/triton"
export TORCHINDUCTOR_COMPILE_THREADS=2
trap 'printf "%s\n" "$?" > "$case_dir/exit-code"' EXIT
timeout 900s /opt/torchtitan-venv/bin/python3.12 -Bu \
  runs/linear-fixes/gdn_output_norm_backward_scaling.py \
  --shape "$shape" \
  --fixture "$task/runs/kernel-fusion/artifacts/raw/followup-fixtures-001/component-inputs/gdn-qkv-rank0.pt" \
  --helpers "$task/runs/fusion-followup/gdn_component.py" \
  --resolved-config /workspace-SR008.nfs2/vkropotin/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2/native/config/resolved_config.yaml \
  --output "$case_dir/metrics" --samples 24
