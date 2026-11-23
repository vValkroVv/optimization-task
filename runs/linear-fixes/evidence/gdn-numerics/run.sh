#!/usr/bin/env bash
set -eo pipefail
out="$(cd -- "$(dirname -- "$0")" && pwd)"
task=/workspace-SR008.nfs2/vkropotin/optimization-task
source /etc/profile.d/wm.sh
cd "$task/artifacts/linear-fixes/source"
export FLA_DISABLE_BACKEND_DISPATCH=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PWD/gift/contrib/flash-linear-attention:$PWD/gift"
export TORCHINDUCTOR_CACHE_DIR="$out/inductor" TRITON_CACHE_DIR="$out/triton"
export TORCHINDUCTOR_COMPILE_THREADS=2
trap 'printf "%s\n" "$?" > "$out/exit-code"' EXIT
python=/opt/torchtitan-venv/bin/python3.12
timeout 900s "$python" -Bu "$out/gdn_numerics.py" \
  --fixture "$task/runs/kernel-fusion/artifacts/raw/followup-fixtures-001/component-inputs/gdn-qkv-rank0.pt" \
  --helpers "$task/runs/fusion-followup/gdn_component.py" \
  --config /workspace-SR008.nfs2/vkropotin/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2/native/config/resolved_config.yaml \
  --output "$out/result.json"
timeout 600s "$python" -B -m pytest -q -p no:cacheprovider \
  gift/tests/unit_tests/test_gdn_fla_backward.py --junitxml "$out/direct-scan.xml"
