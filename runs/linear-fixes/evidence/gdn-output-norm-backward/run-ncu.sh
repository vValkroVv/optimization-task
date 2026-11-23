#!/usr/bin/env bash
set -eo pipefail
out="$(cd -- "$(dirname -- "$0")" && pwd)"
task=/workspace-SR008.nfs2/vkropotin/optimization-task
source /etc/profile.d/wm.sh
cd "$out/source"
export FLA_DISABLE_BACKEND_DISPATCH=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${1:?Pass the checked H100 GPU UUID}"
export PYTHONPATH="$PWD/runs/linear-fixes:$PWD/gift/contrib/flash-linear-attention:$PWD/gift"
export TORCHINDUCTOR_CACHE_DIR="$out/inductor" TRITON_CACHE_DIR="$out/triton"
export TORCHINDUCTOR_COMPILE_THREADS=2
trap 'printf "%s\n" "$?" > "$out/ncu-exit-code"' EXIT

metrics=launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum
metrics+=,dram__bytes.sum,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum
for variant in candidate baseline; do
  kernel=layer_norm_gated_bwd_kernel
  if [[ "$variant" == candidate ]]; then kernel=_gdn_output_norm_backward_kernel; fi
  timeout 180s ncu --kernel-name-base function --kernel-name "regex:^${kernel}$" \
    --launch-skip 2 --launch-count 1 --clock-control none --cache-control none \
    --csv --page raw --metrics "$metrics" --log-file "$out/ncu-${variant}.csv" \
    /opt/torchtitan-venv/bin/python3.12 -B -c '
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch
from gdn_output_norm_backward_bench import (
    NORM_ARGS,
    load_inputs,
    make_norm_backward,
    prepare_projection_dgrad,
)

variant, fixture_path, helper_path = sys.argv[1:]
spec = spec_from_file_location("gdn_component_helpers", helper_path)
helpers = module_from_spec(spec)
spec.loader.exec_module(helpers)
torch.manual_seed(2048)
fixture, _ = helpers.fixture_on_device(Path(fixture_path))
inputs, projection, cotangent = load_inputs(fixture, 2048)
dgrad, _, _ = prepare_projection_dgrad(inputs, projection, cotangent, helpers)
backward = make_norm_backward(
    inputs, dgrad, NORM_ARGS, direct=variant == "candidate", compiled=True
)
for _ in range(4):
    backward()
torch.cuda.synchronize()
' "$variant" \
    "$task/runs/kernel-fusion/artifacts/raw/followup-fixtures-001/component-inputs/gdn-qkv-rank0.pt" \
    "$task/runs/fusion-followup/gdn_component.py"
done
