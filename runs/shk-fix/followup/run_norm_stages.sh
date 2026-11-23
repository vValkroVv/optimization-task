#!/usr/bin/env bash
set -eo pipefail

checkout=$1
historical=$2
output=$3
gpu=$4
shift 4
trap 'printf "%s\n" "$?" > "$output/exit"' EXIT
source /etc/profile.d/wm.sh
set -u

replay="$historical/.worktrees/shared-sm90-kernels-fix/runs/shared-sm90-kernels"
fixture="$historical/runs/kernel-fusion/artifacts/raw/followup-fixtures-001"
tool="$checkout/runs/shk-fix/followup/norm_stage_component.py"
cd "$checkout"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$checkout/gift:$checkout/gift/contrib/flash-linear-attention:$replay${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR="$output/inductor"
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 TORCH_LOGS=recompiles,graph_breaks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum "$tool" "$replay"/{residual_replay_probe,norm,measure}.py \
    "$checkout/gift/torchtitan/models/common/norms/cute_gated_norm/forward.py" > "$output/source.sha256"
index=0
for variant in "$@"; do
    index=$((index + 1))
    test -z "$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)"
    base="$output/$index-stages-$variant"
    stages=(--stages "$variant")
    if [[ "$variant" == late ]]; then
        stages=(--stages 0 --late-residual)
    fi
    printf '%s GPU%s %s\n' "$(date -u +%FT%TZ)" "$gpu" "$base"
    timeout --signal=TERM --kill-after=30 300 /opt/torchtitan-venv/bin/python3.12 -B -u "$tool" \
        --fixture "$fixture/component-inputs/cute-rank0.pt" \
        --resolved-config "$fixture/native/config/resolved_config.yaml" \
        "${stages[@]}" --output "$base.json" --profile > "$base.log" 2>&1
done
