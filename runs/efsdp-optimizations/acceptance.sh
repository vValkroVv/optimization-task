#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# != 4 || ! $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    echo "Usage: bash acceptance.sh RUN_PREFIX ORIGINAL_A_SOURCE ORIGINAL_A_REFERENCE DIRECT_DW_BUILD" >&2
    exit 2
fi
runner_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run_prefix=$1
control_source=$2
reference=$3
direct_dw_build=$4

source /etc/profile.d/wm.sh
python=/opt/torchtitan-venv/bin/python3.12
runner="$runner_dir/run.py"
common=(--reference "$reference")
optimized=(
    --optimization bf16-cache
    --optimization fp8-cache
    --optimization direct-dw
    --optimization tail-retention
    --direct-dw-build "$direct_dw_build"
)

for ep in 8 4; do
    "$python" -B "$runner" plan "$run_prefix-ep$ep-a" \
        --ep "$ep" --original-a-source "$control_source" "${common[@]}" >/dev/null
    "$python" -B "$runner" plan "$run_prefix-ep$ep-expert" \
        --ep "$ep" "${common[@]}" "${optimized[@]}" >/dev/null
done

check_idle() {
    local devices apps
    devices=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    if [[ $(wc -l <<<"$devices") -ne 8 ]] || grep -vq 'NVIDIA H100' <<<"$devices"; then
        echo "Expected the verified single-node eight-H100 allocation" >&2
        exit 1
    fi
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ -n $apps ]]; then
        echo "Existing GPU work detected. Serial acceptance will not start a competing arm." >&2
        exit 1
    fi
}

for ep in 8 4; do
    check_idle
    "$python" -B "$runner" run "$run_prefix-ep$ep-a" \
        --ep "$ep" --original-a-source "$control_source" "${common[@]}"
    check_idle
    "$python" -B "$runner" run "$run_prefix-ep$ep-expert" \
        --ep "$ep" "${common[@]}" "${optimized[@]}"
done
