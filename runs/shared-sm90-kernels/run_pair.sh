#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 5 ]]; then
    echo "Usage: bash $0 GPU CASE FIXTURE RESOLVED_CONFIG OUTPUT_DIRECTORY" >&2
    exit 2
fi
gpu=$1
case_name=$2
fixture=$3
resolved_config=$4
output=$5
case "$gpu" in 6|7) ;; *) echo "Only physical GPU 6 or 7 is authorized" >&2; exit 2;; esac

script_dir=$(cd "$(dirname "$0")" && pwd -P)
repo=$(cd "$script_dir/../.." && pwd -P)
cd "$repo"
test -f "$fixture"
test -f "$resolved_config"
mkdir "$output"

set +u
source /etc/profile.d/wm.sh
set -u
uuid=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$uuid"
export PYTHONPATH="$repo/gift:$repo/gift/contrib/flash-linear-attention${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCH_LOGS=recompiles,graph_breaks
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0

python=/opt/torchtitan-venv/bin/python3.12
{
    hostname
    git rev-parse HEAD
    git status --short --branch -- gift
    sha256sum "$fixture" "$resolved_config" "$script_dir"/*.{py,sh}
    nvidia-smi --id="$gpu" --query-gpu=index,uuid,name,driver_version --format=csv
} > "$output/provenance.log"

index=0
for arm in A B B A; do
    index=$((index + 1))
    apps=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "$apps" ]]; then
        echo "Physical GPU $gpu has compute processes: $apps" >&2
        exit 1
    fi
    log_base="$output/$index-$arm"
    nvidia-smi --id="$gpu" --query-gpu=temperature.gpu,clocks.sm,memory.used,utilization.gpu --format=csv \
        > "$log_base.gpu-before.csv"
    export TORCHINDUCTOR_CACHE_DIR="$output/cache-$arm/torchinductor"
    export TRITON_CACHE_DIR="$output/cache-$arm/triton"
    export TILELANG_CACHE_DIR="$output/cache-$arm/tilelang"
    profile=()
    if [[ $index -le 2 ]]; then
        profile+=(--profile)
    fi
    command=("$python" -B -u "$script_dir/measure.py" --case "$case_name" --arm "$arm"
        --fixture "$fixture" --resolved-config "$resolved_config" --output "$log_base.json"
        --warmup 10 --samples 30 "${profile[@]}")
    printf '%q ' "${command[@]}" >> "$output/provenance.log"
    printf '\n' >> "$output/provenance.log"
    timeout --signal=TERM --kill-after=30 900 "${command[@]}" > "$log_base.log" 2>&1
    nvidia-smi --id="$gpu" --query-gpu=temperature.gpu,clocks.sm,memory.used,utilization.gpu --format=csv \
        > "$log_base.gpu-after.csv"
done
