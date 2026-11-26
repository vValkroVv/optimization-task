#!/usr/bin/env bash
set -eo pipefail

source /etc/profile.d/wm.sh
cd "$(dirname "$0")/../../.."

model=$1
gpu=$2
shift 2
case "$model:$gpu" in
    100b:6) fixture_sha=ca9b18abaad9b21ec050854fb2b7a663b81841caa4ecca5f29b205987b8c217e ;;
    700b:7) fixture_sha=85922b3719fe1162e5f0a33e19bb4d2f4ec82202304a9bfecf1f970ea54c6e45 ;;
    *) echo "Expected 100b on GPU 6 or 700b on GPU 7" >&2; exit 2 ;;
esac

out=length-grid/region-tuning/ce
printf '%s  %s\n' "$fixture_sha" "inputs/$model.pt" | sha256sum --check --strict
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CUDAGRAPHS=0 CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/$out/inductor-$model"
export TRITON_CACHE_DIR="$PWD/$out/triton-$model"
export PYTHONPATH="$PWD/integrated/gift:$PWD:$PWD/../../../.worktrees/final-all-fix/gift:$PWD/../../../../lm10-c1/gift"

for seq in "$@"; do
    nvidia-smi -i "$gpu" --query-gpu=index,name,uuid,driver_version,memory.used --format=csv,noheader
    if [[ -n $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader) ]]; then
        echo "GPU $gpu has an active compute process" >&2
        exit 1
    fi
    /opt/torchtitan-venv/bin/python3.12 -Bu "$out/bench_long.py" \
        --model "$model" --seq "$seq" --output "$out/$model-$seq.json"
done
