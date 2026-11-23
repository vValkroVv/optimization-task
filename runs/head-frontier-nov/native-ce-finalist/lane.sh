#!/usr/bin/env bash
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /etc/profile.d/wm.sh
set -u

gpu=${1:?physical GPU}
chunks=${2:?chunks}
series=${3:?series}
shift 3
case "$gpu:$chunks" in
    0:2|1:8) ;;
    *) echo "Expected physical GPU 0 for C2 or GPU 1 for C8" >&2; exit 1 ;;
esac
out="$PWD/$series-c$chunks"
mkdir "$out"
trap 'echo "$?" > "$out/exit-code"' EXIT
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
test -n "$CUDA_VISIBLE_DEVICES"
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,driver_version,memory.used,clocks.sm,clocks.mem --format=csv > "$out/device-start.csv"
unset PYTORCH_CUDA_ALLOC_CONF FUSED_CE_BINARY
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor-$gpu" TRITON_CACHE_DIR="$PWD/cache/triton-$gpu"

i=0
for spec in "$@"; do
    test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
    case "$spec" in
        control|fused) arm=$spec; method=${NATIVE_CE_METHOD:-cuda-13} ;;
        cuda-13|cuda-16) arm=fused; method=$spec ;;
        *) echo "Unknown arm $spec" >&2; exit 1 ;;
    esac
    case "$method" in
        cuda-13) export FUSED_CE_BINARY=centered.so ;;
        cuda-16) export FUSED_CE_BINARY=centered-opt.so ;;
        *) echo "Expected cuda-13 or cuda-16" >&2; exit 1 ;;
    esac
    echo "GPU $gpu C$chunks $method $arm"
    timeout 250 /opt/torchtitan-venv/bin/python3.12 -Bu native_ce.py "$chunks" "$arm" \
        "$out/$i-$spec.json" --method "$method" > "$out/$i-$spec.log" 2>&1
    ((i+=1))
done
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,driver_version,memory.used,clocks.sm,clocks.mem --format=csv > "$out/device-end.csv"
