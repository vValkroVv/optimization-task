#!/usr/bin/env bash
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /etc/profile.d/wm.sh
set -u

gpu=${1:?physical GPU}
chunks=${2:?loss chunks}
series=${3:?run label}
shift 3
case "$gpu:$chunks" in
    2:2|3:8) ;;
    *) echo "This lane owns C2 on GPU 2 and C8 on GPU 3" >&2; exit 1 ;;
esac
out=$PWD/$series-c$chunks
mkdir "$out"
trap 'echo "$?" > "$out/exit-code"' EXIT
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,driver_version,memory.used,clocks.sm,clocks.mem --format=csv > "$out/device-start.csv"
unset PYTORCH_CUDA_ALLOC_CONF
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$PWD/smoke-c8/inductor TRITON_CACHE_DIR=$PWD/smoke-c8/triton
python=/opt/torchtitan-venv/bin/python3.12

i=0
for spec in "$@"; do
    IFS=, read -r arm rest <<< "$spec"
    options=()
    case "$arm" in
        packed|packed-order) options+=(--plan "$PWD/tune-c$chunks/plan.json") ;;
    esac
    [[ $rest != *fp64* ]] || options+=(--fp64)
    [[ $rest != *stages* ]] || options+=(--stages)
    [[ $rest != *alias* ]] || options+=(--alias)
    [[ $rest != *short* ]] || options+=(--warmup 2 --samples 2)
    echo "GPU $gpu C$chunks $spec"
    timeout 250 "$python" -Bu replay_native.py "$chunks" "$arm" "$out/$i-$arm.json" "${options[@]}" > "$out/$i-$arm.log" 2>&1
    ((i+=1))
done

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,driver_version,memory.used,clocks.sm,clocks.mem --format=csv > "$out/device-end.csv"
