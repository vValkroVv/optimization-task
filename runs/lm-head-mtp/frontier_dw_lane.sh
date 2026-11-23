#!/usr/bin/env bash
# One GPU and cache directory per lane. Each arm runs in a fresh process.
set -Eeo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source /etc/profile.d/wm.sh
set -u
gpu=${1:?GPU index}
label=${2:?unique series label}
shift 2
base=$PWD/runs/lm-head-mtp
out=$base/frontier-dw/$label
mkdir "$out"
trap 'code=$?; echo "$code" > "$out/exit-code"' EXIT
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
test -z "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=$uuid CUDACHERRY_DISABLE_JIT=1
export TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=2
export TRITON_CACHE_DIR=$out/triton TORCHINDUCTOR_CACHE_DIR=$out/inductor
export PYTHONPYCACHEPREFIX=$out/pycache
echo "$$ $uuid $(date -u +%FT%TZ)" > "$out/launcher"
python=/opt/torchtitan-venv/bin/python3.12
i=0
for spec in "$@"; do
    IFS=: read -r chunks pair order <<< "$spec"
    for arm in ${order:-cublas lt lt cublas}; do
        output=$out/$chunks-$i-$arm.json
        optional_flags=()
        if [[ $arm == *-num ]]; then
            arm=${arm%-num}
            optional_flags+=(--numerics)
        fi
        printf '%s %s %s %s\n' "$(date -u +%FT%TZ)" "$chunks" "$pair" "$arm"
        timeout 240 "$python" -Bu "$base/frontier_dw_replay.py" "$arm" "$output" \
            --plan "$base/frontier-dw/tune-$chunks.json" --pair "$pair" "${optional_flags[@]}" \
            > "${output%.json}.log" 2>&1
        ((i+=1))
    done
done
