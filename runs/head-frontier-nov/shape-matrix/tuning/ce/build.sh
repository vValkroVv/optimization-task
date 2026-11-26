#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: bash build.sh BASELINE_CUDA_SOURCE_DIR NEW_OUTPUT_DIR [TUNING=3] [PHASE=1]" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
baseline=$(cd "$1" && pwd)
output=$2
tuning=${3:-3}
phase=${4:-1}
case "$phase:$tuning" in
    1:1|1:3|1:7|1:11|1:15|2:19|2:35|2:51|2:67) ;;
    *) echo "Unknown sweep phase and tuning combination" >&2; exit 2 ;;
esac

(
    cd "$baseline"
    sha256sum -c <<'SHA256'
cb3326849f841af744975346509f38dc125c3c54127abaea38f8b79de9196070  centered.cu
fe54ab060e844d3cb43d4d6b98c90cab9c18a730da6fa751d51e676660e9f0c5  exact.cuh
SHA256
)
mkdir "$output"
cd "$output"
mkdir work
cp "$baseline/centered.cu" "$baseline/exact.cuh" work/
patch -d work -p1 < "$here/candidate.patch"
if [ "$phase" = 2 ]; then
    patch -d work -p1 < "$here/second-sweep.patch"
fi

binary="centered-tune-$tuning.so"
/usr/local/cuda-13.2/bin/nvcc -O3 -shared -Xcompiler=-fPIC -lineinfo -Xptxas=-v \
    -arch=sm_90a --fmad=false --cudart=static -DCE_TUNING="$tuning" \
    work/centered.cu -o "$binary" > "build-$tuning.log" 2>&1
sha256sum "$binary"
nm -D "$binary" | grep centered_ce_launch_folded_v1
