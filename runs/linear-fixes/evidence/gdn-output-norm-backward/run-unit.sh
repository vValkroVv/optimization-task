#!/usr/bin/env bash
set -eo pipefail
out="$(cd -- "$(dirname -- "$0")" && pwd)"
source /etc/profile.d/wm.sh
cd "$out/source"
export CUDA_VISIBLE_DEVICES="$1" FLA_DISABLE_BACKEND_DISPATCH=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/gift/contrib/flash-linear-attention:$PWD/gift"
export TORCHINDUCTOR_CACHE_DIR="$out/unit-inductor" TRITON_CACHE_DIR="$out/unit-triton"
export TORCHINDUCTOR_COMPILE_THREADS=2
trap '"'"'printf "%s\n" "$?" > "$out/unit-exit-code"'"'"' EXIT
timeout 420s /opt/torchtitan-venv/bin/python3.12 -B -m pytest -q -p no:cacheprovider   gift/tests/unit_tests/test_gdn_output_norm_fp8.py --junitxml "$out/unit.xml"
