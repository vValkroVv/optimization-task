#!/usr/bin/env bash
set -Eeuo pipefail

control=${1:?Supply the control campaign state directory}
tag=${2:?Supply a fresh component ID}
[[ $# == 2 && $tag =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /etc/profile.d/wm.sh
gift=$(realpath ../../gift)
export PYTHONPATH="$gift/contrib/flash-linear-attention:$gift/contrib/flash-qla:$gift"
export PYTHONDONTWRITEBYTECODE=1
python=/opt/torchtitan-venv/bin/python3.12
state=$(realpath -m "artifacts/raw/$tag")
mkdir -- "$state"
exec >"$state/component.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$state/exit-code.txt"' EXIT
printf '%s\n' "$$" > "$state/pid"

timeout 1200 bash -c '
    while [[ ! -f $1/exit-code.txt ]]; do sleep 30; done
    pid=$(cat "$1/pid")
    while p=$(ps -p "$pid" -o stat=) && [[ $p != Z* ]]; do sleep 5; done
' bash "$control"

for gpu in 5 6 7; do
    if [[ -n $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader) ]]; then
        printf 'GPU %s is occupied\n' "$gpu"
        exit 1
    fi
done

(
    export CUDA_VISIBLE_DEVICES=5
    export TORCHINDUCTOR_CACHE_DIR="$state/dense-inductor"
    export TRITON_CACHE_DIR="$state/dense-triton"
    "$python" -B -u verify_dense_pad.py --output "$state/dense-pad.json"
    export CUDA_VISIBLE_DEVICES=5,7
    export TORCHINDUCTOR_CACHE_DIR="$state/gfsdp-inductor"
    export TRITON_CACHE_DIR="$state/gfsdp-triton"
    "$python" -B -m torch.distributed.run --standalone --nproc-per-node=2 \
        verify_dense_pad.py --gfsdp --output "$state/dense-gfsdp.json"
) >"$state/dense.log" 2>&1 &
dense=$!
(
    export CUDA_VISIBLE_DEVICES=6
    export TORCHINDUCTOR_CACHE_DIR="$state/cute-inductor"
    export TRITON_CACHE_DIR="$state/cute-triton"
    "$python" -B -u verify_cute_residual.py --output "$state/cute-residual.json"
) >"$state/cute.log" 2>&1 &
cute=$!
if wait "$dense"; then dense_status=0; else dense_status=$?; fi
if wait "$cute"; then cute_status=0; else cute_status=$?; fi
printf 'Dense exit=%s\nCuTe exit=%s\n' "$dense_status" "$cute_status"
(( dense_status == 0 && cute_status == 0 ))
