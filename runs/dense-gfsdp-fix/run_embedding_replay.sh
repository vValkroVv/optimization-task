#!/usr/bin/env bash
set -Eeuo pipefail

source_root=$(realpath -- "${1:?Original A worktree}")
input_root=$(realpath -- "${2:?Captured real-data inputs}")
output_root=$(realpath -- "${3:?New run directory}")
python=/opt/torchtitan-venv/bin/python3.12

trap 'rc=$?; printf "%s\n" "$rc" > "$output_root/exit-code.tmp"; mv "$output_root/exit-code.tmp" "$output_root/exit-code"' EXIT
printf '%s\n' "$$" > "$output_root/driver.pid"
[[ $(git -C "$source_root" rev-parse HEAD) == e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f ]]
[[ -z $(git -C "$source_root" status --porcelain -- gift) ]]
[[ $(cat /etc/volcano/VC_MPIMASTER_NUM) == 1 ]]
for rank in {0..7}; do
    test -r "$input_root/rank$rank/receipt.json"
    test -r "$input_root/rank$rank/tensors.pt"
done

source /etc/profile.d/wm.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$source_root/gift/contrib/flash-linear-attention:$source_root/gift/contrib/flash-qla:$source_root/gift"
cd "$source_root"

for arm in baseline owner; do
    gpu_log="$output_root/$arm.gpus.csv"
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader > "$gpu_log"
    [[ $(wc -l < "$gpu_log") -eq 8 ]]
    [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
    awk -F', ' '$2 !~ /H100/ || $3 != "0 MiB" { exit 1 }' "$gpu_log"
    mkdir "$output_root/$arm"
    command=(
        "$python" -Bu -m torch.distributed.run --standalone --nproc_per_node=8
        runs/ffn-row-owner/components.py --execute --variant "$arm"
        --scope integrated --cases fixture --input-fixtures "$input_root"
        --warmup 10 --samples 10 --output "$output_root/$arm"
    )
    printf '%(%FT%TZ)T %s\n' -1 "$arm"
    printf '%q ' "${command[@]}" > "$output_root/$arm.command"
    printf '\n' >> "$output_root/$arm.command"
    timeout --signal=TERM --kill-after=45s 600s "${command[@]}" > "$output_root/$arm.log" 2>&1
done
