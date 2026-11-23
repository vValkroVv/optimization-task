#!/usr/bin/env bash
# Serial phase-6 replay on the same pristine source and captured IDs as run_embedding_replay.sh.
set -Eeuo pipefail

source_root=$(realpath -- "${1:?Pristine Original A worktree}")
input_root=$(realpath -- "${2:?Captured real-data inputs}")
output_root=$(realpath -- "${3:?Empty, existing run directory}")
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python=/opt/torchtitan-venv/bin/python3.12
[[ -z $(ls -A -- "$output_root") ]]
trap 'rc=$?; printf "%s\n" "$rc" > "$output_root/exit-code.tmp"; mv "$output_root/exit-code.tmp" "$output_root/exit-code"' EXIT
printf '%s\n' "$$" > "$output_root/driver.pid"

original_a=e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f
[[ $(git -C "$source_root" rev-parse HEAD) == "$original_a" ]]
[[ -z $(git -C "$source_root" status --porcelain -- gift runs/ffn-row-owner runs/production-data-abcd/component_fixtures.py) ]]
[[ $(cat /etc/volcano/VC_MPIMASTER_NUM) == 1 ]]
for rank in {0..7}; do
    test -r "$input_root/rank$rank/receipt.json"
    test -r "$input_root/rank$rank/tensors.pt"
done

source /etc/profile.d/wm.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$source_root/gift/contrib/flash-linear-attention:$source_root/gift/contrib/flash-qla:$source_root/gift"
export ROW_OWNER_REPLAY_SOURCE_ROOT="$source_root"
cd "$source_root"
artifacts=(
    "$experiment/row_owner_route_experiment.py"
    "$experiment/row_owner_route_numeric.py"
    "$experiment/row_owner_route_summary.py"
    "$experiment/run_row_owner_route_replay.sh"
    "$experiment/embedding-replay-result.json"
    gift/torchtitan/distributed/gfsdp/row_owner_embedding.py
    gift/tests/unit_tests/test_row_owner_embedding.py
    runs/ffn-row-owner/components.py
    runs/ffn-row-owner/embedding_numeric.py
    runs/production-data-abcd/component_fixtures.py
)
sha256sum -- "${artifacts[@]}" > "$output_root/sources.sha256"
"$python" -Bu "$experiment/row_owner_route_summary.py" \
    "$input_root" "$experiment/embedding-replay-result.json" --fixture-only

run_serial() {
    local arm=$1
    shift
    local gpu_log="$output_root/$arm.gpus.csv"
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader > "$gpu_log"
    [[ $(wc -l < "$gpu_log") -eq 8 ]]
    [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
    awk -F', ' '$2 !~ /H100/ || $3 != "0 MiB" { exit 1 }' "$gpu_log"
    local -a command=("$python" -Bu -m torch.distributed.run --standalone --nproc_per_node=8 "$@")
    printf '%(%FT%TZ)T %s\n' -1 "$arm"
    printf '%q ' "${command[@]}" > "$output_root/$arm.command"
    printf '\n' >> "$output_root/$arm.command"
    timeout --signal=TERM --kill-after=45s 900s "${command[@]}" > "$output_root/$arm.log" 2>&1
    printf '0\n' > "$output_root/$arm.exit-code"
}

mkdir "$output_root/owner-tests-checkpoints"
run_serial owner-tests "$experiment/row_owner_route_numeric.py" --execute \
    --owner-checkpoint-dir "$output_root/owner-tests-checkpoints"
run_serial exact "$experiment/row_owner_route_numeric.py" --execute \
    --input-fixtures "$input_root" --seed 2048 --output "$output_root/exact.json"

for arm in A1 B1 A2 B2 diag-A diag-B; do
    flags=(--scope integrated --samples 10)
    if [[ $arm == diag-* ]]; then
        flags=(--scope stages --samples 1 --profile)
    fi
    if [[ $arm == B* || $arm == diag-B ]]; then
        flags+=(--fixed-shape-route)
    fi
    mkdir "$output_root/$arm"
    run_serial "$arm" "$experiment/row_owner_route_experiment.py" --execute \
        --variant owner --cases fixture --input-fixtures "$input_root" \
        --seed 2048 --warmup 10 --output "$output_root/$arm" "${flags[@]}"
done
sha256sum --check --status "$output_root/sources.sha256"
"$python" -Bu "$experiment/row_owner_route_summary.py" \
    "$output_root" "$experiment/embedding-replay-result.json"
