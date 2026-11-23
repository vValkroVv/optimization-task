#!/usr/bin/env bash
# Use fresh processes for each arm. Set only NCCL diagnostic logging variables.
set -Eeuo pipefail

if [[ $# != 3 ]]; then
    printf 'usage: %s ORIGINAL_A CAPTURE EMPTY_OUTPUT\n' "$0" >&2
    exit 2
fi
source_root=$(realpath -- "$1")
input_root=$(realpath -- "$2")
output_root=$(realpath -- "$3")
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python=/opt/torchtitan-venv/bin/python3.12
[[ -d "$output_root" && -z $(ls -A -- "$output_root") ]]
trap 'rc=$?; printf "%s\n" "$rc" > "$output_root/exit-code.tmp"; mv "$output_root/exit-code.tmp" "$output_root/exit-code"' EXIT
printf '%s\n' "$$" > "$output_root/driver.pid"

[[ $(git -C "$source_root" rev-parse HEAD) == e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f ]]
[[ -z $(git -C "$source_root" status --porcelain -- gift runs/production-data-abcd/component_fixtures.py) ]]
[[ $(cat /etc/volcano/VC_MPIMASTER_NUM) == 1 ]]
source /etc/profile.d/wm.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$source_root/gift/contrib/flash-linear-attention:$source_root/gift/contrib/flash-qla:$source_root/gift"
export ROW_OWNER_REPLAY_SOURCE_ROOT="$source_root"
cd "$source_root"
artifacts=(
    "$experiment/"{row_owner_external_memory.py,run_row_owner_external_memory.sh,embedding-replay-result.json,row_owner_route_experiment.py,row_owner_route_numeric.py,row_owner_route_summary.py}
    runs/production-data-abcd/component_fixtures.py
    gift/torchtitan/distributed/gfsdp/{_collectives.py,_param_group.py,row_owner_embedding.py}
)
sha256sum -- "${artifacts[@]}" > "$output_root/sources.sha256"
"$python" -Bu "$experiment/row_owner_route_summary.py" \
    "$input_root" "$experiment/embedding-replay-result.json" --fixture-only

for arm in control row-owner; do
    arm_output="$output_root/$arm"
    mkdir "$arm_output"
    nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used --format=csv,noheader > "$arm_output/gpus.csv"
    [[ $(wc -l < "$arm_output/gpus.csv") -eq 8 ]]
    [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
    awk -F', ' '$3 !~ /H100/ || $5 != "0 MiB" { exit 1 }' "$arm_output/gpus.csv"
    command=(env TZ=UTC NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=INIT,ALLOC,P2P,CALL,DESTROY
        "NCCL_DEBUG_FILE=$arm_output/nccl-%h-%p.log" NCCL_DEBUG_TIMESTAMP_LEVELS=WARN,INFO,TRACE
        'NCCL_DEBUG_TIMESTAMP_FORMAT=[%F %T.%3f%z] '
        "$python" -Bu -m torch.distributed.run --standalone --nproc_per_node=8
        "$experiment/row_owner_external_memory.py" --execute --arm "$arm"
        --fixture "$input_root" --output "$arm_output")
    printf '%q ' "${command[@]}" > "$arm_output/command"
    printf '\n' >> "$arm_output/command"
    timeout --signal=TERM --kill-after=45s 600s "${command[@]}" > "$arm_output/launcher.log" 2>&1
    printf '0\n' > "$arm_output/exit-code"
done
cmp "$output_root/control/gpus.csv" "$output_root/row-owner/gpus.csv"
sha256sum --check --status "$output_root/sources.sha256"
"$python" -Bu "$experiment/row_owner_external_memory.py" --summarize --output "$output_root"
