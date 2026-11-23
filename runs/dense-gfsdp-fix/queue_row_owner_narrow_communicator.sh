#!/usr/bin/env bash
set -eEo pipefail

workspace=/workspace-SR008.nfs2/vkropotin
source_root="$workspace/dg-a"
owner="$workspace/dg-fix/runs/dense-gfsdp-fix"
staging=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
prior="$owner/artifacts/reviewed-port-001/layer0-prefetch-profile-001"
output="$owner/artifacts/raw/row-owner-narrow-communicator-001"
fixture="$workspace/optimization-task/runs/production-data-abcd/artifacts/raw/proddata-component-capture-001/native-inputs"
python=/opt/torchtitan-venv/bin/python3.12
driver="$staging/row_owner_narrow_communicator.py"
trap 'status=$?; printf "%s\n" "$status" > "$staging/queue.exit"; exit "$status"' EXIT

test -s "$prior/queue.pid"
while [[ ! -s "$prior/queue.exit" ]]; do
    kill -0 "$(cat "$prior/queue.pid")"
    (( SECONDS < 14400 ))
    sleep 120
done
(
    cd "$staging"
    sha256sum --check sources.sha256
)
test "$(git -C "$source_root" rev-parse HEAD)" = e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f
test -z "$(git -C "$source_root" status --porcelain -- gift runs/production-data-abcd/component_fixtures.py)"
source /etc/profile.d/wm.sh
set -u
test "$(cat /etc/volcano/VC_MPIMASTER_NUM)" = 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export ROW_OWNER_REPLAY_SOURCE_ROOT="$source_root"
export PYTHONPATH="$source_root/gift/contrib/flash-linear-attention:$source_root/gift/contrib/flash-qla:$source_root/gift"
mkdir "$output"
cd "$source_root"
"$python" -Bu "$staging/row_owner_route_summary.py" "$fixture" \
    "$staging/embedding-replay-result.json" --fixture-only

for arm in shared dedicated-default dedicated-8 dedicated-4; do
    directory="$output/$arm"
    mkdir "$directory"
    nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used --format=csv,noheader > "$directory/gpus.csv"
    test "$(wc -l < "$directory/gpus.csv")" = 8
    test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    awk -F', ' '$3 !~ /H100/ || $5 != "0 MiB" {exit 1}' "$directory/gpus.csv"
    cmp "$output/shared/gpus.csv" "$directory/gpus.csv"
    command=(env TZ=UTC NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=INIT,ALLOC,P2P,CALL,DESTROY
        "NCCL_DEBUG_FILE=$directory/nccl-%h-%p.log" NCCL_DEBUG_TIMESTAMP_LEVELS=WARN,INFO,TRACE
        'NCCL_DEBUG_TIMESTAMP_FORMAT=[%F %T.%3f%z] '
        "$python" -Bu -m torch.distributed.run --standalone --nproc_per_node=8
        "$driver" --arm "$arm" --fixture "$fixture" --output "$directory")
    printf '%q ' "${command[@]}" > "$directory/command"
    printf '\n' >> "$directory/command"
    printf '%s START embedding communicator %s\n' "$(date -u +'%FT%TZ')" "$arm"
    timeout --signal=TERM --kill-after=45s 600s "${command[@]}" > "$directory/launcher.log" 2>&1
done
"$python" -Bu "$driver" --summarize --output "$output"
