#!/usr/bin/env bash
set -Eeuo pipefail

workspace=/workspace-SR008.nfs2/vkropotin
control="$workspace/dg-fix"
compact="$workspace/dg-cp"
owner="$control/runs/dense-gfsdp-fix"
staging=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
prior="$owner/artifacts/reviewed-port-001/row-owner-narrow-communicator-001"
numeric="$staging/compact_owner_numeric.py"
driver="$staging/compact_acceptance.py"
output="$owner/artifacts/raw/compact-native-002"
source_data="$workspace/optimization-task/runs/production-data-abcd/artifacts/raw"
capture="$source_data/proddata-component-capture-001/native-inputs"
census="$source_data/loader-verification/data002-full/depth0"
data="$source_data/data-002/train"
reference="$workspace/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2"
row_owner=gift/torchtitan/distributed/gfsdp/row_owner_embedding.py
python=/opt/torchtitan-venv/bin/python3.12
package_files=(
    queue_compact_acceptance.sh campaign.py compact_owner_numeric.py compact_acceptance.py
    compact_coalescing_replay.py row_owner_route_numeric.py row_owner_route_experiment.py
    embedding-replay-result.json
)

stage_recovery() {
    local destination="$1" name
    for name in "${package_files[@]}"; do
        test -s "$staging/$name"
    done
    mkdir -- "$destination"
    for name in "${package_files[@]}"; do
        cp -- "$staging/$name" "$destination/$name"
    done
    (
        cd "$destination"
        sha256sum -- "${package_files[@]}" > sources.sha256
    )
}

if [[ ${1-} == --stage ]]; then
    test "$#" = 2
    stage_recovery "$2"
    exit
fi

trap 'status=$?; printf "%s\n" "$status" > "$staging/queue.exit"; exit "$status"' EXIT

test "$#" = 3
compact_hash="$1"
control_owner_hash="$2"
compact_owner_hash="$3"
test -s "$prior/queue.pid"
while [[ ! -s "$prior/queue.exit" ]]; do
    kill -0 "$(cat "$prior/queue.pid")"
    (( SECONDS < 14400 ))
    sleep 120
done
printf '%s Finished predecessor queue with status %s\n' "$(date -u +'%FT%TZ')" "$(cat "$prior/queue.exit")"
verify_stage() (
    cd "$staging"
    sha256sum -- "${package_files[@]}" | cmp - sources.sha256
)
verify_stage
printf '%s  %s\n' "$control_owner_hash" "$control/$row_owner" \
    "$compact_owner_hash" "$compact/$row_owner" | sha256sum -c -
for helper in runs/ffn-row-owner/{embedding_numeric.py,components.py} \
    runs/production-data-abcd/component_fixtures.py; do
    cmp "$control/$helper" "$compact/$helper"
done
test -r "$data/index.json"
for rank in {0..7}; do
    test -r "$capture/rank$rank/receipt.json"
    test -r "$capture/rank$rank/tensors.pt"
    test -r "$census/rank$rank.json"
done

source /etc/profile.d/wm.sh
test "$(cat /etc/volcano/VC_MPIMASTER_NUM)" = 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

with_numeric_environment() (
    local source="$1"
    shift
    cd "$source"
    export PYTHONPATH="$source/gift/contrib/flash-linear-attention:$source/gift/contrib/flash-qla:$source/gift"
    export ROW_OWNER_REPLAY_SOURCE_ROOT="$source"
    "$@"
)

preflight_numeric() {
    local source="$1"
    with_numeric_environment "$source" "$python" -B "$numeric" run --help >/dev/null
    with_numeric_environment "$source" "$python" -B - "$staging" "$source" <<'PY'
from pathlib import Path
import runpy
import sys

staging, root = (Path(value).resolve(strict=True) for value in sys.argv[1:])
sys.path.insert(0, str(staging))
import campaign
import compact_owner_numeric as driver
import row_owner_route_numeric as numeric
from compact_coalescing_replay import low_fixture
from streaming.base.local import LocalDataset
from torchtitan.distributed.gfsdp import row_owner_embedding

campaign.require(
    Path(campaign.__file__).resolve() == staging / "campaign.py",
    "Exact replay imported campaign outside its frozen package",
)
campaign.require(
    numeric.SOURCE_ROOT == root
    and Path(row_owner_embedding.__file__).resolve() == root / driver.OWNER,
    "Exact replay imported a different GIFT source root",
)
campaign.require(
    len(campaign.read_json(driver.REFERENCE)["input_ids_sha256_by_rank"]) == numeric.WORLD,
    "Exact replay lacks the eight accepted high-U input pins",
)
runpy.run_path(str(root / "runs/ffn-row-owner/embedding_numeric.py"))
runpy.run_path(str(root / "runs/production-data-abcd/component_fixtures.py"))
PY
}

preflight_numeric "$control"
preflight_numeric "$compact"
env -u PYTHONPATH "$python" -B "$numeric" compare --help >/dev/null
env -u PYTHONPATH "$python" -B "$driver" --help >/dev/null
mkdir "$output"
mkdir "$output/exact"

run_numeric() {
    local arm="$1" source="$2" digest="$3"
    local inventory="$output/exact/$arm.gpus.csv"
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader > "$inventory"
    test "$(wc -l < "$inventory")" = 8
    awk -F', ' '$2 !~ /H100/ || $3 != "0 MiB" {exit 1}' "$inventory"
    test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    local command=("$python" -Bu -m torch.distributed.run --standalone --nproc_per_node=8
        "$numeric" run --arm "$arm" --source-root "$source" --owner-source-sha256 "$digest"
        --high-fixture "$capture" --low-census "$census" --data-root "$data"
        --output "$output/exact/$arm.json")
    printf '%q ' "${command[@]}" > "$output/exact/$arm.command"
    printf '\n' >> "$output/exact/$arm.command"
    printf '%s START exact compact comparison %s\n' "$(date -u +'%FT%TZ')" "$arm"
    with_numeric_environment "$source" timeout --signal=TERM --kill-after=45s 900s "${command[@]}" \
        > "$output/exact/$arm.log" 2>&1
}

run_numeric a "$control" "$control_owner_hash"
run_numeric b "$compact" "$compact_owner_hash"
verify_stage
env -u PYTHONPATH "$python" -Bu "$numeric" compare \
    "$output/exact/a.json" "$output/exact/b.json" --output "$output/exact/comparison.json"
test -s "$output/exact/comparison.json"
printf '%s PASS exact compact comparison\n' "$(date -u +'%FT%TZ')"

"$python" -Bu "$driver" "$output/timing" --ep 8 \
    --original-root "$workspace/dg-a" --candidate-root "$control" --compact-root "$compact" \
    --reference "$reference" --candidate-head e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f \
    --candidate-source-sha256 da8fdaca51a3aaa509eb9f0e5e3108f0c3831c5f88e828932324be0b96898abe \
    --compact-source-sha256 "$compact_hash"
