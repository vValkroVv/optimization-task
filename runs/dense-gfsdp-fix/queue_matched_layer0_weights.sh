#!/usr/bin/env bash
set -eEo pipefail

workspace=/workspace-SR008.nfs2/vkropotin
control="$workspace/dg-a"
candidate="$workspace/dg-fix"
owner="$candidate/runs/dense-gfsdp-fix"
staging=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
prior="$owner/artifacts/reviewed-port-001/owner-add-002"
output="$owner/artifacts/raw/matched-layer0-002"
selection="$owner/artifacts/raw/native-001/initial-weights/comparison.json"
reference="$workspace/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2"
python=/opt/torchtitan-venv/bin/python3.12
trap 'status=$?; printf "%s\n" "$status" > "$staging/queue.exit"; exit "$status"' EXIT

test -s "$prior/queue.pid"
while [[ ! -s "$prior/queue.exit" ]]; do
    kill -0 "$(cat "$prior/queue.pid")"
    (( SECONDS < 14400 ))
    sleep 120
done
test -s "$selection"
(
    cd "$staging"
    sha256sum --check sources.sha256
)

source /etc/profile.d/wm.sh
set -u
test "$(cat /etc/volcano/VC_MPIMASTER_NUM)" = 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir "$output"
driver="$staging/matched_layer0_weights.py"
pins=(
    --original-root "$control" --candidate-root "$candidate" --reference "$reference"
    --candidate-head e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f
    --candidate-source-sha256 da8fdaca51a3aaa509eb9f0e5e3108f0c3831c5f88e828932324be0b96898abe
    --selection "$selection" --initial-rng-probe
)
for arm in a layer0; do
    "$python" -B "$driver" run "$output" "$arm" "${pins[@]}" --preflight-only
done
for arm in a layer0; do
    printf '%s START matched layer-0 weights %s\n' "$(date -u +'%FT%TZ')" "$arm"
    "$python" -Bu "$driver" run "$output" "$arm" "${pins[@]}"
done
"$python" -Bu "$driver" compare "$output"
