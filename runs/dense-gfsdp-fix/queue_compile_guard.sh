#!/usr/bin/env bash
set -eEo pipefail

workspace=/workspace-SR008.nfs2/vkropotin
original="$workspace/dg-a"
candidate="$workspace/dg-fix"
owner="$candidate/runs/dense-gfsdp-fix"
native="$owner/artifacts/raw/native-001"
output="$owner/artifacts/raw/compile-guard-001"
script="$owner/compile_guard_001.py"
reference="$workspace/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2"
py=/opt/torchtitan-venv/bin/python3.12

test $# = 1
printf '%s  %s\n' "$1" "$script" | sha256sum -c -
mkdir "$output"
receipt="$output/queue.exit"
trap 'status=$?; printf "%s\n" "$status" > "$receipt"; exit "$status"' EXIT

test -s "$native/queue-initial-weights.pid"
while [[ ! -s "$native/queue-initial-weights.exit" ]]; do
    kill -0 "$(cat "$native/queue-initial-weights.pid")"
    (( SECONDS < 14400 ))
    sleep 120
done
"$py" -B - "$native/final/results.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    result = json.load(stream)
arms = result["arms"]
if (result["pending"] or set(arms) != {"ep8-a", "ep8-all", "ep4-a", "ep4-all"}
        or any(value["compilation_status"] != "observed_stable" for value in arms.values())):
    raise RuntimeError("The final native timing matrix is incomplete")
PY

source /etc/profile.d/wm.sh
set -u
test "$(cat /etc/volcano/VC_MPIMASTER_NUM)" = 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
pins=(
    --original-root "$original" --candidate-root "$candidate" --reference "$reference"
    --candidate-head e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f
    --candidate-source-sha256 da8fdaca51a3aaa509eb9f0e5e3108f0c3831c5f88e828932324be0b96898abe
)
cd "$candidate"
for arm in a row-owner layer0-row-owner; do
    "$py" -B "$script" "$output" "$arm" "${pins[@]}" --preflight-only
done

failed=0
for arm in a row-owner layer0-row-owner; do
    printf '%s START compile guard %s\n' "$(date -u +'%FT%TZ')" "$arm"
    if "$py" -Bu "$script" "$output" "$arm" "${pins[@]}"; then
        printf '%s PASS compile guard %s\n' "$(date -u +'%FT%TZ')" "$arm"
    else
        printf '%s FAIL compile guard %s\n' "$(date -u +'%FT%TZ')" "$arm"
        failed=1
    fi
done
exit "$failed"
