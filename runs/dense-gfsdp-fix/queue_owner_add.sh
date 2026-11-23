#!/usr/bin/env bash
set -Eeuo pipefail

workspace=/workspace-SR008.nfs2/vkropotin
owner="$workspace/dg-fix/runs/dense-gfsdp-fix"
staging=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
prior="$owner/artifacts/reviewed-port-001/compact-native-002"
output="$owner/artifacts/raw/owner-add-002"
fixtures="$workspace/optimization-task/runs/production-data-abcd/artifacts/raw/proddata-component-capture-001/native-inputs"
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

mkdir "$output"
bash "$staging/run_owner_add_replay.sh" "$workspace/dg-a" "$fixtures" \
    "$output" "$staging/archived_owner_add_kernels.py"
