#!/usr/bin/env bash
set -Eeuo pipefail

workspace=/workspace-SR008.nfs2/vkropotin
owner="$workspace/dg-fix/runs/dense-gfsdp-fix"
staging=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
prior="$owner/artifacts/reviewed-port-001/owner-add-001"
output="$owner/artifacts/raw/row-owner-external-memory-001"
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
    sha256sum --check --status sources.sha256
)
mkdir "$output"
bash "$staging/run_row_owner_external_memory.sh" "$workspace/dg-a" "$fixtures" "$output"
