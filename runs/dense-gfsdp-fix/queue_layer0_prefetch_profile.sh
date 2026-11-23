#!/usr/bin/env bash
set -eEo pipefail

workspace=/workspace-SR008.nfs2/vkropotin
control="$workspace/dg-a"
candidate="$workspace/dg-l0p"
owner="$workspace/dg-fix/runs/dense-gfsdp-fix"
staging=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
prior="$owner/artifacts/reviewed-port-001/matched-layer0-002"
output="$owner/artifacts/raw/layer0-prefetch-profile-002"
reference="$workspace/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2"
python=/opt/torchtitan-venv/bin/python3.12
trap 'status=$?; printf "%s\n" "$status" > "$staging/queue.exit"; exit "$status"' EXIT

test "$#" = 1
candidate_hash="$1"
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

source /etc/profile.d/wm.sh
set -u
test "$(cat /etc/volcano/VC_MPIMASTER_NUM)" = 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir "$output"
pins=(
    --original-root "$control" --candidate-root "$candidate" --reference "$reference"
    --candidate-head e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f
    --candidate-source-sha256 "$candidate_hash"
)
"$python" -B "$staging/diagnostic.py" "$output" 8 all "${pins[@]}" --preflight-only

(
    cd "$candidate/gift"
    export PYTHONPATH="$PWD/contrib/flash-linear-attention:$PWD/contrib/flash-qla:$PWD"
    timeout --signal=TERM --kill-after=45s 300s "$python" -B -m pytest -q -o addopts= \
        tests/unit_tests/test_ffn_overlap.py -k test_nested_wrapping_and_prefetch_entry_order
) > "$output/prefetch-wiring-tests.log" 2>&1
(
    cd "$candidate"
    raw=runs/overnight-streaming-perf/artifacts/raw
    # The runner checks the path without a trailing slash before creating it.
    # Git's directory-only ignore rule needs the directory to exist for that check.
    resolved_raw=$("$python" -B -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve())' "$raw")
    test "$resolved_raw" = "$(pwd -P)/$raw"
    git check-ignore -q -- "$raw/"
    mkdir -p -- "$raw"
    git check-ignore -q -- "$raw"
)
printf '%s START early layer-0 FFN prefetch profile\n' "$(date -u +'%FT%TZ')"
"$python" -Bu "$staging/diagnostic.py" "$output" 8 all "${pins[@]}"

receipt="$output/diagnostic/ep8-all.json"
attempt=$("$python" -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["attempt"])' "$receipt")
"$python" -B "$staging/layer0_prefetch_trace.py" "$receipt" "$attempt" --output "$output/candidate-mechanism.json"
