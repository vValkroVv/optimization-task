#!/usr/bin/env bash
set -eEo pipefail
workspace=/workspace-SR008.nfs2/vkropotin
candidate="$workspace/dg-fix"
original="$workspace/dg-a"
owner="$candidate/runs/dense-gfsdp-fix"
reviewed="$owner/artifacts/reviewed-port-001"
tools="$owner/artifacts/followup-tools-001"
suite="$owner/artifacts/raw/native-001"
raw="$workspace/optimization-task/runs/production-data-abcd/artifacts/raw"
capture="$raw/proddata-component-capture-001/native-inputs"
python=/opt/torchtitan-venv/bin/python3.12
trap 'status=$?; printf "%s\n" "$status" > "$suite/recover-row-owner.exit"; exit "$status"' EXIT

while [[ ! -s "$suite/queue-per-fix.exit" ]]; do
    kill -0 "$(cat "$suite/queue-per-fix.pid")"
    (( SECONDS < 81000 ))
    sleep 120
done
if [[ $(cat "$suite/queue-per-fix.exit") == 0 ]]; then
    exit 0  # The original serial chain already handles a complete direct matrix.
fi

pending=$("$python" -B - "$suite/per-fix/results.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    report = json.load(stream)
pending = sorted(report["pending"])
if (
    not pending
    or any(key not in {"ep4-row-owner", "ep8-row-owner"} for key in pending)
    or not all(arm["compilation_status"] == "observed_stable" for arm in report["arms"].values())
):
    raise RuntimeError("The direct campaign needs recovery outside row-owner")
print("\n".join(pending))
PY
)

source /etc/profile.d/wm.sh
set -u
test "$(cat /etc/volcano/VC_MPIMASTER_NUM)" = 1
source_hash=$("$python" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["tracked_and_untracked_sha256"])' "$reviewed/source.json")
pins=(
    --original-root "$original"
    --candidate-root "$candidate"
    --reference "$workspace/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2"
    --candidate-head e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f
    --candidate-source-sha256 "$source_hash"
)
recovery="$suite/row-owner-recovery"
mkdir "$recovery"
row_owner_failed=0
while IFS= read -r key; do
    ep="${key%%-*}"
    printf 'START %s conditional row-owner comparison\n' "$ep"
    if "$python" -Bu "$owner/row_owner_on_layer0.py" matrix "$recovery" \
        --suite per-fix --case "$ep-layer0-row-owner" "${pins[@]}"; then
        printf 'PASS %s conditional row-owner arm\n' "$ep"
    else
        printf 'FAIL %s conditional row-owner arm\n' "$ep"
        row_owner_failed=1
    fi
done <<< "$pending"

validation_failed=0
if ! bash "$reviewed/finish_native_after_recovery.sh" conditional > "$suite/finish-native-recovery.log" 2>&1; then
    validation_failed=1
fi
"$python" -B - "$suite/final/results.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    report = json.load(stream)
if (
    report["pending"] or set(report["arms"]) != {"ep8-a", "ep8-all", "ep4-a", "ep4-all"}
    or len(report["comparisons"]) != 2
    or not all(pair["measured_compilation_stable"] for pair in report["comparisons"])
):
    raise RuntimeError("The four final 10+10 runs are incomplete")
PY

export TMPDIR="$(mktemp -d "$candidate/.tmp/dg-x.XXXXXX")"
export TMP="$TMPDIR" TEMP="$TMPDIR"
out="$owner/artifacts/raw/fixed-shape-route-001"
if [[ ! -d "$out" ]]; then
    mkdir "$out"
    if ! bash "$tools/run_row_owner_route_replay.sh" "$original" "$capture" "$out" \
        > "$suite/fixed-shape-route-001.log" 2>&1; then
        validation_failed=1
    fi
fi
out="$owner/artifacts/raw/compact-coalescing-001"
if [[ ! -d "$out" ]]; then
    mkdir "$out"
    census="$raw/loader-verification/data002-full/depth0"
    extra=()
    if [[ -r "$census/rank0.json" && -r "$raw/data-002/train/index.json" ]]; then
        extra=("$census" "$raw/data-002/train")
    fi
    if ! bash "$tools/run_compact_coalescing_replay.sh" "$original" "$capture" "$out" \
        "${extra[@]}" > "$suite/compact-coalescing-001.log" 2>&1; then
        validation_failed=1
    fi
fi
exit "$((row_owner_failed || validation_failed))"
