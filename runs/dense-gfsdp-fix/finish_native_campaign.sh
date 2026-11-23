#!/usr/bin/env bash
set -eEo pipefail
workspace=/workspace-SR008.nfs2/vkropotin
original="$workspace/dg-a"
candidate="$workspace/dg-fix"
owner="$candidate/runs/dense-gfsdp-fix"
reviewed="$owner/artifacts/reviewed-port-001"
campaign="$owner/artifacts/raw/native-001"
py=/opt/torchtitan-venv/bin/python3.12
inputs="$workspace/optimization-task/runs/production-data-abcd/artifacts/raw/data-002/train"
mode="${1:-direct}"
case "$mode" in
    direct) receipt="$campaign/finish-native.exit" ;;
    conditional) receipt="$campaign/finish-native-recovery.exit" ;;
    *) exit 2 ;;
esac
trap 'status=$?; printf "%s\n" "$status" > "$receipt"; exit "$status"' EXIT

test -s "$campaign/queue-per-fix.pid"
while [[ ! -s "$campaign/queue-per-fix.exit" ]]; do
    kill -0 "$(cat "$campaign/queue-per-fix.pid")"
    (( SECONDS < 81000 ))
    sleep 120
done
if [[ "$mode" == direct ]]; then
    test "$(cat "$campaign/queue-per-fix.exit")" = 0
fi
"$py" -B - "$campaign/per-fix/results.json" "$mode" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    result = json.load(stream)
pending = set(result["pending"])
mode = sys.argv[2]
if mode == "direct":
    permitted = not pending
else:
    permitted = pending <= {"ep8-row-owner", "ep4-row-owner"}
verified = all(
    arm["compilation_status"] == "observed_stable" for arm in result["arms"].values()
)
if (not permitted or not verified or len(result["arms"]) + len(pending) != 10
        or len(result["comparisons"]) + len(pending) != 8):
    raise RuntimeError("Per-fix timing campaign is incomplete")
PY

source /etc/profile.d/wm.sh
set -u
test "$(cat /etc/volcano/VC_MPIMASTER_NUM)" = 1
gpu_preflight() {
    local snapshot="$1"
    nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv,noheader > "$snapshot"
    test "$(wc -l < "$snapshot")" = 8
    awk -F', ' '$3 !~ /H100/ || $4 != "0 MiB" { exit 1 }' "$snapshot"
    test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
}
stamp() {
    printf '%s %s\n' "$(date -u +'%FT%TZ')" "$*"
}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

mkdir "$campaign/validation"
checks="$campaign/validation"
gpu_preflight "$checks/transition-tests.gpus.csv"
stamp "START transition source tests"
(
    cd "$candidate/gift"
    export CUDA_VISIBLE_DEVICES=0,1
    export PYTHONPATH="$PWD/contrib/flash-linear-attention:$PWD/contrib/flash-qla:$PWD"
    timeout --signal=TERM --kill-after=45s 1200s "$py" -B -m pytest -q -o addopts= \
        tests/unit_tests/test_ffn_overlap.py \
        tests/unit_tests/test_gfsdp_buffer_order.py \
        tests/unit_tests/test_gfsdp_ffn_overlap.py --maxfail=1
) > "$checks/transition-tests.log" 2>&1
stamp "PASS transition source tests"

gpu_preflight "$checks/embedding-exact.gpus.csv"
stamp "START Original A versus row-owner numerical check"
(
    cd "$original"
    export PYTHONPATH="$PWD/gift"
    timeout --signal=TERM --kill-after=45s 900s "$py" -Bu -m torch.distributed.run \
        --standalone --nproc_per_node=8 \
        runs/ffn-row-owner/embedding_numeric.py --execute --fixture "$inputs" \
        --seed 2048 --output "$checks/embedding-exact.json"
) > "$checks/embedding-exact.log" 2>&1
"$py" -B - "$checks/embedding-exact.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    report = json.load(stream)
summary = report["summary"]
if not summary["all_outputs_exactly_equal"] or not summary["all_gradient_checks_exactly_equal"]:
    raise RuntimeError("Original A and row-owner replay has unresolved numerical differences")
PY
stamp "PASS Original A versus row-owner numerical check"

cd "$candidate"
trace_failed=0
for arm in mtp mtp-transition all; do
    stamp "START EP8 trace $arm"
    if bash "$reviewed/launch.sh" trace 8 "$arm" > "$checks/trace-$arm.log" 2>&1; then
        stamp "PASS EP8 trace $arm"
    else
        stamp "FAIL EP8 trace $arm"
        trace_failed=1
    fi
done
if (( trace_failed == 0 )); then
    trace_receipts="$campaign/diagnostic"
    if ! PYTHONPATH="$owner" "$py" -B "$owner/artifacts/followup-tools-001/analyze_diagnostic.py" \
        --receipt "$trace_receipts/ep8-mtp.json" \
        --receipt "$trace_receipts/ep8-mtp-transition.json" \
        --receipt "$trace_receipts/ep8-all.json" \
        --output "$checks/mtp-trace-analysis.json" > "$checks/mtp-trace-analysis.log" 2>&1; then
        stamp "FAIL trace analysis"
        trace_failed=1
    fi
fi

stamp "START final four-arm matrix"
bash "$reviewed/launch.sh" final > "$campaign/final.log" 2>&1
stamp "PASS final four-arm matrix"
exit "$trace_failed"
