#!/usr/bin/env bash
# Run standalone replays on one idle H100 or one fixed native arm on eight.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
kind=${1:?expected components or native}
tag=${2:?supply a fresh campaign ID}
[[ $tag =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
state=$PWD/runs/backward-components/artifacts/raw/$tag
mkdir -p "$(dirname "$state")"
mkdir "$state"
exec > "$state/campaign.log" 2>&1
trap 'status=$?; echo "$status" > "$state/exit-code.txt"' EXIT
echo "$$" > "$state/pid"
source /etc/profile.d/wm.sh
export PYTHONDONTWRITEBYTECODE=1
python=/opt/torchtitan-venv/bin/python3.12
build=$PWD/runs/kernel-fusion/artifacts/raw/efsdp-direct-dw-ep8-003-campaign/build
current_rev=$(git rev-parse HEAD)
echo "revision=$current_rev"
hostname
idle_node() {
    [[ $(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c H100) -eq 8 ]]
    [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
}
idle_node
case "$kind" in
    components)
        fixture=${3:?supply the saved native rank-zero fixture}
        test -s "$fixture"
        export CUDA_VISIBLE_DEVICES=0
        export PYTHONPATH="$PWD/gift${PYTHONPATH:+:$PYTHONPATH}"
        export CUDACHERRY_DISABLE_JIT=1
        export TRITON_CACHE_DIR="$state/triton" TORCHINDUCTOR_CACHE_DIR="$state/inductor"
        mkdir "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
        failed=0
        (
            cd gift
            "$python" -m pytest -q tests/unit_tests/test_mtp.py -k test_folded_ce_opt_in_preserves_cpu_fallback
        ) > "$state/cpu-test.log" 2>&1 || { echo "$?" > "$state/cpu-test-exit.txt"; failed=1; }
        for variant in base folded; do
            "$python" -Bu runs/backward-components/folded_ce.py --fixture "$fixture" \
                --mode benchmark --variant "$variant" --trace --output "$state/ce-$variant.json" \
                > "$state/ce-$variant.log" 2>&1 || failed=1
        done
        "$python" -Bu runs/backward-components/folded_ce.py --fixture "$fixture" \
            --mode numeric --output "$state/ce-numeric.json" > "$state/ce-numeric.log" 2>&1 || failed=1
        "$python" -Bu runs/backward-components/row2col_replay.py --fixture "$fixture" \
            --build-root "$build" --output "$state/row2col" > "$state/row2col.log" 2>&1 || failed=1
        exit "$failed"
        ;;
    native)
        shift 2
        (( $# > 0 ))
        for arm in "$@"; do
            case "$arm" in
                control) extra=(--backward-optimization unfolded-ce) ;;
                folded-ce) extra=(--backward-optimization "$arm") ;;
                *) echo "Unknown backward arm: $arm"; exit 2 ;;
            esac
            for mode in timing trace; do
                idle_node
                test "$(git rev-parse HEAD)" = "$current_rev"
                echo "Starting $arm $mode at $(date -u +%FT%TZ)"
                "$python" -Bu runs/kernel-fusion/run.py production "$mode" "$tag-$arm-$mode" \
                    --source-revision "$current_rev" --direct-dw-build "$build" \
                    --gfsdp-optimization retain-tail-weights --gfsdp-optimization fp8-expert-weights \
                    --gfsdp-optimization direct-expert-dw "${extra[@]}"
            done
        done
        ;;
    *) exit 2 ;;
esac
