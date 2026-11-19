#!/usr/bin/env bash
# Run the selected component-qualified arms once each against the same four-fix control.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
tag=${1:?expected a fresh campaign name}
shift
[[ $tag =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
for arm in "$@"; do [[ $arm =~ ^[a-z0-9-]+$ && $arm != baseline ]]; done
state=$PWD/runs/fusion-followup/artifacts/raw/$tag
mkdir -p "$(dirname "$state")"
mkdir "$state"
exec > "$state/launch.log" 2>&1
trap 'echo "$?" > "$state/exit-code.txt"' EXIT
echo "$$" > "$state/pid"
revision=$(git rev-parse HEAD)
echo "$revision" > "$state/revision"
fixes=(hidden-z-loss gdn-direct-backward direct-dhidden staged-head-dw)
options=()
for fix in "${fixes[@]}"; do options+=(--fusion-fix "$fix"); done
for arm in baseline "$@"; do
    [[ $(git rev-parse HEAD) == "$revision" ]]
    args=("${options[@]}")
    if [[ $arm != baseline ]]; then args+=(--followup-kernel "$arm"); fi
    run="$tag-$arm"
    echo "Starting $run at $(date -u +%FT%TZ)"
    bash runs/fusion-pack/native.sh timing "$run" "${args[@]}"
    raw="runs/kernel-fusion/artifacts/raw/$run"
    /opt/torchtitan-venv/bin/python3.12 -B runs/fusion-pack/receipt.py "$raw" "${fixes[@]}" > "$state/$arm.json"
    if [[ $arm != baseline ]]; then
        for rank in {0..7}; do
            grep -Fq "[default$rank]:kernel_fusion followup_kernel=$arm" "$raw/logs/launcher.log"
        done
    fi
    echo "Completed $run at $(date -u +%FT%TZ)"
done
