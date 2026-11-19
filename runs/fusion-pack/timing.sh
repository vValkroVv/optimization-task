#!/usr/bin/env bash
# All arms share the folded-CE production baseline and the same source revision.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
tag=${1:?expected a fresh campaign ID}
shift
(( $# > 0 ))
[[ $tag =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
state=$PWD/runs/fusion-pack/artifacts/raw/$tag
mkdir -p "$(dirname "$state")"
mkdir "$state"
exec > "$state/campaign.log" 2>&1
trap 'echo "$?" > "$state/exit-code.txt"' EXIT
echo "$$" > "$state/pid"
rev=$(git rev-parse HEAD)
echo "$rev" > "$state/revision"
while (( $# )); do
    arm=$1
    shift
    case "$arm" in
        baseline) fixes=() ;;
        pack) fixes=(hidden-z-loss gdn-direct-backward direct-dhidden) ;;
        head) fixes=(staged-head-dw) ;;
        *) echo "Unknown timing arm $arm"; exit 2 ;;
    esac
    opts=()
    for fix in "${fixes[@]}"; do opts+=(--fusion-fix "$fix"); done
    [[ $(git rev-parse HEAD) == "$rev" ]]
    run="$tag-$arm"
    echo "Starting $arm at $(date -u +%FT%TZ)"
    bash runs/fusion-pack/native.sh timing "$run" "${opts[@]}"
    /opt/torchtitan-venv/bin/python3.12 -B runs/fusion-pack/receipt.py \
        "runs/kernel-fusion/artifacts/raw/$run" "${fixes[@]}" > "$state/$arm.json"
    echo "Measured $arm at $(date -u +%FT%TZ)"
done
