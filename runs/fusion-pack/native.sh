#!/usr/bin/env bash
# One detached native launch with persistent evidence under the task checkout.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
mode=${1:?expected capture or timing}
tag=${2:?expected a fresh run ID}
shift 2
[[ $mode == capture || $mode == timing ]]
[[ $tag =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
state=$PWD/runs/fusion-pack/artifacts/raw/$tag
mkdir -p "$(dirname "$state")"
mkdir "$state"
exec > "$state/launch.log" 2>&1
trap 'echo "$?" > "$state/exit-code.txt"' EXIT
echo "$$" > "$state/pid"
source /etc/profile.d/wm.sh
export PYTHONDONTWRITEBYTECODE=1
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c H100) -eq 8 ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
build=runs/kernel-fusion/artifacts/raw/efsdp-direct-dw-ep8-003-campaign/build
/opt/torchtitan-venv/bin/python3.12 -Bu runs/kernel-fusion/run.py production "$mode" "$tag" \
    --source-revision "$(git rev-parse HEAD)" --direct-dw-build "$build" \
    --gfsdp-optimization retain-tail-weights --gfsdp-optimization fp8-expert-weights \
    --gfsdp-optimization direct-expert-dw "$@"
