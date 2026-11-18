#!/usr/bin/env bash
set -Eeuo pipefail

run_id=${1:?Supply a new run ID}
[[ $# == 1 && $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
task_root=$PWD
script_dir=$task_root/runs/capacity-selector
raw=$script_dir/artifacts/raw
cache=$task_root/runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache
python=/opt/torchtitan-venv/bin/python3.12
source /etc/profile.d/wm.sh
export PYTHONDONTWRITEBYTECODE=1

mkdir -p -- "$raw"
state=$raw/$run_id-campaign
mkdir -- "$state"
exec > >(tee -a "$state/campaign.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$state/exit-code.txt"' EXIT
printf '%s\n' "$$" > "$state/pid"

git rev-parse HEAD
hostname
echo "Capturing real routing inputs"
"$python" -B "$script_dir/native.py" capture --run-id "$run_id-capture"

echo "Checking and measuring isolated capacity preparation"
env PYTHONPATH="$task_root/gift" PYTORCH_ALLOC_CONF=expandable_segments:True \
  TORCHINDUCTOR_CUDAGRAPHS=0 TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 \
  "$python" -B -m torch.distributed.run --standalone --nproc-per-node=8 \
  "$script_dir/component.py" \
  --fixture-dir "$raw/$run_id-capture/routing-fixtures" \
  --output "$raw/$run_id-component" --cache-root "$cache" \
  --capacity-source "$task_root/runs/capacity-group-allreduce/capacity.original.py"

for mode in control candidate trace; do
  echo "Running $mode"
  "$python" -B "$script_dir/native.py" "$mode" --run-id "$run_id-$mode"
done
