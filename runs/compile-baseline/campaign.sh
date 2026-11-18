#!/usr/bin/env bash
set -Eeuo pipefail

run_id=${1:?Supply a fresh campaign ID}
[[ $# == 1 && $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p artifacts/raw
state=$PWD/artifacts/raw/$run_id
mkdir -- "$state"
exec >"$state/campaign.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$state/exit-code.txt"' EXIT

source /etc/profile.d/wm.sh
export PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$$" > "$state/pid"
git rev-parse HEAD
hostname
python=/opt/torchtitan-venv/bin/python3.12
for mode in compile no_compile; do
    printf 'Starting %s at %s\n' "$mode" "$(date -u +%FT%TZ)"
    "$python" -B run.py "$mode" "$run_id-$mode"
done
