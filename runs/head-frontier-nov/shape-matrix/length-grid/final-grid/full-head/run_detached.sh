#!/usr/bin/env bash
set -u
set -o noclobber

run_tag=$1
shift
if [[ ! $run_tag =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    exit 2
fi
cd -- "$(dirname -- "$0")" || exit 1
mkdir -p control
printf '%s\n' "$$" > "control/$run_tag.pid" || exit 1
date -u +%FT%TZ > "control/$run_tag.started"
{
    source /etc/profile.d/wm.sh && /opt/torchtitan-venv/bin/python -Bu run_grid.py "$@"
} > "control/$run_tag.log" 2>&1
status=$?
date -u +%FT%TZ > "control/$run_tag.finished"
printf '%s\n' "$status" > "control/$run_tag.exit"
exit "$status"
