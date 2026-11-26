#!/usr/bin/env bash
set -u
set -o noclobber

tag=${1:?Specify a new control tag}
if [[ ! $tag =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    exit 2
fi
cd -- "$(dirname -- "$0")" || exit 1
mkdir -p control
printf '%s\n' "$$" > "control/$tag.pid" || exit 1
date -u +%FT%T.%NZ > "control/$tag.started"
{
    source /etc/profile.d/wm.sh && /opt/torchtitan-venv/bin/python -Bu run_matrix.py
} > "control/$tag.log" 2>&1
status=$?
date -u +%FT%T.%NZ > "control/$tag.finished"
printf '%s\n' "$status" > "control/$tag.exit"
exit "$status"
