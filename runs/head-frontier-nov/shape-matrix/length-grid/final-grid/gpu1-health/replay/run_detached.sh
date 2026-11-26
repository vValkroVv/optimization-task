#!/usr/bin/env bash
set -u
set -o noclobber
cd -- "$(dirname -- "$0")" || exit 1
printf '%s\n' "$$" > replay.pid || exit 1
date -u +%FT%T.%NZ > replay.started
{
    source /etc/profile.d/wm.sh && /opt/torchtitan-venv/bin/python -Bu run_replay.py
} > replay.log 2>&1
status=$?
date -u +%FT%T.%NZ > replay.finished
printf '%s\n' "$status" > replay.exit
exit "$status"
