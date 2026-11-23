#!/usr/bin/env bash
set -euo pipefail

checkout=$1
campaign=$2
candidate=$3
launcher=$4
previous=$5
trap 'printf "%s\n" "$?" > "$launcher/exit"' EXIT

previous_pid=$(<"$previous/pid")
deadline=$((SECONDS + 10800))
printf 'Waiting for task launcher PID %s in %s\n' "$previous_pid" "$previous"
while [[ ! -f "$previous/exit" ]]; do
    process_args=$(ps -ww -p "$previous_pid" -o args= || true)
    if [[ -f "$previous/exit" ]]; then
        break
    fi
    if [[ "$process_args" != *"/run_acceptance.sh "*" $previous" ]] || (( SECONDS >= deadline )); then
        printf 'The preceding launcher is missing or exceeded its wait deadline\n' >&2
        exit 1
    fi
    sleep 120
done
if [[ $(<"$previous/exit") != 0 ]]; then
    printf 'The preceding acceptance campaign failed. See %s\n' "$previous/run.log" >&2
    exit 1
fi

exec bash "$checkout/runs/shk-fix/followup/run_acceptance.sh" \
    "$checkout" "$campaign" "$candidate" "$launcher"
