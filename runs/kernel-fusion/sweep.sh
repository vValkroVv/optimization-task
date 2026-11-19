#!/usr/bin/env bash
set -Eeuo pipefail

run_id=${1:?Supply a fresh sweep ID}
plan=${2:?Supply the tab-separated variant/source/component plan}
predecessor=${3:?Supply the prerequisite campaign directory}
[[ $# -eq 3 && $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
state=$PWD/artifacts/raw/$run_id
mkdir -- "$state"
exec >"$state/sweep.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$state/exit-code.txt"' EXIT
printf '%s\n' "$$" > "$state/pid"
cp -- "$plan" "$state/plan.tsv"

timeout 3600 bash -c '
    while [[ ! -f $1/exit-code.txt ]]; do sleep 120; done
    [[ $(<"$1/exit-code.txt") == 0 ]]
' bash "$predecessor"

status=0
while IFS=$'\t' read -r variant source receipt; do
    attempt="$run_id-$variant"
    printf 'Starting %s at %s\n' "$variant" "$(date -u +%FT%TZ)"
    if bash launch.sh "$variant" "$attempt" "$source" "$receipt"; then
        result=0
    else
        result=$?
        status=1
    fi
    printf '%s\t%s\t%s\n' "$variant" "$attempt" "$result" >> "$state/results.tsv"
done < "$state/plan.tsv"
exit "$status"
