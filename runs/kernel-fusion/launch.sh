#!/usr/bin/env bash
set -Eeuo pipefail

variant=${1:?Supply the variant}
run_id=${2:?Supply a fresh run ID stem}
shift 2
[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
if [[ $variant == baseline ]]; then
    [[ $# -eq 0 ]]
else
    [[ $# -eq 2 && -s $2 ]]
    printf 'Reference component result %s\n' "$2"
    set -- --candidate-source "$1"
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p artifacts/raw
state=$PWD/artifacts/raw/$run_id
mkdir -- "$state"
exec >"$state/campaign.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$state/exit-code.txt"' EXIT

source /etc/profile.d/wm.sh
export PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$$" > "$state/pid"
hostname
nvidia-smi --query-gpu=index,name,memory.used --format=csv
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 8 ||
      -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
    printf '%s\n' "Expected eight GPUs and no active GPU compute processes."
    exit 1
fi

python=/opt/torchtitan-venv/bin/python3.12
for mode in timing trace; do
    printf 'Starting %s/%s at %s\n' "$variant" "$mode" "$(date -u +%FT%TZ)"
    "$python" -B run.py "$variant" "$mode" "$run_id-$mode" "$@"
done
