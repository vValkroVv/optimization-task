#!/usr/bin/env bash
set -Eeuo pipefail

campaign=${1:?Supply a fresh campaign ID}
candidate=${2:?Supply the isolated candidate worktree}
[[ $# -eq 2 && $campaign =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p artifacts/raw
state=$PWD/artifacts/raw/$campaign
mkdir -- "$state"
exec >"$state/campaign.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$state/exit-code.txt"' EXIT
printf '%s\n' "$$" > "$state/pid"

source /etc/profile.d/wm.sh
export PYTHONDONTWRITEBYTECODE=1
python=/opt/torchtitan-venv/bin/python3.12
sha256sum run.py run-140.sh
hostname

run_arm() {
    local arm=$1
    shift
    if [[ $(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c 'H100') -ne 8 ||
          -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
        printf '%s\n' "Expected eight H100s with no GPU compute processes."
        return 1
    fi
    printf 'Starting %s at %s\n' "$arm" "$(date -u +%FT%TZ)"
    "$python" -B run.py "$arm" timing-140 "$campaign-$arm" "$@"
}

run_arm baseline
(cd -- "$candidate" && test -s "$state/combined-source.sha256" &&
    sha256sum --check "$state/combined-source.sha256")
run_arm combined-five --candidate-source "$candidate"
