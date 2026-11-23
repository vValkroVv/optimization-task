#!/usr/bin/env bash
set -euo pipefail
run_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python=/opt/torchtitan-venv/bin/python3.12

if [[ $# -lt 2 ]]; then
    printf 'Usage: launch.sh <series> <chunks>...\n' >&2
    exit 2
fi
series=$1
shift
cd /workspace-SR008.nfs2/vkropotin/lm10-c1
for chunks in "$@"; do
    if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
        printf 'GPU compute processes present before C%s\n' "$chunks" >&2
        exit 3
    fi
    run_id="lhce-${series}-c${chunks}"
    printf '%s starting %s\n' "$(date -u +%FT%TZ)" "$run_id"
    status=0
    "$python" -Bu "$run_root/run_arm.py" "$chunks" "$run_id" \
        > "$run_root/$run_id.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$run_root/$run_id.exit"
    if [[ "$status" -ne 0 ]]; then
        exit "$status"
    fi
    printf '%s completed %s\n' "$(date -u +%FT%TZ)" "$run_id"
done
