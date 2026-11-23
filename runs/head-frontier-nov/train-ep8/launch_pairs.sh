#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
personal=/workspace-SR008.nfs2/vkropotin
python=/opt/torchtitan-venv/bin/python3.12
series=${1:?expected a series ID}
shift
if [[ $# -eq 0 ]]; then
    set -- a:2 b:2 b:8 a:8 a:32 b:32
fi

printf '%s host=%s\n' "$(date -u +%FT%TZ)" "$(hostname)"
for cell in "$@"; do
    chunks=${cell#*:}
    case "${cell%%:*}" in
        a) arm=a; runner=run_baseline.py; checkout=a10-cute-001 ;;
        b) arm=b; runner=run_arm.py; checkout=lm10-c1 ;;
        *) printf 'Unknown arm: %s\n' "$cell" >&2; exit 2 ;;
    esac
    if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
        printf 'GPU compute processes present before %s\n' "$cell" >&2
        exit 3
    fi

    cd "$personal/$checkout"
    run_id="lhce-${series}-${arm}-c${chunks}"
    if [[ -e "$root/$run_id.log" || -e "$root/$run_id.exit" ]]; then
        printf 'Run already exists: %s\n' "$run_id" >&2
        exit 2
    fi
    printf '%s starting %s\n' "$(date -u +%FT%TZ)" "$run_id"
    status=0
    "$python" -Bu "$root/$runner" "$chunks" "$run_id" \
        > "$root/$run_id.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$root/$run_id.exit"
    if [[ "$status" -ne 0 ]]; then
        exit "$status"
    fi
    printf '%s completed %s\n' "$(date -u +%FT%TZ)" "$run_id"
done
