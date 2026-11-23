#!/usr/bin/env bash
# Run the fourteen Linear-fixes acceptance arms serially on the existing SR008 allocation.
set -Eeuo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
campaign=$(realpath -- "${1:?Expected the prepared campaign directory}")
[[ $campaign == "$here/artifacts/raw/"* && -f $campaign/inputs.json ]]
[[ ! -e $campaign/matrix.tsv ]]
source /etc/profile.d/wm.sh
python=/opt/torchtitan-venv/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1
trap 'status=$?; printf "%s\n" "$status" > "$campaign/exit-code.txt.tmp"; mv "$campaign/exit-code.txt.tmp" "$campaign/exit-code.txt"' EXIT

"$python" -Bu "$here/acceptance.py" verify-ready "$campaign"
entries=$("$python" -B "$here/acceptance.py" order)
printf '%s\n' "$$" > "$campaign/driver.pid"
printf 'ep\tarm\tstatus\ttimestamp_utc\n' > "$campaign/matrix.tsv"
printf 'host=%s python=%s\n' "$(hostname)" "$python"
sha256sum "$here/acceptance.py" "$here/acceptance.sh" "$here/activation.py" "$campaign/inputs.json"

result=0
while IFS=: read -r ep arm; do
    preflight="$campaign/ep$ep-$arm.gpus.csv"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader > "$preflight"
    [[ $(wc -l < "$preflight") -eq 8 ]]
    compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ -n $compute_apps ]] || ! awk -F, '$3 != " 0 MiB" { exit 1 }' "$preflight"; then
        printf 'GPU preflight found active GPU memory or compute for EP%s %s\n' "$ep" "$arm" >&2
        exit 1
    fi
    printf '%s\t%s\tSTART\t%s\n' "$ep" "$arm" "$(date -u +%FT%TZ)" >> "$campaign/matrix.tsv"
    log="$campaign/ep$ep-$arm.runner.log"
    status=PASS
    if ! "$python" -Bu "$here/acceptance.py" run "$campaign" "$ep" "$arm" > "$log" 2>&1 ||
       ! "$python" -Bu "$here/acceptance.py" summarize "$campaign" "$ep" "$arm" >> "$log" 2>&1; then
        status=FAIL
        result=1
        tail -n 15 "$log" >&2
    fi
    printf '%s\t%s\t%s\t%s\n' "$ep" "$arm" "$status" "$(date -u +%FT%TZ)" >> "$campaign/matrix.tsv"
done <<< "$entries"
"$python" -Bu "$here/acceptance.py" matrix "$campaign" || result=1
exit "$result"
