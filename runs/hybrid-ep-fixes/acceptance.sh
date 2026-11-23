#!/usr/bin/env bash
# Run the eight HybridEP comparisons serially on the existing SR008 allocation.
set -Eeuo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
campaign=$(realpath -- "${1:?Expected the prepared campaign directory}")
[[ $campaign == "$here/artifacts/raw/"* && -f $campaign/candidate/routing_metadata.py ]]
[[ ! -e $campaign/matrix.tsv && -s $campaign/dry/ep8-baseline.verified.json &&
   -s $campaign/dry/ep4-both.verified.json ]]
source /etc/profile.d/wm.sh
python=$(command -v python)
export PYTHONDONTWRITEBYTECODE=1
trap 'status=$?; printf "%s\n" "$status" > "$campaign/exit-code.txt.tmp"; mv "$campaign/exit-code.txt.tmp" "$campaign/exit-code.txt"' EXIT

printf '%s\n' "$$" > "$campaign/driver.pid"
printf 'ep\tarm\tstatus\ttimestamp_utc\n' > "$campaign/matrix.tsv"
printf 'host=%s python=%s\n' "$(hostname)" "$python"
sha256sum "$here/acceptance.py" "$here/acceptance.sh" "$campaign/candidate/routing_metadata.py"

result=0
for entry in 8:baseline 4:baseline 8:routing 4:selector 8:selector 4:routing 8:both 4:both; do
    ep=${entry%:*}
    arm=${entry#*:}
    preflight="$campaign/ep$ep-$arm.gpus.csv"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader > "$preflight"
    [[ $(wc -l < "$preflight") -eq 8 ]]
    if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] ||
        ! awk -F, '$3 != " 0 MiB" { exit 1 }' "$preflight"; then
        printf 'GPU preflight found active GPU memory or compute for EP%s %s\n' "$ep" "$arm" >&2
        exit 1
    fi
    printf '%s\t%s\tSTART\t%s\n' "$ep" "$arm" "$(date -u +%FT%TZ)" >> "$campaign/matrix.tsv"
    attempt_log="$campaign/ep$ep-$arm.runner.log"
    status=PASS
    if ! "$python" -Bu "$here/acceptance.py" run "$campaign" "$ep" "$arm" > "$attempt_log" 2>&1 ||
       ! "$python" -Bu "$here/acceptance.py" summarize "$campaign" "$ep" "$arm" >> "$attempt_log" 2>&1; then
        status=FAIL
        result=1
        tail -n 12 "$attempt_log" >&2
    fi
    printf '%s\t%s\t%s\t%s\n' "$ep" "$arm" "$status" "$(date -u +%FT%TZ)" >> "$campaign/matrix.tsv"
done
exit "$result"
