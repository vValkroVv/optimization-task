#!/usr/bin/env bash
# Run one matched pair per topology, reversing arm order on EP4.
set -Eeuo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
tag=${1:?Supply a fresh campaign ID}
[[ $tag =~ ^pf[0-9]{3}$ ]]
raw="$here/artifacts/raw"
campaign="$raw/$tag"
mkdir -- "$campaign"
trap 'result=$?; printf "%s\n" "$result" > "$campaign/exit-code.txt.tmp"; mv "$campaign/exit-code.txt.tmp" "$campaign/exit-code.txt"' EXIT
printf '%s\n' "$$" > "$campaign/driver.pid"
source /etc/profile.d/wm.sh
python=/opt/torchtitan-venv/bin/python3.12

printf 'ep\tdepth\tstatus\ttimestamp_utc\n' > "$campaign/progress.tsv"
sha256sum "$here/run.py" "$here/matrix.sh" > "$campaign/source.sha256"
for entry in '8 0 off' '8 32 on' '4 32 on' '4 0 off'; do
    read -r ep depth arm <<< "$entry"
    id="$tag-e$ep-$arm"
    nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv,noheader > "$campaign/$id.gpus.csv"
    [[ $(wc -l < "$campaign/$id.gpus.csv") -eq 8 ]]
    awk -F, '$2 !~ /H100/ || $4 != " 0 MiB" {exit 1}' "$campaign/$id.gpus.csv"
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    [[ -z $apps ]]
    printf '%s\t%s\tSTART\t%s\n' "$ep" "$depth" "$(date -u +%FT%TZ)" >> "$campaign/progress.tsv"
    if "$python" -Bu "$here/run.py" "$ep" "$depth" "$id" > "$campaign/$id.log" 2>&1; then
        status=PASS
    else
        printf '%s\t%s\tFAIL\t%s\n' "$ep" "$depth" "$(date -u +%FT%TZ)" >> "$campaign/progress.tsv"
        tail -n 15 "$campaign/$id.log" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\t%s\n' "$ep" "$depth" "$status" "$(date -u +%FT%TZ)" >> "$campaign/progress.tsv"
done
