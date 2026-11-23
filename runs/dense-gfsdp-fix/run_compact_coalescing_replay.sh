#!/usr/bin/env bash
# Run the isolated source-op comparison on the idle, single-node H100 allocation.
set -Eeuo pipefail

if [[ $# != 3 && $# != 5 ]]; then
    printf 'usage: %s ORIGINAL_A CAPTURE EMPTY_OUTPUT [DEPTH0_CENSUS MDS_TRAIN]\n' "$0" >&2
    exit 2
fi
source_root=$(realpath -- "$1")
input_root=$(realpath -- "$2")
output_root=$(realpath -- "$3")
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python=/opt/torchtitan-venv/bin/python3.12
[[ -d "$output_root" && -z $(ls -A -- "$output_root") ]]
trap 'rc=$?; printf "%s\n" "$rc" > "$output_root/exit-code.tmp"; mv "$output_root/exit-code.tmp" "$output_root/exit-code"' EXIT
printf '%s\n' "$$" > "$output_root/driver.pid"

original_a=e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f
[[ $(git -C "$source_root" rev-parse HEAD) == "$original_a" ]]
[[ -z $(git -C "$source_root" status --porcelain -- gift runs/production-data-abcd/component_fixtures.py) ]]
[[ $(cat /etc/volcano/VC_MPIMASTER_NUM) == 1 ]]
low_flags=()
if [[ $# == 5 ]]; then
    census=$(realpath -- "$4")
    data_root=$(realpath -- "$5")
    for rank in {0..7}; do
        test -r "$census/rank$rank.json"
    done
    test -r "$data_root/index.json"
    low_flags=(--low-census "$census" --data-root "$data_root")
fi

source /etc/profile.d/wm.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$source_root/gift/contrib/flash-linear-attention:$source_root/gift/contrib/flash-qla:$source_root/gift"
export ROW_OWNER_REPLAY_SOURCE_ROOT="$source_root"
cd "$source_root"
sha256sum -- "$experiment/"{compact_coalescing_replay.py,run_compact_coalescing_replay.sh,embedding-replay-result.json,row_owner_route_experiment.py,row_owner_route_numeric.py,row_owner_route_summary.py} \
    runs/production-data-abcd/component_fixtures.py \
    gift/torchtitan/distributed/gfsdp/row_owner_embedding.py > "$output_root/sources.sha256"
"$python" -Bu "$experiment/row_owner_route_summary.py" \
    "$input_root" "$experiment/embedding-replay-result.json" --fixture-only

nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader > "$output_root/gpus.csv"
[[ $(wc -l < "$output_root/gpus.csv") -eq 8 ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
awk -F', ' '$2 !~ /H100/ || $3 != "0 MiB" { exit 1 }' "$output_root/gpus.csv"
command=("$python" -Bu -m torch.distributed.run --standalone --nproc_per_node=8
    "$experiment/compact_coalescing_replay.py" --execute --profile
    --high-fixture "$input_root" --output "$output_root" "${low_flags[@]}")
printf '%q ' "${command[@]}" > "$output_root/command"
printf '\n' >> "$output_root/command"
timeout --signal=TERM --kill-after=45s 900s "${command[@]}" > "$output_root/launcher.log" 2>&1
sha256sum --check --status "$output_root/sources.sha256"
test -s "$output_root/parity.json"
test -s "$output_root/summary.json"
for rank in {0..7}; do
    test -s "$output_root/rank$rank.json"
done
