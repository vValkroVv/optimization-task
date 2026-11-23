#!/usr/bin/env bash
# Run one isolated A1/B1/A2/B2 replay on the idle single-node H100 allocation.
set -Eeuo pipefail

if [[ $# != 4 ]]; then
    printf 'usage: %s ORIGINAL_A CAPTURE EMPTY_OUTPUT ARCHIVED_KERNEL_SOURCE\n' "$0" >&2
    exit 2
fi
source_root=$(realpath -- "$1")
input_root=$(realpath -- "$2")
output_root=$(realpath -- "$3")
archive_kernel=$(realpath -- "$4")
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python=/opt/torchtitan-venv/bin/python3.12
[[ -d "$output_root" && -z $(ls -A -- "$output_root") ]]
trap 'rc=$?; printf "%s\n" "$rc" > "$output_root/exit-code.tmp"; mv "$output_root/exit-code.tmp" "$output_root/exit-code"' EXIT
printf '%s\n' "$$" > "$output_root/driver.pid"

[[ $(git -C "$source_root" rev-parse HEAD) == e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f ]]
[[ -z $(git -C "$source_root" status --porcelain -- gift runs/production-data-abcd/component_fixtures.py) ]]
[[ $(cat /etc/volcano/VC_MPIMASTER_NUM) == 1 ]]
cp -- "$archive_kernel" "$output_root/archived_owner_add_kernels.py"
archive_kernel="$output_root/archived_owner_add_kernels.py"
printf 'a8474c8da675e48e6a36d6fbbe566315c8a3420a11b30f18576ee4edba43fc1c  %s\n' "$archive_kernel" | sha256sum --check --status

source /etc/profile.d/wm.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$source_root/gift/contrib/flash-linear-attention:$source_root/gift/contrib/flash-qla:$source_root/gift"
export ROW_OWNER_REPLAY_SOURCE_ROOT="$source_root"
cd "$source_root"
artifacts=(
    "$experiment/"{owner_add_replay.py,owner_add_ftz_kernel.py,run_owner_add_replay.sh,embedding-replay-result.json,row_owner_route_experiment.py,row_owner_route_numeric.py,row_owner_route_summary.py}
    "$archive_kernel"
    runs/production-data-abcd/component_fixtures.py
    gift/torchtitan/distributed/gfsdp/row_owner_embedding.py
)
sha256sum -- "${artifacts[@]}" > "$output_root/sources.sha256"
"$python" -Bu "$experiment/row_owner_route_summary.py" \
    "$input_root" "$experiment/embedding-replay-result.json" --fixture-only

nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader > "$output_root/gpus.csv"
[[ $(wc -l < "$output_root/gpus.csv") -eq 8 ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
awk -F', ' '$2 !~ /H100/ || $4 != "0 MiB" { exit 1 }' "$output_root/gpus.csv"
command=("$python" -Bu -m torch.distributed.run --standalone --nproc_per_node=8
    "$experiment/owner_add_replay.py" --execute
    --fixture "$input_root" --archive-kernel "$archive_kernel" --output "$output_root")
printf '%q ' "${command[@]}" > "$output_root/command"
printf '\n' >> "$output_root/command"
timeout --signal=TERM --kill-after=45s 900s "${command[@]}" > "$output_root/launcher.log" 2>&1
sha256sum --check --status "$output_root/sources.sha256"
test -s "$output_root/parity.json"
test -s "$output_root/summary.json"
for rank in {0..7}; do
    test -s "$output_root/rank$rank.json"
done
