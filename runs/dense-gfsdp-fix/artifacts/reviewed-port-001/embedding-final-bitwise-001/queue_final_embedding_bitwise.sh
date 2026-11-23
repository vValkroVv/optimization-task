#!/usr/bin/env bash
set -Ee -o pipefail

workspace=/workspace-SR008.nfs2/vkropotin
control="$workspace/dg-fix"
owner="$control/runs/dense-gfsdp-fix"
staging=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output="$owner/artifacts/raw/embedding-final-bitwise-001"
helpers="$owner/artifacts/reviewed-port-001/embedding-combined-001"
source_data="$workspace/optimization-task/runs/production-data-abcd/artifacts/raw"
capture="$source_data/proddata-component-capture-001/native-inputs"
census="$source_data/loader-verification/data002-full/depth0"
data="$source_data/data-002/train"
python=/opt/torchtitan-venv/bin/python3.12
driver="$staging/final_embedding_bitwise.py"
owner_file=gift/torchtitan/distributed/gfsdp/row_owner_embedding.py
minimum_free_mib=78000

test ! -e "$staging/queue.exit"
trap 'status=$?; printf "%s\n" "$status" > "$staging/queue.exit"; exit "$status"' EXIT
source /etc/profile.d/wm.sh
set -u
test "$(cat /etc/volcano/VC_MPIMASTER_NUM)" = 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

mkdir "$output"
printf 'Queue log: %s\n' "$output/queue.log"
exec >> "$output/queue.log" 2>&1
test -r "$driver"
test -d "$helpers"
test -d "$capture"
test -d "$census"
test -d "$data"

"$python" -B - > "$output/runtime.json" <<'PY'
import json
import os

import torch
import torch.distributed as dist

knobs = (
    "TORCH_DISTRIBUTED_USE_TORCHCOMMS", "TORCH_DIST_USE_NCCL2", "NCCL_MIN_CTAS",
    "NCCL_MAX_CTAS", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
    "NCCL_MIN_NRINGS", "NCCL_MAX_NRINGS", "NCCL_MIN_P2P_NCHANNELS",
    "NCCL_MAX_P2P_NCHANNELS",
)
env = {key: os.environ.get(key) for key in knobs}
runtime = (torch.version.git_version, torch.version.cuda, torch.cuda.nccl.version())
print(json.dumps({"runtime": runtime, "environment": env}, indent=2))
assert runtime == ("8b353065ce260336c5fdfa74d754e8bd9b91b062", "13.0", (2, 30, 7)), runtime
assert not any(env.values()) and not dist.distributed_c10d._use_torchcomms_enabled(), env
options = dist.ProcessGroupNCCL.Options()
options.config.max_ctas = 4
assert options.config.max_ctas == 4
PY

run_exact() (
    local name="$1" root="$2" digest="$3"
    local dest="$output/$name"
    mkdir "$dest"
    nvidia-smi --query-gpu=index,name,memory.free,memory.total,uuid --format=csv,noheader,nounits \
        > "$dest/gpus.csv"
    if ! awk -F',[[:space:]]*' -v minimum="$minimum_free_mib" '
        $1 !~ /^[0-9]+$/ || ++seen[$1] != 1 || $2 !~ /H100/ ||
        $3 !~ /^[0-9]+$/ || $3 < minimum {bad = 1}
        END {exit (bad || NR != 8)}
    ' "$dest/gpus.csv"; then
        printf 'FAIL %s requires eight distinct H100s with at least %s MiB free each\n' \
            "$name" "$minimum_free_mib" >&2
        exit 1
    fi
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        > "$dest/compute-apps.csv"
    if grep -q '[^[:space:]]' "$dest/compute-apps.csv"; then
        printf 'FAIL %s found active GPU compute applications\n' "$name" >&2
        exit 1
    fi
    if [[ "$(sha256sum "$root/$owner_file" | cut -d ' ' -f 1)" != "$digest" ]]; then
        printf 'FAIL %s owner source SHA-256 differs\n' "$name" >&2
        exit 1
    fi

    cd "$root"
    export PYTHONPATH="$root/gift/contrib/flash-linear-attention:$root/gift/contrib/flash-qla:$root/gift:$helpers"
    export ROW_OWNER_REPLAY_SOURCE_ROOT="$root"
    export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT NCCL_DEBUG_FILE="$dest/nccl.%h.%p.log"
    local command=("$python" -Bu -m torch.distributed.run --standalone --nproc_per_node=8
        "$driver" run --variant "$name" --source-root "$root" --owner-source-sha256 "$digest"
        --high-fixture "$capture" --low-census "$census" --data-root "$data"
        --output "$dest/receipt.json")
    printf '%q ' "${command[@]}" > "$dest/command"
    printf '\n' >> "$dest/command"
    printf '%s START exact replay %s\n' "$(date -u +'%FT%TZ')" "$name"
    timeout --signal=TERM --kill-after=45s 900s "${command[@]}" > "$dest/launcher.log" 2>&1
)

check_channels() {
    "$python" -B - "$output/$1" "$2" <<'PY'
import json
from pathlib import Path
import re
import sys

root, owner_channels = Path(sys.argv[1]), int(sys.argv[2])
by_rank = {}
pattern = re.compile(r"\[(\d+)\] NCCL INFO[^\n]*?, (\d+) p2p channels per peer")
for path in sorted(root.glob("nccl.*.log")):
    for rank, channels in pattern.findall(path.read_text()):
        by_rank.setdefault(int(rank), set()).add(int(channels))
with (root / "effective-channels.json").open("x") as handle:
    json.dump({rank: sorted(values) for rank, values in sorted(by_rank.items())}, handle, indent=2)
    handle.write("\n")

# The dedicated replay initializes a second communicator alongside dense.
expected = {32, owner_channels}
assert sorted(by_rank) == list(range(8)) and all(
    values == expected for values in by_rank.values()
), (expected, by_rank)
PY
}

run_exact shared "$control" f3a01c6cf99b981db7b4ef66866fe46259af84e9d2c21e2044e9ae18ff1696fd
check_channels shared 32
run_exact narrow-4 "$workspace/dg-n4" 5d6fa6bf6b06eb30b4c234550748664c8df67bad08ca29a3ff21a97d2719128d
check_channels narrow-4 4
run_exact combined "$workspace/dg-n4c" e2603ac17f2adf9a3d7948f53290700e4133fa698ca15a47122b116a6aabdf72
check_channels combined 4

PYTHONPATH="$helpers" "$python" -Bu "$driver" compare \
    --shared "$output/shared/receipt.json" \
    --narrow "$output/narrow-4/receipt.json" \
    --combined "$output/combined/receipt.json" \
    --output "$output/comparison.json" > "$output/compare.log" 2>&1
printf '%s PASS exact replays and comparison\n' "$(date -u +'%FT%TZ')"
