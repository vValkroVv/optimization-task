#!/usr/bin/env bash
set -Eeuo pipefail

workspace=/workspace-SR008.nfs2/vkropotin
control="$workspace/dg-fix"
owner="$control/runs/dense-gfsdp-fix"
staging=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
prior="$owner/artifacts/reviewed-port-001/matched-layer0-002"
output="$owner/artifacts/raw/embedding-combined-001"
compact_exact="$owner/artifacts/raw/compact-native-002/exact/comparison.json"
source_data="$workspace/optimization-task/runs/production-data-abcd/artifacts/raw"
capture="$source_data/proddata-component-capture-001/native-inputs"
census="$source_data/loader-verification/data002-full/depth0"
data="$source_data/data-002/train"
reference="$workspace/a10-cute-001/runs/production-data-abcd/artifacts/raw/original-a-vs-cute-001-a2"
python=/opt/torchtitan-venv/bin/python3.12
driver="$staging/narrow_native_acceptance.py"
numeric="$staging/narrow_owner_exact.py"
owner_file=gift/torchtitan/distributed/gfsdp/row_owner_embedding.py

trap 'status=$?; printf "%s\n" "$status" > "$staging/queue.exit"; exit "$status"' EXIT
test -s "$prior/queue.pid"
while [[ ! -s "$prior/queue.exit" ]]; do
    kill -0 "$(cat "$prior/queue.pid")"
    (( SECONDS < 14400 ))
    sleep 120
done
printf '%s Predecessor finished with status %s\n' "$(date -u +'%FT%TZ')" "$(cat "$prior/queue.exit")"

(cd "$staging" && sha256sum --check sources.sha256)
source /etc/profile.d/wm.sh
set -u
test "$(cat /etc/volcano/VC_MPIMASTER_NUM)" = 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

mkdir "$output"
mkdir "$output/exact"
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
for cap in (4, 8):
    options = dist.ProcessGroupNCCL.Options()
    options.config.max_ctas = cap
    assert options.config.max_ctas == cap
PY

pin() {
    "$python" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' \
        "$staging/source-pins.json" "$1" "$2"
}
source_hash=tracked_and_untracked_sha256
pins=(
    --original-root "$workspace/dg-a" --candidate-root "$control" --reference "$reference"
    --candidate-head e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f
    --candidate-source-sha256 da8fdaca51a3aaa509eb9f0e5e3108f0c3831c5f88e828932324be0b96898abe
    --narrow4-root "$workspace/dg-n4" --narrow4-source-sha256 "$(pin dg-n4 "$source_hash")"
    --narrow8-root "$workspace/dg-n8" --narrow8-source-sha256 "$(pin dg-n8 "$source_hash")"
    --combined-root "$workspace/dg-n4c" --combined-source-sha256 "$(pin dg-n4c "$source_hash")"
    --combined-max-ctas 4 --compact-exact-receipt "$compact_exact"
)
all_arms=(--arm a --arm shared --arm narrow-4 --arm narrow-8 --arm combined)
for ep in 8 4; do
    "$python" -B "$driver" "$output/ep$ep" --ep "$ep" "${all_arms[@]}" "${pins[@]}" --preflight-only
done

run_exact() (
    local name="$1" root="$2" digest="$3" arm="$4"
    shift 4
    local dest="$output/exact/$name"
    mkdir "$dest" || exit 1
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader > "$dest/gpus.csv" || exit 1
    test "$(wc -l < "$dest/gpus.csv")" = 8 || exit 1
    awk -F', ' '$2 !~ /H100/ || $3 != "0 MiB" {exit 1}' "$dest/gpus.csv" || exit 1
    local apps
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 1
    test -z "$apps" || exit 1
    test "$(sha256sum "$root/$owner_file" | cut -d ' ' -f 1)" = "$digest" || exit 1
    cd "$root" || exit 1
    export PYTHONPATH="$root/gift/contrib/flash-linear-attention:$root/gift/contrib/flash-qla:$root/gift"
    export ROW_OWNER_REPLAY_SOURCE_ROOT="$root"
    export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT NCCL_DEBUG_FILE="$dest/nccl.%h.%p.log"
    local command=("$python" -Bu -m torch.distributed.run --standalone --nproc_per_node=8
        "$numeric" run --arm "$arm" --source-root "$root" --owner-source-sha256 "$digest"
        --high-fixture "$capture" --low-census "$census" --data-root "$data"
        --output "$dest/receipt.json" "$@")
    printf '%q ' "${command[@]}" > "$dest/command" || exit 1
    printf '\n' >> "$dest/command" || exit 1
    printf '%s START exact replay %s\n' "$(date -u +'%FT%TZ')" "$name"
    timeout --signal=TERM --kill-after=45s 900s "${command[@]}" > "$dest/launcher.log" 2>&1
)

check_channels() {
    "$python" -B - "$output/exact/$1" "$2" <<'PY'
import json
from pathlib import Path
import re
import sys

root, cap = Path(sys.argv[1]), int(sys.argv[2])
by_rank = {}
for path in root.glob("nccl.*.log"):
    for rank, value in re.findall(r"\[(\d+)\] NCCL INFO[^\n]*?, (\d+) p2p channels per peer", path.read_text()):
        by_rank.setdefault(int(rank), set()).add(int(value))
assert sorted(by_rank) == list(range(8)) and all(cap in values for values in by_rank.values()), by_rank
with (root / "effective-channels.json").open("x") as handle:
    json.dump({rank: sorted(values) for rank, values in sorted(by_rank.items())}, handle, indent=2)
    handle.write("\n")
PY
}

shared_owner=f3a01c6cf99b981db7b4ef66866fe46259af84e9d2c21e2044e9ae18ff1696fd
valid=()
exact_status=0
if run_exact shared "$control" "$shared_owner" a && check_channels shared 32; then
    for name in narrow-4 narrow-8 combined; do
        extra=()
        case "$name" in
            narrow-4) checkout=dg-n4; cap=4 ;;
            narrow-8) checkout=dg-n8; cap=8 ;;
            combined) checkout=dg-n4c; cap=4; extra=(--compact-source) ;;
        esac
        if run_exact "$name" "$workspace/$checkout" "$(pin "$checkout" owner_sha256)" b \
            --max-ctas "$cap" "${extra[@]}" \
            && check_channels "$name" "$cap" \
            && "$python" -B "$numeric" compare "$output/exact/shared/receipt.json" \
                "$output/exact/$name/receipt.json" --output "$output/exact/$name/comparison.json"; then
            valid+=("$name")
        else
            exact_status=1
            printf 'FAIL exact replay %s\n' "$name"
        fi
    done
else
    printf 'FAIL exact replay shared\n'
    exact_status=1
fi
printf '%s\n' "${valid[@]}" > "$output/exact/passed-arms.txt"

native_status=0
for ep in 8 4; do
    if [[ $ep == 8 ]]; then
        ordered=(a shared narrow-4 narrow-8 combined)
    else
        ordered=(shared a narrow-8 narrow-4 combined)
    fi
    selected=()
    for name in "${ordered[@]}"; do
        if [[ "$name" == a || "$name" == shared || " ${valid[*]} " == *" $name "* ]]; then
            selected+=(--arm "$name")
        fi
    done
    printf '%s START EP%s native 10+10\n' "$(date -u +'%FT%TZ')" "$ep"
    "$python" -Bu "$driver" "$output/ep$ep" --ep "$ep" "${selected[@]}" "${pins[@]}" || native_status=1
done
test "$exact_status" = 0 && test "$native_status" = 0
