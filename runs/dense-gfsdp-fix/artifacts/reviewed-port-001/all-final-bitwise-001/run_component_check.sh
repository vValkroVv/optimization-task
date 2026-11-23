#!/usr/bin/env bash
set -Ee -o pipefail

workspace=/workspace-SR008.nfs2/vkropotin
stage=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output="$workspace/dg-fix/runs/dense-gfsdp-fix/artifacts/raw/all-final-bitwise-001"
name="$1"
checkout="$2"
shift 2

case "$name" in
    layer0|mtp-ffn|mtp-transition|embedding) ;;
    *) exit 2 ;;
esac
case "$checkout" in
    dg-a|dg-fix|dg-n4c) ;;
    *) exit 2 ;;
esac
root="$workspace/$checkout"
python=/opt/torchtitan-venv/bin/python3.12
[[ "$1" == "$stage/"* && -f "$1" && -d "$root/gift" ]]
dest="$output/$name"
mkdir "$dest"
trap 'status=$?; printf "%s\n" "$status" > "$dest/exit"; exit "$status"' EXIT
source /etc/profile.d/wm.sh
set -u
test "$(cat /etc/volcano/VC_MPIMASTER_NUM)" = 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/gift/contrib/flash-linear-attention:$root/gift/contrib/flash-qla:$root/gift"
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT NCCL_DEBUG_FILE="$dest/nccl.%h.%p.log"

# One run at a time in this allocation. Leave the task's other processes alone.
exec 9> "$output/gpu.lock"
flock -w 3600 9
nvidia-smi --query-gpu=index,name,memory.free,uuid --format=csv,noheader,nounits > "$dest/gpus.csv"
awk -F',[[:space:]]*' '
    $1 !~ /^[0-9]+$/ || ++seen[$1] != 1 || $2 !~ /H100/ || $3 < 75000 {bad = 1}
    END {exit (bad || NR != 8)}
' "$dest/gpus.csv"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits > "$dest/compute-apps.csv"
if grep -q '[^[:space:]]' "$dest/compute-apps.csv"; then
    printf '%s\n' 'GPU compute applications were present at launch.' >&2
    exit 1
fi

git -C "$root" rev-parse HEAD > "$dest/source-head"
sha256sum "$1" "$root/gift/torchtitan/distributed/gfsdp/"{_state.py,_param_group.py,row_owner_embedding.py} \
    "$root/gift/torchtitan/models/gift_moe/_base/parallelize.py" > "$dest/source.sha256"

cd "$root"
command=("$python" -Bu -m torch.distributed.run --standalone --nproc_per_node=8 "$@")
printf '%q ' "${command[@]}" > "$dest/command"
printf '\n' >> "$dest/command"
printf '%s START %s\n' "$(date -u +'%FT%TZ')" "$name"
timeout --signal=TERM --kill-after=45s 1500s "${command[@]}" > "$dest/launcher.log" 2>&1
printf '%s PASS harness execution %s\n' "$(date -u +'%FT%TZ')" "$name"
