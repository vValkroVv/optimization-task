#!/usr/bin/bash
# Launch torchtitan across all nodes of an infinite job, from inside the
# container (mpimaster-0). One torchrun per node; ranks rendezvous on master.
set -euo pipefail

# --- edit these -------------------------------------------------------------
CODE_PATH="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"  # torchtitan repo
CONFIG_FILE="${CONFIG_FILE:-configs/experiments/llama3_8b.yaml}"  # experiment YAML, relative to CODE_PATH
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"           # dot-path overrides forwarded to torchtitan.train
# ---------------------------------------------------------------------------

MASTER_PORT=29500
export OPAL_PREFIX=/opt/hpcx/ompi

bash "$(dirname "$(realpath "$0")")/clean.sh"

HOSTLIST=/tmp/hostlist
cat /etc/volcano/mpimaster.host > "$HOSTLIST"
echo >> "$HOSTLIST"   # the host files lack a trailing newline; keep hosts on separate lines
# mpiworker.host is absent on single-node jobs (no workers); skip it if missing
# (use `if`, not `&&`, so it does not trip set -e + pipefail).
if [ -f /etc/volcano/mpiworker.host ]; then
  cat /etc/volcano/mpiworker.host >> "$HOSTLIST"
  echo >> "$HOSTLIST"
fi
MASTER_ADDR="$(grep -m1 mpimaster-0 "$HOSTLIST" | tr -d '[:space:]')"

LOGS_DIR="inf-job-logs/$(date +%Y%m%d-%H%M%S)"

mpirun -N 1 -np "$NNODES" \
  --hostfile "$HOSTLIST" \
  -bind-to none \
  -x MASTER_ADDR="$MASTER_ADDR" \
  -x MASTER_PORT="$MASTER_PORT" \
  -x NNODES \
  -x NGPUS_PER_NODE \
  -x CONFIG_FILE \
  -x EXTRA_ARGS \
  -x CODE_PATH \
  -x PYTHONPATH="$CODE_PATH" \
  -x PYTHONUNBUFFERED=1 \
  -x OMP_NUM_THREADS=1 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_HCA=^mlx5_bond \
  -x PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_HOME \
  -x OPAL_PREFIX \
  -mca pml ucx \
  -mca plm_rsh_args "-p 2222" \
  -output-filename "$LOGS_DIR:nojobid" \
  -merge-stderr-to-stdout \
  -tag-output \
  /bin/bash -c '
    [ -f /etc/profile.d/wm.sh ] && source /etc/profile.d/wm.sh
    NODE_RANK=$(hostname | python3 -c "h=input(); print(0 if h.endswith(\"mpimaster-0\") else int(h.split(\"-\")[-1]) + 1, end=\"\")")
    cd "$CODE_PATH"
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    torchrun \
      --nproc_per_node "$NGPUS_PER_NODE" \
      --nnodes "$NNODES" \
      --node_rank "$NODE_RANK" \
      --rdzv_backend c10d \
      --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
      --role rank --tee 3 \
      -m torchtitan.train "$CONFIG_FILE" $EXTRA_ARGS
  '
