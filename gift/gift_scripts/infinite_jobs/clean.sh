#!/usr/bin/bash
# Kill leftover torchtitan/torchrun processes and free shared memory across all
# nodes. Run after a job is stopped or crashes (run.sh also calls it first).
set -uo pipefail

export NNODES="${NNODES:-1}"
export OPAL_PREFIX=/opt/hpcx/ompi

HOSTLIST=/tmp/hostlist
cat /etc/volcano/mpimaster.host > "$HOSTLIST"
echo >> "$HOSTLIST"   # the host files lack a trailing newline; keep hosts on separate lines
# mpiworker.host is absent on single-node jobs (no workers).
if [ -f /etc/volcano/mpiworker.host ]; then
  cat /etc/volcano/mpiworker.host >> "$HOSTLIST"
  echo >> "$HOSTLIST"
fi

mpirun --hostfile "$HOSTLIST" \
  -N 1 -np "$NNODES" \
  -bind-to none \
  -x OPAL_PREFIX \
  -mca plm_rsh_args "-p 2222" \
  /bin/bash -c '
    pkill -9 -f "[t]orchtitan.train" || true
    pkill -9 -f "[t]orchrun" || true
    pkill -9 -f "[t]orch.distributed" || true
    rm -f /dev/shm/torch_* /dev/shm/nccl-* /dev/shm/[0-9][0-9][0-9][0-9][0-9][0-9]_* 2>/dev/null || true
  '
