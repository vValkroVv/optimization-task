#!/usr/bin/env bash
set -uo pipefail
ATTEMPT_DIR=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-AA/attempts/002
CACHE_DIR=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NATIVE-SMOKE/001/cache
CACHE_ROOT=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/cache
GIFT_DIR=/workspace-SR008.nfs2/vkropotin/optimization-task/gift
PYTHON=/opt/torchtitan-venv/bin/python3.12
umask 077
cd "$GIFT_DIR"
export PYTHONPATH="$GIFT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$CACHE_DIR/huggingface/datasets"
export HF_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export DG_JIT_SHARED_CACHE_DIR="$CACHE_ROOT/deepgemm_shared"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHFT_LIGHTHOUSE=http://localhost:29510
MEMORY_SAMPLER_PID=
printf 'runner_pid=%s\n' "$$" > "$ATTEMPT_DIR/runner.pid"
{
  printf 'started_at_utc='
  date -u --iso-8601=ns
  printf 'cwd=%s\n' "$PWD"
  printf 'task_local_import_path_prepend=%s\n' "$GIFT_DIR"
  printf 'home_environment=preserved\n'
  printf 'shared_existing_cache=%s\n' "$CACHE_DIR"
} > "$ATTEMPT_DIR/lifecycle.log"
finish() {
  rc=$?
  if [[ -n "$MEMORY_SAMPLER_PID" ]]; then
    if kill -0 "$MEMORY_SAMPLER_PID" 2>/dev/null; then
      kill -TERM "$MEMORY_SAMPLER_PID" 2>/dev/null || true
    fi
    wait "$MEMORY_SAMPLER_PID" 2>/dev/null || true
  fi
  {
    printf 'finished_at_utc='
    date -u --iso-8601=ns
    printf 'exit_code=%s\n' "$rc"
    printf 'memory_sampler_pid=%s\n' "$MEMORY_SAMPLER_PID"
  } >> "$ATTEMPT_DIR/lifecycle.log"
  "$PYTHON" - "$ATTEMPT_DIR/exit-status.json" "$rc" <<'STATUS'
from datetime import datetime, timezone
from pathlib import Path
import json, sys
Path(sys.argv[1]).write_text(json.dumps({
    "exit_code": int(sys.argv[2]),
    "finished_at_utc": datetime.now(timezone.utc).isoformat()
}, indent=2) + "\n")
STATUS
}
trap finish EXIT
printf 'timestamp_utc,index,memory_used_mib\n' > "$ATTEMPT_DIR/device-memory.csv"
TZ=UTC nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits --loop-ms=500 \
  >> "$ATTEMPT_DIR/device-memory.csv" 2> "$ATTEMPT_DIR/nvidia-smi.stderr" &
MEMORY_SAMPLER_PID=$!
printf '%s\n' "$MEMORY_SAMPLER_PID" > "$ATTEMPT_DIR/memory-sampler.pid"
"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --max-restarts=0 \
  --log-dir="$ATTEMPT_DIR/torchrun" \
  --redirects=3 \
  --tee=3 \
  -m torchtitan.train \
  "$ATTEMPT_DIR/config/observer.yaml" \
  --training.steps 71 \
  --lr_scheduler.total_steps 30000 \
  --debug.seed 42 \
  --checkpoint.enable false \
  --dump_folder "$ATTEMPT_DIR/native"
exit $?
