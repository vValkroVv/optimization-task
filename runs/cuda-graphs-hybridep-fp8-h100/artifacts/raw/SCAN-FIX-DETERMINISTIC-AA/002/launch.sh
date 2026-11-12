#!/usr/bin/env bash
set -uo pipefail
ATTEMPT_DIR=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/SCAN-FIX-DETERMINISTIC-AA/002
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
export GIFT_HYBRID_EP_SCAN_RUNTIME_DIR="$ATTEMPT_DIR/scan-runtime"
export TORCHINDUCTOR_CUDAGRAPHS=0
export TORCHFT_LIGHTHOUSE=http://localhost:29510
printf 'runner_pid=%s\n' "$$" > "$ATTEMPT_DIR/runner.pid"
{
  printf 'started_at_utc='
  date -u --iso-8601=seconds
  printf 'cwd=%s\n' "$PWD"
  printf 'task_local_import_path_prepend=%s\n' "$GIFT_DIR"
  printf 'home_environment=preserved\n'
  printf 'shared_existing_cache=%s\n' "$CACHE_DIR"
} > "$ATTEMPT_DIR/lifecycle.log"
write_status() {
  rc=$?
  {
    printf 'finished_at_utc='
    date -u --iso-8601=seconds
    printf 'exit_code=%s\n' "$rc"
  } >> "$ATTEMPT_DIR/lifecycle.log"
  "$PYTHON" - "$ATTEMPT_DIR/exit-status.json" "$rc" <<'STATUS'
from datetime import datetime, timezone
from pathlib import Path
import json, sys
Path(sys.argv[1]).write_text(json.dumps({
    "exit_code": int(sys.argv[2]),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")
}, indent=2) + "\n")
STATUS
}
trap write_status EXIT
"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --max-restarts=0 \
  --log-dir="$ATTEMPT_DIR/torchrun" \
  --redirects=3 \
  --tee=3 \
  -m torchtitan.experiments.cuda_graph_campaign.train_scan_fix \
  configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml \
  --training.steps 10 \
  --lr_scheduler.total_steps 30000 \
  --debug.seed 42 \
  --debug.deterministic \
  --debug.deterministic_warn_only false \
  --checkpoint.enable false \
  --metrics.enable_tensorboard true \
  --metrics.log_freq 1 \
  --dump_folder "$ATTEMPT_DIR/native"
exit $?
