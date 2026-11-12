#!/usr/bin/env bash
set -euo pipefail
umask 077
ulimit -c 0

TASK_ROOT=/workspace-SR008.nfs2/vkropotin/optimization-task
RAW="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw"
ATTEMPT_DIR="$RAW/SANITIZER-ORIGIN/attempts/001"
GIFT_DIR="$TASK_ROOT/gift"
WATCHER="$ATTEMPT_DIR/watch_binary_provenance.py"
GATE="$ATTEMPT_DIR/launch-gate.json"
PYTHON=/opt/torchtitan-venv/bin/python3.12
SANITIZER=/usr/local/cuda/bin/compute-sanitizer

"$PYTHON" "$WATCHER" verify-ready --attempt-dir "$ATTEMPT_DIR"
"$PYTHON" "$ATTEMPT_DIR/verify_pins.py"

for output in native compiler-options scan-runtime torchrun lifecycle.log runner-status.json \
  launch-receipt.json process-discovery.json helper-started.json helper-completion.json \
  launch-ack.json task-processes.jsonl task-stop.json helper-process.json \
  helper.stdout.log helper.stderr.log binary-provenance-receipt.json storage-watch.jsonl \
  binary-provenance-receipt.pending storage-stop.json watcher-failure.json watcher-complete.json; do
  if [[ -e "$ATTEMPT_DIR/$output" ]]; then
    printf 'refusing to overwrite existing attempt output: %s\n' "$output" >&2
    exit 2
  fi
done
if compgen -G "$ATTEMPT_DIR/sanitizer.*.log" >/dev/null; then
  printf 'refusing to reuse an existing sanitizer log\n' >&2
  exit 2
fi
if [[ ! -f "$GATE" ]]; then
  printf 'refusing launch without the reviewed prelaunch gate\n' >&2
  exit 2
fi

"$PYTHON" "$WATCHER" launch-receipt --attempt-dir "$ATTEMPT_DIR" \
  --runner-pid "$$" --launcher-path "$ATTEMPT_DIR/launch.sh" --gate-path "$GATE"
"$PYTHON" "$WATCHER" wait-ack --attempt-dir "$ATTEMPT_DIR" --timeout-seconds 15
( set -o noclobber; : > "$ATTEMPT_DIR/lifecycle.log" )
{
  printf 'started_at_utc='
  date -u --iso-8601=ns
  printf 'runner_pid=%s\n' "$$"
  printf 'source_manifest_sha256=2585e504350276cb4689ae9ea45a33706e9f898bb61969abe5d08f2650c55620\n'
  printf 'source_archive_sha256=0fc5ddad83c5f99b4036a47dd7f485ba2f42d1284c42f086d5f051cebad66082\n'
  printf 'graph_policy=TORCHINDUCTOR_CUDAGRAPHS=0; mix_order_reduction=0; graph_partition=true\n'
  printf 'public_step_boundary=campaign_cudagraph_step_boundary\n'
  printf 'sanitizer=memcheck; binary_patching=0; print_limit=50; no profiling or completion/timing observers\n'
} >> "$ATTEMPT_DIR/lifecycle.log"

CACHE_DIR="$RAW/NATIVE-SMOKE/001/cache"
CACHE_ROOT="$RAW/cache"
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
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
export NV_COMPUTE_SANITIZER_BINARY_PATCHING=0
unset TORCH_LOGS TORCH_COMPILE_DEBUG TORCH_TRACE
export TORCHFT_LIGHTHOUSE=http://localhost:29510

TIMEOUT_PID=
finish() {
  rc=$?
  trap - EXIT INT TERM
  if [[ -n "${TIMEOUT_PID:-}" ]] && kill -0 "$TIMEOUT_PID" 2>/dev/null; then
    kill -TERM "$TIMEOUT_PID" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$TIMEOUT_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$TIMEOUT_PID" 2>/dev/null; then
      kill -KILL "$TIMEOUT_PID" 2>/dev/null || true
    fi
    for _ in {1..20}; do
      kill -0 "$TIMEOUT_PID" 2>/dev/null || break
      sleep 0.1
    done
  fi
  {
    printf 'finished_at_utc='
    date -u --iso-8601=ns
    printf 'exit_code=%s\n' "$rc"
  } >> "$ATTEMPT_DIR/lifecycle.log" 2>/dev/null || true
  "$PYTHON" - "$ATTEMPT_DIR/runner-status.json" "$rc" <<'STATUS' 2>/dev/null || true
from datetime import datetime, timezone
import json, os, sys, tempfile
target = sys.argv[1]
payload = (json.dumps({"exit_code": int(sys.argv[2]), "finished_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n").encode()
fd, temporary = tempfile.mkstemp(prefix=".runner-status.", suffix=".tmp", dir=os.path.dirname(target))
try:
    with os.fdopen(fd, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, target)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
STATUS
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' INT TERM

cd "$GIFT_DIR"
set +e
timeout --signal=TERM --kill-after=10s 20m "$SANITIZER" \
  --tool memcheck \
  --target-processes all \
  --error-exitcode 86 \
  --show-backtrace yes \
  --log-file "$ATTEMPT_DIR/sanitizer.%p.log" \
  --print-limit 50 \
  --launch-timeout 180 \
  "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --max-restarts=0 \
  --log-dir="$ATTEMPT_DIR/torchrun" \
  --redirects=3 \
  --tee=3 \
  "$ATTEMPT_DIR/compiler_entry.py" \
  "$ATTEMPT_DIR/config/sanitizer_origin.yaml" \
  --training.steps 60 \
  --lr_scheduler.total_steps 30000 \
  --debug.seed 42 \
  --checkpoint.enable false \
  --dump_folder "$ATTEMPT_DIR/native" &
TIMEOUT_PID=$!
printf 'timeout_pid=%s\n' "$TIMEOUT_PID" >> "$ATTEMPT_DIR/lifecycle.log"
wait "$TIMEOUT_PID"
run_rc=$?
set -e
exit "$run_rc"
