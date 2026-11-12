#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=/workspace-SR008.nfs2/vkropotin/optimization-task
ATTEMPT_DIR="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-ASSESSMENT/attempts/002"
CACHE_DIR="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NATIVE-SMOKE/001/cache"
CACHE_ROOT="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/cache"
GIFT_DIR="$TASK_ROOT/gift"
SOURCE_MANIFEST="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/SOURCE-SNAPSHOT/010/manifest.json"
SOURCE_ARCHIVE="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/SOURCE-SNAPSHOT/010/changed-files.zip"
RELEASE_FILE="$ATTEMPT_DIR/release.json"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?Root release must supply the SOURCE-SNAPSHOT/010 manifest SHA-256}"
PRESET="$GIFT_DIR/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml"
PYTHON=/opt/torchtitan-venv/bin/python3.12
PRESET_SHA256=b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b

umask 077
test "$(sha256sum "$SOURCE_MANIFEST" | cut -d' ' -f1)" = "$EXPECTED_SOURCE_MANIFEST_SHA256"
test "$(sha256sum "$PRESET" | cut -d' ' -f1)" = "$PRESET_SHA256"
test "$(sha256sum "$ATTEMPT_DIR/compiler_entry.py" | cut -d' ' -f1)" = "ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065"
test "$(sha256sum "$ATTEMPT_DIR/config/observer.yaml" | cut -d' ' -f1)" = "01675fbc967e2e0a7e3798d333edc117d18bf5ad3746d999a86f566a3ff554f8"
test "$(sha256sum "$ATTEMPT_DIR/phase_entry.py" | cut -d' ' -f1)" = "cc81cc8340fb1d9becc8196095aedd75f40b35f6db07a35775f3a9839ca54f59"
"$PYTHON" - "$RELEASE_FILE" "$EXPECTED_SOURCE_MANIFEST_SHA256" "01675fbc967e2e0a7e3798d333edc117d18bf5ad3746d999a86f566a3ff554f8" <<'RELEASE_GATE'
import json
from pathlib import Path
import sys

release = json.loads(Path(sys.argv[1]).read_text())
if release["status"] != "COORDINATOR_CONFIRMED_FOR_LAUNCH":
    raise SystemExit("waiting for coordinator's final launch confirmation")
if release["scope"] != "PERFORMANCE-ASSESSMENT/attempts/002":
    raise SystemExit("release scope does not match this attempt")
if release["source_manifest_sha256"] != sys.argv[2]:
    raise SystemExit("release source pin does not match the supplied manifest SHA-256")
if release["config_sha256"] != sys.argv[3]:
    raise SystemExit("release config pin does not match this attempt")
RELEASE_GATE
"$PYTHON" - "$TASK_ROOT" "$SOURCE_MANIFEST" "$SOURCE_ARCHIVE" <<'SOURCE_PIN'
import hashlib
import json
from pathlib import Path
import sys

root, manifest_path, archive_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
if archive_digest != manifest["archive_sha256"]:
    raise SystemExit("SOURCE-SNAPSHOT/010 archive SHA-256 mismatch")
for relative_path, expected in manifest["files"].items():
    path = root / relative_path
    data = path.read_bytes()
    if len(data) != expected["bytes"]:
        raise SystemExit(f"source byte-count mismatch: {relative_path}")
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise SystemExit(f"source SHA-256 mismatch: {relative_path}")
print(f"verified_source_files={len(manifest['files'])}")
print(f"verified_source_archive_sha256={archive_digest}")
SOURCE_PIN

for output in runner.pid lifecycle.log exit-status.json device-memory.csv memory-sampler.pid nvidia-smi.stderr native torchrun phase_memory compiler-options runtime; do
  if [[ -e "$ATTEMPT_DIR/$output" ]]; then
    printf 'refusing to overwrite existing attempt output: %s\n' "$ATTEMPT_DIR/$output" >&2
    exit 2
  fi
done
if [[ -e "$ATTEMPT_DIR/scan-runtime" ]]; then
  printf 'refusing to reuse HybridEP runtime directory: %s\n' "$ATTEMPT_DIR/scan-runtime" >&2
  exit 2
fi

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
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
export CUDA_VISIBLE_DEVICES="GPU-519db4ab-52c1-7730-5f03-bb8ecba718c3,GPU-f0ff1e03-5cda-3f99-6b5b-22016cc776bf,GPU-35c5887c-b022-4a95-bdc4-f6c6e2175d52,GPU-384c07d4-8c7d-6c6f-c4a8-66a7c18316eb,GPU-d52fef45-0944-a850-6d82-86acee633ce8,GPU-9a1e0c80-7b57-12a4-6220-6a2b9a267466,GPU-5f286803-df1a-a5db-5868-d31d1dde3201,GPU-d03bb98d-079d-b834-e5a0-558188e55667"
unset TORCH_LOGS TORCH_COMPILE_DEBUG TORCH_TRACE CUDA_LAUNCH_BLOCKING
export TORCHFT_LIGHTHOUSE=http://localhost:29510

MEMORY_SAMPLER_PID=
printf 'runner_pid=%s\n' "$$" > "$ATTEMPT_DIR/runner.pid"
{
  printf 'started_at_utc='
  date -u --iso-8601=ns
  printf 'cwd=%s\n' "$PWD"
  printf 'task_local_import_path_prepend=%s\n' "$GIFT_DIR"
  printf 'source_snapshot_manifest=%s\n' "$SOURCE_MANIFEST"
  printf 'source_snapshot_manifest_sha256=%s\n' "$EXPECTED_SOURCE_MANIFEST_SHA256"
  printf 'fresh_hybrid_ep_runtime_dir=%s\n' "$ATTEMPT_DIR/scan-runtime"
  printf 'home_environment=preserved\n'
  printf 'shared_existing_cache=%s\n' "$CACHE_DIR"
  printf 'graph_policy=TORCHINDUCTOR_CUDAGRAPHS=0; extra_compiler_logging=off\n'
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

set +e
"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --max-restarts=0 \
  --log-dir="$ATTEMPT_DIR/torchrun" \
  --redirects=3 \
  --tee=3 \
  "$ATTEMPT_DIR/phase_entry.py" \
  "$ATTEMPT_DIR/config/observer.yaml" \
  --training.steps 41 \
  --lr_scheduler.total_steps 30000 \
  --debug.seed 42 \
  --checkpoint.enable false \
  --dump_folder "$ATTEMPT_DIR/native"
run_rc=$?
exit "$run_rc"
