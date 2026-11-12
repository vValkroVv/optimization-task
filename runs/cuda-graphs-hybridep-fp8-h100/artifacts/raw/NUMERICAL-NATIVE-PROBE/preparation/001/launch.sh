#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=/workspace-SR008.nfs2/vkropotin/optimization-task
PREP_DIR="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/001"
ATTEMPT_ROOT="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/attempts/001"
RAW_DIR="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw"
GIFT_DIR="$TASK_ROOT/gift"
PYTHON=/opt/torchtitan-venv/bin/python3.12
RELEASE_FILE="$PREP_DIR/root-release.json"
CANONICAL_CONFIG="$GIFT_DIR/configs/experiments/cuda_graph_campaign_continuation.yaml"
CREATE_CONFIG="$PREP_DIR/config/create_s0.yaml"
OBSERVE_CONFIG="$PREP_DIR/config/observe.yaml"
S0_DUMP="$ATTEMPT_ROOT/create_s0"
OBSERVE_DUMP="$ATTEMPT_ROOT/observe"
S0_STEP1="$S0_DUMP/checkpoint/step-1"
CACHE_DIR="$RAW_DIR/NATIVE-SMOKE/001/cache"
CACHE_ROOT="$RAW_DIR/cache"
EXPECTED_GPU_UUIDS=(
  GPU-519db4ab-52c1-7730-5f03-bb8ecba718c3
  GPU-f0ff1e03-5cda-3f99-6b5b-22016cc776bf
  GPU-35c5887c-b022-4a95-bdc4-f6c6e2175d52
  GPU-384c07d4-8c7d-6c6f-c4a8-66a7c18316eb
  GPU-d52fef45-0944-a850-6d82-86acee633ce8
  GPU-9a1e0c80-7b57-12a4-6220-6a2b9a267466
  GPU-5f286803-df1a-a5db-5868-d31d1dde3201
  GPU-d03bb98d-079d-b834-e5a0-558188e55667
)

umask 077
test -x "$PYTHON"
test -f "$RELEASE_FILE" || { echo "missing root release: $RELEASE_FILE" >&2; exit 2; }
"$PYTHON" "$PREP_DIR/verify_release.py" "$RELEASE_FILE"
test ! -e "$ATTEMPT_ROOT" || { echo "refusing existing attempt output: $ATTEMPT_ROOT" >&2; exit 2; }

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
export TORCHINDUCTOR_CUDAGRAPHS=0
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
export CUDA_VISIBLE_DEVICES="$(IFS=,; printf '%s' "${EXPECTED_GPU_UUIDS[*]}")"
unset TORCH_LOGS TORCH_COMPILE_DEBUG TORCH_TRACE

RUN_STAMP=$(date -u +%Y%m%dT%H%M%S%N)
CONFIG_CHECK_DIR="$PREP_DIR/config-checks/$RUN_STAMP"
mkdir -p -m 0700 "$PREP_DIR/config-checks"
mkdir -m 0700 "$CONFIG_CHECK_DIR"
check_config() {
  local label=$1 config=$2 dump=$3 report=$4
  shift 4
  local -a args=("$config" --dump_folder "$dump" "$@")
  if ! "$PYTHON" -m torchtitan.config.pre_submit \
    "${args[@]}" --world-size 8 --report "$report" \
    > "$CONFIG_CHECK_DIR/$label.log" 2>&1; then
    cat "$CONFIG_CHECK_DIR/$label.log" >&2
    echo "config resolution failed for $label; no trainer was started" >&2
    exit 3
  fi
}
check_config create_s0 "$CREATE_CONFIG" "$CONFIG_CHECK_DIR/create_s0-dump" \
  "$CONFIG_CHECK_DIR/create_s0-report.yaml"
check_config observe "$OBSERVE_CONFIG" "$CONFIG_CHECK_DIR/observe-dump" \
  "$CONFIG_CHECK_DIR/observe-report.yaml" \
  --checkpoint.initial_load_path "$S0_STEP1"
test ! -e "$ATTEMPT_ROOT" || { echo "config check created or found the attempt output" >&2; exit 2; }
test ! -e "$CONFIG_CHECK_DIR/create_s0-dump/checkpoint" || { echo "create_s0 config check created a checkpoint folder" >&2; exit 2; }
test ! -e "$CONFIG_CHECK_DIR/observe-dump/checkpoint" || { echo "observe config check created a checkpoint folder" >&2; exit 2; }

mkdir -m 0700 "$ATTEMPT_ROOT"
CONTROLLER_PID=$$
CONTROLLER_START=$(awk '{print $22}' "/proc/$CONTROLLER_PID/stat")
CONTROLLER_PPID=$(awk '/^PPid:/ {print $2}' "/proc/$CONTROLLER_PID/status")
CONTROLLER_PGID=$(ps -o pgid= -p "$CONTROLLER_PID" | tr -d ' ')
CONTROLLER_SID=$(ps -o sid= -p "$CONTROLLER_PID" | tr -d ' ')
if [[ "$CONTROLLER_PGID" != "$CONTROLLER_PID" || "$CONTROLLER_SID" != "$CONTROLLER_PID" ]]; then
  echo "launch.sh must run detached in its own session (setsid)" >&2
  exit 2
fi
"$PYTHON" - "$ATTEMPT_ROOT/controller.json" "$CONTROLLER_PID" "$CONTROLLER_START" "$CONTROLLER_PPID" "$CONTROLLER_PGID" "$CONTROLLER_SID" <<'CONTROLLER'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps({
    "pid": int(sys.argv[2]),
    "start_ticks": int(sys.argv[3]),
    "ppid": int(sys.argv[4]),
    "pgid": int(sys.argv[5]),
    "sid": int(sys.argv[6]),
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
CONTROLLER
printf 'started_at_utc=%s\n' "$(date -u --iso-8601=ns)" > "$ATTEMPT_ROOT/lifecycle.log"
printf 'controller_pid=%s start_ticks=%s ppid=%s pgid=%s sid=%s\n' \
  "$CONTROLLER_PID" "$CONTROLLER_START" "$CONTROLLER_PPID" "$CONTROLLER_PGID" "$CONTROLLER_SID" \
  >> "$ATTEMPT_ROOT/lifecycle.log"
printf 'phase\ttimestamp_utc\towned_host_rss_bytes\towned_process_count\tphase_bytes\tattempt_bytes\tfs_available_bytes\tgpu_memory_mib\n' \
  > "$ATTEMPT_ROOT/resource-samples.tsv"

ACTIVE_PID=
ACTIVE_START=
ACTIVE_PHASE=
ACTIVE_LEDGER=
owner_identity_matches() {
  [[ -n "$ACTIVE_PID" && -r "/proc/$ACTIVE_PID/stat" ]] || return 1
  local start
  start=$("$PYTHON" -c 'from pathlib import Path; import sys; raw=Path(f"/proc/{sys.argv[1]}/stat").read_text(); print(raw[raw.rfind(")")+2:].split()[19])' "$ACTIVE_PID" 2>/dev/null) || return 1
  [[ "$start" == "$ACTIVE_START" ]]
}
stop_active_owner() {
  [[ -n "$ACTIVE_PID" ]] || return 0
  if owner_identity_matches; then
    printf 'sending TERM to owned phase owner pid=%s start_ticks=%s phase=%s\n' \
      "$ACTIVE_PID" "$ACTIVE_START" "$ACTIVE_PHASE" >> "$ATTEMPT_ROOT/lifecycle.log"
    kill -TERM "$ACTIVE_PID" 2>/dev/null || true
  fi
  wait "$ACTIVE_PID" 2>/dev/null || true
  if [[ -n "$ACTIVE_LEDGER" && -f "$ACTIVE_LEDGER" ]]; then
    if ! "$PYTHON" "$PREP_DIR/phase_process_owner.py" verify "$ACTIVE_LEDGER" \
      >> "$ATTEMPT_ROOT/lifecycle.log" 2>&1; then
      printf 'owned identities remain or cannot be verified for %s\n' "$ACTIVE_PHASE" \
        >> "$ATTEMPT_ROOT/lifecycle.log"
      return 1
    fi
  fi
  ACTIVE_PID=
  ACTIVE_START=
  ACTIVE_PHASE=
  ACTIVE_LEDGER=
}
finish() {
  local rc=$?
  trap - EXIT INT TERM
  if ! stop_active_owner; then rc=70; fi
  {
    printf 'finished_at_utc=%s\n' "$(date -u --iso-8601=ns)"
    printf 'exit_code=%s\n' "$rc"
  } >> "$ATTEMPT_ROOT/lifecycle.log"
  "$PYTHON" - "$ATTEMPT_ROOT/exit-status.json" "$rc" <<'STATUS'
from datetime import datetime, timezone
from pathlib import Path
import json, sys
Path(sys.argv[1]).write_text(json.dumps({
    "exit_code": int(sys.argv[2]),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
STATUS
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

sample_resources() {
  local phase=$1 dump_dir=$2 ledger=$3 count=$4
  local now rss_line rss_bytes process_count phase_bytes attempt_bytes free_bytes gpu_memory
  now=$(date -u --iso-8601=ns)
  rss_line=$("$PYTHON" "$PREP_DIR/phase_process_owner.py" rss "$ledger" 2>>"$ATTEMPT_ROOT/sampler.stderr" || true)
  if [[ "$rss_line" =~ ^([0-9]+)[[:space:]]+([0-9]+)$ ]]; then
    rss_bytes=${BASH_REMATCH[1]}
    process_count=${BASH_REMATCH[2]}
  else
    rss_bytes=unavailable
    process_count=unavailable
  fi
  if (( count % 3 == 0 )); then
    phase_bytes=$(timeout --signal=TERM --kill-after=1s 8s du -sb "$dump_dir" 2>/dev/null | awk '{print $1}' || true)
    attempt_bytes=$(timeout --signal=TERM --kill-after=1s 8s du -sb "$ATTEMPT_ROOT" 2>/dev/null | awk '{print $1}' || true)
    free_bytes=$(timeout --signal=TERM --kill-after=1s 5s df -B1 --output=avail "$PREP_DIR" 2>/dev/null | tail -n 1 | tr -d ' ' || true)
  else
    phase_bytes=not_sampled
    attempt_bytes=not_sampled
    free_bytes=not_sampled
  fi
  gpu_memory=$(timeout --signal=TERM --kill-after=1s 5s nvidia-smi \
    --query-gpu=uuid,memory.used --format=csv,noheader,nounits \
    2>>"$ATTEMPT_ROOT/nvidia-smi.stderr" | tr '\n' ';' || true)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$phase" "$now" "$rss_bytes" "$process_count" \
    "${phase_bytes:-}" "${attempt_bytes:-}" "${free_bytes:-}" "${gpu_memory:-unavailable}" \
    >> "$ATTEMPT_ROOT/resource-samples.tsv"
}
verify_config() {
  local dump_dir=$1 steps=$2 initial_load_path=$3
  "$PYTHON" - "$dump_dir/config/resolved_config.yaml" "$steps" "$initial_load_path" <<'CONFIG'
import sys
import yaml
from pathlib import Path

path, expected_steps, initial_load_path = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
data = yaml.safe_load(path.read_text(encoding="utf-8"))
assert data["trainer"]["type"] == "cuda_graph_campaign"
assert data["debug"]["seed"] == 42
training = data["training"]
assert (training["steps"], training["local_batch_size"], training["global_batch_size"], training["seq_len"]) == (expected_steps, 8, 256, 4096)
assert training["global_batch_size"] // (training["local_batch_size"] * 8) == 4
assert (training["dtype"], training["mixed_precision_param"], training["mixed_precision_reduce"]) == ("float32", "bfloat16", "float32")
assert (data["optimizer"]["name"], data["optimizer"]["implementation"]) == ("AdamW", "fused")
assert (data["lr_scheduler"]["total_steps"], data["lr_scheduler"]["warmup_steps"]) == (30000, 2000)
assert data["compile"]["enable"] is True and data["compile"]["components"] == ["model", "loss"]
assert data["activation_checkpoint"]["mode"] == "selective"
parallel = data["parallelism"]
assert (parallel["fsdp_backend"], parallel["expert_parallel_degree"], parallel["tensor_parallel_degree"], parallel["pipeline_parallel_degree"]) == ("fsdp2", 8, 1, 1)
blocks = data["model"]["config"]["blocks"]
assert blocks["mla_dense"]["attention"]["inner_attention"]["type"] == "flex"
assert blocks["mla_moe"]["attention"]["inner_attention"]["type"] == "flex"
assert blocks["mla_moe"]["moe"]["experts"]["token_dispatcher"]["type"] == "hybridep"
assert data["fp8"]["sparse_use_grouped_gemm"] is True
assert data["dataloader"]["dataset"] == "c4_test"
assert list(data["callbacks"]) == ["campaign_rng_continuation", "campaign_numerical_probe"]
numerical = data["callbacks"]["campaign_numerical_probe"]
expected_mode = "create_s0" if expected_steps == 1 else "observe"
expected_output = "checkpoint/numerical_probe" if expected_steps == 1 else "numerical_probe"
assert (numerical["mode"], numerical["updates"], numerical["output_dir"]) == (expected_mode, 1, expected_output)
checkpoint = data["checkpoint"]
for name, value in {
    "enable": True,
    "folder": "checkpoint",
    "interval": 1,
    "async_mode": "disabled",
    "keep_latest_k": 0,
    "create_seed_checkpoint": False,
    "initial_load_model_only": False,
    "initial_load_in_hf": False,
    "load_step": -1,
    "last_save_model_only": False,
    "save_in_hf": False,
    "last_save_in_hf": False,
    "exclude_from_loading": [],
    "load_only": expected_steps == 2,
}.items():
    assert checkpoint[name] == value, (name, checkpoint[name], value)
assert checkpoint.get("initial_load_path") == (initial_load_path or None)
print(f"resolved_config_verified={path}")
CONFIG
}
verify_compiler_options() {
  local phase_dir=$1
  "$PYTHON" - "$phase_dir/compiler-options" <<'COMPILER'
import json
from pathlib import Path
import sys
directory = Path(sys.argv[1])
for rank in range(8):
    path = directory / f"rank-{rank:05d}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["rank"] == rank
    assert data["triton.mix_order_reduction"] is False
    assert data["triton.cudagraphs"] is False
    assert data["graph_partition"] is True
print(f"compiler_options_verified={directory}")
COMPILER
}
verify_probe_manifest() {
  local manifest=$1 mode=$2 phase=$3 step=$4
  "$PYTHON" - "$manifest" "$mode" "$phase" "$step" <<'MANIFEST'
import json
from pathlib import Path
import sys
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode, phase, step = sys.argv[2], sys.argv[3], int(sys.argv[4])
assert data["mode"] == mode and data["phase"] == phase
assert data["world_size"] == 8 and data["complete"] is True
assert len(data["ranks"]) == 8
assert sorted(item["rank"] for item in data["ranks"]) == list(range(8))
assert all(item["ok"] and item["step"] == step for item in data["ranks"])
print(f"probe_manifest_verified={sys.argv[1]}")
MANIFEST
}
run_phase() {
  local phase=$1 config=$2 dump_dir=$3 entry_source=$4 steps=$5 load_path=$6
  local phase_dir="$ATTEMPT_ROOT/$phase"
  local ledger="$phase_dir/owned-processes.jsonl"
  local owner_status="$phase_dir/owner-status.json"
  local owner_record="$phase_dir/owner.json"
  mkdir -m 0700 "$phase_dir"
  cp "$entry_source" "$phase_dir/compiler_entry.py"
  cmp -s "$entry_source" "$phase_dir/compiler_entry.py"
  export GIFT_HYBRID_EP_SCAN_RUNTIME_DIR="$phase_dir/scan-runtime"
  local -a command=(
    "$PYTHON" -m torch.distributed.run
    --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0
    "--log-dir=$phase_dir/torchrun" --redirects=3 --tee=3
    "$phase_dir/compiler_entry.py" "$config" --dump_folder "$dump_dir"
  )
  if [[ -n "$load_path" ]]; then
    command+=(--checkpoint.initial_load_path "$load_path")
  fi
  printf '%q ' "${command[@]}" > "$phase_dir/command.txt"
  printf '\n' >> "$phase_dir/command.txt"

  "$PYTHON" "$PREP_DIR/phase_process_owner.py" \
    --phase "$phase" \
    --parent-pid "$$" --parent-start-ticks "$CONTROLLER_START" \
    --entrypoint "$phase_dir/compiler_entry.py" --cwd "$GIFT_DIR" \
    --ledger "$ledger" --status "$owner_status" --owner-record "$owner_record" \
    --log "$phase_dir/launcher.log" --deadline-seconds 1800 \
    --term-grace 8 --kill-grace 8 \
    -- "${command[@]}" > "$phase_dir/owner.log" 2>&1 &
  ACTIVE_PID=$!
  ACTIVE_START=$("$PYTHON" -c 'from pathlib import Path; import sys; raw=Path(f"/proc/{sys.argv[1]}/stat").read_text(); print(raw[raw.rfind(")")+2:].split()[19])' "$ACTIVE_PID")
  ACTIVE_PHASE=$phase
  ACTIVE_LEDGER=$ledger
  printf 'phase=%s owner_pid=%s owner_start_ticks=%s deadline_seconds=1800\n' \
    "$phase" "$ACTIVE_PID" "$ACTIVE_START" \
    >> "$ATTEMPT_ROOT/lifecycle.log"
  local sample_count=0 owner_state
  while owner_identity_matches; do
    owner_state=$(awk '/^State:/ {print $2}' "/proc/$ACTIVE_PID/status" 2>/dev/null || true)
    [[ "$owner_state" == Z ]] && break
    sample_resources "$phase" "$dump_dir" "$ledger" "$sample_count"
    sample_count=$((sample_count + 1))
    sleep 10
  done
  local run_status
  if wait "$ACTIVE_PID"; then run_status=0; else run_status=$?; fi
  if ! "$PYTHON" "$PREP_DIR/phase_process_owner.py" verify "$ledger" \
    >> "$ATTEMPT_ROOT/lifecycle.log" 2>&1; then
    echo "$phase exited while recorded task identities remain or cannot be verified" >&2
    return 70
  fi
  if owner_identity_matches; then
    echo "$phase phase owner PID/start-tick remains live after wait" >&2
    return 70
  fi
  "$PYTHON" - "$owner_status" "$run_status" <<'OWNER_STATUS'
import json
from pathlib import Path
import sys
status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
assert status["exit_code"] == expected, (status["exit_code"], expected)
assert status["all_recorded_identities_absent"] is True, status
assert status["remaining_identities"] == [], status
OWNER_STATUS
  ACTIVE_PID=
  ACTIVE_START=
  ACTIVE_PHASE=
  ACTIVE_LEDGER=
  printf 'phase=%s run_exit_code=%s\n' "$phase" "$run_status" >> "$ATTEMPT_ROOT/lifecycle.log"
  if (( run_status != 0 )); then
    echo "$phase failed with exit code $run_status; preserving outputs and stopping chain" >&2
    return "$run_status"
  fi
  verify_config "$dump_dir" "$steps" "$load_path"
  verify_compiler_options "$phase_dir"
}

verify_phase_transition_gpu_release() {
  local phase=$1 gpu_file="$ATTEMPT_ROOT/$phase/post-phase-gpu.csv" apps_file="$ATTEMPT_ROOT/$phase/post-phase-apps.csv"
  timeout --signal=TERM --kill-after=2s 10s nvidia-smi \
    --query-gpu=index,uuid,memory.used,utilization.gpu \
    --format=csv,noheader,nounits > "$gpu_file"
  timeout --signal=TERM --kill-after=2s 10s nvidia-smi \
    --query-compute-apps=pid,used_memory --format=csv,noheader,nounits > "$apps_file"
  "$PYTHON" - "$gpu_file" "$apps_file" <<'GPU_RELEASE'
import csv
from pathlib import Path
import sys
expected = [
    "GPU-519db4ab-52c1-7730-5f03-bb8ecba718c3",
    "GPU-f0ff1e03-5cda-3f99-6b5b-22016cc776bf",
    "GPU-35c5887c-b022-4a95-bdc4-f6c6e2175d52",
    "GPU-384c07d4-8c7d-6c6f-c4a8-66a7c18316eb",
    "GPU-d52fef45-0944-a850-6d82-86acee633ce8",
    "GPU-9a1e0c80-7b57-12a4-6220-6a2b9a267466",
    "GPU-5f286803-df1a-a5db-5868-d31d1dde3201",
    "GPU-d03bb98d-079d-b834-e5a0-558188e55667",
]
with Path(sys.argv[1]).open(newline="", encoding="utf-8") as stream:
    rows = [[value.strip() for value in row] for row in csv.reader(stream)]
assert len(rows) == 8 and all(len(row) == 4 for row in rows), rows
assert [row[1] for row in rows] == expected, rows
apps = Path(sys.argv[2]).read_text(encoding="utf-8").strip()
assert not apps or apps == "No running processes found", apps
print("post_phase_gpu_release=verified_exact_8_uuid_no_compute_apps")
GPU_RELEASE
}

run_phase create_s0 "$CREATE_CONFIG" "$S0_DUMP" \
  "$PREP_DIR/compiler_entry_create_s0.py" 1 ""
test -d "$S0_STEP1" || { echo "native S0 step-1 checkpoint is missing" >&2; exit 4; }
verify_probe_manifest "$S0_DUMP/checkpoint/numerical_probe/manifest.json" create_s0 s0 1
test ! -e "$OBSERVE_DUMP/checkpoint" || { echo "observe checkpoint folder must be absent before loading S0" >&2; exit 4; }
verify_phase_transition_gpu_release create_s0

run_phase observe "$OBSERVE_CONFIG" "$OBSERVE_DUMP" \
  "$PREP_DIR/compiler_entry_observe.py" 2 "$S0_STEP1"
verify_probe_manifest "$OBSERVE_DUMP/numerical_probe/manifest.json" observe step_2 2
test ! -e "$OBSERVE_DUMP/checkpoint" || { echo "load_only observe created a checkpoint" >&2; exit 4; }
printf 'native_probe_chain=complete; optimizer_updates=2; timing=unmeasured\n' \
  >> "$ATTEMPT_ROOT/lifecycle.log"
