#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: $0 TASK_ROOT PYTHON DEEPGEMM_PACKAGE_PARENT ATTEMPT_ID [patched|stock]" >&2
  exit 2
}
die() { echo "smoke: $*" >&2; exit 1; }
[[ $# == 4 || $# == 5 ]] || usage

TASK_ROOT=$(cd -- "$1" && pwd -P)
PYTHON=$2
DEEPGEMM_PACKAGE_PARENT=$(cd -- "$3" && pwd -P)
ATTEMPT_ID=$4
MODE=${5:-patched}
[[ $MODE == patched || $MODE == stock ]] || die "mode must be patched or stock"
[[ $ATTEMPT_ID =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid attempt ID"
[[ $PYTHON == /* ]] || die "PYTHON must be an absolute executable path"
[[ -x $PYTHON ]] || die "Python executable not found: $PYTHON"
# The job image exposes its CUDA forward-compatibility libraries through this
# profile; a noninteractive SSH command otherwise loads the older host driver.
[[ -r /etc/profile.d/wm.sh ]] || die "job image runtime profile is unavailable"
source /etc/profile.d/wm.sh
CUDA_HOME_DIR=${CUDA_HOME:-/usr/local/cuda}
[[ -x $CUDA_HOME_DIR/bin/nvcc ]] || die "CUDA nvcc is unavailable under $CUDA_HOME_DIR"

SOURCE_ROOT="$TASK_ROOT/gift"
CONFIG="$SOURCE_ROOT/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml"
PRESET="$SOURCE_ROOT/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml"
PACKAGE="$DEEPGEMM_PACKAGE_PARENT/deep_gemm"
HEADER="$PACKAGE/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh"
ORIGINAL_HEADER_SHA=fd8a620440cc31ae7d01010fde1f914aae6998f2dcf6ede2a65e3b5e76001726
PATCHED_HEADER_SHA=ed8bfb6a182d6a8a225d36b359287f17dba73e95d5cc50150ca177ea979557d2
if [[ $MODE == patched ]]; then
  EXPECTED_HEADER_SHA=$PATCHED_HEADER_SHA
  TRAIN_ENTRYPOINT=torchtitan.experiments.cuda_graph_campaign.train_scan_fix
else
  EXPECTED_HEADER_SHA=$ORIGINAL_HEADER_SHA
  TRAIN_ENTRYPOINT=torchtitan.train
fi
[[ -d $SOURCE_ROOT/torchtitan && -f $SOURCE_ROOT/torchtitan/train.py ]] || die "GIFT source tree is missing"
[[ -f $CONFIG && -f $PRESET ]] || die "canonical experiment or preset is missing"
[[ -f $PACKAGE/__init__.py && -f $HEADER ]] || die "DeepGEMM package or header is missing"

GIT_ROOT=$(git -C "$TASK_ROOT" rev-parse --show-toplevel 2>/dev/null) || die "task root is not a Git repository"
[[ $(cd -- "$GIT_ROOT" && pwd -P) == "$TASK_ROOT" ]] || die "task root is not the Git root"
dirty_source=$(git -C "$TASK_ROOT" status --porcelain=v1 --untracked-files=all -- gift/) || die "could not inspect GIFT source status"
[[ -z $dirty_source ]] || { printf '%s\n' "$dirty_source" >&2; die "GIFT source tree must be clean"; }
HEADER_SHA=$(sha256sum -- "$HEADER" | cut -d ' ' -f 1)
[[ $HEADER_SHA == "$EXPECTED_HEADER_SHA" ]] || die "patched DeepGEMM header SHA-256 mismatch"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
command -v timeout >/dev/null || die "GNU timeout is unavailable"
command -v sha256sum >/dev/null || die "sha256sum is unavailable"
[[ $(timeout --version | sed -n '1p') == *"(GNU coreutils)"* ]] || die "GNU coreutils timeout is required"

ATTEMPT_PARENT="$TASK_ROOT/runs/100b-crop8-ep8-smoke/artifacts/raw"
ATTEMPT="$ATTEMPT_PARENT/$ATTEMPT_ID"
[[ ! -e $ATTEMPT && ! -L $ATTEMPT ]] || die "attempt already exists: $ATTEMPT"
mkdir -p "$ATTEMPT_PARENT"
mkdir "$ATTEMPT"
LOG_DIR="$ATTEMPT/logs"
PREFLIGHT_DIR="$ATTEMPT/preflight"
CONFIG_DIR="$ATTEMPT/config"
NATIVE_DIR="$ATTEMPT/native"
mkdir -p "$LOG_DIR" "$PREFLIGHT_DIR" "$CONFIG_DIR"

record_exit() {
  local status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$ATTEMPT/exit-code.txt" || status=1
  exit "$status"
}
trap record_exit EXIT

cp -- "$CONFIG" "$CONFIG_DIR/source-experiment.yaml"
cp -- "$PRESET" "$CONFIG_DIR/source-model-preset.yaml"
CONFIG_SHA=$(sha256sum -- "$CONFIG" | cut -d ' ' -f 1)
PRESET_SHA=$(sha256sum -- "$PRESET" | cut -d ' ' -f 1)
RESOLVED_CONFIG="$NATIVE_DIR/config/resolved_config.yaml"
{
  printf 'task_head=%s\n' "$(git -C "$TASK_ROOT" rev-parse HEAD)"
  printf 'experiment_sha256=%s\n' "$CONFIG_SHA"
  printf 'model_preset_sha256=%s\n' "$PRESET_SHA"
  printf 'mode=%s\ntraining_entrypoint=%s\n' "$MODE" "$TRAIN_ENTRYPOINT"
  printf 'deepgemm_header_sha256=%s\n' "$HEADER_SHA"
  printf 'deepgemm_original_header_sha256=%s\ndeepgemm_patched_header_sha256=%s\n' "$ORIGINAL_HEADER_SHA" "$PATCHED_HEADER_SHA"
  printf 'resolved_config_path=%s\n' "$RESOLVED_CONFIG"
} > "$ATTEMPT/metadata.txt"

if ! compute_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null); then
  echo "query failed" > "$PREFLIGHT_DIR/compute-apps.txt"
  die "could not inspect GPU compute applications"
fi
if [[ -n ${compute_apps//[[:space:]]/} ]]; then
  echo "non-empty" > "$PREFLIGHT_DIR/compute-apps.txt"
  die "GPU compute applications are already present"
fi
echo "empty" > "$PREFLIGHT_DIR/compute-apps.txt"
nvidia-smi --query-gpu=name,memory.total,compute_mode,driver_version --format=csv,noheader > "$PREFLIGHT_DIR/gpus.csv" || die "could not inspect GPUs"
nvidia-smi topo -m > "$PREFLIGHT_DIR/topology.txt" || die "could not capture GPU topology"
if ! "$CUDA_HOME_DIR/bin/nvcc" --version > "$PREFLIGHT_DIR/nvcc-version.txt" 2>&1; then
  cat "$PREFLIGHT_DIR/nvcc-version.txt" >&2
  die "could not inspect nvcc"
fi
grep -q 'V13.2.51' "$PREFLIGHT_DIR/nvcc-version.txt" || { cat "$PREFLIGHT_DIR/nvcc-version.txt" >&2; die "expected nvcc 13.2.51"; }
if ! "$PYTHON" - "$PREFLIGHT_DIR/gpus.csv" > "$PREFLIGHT_DIR/runtime.txt" 2>&1 <<'PY'
import csv
import sys
import torch

with open(sys.argv[1], newline="", encoding="utf-8") as source:
    rows = list(csv.reader(source))
if len(rows) != 8 or any("H100" not in row[0] for row in rows):
    raise SystemExit("expected exactly eight H100 GPUs")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit("PyTorch does not see eight available CUDA devices")
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
for index in range(8):
    device = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    if "H100" not in device.name or capability != (9, 0):
        raise SystemExit(f"GPU {index} is not an H100/SM90: {device.name} {capability}")
    print(f"gpu{index}={device.name} capability={capability[0]}.{capability[1]}")
PY
then
  cat "$PREFLIGHT_DIR/runtime.txt" >&2
  die "GPU runtime preflight failed"
fi

RAW_CACHE="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw"
HF_CANDIDATE="$RAW_CACHE/NATIVE-SMOKE/001/cache/huggingface"
DG_CANDIDATE="$RAW_CACHE/cache/deepgemm_shared"
HF_HOME_DIR="$ATTEMPT/cache/huggingface"
DG_CACHE_DIR="$ATTEMPT/cache/deepgemm_shared"
if [[ -d $HF_CANDIDATE && ! -L $HF_CANDIDATE ]]; then HF_HOME_DIR=$(cd -- "$HF_CANDIDATE" && pwd -P); fi
if [[ $MODE == patched && -d $DG_CANDIDATE && ! -L $DG_CANDIDATE ]]; then DG_CACHE_DIR=$(cd -- "$DG_CANDIDATE" && pwd -P); fi
case "$HF_HOME_DIR/" in "$TASK_ROOT/"*) ;; *) die "HF cache is outside the task tree" ;; esac
case "$DG_CACHE_DIR/" in "$TASK_ROOT/"*) ;; *) die "DeepGEMM cache is outside the task tree" ;; esac
mkdir -p "$HF_HOME_DIR" "$DG_CACHE_DIR" "$ATTEMPT/cache/torchinductor" "$ATTEMPT/cache/triton"

if [[ $MODE == patched ]]; then
  export PYTHONPATH="$DEEPGEMM_PACKAGE_PARENT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$SOURCE_ROOT"
fi
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$HF_HOME_DIR" HF_HUB_CACHE="$HF_HOME_DIR/hub" HF_DATASETS_CACHE="$HF_HOME_DIR/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$ATTEMPT/cache/torchinductor" TRITON_CACHE_DIR="$ATTEMPT/cache/triton"
export DG_JIT_SHARED_CACHE_DIR="$DG_CACHE_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_CUDAGRAPHS=0
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 GIFT_LOG_RANK=all TORCHFT_LIGHTHOUSE=http://localhost:29510
export CUDA_HOME="$CUDA_HOME_DIR" PATH="$CUDA_HOME_DIR/bin:$PATH"
if [[ $MODE == stock ]]; then
  export HYBRID_EP_CACHE_DIR="$ATTEMPT/cache/hybridep"
  unset GIFT_HYBRID_EP_SCAN_RUNTIME_DIR PROPAGATE_DG_CACHE PROPAGATE_DG_CACHE_DIR
  mkdir -p "$HYBRID_EP_CACHE_DIR"
else
  export GIFT_HYBRID_EP_SCAN_RUNTIME_DIR="$ATTEMPT/scan-runtime"
fi
unset TORCH_LOGS TORCH_COMPILE_DEBUG TORCH_TRACE CUDA_LAUNCH_BLOCKING TITAN_STRUCT_LOGGER_HANDLERS
unset CUDA_ENABLE_COREDUMP_ON_EXCEPTION CUDA_COREDUMP_GENERATION_FLAGS CUDA_COREDUMP_SHOW_PROGRESS CUDA_COREDUMP_FILE
printf 'hf_cache=%s\ndeepgemm_cache=%s\n' "$HF_HOME_DIR" "$DG_CACHE_DIR" >> "$ATTEMPT/metadata.txt"
if ! "$PYTHON" - "$DEEPGEMM_PACKAGE_PARENT" > "$PREFLIGHT_DIR/deepgemm-runtime.txt" 2>&1 <<'PY'
from pathlib import Path
import sys
import deep_gemm

package = Path(sys.argv[1]).resolve() / "deep_gemm"
if Path(deep_gemm.__file__).resolve() != package / "__init__.py":
    raise SystemExit(f"unexpected DeepGEMM package path: {deep_gemm.__file__}")
extension = Path(deep_gemm._C.__file__).resolve()
if extension.parent != package:
    raise SystemExit(f"unexpected DeepGEMM extension path: {extension}")
print(f"package={deep_gemm.__file__}")
print(f"extension={extension}")
print(f"build_version={deep_gemm.__build_version__}")
PY
then
  cat "$PREFLIGHT_DIR/deepgemm-runtime.txt" >&2
  die "DeepGEMM import path verification failed"
fi
if [[ $MODE == stock ]]; then
  if ! "$PYTHON" > "$PREFLIGHT_DIR/hybridep-runtime.txt" 2>&1 <<'PY'
from hashlib import sha256
import importlib.util
from pathlib import Path
import sysconfig

site = Path(sysconfig.get_path("purelib")).resolve()
wrapper = site / "deep_ep" / "hybrid_ep_buffer.py"
backend = wrapper.parent / "backend" / "hybrid_ep_backend.cuh"
spec = importlib.util.find_spec("deep_ep.hybrid_ep_buffer")
if spec is None or spec.origin is None or Path(spec.origin).resolve() != wrapper.resolve():
    raise SystemExit(f"unexpected imported HybridEP wrapper: {None if spec is None else spec.origin}")
for path, expected in ((wrapper, "63e425916b3c6b0c881ad402af35a2b00c4c84203fcde611a6715744b89210cb"), (backend, "b507f65cfa4580f16342076fc2d00afe81e43f64ddefcc4a3cf43681aef11b4b")):
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"unexpected HybridEP source: {path}")
if "thread_local_all_block_local_experts_sum[NUM_OF_EXPERTS_PER_RANK - 1]" not in backend.read_text(encoding="utf-8"):
    raise SystemExit(f"HybridEP backend is missing the E-1 scan: {backend}")
print(f"wrapper={wrapper}\nbackend={backend}")
PY
  then
    cat "$PREFLIGHT_DIR/hybridep-runtime.txt" >&2
    die "installed HybridEP source verification failed"
  fi
fi

cd -- "$SOURCE_ROOT"
set +e
timeout --signal=TERM --kill-after=60s 3600s "$PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 \
  --log_dir "$LOG_DIR/torchrun" --redirects=3 --tee=3 \
  -m "$TRAIN_ENTRYPOINT" \
  configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml \
  --training.steps 10 --lr_scheduler.total_steps 65000 \
  --callbacks.monitoring.log_interval 5 --checkpoint.enable false \
  --validator.enable false --profiler.enable_profiling false \
  --profiler.enable_memory_snapshot false --dump_folder "$NATIVE_DIR" \
  2>&1 | tee "$LOG_DIR/launcher.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
run_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
if (( run_status == 0 && tee_status != 0 )); then run_status=$tee_status; fi
if (( run_status == 0 )) && [[ ! -f $RESOLVED_CONFIG ]]; then
  echo "smoke: training exited successfully without a resolved config" >&2
  run_status=1
fi
exit "$run_status"
