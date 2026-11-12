#!/usr/bin/env bash
set -euo pipefail
umask 077
/opt/torchtitan-venv/bin/python3.12 - <<'SOURCE_GATE'
from pathlib import Path
import hashlib,json
root=Path('/workspace-SR008.nfs2/vkropotin/optimization-task')
raw=root/'runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw'
a=raw/'BASELINE-STABILITY/attempts/002'
release=json.loads((a/'release.json').read_text())
assert release['status']=='ROOT_RELEASED_NATIVE_DIAGNOSTIC_200'
m=raw/'SOURCE-SNAPSHOT/010/manifest.json'
assert hashlib.sha256(m.read_bytes()).hexdigest()=='b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20'
for rel,expected in json.loads(m.read_text())['files'].items():
 assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected['sha256'],rel
assert hashlib.sha256((root/'gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml').read_bytes()).hexdigest()=='b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b'
for name in ['native','torchrun','scan-runtime','cores']:
 assert not (a/name).exists(),name
print('Source010 all98 verified; native graph-off diagnostic200 admitted',flush=True)
SOURCE_GATE
TASK_ROOT=/workspace-SR008.nfs2/vkropotin/optimization-task
GIFT_DIR="$TASK_ROOT/gift"
ATTEMPT_DIR="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/attempts/002"
CACHE_DIR="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NATIVE-SMOKE/001/cache"
CACHE_ROOT="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/cache"
PYTHON=/opt/torchtitan-venv/bin/python3.12
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
export TORCHFT_LIGHTHOUSE=http://localhost:29510
export GIFT_LOG_RANK=all
unset TORCH_LOGS TORCH_COMPILE_DEBUG TORCH_TRACE CUDA_LAUNCH_BLOCKING
unset TITAN_STRUCT_LOGGER_HANDLERS
unset TORCHTITAN_PHASE_MEMORY_RUN_ID TORCHTITAN_PHASE_MEMORY_SOURCE_ID TORCHTITAN_PHASE_MEMORY_CONFIG_ID

export CUDA_VISIBLE_DEVICES="GPU-c5efeb4a-3298-d3d3-d740-9aeaa1cdd60c,GPU-7e4c74eb-40df-7c3c-0886-8d158b9b48d9,GPU-2f380105-fce8-292b-f1a6-54dbea10840a,GPU-7d17ce5f-a48a-731b-4fa1-a8c6eb959117,GPU-1c5ac011-cc53-c978-d10e-03f900f8fac7,GPU-aeea52a7-1100-c642-3465-82d9ab487492,GPU-9974d8fe-53b0-a7e0-68d3-8d0fdd5ed653,GPU-bb2558b3-a4fc-6fce-46a1-45180afa9681"
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_GENERATION_FLAGS=skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory,skip_abort
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_FILE="$ATTEMPT_DIR/cores/core.%p.%t.nvcudmp"
mkdir "$ATTEMPT_DIR/cores"

"$PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
  --log-dir="$ATTEMPT_DIR/torchrun" --redirects=3 --tee=3 \
  -m torchtitan.experiments.cuda_graph_campaign.train_scan_fix \
  configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml \
  --training.steps 200 \
  --lr_scheduler.total_steps 30000 \
  --debug.seed 42 \
  --checkpoint.enable false \
  --validator.enable false \
  --profiler.enable_profiling false \
  --profiler.enable_memory_snapshot false \
  --dump_folder "$ATTEMPT_DIR/native"
