#!/usr/bin/env bash
set -euo pipefail
umask 077

TASK_ROOT=/workspace-SR008.nfs2/vkropotin/optimization-task
CAMPAIGN="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY"
GIFT_DIR="$TASK_ROOT/gift"
DEPENDENCY_DIR="$CAMPAIGN/dependencies/descriptor-fix-001"
ATTEMPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ATTEMPT_ID="${ATTEMPT_DIR##*/}"
CACHE_DIR="$TASK_ROOT/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NATIVE-SMOKE/001/cache"
PYTHON=/opt/torchtitan-venv/bin/python3.12

case "$ATTEMPT_ID" in
  003|004) ;;
  *) echo "unsupported native200 attempt directory: $ATTEMPT_ID" >&2; exit 2 ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DEPENDENCY_DIR:$GIFT_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - "$ATTEMPT_ID" <<'SOURCE_GATE'
from pathlib import Path
import hashlib
import json
import sys

root = Path('/workspace-SR008.nfs2/vkropotin/optimization-task')
raw = root / 'runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw'
campaign = raw / 'BASELINE-STABILITY'
attempt = campaign / 'attempts' / sys.argv[1]
release = json.loads((attempt / 'release.json').read_text())
assert release.get('status') == 'ROOT_RELEASED_REPAIRED_NATIVE_200'

source_manifest_path = raw / 'SOURCE-SNAPSHOT/010/manifest.json'
assert hashlib.sha256(source_manifest_path.read_bytes()).hexdigest() == (
    'b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20'
)
source_manifest = json.loads(source_manifest_path.read_text())
assert len(source_manifest['files']) == 98
for relative, expected in source_manifest['files'].items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected['sha256'], relative
assert hashlib.sha256(
    (root / 'gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml').read_bytes()
).hexdigest() == 'b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b'

dependency = campaign / 'dependencies/descriptor-fix-001'
original_path = dependency / 'original-manifest.json'
candidate_path = dependency / 'candidate-manifest.json'
assert hashlib.sha256(original_path.read_bytes()).hexdigest() == (
    '1fb37e4faf9d55f92a736a3d81aa17613de68a767277f981b6733342f771de23'
)
assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == (
    'e1edfcef5135d401b402b8497bb103c69c55f8c896b09d7abc343dbc33dcafa7'
)
original = json.loads(original_path.read_text())
candidate = json.loads(candidate_path.read_text())
header = 'include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh'
extension = '_C.cpython-312-x86_64-linux-gnu.so'
assert original['package'] == '/opt/torchtitan-venv/lib/python3.12/site-packages/deep_gemm'
assert original['build'] == candidate['base_build'] == '2.3.0+f093a4c'
assert candidate['changed_files'] == [header]
assert len(original['files']) == len(candidate['files']) == 844
assert set(original['files']) == set(candidate['files'])
assert original['files'][header]['sha256'] == (
    'fd8a620440cc31ae7d01010fde1f914aae6998f2dcf6ede2a65e3b5e76001726'
)
assert candidate['files'][header]['sha256'] == (
    'ed8bfb6a182d6a8a225d36b359287f17dba73e95d5cc50150ca177ea979557d2'
)
assert original['files'][extension]['sha256'] == candidate['files'][extension]['sha256'] == (
    '99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26'
)
changed = {
    name for name in original['files']
    if original['files'][name]['sha256'] != candidate['files'][name]['sha256']
}
assert changed == {header}


def verify_files(package_root: Path, manifest: dict, label: str) -> None:
    package_root = package_root.resolve(strict=True)
    if not package_root.is_dir():
        raise RuntimeError(f'{label} package root is not a directory: {package_root}')
    for relative, expected in manifest['files'].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise RuntimeError(f'{label} manifest path escapes package root: {relative}')
        path = package_root / relative_path
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(package_root) or not resolved.is_file():
            raise RuntimeError(f'{label} manifest path is not a package file: {relative}')
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected['sha256']:
            raise RuntimeError(f'{label} SHA-256 mismatch: {relative}')


verify_files(Path(original['package']), original, 'original DeepGEMM')
verify_files(dependency / 'deep_gemm', candidate, 'candidate DeepGEMM')

for name in ('native', 'torchrun', 'scan-runtime', 'deepgemm-cache',
             'deepgemm-shared-cache', 'cores'):
    output = attempt / name
    assert not output.exists() and not output.is_symlink(), f'refusing existing output: {name}'

print('Source010 all98 and preset verified; original/candidate DeepGEMM all844 verified; repaired native200 admitted', flush=True)
SOURCE_GATE

cd "$GIFT_DIR"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$CACHE_DIR/huggingface/datasets"
export HF_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export DG_JIT_CACHE_DIR="$ATTEMPT_DIR/deepgemm-cache"
export DG_JIT_SHARED_CACHE_DIR="$ATTEMPT_DIR/deepgemm-shared-cache"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export GIFT_HYBRID_EP_SCAN_RUNTIME_DIR="$ATTEMPT_DIR/scan-runtime"
export TORCHINDUCTOR_CUDAGRAPHS=0
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
export TORCHFT_LIGHTHOUSE=http://localhost:29510
export GIFT_LOG_RANK=all
unset DG_CACHE_FORCE_SKIP DG_JIT_DEBUG DG_PRINT_CONFIGS DG_SILENT_CACHE_MISS
unset CUDA_ENABLE_COREDUMP_ON_EXCEPTION CUDA_COREDUMP_GENERATION_FLAGS CUDA_COREDUMP_SHOW_PROGRESS
unset CUDA_COREDUMP_FILE CUDA_LAUNCH_BLOCKING TORCH_LOGS TORCH_COMPILE_DEBUG TORCH_TRACE
unset TITAN_STRUCT_LOGGER_HANDLERS
unset TORCHTITAN_PHASE_MEMORY_RUN_ID TORCHTITAN_PHASE_MEMORY_SOURCE_ID TORCHTITAN_PHASE_MEMORY_CONFIG_ID

export CUDA_VISIBLE_DEVICES="GPU-c5efeb4a-3298-d3d3-d740-9aeaa1cdd60c,GPU-7e4c74eb-40df-7c3c-0886-8d158b9b48d9,GPU-2f380105-fce8-292b-f1a6-54dbea10840a,GPU-7d17ce5f-a48a-731b-4fa1-a8c6eb959117,GPU-1c5ac011-cc53-c978-d10e-03f900f8fac7,GPU-aeea52a7-1100-c642-3465-82d9ab487492,GPU-9974d8fe-53b0-a7e0-68d3-8d0fdd5ed653,GPU-bb2558b3-a4fc-6fce-46a1-45180afa9681"

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
