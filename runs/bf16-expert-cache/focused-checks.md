# Focused BF16 expert-cache checks

This record is limited to the deterministic actual-GFSDP fixture. The feature
implementation and full accepted training crop have separate validation gates.

## Fixture and acceptance

`gift/tests/unit_tests/test_gfsdp_bf16_expert_cache.py` uses the GIFT debug MoE
model with three sequential `mla_moe` blocks. It applies selective activation
checkpointing and the real GIFT GFSDP integration, with a two-rank dense mesh,
EP degree two, and singleton expert-data-parallel groups. The fixture compares
cache off/on from identical initialized FP32 masters. Inputs and labels are
deterministic; seed is **78123**, set on each model immediately before
`init_weights` after `to_empty`, using `GFSDP2Backend.weight_init_context` in the
same order as `Trainer`. It uses BF16 parameters and FP32 reduction, two
microbatches per update, two AdamW fused updates, and the actual
`OptimizersContainer` hooks. Small hidden/sequence/vocabulary sizes are
`256/4/2048`.

The main test checks exact loss, input-gradient, elementwise expert-gradient and
updated FP32-master parity; active-master-to-BF16 casts at consumers; SAC
recomputation; eligible expert-group selection, output/input aliasing, separate
persistent buffers for groups 0 and 2, generation counters and dense size-two
fallback. It exercises ordinary partial `load_state_dict`, a temporary actual
GFSDP DCP save/load, EMA swap in/back, then a same-dtype value-changing `_apply`
followed by identity checks and another real optimizer step with parity checks.
A second distributed test covers the all-gather-extension fallback. A small
configuration test asserts the feature default is off.

The distributed tests use `DTensorTestBase(world_size=2)` and
`@with_comms`; pytest itself spawns two local processes and binds them to device
indices 0 and 1. No `torchrun` wrapper is needed.

## Checks and command

Completed source-only checks:

- `/opt/torchtitan-venv/bin/ruff check --select F821,F811,F841 --no-cache gift/tests/unit_tests/test_gfsdp_bf16_expert_cache.py` — passed.
- `/opt/torchtitan-venv/bin/python3.12 -m py_compile gift/tests/unit_tests/test_gfsdp_bf16_expert_cache.py` — passed.
- `git diff --check -- gift/tests/unit_tests/test_gfsdp_bf16_expert_cache.py` — passed.

The test writes temporary DCP checkpoints under the ignored task-local
`runs/bf16-expert-cache/artifacts/raw/` directory and removes them only after a
fully successful main test. Failed-test checkpoints are retained.

## Runtime result

Parent preflight on 13 November 2026 confirmed existing one-node allocation
`lm-mpi-job-8dff1e19-674b-42ed-96a9-bcd042dfefe5`, task NFS4 route, and all eight
H100 80 GB GPUs at 0 MiB with no compute applications. This fixture uses only
local devices 0 and 1; existing task Perfetto servers were left untouched.

The first authorized command ran from `gift/` with `/etc/profile.d/wm.sh`
sourced and exited 1: **2 failed, 1 passed** in 16 seconds. Full output is
[pytest-cache-fixture-001.log](artifacts/raw/pytest-cache-fixture-001.log). It
showed that direct `torch.manual_seed` did not initialize the DTensor RNG tracker
required by `RaggedShardDTensor`; the fixture then adopted DTensor's
`manual_seed(seed, mesh)`, matching Trainer's tracker setup. It also showed the
extension methods on the original `nn.Parameter` did not survive sharding.

Exact command:

```bash
source /etc/profile.d/wm.sh
set -o pipefail
/opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  tests/unit_tests/test_gfsdp_bf16_expert_cache.py \
  --basetemp=../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-001 \
  2>&1 | tee ../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-001.log
```

The second authorized command used the same profile, but no explicit task cache
overrides. The resolved values were `TMPDIR` and `XDG_CACHE_HOME` both set to
`/workspace-SR008.nfs2/vkropotin/.sr008-gpu/.cache`; Inductor, Triton, CUDA and
Torch extension cache variables were unset. It exited 1 after **2 failed,
1 passed** in 310.50 seconds. Full output is
[pytest-cache-fixture-002.log](artifacts/raw/pytest-cache-fixture-002.log). The
extension stand-in reached GFSDP's hooks, but its `post_all_gather` asserted
metadata was always present; GFSDP calls that hook during lazy initialization
with `None`, before a consumer calls `pre_all_gather`. The main test did not
produce an interpretable freshness result: rank 1 remained in CUDA host-value
checking inside `torch.testing.assert_close` at the first post-DCP consumer when
the 300-second timeout fired; rank 0 was waiting in the test's unconditional
cleanup barriers.

Exact command:

```bash
source /etc/profile.d/wm.sh
set -o pipefail
/opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  tests/unit_tests/test_gfsdp_bf16_expert_cache.py \
  --basetemp=../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-002 \
  2>&1 | tee ../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-002.log
```

The fixture now accepts the documented initial `None` metadata while checking
that consumer hooks receive the metadata from `pre_all_gather`. Cleanup barriers
and DCP removal run only after a fully successful test; failures retain the
checkpoint and propagate the original exception. Scoped distributed test
timeouts are 90 seconds for the main fixture and 60 seconds for the extension
fixture.

For the next two runs, the profile's default cache paths were overridden to
task-local paths under
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/bf16-expert-cache/artifacts/raw/`:
`TMPDIR=tmp`, `XDG_CACHE_HOME=xdg-cache`,
`TORCHINDUCTOR_CACHE_DIR=torchinductor`, `TRITON_CACHE_DIR=triton`,
`CUDA_CACHE_PATH=cuda`, and `TORCH_EXTENSIONS_DIR=torch-extensions`. Both ran
after a fresh one-node/all-eight-GPU preflight; all GPUs were idle, and only
devices 0 and 1 were used by pytest.

The authorized targeted main-test command was:

```bash
source /etc/profile.d/wm.sh
export BF16_CACHE_RAW=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/bf16-expert-cache/artifacts/raw
export TMPDIR="$BF16_CACHE_RAW/tmp"
export XDG_CACHE_HOME="$BF16_CACHE_RAW/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$BF16_CACHE_RAW/torchinductor"
export TRITON_CACHE_DIR="$BF16_CACHE_RAW/triton"
export CUDA_CACHE_PATH="$BF16_CACHE_RAW/cuda"
export TORCH_EXTENSIONS_DIR="$BF16_CACHE_RAW/torch-extensions"
set -o pipefail
/opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  tests/unit_tests/test_gfsdp_bf16_expert_cache.py \
  -k test_cache_matches_uncached_training_and_weight_lifecycles \
  --basetemp=../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-003 \
  2>&1 | tee ../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-003-main.log
```

It exited 1 after **1 failed, 2 deselected** in 95.77 seconds, using the scoped
90-second distributed timeout. The two-rank trace placed the failure at the
first post-DCP consumer in `_assert_fresh_bf16_cast`. `with_comms` caught an
exception, destroyed the process group, then re-raised; this teardown masked the
rank's original exception, so the log does not establish whether `assert_close`
found a mismatch. The peer remained in CUDA host-value synchronization. The two
DCP checkpoint artifacts were retained. Full log:
[pytest-cache-fixture-003-main.log](artifacts/raw/pytest-cache-fixture-003-main.log).

The separately targeted extension command was:

```bash
source /etc/profile.d/wm.sh
set -o pipefail
export BF16_CACHE_RAW=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/bf16-expert-cache/artifacts/raw
export TMPDIR="$BF16_CACHE_RAW/tmp"
export XDG_CACHE_HOME="$BF16_CACHE_RAW/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$BF16_CACHE_RAW/torchinductor"
export TRITON_CACHE_DIR="$BF16_CACHE_RAW/triton"
export CUDA_CACHE_PATH="$BF16_CACHE_RAW/cuda"
export TORCH_EXTENSIONS_DIR="$BF16_CACHE_RAW/torch-extensions"
/opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  tests/unit_tests/test_gfsdp_bf16_expert_cache.py \
  -k test_all_gather_extension_keeps_original_output_buffer \
  --basetemp=../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-004 \
  2>&1 | tee ../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-004-extension.log
```

It exited 0:
**1 passed, 2 deselected** in 12.61 seconds. This verifies the extension
pre/post hooks run on an actual GFSDP local tensor and the group retains distinct
gather input/output buffers with caching disabled. Full output is
[pytest-cache-fixture-004-extension.log](artifacts/raw/pytest-cache-fixture-004-extension.log).

The current fixture adds `_report_rank_exception` inside `@with_comms` on the
main test. It prints and flushes the original rank traceback before the wrapper
tears down the process group. This changes only failure reporting; the consumer
hook boundary and exact cast check are unchanged. The parent is reviewing this
boundary before any further main-test execution.

After parent review, one main-only diagnostic rerun used the same task-local
cache exports with `--basetemp=../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-005`
and tee output `../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-005-main.log`.

Exact command:

```bash
source /etc/profile.d/wm.sh
set -o pipefail
export BF16_CACHE_RAW=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/bf16-expert-cache/artifacts/raw
export TMPDIR="$BF16_CACHE_RAW/tmp"
export XDG_CACHE_HOME="$BF16_CACHE_RAW/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$BF16_CACHE_RAW/torchinductor"
export TRITON_CACHE_DIR="$BF16_CACHE_RAW/triton"
export CUDA_CACHE_PATH="$BF16_CACHE_RAW/cuda"
export TORCH_EXTENSIONS_DIR="$BF16_CACHE_RAW/torch-extensions"
/opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  tests/unit_tests/test_gfsdp_bf16_expert_cache.py \
  -k test_cache_matches_uncached_training_and_weight_lifecycles \
  --basetemp=../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-005 \
  2>&1 | tee ../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-005-main.log
```

It exited 1 after **1 failed, 2 deselected** in 95.67 seconds. The newly printed
rank-0 traceback identifies the failing assertion: immediately after
`GFSDPCheckpointAdapter.load(..., load_model=True)`, exact local FP32 masters do
not match the snapshots captured before DCP save. `torch.testing.assert_close`
reported **1,572,665 of 1,572,864** elements different, greatest absolute
difference `0.003996892832219601` at index `(225242,)`, and greatest relative
difference `583439.9375` at `(1536933,)`. Rank 1 proceeded to the next consumer
and remained there while the distributed harness timed out. The pre-DCP
two-update loss/input-gradient/expert-gradient/master parity, ordinary load
consumer, and DCP save/load invocation had executed before this assertion; the
post-DCP freshness, EMA and `_apply` checks were not reached.

The retained run005 DCP checkpoint is
[`focused-dcp-a9f17d709cd84320adac17f40f9481de`](artifacts/raw/focused-dcp-a9f17d709cd84320adac17f40f9481de).
The `checkpoint_masters` expected snapshot existed only in process memory; the
DCP artifact is the saved expected state. The full run output is
[pytest-cache-fixture-005-main.log](artifacts/raw/pytest-cache-fixture-005-main.log).
The independent review diagnosed the mismatch as a fixture topology error, not a
production checkpoint bug: the test passed `ep_degree=2` to GFSDP without first
running GIFT `model.parallelize(parallel_dims)`, so expert parameters remained
full replicas on both ranks. DCP therefore saw overlapping replica chunks and
legitimately restored one rank's copy. No source checkpoint fix or tolerance
change is indicated.

The fixture now creates one consistent `ParallelDims` mesh, configures and runs
GIFT model parallelization before AC/GFSDP, and asserts expert `DTensor`
`Shard(0)` placement both before and after sharding. Rank-specific inputs and
the exact FP32 DCP roundtrip comparison are preserved. Static checks pass; this
topology correction had not yet run on GPU at that point; the later run006 and
run007 records below capture its authorized executions.

The topology fixture's device lifecycle was checked against the production
Trainer path before rerun. Trainer builds the model under `torch.device("meta")`,
then applies GIFT parallelization/GFSDP before `to_empty` and weight
initialization (`gift/torchtitan/trainer.py`). GFSDP's `fully_shard` moves the
managed parameters to the mesh device before constructing `GFSDPParam`
(`gift/torchtitan/distributed/gfsdp/fully_shard.py` and `_init.py`), satisfying
the constructor's device assertion. The fixture now uses the same meta build
and later `to_empty`/backend initialization order, with an assertion that
parameters are meta immediately after build. Ruff F821/F811/F841, Python
compilation, and `git diff --check` pass after this correction. This static
path check preceded the GPU reruns below.

After parent authorization, run006 used one main-only rerun. Immediately
before launch, `nvidia-smi` showed GPUs 0 and 1 at 0 MiB and 0% utilization,
with no compute apps; all eight GPUs were idle. The run used the existing
one-node allocation, two test ranks on GPUs 0 and 1, seed `78123`, the shared
EP/DP `ParallelDims` topology, and task-owned temporary and compiler caches.
The exact command was:

```bash
source /etc/profile.d/wm.sh
set -o pipefail
export BF16_CACHE_RAW=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/bf16-expert-cache/artifacts/raw
mkdir -p "$BF16_CACHE_RAW/tmp" "$BF16_CACHE_RAW/xdg-cache" "$BF16_CACHE_RAW/torchinductor" "$BF16_CACHE_RAW/triton" "$BF16_CACHE_RAW/cuda" "$BF16_CACHE_RAW/torch-extensions"
export TMPDIR="$BF16_CACHE_RAW/tmp"
export XDG_CACHE_HOME="$BF16_CACHE_RAW/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$BF16_CACHE_RAW/torchinductor"
export TRITON_CACHE_DIR="$BF16_CACHE_RAW/triton"
export CUDA_CACHE_PATH="$BF16_CACHE_RAW/cuda"
export TORCH_EXTENSIONS_DIR="$BF16_CACHE_RAW/torch-extensions"
/opt/torchtitan-venv/bin/python3.12 -m pytest -q tests/unit_tests/test_gfsdp_bf16_expert_cache.py -k test_cache_matches_uncached_training_and_weight_lifecycles --basetemp=../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-006 2>&1 | tee ../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-006-main.log
```

Run006 exited 1 after **1 failed, 2 deselected** in 19.41 seconds. It passed
the DCP FP32 master roundtrip, post-DCP freshness, EMA swap-in/back freshness
and master checks, and `_apply` Parameter identity and equal-master checks. It
then failed on the actual post-`_apply` optimizer-step comparison:
`post_apply.loss` at test line 598, with exact comparison (`rtol=0`, `atol=0`).
The logged assertion does not include a difference magnitude, so the numerical
gap is unresolved. Full output is
[pytest-cache-fixture-006-main.log](artifacts/raw/pytest-cache-fixture-006-main.log).

The retained run006 checkpoint is
[`focused-dcp-0d1c0810df6f429bb8d6aba894c7dc70`](artifacts/raw/focused-dcp-0d1c0810df6f429bb8d6aba894c7dc70).
Read-only `FileSystemReader.read_metadata()` inspection confirms actual EP
partition chunks: for each of the three layers, `w13_E2FD` has global length
1,048,576 and nonoverlapping chunks `(offset 0, length 524,288)` and
`(offset 524,288, length 524,288)`; `w2_EDF` has global length 524,288 and
chunks at offsets 0 and 262,144, each length 262,144. EMA expert tensors have
the same two-way partition. The earlier run005 DCP mismatch was therefore
resolved by the fixture's real EP topology; no tolerance or source checkpoint
change was made.

The default-off assertion passed in the two whole-file attempts, and the
extension test passed separately. After the run006 failure, the fixture added
a comparison of all GFSDP groups' FP32 main masters before the post-apply
optimizer step, plus dense and expert fresh-cast assertions at the existing
post-GFSDP consumer hook. Exact post-step comparisons were unchanged. Ruff
F821/F811/F841, Python compilation, and `git diff --check` passed for those
test edits.

After the source-side `_apply` invalidation and post-write ordering fix, parent
authorized one main-only run007. The immediate idle preflight found GPUs 0 and
1 at 0 MiB and 0% utilization, no compute apps, and all eight GPUs idle. It
used the same existing one-node allocation, seed `78123`, two ranks on GPUs 0
and 1, shared EP/DP `ParallelDims` topology and task-owned caches as run006.
Exact command:

```bash
source /etc/profile.d/wm.sh
set -o pipefail
export BF16_CACHE_RAW=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/bf16-expert-cache/artifacts/raw
mkdir -p "$BF16_CACHE_RAW/tmp" "$BF16_CACHE_RAW/xdg-cache" "$BF16_CACHE_RAW/torchinductor" "$BF16_CACHE_RAW/triton" "$BF16_CACHE_RAW/cuda" "$BF16_CACHE_RAW/torch-extensions"
export TMPDIR="$BF16_CACHE_RAW/tmp"
export XDG_CACHE_HOME="$BF16_CACHE_RAW/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$BF16_CACHE_RAW/torchinductor"
export TRITON_CACHE_DIR="$BF16_CACHE_RAW/triton"
export CUDA_CACHE_PATH="$BF16_CACHE_RAW/cuda"
export TORCH_EXTENSIONS_DIR="$BF16_CACHE_RAW/torch-extensions"
/opt/torchtitan-venv/bin/python3.12 -m pytest -q tests/unit_tests/test_gfsdp_bf16_expert_cache.py -k test_cache_matches_uncached_training_and_weight_lifecycles --basetemp=../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-007 2>&1 | tee ../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-007-main.log
```

Run007 passed: **1 passed, 2 deselected** in 20.74 seconds (pytest process
21.99 seconds). The run exercised actual two-rank GIFT EP and GFSDP, the
default-off and eligible/fallback structures, two accumulated microbatches
across two optimizer updates with SAC recomputation, exact cached/uncached
loss/input-gradient/expert-gradient/master parity, and exact fresh BF16 casts
for dense and expert groups at each consumer. It also passed ordinary weight
load, actual DCP FP32 master roundtrip, EMA swap-in/back, `_apply` identity and
all-group master checks, and an optimizer step after `_apply` with exact
post-step loss, input-gradient, expert-gradient and updated FP32 expert-master
parity. Full run output is
[pytest-cache-fixture-007-main.log](artifacts/raw/pytest-cache-fixture-007-main.log).

Run006 remains recorded as a failed behavioral regression that exposed stale
dense fallback input after `_apply`. Run007 passed after the source correction;
the retained run006 checkpoint metadata establishes two nonoverlapping EP
chunks per expert and EMA expert tensor. The focused main, default-off and
extension checks passed.

Parent requested one additional focused `_apply` stream-order regression after
run007. The fixture now runs the same value-changing apply on a separate CUDA
writer stream for each of the cached and uncached models. Each writer waits for
the current stream's incoming writes, uses `torch.cuda._sleep(1_000_000)` when
available, then applies `+0.125`; the same input is immediately forwarded on
the default stream without a caller wait. The existing all-group BF16
freshness hooks, all-master equality, Parameter identity and exact optimizer
comparisons remain. The pinned interpreter's `torch.cuda._sleep(cycles)` is
defined in `torch/cuda/__init__.py` and calls `torch._C._cuda_sleep(cycles)`;
there is no GIFT-local sleep helper. Ruff F821/F811/F841, Python compilation
and `git diff --check` pass after this test-only change. This stream case has
passed static review by the parent and reviewer. Parent then authorized one
main-only run008. Immediately before launch, `nvidia-smi` showed GPUs 0 and 1 at
0 MiB and 0% utilization, with no compute apps; all eight GPUs were idle. The
run used the same seed, two-rank topology, allocation, and task-local cache
paths as run007. Exact command:

```bash
source /etc/profile.d/wm.sh
set -o pipefail
export BF16_CACHE_RAW=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/bf16-expert-cache/artifacts/raw
mkdir -p "$BF16_CACHE_RAW/tmp" "$BF16_CACHE_RAW/xdg-cache" "$BF16_CACHE_RAW/torchinductor" "$BF16_CACHE_RAW/triton" "$BF16_CACHE_RAW/cuda" "$BF16_CACHE_RAW/torch-extensions"
export TMPDIR="$BF16_CACHE_RAW/tmp"
export XDG_CACHE_HOME="$BF16_CACHE_RAW/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$BF16_CACHE_RAW/torchinductor"
export TRITON_CACHE_DIR="$BF16_CACHE_RAW/triton"
export CUDA_CACHE_PATH="$BF16_CACHE_RAW/cuda"
export TORCH_EXTENSIONS_DIR="$BF16_CACHE_RAW/torch-extensions"
/opt/torchtitan-venv/bin/python3.12 -m pytest -q tests/unit_tests/test_gfsdp_bf16_expert_cache.py -k test_cache_matches_uncached_training_and_weight_lifecycles --basetemp=../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-008 2>&1 | tee ../runs/bf16-expert-cache/artifacts/raw/pytest-cache-fixture-008-main.log
```

Run008 passed: **1 passed, 2 deselected** in 19.68 seconds (pytest process
20.92 seconds). The auxiliary-stream value-changing `_apply` completed for both
cached and uncached models, then each used the same input for an immediate
default-stream no-grad forward without a caller wait. All-group dense/expert
freshness, FP32 master equality, stable Parameter identities, and exact
post-step loss/input-gradient/expert-gradient/master comparisons passed. The
full output is
[pytest-cache-fixture-008-main.log](artifacts/raw/pytest-cache-fixture-008-main.log).

The focused main lifecycle test, default-off check, and extension fallback
check have all passed. Run006 remains as the recorded regression failure that
motivated the dense fallback invalidation fix; run007 passed the initial source
fix, and run008 passed the final nondefault-stream regression. No further GPU
execution is authorized by this fixture task.
