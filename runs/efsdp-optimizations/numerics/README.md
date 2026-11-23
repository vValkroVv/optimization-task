# First-update EP8 numerical check

This diagnostic compares two repeats of the pinned original A with the Expert
GFSDP candidate. Each arm runs one native optimizer update with all 32
accumulation microbatches on eight H100s. It preserves the original dataset and
tokenizer, EP8/eFSDP1, FP8, native FLA and HybridEP, model and loss compilation,
selective activation checkpointing, MTP, and the LR schedule of 65,000 total
and 2,000 warmup updates. The diagnostic sets `debug.deterministic=true` and
keeps `debug.deterministic_warn_only=false`. GIFT applies the deterministic
PyTorch settings before native construction and selects its deterministic
FlexAttention compile mode. This diagnostic has no timing window.

## Plan and run

Use an idle task-owned checkout and the pinned native runtime for the existing
[Expert GFSDP runner](../README.md). The [launcher patch](native-numerics.patch)
targets the unprofiled runner. It belongs on the candidate launcher checkout
only. Apply it after checking the working copy and the patch.

```bash
git apply --check runs/efsdp-optimizations/numerics/native-numerics.patch
git apply runs/efsdp-optimizations/numerics/native-numerics.patch
python -B runs/efsdp-optimizations/run.py plan ep8-num-original --numerical-ep8 \
  --original-a-source "$clean_a_checkout"
python -B runs/efsdp-optimizations/run.py plan ep8-num-expert --numerical-ep8 \
  --direct-dw-build "$verified_build" \
  --optimization bf16-cache --optimization fp8-cache \
  --optimization direct-dw --optimization tail-retention
```

Use `run` in place of `plan` once the native GPU preflight passes and add
`--reference "$reference"` pointing to the completed original A receipt.
Launch two original arms under distinct run IDs. Use fresh per-arm compiler
caches, the same runtime and nodes, and the same reference for all arms. The
original runner verifies the reference, clean GIFT revision, and verified input
bytes. Every rank compares its effective typed config against the launch
digest before native trainer construction. The ordinary native post-run
config, source, rank-completion, and feature-activation checks also run.

Set each path below to its full attempt directory under
`runs/overnight-streaming-perf/artifacts/raw` in the corresponding checkout.

```bash
python -B runs/efsdp-optimizations/numerics/ep8.py "$a1" "$a2" > ep8-control.json
python -B runs/efsdp-optimizations/numerics/ep8.py "$a1" "$a2" "$expert" > ep8-parity.json
```

## Captured data

- Before the first native step, the observer hashes local parameter shards,
  model buffers, existing optimizer state, optimizer groups, schedulers, token
  counts and Python, NumPy, Torch CPU and rank-local CUDA RNG state. The cap is
  16 GiB of state read per rank in chunks of up to 16 MiB. It writes digests and
  ragged-shard metadata. It does not persist whole tensors or capture private
  DTensor tracker internals.
- The observer hashes callback-mutated CPU inputs and labels for all 32
  microbatches. It retains their native unrounded, normalized rank-local loss
  scalars, the native rank-local update loss, global token denominator and
  unclipped gradient norm.
- Immediately before native GFSDP reduction, the observer snapshots 4,096
  stratified FP32 elements of each local `w13_E2FD` and `w2_EDF` gradient in
  routed layers 1, 4 and 7. Capture covers all eight ranks and all 32
  microbatches. It reads the packed accumulation view shared by the baseline
  and direct dW paths before the native reduction can change or release it.
  It follows sharded parameter identity through SAC and compile wrappers.
  The observation is after HybridEP communication and gradient computation.
- The same indices sample the initial weights, pre-clip and post-clip
  optimizer-visible gradients, updated weights, and AdamW moments. Receipts
  also retain the native step counters, scheduler state and the 128-element
  FP32 router count and bias vectors around the optimizer.
- The active stream makes the snapshot into private GPU storage. Pinned host
  copies and their one completion wait follow the complete native update. Raw
  numeric arrays are capped at 4 MiB per rank. The NPZ container adds metadata.
  CPU input hashes run before forward and backward. A diagnostic run changes
  device launch pressure and cannot establish production performance or
  asynchronous event correctness.

## Interpretation

An initial-state, input, recipe, coverage or counter mismatch invalidates the
comparison. If A/A matches exactly, exact A/B attests the captured fields for
one update. If A/A differs, use the earliest differing sampled boundary and
the exact, max-absolute, p99-absolute and relative-L2 metrics to localize the
variation. A single A/A pair cannot establish a statistical tolerance. The
comparator returns status 1 for any numeric difference and retains the detailed
results. It does not infer a precision regression or accept an A/B difference
from a rounded norm.

Each arm must run with the same capture helper bytes. The offline comparator can
read existing captures after a validator correction. Keep the capture helper
already used by A1 and A2 for a later candidate arm in that series.

HybridEP dispatch, combine and their backward operators invoke native
`deep_ep` kernels. PyTorch's deterministic flag does not certify their routing,
permutation or accumulation order. DeepGEMM and other custom kernels also need
empirical repeatability. If initial state matches and the strict path errors or
A/A still differs materially, retain that evidence and localize the difference
with matched operands at the affected expert boundary. If A/A repeats and A/B
differs first at pre-reduction, use the selected layer and microbatch to guide
a matched-input reference check before assigning a cause. The sample omits
most gradient elements and covers the first native EP8 update only.

## Local checks

```bash
python3.12 -B runs/efsdp-optimizations/numerics/test_ep8.py
ruff check --select F821,F811,F841 --no-cache runs/efsdp-optimizations/numerics
```

The plan was checked for both a clean original A root at `e9b7cdbb4` and the
candidate root with the actual bridge and native parsers. Seven CPU fixture
checks passed, including the no-grad requirement for ragged shard reads and
validation of the distinct weight and router vector shapes.
The observer also read and hashed an actual optimizer-owned
`RaggedShardDTensor` under a one-rank Gloo process group.
Both original runs and the expert candidate completed one full update in
the pinned eight-H100 runtime. All rank-local losses matched bitwise.
The sampled pre-reduction gradients varied between the original repeats
and between each original and the candidate. The receipt records the two
controls and both candidate comparisons.
[Observed numerical differences](../artifacts/raw/ep8num-01/ep8-parity-both-controls.json)
