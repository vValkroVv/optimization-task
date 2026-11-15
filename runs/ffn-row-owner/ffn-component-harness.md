# Native FFN component harness

## Acceptance and current evidence

The bounded deliverable is a task-local runnable native factory for `layers.0`
and `mtp.layers.0`, matched A/A/B local numerical checks, complete component
F/replay/B/communication timing, and bounded all-rank schedule diagnostics.
GPU execution belongs to the coordinator after source freeze. This worker has
executed **no GPU operation**. Current implementation baseline is root revision
`2ada87125e433e1f76c4cb2bfad244b641a78a93` plus the integrated task diff; every
receipt records HEAD, every active `gift/torchtitan/**/*.py` SHA256, harness
SHA256, arguments, resolved config, software and hardware.

The CPU factory plans passed for both components. They instantiate native
`FP8Linear` w1/w2/w3 with `[10944,4096]`, `[4096,10944]`, `[10944,4096]` weights.
Selected native parameter counts are 202,424,512 (layer0) and 234,271,232 (MTP).
The seven CPU tests cover native meta construction, FQNs, FP32 master dtype,
independent controls, unchanged original eight-layer configuration, error
report coverage and bounded CLI options. Scoped Ruff passes. These checks do
not establish CUDA, NCCL, Inductor execution, FP8 numerical acceptance or overlap.

Initial retained failure: config loading from the repository root rejected the
relative tokenizer path and WORLD_SIZE1. The harness now supplies the actual
absolute tokenizer asset path and temporarily supplies WORLD_SIZE8 only during
config preflight; it restores the caller environment and independently requires
real torchrun WORLD_SIZE8 before GPU execution. No download/install was used.
Raw plans/logs are under `artifacts/raw/ffn-plan/` and `artifacts/raw/`.

## Factory and supported scope

`resolved_config` uses `load_trainer_config` on the canonical experiment and
`model_spec.model.owner().prepare_config`. This is the real materialized config,
including native FP8 conversion, initializer specifications, positional block
configuration, sharding contracts and the current SAC config. The task recipe
prefetch32, epoch102400 and BF16 expert reuse are resolved explicitly. Routing,
CE and loader execution lie outside these two dense components.

`meta_component` copies that prepared configuration, selects its original placed
layer0 and MTP, builds the native model on meta, then removes unused registered
tables/head/norm and unrelated blocks before materialization. No MoE dispatcher
is built. The original resolved config is retained unmodified in the receipt.
The native root, RoPE cache and original source FQNs survive. The task-local root
forward calls the exact selected block. MTP uses its native container, preserving
the shared embedding's left shift and zero tail; both hidden and unshifted
embedding inputs remain differentiation inputs. Production FFN dimensions,
FP8 recipes, BF16 compute, FP32 masters/reductions, root masks and positions,
Inductor fullgraph and resolved SAC remain active.

`build` calls the actual model-spec `parallelize_fn`, including native EP8 /
denseGFSDP8 mesh handling, `apply_ac`, `apply_compile`, `apply_gfsdp`, shard order,
prefetch order and reduction factor1. It materializes with `to_empty`, then uses
`GFSDP2Backend.weight_init_context(match_fsdp2_init=True)` and native
`init_weights`, matching Trainer's ordering. CUDA Graphs are disabled with the
same `TORCHINDUCTOR_CUDAGRAPHS=0` environment as the native campaign. Native
`TORCHINDUCTOR_MIX_ORDER_REDUCTION=0`, expandable-segment allocator settings and
`set_determinism`/DTensor seed setup are also retained. All compiler,
Triton and TileLang caches live in the run-owned output directory. The real
FlashQLA `install_backend('external_gate')` runs on every GPU rank.

Only the selected block, parameter-free root/RoPE, its native GFSDP buffers and
synthetic BF16 inputs/upstream gradients occupy the component memory scope.
No full embedding table, output head or seven unrelated trunk/MoE blocks remain.
This is a block VJP workload, not model/loss/optimizer throughput. Native loss
config, MTP weight0.3 and LR65000/warmup2000 are recorded but the model loss,
scheduler, loader, EMA and native checkpoint lifecycle do not run here.
Synthetic block weights are initialized from seed2048; they are matched across
A/A/B, not claimed to equal a full Trainer initialization after every unrelated
module has consumed RNG. Inputs use seed2048 + rank*1000 + microbatch, B2S4096D4096,
int64 contiguous full-document positions, BF16 upstream gradients divided by32.

## Numerical mode

`check` builds A, A/A and B sequentially with identical logical FP32 initial
parameters (elementwise equality is a precondition). Every update compares
selected block outputs, hidden-input gradients, both MTP input gradients, every
logical parameter gradient, clipped/updated FP32 masters and fused AdamW moment
state. Two updates reuse each model's optimizer, so the second update exercises
existing moments and weight decay. `--accumulation 32` accumulates every
microbatch's parameter gradients in the native path before clipping/Adam.
Output/input-gradient captures are from the final microbatch of each update;
all parameter gradients cover the complete accumulation window. The ordinary
`--accumulation 1` check isolates the complete affected local VJP.

CPU reference tensors remain only in this process; no full-model tensor export
framework or raw tensor dump is created. Per-tensor JSON reports shape/dtype,
unequal count, nonfinite count, max absolute error, RMS and relative L2, separately
for A/A and A/B. Chunked FP64 CPU error arithmetic limits temporary allocations.
No universal FP8 tolerance is imposed. Nonzero differences require interpretation
against A/A, actual rounding/reduction changes and, if needed, an affected-block
higher-precision reference. A completed check command is not by itself a
numerical acceptance verdict. Component clipping uses the actual EP-aware
clipping routine; this is not mixed embedding-owner clipping coverage.

## Timing, buffers and diagnostic mode

`benchmark` runs ten complete warmup F/B calls by default, then ten samples.
Inputs are prepared outside timing. Each sample includes zero_grad, mask setup,
selected native forward, SAC replay, backward, packing and all required native
GFSDP communications. A pre-sample distributed barrier is outside the timer;
`torch.cuda.synchronize` waits for all device streams before and after the
measured work. No profiler or observer is installed. The receipt has each rank's
sample series/median/min/max, per-sample slowest rank, and rank-local peak
allocated/reserved bytes. Startup construction/materialization/lazy-init time
is separate; lazy compilation is warmed before samples. No integrated speedup
may be inferred from isolated kernel sums.

`buffer_inventory` counts unique native data/gradient storage bytes by
`untyped_storage().data_ptr()` and records each view's byte offset/length,
FQN and dtype. This supports the superseding requirement that the FFN and
remainder use aligned non-overlapping views of existing alternating slots with
**no storage growth**, including the buffer-region owner's new contexts.
It makes no private-parent assumption. Compare control/split unique storage
bytes and lifecycle peaks; inspect overlapping view intervals and release events
in addition to totals. Standalone component scope does not establish the full
root's table-dominated pool budget or row-owner on/off combinations.

`diagnostic` warms unobserved compute first, then instruments the native GFSDP
eager pre-forward and post-backward methods, outside compiled compute. It records
FFN input-gradient arrival, all parameter gradients present before post-backward,
exactly one FFN reduce call per microbatch and input-gradient-before-reduce order.
It captures at most three complete microbatches on every rank with native
`GFSDP::post_backward_reduce` ranges. Traces and JSON are rank-specific.
All eight ranks require consistent readiness, reduction, prefetch and buffer
release ordering. Host markers alone do not prove device readiness or overlap:
MCP Perfetto must link actual dInput/all-three-wgrad kernels, event dependencies,
packing/RS launch, remaining backward, and storage release/reuse on every rank.
Diagnostic synchronizations and observer costs are excluded from magnitude.

## Executable commands

CPU only, from the task root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  runs/ffn-row-owner/tests/test_ffn_components.py --disable-warnings --tb=short
CUDA_VISIBLE_DEVICES='' /opt/torchtitan-venv/bin/python3.12 \
  runs/ffn-row-owner/ffn_components.py --component mtp --mode plan \
  --output runs/ffn-row-owner/artifacts/raw/ffn-plan
```

After the coordinator's current ownership preflight/source freeze, run each
command serially on the verified eight-H100 allocation, substituting `layer0`
for `mtp` for the other independent comparison and using fresh output folders:

```bash
/opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run \
  --standalone --nproc_per_node=8 runs/ffn-row-owner/ffn_components.py \
  --component mtp --mode check --accumulation 1 \
  --output runs/ffn-row-owner/artifacts/raw/mtp-vjp
/opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run \
  --standalone --nproc_per_node=8 runs/ffn-row-owner/ffn_components.py \
  --component mtp --mode check --accumulation 32 \
  --output runs/ffn-row-owner/artifacts/raw/mtp-accum32
/opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run \
  --standalone --nproc_per_node=8 runs/ffn-row-owner/ffn_components.py \
  --component mtp --mode benchmark --variant control --warmup 10 --samples 10 \
  --output runs/ffn-row-owner/artifacts/raw/mtp-control
/opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run \
  --standalone --nproc_per_node=8 runs/ffn-row-owner/ffn_components.py \
  --component mtp --mode benchmark --variant split --warmup 10 --samples 10 \
  --output runs/ffn-row-owner/artifacts/raw/mtp-split
/opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run \
  --standalone --nproc_per_node=8 runs/ffn-row-owner/ffn_components.py \
  --component mtp --mode diagnostic --variant split --trace-microbatches 3 \
  --output runs/ffn-row-owner/artifacts/raw/mtp-diagnostic
```

Setup must print selected `FFN overlap compute regions active` and
`GFSDP FFN overlap active` markers for B, native FP8 module identity, and resolved
SAC/Inductor/GFSDP settings. A must have neither split active. Native component
execution, eight-rank numerical interpretation, buffer-budget acceptance,
all-rank traced overlap, phase memory comparisons, and the combined native
training/EMA/continuation gates remain open. The coordinator owns the component
result table and decisions; this harness introduces no training-source change.
