# Row-owner embedding implementation

This is the historical initial implementation record. The current borrowed
storage and standard-autograd design is documented in [owner storage](owner-storage.md);
the [final component results](embedding-component-results.md) own its measured costs.

## Finite acceptance and preimplementation review

Scope: implement the complete owner lookup/autograd and excluded-parameter
state lifecycle in this task's `gift/`, with focused CPU/Gloo checks. Root
integration owns config, wrapping, compilation and SAC. GPU/NCCL/component
timing and native training are open gates; this worker may not run them.

Acceptance: (1) FP32 Ragged owner storage and baseline flat checkpoint shape;
(2) unique row requests, BF16 lookup, per-invocation inverse routing;
(3) native full-vocabulary BF16 local coalescing, BF16 transport, widening
each source before FP32 owner SUM and ordinary autograd accumulation;
(4) dense AdamW, mixed Ragged clipping, EMA lifecycle and continuation;
(5) coordinated invalid inputs and empty/skew traffic; (6) focused checks,
scoped lint and final source/runtime-cost review.

The parent supplied source authorization for this task copy only. Source base:
`2ada87125e433e1f76c4cb2bfad244b641a78a93`. Runtime inspected locally:
`/opt/torchtitan-venv/bin/python`, torch `2.15.0.dev20260822+cu130`, installed
under `/opt/torchtitan-venv/lib/python3.12/site-packages`.

Integration points inspected: `Module.init_states` recursively initializes
Module children after trainer `parallelize -> to_empty -> init_weights`;
GFSDP single-table packing uses alignment 8 and flat `RaggedShard` storage;
`RegisterPostBackwardFunction` calls group post-backward before propagating
input gradients; the root shares one all-gather/reduce-scatter stream.
The excluded owner will use that stream, explicit producer/completion events,
and independent allocations. Returned owner gradients must be current-stream
ready because the GFSDP root does not wait for excluded parameters.

Precision: fresh selected FP32 master rows cast BF16 each invocation; native
`aten.embedding_dense_backward` receives original signed int64 IDs and the
combined BF16 derivative. Its BF16 row result is selected before transport;
source contributions widen separately before owner addition. There is no
normalization, sparse update, persistent BF16 cache or side-effect `.grad`.

Reference: NeMo Automodel commit
`3ea1de8d0aa902c3329b585d194c2e5274a85f3f`,
`nemo_automodel/components/models/qwen3_8_flash_next/engram.py`, inspected via
the pinned upstream raw source. Reuse its owner/reverse transport pattern;
reject occurrence-gradient accumulation and first-batch fixed capacity, which
do not preserve this task's duplicate rounding and unbounded later counts.
Native CUDA reference: PyTorch commit
`8b353065ce260336c5fdfa74d754e8bd9b91b062`,
`aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu`; retain native dispatch
and original IDs rather than approximating its reduction order.

Expected per microbatch costs at V128256/D4096/eight owners: sorting/unique,
int64 count scatter-add, one fixed count/validity all-gather and one host matrix
read, variable request/value exchanges, inverse reconstruction, native full
BF16 coalescing (1002 MiB output plus its native FP32 scratch; low-parallelism
branch may allocate a full 2004 MiB FP32 temporary), BF16 gradient exchange,
and dense 250.5 MiB local FP32 gradient zero/materialization. Owner reduction
widens one source segment at a time to bound temporary memory. Maximum unique
row payload per requester is 64 MiB BF16 for 8192 IDs, with skew-dependent
owner receive traffic; no payload padding. Saved IDs/inverse/request metadata
live per invocation through backward. EMA adds 250.5 MiB persistent local
FP32 storage. No whole-table gather or dense gradient reduce-scatter remains
in this path. Eager stem boundary remains outside compiled/SAC blocks.

Required measurements remain open: actual native scratch/peak memory,
metadata synchronization, transport imbalance, complete latency and cross-
stream lifetime under H100/NCCL. CPU/Gloo can establish routing/state behavior,
not these GPU properties or a speedup.

## Implemented interfaces and final cost review

`RowOwnerEmbedding.from_embedding(embedding, dense_mesh, shard_alignment=8)`
returns a `Module` with the final registered owner `weight`. The caller must
exclude that exact parameter from **every** enclosing GFSDP wrapper, skip the
old embedding wrapper/prefetch entry and call `attach_comm_context(root)`.
Attachment verifies exclusion and emits `row_owner_embedding active` with
FQN/global shape/local rows/group rank and the ownership marker. The `_apply`
implementation preserves parameter identity through meta materialization and
the original `_param_init` is retained for `Module.init_states` traversal.

Optional `stage_observer(name)` supplies context managers for
`partition_ids`, `counts`, `request_exchange`, `owner_lookup`, `value_exchange`,
`reconstruct`, `source_coalescing`, `gradient_exchange`, `owner_reduction`.
It defaults to `None`; normal timing has no profiler stage annotations or
stage synchronization. Ordinary module forward and tensor backward cover the
complete path. There is no persistent row-value cache.

State integration is explicit in `BackendBufferStore` and
`GFSDPCheckpointAdapter`: owner enumeration, EMA allocation/seed/update and
paired backing swaps, flat main/EMA views, baseline FQN bindings and missing
EMA reseeding. The main checkpoint view continues to select LIVE weights
while EMA weights are temporarily swapped in for evaluation. Existing Ragged
clipping and fused AdamW dispatch are reused, including their dense untouched
row behavior and the existing optimizer checkpoint wrapper.

Final stream review found and fixed a material lifetime issue before handoff:
producer/completion events alone do not tell the caching allocator about
compute-stream consumers of communication-stream allocations. Both returned
output and gradient now `record_stream` on the caller's compute stream. Input
allocations record the communication stream, and independent per-invocation
scratch never aliases GFSDP buffers. Current GPU stages all execute on the
shared communication stream. This safely orders transport behind prior GFSDP
work, but also serializes the native dense source-coalescing kernel behind
that communication; component measurements must count it. If measured as a
bottleneck, a compute-stream coalescing phase with explicit dependencies is a
possible targeted repair, not an established improvement.

One explicit count-matrix `.cpu()` supplies all dynamic split metadata and
coordinated validity. `torch.unique` has dynamic output size and may introduce
additional allocator/device synchronization; the explicit host-read count is
not a claim that there is only one total synchronization. Per-invocation dense
local gradient allocation can coexist with the existing accumulated FP32
gradient. Cross-source FP32 addition follows source-rank order; a NCCL baseline
tree may round differently, despite identical per-source BF16 coalescing.

## Verification and failed attempts

No GPU/NCCL validation command has been executed by this worker. CPU workers
explicitly select Gloo and a CPU DeviceMesh. A later no-CUDA assertion exposed
that the installed DCP FileSystemWriter initializes a CUDA context when GPUs
are visible even for CPU payloads: `filesystem.py::_write_files_from_queue`
selects `_OverlappingCpuLoader`, whose constructor requests the available
device's current stream. Earlier checks are CPU/Gloo tensor/backend evidence,
but do **not** establish that no device context was initialized. Those worker
processes exited. Production DCP behavior was not altered. Final CPU execution
uses process-scoped `CUDA_VISIBLE_DEVICES=''`, and the pytest fixture also
hides GPUs before spawning fresh CPU workers. The explicit standalone NCCL
route does not use the pytest fixture.

Commands and outcomes:

- `PYTHONPATH=gift python` one-rank Gloo probe: stable parameter identity,
  ordinary Ragged autograd gradient and fused AdamW passed. The first ad hoc
  probe failed because the harness copied `[16,8]` into a flat `[128]` local
  tensor; corrected to a flat copy.
- `PYTHONPATH=gift python -m pytest -q
  gift/tests/unit_tests/test_row_owner_embedding.py --disable-warnings -x`:
  first run reached state checks after routing/accumulation and failed on the
  harness's unsupported `full_like(Ragged)`; changed helper to
  `zeros_like(...).fill_(...)`. Logs `embedding-cpu-checks.log` (failed) and
  `embedding-cpu-checks-2.log` (2 passed).
- The affected owner/DCP/FQN/EMA set had 19 passes before an existing EMA test
  failed on its cwd-relative config path. Log
  `embedding-cpu-checks-final.log`. The remaining selected EMA cases passed
  from `gift/`: 8 passed, 7 deselected in `embedding-ema-remaining.log`.
- Strengthened source-coalescing, meta initialization and resumed Adam checks
  passed before the added EMA lookup expectation failed. The harness wrongly
  assumed all EMA rows were equal, although baseline EMA was seeded before
  the constant LIVE weight assignment. Fixed the expectation to use the
  actual gathered EMA rows. Failure: `embedding-cpu-checks-complete.log`.
- The corrected state assertions all completed, then the added final
  no-CUDA assertion detected the DCP context initialization described above.
  `embedding-cpu-state-final.log` records initial backend Gloo, CPU mesh and
  `cuda_initialized=False`, plus the final assertion failure.
- Final targeted GPU-hidden rerun:
  `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m pytest -q -s
  tests/unit_tests/test_row_owner_embedding.py::test_cpu_gloo_routing_accumulation_and_state
  --disable-warnings -x`, from `gift/`. Result is in
  `embedding-cpu-hidden-final.log`: **1 passed in 7.90s**, both ranks explicitly
  report Gloo/CPU and `cuda_initialized=False`; final per-rank no-CUDA
  assertions also passed. All child processes exited.
- Scoped Ruff (F821/F811/F841 and the checkout's complete default rule set)
  passed on the four owned Python files. `git diff --check` passed.

The focused checks exercise exact BF16 lookup/order, duplicates/skew/empty/
boundary inputs and increasing later counts, overlapping invocation metadata,
shared two-consumer derivatives, native BF16 coalescing that rejects widened
occurrence accumulation, FP32 accumulation over 32 microbatches, zero gradients,
coordinated range/dtype/device failure and no-sync rejection. State checks use
real GFSDP-managed probe parameters alongside the excluded Ragged owner, real
global clipping, fused dense AdamW, baseline flat checkpoint+moments loading,
next updates and untouched rows, EMA interpolation/backing swaps/fresh lookup,
main-save while swapped, owner checkpoint roundtrip and missing-EMA reseeding.
The module initialization check exercises meta -> to_empty -> recursive
`Module.init_states` with an explicit initializer and stable parameter identity.

Remaining required gates: fresh independent integration/runtime-cost review,
actual eight-rank H100/NCCL focused check and production-shape component table,
compiled/SAC/native model integration and all campaign training/trace gates.

## Maintained documentation text for the thin integrator

Suggested addition to `gift/docs/fsdp.md`, beside the FFN controls:

> `gfsdp_row_owner_embeddings` defaults to `false`. In the supported GFSDP-only
> TP/PP/CP1, untied, complete-row layout, it keeps the input table's FP32 owner
> shards registered under `tok_embeddings.weight`, exchanges unique signed
> int64 row requests, and returns freshly cast BF16 rows in original token
> order. One shared lookup feeds trunk and MTP. Backward uses native full-
> vocabulary BF16 embedding coalescing before lossless BF16 transport and
> per-source FP32 owner SUM; dense FP32 accumulation, clipping, AdamW, EMA and
> flat DCP compatibility are retained. No full-table gather or dense embedding
> reduce-scatter is issued. The initial implementation retains the native
> 1002 MiB BF16 full-gradient temporary and native scratch at the crop dimensions;
> dynamic metadata/host synchronization and owner skew remain measured costs.
> HSDP, partial rows/table padding, tied weights, sparse/frequency-scaled or
> max-norm lookup, offload, deferred/no-sync reduction and higher-order
> differentiation are unsupported. The stem stays eager outside SAC/compiled
> blocks; existing model/loss compilation and SAC remain enabled. Focused
> checks are in `tests/unit_tests/test_row_owner_embedding.py`; its explicit
> `torchrun --backend nccl --checkpoint-dir PATH` route must only run on an
> authorized allocation.
