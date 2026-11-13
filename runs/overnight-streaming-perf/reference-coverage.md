# Reference applicability review

Scope: read-only source reconciliation for K1–K3, Z1 and V1–V5. No training,
benchmark, dependency installation or training-source change was performed.
Acceptance: identify independent missing mechanisms, inspect current consumers
and inherited lifetimes, retain fixed workload constraints, and give a finite
trace-dependent experiment order. Source inspection completes this deliverable;
GPU correctness and performance remain separate campaign gates.

## Source identity and method

Task HEAD at inspection: `a2a59e14545b55ca777f94a28e332759e7c06087`.
Current active files were inspected directly; line references below describe
that working-tree state, before concurrent campaign implementation.

| Snapshot | Pin | Verified files |
| --- | --- | --- |
| Kindulov | `b983803337dfcc34fdd571b38aaf5e01df583bbe` | 31 |
| Hidden z-loss | `81054a3383dc349a6690cc2b4aa0bce0dc73b5e9` | 36 |
| Vision runtime | `860152d111e4bac25daecbe3cc0cd84c68c8af9c` | 42 |

All 109 file SHA-256 values match their manifests. These are selected snapshots,
not independently installable trees. The exact promised vision list is still
unconfirmed. The missing CudaCherry source pin is
`585bd50a641857ccae8b5ec691bb15791c9529e7` (hidden-z-loss Dockerfile:13).
Neither gap authorizes objective changes or substitution of unverified code.

Paths in this report are relative to the repository root. `K/`, `Z/` and `V/`
mean `references/kindulov/source/`, `references/hidden-z-loss/source/` and
`references/vision-runtime/source/` respectively. `G/` means `gift/`.

## Candidate coverage

| ID | Source reconciliation and applicability | Expected costs, risks and decision gate |
| --- | --- | --- |
| K1a: cap-zero fused CE | **Independent algorithm remains applicable; proposed FP32-buffer removal is already present.** Actual G3 AOT forward saves BF16 logits/labels and two FP32 row statistics; backward writes BF16 gradients without a full-vocabulary FP32 buffer. Donor K saves one FP32 row LSE and directly computes softmax-minus-one-hot, removing the baseline backward's vocabulary-wide one-hot-gradient reduction. See [compiled contract and four pinned cache wrappers](k1a-compiled-contract.md). | Ordinary paths both use three kernels per CE chunk; the donor is not a launch-count reduction. Remaining arithmetic benefit and small row-stat lifetime difference are unmeasured. Adapt donor integration to preserve compiled CE rather than its eager bypass, without adding the unsupported head softcap keyword at cap zero. Unchanged donor also loses normalization for finite large equal logits: the pinned formula proof requires separate FP32 maximum/log-denominator statistics and an actual CUDA/compiled regression before acceptance. Preserve SUM/global-valid-token scaling, MTP weights, signed int64 labels, ignored/all-ignored behavior, head hooks, monitoring 2/32 chunks and FP32 hidden-gradient accumulation. Eager/compiled elementwise checks, native integration, A/B and comparable traces remain open. |
| K1b: completed chunk lifetime | **Applicable and missing independently of fusion.** Current G/loss.py:738–762 keeps Python `logits` and `chunk_loss` references until the next iteration rebinds them; the next head output is allocated while the prior logits variable is still live. K/loss.py:751 releases both after inner backward. Current code already clears `h_chunk.grad`; :703–706 still keeps all contiguous hidden/label chunks, and :708–712 retains the FP32 accumulated hidden gradient. | Explicit deletion can remove adjacent-logits overlap without changing computation. It does not eliminate all chunk copies or head buffers. Measure phase-aligned allocated/reserved peaks and actual end-to-end effect; allocator reuse and compiler ownership determine magnitude. Keep argmax sink/metrics and exception cleanup intact. |
| K2: dense/shared local FFN recomputation | **Applicable, missing as a local option; incremental benefit unresolved.** Current G/models/common/ffn/__init__.py:85–129 already uses FP8 projections, fused SwiGLU quantization and FP8-input storage, but has no local checkpoint. K/same-file:132–164 wraps the FP8-capable implementation in nonreentrant checkpoint and commits temporary clip counts outside that boundary. Shared experts use the same component through G/models/common/moe/__init__.py:69–70,183. | Fixed block SAC already replays these paths: G/distributed/activation_checkpoint.py:187–210 distinguishes dense `mm` from grouped `experts_up`; saving routed up projections does not mean dense/shared projections are saved. Nesting local checkpoints may add replay, retain an extra boundary input and change compiler partitioning. Dense layer0 uses recompute; most MoE blocks save routed up results only; the outer MTP segment falls through to global selective SAC with an empty save list. Do not call this inapplicable merely because SAC exists, or a saving merely because the donor supports FP8. Trace/saved-tensor evidence must show additional removable lifetime without unacceptable repeated GEMMs. Preserve FP8 `orig_shape` handling and clip/aux counters under both replay levels. |
| K3: routed local recomputation | **Inapplicable unchanged under required routed FP8.** Actual guard is K/models/common/moe/experts/gmm.py:111–115, not the package `__init__.py`: it raises when recompute and FP8 grouped GEMM are both enabled. BF16 checkpoint branch is at :212. | Converting routed experts to BF16 violates the workload. Existing SAC and FP8 grouped GEMM remain. A new FP8 algorithm is beyond a direct donor port; no performance claim follows from the BF16 implementation. |
| Z1: fused gate/up z-loss | **Inapplicable to the active objective, with missing dependency source.** G/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml:45–46 selects norm-input z-loss only. G/ffn/__init__.py:73–82 guards gate/up z-loss with fused FP8. Z/ffn/__init__.py:122–144,179–195 implements gate/up gradient fusion, not final-norm-input fusion. Z/ops/float8/_cudacherry.py:72 imports external fused kernels. | Adding gate/up coefficients changes computation and is explicitly disallowed. The no-gate/up path already has fused SwiGLU/quantization in the baseline; enabling that is not a new gain. Donor nonzero gate/up z-loss requires higher-precision saved inputs and scaling/replay validation. Missing external kernels stay a source limit; they do not block K1/K2/V2/V3. |
| V1: CompactCE | **Inapplicable unchanged.** V/components/loss.py:1153–1164 selects supervised positions before vocabulary projection and explicitly requires ordinary loss for MTP (also TP/CP/PP/full DTensor). Crop MTP1 is mandatory. | Do not disable MTP. A new MTP-capable compact objective path would require shifting/boundary/weight/denominator integration absent from this donor. Full-length streaming masks only certain targets; ignored fraction must be measured before predicting a material opportunity. |
| V2: bounded CPU streaming prefetch | **Applicable and missing.** Current G/datasets/streaming/core/dataloader.py:113–140 synchronously yields Mosaic batches and snapshots only its source cursor; current text loader :104–118 uses this base. V/core/prefetch.py is a reusable producer helper, but `rg PrefetchIterator V/` finds only its class and tests: selected donor files contain no text-loader consumer or checkpoint integration. | Adds a thread, bounded ready-batch memory, locking and possibly contention. Existing Mosaic workers/prefetch_factor are already supported (G/core/dataloader.py:106–109), so this is additional main-iterator overlap, not first-ever prefetch. The trainer fetches all 32 accumulation microbatches before compute (G/trainer.py:733–750); the producer may overlap next-update loading with current-update GPU work, but cannot interleave current-step compute without a separate trainer change. Trace exposed `fetching_batch` and worker gaps before ranking. No CUDA streams or H2D overlap are implemented by this CPU helper. |
| V3: GFSDP runtime collection | **Partly already present, with independent missing pieces below.** Current G/models/gift_moe/_base/parallelize.py:195–223 already separates norm/head and gives chunked head a private gradient buffer; :235–266 orders embedding, blocks, norm, MTP, then head and explicitly prefetches forwards/backwards for EP8. Do not count enabling GFSDP, explicit order, double buffering or head private gradients as new mechanisms. | Missing buffer planner, tail retention, callback guard and startup slicing have distinct evidence/cost scopes. Port small coherent pieces; donor Omni branch wiring is not a drop-in text-model replacement. Required GPU acceptance covers EP8/GFSDP2, compiled SAC, MTP, ordinary and monitoring inner backwards and repeated updates. |
| V4: frozen BF16 storage | **INAPPLICABLE to this fixed crop.** V/trainer.py:82–92 casts only non-trainable floating parameters before sharding/optimizer; documentation describes FP32 checkpoint rounding. Current crop configuration contains no freeze policy. | Native002 [runtime census](g1-smoke-002-receipt.json) confirms zero frozen tensors/elements on every rank. Non-trainable buffers and EMA copies are not eligible parameters. Do not freeze layers or cast trainable FP32 storage. Mixed frozen/trainable GFSDP partitioning is therefore not a crop speed experiment. |
| V5a: input-token metric accounting | **Missing accounting, not a speedup by itself.** Current G/trainer.py:748 counts physical label slots for console metrics, :749–760 reduces supervised targets for loss, and callbacks retain `ntokens_seen`. V/trainer.py:983,997–1019 packs input token counts into the same collective; V/callbacks/sources/trainer_source.py:322–360 publishes separate input-token TPS. | Keep physical input TPS and supervised-target TPS distinct on streaming masks/MTP. Extra counters/history/reporting add small work; a new collective is unnecessary. Do not import donor sample-normalization or CompactCE behavior merely to add this metric. |
| V5b: retained device count through producer and consumers | **Applicable after correcting the earlier premise.** G/trainer.py:749–758 reduces clamped local int64 counts through `dist_sum`, whose G/distributed/utils.py:70–72 returns a Python float via `.item()`. Donor V/trainer.py:997–1024 retains an int64 device tensor through direct functional reduction and adds it at:1131. | The current `int()` at:830 is CPU-only; the removable device read is at the producer before32 microbatches. A consumer-only replacement is invalid. A minimal independent port can retain the int64 reduction tensor without donor CompactCE/sample normalization/loss-unit vector. Preserve current local clamp-before-reduction, normalization precision, main denominator for MTP, optimizer/scheduler/callback/counter order and int64 persisted state. Verify compiled loss/gradient/aux-scale parity, empty/uneven local counts, callback observations and save/load; later logging/batch-end reads may bound benefit. |

## K2 local checkpoint and fixed SAC contract

Read-only integration inspection at `72a814f2` retains K2 as applicable, with no
demonstrated incremental benefit. The donor pin above adds a strict default-off
`FeedForwardConfig.recompute` and checkpoints `_forward_with_counts` with
`use_reentrant=False` (K/ffn/__init__.py:132–164). Temporary counts are committed
outside that local boundary. Current task fused FP8 output passes explicit
`orig_shape=(*base_shape[:-1], self.w1.out_features)`; the donor does not preserve
that active custom-op contract. Port the selector/count boundary without replacing
the task FFN or its functional graph-safe GEMM/schema fixes. Root inspected the
donor/current forward boundary and preset selection.

There are nine eligible FFNs: trunk layer0 and MTP each H10944, and seven shared
H1536. The preset MTP template aliases `&dense`; validate resolved selection and
keep trunk/shared/MTP choices independent. Shared experts use the existing
`FeedForwardConfig`, so no separate shared-expert API is needed. Local checkpointing
must exclude router, dispatch/combine, synchronization and auxiliary-loss sites.
The enclosing GFSDP unit owns live gathered parameters; longer replay can shift
its buffer lifetime and shared-expert/combine overlap without adding a collective.

MTP is wrapped as an outer segment and falls through to global selective SAC with
an empty save list. Dense/shared backward intermediates are already discarded and
regenerated by block SAC. Possible K2 savings concern regenerated lifetimes or
compiler save decisions, not removal of the entire original-forward retention
again. A local boundary also retains input and can add replay/quantization and
change fusion. At8192 tokens, BF16 boundary input is64 MiB, a dense gate/up output
is172 MiB each and a shared output24 MiB each, before aliases/scales/FP8 storage.
One additional complete nine-FFN replay would add27 GEMMs/microbatch or864/update
at accumulation32 (about210.28 TFLOP/rank/update projection arithmetic). These
are conditional sizes/work, not measured savings or overhead; checkpoint early
stop and compiled partitioning can change the actual count.

Keep FP8 scale/rounding/padding and saved wrapper metadata intact. `_is_recompute`
in the current GEMM is not a K2 switch and does not establish weight caching.
Always-save `aten.max` does not prove opaque quantizer internal amax is cached;
the active external quantizer's replay/storage contract still needs inspection
if this candidate is selected. Donor historical L8 results disabled block AC and
do not establish benefit under this fixed SAC. Its preserved recompute test is
primarily routed BF16 K3, not executable K2 FP8 acceptance evidence.

The donor scratch-count commit remains inside the outer block SAC boundary;
therefore it does not prove no duplicate mutation under both replay levels.
Test persistent clip slots, published counts and projection/module hooks before
and after backward, plus router/aux/bias state and two AdamW updates. Ordinary
compiled execution and monitoring differ: current MonitoringCallback opens an
`EagerWindow` on logged updates for module hooks. Required update50 coverage must
exercise that path and chunks32 as well as ordinary chunks2. Native A/A and actual
nested-SAC eager/compiled elementwise loss/input/projection-gradient/state checks
remain required; no bitwise native assumption or tolerance is established.

After G3, identify a concrete removable lifetime at the peak, replay GEMM/quantizer
counts and exposed shared-combine/GFSDP waits before choosing a port order or
selection. The campaign still retains only its agreed ordinary-update trace pair;
monitoring behavior uses the native cadence and targeted path checks rather than
silently adding a second capture campaign. No K2 implementation is active.

## V2 lifetime and state findings

### Selected minimal port and acceptance

**Specification only; no implementation or performance acceptance.** Reuse the
bounded `PrefetchIterator` from donor `860152d111e4bac25daecbe3cc0cd84c68c8af9c`
(V/core/prefetch.py), with a thin finite-epoch adapter around the existing Mosaic
loader. Keep `streaming_text`, its dataset/collator and scalar checkpoint format.
The finite deliverable is config wiring, producer/consumer cursor separation,
consumer-owned epoch entry and cleanup, behavior regressions, then the existing
native numerical/performance gates. No loader replacement, new state framework,
GPU copy/stream, trainer accumulation/normalization rewrite or preset change is
part of V2.

Installed Streaming0.13.0 source remains authoritative: dataloader SHA-256
`5328e3970ad2557df6bceeab64b18fc60691dbc39cb5ccb88631c985ab9cbfce` and dataset
`3bac8df314d3a9faac01b9f65e8a4de01420639feb67949194a6d6cbc3df02fb`, matching
[the data receipt](mds-data.json). Reinspected PyTorch2.15 development
`torch/utils/data/dataloader.py` is
`793f8e638fe953b55111b54c6efe007a44cc57abf031b3105f6a487ec666d58a`; its
`_utils/signal_handling.py` is
`20f1545bbf8f20445a89971fca8fb7f44fd9337f5aa6c0df336cf17f8246579e`.
These identify the installed `/opt/torchtitan-venv/lib/python3.12/site-packages`
files; current online documentation is background, not a replacement pin.
The official [PyTorch DataLoader documentation](https://docs.pytorch.org/docs/main/data.html)
and [Mosaic checkpoint interface](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataLoader.html)
confirm existing worker prefetch and loader state interfaces. Installed source
shows the concrete gap: neither supplies a bounded asynchronous main iterator
whose checkpoint cursor advances only when GIFT consumes a batch. The donor
supplies queue mechanics but no text-loader integration. A thin adaptation is
therefore required; replacing Mosaic would expand checkpoint/partition contracts.

### Queue depth and what the trace can establish

Use default-off `num_prefetch_batches: int = 0`; reject negative values. Select
**32** only in the isolated candidate override, superseding the earlier provisional
trial depth2. `Trainer.train_step` consumes32 microbatches before any model compute;
a queue of2 can hold only two already-materialized batches across that compute
interval. Depth32 can hold a full next fetch burst when the finite epoch has that
many remaining batches. This is a workload-based trial choice, not evidence of
an optimal depth or a speedup. Do not add a queue-depth sweep to this campaign.

The actual `TextPretrainCollator` returns int64 `input` and labels, plus int32
`positions`, each B2/S4096: `(8+8+4)*2*4096 = 163840` bytes, **160KiB/batch**.
Thus depth2 is320KiB and depth32 is **5MiB/rank** of ready tensors, not4.2MiB.
This excludes Python/state dictionaries, IPC/storage ownership, existing worker
queues and Mosaic preparation threads. The trainer already holds up to32 CPU
microbatches; the feature may retain another32 while compute runs. No tensor clone
is required by the queue. Actual storage lifetime and process RSS remain measured
gates, especially while trainer inputs migrate to GPU and workers recycle buffers.
Two workers with null `prefetch_factor` already resolve to two tasks/worker,
four outstanding tasks; V2 adds main-iterator materialization beyond that layer.

The supplied G3 MCP evidence for ordinary update111 reports rank0/1 underlying
DataLoader33-next sums591.856/631.059ms, before first microbatch at659.616/660.192ms;
the largest input prefix is607/644ms. CPU frame14.130s differs from unprofiled
ordinary roughly13.79s. All-GPU idle1.086/1.133s includes causes beyond input.
896 rows /8 ranks /B2 gives56 local batches/epoch; update111 starts at
`110*32 mod56 = 48`, consuming8 old-epoch then24 new-epoch batches. Thirty-three
calls are consistent with32 deliveries plus `StopIteration`; a355/346ms call
near wrap is observed, but worker/epoch causality is not directly established.

A safe producer cannot start the next epoch ahead of the consumer. First-batch
epoch entry remains synchronous, so the port cannot be credited with eliminating
all0.6s of input time or the wrap call. It may overlap the ready portion of the
next burst with the current update's GPU work; a depth32 queue cannot cross the
barrier to prepare those24 new-epoch batches. Keep the fixed native111–160 window
(which covers every epoch phase), accepted G2 A/A controls and candidate profile111.
No new trace campaign is needed. The unprofiled comparison must retain monitor150,
all FP8/compile/CuTe/SAC/MTP settings, B2/S4096/accum32, FSDP8/EP8, seed2048 and
scheduler65000/2000; the G2 baseline is124454.812 global input tokens/s with
A2/A1−0.074173%, not a promised V2 gain.

### Exact integration boundary and cursor ownership

Proposed owned training files and APIs (names below are port contracts, not
claims that these methods already exist):

- `torchtitan/config/schema/datasets/streaming/core/dataloader.py`:
  `BaseStreamingDataLoaderConfig.num_prefetch_batches`, default0 and lightweight
  nonnegative validation. Document one candidate override; keep canonical YAML0.
- `torchtitan/datasets/streaming/core/prefetch.py`: adapted donor
  `PrefetchIterator`, retaining `paused()`, `close()` and `join(timeout)` semantics.
  Preserve donor provenance/license. Do not edit the inert reference snapshot.
- `torchtitan/datasets/streaming/core/dataloader.py`:
  `_TitanStreamingDataLoader.begin_prefetch_epoch()` constructs/resets the raw
  PyTorch iterator with `torch.utils.data.DataLoader.__iter__(self)` and resets
  `num_samples_yielded=0` on the consumer. It **primes the first batch on the
  consumer**, increments the counter with existing `_get_batch_size`, and snapshots
  through existing `StreamingDataLoader.state_dict()`. Return this primed
  `(batch,state_after_batch)` and a finite remainder source. Each producer read
  mirrors that same increment/snapshot sequence. Do not use the lazy Mosaic
  `__iter__` wrapper as an allegedly already-initialized source.
- The same file's `BaseStreamingDataLoader.__iter__` owns the outer infinite
  loop and the enabled epoch producer. Start a thread only after successful
  priming; enqueue the first pair, then finite remainder pairs. On delivery,
  replace a separate `_committed_state` with that pair's scalar state **before
  yielding**. Do not mutate batches, labels or positions. Preserve the existing
  finite exhaustion and two-empty-epoch guard. New iteration closes the prior
  producer first. Only one active iteration is supported per loader, as with
  the underlying persistent-worker iterator.
- Its `state_dict()` returns a fresh copy under `dp_rank_0` of committed scalar
  metadata, never source-ahead state or queued tensors. At a checkpoint, use
  `paused()` to surface producer failure/quiesce an active read, then copy the
  committed cursor; pause does not serialize or discard the queue. Saving before
  first delivery retains the existing loader-state behavior; loading initializes
  committed state from the supplied scalar dictionary. Preserve missing/empty-key
  behavior. Saving and resuming must work with old checkpoints and changed
  world size/replication; retain Mosaic's sample-count/replication arithmetic.
- `torchtitan/datasets/streaming/text/dataloader.py`: pass the new config value
  through the existing `_init_streaming_loader` call. Text dataset, collation,
  transforms, sample partitioning and output dtypes remain unchanged.

`load_state_dict()` first closes/joins the enabled producer and invalidates the
old outer iterator so it cannot deliver stale batches. Only after successful join
may it discard queued pairs and apply existing `_loader.load_state_dict(state)`.
Installed `StreamingDataset.load_state_dict` only writes shared resume JSON;
`_resume` ignores an older saved epoch when `next_epoch` has advanced. Therefore
same-instance restore must also set the **public** `dataset.next_epoch` to the
saved `epoch`, and reset `num_samples_yielded`, before creating the new iterator.

This scalar is shared across local ranks/workers. A per-rank unsynchronized reset
can rewind it after a faster rank already starts workers. For the enabled
real-distributed load path use a consumer data-world quiescence barrier after
all producers join, then each owner loads the identical scalar state, the
node/data-world local leader resets `next_epoch`, and a completion barrier
precedes any new iterator. The exact existing route is
`Trainer.__init__` -> `DataWorld.from_parallel_dims(parallel_dims, role="train")`
-> `parallel_dims.get_optional_mesh("data")` -> `DataWorld.from_mesh(...)` ->
`process_group=data_mesh.get_group()`. `ParallelDims.get_optional_mesh` flattens
the existing `dataloading["batch", "cp", "tp"]` mesh using
`_flatten("data_mesh", backend_override="gloo")`; `DataWorld` already exposes the
validated `process_group`, `local_rank` and `world_size` fields. The text loader
stores this object as `BaseStreamingDataLoader.data_world` through the existing
`_init_streaming_loader` call. Therefore the two enabled-load barriers are exactly
`dist.barrier(group=self.data_world.process_group)` when data-world size>1;
the shared-scalar writer is exactly `self.data_world.local_rank == 0`.
No new group creation or group plumbing is required on the native caller path.
For enabled distributed construction with world_size>1, validate that this
explicit group exists rather than interpreting None as the default/global group;
manual diagnostic callers must pass their actual group via existing
`DataWorld(process_group=...)`. Size1/non-distributed cases need no barriers.

`Trainer.__init__` computes `DataWorld` before `build_replicated_dataloader`, so
existing mesh construction still involves its required participants. The latter
builds real loaders only on local rank0 of the PP replication mesh; its other
ranks receive wrappers. The `data` group excludes PP and spans one PP coordinate.
Consequently all real loader owners of the first PP coordinate participate in
that coordinate's restore barriers, while other PP coordinates do not enter
these loader barriers or build real loaders. `ReplicatedBatchDataLoader` already
forwards `load_state_dict` only on its owner and preserves receiver empty state.
Keep that routing. The selected PP1/TP1/CP1/EP8 workload has eight real owners in
one existing Gloo data group. `DataWorld.from_parallel_dims(..., role="validation")`
in `BaseValidator.validate` supplies the same API for a separately built
validation loader. No background collective is allowed. This is a
checkpoint-load-only coordination cost and an explicit eight-rank integration
gate, not a per-batch collective. Do not mutate Mosaic private shared-memory
objects or its barrier; do not use the checkpointer's group as a guessed substitute.

### Epoch, RNG, errors and close lifecycle

The donor condition guards active read, bounded ready queue and pause; capacity
is checked before `next`. It stores errors, drains earlier ready items, then raises.
`paused()` waits for an active read and rejects a saved error. `close()` requests
shutdown; it cannot interrupt blocked source I/O. `join(timeout)` raising is a
failed lifecycle gate: never reset a live source or silently abandon its thread.
Retain context propagation with `copy_context()` and close the finite generator
on its producer thread. The consumer releases the raw iterator only after join.

Do not wrap the infinite generator: one rank can enter Mosaic's node-shared
next-epoch worker barrier while another pauses with an old-epoch queue. Even a
finite source needs consumer priming, since dataset `__iter__` is lazy and enters
`_resume_incr_epoch` on first `next`. Prime before starting the producer, so the
producer never owns epoch entry. Drain the queue, observe exhaustion, join, and
only then let the consumer reset/prime the next epoch. An empty epoch starts no
producer. Keep data-world contexts active on each owning thread; PP broadcasts
and checkpoint collectives remain on consumer paths.

Create/reset/prime at the same first-`next` boundary as the baseline, not eagerly
at `iter(loader)` or constructor time. PyTorch `_BaseDataLoaderIter` consumes a
CPU Torch base-seed draw at construction; persistent resets reuse it, while
nonpersistent iterators draw anew. `_set_SIGCHLD_handler()` is a no-op outside the
main thread. Preserve both by consumer ownership. The inspected text path and
Mosaic sampling use local NumPy generators; FIM seeds `(seed,epoch,idx)`, not the
process-global Python/NumPy/Torch RNG. Do not install a different global RNG state
in the producer or perform per-batch global-RNG save/restore. Test observable draw
timing around deliveries, wraps and restores, including `num_workers=0`; arbitrary
new transforms that consume process-global RNG in the producer need a guarded
synchronous path or explicit further design, not silent acceptance.

Explicit cleanup is required because existing trainer/validation paths do not
close an owned asynchronous iterator on every exit. The exact interface is
`BaseDataLoader.close_iteration() -> None` and `BaseDataLoader.close() -> None`,
with no-op defaults in `torchtitan/components/dataloader.py`; these names are new
proposed methods, not existing APIs. Implement idempotent overrides on
`BaseStreamingDataLoader`: `close_iteration()` stops/joins/releases the current
source and retains reusable persistent workers; `close()` additionally performs
final worker shutdown. After load/close_iteration an old epoch iterator must
terminate rather than restart another producer; only a new `iter(loader)` may
start the replacement generation. This can use a per-iteration generation token
owned by the base streaming adapter, with no source/checkpoint-format change.
Keep PyTorch's existing shutdown behavior isolated in `_TitanStreamingDataLoader`;
its private worker-shutdown API is already used by Mosaic's destructor and must
be checked against the pinned installed source. Forward these methods through
`ReplicatedBatchDataLoader` in `torchtitan/components/dataloader.py` only on its
owner; clear its pending/iterator references. Add no-op base lifecycle methods
for loaders without owned asynchronous resources. In
`torchtitan/components/validate.py`, retain the validation iterator and call
`close_iteration()` in `finally` for early break and failure, preserving cached
loader reuse. `Trainer.close()` closes the loader through the wrapper, including
failure/normal completion; `batch_generator` propagates explicit iterator close
in `finally`. These are lifecycle edits only: preserve all32-fetch callbacks,
valid-token counting, device transfers, forward/backward and optimizer order.

### Finite verification gates and runtime-cost review

1. Add focused helper tests (adapt donor queue/order/bounds/pause/drain-error/
   exhaustion tests) and text-loader integration tests in existing streaming
   suites. Cover queue-ahead save/replay, exact end-of-epoch save, fresh and
   same-instance old-epoch restore, repeated/chained loads, changed data-world
   size and replication, unchanged state keys, zero/one/two workers, and two-worker
   persistent/nonpersistent modes. Assert tensor contents/dtypes, labels/masks,
   positions and elementwise continuation, not just sample counts or rounded norms.
2. Test lazy RNG draw timing against default-off at startup, each epoch reset,
   pause/save and restore; include Python, NumPy and Torch CPU draws. Verify the
   actual PyTorch iterator is initialized on the consumer and worker failure
   reaches it. Exercise validation early break/reiteration, producer exception,
   bounded join failure, explicit close, no surviving producer, old iterator
   invalidation and empty-epoch guard. Demonstrate each regression fails on the
   corresponding broken adapter, then passes. Run affected config decoding,
   torch-free/golden tests and `ruff F821,F811,F841` for changed Python files.
3. Before GPU performance, extend the existing real eight-rank/two-worker MDS
   check with off/on32 matched batch/label/position hashes through at least a wrap,
   queue-ahead checkpoint and same-instance older-epoch restore. Exercise skewed
   rank consumption/pause around the boundary with a deadline; retain the actual
   baseline rank partition, no missing/duplicate rows and identical continuation.
   Run both persistent and nonpersistent worker cases. Reuse the existing
   changed-world/replication/chained-resume tests with prefetch enabled, plus
   focused `ReplicatedBatchDataLoader` state/close forwarding regressions; these
   cover supported scalar/wrapper contracts without a new native PP/TP/CP matrix.
   The production target remains PP1/TP1/CP1/EP8.
4. Checkpoint verification is **loader-only**. Exercise actual installed
   `torch.distributed.checkpoint.save({"dataloader": loader}, checkpoint_id=...,
   process_group=data_world.process_group)` and matching `dcp.load` on all eight
   real Gloo owners, using the real MDS loader and a task-owned small checkpoint.
   DCP calls the loader's `state_dict()` during serialization/target creation and
   `load_state_dict()` after deserialization (installed
   `state_dict_saver._stateful_to_state_dict` and `state_dict_loader.load`). Save
   while ahead, advance across a wrap, restore the same loader to the older
   serialized cursor, and compare subsequent batch/label/position hashes to the
   off control/fresh restore under a deadline. Assert actual persisted
   `dataloader.dp_rank_0` scalar keys and values, not an in-memory dict roundtrip.
   This is the production Stateful serialization/load boundary: GIFT
   `CheckpointManager` stores the same loader under `DATALOADER="dataloader"`;
   `GFSDPCheckpointAdapter.load`/`load_gfsdp_dcp` retains nonmodel loader objects
   in the target and delegates to DCP. No new CheckpointManager/GFSDP code edit,
   full model/optimizer/EMA checkpoint, data download or 228GB artifact is needed.
   This closes the affected loader persistence/coordination claim if it passes;
   it does not establish full-model numerical checkpoint continuation.
5. Use the established native numerical harness for fixed-workload deterministic
   loss/elementwise-gradient/relevant-state comparison on ordinary and monitoring
   paths. CPU/Gloo loader-only checks do not establish GPU transfer or native
   training numerics; those are this native gate. No additional GPU/NCCL
   checkpoint gate is material to the selected loader-only change if its source
   review confirms state remains scalar/CPU, the data group remains Gloo, and
   no model, optimizer, EMA, GPU buffer or checkpoint staging path is changed.
   A change to those assumptions reopens only its affected gate. Full-model
   continuation is not claimed. Keep campaign checkpoints off throughout native
   numerical/performance runs. Then compare the existing alternating unprofiled
   A/B111–160 windows and profile111 for mechanism; retain failed/skipped gates
   and source/config/runtime identities. No additional native checkpoint run or
   broad parallelism campaign is requested by this specification.

Expected cost per enabled batch is one queue push/pop and condition signaling,
state counter arithmetic plus a fresh scalar snapshot (shared-memory read and
JSON decode), and producer Python/GIL work. There are no additional GPU kernels,
H2D transfers, device synchronization, training collectives, model arithmetic,
compilation boundaries or optimizer changes. There is one producer per rank,
thread start/join and synchronous first-batch entry per finite epoch; eight ranks
contend with sixteen existing workers and Mosaic preparation threads. The queue
retains CPU/IPC storage, so actual RSS/lifetime and CPU contention can outweigh
hidden fetch time. Checkpoint pause/load/close may wait for one active read;
load adds the explicit data-world coordination above. The default0 path must
keep original iteration/state logic and allocate no producer/queue/snapshots or
extra distributed coordination. Close forwarding adds only lifecycle calls.

Review the final implementation once against this contract and its measured
costs. Promotion requires all applicable behavior/native gates and a complete
performance comparison; queue32, generic donor tests and this specification are
not production acceptance. No unsupported combinations or claimed gains have
been established by this read-only work.

## V3 independent mechanisms and inherited constraints

1. **Startup shard slicing**: V/gfsdp/_param_group.py:275–336 replaces full
   unsharded parameter-data/index/element-index scratch tensors, GPU `eq/sum`
   loops and `.item()` reads with Python interval intersections and local slice
   copies. Current G/same-file:274–377 has the old implementation. This can reduce
   initialization memory/work and is applicable, but cannot improve warmed step
   TPS through that mechanism. Validate ragged offsets, empty shard slices,
   dtype and initialization equality; do not include startup in stable TPS.
2. **Data-buffer bank/lane planner**: V/gfsdp/_state.py:752–835 uses two banks of
   size-ranked lanes, grouping compatible process-group/all-gather layouts and
   giving retained/grouped/ancestor/private units dedicated contexts. Current
   G/_state.py:224–233 simply requests two contexts per process group. At EP8 a
   MoE block has both dense FSDP8 and expert FSDP1 partitions; their lifetimes and
   widths differ. Mapping can reduce arena waste or change which allocations
   remain live. Do not assume a fixed byte saving without enumerating resolved
   groups; dense embedding/head weights may already dominate both banks.
3. **Automatic tail retention and safe eviction**: V/_state.py:320–340 retains
   two final units with disjoint contexts; current G/_state.py keeps the final
   group of every context instead. V/_param_group.py:591–612 evicts only idle,
   auto-retained, unsharded owners without in-flight all-gather, then waits for
   the buffer release event before reuse. Current G/_param_group.py:655–664
   raises on any other owner. Retention can remove immediate tail regathers but
   increase live gathered memory; planner and owner/event handling form one
   lifetime-sensitive adaptation, not independent unsafe toggles.
4. **Partial-inner-backward callback guard**: V/_state.py:609–624 leaves groups
   untouched on a nonfinal backward unless PRE_BACKWARD or POST_BACKWARD.
   Current G/_state.py:559–565 calls `post_backward()` for every other group and
   resets it IDLE. The chunked head already sets `is_last_backward=False` across
   inner backwards (G/loss.py:719–724). Thus the guard is relevant with ordinary
   MTP chunking, not restricted to CompactCE; it can preserve prefetched decoder
   units until real decoder backward. Inspect actual prefetch/reshard traces and
   test head exceptions/flag restoration plus ordinary/monitoring sequences.
5. **Private data buffers / Omni branch chains**: donor API
   V/gfsdp/fully_shard.py:329–345 and Omni parallelize.py:177–200 protect branch
   lifetimes. Full Omni towers are absent from the crop. The text embedding may
   still be an independent autograd leaf, so do not port reversed-chain
   assumptions without testing actual MTP/head order. Current explicit text
   chain is already present and should be retained unless evidence identifies a
   specific flaw.
6. **Mixed storage/trainability partitions**: V/fully_shard.py:118–145,237–246
   and V/_state.py:838–860 split by storage dtype/grad. Useful for V4/frozen
   multimodal models; no fixed-crop benefit established. Do not add this broad
   semantic surface to obtain the unrelated buffer/callback mechanisms.

Inherited stream protocol must remain: G/_param_group.py:652–664 obtains the
all-gather stream and waits on `release_event`; :705–708 publishes consumer-stream
release before clearing ownership; :927–938 skips prefetch into another live
owner. Changing arena reuse must preserve all three, forward/backward multiplicity,
head private gradient lifetime and EMA registered parameter-group ownership.
Snapshot tests include data-pool planning, initialization and partial backwards;
their presence is not executed GPU evidence.

## Finite order and trace decisions

Start with G1 full-length streaming smoke, G2 unprofiled controls, and G3 ordinary
update trace pair; these source findings do not replace those gates. Then run at
most the following independently attributable changes before any combined winner:

1. K1b explicit chunk release; smallest isolated lifetime change. If no peak or
   TPS improvement survives controls, retain the negative result.
2. K1a cap-zero fused CE if CE/intermediate allocation has meaningful exposed cost;
   require local-CUDA path activation and parity before timing.
3. V3 partial-backward guard if traces show premature decoder releases/regathers;
   use a targeted state/lifetime regression before full updates.
4. V2 CPU prefetch if the loader/update boundary exposes input wait; preserve the
   loader state/exception contract and compare host queue memory.
5. V3 arena/retention adaptation if allocation map or exposed all-gathers show an
   opportunity; enumerate actual EP8 groups first and port coherent lifetimes.
6. V5b retained device count across the producer/reduction/normalization/counter
   path if the early producer `.item()` is exposed. The consumer's Python `int()`
   is not a GPU read. V5a accounting alone is not credited as a speedup.
7. K2 dense/shared local recomputation only if activation-lifetime evidence shows
   an incremental saving beyond fixed SAC; profile nested replay costs first.

V3 startup slicing is separately eligible when initialization peak/time is a
campaign blocker; it has no warmed-TPS priority. K3, Z1, V1 and V4 remain
inapplicable under current constraints, without disabling FP8/MTP, changing the
objective or freezing trainable parameters. Reorder applicable trials only when
the baseline traces resolve one of the stated cost questions. Every executed
candidate needs elementwise numerical/state checks, repeated unprofiled A/B/A/B,
phase-matched memory, and a representative two-rank trace pair. No speed, memory
percentage, production readiness or complete vision-list coverage is claimed here.
