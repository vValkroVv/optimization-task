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
| K1a: cap-zero fused CE | **Applicable and missing.** G/torchtitan/components/loss.py:34–44 converts each logits chunk to FP32 for CE. K/torchtitan/ops/softcap_cross_entropy.py:12–55 saves logits, labels and FP32 row LSE; :58–93 permits cap zero and int64 targets. K/torchtitan/ops/_softcap_cross_entropy.py:9–15 explicitly skips softcap at cap zero, :22–43 computes FP32 online LSE with ignored labels, and :50–65 writes logits-dtype gradient. MTP inherits the donor chunk loss. | Can avoid full-vocabulary FP32 CE intermediates while retaining low-precision logits; still computes logits and scans vocabulary. Triton uses one row program and 4096-column tiles, so V128256 is a concrete performance question. GPU summation/rounding may differ. Preserve SUM/global-valid-token scaling, MTP weights, int64 labels, first-order gradients, head hooks, monitoring 2/32 chunks and FP32 hidden-gradient accumulation. Require eager/compiled elementwise loss/gradient comparisons and full fixed-workload runtime evidence. |
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

The donor producer owns source iteration and closes it on that same thread
(V/core/prefetch.py:51–107). `copy_context()` carries context variables. The
condition guards `_active`, ready queue and pause; it sets active before `next`
and clears it in `finally`. Capacity checks happen before reading the next item.
Errors are stored, queued items drain first, then consumer raises (:109–120).
`paused()` waits for any active read and rejects a saved error (:123–134).
`close()` requests shutdown but does not interrupt a blocking source read;
`join(timeout)` raises if it still runs (:136–147). These are concrete contracts,
not proof of correct text-loader integration.

Deeper installed-source inspection changes the proposed checkpoint integration:
use a **committed consumer cursor**, not ready-queue serialization. GIFT stores
rank-identical `dp_rank_0` state and supports changed world size/replication;
rank-specific queued batches do not fit that elastic contract. Installed
`StreamingDataLoader.__iter__` advances its cursor before yielding, and its local
`state_dict()` performs no collective. Queue `(batch,state-after-batch)` and
commit metadata only on delivery, saving the same scalar-state format; recompute
undelivered batches on restore. This adds a shared-memory read/JSON decode per
produced batch and must be included in the cost comparison.

Do not background-wrap the existing infinite generator: one producer can enter
Mosaic's node-shared next-epoch barrier while another queue pauses before it,
deadlocking checkpoint pause/join. Use a finite single-epoch producer and restart
on the consumer only after draining. Construct/reset/prime DataLoader iterators
on the consumer thread to preserve global CPU RNG draw timing and PyTorch's
main-thread SIGCHLD setup. Keep generator/data-world context and checkpoint/PP
collectives on their existing consumer paths. Same-instance load must also reset
Mosaic epoch bookkeeping: forwarding an older saved epoch can otherwise be
ignored after `next_epoch` advanced. Closing cannot cancel blocked `next()`;
join timeout is a failure, not permission to reset a live source.

The minimal proposed selector is default-off `num_prefetch_batches: 0`, with two
as an initial isolated trial. At B2/S4096 its additional ready-batch tensors are
about320 KiB/rank, excluding producer/Python/IPC overhead and existing worker
buffers. Installed PyTorch already resolves null prefetch factor to two per
worker (four outstanding with two workers); Mosaic also has preparation threads.
These established paths do not supply a checkpoint-aware asynchronous main
iterator. See official [PyTorch DataLoader](https://docs.pytorch.org/docs/main/data.html),
[Mosaic tuning](https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/performance_tuning.html)
and [checkpoint interface](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataLoader.html).

The donor tests cover generic queue/pause/error/exhaustion, not actual elastic
Mosaic state. Required integration tests include queue-ahead exact continuation,
unequal-rank epoch-boundary pause, existing changed-world-size/replication/chained
resume, CPU RNG parity, persistent/nonpersistent workers, same-instance load,
validation early break/reiteration and thread cleanup. The real eight-rank
fixture must retain exact row/label/position ordering through wrap and resume.
Campaign saves are off, so successful training alone cannot close continuation.

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
