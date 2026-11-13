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
| K2: dense/shared local FFN recomputation | **Applicable, missing as a local option; incremental benefit unresolved.** Current G/models/common/ffn/__init__.py:85–129 already uses FP8 projections, fused SwiGLU quantization and FP8-input storage, but has no local checkpoint. K/same-file:132–164 wraps the FP8-capable implementation in nonreentrant checkpoint and commits temporary clip counts outside that boundary. Shared experts use the same component through G/models/common/moe/__init__.py:69–70,183. | Fixed block SAC already replays these paths: G/distributed/activation_checkpoint.py:187–210 distinguishes dense `mm` from grouped `experts_up`; saving routed up projections does not mean dense/shared projections are saved. Nesting local checkpoints may add replay, retain an extra boundary input and change compiler partitioning. Dense layer0 uses recompute; most MoE blocks save routed up results only; MTP policy also needs resolved-config inspection. Do not call this inapplicable merely because SAC exists, or a saving merely because the donor supports FP8. Trace/saved-tensor evidence must show additional removable lifetime without unacceptable repeated GEMMs. Preserve FP8 `orig_shape` handling and clip/aux counters under both replay levels. |
| K3: routed local recomputation | **Inapplicable unchanged under required routed FP8.** Actual guard is K/models/common/moe/experts/gmm.py:111–115, not the package `__init__.py`: it raises when recompute and FP8 grouped GEMM are both enabled. BF16 checkpoint branch is at :212. | Converting routed experts to BF16 violates the workload. Existing SAC and FP8 grouped GEMM remain. A new FP8 algorithm is beyond a direct donor port; no performance claim follows from the BF16 implementation. |
| Z1: fused gate/up z-loss | **Inapplicable to the active objective, with missing dependency source.** G/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml:45–46 selects norm-input z-loss only. G/ffn/__init__.py:73–82 guards gate/up z-loss with fused FP8. Z/ffn/__init__.py:122–144,179–195 implements gate/up gradient fusion, not final-norm-input fusion. Z/ops/float8/_cudacherry.py:72 imports external fused kernels. | Adding gate/up coefficients changes computation and is explicitly disallowed. The no-gate/up path already has fused SwiGLU/quantization in the baseline; enabling that is not a new gain. Donor nonzero gate/up z-loss requires higher-precision saved inputs and scaling/replay validation. Missing external kernels stay a source limit; they do not block K1/K2/V2/V3. |
| V1: CompactCE | **Inapplicable unchanged.** V/components/loss.py:1153–1164 selects supervised positions before vocabulary projection and explicitly requires ordinary loss for MTP (also TP/CP/PP/full DTensor). Crop MTP1 is mandatory. | Do not disable MTP. A new MTP-capable compact objective path would require shifting/boundary/weight/denominator integration absent from this donor. Full-length streaming masks only certain targets; ignored fraction must be measured before predicting a material opportunity. |
| V2: bounded CPU streaming prefetch | **Applicable and missing.** Current G/datasets/streaming/core/dataloader.py:113–140 synchronously yields Mosaic batches and snapshots only its source cursor; current text loader :104–118 uses this base. V/core/prefetch.py is a reusable producer helper, but `rg PrefetchIterator V/` finds only its class and tests: selected donor files contain no text-loader consumer or checkpoint integration. | Adds a thread, bounded ready-batch memory, locking and possibly contention. Existing Mosaic workers/prefetch_factor are already supported (G/core/dataloader.py:106–109), so this is additional main-iterator overlap, not first-ever prefetch. The trainer fetches all 32 accumulation microbatches before compute (G/trainer.py:733–750); the producer may overlap next-update loading with current-update GPU work, but cannot interleave current-step compute without a separate trainer change. Trace exposed `fetching_batch` and worker gaps before ranking. No CUDA streams or H2D overlap are implemented by this CPU helper. |
| V3: GFSDP runtime collection | **Partly already present, with independent missing pieces below.** Current G/models/gift_moe/_base/parallelize.py:195–223 already separates norm/head and gives chunked head a private gradient buffer; :235–266 orders embedding, blocks, norm, MTP, then head and explicitly prefetches forwards/backwards for EP8. Do not count enabling GFSDP, explicit order, double buffering or head private gradients as new mechanisms. | Missing buffer planner, tail retention, callback guard and startup slicing have distinct evidence/cost scopes. Port small coherent pieces; donor Omni branch wiring is not a drop-in text-model replacement. Required GPU acceptance covers EP8/GFSDP2, compiled SAC, MTP, ordinary and monitoring inner backwards and repeated updates. |
| V4: frozen BF16 storage | **INAPPLICABLE to this fixed crop.** V/trainer.py:82–92 casts only non-trainable floating parameters before sharding/optimizer; documentation describes FP32 checkpoint rounding. Current crop configuration contains no freeze policy. | Native002 [runtime census](g1-smoke-002-receipt.json) confirms zero frozen tensors/elements on every rank. Non-trainable buffers and EMA copies are not eligible parameters. Do not freeze layers or cast trainable FP32 storage. Mixed frozen/trainable GFSDP partitioning is therefore not a crop speed experiment. |
| V5a: input-token metric accounting | **Missing accounting, not a speedup by itself.** Current G/trainer.py:748 counts physical label slots for console metrics, :749–760 reduces supervised targets for loss, and callbacks retain `ntokens_seen`. V/trainer.py:983,997–1019 packs input token counts into the same collective; V/callbacks/sources/trainer_source.py:322–360 publishes separate input-token TPS. | Keep physical input TPS and supervised-target TPS distinct on streaming masks/MTP. Extra counters/history/reporting add small work; a new collective is unnecessary. Do not import donor sample-normalization or CompactCE behavior merely to add this metric. |
| V5b: device-side tokens-seen update | **Applicable and missing independently of the metric.** G/trainer.py:830 calls `int(global_valid_tokens)` before updating a device int64 counter; donor V/trainer.py:1131 uses tensor `add_` directly. | Current scalar conversion can introduce a GPU-to-host wait every optimizer update. Removing it preserves the count if tensor dtype/device/scalar contract is maintained; inspect callback consumers before editing. Trace whether that wait is exposed before claiming gain. Preserve intentional log-time reads and persisted `ntokens_seen` state; do not copy the donor's broader normalization changes. |

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

To port safely, serialize ready items, finished flag and Mosaic source state
inside the same pause; source cursor alone skips prefetched batches. Consumer
must own checkpoint collectives. Preserve generator/data-world context,
reiteration and infinite-epoch behavior; close/join the old producer before
load/reset. The selected base donor additionally has worker startup accounting
(`dataloader.py:57–76`) but no ready-state consumer: do not assume it solves text
resume. Tests V/tests/unit_tests/test_streaming_prefetch.py cover queue ordering,
pause/cursor, delayed error, close and exhausted restore for generic Python
sources; they were inspected, not executed, and do not cover Mosaic/distributed
startup. Campaign checkpoint saves are off, so successful training would not
prove continuation. A task-scoped integration must retain that open gate rather
than silently weakening the existing loader state API.

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
6. V5b scalar-read removal if the tokens-seen conversion is an exposed wait;
   V5a may accompany accurate reporting but is not credited as a speedup.
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
