# Independent integration and runtime-cost review

14 November 2026; common source revision
`6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507`.
This review applies `code-rules`, `perf-experiments`, GIFT contributing/review
requirements and the applicable model, distributed, integer-width, experiment
and config rules. No reviewer source edit, GPU execution or launch occurred.

## Finite acceptance and current verdict

Review exactly the composition of existing V2 depth32, BF16 expert reuse,
[routing activation](../four-fusion/routing-metadata.patch) and
[CE activation](../four-fusion/ce.patch); inspect affected unchanged ownership,
precision, autograd and scheduling contracts; reconcile existing evidence and
new composition checks; name remaining native execution gates.

**Pre-run review passed: no unresolved material composition finding.** The two
patches change disjoint binding sites and introduce no new mechanism. Focused
composition tests, protocol parsing, scoped lint and exact baseline restoration
passed. The already authorized native execution gates remain open; this verdict
does not establish production acceptance or a speedup.

## Source integration and inherited contracts

- `hybridep.py` changes only the `indices_to_map` import. `_dispatch_impl`
  produces metadata before its existing communication-stream launch;
  `_launch_on_comm_stream` waits on the compute stream, records allocator
  ownership and joins results before consumption. The new metadata allocations
  retain those paths. `_dispatch_backward` still calls native
  `gather_route_gradients` after `combine_bwd`; the helper's standalone
  registered backward does not add another gather inside opaque dispatch.
  Top-k distinct signed IDs and capacity dropping remain producer invariants;
  `routing_metadata.py::_can_use_cuda_kernel` checks shape/dtype/offset limits
  without reading GPU values. Kernel offsets and sink addresses use int64.
- `loss.py::{CrossEntropyLoss,ChunkedCELoss}.__init__` replaces `self.fn`
  before `_maybe_compile`. The native callable remains defined for DTensor
  fallback, so the helper's local fallback import cannot recurse through the
  replacement. `MTPLoss` inherits that constructor. Its shifted targets,
  two sequence chunks, wrapped head calls, inner backward and shared main-token
  normalization remain active; `_chunk_loss` applies weights 1.0/0.3 after
  SUM CE. FP32 hidden-gradient accumulation and native head synchronization
  remain unchanged. The stable helper retains FP32 row statistics, SUM and
  incoming gradient arithmetic, with one final logits-dtype gradient store.
  It explicitly supports first-order gradients, the native training contract.
- `dataloader.py::_iter_prefetched` confines materialization to the existing
  CPU producer; epoch entry and committed yielded-state ownership stay on the
  consumer. Neither patch adds device work to that thread or changes its
  pause/close/load protocol. The logical epoch102400 covers 140 updates and
  the depth32 queue, avoiding the prior short-epoch boundary confound.
- `parallelize.py::apply_gfsdp` selects exact expert groups before lazy init.
  `_param_group.py` restricts reuse to FP32 masters, BF16 inputs, singleton
  shard groups and supported layouts. Shared-buffer ownership/release events,
  private ready/reader/write events and `wait_for_unshard` continue to order
  refresh and consumers. The two patches do not change master writers,
  Parameter identity, FP8 quantization or these events. Optimizer pre/post
  hooks, load/DCP, EMA and `_apply` retain invalidation/write publication.
  `_apply` also clears fallback validity, joins old shared readers before
  mutation and publishes the new writer event afterward.

Prior findings have explicit dispositions: the exact-group activation,
initialized fixture, true EP checkpoint layout, stable Parameter identity,
fallback freshness and nondefault-writer event-order findings are resolved in
[the cache review](../bf16-expert-cache/review.md) and
[focused checks](../bf16-expert-cache/focused-checks.md), including main008.
The patches cannot invalidate those resolutions because they change neither
writers nor cache ownership. Loader ownership/rollover/DCP evidence remains
[the V2 report](../v2-prefetch-baseline/report.md). Routing numerical/layout
checks and native activation remain
[the routing implementation](../four-fusion/routing-implementation.md).
Compiled full-vocabulary two-chunk main/MTP replay and precision interpretation
remain [CE numerics](../four-fusion/ce-numerics.md); neither the helper bytes nor
normalization path is being rewritten. These checks are reused, not rerun as
an individual-candidate campaign.

## Runtime-cost review

| Scope | Final composed path and residual measurement gate |
| --- | --- |
| CPU/host | Existing depth32 queue/thread overlaps materialization, retaining up to 32 queued CPU batches and their state. Cache adds Python generation checks, event records and update hooks. Disabled diagnostic counters still have their existing cheap guard. Routing removes eager metadata submissions; CE retains its opaque-op dispatch and SUM launch. Combined host pressure and submission gaps require native evidence. |
| Device computation | Routing does T×E×K comparisons:8192×128×8 per call, stores bool/FP32 dense maps plus sinks, and still allocates both outputs. Stable CE scans logits for row statistics and backward, retains head GEMMs, and does not fuse Linear+CE. Cache removes repeated eligible casts/local copies while retaining refresh and FP8 quantization. SAC repeats routing/weight consumers; prior three-microbatch routing traces counted 39 calls/rank. |
| Memory/lifetime | Cache aliases existing private BF16 inputs; shared output pools remain allocated. CE retains BF16 logits/dlogits, small FP32 row statistics and the native FP32 hidden-gradient accumulator. Depth32 increases CPU residency. Faster submission or shorter metadata/loss work can shift overlapping allocation peaks; no additive savings or lower peak is established by source. |
| Synchronization | No new patch-level host scalar extraction, explicit device synchronization or collective. Existing cache stream waits/shared-pool serialization, token-drop control and capacity checks remain. Earlier readiness can expose different waits or contention. Separate host waits, device gaps, copies and communication from compute when interpreting traces. |
| Communication | Native DeepEP FP8 dispatch/combine and routing-gradient gather remain. Only eligible singleton expert gather/copy bypass is reused; dense gathers and FP32 gradient reductions retain their paths. No collective-count reduction outside that singleton path is claimed. |
| Compile/state | Existing custom-op fake/autograd definitions stay unchanged; no new compile boundary. Check actual warmup settling, not merely cache entries. No new model/checkpoint key or optimizer state; loader cursor and EMA/load writers retain their owners. Native short runs cannot establish convergence or exact continuation. |

Integrated latency/TPS and allocator peaks must come from the matched unprofiled
runs. Trace kernel sums across concurrent streams are not elapsed time and must
not be added to host/communication spans. Previous isolated effects do not
predict an additive combined speedup.

## Focused evidence and remaining gates

[Integration checks](integration-checks.md) record **3 passed, 4 deselected, 0
skipped** in 8.33 s: actual routing-dispatch binding/kernel and native_float
two-chunk MTP eager/compiled arms. One `pynvml` deprecation warning remains;
there was no test failure or inactive-patch/CUDA skip. Four other MTP denominator
parameters were intentionally deselected. The routing fixture uses a recording
DeepEP buffer and disables its communication stream; these new checks close the
binding-composition question, while actual DeepEP/stream/FP8/SAC integration
remains a native gate. Scoped Ruff F821/F811/F841 and `git diff --check` passed.

[Source identities](source-identities.json) record both exact activation-patch
hashes, candidate core hashes and restored baseline hashes. The restored
`loss.py`/`hybridep.py` hashes equal the assessment pins; only those two core
files changed during transient activation. The
[combined source diff](candidate-core.patch) is frozen at SHA256
`9615f2d80160af46a2048531f2ad12c9728fd13044c8a2ae03663f25c7496950`.
The [protocol](protocol.json) was generated through the actual runner-to-helper
parsers without starting training: timing variants have 140/40/100, epoch102400,
seed2048 and depth0/reuseFalse versus depth32/reuseTrue; diagnostic has 21/20/0,
update 21, three trace microbatches, ranks 0/1, counters and microbatch receipts.
Both timing variants disable those diagnostics. No runner rewrite was needed.
Actual rank-0 resolved configuration and runtime activation are still checked
at launch, rather than inferred from this argument parse.

Native gates remain: baseline depth0/reuseFalse/original bindings and candidate
depth32/reuseTrue/both bindings; epoch102400/seed2048, steps140, warmup40,
window100 (updates 41–140; 209715200 global physical tokens). Preserve all agreed
crop settings, FP32 masters/accumulation, FP8, compile, SAC, EMA, MTP and
LR65000/warmup2000. Timing counters, profiler and microbatch capture stay off.
Freeze patches/helpers/runner before baseline, keep its source unchanged until
all ranks exit, then apply only the two exact patches before candidate.

The separate candidate diagnostic must complete 21 updates, all 32 microbatches
and optimizer on all 8 ranks, with update 21 traces limited to three complete
microbatches on ranks 0/1. Confirm queue use, eligible cache refresh/hits and
routing/CE observables; input fusion and LinearCE must remain absent. Record
resolved config and activation, integrated timing, paired lifecycle memory,
retry/OOM deltas, and trace attribution limits. One fresh pair is provisional;
it cannot establish repeatability, general speedup, convergence or PP4/EP32
production behavior. No further benchmark or optimization campaign follows.
