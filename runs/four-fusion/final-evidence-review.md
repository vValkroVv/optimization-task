# Final independent acceptance-evidence review

## Verdict and finite scope

No new material integration, numerical, or runtime-cost finding was found in
the reviewed evidence. The recorded synthetic CE replay supports its stated
local numerical gate. Activated FLCE has a measured local regression, and the
earlier inactive FLCE pair establishes neither its speed nor its memory cost.
Final throughput/retention decisions remain open until the coordinator integrates
the already running matched comparisons and repeats.

This bounded review applies `code-rules` and `perf-experiments` to four items:
reconcile prior required findings against the final pins and native execution;
review the new CE replay and its limits; assess equivalent trace boundaries and
cost interpretation; identify only missing evidence that changes acceptance.
It reuses the independent implementation reviews and completed Perfetto work.
No source edit, test rerun, GPU operation, benchmark, installation, or publication
was performed. The only write is this review.

## Final identity and integration

Source HEAD is `33b336a592746340d327d86e34149a23dc8688ee` plus the recorded
experiment changes. A read-only SHA-256 comparison found **all 15 files** in
[post-wrapper implementation pins](runtime/post-wrapper-implementation-pins.json)
unchanged, including all helpers, independent activation patches, the selector,
canonical module metadata and focused tests. The CE replay's helper/script pins
and retained raw-receipt hash also match. Commit identity alone is not the
executed-source identity.

| Prior finding | Final disposition and supporting evidence |
| --- | --- |
| R1: FLCE target metadata | Resolved: dtype, shape and device guards remain before reshape/dispatch; [correction evidence](loss-metadata-fix.md) covers eager/compiled rejection and valid native integration. |
| K1: input's native Python-float denominator | Resolved: the scalar arm preserves the inspected ATen reciprocal/multiplication order and auxiliary-state identities; [targeted independent verification](kernel-review.md#targeted-correction-verification) and [input checks](input-implementation.md) cover the producer boundary. Native scalar-kernel activation is observed. |
| K2: FLCE's native denominator and count specialization | Resolved: exact integral host-count bridging, local dynamic compilation, retained metadata guards and Tensor/fallback arms; [correction evidence](loss-native-denominator-fix.md) includes actual producer outputs, stable compilation counts, and GFSDP accumulation. |
| K3: input patch packaging | Resolved: the independent patch contains its Trainer integration only; actual-workspace application checks and native input runs passed. |
| Canonical parallel-wrapper fallback and reviewer follow-ups | Resolved within the required TP1 crop: exact wrapper/original-owner provenance, no active boundary/state mesh, no local-map/full-DTensor bypass, and complete fallback. [Final correction evidence](loss-parallel-wrapper-fix.md) covers old-fail/new-pass activation, eight CPU cases and three CUDA cases, including actual parallelize/GFSDP eager/compiled accumulation and DTensor fallback. Corrected native traces are active on both requested ranks. |

The final wrapper correction retains `head(...)`: GFSDP's prepended pre-forward
hook unshards before the adapter reads `module.weight`. Placement recognition
runs at installation; the metadata adds Python references and no per-chunk
tensor, collective, transfer or mesh resolution. Required FP8, compilation,
per-layer SAC, MTP1, EP8/dense FSDP8 and accumulation32 remain in the resolved
native setup. No universal claim about arbitrary wrappers, mutable post-install
sharding configuration, or arbitrary borrowed reshard policies follows.

## CE numerical gate

[The script](ce-numerical-replay.py) matches the inspected native CE compilation
boundary and `MTPLoss._chunk_loss` arithmetic: compile CE itself, divide its FP32
SUM by the native Python-float count, multiply by head weight, sum heads, then
unit-upstream backward. Main/MTP label shifting occurs before sequence chunking;
the replay explicitly checks the cross-chunk target and final ignored positions.
Both variants consume identical BF16 logits storage and int64 targets.

The [results](ce-numerics.json) compare all **2,101,346,304** BF16 gradient
elements across two `[4,2048,128256]` chunks. Tiled comparison is exhaustive over
those gradients, checks finiteness and exact ignored-gradient zeros, and uses
int64 element counts. Incoming FP32 scale bits match. There are 133,963 one-ULP
differences and none above one ULP; relative L2 differences are about `1.7e-7`.
One-FP32-ULP raw-loss differences become at most `1.8626e-9` weighted chunk-loss
differences; the recorded total weighted FP32 loss is equal.

The centered FP64 oracle uses represented BF16 inputs and the captured incoming
FP32 scales, subtracts the target before scaling, and covers 42 selected rows,
including boundaries, ignored rows and worst changed rows. Both implementations
remain within one BF16 ULP of that rounded oracle on those rows and have
comparable reference errors. Together with the separate large-common-offset
regression, these facts support the precision-aware verdict in
[ce-numerics.md](ce-numerics.md). Equality of the total alone would not do so.

This gate is explicitly one seeded synthetic local-block case. It establishes
neither model-logit distribution coverage, universal tolerance, convergence nor
continuation. The documents keep those limits visible, so no broader numerical
claim needs correction or an additional whole-model-gradient gate.

## Actual runtime-cost disposition

The [input](input-trace-analysis.md), [routing](routing-trace-analysis.md),
[CE](ce-trace-analysis.md) and [active FLCE](active-linear-ce-trace-analysis.md)
analyses establish activation and equivalent affected regions on ranks 0/1.
All eight ranks execute the required native accumulation; the requested capture
is bounded to three microbatches on two ranks. Per-kernel captures on ranks 2–7
are not an additional acceptance requirement.

Input and routing remove local launches and reduce the measured local regions;
their wider waits/communication remain. CE reduces equivalent CE elapsed in all
12 chunks, while two backward-only samples regress and wider downstream results
are mixed. None establishes a whole-update gain from kernel sums alone. The
initial input effect is negligible; initial routing/CE effects are comparable
to the recorded control variation, so the pending repeats can change retention.

Activated FLCE computes dH/dW inside its forward-labelled custom op. The common
loss guard includes both gradients, accumulation, hooks and communication rather
than comparing that forward to baseline forward alone. Device union and span,
kernel sums, host submission, copies, communication and idle are separately
reported. All six full-loss device spans regress, including the first chunk
without owned NCCL. Tiled FP32 dW costs `38.17–42.37 ms` per two-chunk microbatch
versus `21.32–22.85 ms` for baseline dW; the smaller CE cost cannot offset it.
This identifies measured cost without claiming that precision, tile shape or
library algorithm was independently isolated as the cause.

The FP32 dW accumulator/final BF16 cast and transient tile lifetimes remain real
costs. Buffer-size arithmetic is not a measured peak saving. The inactive pair's
memory reduction remains excluded. No new faster-kernel or memory-saving claim
is supported for FLCE.

## Remaining acceptance items

1. Integrate the fresh **active** FLCE matched unprofiled TPS/paired-memory result;
   retain the failed-activation attempt as invalid for candidate attribution.
2. Integrate the pending input/routing/CE reverse comparisons and controls, using
   complete measured windows and phase-matched memory, before deciding whether
   their small initial throughput differences are reproducible.
3. Consolidate those outcomes in the existing report/status/plan owners. Close
   acceptance at the measured crop scope; do not turn general long-run stability,
   convergence, continuation or additional rank captures into new requirements
   for rejecting FLCE. A representative FLCE numerical replay would be needed if
   its retention were later reconsidered; its current small tests do not claim
   full-shape numerical equivalence.

These are pending evidence/decision items, not a newly discovered source defect.
There is no source-fix request from this review.
