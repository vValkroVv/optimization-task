# Numerical acceptance amendment: variability and precision

Design only, 2026-11-11. No source changes, GPU access, runs or installations.
The [current owner contract](../../../../report.md#current-acceptance-contract--owner-update)
supersedes the exact floating default and strict A/A gate in
[design 001](../001/handoff.md). Its native hooks, local-shard inventory and native
checkpoint route remain applicable, subject to the trajectory correction below.
This document proposes a finite calibration procedure. The coordinator may
establish, review and freeze criteria within existing user authorization; this
introduces no additional human approval gate. Material baseline or precision
changes still require the applicable authorization.

## Acceptance checklist and current evidence

1. Define tensor/scalar metrics, precision constraints and exact invariants.
2. Define the production A/A measurements needed before freezing A/B criteria.
3. Separate one-update arithmetic from ten-update trajectory variability.
4. Preserve continuation coverage with bounded retention and honest verdicts.
5. Check this document's local links and whitespace.

Keep the exact 3B preset, FlexAttention, HybridEP, dense/sparse FP8, FSDP2/EP8,
SAC, compiled model/loss, batch 8/global 256/sequence 4096, four microbatches,
and 2,000/30,000 schedule. Do not substitute debug deterministic compiler options
for production policy. Capture repair has priority: one public step marker at
native `batch_start` addresses the actual mask-lifetime error. No runnable graph
candidate exists yet; do not build a universal observer ahead of that result.

The original longer graph-off control failed at update 22. Corrected scan-bound
smoke completed 25 updates, but corrected 71-update control 001 failed during
update 33 after 32 optimizer completions; 002 was not run. No valid 50-update
noise pair exists. Original/corrected strict ten-update comparisons differ at
updates 2–10, but strict mode overrides FlexAttention compile options. These are
retained diagnostics, not full production calibration. Scalar logs, even with
10/20 samples, cannot calibrate elementwise gradients or optimizer state. CUDA
faults remain a separate defect, not numerical variation.

**No numeric passing tolerances can be justified from the current evidence.**
Finite A/A data can establish observed variability under its recorded conditions;
it cannot alone establish a universal safe perturbation or 30,000-update quality.
FP8/BF16/FP32 format names alone also cannot establish such a bound.

## Metrics and ownership

Use design 001's canonical parameter/alias mapping and actual rank-local DTensor
shards. Never gather/dequantize/recast the recorded source tensors to save disk.
CPU comparison may promote copies to FP64 for arithmetic. For each update and
family, retain every layer/expert/FQN/shard result; a global average cannot hide
one bad expert or rank. Deduplicate aliases while retaining their identity map.

For reference `x`, comparison `y`, difference `d = y - x`, report:

- Elementwise absolute errors, their maximum and worst coordinates; nonfinite
  counts and count/fraction outside each frozen elementwise band.
- Relative error `abs(d_i)/abs(x_i)` only where `abs(x_i) > z_s`; elsewhere
  report absolute errors separately. `z_s` is a frozen, dimensional threshold
  for stratum `s`, justified by an operation's absolute error/quantization scale
  or production A/A pilot magnitudes. Never divide by an arbitrary epsilon.
- Relative L2 `norm(d)/norm(x)` when `norm(x) > Z_s`; otherwise report absolute
  L2/RMS with count. Freeze `Z_s` with the same rationale. Also report symmetric
  RMS-normalized difference when both sides have nonzero scale, as a diagnostic.
- Native dtype ULP distance where meaningful, signed-zero differences separately,
  and zero-to-nonzero transitions; ULP distance is descriptive, not a universal
  FP8 or training tolerance. Empty tensors have zero counts and no maximum.

Strata are tensor family × layer/module type × dense/expert role × dtype;
shards remain individually visible. Pool calibration only where producer,
precision, scale and reduction path justify it. Small/empty experts never borrow
large dense-layer averages. Freeze the grouping before inspecting candidate
errors; retain worst individual tensors even when reporting grouped quantiles.

Families: accumulated preclip gradient, actual postclip gradient, first moment,
second moment, post-update parameter, and actual parameter delta. Delta is
`FP64(P_after) - FP64(P_before)` from native endpoints, not a reconstructed Adam
formula. Compare deltas directly: large parameter values must not conceal a wrong
small update. Record native unrounded main/auxiliary losses and their actual
normalizers, returned global norm, clipping coefficient, LR used/next LR, and
floating persistent/router/FP8/candidate state where present. Equal norms do not
substitute for equal elementwise gradients. Keep per-microbatch loss/count records
and four-backward/one-optimizer phase counts.

## Exact invariants versus floating state

Fail on missing/extra tensors, wrong `grad is None`, aliases, shape/dtype/stride,
placements/mesh/rank coverage, parameter ownership, optimizer groups/step,
schedule/horizon/LR construction, input/label elements, masks/packed boundaries,
loader position, valid-token and auxiliary normalization semantics, or lifecycle
phase/order differences. Do not turn `None` into zero or narrow integer counts.

RNG ownership, saved bytes and restoration are exact at matched start/load
boundaries; required random draws and SAC recomputation retain baseline semantics.
Compare integer/discrete bookkeeping exactly when it is prescribed by the same
inputs/state. Routing assignments and resulting expert counts may diverge later
through floating near-ties: do not require all such values to match across
independent trajectories. They must obey exact capacity/no-drop/count/reset
semantics and remain measured; unexpected divergence in the first common-state
update requires explanation, not a larger gradient tolerance. Persistent floating
router bias, moments, parameters and FP8 state use their frozen floating criteria.
Nonfinite values or an unexplained state mutation fail; do not calibrate them away.

Fresh-process checkpoint load must reproduce the saved state itself exactly,
including floating bytes where native serialization promises that round trip.
The subsequent stochastic/nondeterministic trajectory uses floating criteria;
exact saved-state restoration does not imply bitwise future loss equality.

## What precision establishes

Inspected source: [dense FP8 GEMM](../../../../../../gift/torchtitan/ops/float8/cuda_gemm.py)
quantizes current operands, requests split accumulation and a chosen output dtype;
[grouped GEMM](../../../../../../gift/torchtitan/ops/float8/grouped_gemm.py)
uses E4M3 payloads and BF16 output buffers. The
[block quantizer](../../../../../../gift/torchtitan/ops/float8/triton_kernels/block_quantization.py)
converts arithmetic to FP32 and stores FP32 scales. These wrapper facts do not prove
the installed custom kernel's complete accumulation/reduction error bound.
[Training config](../../../../../../gift/torchtitan/config/configs.py) distinguishes
FP32 master training state from BF16 mixed-precision parameters and FP32 reduction;
[optimizer construction](../../../../../../gift/torchtitan/components/optimizer.py)
has a separate BF16-state mode. Inventory resolved actual dtypes, active mode and
installed kernel pins rather than assuming five FP32 families from the estimate.
[Native train_step](../../../../../../gift/torchtitan/trainer.py) accumulates four
backwards, then clips, steps optimizer/scheduler and sums detached losses.

Preserve these operations, accumulation precision, quantizer scales/rounding,
reduction policy and optimizer epsilon. A local rounding bound must name the
actual operation, input magnitude, scale, reduction length/order, output cast and
nonlinear conditioning. Do not propagate one FP8 representable spacing as a
relative tolerance for all gradients/moments. Quantizer thresholds, cancellation,
router top-k changes, clipping and Adam normalization can amplify a small input
change. Their end-to-end bounds remain unresolved. Existing transient-scale
observation gaps from design 001 remain explicit; do not invent persistent amax
history or add a universal tracing wrapper.

## Finite calibration and freeze procedure

This is conditional on a runnable candidate and an operational production path;
no run is authorized by this design. Preserve identical graph-off prerequisite
fixes and all non-graph compiler settings. Numerical runs are separate from speed
measurements. Reuse still-valid data; performance A/A and numerical A/A answer
different questions.

1. Pin source/runtime, resolved rank-zero production config, all-rank topology,
   feature activation, saved full state S0, data/loader/RNG and observer identity.
   Validate one native observed update and complete shard coverage using design
   001's small implementation checks. A changed prerequisite invalidates only
   affected calibration, not unrelated completed evidence.
2. Start with **one matched production A/A pair** from S0 over ten real updates,
   with all four changing microbatches. Propose one additional independent pair
   only when its possible outcomes can change calibration or an outlier decision;
   record that question and decision first. Sample count and uncertainty remain
   explicit; this is not a tail-probability or equivalence proof.
   Observe all required tensor families over the required uninterrupted ten-update
   trajectories; storage feasibility remains open as below. Record per-update losses, gradients, deltas, moments/state,
   phase records and stratum metrics; scalar-only repeats do not satisfy this.
3. For the first common-state update, distinguish arithmetic perturbation from
   trajectory amplification. Compare exact start tensors/state and record each
   phase's input/output precision. If a local precision check is needed, replay
   that operation from identical saved inputs; do not restore before every
   update and call it a ten-update trajectory. Later updates retain each run's
   evolving state; calibrate their errors by update index, never pool them as
   independent samples. Cross-run pairs, not elements or steps, are replicates.
4. Create a criteria manifest **before viewing graph-on numerical differences**:
   metric formulas/strata/near-zero thresholds, absolute-plus-relative bands
   `abs(d_i) <= a_s + r_s * abs(x_i)`, delta/moment/scalar limits, aggregation,
   number of allowed exceedances (normally zero unless justified), source of
   every number, finite sample count and uncertainty, and explicit verdict rules.
   Attach observed A/A ranges/quantiles and local operation evidence. Set a band
   from a defensible precision/perturbation budget, then ask whether A/A fits it;
   never set `a_s/r_s = multiplier * observed AA_max` merely to obtain a pass.
5. The coordinator reviews and freezes the manifest/hash, margins and limited
   claim within existing authorization; no additional human approval is required. If operation analysis plus A/A cannot justify a number, leave
   that metric INCONCLUSIVE and name the missing bound or practical-equivalence
   decision. At most one targeted additional A/A pair is proposed to resolve an
   unstable calibration result; if still unresolved, report the gate open rather
   than expand into a baseline-determinism or statistical research project.
6. Compare graph-off/on using the frozen manifest; any threshold change creates
   a new version requiring independent confirmation, not retroactive acceptance.
   A relevant source/path change requires affected recalibration. Keep the
   original baseline separate when prerequisites changed it.

A/A supplies variability evidence, not a physical error budget. If the owner
chooses a finite-sample predictive/outlier rule instead of an analytical bound,
the coordinator must record its risk level and independent-pair count before A/B.
For example, with exchangeable independent pair scores, rank resolution is
`1/(n+1)`; two pairs cannot support a 5% tail claim. Predeclare a single joint
worst-stratum score or account for multiple comparisons. A maximum from a few
pairs cannot silently become a universal bound. This optional statistical route
is not a requirement to collect a large new campaign, and a non-outlier result
alone is not evidence of practical equivalence or training quality.

## Decisions and outliers

- **FAIL:** exact contract violation, nonfinite, stale input/state, illegal memory
  behavior, or a confirmed frozen precision/equivalence-band violation.
- **INCONCLUSIVE:** absent calibration/margin, incomplete coverage/faulted run,
  unjustified pooling, unstable A/A, or an isolated outlier with unresolved cause.
  Preserve its location/phase and relevant slices. One matched repeat of the
  affected comparison may distinguish corruption/observer error from recurrence;
  do not discard a valid original outlier or select only passing repeats.
- **PASS for this scope:** every required structural and frozen floating check
  passes over the observed ten updates and continuation paths. Report actual
  error metrics and A/A sample count alongside that limited claim. No inference
  of long-horizon convergence follows; failure to reject a difference is not a pass.

Capture/performance research may proceed with numerical/continuation gates open.
Baseline nondeterminism does not itself block that work; faults still invalidate
measurements of the affected window.

## Bounded disk and trajectory correction

Five-family stripes from design 001 can compare each family separately, but
independent nondeterministic runs cannot be called one joint trajectory merely
because common scalar metadata matches. Keep their per-family claims separate.
For bounded preliminary joint gradients/moments/deltas diagnostics, use
**two-update blocks with every family observed together**; for block 3–4, for example, fresh reference and consumer each
execute the real prefix 1–4 from S0 and retain only updates 3–4. Repeat for 1–2,
5–6, 7–8 and 9–10. Each block is a valid same-run joint observation after its
native prefix; across blocks these are repeated trajectories, not one exactly
reconstructed trajectory. These blocks are partial/cohort diagnostics and do
**not** satisfy the required coherent, uninterrupted ten-update comparison. Keep
common scalar/structural records for the full prefix and label this limitation. This costs
60 applied updates and ten launches per matched ten-update block collection,
excluding warmup; it is validation cost, never throughput evidence.

At the provisional 12.17-GiB/family FP32 estimate, two updates of four
nonparameter families plus three parameter endpoints use 11 families, about
133.8 GiB; S0's estimated 36.5 GiB brings retained payload to about 170.3 GiB.
One temporary preparameter family adds about 12.17 GiB (182.5 GiB total).
Consumer comparisons stream tensorwise, with FP64 delta scratch released per
tensor. Real aliases/replication/dtypes/checkpoint bytes must be inventoried
before writing; the **192-GiB figure is a provisional design target**, not an
owner-imposed hard cap or allocated disk. Include continuation checkpoint
lifetimes and metadata/headroom, and never keep all diagnostic blocks/pairs
simultaneously. One-update prefix diagnostics reduce space further but do not
close the uninterrupted-trajectory gate. Do not remove families or reduce
snapshot precision.

The full joint ten-update gate therefore retains an open storage decision.
Inventory actual dtypes/shards/aliases and serialized bytes; assess standard
lossless compression and free-space headroom before choosing. A single complete
reference of all five families plus P0 is approximately 620.5 GiB uncompressed
under the FP32 estimate, plus S0/metadata; the consumer need not write a second
archive. If this is the simplest feasible route, justify its measured bounded
peak and retain only that reference until all named consumers complete, then
delete with inventory. A larger temporary allowance needs the coordinator’s
explicit decision under the owner’s tight-retention preference. Do not invent
a concurrent dual-model/IPC harness to force the provisional estimate. Family
stripes or prefix blocks may inform calibration, but cannot silently replace
required same-run cross-family ten-update coverage. Working capture and the
actual byte inventory precede this storage decision.

For 5+5 bidirectional full-state continuation, retain only the native checkpoint
needed by named consumers and the active tail block. Compare loaded bytes before
execution, then resumed trajectories against matched uninterrupted controls with
the same frozen criteria; partial blocks alone do not close this trajectory gate. Keep full horizon and exact loader/RNG ownership;
recreate graphs, restore warmup/capture side effects, and never serialize graph
executables. Delete a checkpoint/reference only after every named consumer
finishes and a hash/result/deletion inventory is persisted. Do not retain unneeded final checkpoints or completed tensor payloads; retain
the justified single ten-update reference only until its named consumers finish.

## Cost and limits

This document changes no production code or runtime. Proposed validation copies
all selected local tensor elements to CPU and deliberately synchronizes, serializes
and compares them; prefix replay adds compilation/startup/training work. It adds
no comparison all-gather and must be off in performance runs. Observer impact on
capture, scheduling and peak memory remains a measured implementation gate.
The amendment supplies definitions and a freeze procedure, not numeric tolerance,
GPU correctness, throughput, continuation or quality evidence. Stop here until a
runnable candidate makes these conditional checks relevant.
