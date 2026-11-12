# Streaming numerical comparison contract

Design judgment, 2026-11-12. **Yes: reuse one direct ten-update reference for
production A/A, then A/B, without a second five-family archive.** No numerical
passing values are established here. Uses code-rules/perf-experiments; this
document changes no runtime. No source edits, GPU work, installs or cleanup.

Finite checklist: resolve reference reuse and delta storage; define metrics and
exact invariants; identify reusable interfaces and concrete producer omissions;
specify one-update verification and calibration/freeze gates. This does not
repeat the [producer review](../../NUMERICAL-PROBE-REVIEW/001/REVIEW.md).

## Storage and integration decision

[006](../../NUMERICAL-DESIGN/006/handoff.md) supersedes historical encoding in
[003](../../NUMERICAL-DESIGN/003/handoff.md) and
[005](../../NUMERICAL-DESIGN/005/handoff.md): retain native full S0 plus actual
preclip/postclip/P/m/v for ten uninterrupted updates. Nominal reference is
644.9121 GiB; measured metadata, small state, partials and checkpoints must fit
the **700-GiB total live cap plus 32-GiB free reserve**. Include comparison
outputs and simultaneous rank reservations. Shared free space is not reserved.

Each consumer performs an actual native full S0 restore and proves its hashes,
ownership, optimizer, loader and RNG match. Initialize its previous-P CPU cache
from that restored P0: one FP32 family, nominal 1.521 GiB/rank. The verified
common P0 also supplies the first reference delta. For update t>1, read reference
P[t-1] and P[t] from adjacent stored endpoints. Compute both actual deltas
`float64(P[t])-float64(P[t-1])`, then compare them. Replace each consumer cache
entry only after its delta is measured. Check each trajectory's own pre-update
P against its previous endpoint. Endpoint equality cannot replace delta metrics.
No P0 export, partial-DCP reader, decoder, second GPU model or RAM server.

After the mandatory one-update native readiness probe, implement an experiment-
owned comparison sink alongside the existing payload sink: `begin_update`,
`consume(identity, family, phase, native_cpu_tensor_or_None, metadata)`, and
`end_update`. These are proposed interfaces. Reuse callback order, local-shard
readiness, ownership records and all-rank status boundaries; do not copy the
trainer loop. Reference mode writes; consumer mode reads matched reference
entries and updates compact metrics without persisting its large tensors.
Keep tensors immutable through synchronous consumption; release scratch per
tensor. Mark completion only after every required rank/update/family finishes.

## Metrics

For every canonical FQN/local shard/update/phase, promote comparison copies to
FP64 before subtraction. With reference x, consumer y and d=y-x, retain integer
element/nonfinite/bit-mismatch counts, sum(x), sum(y), sum(d), sum(abs(d)),
sum(x²), sum(y²), sum(x*y), sum(d²), max(abs(d)) and its coordinate/values.
Accumulate chunk contributions in fixed order with FP64 sums. These support
bias, MAE, RMSE, absolute L2, relative L2 and cosine; report undefined zero-norm
ratios explicitly, without epsilon denominators. Also retain magnitude-stratum
counts/error summaries, declared relative-error maxima away from a justified
near-zero boundary, and absolute error near zero. Keep bounded worst-coordinate
slices and explicitly labelled approximate quantiles if used. No full error
arrays are required.

Apply this to actual per-microbatch loss and native summed update loss,
elementwise accumulated preclip and actual postclip gradients, native returned
norm, P/m/v endpoints, actual deltas, and required floating auxiliary/router/
persistent state. Compare native optimizer step exactly, including dtype and
None state. Derive clipping coefficient separately from native norm, configured
max_norm and the pinned native scalar arithmetic; label it **derived**, never
observed. Postclip tensors remain the actual clipping-output evidence.

Preserve individual shard/expert/layer results; global summaries cannot mask a
bad expert. Deduplicate actual aliases, retain replicas and ownership, and do
not sum replicated elements as independent samples. Record zero/empty tensors
and nonfinite locations explicitly; nonfinite arithmetic fails. Independent
bitwise loss/norm equality is diagnostic only. Same-execution payload integrity
and native saved-state restoration still require exact bytes.

## Exact contracts and producer data

Require matching input/label elements, signed integer widths, masks/packed
boundaries/context, valid-token counts, loss weights/normalizers, four ordered
backwards before clipping, one native optimizer/scheduler step, used versus
next LR, group/state keys, aliases, shape/dtype/stride/storage offset, mesh/
placements/rank coverage, ownership, None versus zero, empty shards and phase
order. Preserve AC recomputation and checkpoint semantics. Match loader position
and exact saved/pending/restored RNG at their owning boundaries; preserve RNG
ownership/draw semantics without imposing independent future floating equality.

Routing must retain capacity/no-drop/count/reset semantics. Floating near-ties
can change later assignments/counts; compare those observations without making
cross-trajectory identity a blanket requirement. Explain unexpected divergence
on the first common-state update. Preserve actual FP32 P/grad/moments and
accumulation/reduction precision, BF16/FP8 operation/cast policies and quantizer
scales; verify actual dtypes instead of assuming the storage estimate. FP8
spacing is not an end-to-end gradient or Adam tolerance.

Concrete small-payload additions are needed **before ten-update reference
generation**, not before the current one-update inventory/readback readiness:

- `after_forward` currently stores auxiliary totals as hashes: retain actual
  totals plus names/alias metadata; existing native-weigh scale payload stays.
- `_router_state` currently hashes counts/bias at preclip, after container hook
  and post-update: retain those actual small tensors so reset/count semantics
  and floating bias differences are measurable.
- `_buffer_inventory` currently retains hashes: use the one-update inventory
  to classify necessary mutable persistent floating state and retain actual
  boundary values. Immutable reconstructed caches retain reconstruction checks;
  aux scale remains checked after native weighing. No general state walker or
  transient FP8 tracer is required. Unexposed necessary state remains a gate.

## Reuse and verification before scaling

Source references below are relative to
[the staged gift tree](../../NUMERICAL-PROBE-IMPLEMENTATION/001/staged/gift/):
`numerical_probe.py` under `torchtitan/experiments/cuda_graph_campaign` supplies
`_TensorPackReader.read/verify_all` (386–467), `_state_equal` (1412), prepared
input capture (1760), native loss capture (1935), preclip (2004), inner pre-hook
(2094), endpoints (2161), and batch-end completion (2264). Lines reflect the
inspected evolving staging, not a frozen source revision. Reuse these contracts
after the ongoing R1/R2/R3 corrections; their review remains separately owned.
The staged `test_campaign_numerical_probe.py:199,386,436,449` already supplies
native CPU accumulation, callback-order controls and corrupt-readback fixtures.

Targeted existing comparator search found
[compare_numerics.py](../../../../../../gift/agent_tooling/numerics_debugging/compare_numerics.py)
(activation summary/fuzzy matching) and
[loss_compare.py](../../../../../../gift/scripts/loss_compare.py)
(scalar TensorBoard losses, forced deterministic mode). Neither implements
this exact-key elementwise native-state/delta contract; do not inherit their
matching heuristics, denominator floors or deterministic launch overrides.
Reuse the staged reader rather than introducing a storage library.

Before ten-update A/A, exercise the actual sink/reader on one-update producer
records: self-comparison gives zero errors and defined zero/empty behavior;
known element perturbations produce independently calculated FP64 metrics;
equal norms with different gradients and equal final P with different previous
P expose the required differences. Test missing/extra/wrong-phase/None/dtype/
ownership records, incorrect step/LR, corrupted/truncated payload, nonfinite
values, and missing-rank completion. Verify measured CPU peak/cache lifetime,
bounded outputs and failure propagation. CPU fixtures prove comparer behavior;
the native target record proves producer-to-reader integration, not GPU parity.

## Calibration, freeze and remaining gates

Follow [002](../../NUMERICAL-DESIGN/002/acceptance-amendment.md). One independent
graph-off consumer against the stored graph-off trajectory supplies one A/A
pair, not ten independent replicates. Retain all per-update/per-family/stratum
metrics, magnitude/error distributions, clipping regimes, exact-invariant
results, source/runtime/config/topology/activation identities and first-update
versus later amplification. Shared reference reuse also correlates comparisons.
No universal tail, long-horizon quality or equivalence claim follows.

Predetermine these metric outputs for both A/A and A/B; each is computed from
actual elementwise differences, rather than differences between aggregate norms.
Before viewing A/B differences, justify metric/scalar/delta limits and near-zero
policies from actual operation precision and recorded A/A behavior. Freeze
formulas, strata, values/provenance and verdicts in a hashed manifest. Compact
summaries do not answer arbitrary retrospective elementwise-band queries; no
such query or second archive is required by this contract. Do not set each
limit to its A/A maximum or a multiplier
of it. If precision evidence cannot justify a criterion, leave it INCONCLUSIVE.
Only an actual unresolved variability/outlier decision justifies proposing one
additional independent A/A pair with its question and possible decisions stated
first; more samples cannot repair a missing precision argument. No extra
nondeterminism-elimination campaign. Missing calibration fails no numerical
band by itself and grants no PASS.

Open gates: producer corrections/review, native one-update readiness and actual
inventory/bytes/host peak, comparer implementation/behavior integration, complete
ten-update A/A and justified frozen criteria, graph-on observation inertness,
ten-update A/B, bidirectional full-state continuation and existing CUDA fault/
campaign performance gates. Streaming adds full-family D2H/FP64 CPU work and
reference reads (including prior P); measure host headroom/I/O, avoid comparison
all-gathers and keep observation absent from throughput runs. No performance,
production-readiness or numerical-equivalence claim is made.
