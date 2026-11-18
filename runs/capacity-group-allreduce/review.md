# Independent capacity aggregation integration and runtime-cost review

## Verdict and scope

No material source or test finding in the requested scope. The candidate is
ready for the authorized native control/candidate pair and final diagnostic.
This verdict does not close those execution gates or claim production speedup.

Reviewed once on 15 November 2026 using the `code-rules` and `perf-experiments`
skills, GIFT operating/review rules and applicable model/integer rules. Scope:
the arithmetic/collective-order diff, shared selector consumers, compile/SAC
visibility, added tests and measured costs. No source edit, GPU operation,
new test campaign or unrelated model investigation was performed.

Candidate `capacity.py` SHA256:
`164af05bd72ceb0675f0d9f756ecf04926ced17430a58dd2e38bc160c514dfdc`.
Original snapshot SHA256:
`2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120`.
The [reverse patch](original-order.patch) restores only the reviewed change.

## Integration findings and dispositions

- **Exact arithmetic preserved.** `select_capacity_routes` retains the int64
  histogram and excludes its inactive sentinel bucket from both group sums.
  It now evaluates `SUM_r(sum_e_in_group(counts[r,e]))` instead of
  `sum_e_in_group(SUM_r(counts[r,e]))`. Nonnegative bounded counters with sums
  within int64 have identical results. There is no cast or float reduction.
  `TokenDropConfig.source_capacity` still validates positive EP/expert counts
  and divisibility before the reshape; capacity and budget are unchanged.
- **Gate and selection preserved.** The original local expert histogram still
  supplies the local overload condition and branchless selector. The global
  condition remains an AND with any destination-group overload. Stable score
  selection, inactive masks and all host/branchless decisions are unchanged.
  The five added oracle cases check the old mathematical group loads, both
  gate conditions and expected masks, including local-only/global-only
  overload, ties and overload at another destination.
- **Shared consumers remain integrated.** Local, AllToAll, DeepEP and HybridEP
  dispatchers still enter `BaseEPTokenDispatcher._drop_routes`, which applies
  the keep mask to scores and IDs before transport. The changed value is an
  internal gate input; no consumer signature, route representation or gradient
  path changes. No configuration or persistent state changes affect optimizer,
  evaluation or checkpoint format.
- **Collective participation and order preserved.** There is still exactly one
  functional `all_reduce(..., "sum", ep_group)` in the group-gated path,
  using the same group. It remains before the conditional custom op and all
  empty/quota early returns. Empty sources contribute an EP-shaped zero vector
  and participate. No collective is hidden inside the custom op.
- **Compile/SAC boundary preserved.** The collective remains the same
  `_c10d_functional.all_reduce` operator classified as `comm` by
  `activation_checkpoint._op_categories`. The configured selective policy is
  unchanged. The int64 local sum can be recomputed under SAC; saved collective
  output now has EP elements. Existing two-rank Gloo tests check actual process
  group sequence counts, masks, loss and gradients across eager/compiled and
  checkpoint modes, including the selective empty-source path. Their existing
  compiled/full-AC/empty-source exclusion is retained and is not evidence for
  that excluded combination; the requested native selective policy remains
  the GPU acceptance gate.

## Runtime-cost reconciliation

The [pre-implementation review](preimplementation-costs.md) matches the final
source. One local `(EP, E/EP).sum(1)` moves across the collective; no additional
source-level tensor pass, host scalar read, synchronization, transfer or
instrumentation is added. The E-element histogram remains needed locally.
The functional collective's input/output tensor shrinks from E to EP counters
(1024 to 64 bytes for E128/EP8); these are tensor payload sizes, not wire bytes.
No persistent allocation is added. Runtime frequency remains one group-gated
selector call per affected forward/recomputation path, with existing SAC policy
determining whether communication is saved or replayed.

Moving the sum can change fusion, launch placement, intermediate lifetime and
overlap. Source equivalence does not establish identical compiled launches or
peak memory. The [component measurement](component-results.md) correctly labels
CUDA-event spans as inclusive eager call latency with host submission gaps,
communication and peer waits. Its alternating paired maximum-rank B-minus-A
medians are +0.128 us for the collective and +0.912 us for the full fragment;
neither establishes a resolved latency improvement. The report retains tails
and per-rank distributions and makes no isolated-kernel or whole-training claim.
Native compile/SAC timing, memory and final trace remain open gates owned by
the coordinating agent. Reduced payload alone must not be reported as speedup.

## Reused verification

Inspected the changed tests and supporting logs; no repeat was needed:

- [Formatted focused suite](group-tests-formatted.log): 13 passed.
- [Dispatcher, gradient and real Gloo checks](dispatcher-gradient-gloo.log):
  9 passed, including empty and unequal sources.
- [CPU fullgraph checks](compile-tests.log): 12 passed, 12 deselected;
  this receipt establishes only the selected CPU checks.
- [Real two-rank Gloo compile/SAC check](sac-gloo-tests.log): 1 passed.
- [Original-order regression](original-order-regression.log): the payload test
  actually fails when given 128 counters instead of the required 8.
  [Candidate payload check](payload-candidate.log): 1 passed.
- [NCCL component evidence](component-results.md): eight ranks pass exact
  group-load equality for bounded counts, values above int32 and an empty
  source with unequal peers. This is aggregation evidence, not full-model
  GPU gradient evidence.

The source worker is finalizing the command/lint receipts and correctness
summary. At this review, `correctness.md` and the report checklist still contain
pending text despite completed test logs; the coordinator must reconcile those
existing records before final handoff. This documentation status is not a
source blocker for the scheduled native runs. No material source finding is
open, and no additional review campaign is requested.
