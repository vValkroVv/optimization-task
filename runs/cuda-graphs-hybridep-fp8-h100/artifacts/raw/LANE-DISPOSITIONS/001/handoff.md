# Finite candidate-lane dispositions

## Result

**The campaign remains inconclusive.** A008 proves a working persistent loss
forward/backward graph mechanism in native training. Existing evidence does not
justify another explicit region or a static-count implementation. This resolves
lane selection for the available evidence; it does not close finalist acceptance
or demonstrate that every possible CUDA Graph boundary is incompatible.

Acceptance for this bounded review: reconcile task sections 3–4 and completion
rules; give A/B/C a disposition grounded in completed evidence; distinguish
mechanism, performance, numerical and continuation gates; check local links and
whitespace. No new trace analysis, experiment, source edit or GPU inquiry is part
of this review. This document has no production-runtime impact.

| Lane | Finite disposition | Evidence and decision |
| --- | --- | --- |
| A — compiler regions | **Mechanism PASS for loss-only A008; final acceptance unresolved.** Dense/loss A007 remains an unsettled capture policy. | A008 reuses actual CE forward/backward graphs on every rank with no recording/instantiation in updates 21–23. Retain this candidate for the existing acceptance gates. A007 dense backward records on 96/96 calls per rank; that proves churn, not universal dense-capture incompatibility or a final performance verdict. |
| B — one explicit region | **Unattempted: no demonstrated material uncovered host-submission target in the completed evidence.** | Baseline analysis supplies no qualifying attention/shared-FFN/loss region. A008 already captures actual CE forward/backward, so implementing that same regional boundary again would duplicate a resolved mechanism question. Whole chunked-loss/head capture is not established, but no material exposed gap and safe integration contract for that larger boundary has been demonstrated. |
| C — graph-safe MoE counts | **Unattempted: conditional prerequisites remain unproven.** | Blocking counts are active. Their host duration is not exposed elapsed time, and no material remaining count-related replay opportunity after A/B is demonstrated. A legal complete Hopper count-consumer/capacity path is also missing. No static-count source change is justified by this evidence. |

## Evidence reconciliation

### Source and original baseline

The [task](../../../../../../backlog/tasks/opt-004-cuda-graphs-hybridep-fp8.md)
freezes upstream GIFT `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2` and
the exact 3B MoE/HybridEP/FP8/FlexAttention preset, batches, FSDP2, SAC and model/loss
compilation. [Source006](../../SOURCE-SNAPSHOT/006/manifest.json) pins A007/A008
and their matched controls. [Source007](../../SOURCE-SNAPSHOT/007/manifest.json)
records only the diagnostic completion-window extension and its tests/README
relative to 006; it is not a new graph candidate or performance control.

The [all-rank baseline analysis](../../BASELINE-TRACE/analysis/001/handoff.md)
and [summary](../../BASELINE-TRACE/analysis/001/summary.json) establish no graph
launches with the original graph policy. Across its complete profiled windows,
host launch unions are 913.940–970.543 ms and overlap device work. All-stream
idle is 56.267–70.887 ms in 3283.130–3331.349 ms updates. HybridEP dispatch host
synchronization is 366.439–387.780 ms, but its intersection with all-stream idle
is only 8.512–9.132 ms. None of these scopes is a removable-cost estimate or a
speedup bound: communication can hide compute gaps. The largest examined compute
gaps end at kernels already queued before the gaps, so they do not establish a
late-host-submission region for B. No new critical-path inference is made here.

### A: actual capture, finite fallback and unresolved acceptance

[A007's mechanism review](../../LANE-A/analysis/007/mechanism-review.md) proves
actual dense/loss forward/backward graph launches on every rank, with dense
backward fresh recording on all 96 calls per rank in the three-update window.
Static-pointer guard failures persist near shutdown. The exact address producer
is unresolved; assigning every mismatch to FSDP would exceed the evidence.
The earlier failures and their scoped remedies are retained in the
[campaign report](../../../../report.md): Flex block-mask lifetime, hidden private
workspace in unsafe dense GEMM, nested HybridEP pool ownership and generated
backward `buf66`. The task-local `torch.Tag.cudagraph_unsafe` adapter around
unchanged vendor dense GEMM isolates its hidden workspace allocation from graph
partitions; it does not exclude the dense block from compilation or execution.
That capture-safety adaptation and `mix_order_reduction=false` are causal
prerequisites shared by graph-off/on controls, with the original baseline retained
separately. A007 excludes the 17 MoE blocks from graphs; A008 excludes all 18 model
blocks. These block exclusions are distinct from the GEMM adapter. Neither
prerequisite can be credited as a replay effect.

The [quiet dense/loss pair](../../MATCHED-RECAPTURE/attempts/002/paired-handoff.md)
measures five-update windows at 318,557.08 off versus 36,055.46 on global useful
tokens/s. These are descriptive only: one pair supplies no A/A noise, and C1's
run-level event-completion gate fails after ENOSPC/persistence errors. Missing
markers are not proof of skipped optimizer updates. Here C0/C1 name controls,
not an implementation of lane C. Neither this pair nor capture churn supports
a completed negative performance or incompatibility verdict.

[A008 replay review](../../LANE-A/analysis/008/mechanism-review.md) and
[native validity](../../LANE-A/attempts/008/monitor-handoff.md) jointly establish
25 native updates and 100 forward/backward microbatches per rank, intact logs,
all 18 blocks still compiled/SAC, and persistent CE replay on all eight ranks.
Updates 21–23 have 32 CE forward and 32 backward launches per update/rank, the
same eight IDs per phase, and zero late captures/instantiations. Head GEMM,
model, optimizer and communication graph capture are not proved. Keeping those
operations outside captured regions while retaining required compilation/SAC
preserves the native full-update path; the optimizer remains native eager.
They must still be included in integrated measurement
and correctness checks. Persistent whole-update capture is not required by the
regional lane, and actual forward/backward integration is demonstrated.

The first [loss-only performance control](../../LOSS-ONLY-PERF/preparation/001/handoff.md)
subsequently fails with CUDA illegal instruction during update 48 after 47
completed updates; its outcome and held attempts are retained in the report.
There is no completed 50-update window, A/A noise or repeated loss-only A/B/A/B
result. Earlier graph-off faults at 22/33 remain distinct evidence. This fault
does not reject the independently demonstrated graph-on mechanism, and cannot
be dismissed as nondeterminism. The separate completion-prefix diagnostic and
loaded-binary receipt are localization work, not a repair or performance control.

### B/C: why additional implementation is not selected

A008 answers the CE regional mechanism question B could otherwise duplicate.
The larger loss/head loop retains FP32 hidden-gradient assembly and FSDP head
hook semantics; absence of its capture is not permission to remove those required
behaviors. Existing evidence provides no concrete uncovered region meeting B's
material host-submission and preservable-lifetime prerequisites. This is an
evidence-limited unattempted disposition, not proof that another region cannot
help. A demonstrated material uncovered region with a legal full
forward/backward/SAC integration contract would change the selection decision;
none is identified in the completed analysis.

The report's **Research and integration findings** records the existing pinned
research: blocking exact-count dispatch; sparse FP8 rejection of nonblocking
capacity mode; Hopper DeepGEMM NT weight gradients requiring host group sizes;
device-count TN/PSUM and ECHO alternatives targeting SM100/Blackwell. Simply
removing dispatch synchronization leaves capture-time WGRAD metadata stale.
Its worst-case capacity arithmetic also rules out assuming balanced routing as
a safety proof. Preserve changing routes, empty/skewed experts, safe capacity,
no dropped tokens and the FP8 recipe. These are required contracts, not deferred
optimizations. No inspected legal complete count-consumer route and no trace
demonstrated material remaining count-related replay blocker jointly justify C.
Either missing prerequisite cannot be converted into universal incompatibility.

## Remaining acceptance gates and handoff

Retain A008 as a runnable mechanism candidate with **inconclusive / gate open**
campaign status. Required production-policy A/A variability and precision-based
criteria must precede interpreting matched floating comparisons. Preserve strict
A/A bit inequality as diagnostic only; do not invent tolerances. Ten-update
elementwise gradients/clipping/updates/floating state, structural invariants,
FP8/SAC/routing behavior, and fresh-process bidirectional full-state continuation
remain open. Baseline/finalist repeated full-update throughput, matched lifecycle
memory, 500-update sustained runs and applicable integration/runtime-cost review
remain required. Actual fault resolution and acceptance evidence determine
whether the runnable candidate can advance; this review schedules no new GPU work.

The completion exception for all justified lanes being reproducibly incompatible
does not apply: A008 is runnable. B/C dispositions do not waive A's numerical,
performance, memory or continuation gates. No beneficial, feasible-no-benefit or
pinned-target-incompatible final result is supported yet.
