# Four independent fusion experiments

Status: complete, 13 November 2026. The [assignment checklist](../../planning/README.md#active-assignment-four-independent-fusion-experiments)
records the finite acceptance scope. All four explicit-kernel variants completed
their scoped implementation/regression checks, native activation checks,
matched traces and required unprofiled comparisons. Input, routing and cap-zero
CE also completed the decision-relevant repeats; the active linear-CE port was
rejected after its first valid active pair. The results do not accept or enable
any candidate.

## Initial reconciliation

The starting task revision is clean `main` at
`da6774a1c2191bfec0e1cb4aa1dc34019b3b7df8`; `gift/` is the flattened active source.
The accepted control is V2 with explicit `num_prefetch_batches: 32` and
`dataloader.dataset.epoch_size: 51200`. Canonical defaults remain unchanged.
Consumer-owned epoch entry and corrected loader/checkpoint behavior remain.
The [V2 acceptance](../v2-prefetch-baseline/report.md#mcp-perfetto-comparison-and-acceptance)
is workload-specific; its one-pair throughput magnitude remains provisional.
The [baseline](../../docs/baseline.md) owns all other workload settings.

At the owner's request, `origin/main` was fetched and fast-forwarded to
`33b336a5` while preserving all experiment edits. That commit changes only
`.gitignore` and adds the requested [profile notes](../../docs/profile-my-ideas.md);
training-source bytes are unchanged. The previously missing notes have now been
read and reconciled: their ten eager preparation kernels and post-capacity
`indices_to_map` launch gaps match the selected boundaries. Historical slice
IDs/timings are pointers, not new V2 evidence. The separate BF16 expert-weight
reuse idea in those notes is outside these four experiments.

## Evidence and decisions

The [runtime inventory](runtime/inventory.md) and [machine-readable pins](runtime/pins.json)
verify direct execution on the existing one-node allocation, eight idle H100s,
the canonical Python/Torch/CUDA stack and all 29 fixture shards plus index against
the data receipt. No allocation or installation occurred. Each run rechecked
GPU availability and preserved its complete source/diff/config/runtime identities.
The existing runner and timer remain unchanged; raw native artifacts use its
existing Git-ignored `runs/overnight-streaming-perf/artifacts/raw` root with new
four-fusion run IDs. Compact evidence stays in this directory.

The owner subsequently required explicit handwritten Triton/CUDA/CuTe kernels
for all four variants. The original [input/routing design](prefix-routing-design.md)
retains its contract/source research, but its Inductor-helper selection is
superseded. Those provisional helpers are not accepted implementations and have
no performance evidence. The replacement must first reuse/adapt existing written
kernels or document their concrete interface gap. Capacity checks, token
dropping and host predicates remain unchanged. Existing V2 model/loss compilation
stays enabled; `torch.compile` around tensor expressions is insufficient.
The replacement [explicit-kernel design](explicit-kernel-design.md) records
inspected handwritten FLA/TE sources and concrete interface gaps. It specifies a
single-CTA preparation kernel and a direct per-token routing kernel. Preparation
scale fusion is conditional on exact ordered-division parity; existing aux-state
tensor identities and copy updates remain mandatory.

The [loss design](loss-design.md) selects the bundled FLA token-chunk/GEMM
implementation and the preserved Kindulov cap-zero CE algorithm as independent
alternatives. Both require centered statistics; linear-CE also requires correct
pre-cast head/denominator scaling, single target-gradient rounding and FP32
internal weight-gradient accumulation. Native PyTorch, Liger and CCE were
inspected as alternatives; exact pins, primary links and compatibility gaps
are recorded in the design. No dependency installation is required.

Each candidate retains an independent patch against V2. Unapplied helper modules
have no active imports in the control. Integration patches were applied one at
a time and source remained frozen during each run. The two loss experiments
never ran together. Explicit-kernel implementations, scoped checks, native
integration, measurements and trace gates are complete.

The [implementation manifest](runtime/implementation-pins.json) pins the frozen
helper modules, activation patches and selector at launch. The first fresh V2
control is `four-fusion-control-input-pair-control-001`; its native receipts live
under the existing raw-artifact root. The completed initial matrix used four
fresh control/candidate pairs, one matched V2 trace and one trace per candidate. Input,
routing and cap-zero CE also completed decision-relevant repeats. Because the
owner prohibited further V2 runs, queued routing/CE control repeats were
cancelled before launch; their candidate-only repeats reused the completed V2
input-repeat control, with order and elapsed gaps recorded in the receipts.

The first [input pair](input-pair-001.json) passed all 20 updates on eight ranks,
with ten measured updates and unchanged source/config during each run. Input
TPS is 156,318.29 versus 156,325.51 (+0.0046%); valid-target TPS is 156,280.12
versus 156,287.35. Rank-0 median update intervals are 13.3660 versus 13.3797 s.
Both series rise toward the end of the window; this observation does not by
itself identify compilation or justify a warmup extension. Window peak allocated
memory across ranks is 69.5350 versus 69.5222 GiB, with zero allocator retries or
OOMs. The initial pair and its repeat show no demonstrated TPS or memory benefit.
The [matched input trace analysis](input-trace-analysis.md)
verifies activation on both ranks and all three microbatches: preparation drops
from 11 launches to two, host elapsed falls 201–326 microseconds, and completion
through the first embedding improves 158–314 microseconds. The full sampled
three-F/B region is 2.645/2.790 ms longer. This establishes local acceleration
in these samples without an overall training benefit; the later repeat also does
not establish a training-throughput gain.

The first [routing pair](routing-pair-001.json) also passed both native 20-update
runs. Input TPS is 156,373.29 versus 156,819.10 (+0.2851%); valid-target TPS is
156,335.11 versus 156,780.82. Rank-0 median intervals are 13.3697 versus 13.3275 s;
peak allocated memory is 69.5308 versus 69.5293 GiB. The two identical V2
controls differ by +0.0352% TPS. The repeat is complete; no memory saving is
established. The [matched routing analysis](routing-trace-analysis.md)
verifies 39 calls per rank: seven original dispatches and six SAC recomputes per
microbatch, with layer-5 communication saved. Each metadata helper changes from
11 kernels and one copy to one explicit kernel. Mean host metadata time falls
248/270 to 111/107 microseconds on ranks 0/1; enclosing dispatch completion falls
1,631.8/1,613.2 to 1,460.8/1,471.2 microseconds. Capacity collectives, token-drop
predicates and stream synchronization remain. Rank-1 synchronization increases,
so no communication-removal benefit is claimed. Production-shape exact metadata
and standalone/compiled gradient comparisons are already covered by the
[scoped checks](routing-implementation.md); no full-model gradient dump is needed.

The first [linear-CE attempt](linear-ce-pair-001.json) passed both full native runs,
but subsequent actual Perfetto inspection found **zero FLCE custom ops or
explicit CE kernels on both ranks**. Device event counts exactly match V2.
This is a failed activation gate: the following observations do not measure
the proposed FLCE kernel and cannot establish its speed or memory effect.
Input TPS is 155,988.30 versus 153,502.51 (−1.5936%); valid-target TPS is
155,950.21 versus 153,465.03. Peak allocated memory falls from 69.5363 to
67.5651 GiB. Candidate rank-0 median interval is 13.4415 s versus 13.4167 s,
while its maximum is 15.6194 s versus 13.7048 s. The long candidate interval is
retained in the complete window. Native microbatch records localize it to
update 11, microbatch 18: 2,606–2,608 ms across all eight ranks, versus roughly
378 ms ordinarily. This accounts for approximately 2.23 seconds, while the
paired window deficit is approximately 2.18 seconds. Logged GC is negligible;
observers were disabled and there is no actual compilation marker. The cause
remains unknown. A warmup extension was not justified by this evidence. At the
time, the native fallback had to be fixed and verified before a fresh matched
comparison and trace; the correction and follow-up are recorded below. The
inactive attempt remains preserved.

The fallback was traced to the canonical `Module.parallelize()` head wrapper,
which the direct-head fixtures omitted. The [wrapper correction](loss-parallel-wrapper-fix.md)
records exact setup provenance and accepts only verified no-op state/input/output
redistribution; other wrappers keep the complete original path. Independent
[review](loss-parallel-wrapper-fix-review.md) and focused old-fail/new-pass
regressions cover actual parallelization plus eager/compiled GFSDP, two outer
chunks, two-microbatch no-sync accumulation, components, dH/dW and flag/state
restoration. Eight CPU cases and three CUDA cases pass. The common metadata is
setup-only and adds no tensor or communication work. Corrected runs use the
[new source manifest](runtime/post-wrapper-implementation-pins.json) and a fresh
matched baseline capture. Corrected traces now verify activation on both ranks:
six native linear-CE calls, 48 explicit CE forward and 48 backward kernels,
and 144 projection/dH/dW GEMMs per rank, with no original compiled CE kernels.
The [matched active analysis](active-linear-ce-trace-analysis.md) measures the
same whole-loss forward/dH/dW guard: device span rises from 88.78–91.36 to
105.90–107.06 ms in all six samples. Projection and dH costs are similar, while
tiled FP32 dW accumulation costs approximately 38–42 ms per microbatch versus
21–23 ms for the baseline weight-gradient GEMMs. This outweighs the smaller
CE cost. The first chunk also regresses without overlapping owned NCCL.
The [fresh active pair](linear-ce-active-pair-002.json) completes the required
20 native updates per arm. Complete measured-window input TPS is 156,368.31
versus 151,224.14 (**−3.2898%**); valid-target TPS is 156,330.13 versus
151,187.22. Rank-0 median intervals rise from 13.3957 to 13.8164 s. Matched
window peak allocated memory falls from 69.5336 to 66.8953 GiB (−2.6384 GiB),
and reserved from 70.2012 to 67.7012 GiB. All ranks have zero retries/OOMs.
**Reject this 1,024-token-tile FLCE port for throughput.** Its negative effect
and local regression are clear relative to observed controls, so no repeat or
full-shape retained-candidate numerical replay is needed. The single-pair
magnitude remains provisional. This does not reject other FLCE implementations;
the existing small dH/dW and GFSDP checks retain their limited correctness scope.
The inactive attempt above remains excluded from candidate performance claims.

The first [cap-zero CE pair](ce-pair-001.json) passed both native 20-update runs:
input TPS 156,177.10 versus 156,650.91 (+0.3034%), valid-target TPS 156,138.98
versus 156,612.66, and rank-0 median intervals 13.3934 versus 13.3301 s. Peak
allocated memory is 69.5181 versus 69.5374 GiB; no memory reduction is observed.
The [matched CE analysis](ce-trace-analysis.md) verifies explicit forward and
backward calls in generated rank-specific compiled graphs and the corresponding
GPU kernels. Equivalent CE elapsed, including both heads, forward/backward,
normalization and internal gaps, falls in all 12 chunks by 0.434–1.306 ms.
Two backward-only samples regress. Whole-head stream-7 elapsed improves in five
of six samples, with a 0.997 ms regression in the sixth; downstream starts also
remain mixed. Both implementations read BF16 logits once and retain two FP32
row-stat vectors. The improvement changes recurrence, exponential evaluation
and backward reduction work; it removes no full-vocabulary FP32 buffer or
logical kernel launch. The [representative local replay](ce-numerics.md) passes:
two full-vocabulary folded-head chunks have identical total weighted FP32 loss,
all gradient differences are at most one BF16 ULP, ignored gradients are zero,
and sampled FP64-reference errors remain comparable to V2. This establishes the
recorded synthetic local-block gate, not convergence or continuation. The
candidate-only throughput repeat is complete and reuses the latest completed V2
control, as the owner prohibits further V2 runs.
Across the four initial identical controls, input TPS ranges from 155,988.30 to
156,373.29 (approximately 0.247%); this is descriptive control variation, not a
confidence interval. The completed repeat effects for input, routing and CE are
small and vary from their initial-pair effects; none establishes a robust
whole-training throughput gain.

## Comparison

The loss [implementation/check record](loss-implementation.md) and independent
[integration/runtime-cost review](loss-review.md) are complete at source level.
Its material finding, FLCE target-metadata validation, is
[fixed and verified](loss-metadata-fix.md). Subsequent producer/wrapper findings
and native activation are reconciled below. Small GFSDP checks and bounded
captures do not establish general convergence or continuation.

The subsequent cross-boundary [kernel review](kernel-review.md) found that the
actual DP8 producer returns a Python float denominator, missed by the synthetic
Tensor-denominator input/FLCE fixtures. Both adapters now handle that native
producer: the [input checks](input-implementation.md) cover exact auxiliary-state
parity, and the [linear-CE correction](loss-native-denominator-fix.md) covers
compiled reuse as the count changes without adding a tensor transfer. The input
activation patch now contains only its Trainer integration hunk. All five
actual-workspace driver dry runs passed before launch. Focused independent
review resolved all three findings without a new material finding. Routing has
no material source finding within its documented
producer contract. Its [check record](routing-implementation.md) distinguishes
real CUDA tests, mocked dispatch evidence and the pinned Torch opcheck limitation.

The [final independent evidence review](final-evidence-review.md) reconciles
all prior findings against the 15 final source pins, the corrected native
activation, representative CE replay and actual trace cost interpretation.
It found no new material source or numerical issue. The completed active FLCE
pair supports rejection. All four finite comparison gates are complete. Input's
repeat used a fresh control in reverse order; routing and CE candidate-only
repeats reuse the completed input-repeat V2 control. The independent review's
prior pending wording was written before these final comparisons.

| Experiment | Source and patch pins | Checks and native activation | Matched target elapsed | Global input/valid-target TPS; peak allocated memory Δ (candidate − control) | Numerical findings and verdict |
| --- | --- | --- | --- | --- | --- |
| GPU preparation | Task HEAD `33b336a5`; [explicit-kernel design](explicit-kernel-design.md); helper pin `db35fc56`; Trainer-only patch `e0eea03e` | 16 focused checks; native 20-update pair/repeat; explicit scalar/kernel path active on ranks 0/1 | Host preparation −201–326 µs and completion through first embedding −158–314 µs; sampled three-F/B guard is +2.645/+2.790 ms longer | Initial **+0.0046%**, repeat **+0.1630%**; memory **−0.0128/−0.0181 GiB** | Native Python-float and tensor-denominator parity plus auxiliary-state checks pass. **Inconclusive** for training TPS. Local region improves; no material memory saving. |
| Routing metadata | Task HEAD `33b336a5`; [explicit-kernel design](explicit-kernel-design.md); helper pin `2fb70d05`; import patch `4a9e17c8` | 19 passed, 1 skipped, 1 deselected; native 20-update pair/repeat; 39 dispatch metadata calls and kernels per rank active | Host metadata 248/270 → 111/107 µs; dispatch completion 1,631.8/1,613.2 → 1,460.8/1,471.2 µs; capacity collectives, predicates and stream waits remain | Initial **+0.2851%**, repeat **+0.0897%**; memory **−0.0015/−0.0143 GiB** | Exact production-shape metadata and standalone/compiled gradient checks pass. **Inconclusive** for training TPS. Local region improves; no material memory saving. |
| Linear-CE | FLA `1d2281ff` plus centered Kindulov `b9838033` CE; helper pin `d7cda84d`; [linear patch](linear-ce.patch) `a38757cd`, wrapper fix `f675510a`; failed inactive pair excluded | Corrected focused wrapper/GFSDP checks; active native 20-update pair; six native calls plus 144 projection/dH/dW GEMMs per rank observed | Complete loss forward/dH/dW device span regresses in all six samples: 88.78–91.36 → 105.90–107.06 ms; tiled FP32 dW is 38–42 vs 21–23 ms | **−3.2898%**; peak allocated memory **−2.6384 GiB** (69.5336 → 66.8953) | Small dH/dW and accumulation/lifecycle checks pass. **Reject this 1,024-token-tile port**: slower whole head and training despite lower memory. Full-shape replay is unnecessary for this rejected candidate. |
| Cap-zero CE | Kindulov `b9838033`; helper pin `d101584f`; [CE patch](ce.patch) `d7cc5e5e` | Scoped checks; native 20-update pair/repeat; explicit forward/backward kernels active in compiled graphs on ranks 0/1 | Equivalent CE elapsed falls 0.434–1.306 ms in all 12 chunks; two backward-only regressions and mixed whole-head/downstream effects remain | Initial **+0.3034%**, repeat **+0.0560%**; memory **+0.0194/−0.0074 GiB** | Precision-aware synthetic local replay passes: equal total weighted FP32 loss, at most one BF16 ULP across checked gradients, ignored gradients zero, sampled FP64 errors comparable. **Inconclusive** for training TPS. No convergence/continuation claim. |

Input's repeat ran candidate first, finishing 0.161 seconds before its fresh V2
control began. Routing began 0.111 seconds after that same control ended. The
cap-zero CE candidate began 1,167.022 seconds after that control ended. These
are completed candidate-only comparisons, not independent fresh controls for
routing and CE. All pairs used 10 warmup plus 10 measured updates on eight ranks
with identical resolved V2 configuration apart from the run dump path. Each
measured arm processed 20,971,520 physical input and 20,966,400 valid supervised
tokens, making the input and valid-target TPS percentage changes equal. Exact
values, run series, memory phases, identities, commands and hashes are retained
in the [initial input](input-pair-001.json), [input repeat](input-repeat-pair-002.json),
[initial routing](routing-pair-001.json), [routing repeat](routing-repeat-pair-002.json),
[initial CE](ce-pair-001.json), [CE repeat](ce-repeat-pair-002.json), and
[active FLCE](linear-ce-active-pair-002.json) receipts.

The finite comparison is complete at the accepted crop scope. No candidate is
enabled or accepted; the independent patches and helpers remain reproducible.
General convergence and continuation were not required gates for this bounded
comparison and are not established by it. The original FP8, compile, SAC and
MTP settings remained active. The inactive FLCE attempt remains recorded as a
failed activation and is excluded from its candidate cost claims.

Final root checks confirmed all five core activation targets clean against
`33b336a5`, with the sole tracked `gift/` change being the expected
`gift/torchtitan/protocols/module.py` correction. All 39 maintained root/runtime/
driver-record JSON files parsed. No GPU had a compute application, and the owned
controller/driver PIDs were absent. No new run or training-source change followed
these checks. The owner's subsequent request authorizes committing and pushing
the completed changes to `origin/main`.
