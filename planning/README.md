# Work plan

[Current state](../docs/status.md) owns results and unknowns;
[candidates](../docs/candidates.md) owns mechanisms/source pointers.
Follow [workflow](workflow.md) and [validation](validation.md).

## Active assignment: expert GFSDP optimizations

Use English for all updates, assignments, code, and documentation for this goal.
The starting recipe is [S + GPU capacity selector + five Hopper fusions](../docs/baseline.md#current-final-task-baseline).
Use the existing eight-H100 `sr008-gpu1` allocation and the task checkout at
`/workspace-SR008.nfs2/vkropotin/optimization-task`. Reuse the verified compiler
cache. Keep the selected architecture and setup fixed in every arm, including
EP8/eFSDP1, dense FSDP8, TP/PP/CP1, production-tokenizer data, compile, SAC,
EMA, MTP, and 32-microbatch accumulation.

1. Inspect the LLM Foundry FP8 gather and adapt the applicable parts for cached
   routed-expert shards and the GFSDP gather path. Measure per-generation
   quantization and reuse on the existing singleton expert groups.
2. Write routed-expert dW to the existing FP32 reduce-scatter slot from the
   DeepGEMM epilogue, preserving the BF16 rounding boundary.
3. Preserve tail expert weights across nonfinal LM-head backward callbacks and
   verify the dense MTP re-gathers separately.

For each candidate, document pinned implementation research and cost, run
focused correctness checks, then collect a separate 10+10 run and matched
profile against the fixed EP8 control. Measure a combined arm and promote only the
candidates with a verified gain. Commit and push reviewed milestones to `main`,
then fast-forward the server checkout and measure there. Continue profiling
and validating additional optimizations after the three-candidate campaign,
prioritizing throughput gains that persist as model depth grows.
[Candidate designs](../docs/profile-my-ideas.md#fp8-all-gather-for-sharded-routed-expert-weights)

For tail retention, adapt the pinned
[vision-runtime callback guard](../references/vision-runtime/source/torchtitan/distributed/gfsdp/_state.py#L607).
The [pinned PyTorch callback](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/distributed/fsdp/_fully_shard/_fsdp_state.py#L379-L416)
uses the final-backward flag to gate finalization. GFSDP also needs to retain
nonparticipating groups' shared-buffer ownership through nonfinal callbacks.
Test owner-safe release and pending gather completion alongside participation.
The candidate adds a host-side state check and uses existing buffer ranges and
events. On EP8, the two expert scopes per microbatch are singleton paths. The
two dense MTP re-gathers in the selected trace are DP8 collectives.

For FP8 weights, the inspected LLM Foundry revision is
`eca85af5e254520722efdc3de1a7c9bb834d1a2a`, with GigaFSDP gitlink
`27a1825817bec332a5e327ddcb23cabb122bb32e`. Its expert-preparation,
byte-gather, zero-copy view, and consumer handoff patterns inform this port.
Keep GIFT's 128-by-128 quantizer arithmetic and the FP32-to-BF16 rounding
boundary. On EP8/eFSDP1, replace the per-generation BF16 cache with E4M3 data
and FP32 scales; no expert-weight network payload is involved. Verify that
compiled forward, backward, and SAC resolve the intended weight generation,
including layer 5 and cache invalidation after optimizer, EMA, and load.
[Megatron FP8 lifecycle reference](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.20.0/megatron/core/fp8_utils.py#L620-L650)

For direct dW, the deployed DeepGEMM `2.3.0+f093a4c` SM90 kernel header has
SHA-256 `fd8a620440cc31ae7d01010fde1f914aae6998f2dcf6ede2a65e3b5e76001726`.
Its BF16 store uses TMA reduce-add into a zeroed destination. The candidate
needs a dedicated Hopper K-grouped epilogue with the same rounded values in
the existing FP32 slot, covering experts with zero tokens. Preserve the
existing singleton collective, FP32 shard accumulation, and release event.
Verify aligned output views, special values, and the compiled backward's
mutation and data lifetimes before timing the shortened packing path.
[NVIDIA TMA semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)

<a id="active-assignment-seven-configuration-timings-profiles-and-drift"></a>
## Completed assignment: seven-configuration timings, profiles and drift

The revised 15 November goal authorizes minimal native embedding-stage selection
in this task's `gift/`, targeted checks, serial A/B/C/D0/S/SE/SE-fused timings
at exactly 40+100 updates, four separate update-21 profiles and bounded drift
research on the current eight-H100 server. The
[single report](../runs/schedule-embedding/report.md#revised-measurement-assignment)
owns the finite checklist. Preserve the production-data workload, precision,
compile/SAC, accumulation and existing storage budgets. Reuse the saved D cache.
No new allocation, installation or publication is authorized. The independent
capacity edits and incomplete evidence below remain preserved.

Stage selection, all seven timing proofs, the current-data drift record and all
four native captures are complete. The actual Perfetto imports, comparisons and
GPU-tail acceptance pass, and the [final integration review](../runs/schedule-embedding/final-campaign-review.md)
is complete. The [campaign report](../runs/schedule-embedding/report.md#revised-perfetto-profile-findings)
and [current state](../docs/status.md#seven-configuration-campaign-measurements-complete-layers7-overlap-unmet)
own the results. The original layers.7 overlap objective remains unmet, and
MTP-prefix-specific overlap ownership is unestablished; neither is hidden by
the positive generic local-compute overlap. Drift cause and fused-specialization
settling remain unknown. No further GPU work is scheduled for this assignment.

## Separate assignment: capacity destination-group all-reduce

Authorized by the 15 November 2026 goal attachment. Change only the local/group
aggregation order in this task's capacity path, preserving exact drop behavior
and compile/SAC semantics. The [single report](../runs/capacity-group-allreduce/report.md#scope-and-finite-acceptance)
owns the finite checklist: focused correctness and one scoped review, warmed
eight-rank NCCL component/fragment measurements, exactly one native production-data
D control/candidate 10+10 pair, and a bounded final candidate update-21 trace.
Use only the existing owner allocation; no installation, new allocation or
publication is authorized. The completed production-data input census and
accepted MTP/FP8 decisions are reused.

## Global priority: rebaseline all runs on real data

Owner instruction, 15 November 2026. The
[global input requirement](../docs/baseline.md#global-requirement-real-data-and-production-tokenizer)
supersedes the old test-tokenizer fixture across all campaigns. The requested
four-variant cumulative rerun and three affected component comparisons are
complete; historical per-feature experiments retain their original input scope.

- [x] Locate the named production 100B tokenizer and a diverse real corpus;
  record their identities and packing rules.
- [x] Align canonical configuration, active runners and preparation; verify
  actual loaded token IDs, diversity and embedding-owner distributions.
- [x] Measure baseline, four fixes, FlashQLA and the three new fixes cumulatively,
  with matched new-data component timings and rank-0/1 diagnostics.
- [x] Preserve historical evidence and reconcile current speed and memory
  conclusions in the [campaign report](../runs/production-data-abcd/report.md).

Future measurements must keep the corrected input contract. Reuse accepted
correctness/lifecycle evidence and do not reopen the accepted MTP/FP8 A/A work.
This completed entry launches no further experiment.

## Completed assignment: production-data cumulative A/B/C/D rerun

Completed 15 November 2026.

Authorized by the 15 November 2026 goal attachment. Correct the shared input
baseline using the verified production 100B tokenizer and existing real diverse
corpus, then refresh the three component timings and run exactly the cumulative
A/B/C/D comparison with matching bounded diagnostics. Preserve the completed
implementation, workload, contributor edits and historical evidence. Use only
the existing owner-confirmed eight-H100 allocation; no allocation, installation,
download, commit or push. Do not reopen accepted MTP/FP8 A/A work.

The [single campaign report](../runs/production-data-abcd/report.md#scope-and-finite-acceptance)
owns the finite checklist, variant matrix, commands, results and remaining gates.
Real-data input acceptance is complete: `data-002` is prepared, the native
CPU/Gloo loader passes at depths 0 and 32 across all eight ranks, and the actual
base-runner readiness check passes. The [preparation receipt](../runs/production-data-abcd/data-002.json),
[loader receipt](../runs/production-data-abcd/data-002-loader-summary.json) and
[readiness receipt](../runs/production-data-abcd/input-readiness.json) hold the
details. `data-001` remains excluded as a failed partial. This does not establish
GPU behavior. The input gate and six component refresh processes passed, and all
four serial A/B/C/D timing runs are complete. Their fixed-order ratios are
provisional; the report records the timer, activation, memory and cadence scope.
All eight timing/diagnostic rows completed without failure, source/config
identities matched, and the GPU lease is closed. MCP analysis is complete, with
parser and coverage limits recorded in the report. No default recipe or feature
setting was promoted. The owner subsequently authorized consolidating the local
and server copies, keeping four original rank-0 traces in ignored `artifacts/`,
and committing/pushing the completed implementation, documentation and compact
evidence. This publication request authorizes no additional GPU run. The
completed assignments below remain historical.

<a id="active-assignment-ffn-overlap-and-row-owner-embeddings"></a>
## Completed assignment: FFN overlap and row-owner embeddings

Completed 14 November 2026.

Authorized by the 14 November 2026 goal attachment. Starting checkout is clean
`main` at `2ada87125e433e1f76c4cb2bfad244b641a78a93`. Implementation is limited
to this task's `gift/` and task-owned tests, runners and evidence. The current
[final baseline](../docs/baseline.md#current-final-task-baseline) remains the
control, including all four accepted improvements and FlashQLA `external_gate`.
Independent controls default off. The initial native protocol is 10 warmup plus
10 measured complete updates with accumulation 32; LR schedule stays 65000/2000.
The owner subsequently limited remaining numerical verification to embeddings
and accepted the investigated MTP/FP8 A/A variability. Further MTP/layer-0 A/A
investigations are stopped and their results do not block this campaign.
The owner confirmed the current eight-H100 allocation and authorized the goal's
specified GPU work after the initial read-only reconciliation; historical
campaigns grant no new permission. No allocation, installation,
shared-environment change, original-checkout write or publication is authorized.

Finite acceptance, in order where dependencies apply:

- [x] Reconcile current source, recorded runtime and existing research once;
  inspect the current `external_gate` trace with MCP Perfetto and record the
  integration/precision boundaries, collective ordering and expected costs.
- [x] Integrate separate opt-in MTP and `layers.0` FFN groups, each containing
  w1/w2/w3 together with callable, compile and GFSDP boundaries; preserve SAC,
  FQNs, input gradients, precision and state semantics. The owner's subsequent
  buffer requirement supersedes private parent contexts: use aligned contiguous
  non-overlapping regions of existing weight/gradient storage, preserve the
  alternating weight slots and original allocation budget, and track each
  region's ownership/completion. Add no GPU storage, pool growth or duplicate
  staging tensors. Component receipts show unchanged unique pool bytes and
  measured peaks; pending-reader tests and the all-rank component trace cover
  their stated scopes. The combined native initialized pool/peak and allocation
  lifetime pass in B; see the [native memory receipt](../runs/ffn-row-owner/native-memory-results.md).
- [x] Integrate complete row-owner lookup and backward with requester/owner
  deduplication, native source-local BF16 rounding, FP32 owner SUM and dense
  accumulated gradients; remove the embedding from all GFSDP wrappers and
  integrate clipping, dense AdamW, EMA and checkpoint continuation. The owner's
  subsequent memory requirement supersedes independent embedding intermediates:
  borrow existing GFSDP storage without pool growth or a separate workspace,
  protect reuse with events, retain backward routing state safely and keep one
  persistent FP32 owner gradient across first/later/reset accumulation. The
  full-shape fixture includes native coalescing scratch and routing metadata;
  its matched-cache peak is below baseline. The full native initialized pool
  and common-baseline memory cap also pass in the integrated native receipt.
- [x] Close embedding-specific output/gradient, invalid/empty/skew traffic,
  trunk+MTP derivative, accumulation-32, optimizer, clipping, EMA and state
  checks; retain failed/skipped checks. Component outputs/gradients and the
  listed traffic/state cases pass; the all-three native update also passes with
  compiled/SAC ordering and the measured allocation lifetime.
- [x] Measure the actual eight-H100/NCCL embedding stages and complete paths,
  plus affected FFN blocks/schedules, with per-rank latency, traffic and memory;
  publish the component result table before native timing. The six-case owner
  table and both FFN component tables are published; these isolated
  measurements do not establish combined training performance.
- [x] Complete one all-three-change native optimizer update with all 32
  microbatches on eight ranks under the accepted four fixes plus FlashQLA
  `external_gate`, SAC and model/loss compilation. Record initialized per-dtype
  pool capacities and actual full-model peak; source-derived pool arithmetic is
  not a substitute. The [native memory receipt](../runs/ffn-row-owner/native-memory-results.md)
  records the passing eight-rank update, pool and peak comparison.
- [x] Run matched native A/B 10+10 measurements with accepted baseline A and
  exactly all three new changes in B; report full series, physical/valid TPS,
  elapsed time, memory and cadence uncertainty. The [campaign report](../runs/ffn-row-owner/report.md#final-verdict)
  links the primary performance result and its JSON, with canonical A/B
  summaries, process metadata and per-rank full-step series; the bounded native
  trace confirms activation. One sequential pair measured a provisional
  +1.27% fixed-window physical-TPS difference with roughly 3% within-window
  cadence drift, so no general or causal gain is claimed.
- [x] Use bounded diagnostics (at most three complete microbatches per traced
  rank) and MCP Perfetto to establish early reductions and removal of embedding
  table collectives; reconcile final source/costs and give per-change and
  combined verdicts with explicit remaining production gates. The
  [native trace report](../runs/ffn-row-owner/native-trace-results.md) confirms
  the mechanism on 12 FFN rank/microbatch cases, the once-per-microbatch routed
  owner path, and absence of embedding-table GFSDP collectives. It retains the
  unchanged `lm_head` collectives and event-wait causality limits; coverage is
  three of 32 microbatches on ranks 0/1 and excludes optimizer device work.

The single [campaign evidence area](../runs/ffn-row-owner/report.md) records
source identities, commands, design decisions, measurements, failures and
remaining limits. All finite assignment items are complete. No defaults are
promoted and no commit or publication follows this assignment.

## Completed assignment: FlashQLA gate variants

Completed 14 November 2026. The block comparison (18 reports), all three native
eight-rank measurements, two new rank-0/1 trace captures and their MCP analyses
passed their recorded checks. The detailed [execution plan](../runs/flash-qla/plan.md),
[integrated report](../runs/flash-qla/report.md), and linked block, native, and
trace records retain the source pins, protocols, runtime costs, limits and failed
attempts. The single-order native measurements still have roughly 10.4% cadence
drift and do not establish general or causal speedup.

The owner subsequently selected `external_gate` as the explicit GDN backend in
the [current final task baseline](../docs/baseline.md#current-final-task-baseline),
on top of the combined-four foundation and already accepted V2 prefetch. The
selection is a task-recipe decision; it does not change generic backend defaults
or establish production validation. Convergence, checkpoint/resume, EMA updates,
CP warmup, and PP4/EP32 behavior remain open.

The user's subsequent instruction authorizes consolidating all three copies and
committing and pushing the validated implementation, documentation, and compact
evidence. No new training experiment, package installation, allocation, or
unvalidated implementation change is authorized.

## Active assignment: combined four improvements

Authorized 14 November 2026 and complete. The baseline/candidate pair and
combined diagnostic used task source HEAD
`6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507`, logical epoch 102,400, seed 2048,
and the unchanged [crop workload](../docs/baseline.md). The candidate enabled
prefetch 32, BF16 expert-weight reuse, explicit routing metadata, and stable
cap-zero CE. GPU input preparation fusion and Linear+CE were excluded. No new
allocation or installation was used.

Finite acceptance:

- [x] Reconcile existing implementations and numerical/lifecycle evidence once;
  integrate only necessary activation plumbing and affected checks.
- [x] Resolve the independent integration/runtime-cost review and scoped lint.
- [x] Verify the existing eight-H100 route, actual child arguments, resolved
  configuration, unchanged model contract, and compiler-cache policy.
- [x] Complete one baseline and one combined candidate, each 140 updates with
  warmup 1–40 and measurement 41–140; timing counters/profiling off and no epoch
  rollover.
- [x] Complete one diagnostic: 21 updates, warmup 20, update-21 traces on
  ranks 0/1 capped at three microbatches; all ranks completed 32 microbatches
  and the optimizer.
- [x] Verify combined queue, BF16, routing, CE and FP8 trace observables; record
  complete import coverage and unavailable exact event-wait attribution.
- [x] Produce the [report](../runs/combined-four/report.md), [compact results](../runs/combined-four/results.json),
  exact protocol/source identities, full per-rank cadence, lifecycle receipts,
  trace analysis, and retained failure record.

The [report](../runs/combined-four/report.md) owns the provisional fixed-window
result and limits. It is one sequential A/B with continuing cadence settling;
there is no A/A estimate or repeat. It establishes neither steady-state/general
speedup nor convergence, exact continuation, or production-scale behavior. The
first detached launcher and later incomplete `baseline-002` MDS lock failure
remain excluded. All actual run groups exited, source bindings were restored,
and the four-feature trace gate passed within its explicit timing/attribution
limits. No further run, repeat, individual-feature comparison, or optimization
campaign is authorized. The owner handles final review and publication under
the existing user authorization.

## Active assignment: BF16 expert-weight reuse

Completed, 14 November 2026. The anchor is retained for existing links. The
[report](../runs/bf16-expert-cache/report.md) owns the closed results and limits;
no additional run is outstanding.

The 14 November 2026 goal authorizes this task's `gift/` implementation,
focused checks and two candidate launches on the existing eight-GPU allocation.
Starting source is clean `main` at `657e77a4f8ddb116b75925e5671b00642a0e4d32`.
Preserve accepted V2 (prefetch 32, logical epoch 51,200), all crop settings,
FP32 masters/gradient accumulation, FP8, compile, SAC, EMA and MTP. All four
fusion candidates stay disabled. No original-checkout writes, installations,
new allocations, publication or fresh V2 controls are authorized.

The subsequent owner instruction authorizes committing and pushing the completed
BF16 experiment to `origin/main`, updating the local checkout, and downloading
the two candidate traces. It authorizes no additional training or experiments.

Finite acceptance checklist:

- [x] Reconcile existing source, V2 controls/traces and native execution route
  once; inspect the pinned Megatron lifecycle reference and the actual GFSDP
  optimizer, load, EMA and storage-replacement consumers before editing.
  See [design](../runs/bf16-expert-cache/design.md) and
  [execution readiness](../runs/bf16-expert-cache/execution-readiness.md).
- [x] Integrate opt-in reuse of persistent BF16 expert weights per weight
  generation and direct consumption for eligible size-one expert shard groups;
  preserve fallback, parameter identity, gradients, stream ordering and buffer
  ownership without another full BF16 allocation or new host synchronization.
- [x] Verify deterministic actual-GFSDP accumulation across two optimizer
  updates, recomputation, two groups sharing the original working-buffer pool,
  changed-weight loading, EMA swap/restoration, activation and fallback. Compare
  losses, input/weight gradients, FP32 masters and fresh BF16 casts.
  The [focused evidence](../runs/bf16-expert-cache/focused-checks.md) retains
  all failed harness attempts and the passing default-off, extension and
  real-EP main008 checks, including a nondefault writer/default consumer.
- [x] Pass scoped lint and resolve one independent integration/runtime-cost
  review, verifying material fixes.
  All findings in the [review](../runs/bf16-expert-cache/review.md) are resolved.
- [x] Verify parsed arguments and actual child commands; run one unprofiled
  candidate with `--steps 60 --warmup 20 --window 40` (updates 21–60) and one
  diagnostic candidate with `--steps 21 --warmup 20 --window 0
  --profile-update 21 --profile-microbatches 3` (ranks 0/1). Keep accumulation 32,
  LR horizon 65,000 and LR warmup 2,000. Counters are diagnostic-only and cover
  all microbatches and update boundaries.
- [x] Establish cast/copy removal and reader/refresh ordering from bounded
  traces and counters; report TPS, update times, peak allocated/reserved memory,
  exact saved-control windows and comparison limits in one result table.
  Record commands, source/config identities and artifact paths under
  `runs/bf16-expert-cache/`; update the existing owner documents.

An ambiguous performance result is inconclusive and ends this experiment.
Additional execution requires a concrete failed required check or invalid
measurement; repair and repeat only that check. No automatic repeats, new
controls, convergence campaign, full-model numerical capture or review campaign.
A positive speedup is not required; unresolved correctness must remain explicit.

## Active assignment: four independent fusion experiments

Completed, 13 November 2026. The anchor is retained for existing links; the
finite checklist below is closed and no further V2 run is outstanding.

The 13 November 2026 goal authorizes source changes in this task's `gift/`,
task-local runners/tests/evidence, and measurements on the existing allocation.
It supersedes historical cleanup-only stop statements for these four experiments.
At launch, no new allocation, installation, shared-environment change or
publication was authorized. Starting source is clean `main` at
`da6774a1c2191bfec0e1cb4aa1dc34019b3b7df8`.
The owner's subsequent pull request advanced `main` to `33b336a5`, adding the
[profile notes](../docs/profile-my-ideas.md) and ignore rules only; training source
is unchanged and all experiment edits are preserved.

Finite acceptance checklist:

- [x] Reconcile the accepted V2 evidence, source/diff, dependency/data identities
  and runtime route once; preserve prefetch 32, logical epoch 51,200 and the
  complete [workload contract](../docs/baseline.md).
- [x] Research, pin and implement four independent variants using explicit
  handwritten Triton/CUDA/CuTe kernels: GPU preparation,
  HybridEP routing metadata, fused linear-CE, and stable cap-zero CE. Record
  reuse decisions, compatibility gaps, costs and scoped regression checks.
- [x] For each variant, verify native activation and complete a matched
  unprofiled V2/candidate comparison: ten warmup plus ten measured full updates,
  input/valid-target TPS, update distributions and matched memory. Complete the
  decision-relevant input/routing/CE repeats; routing and CE reuse the completed
  input-repeat control because further V2 runs are prohibited. Record the FLCE
  activation-failure attempt separately from the corrected active comparison.
- [x] Capture matched rank-0/1 diagnostics after warmup, at most three complete
  microbatches while all ranks execute all 32; analyze equivalent target forward
  and affected backward regions with Perfetto, launch flows and stream overlap.
- [x] Complete required local correctness gates: exact input scalar/auxiliary
  state parity, routing metadata/gradient checks, the CE precision-aware
  synthetic replay, and
  scoped FLCE dH/dW/lifecycle checks. Reject active FLCE without a full-shape
  replay; it is not retained. Resolve independent integration/runtime-cost
  review findings and verify affected behavior.
- [x] Record commands, pins, resolved configuration, activation, logs, trace
  paths/hashes, failures and the four-way comparison table under
  [experiment evidence](../runs/four-fusion/report.md); update result owners
  and verify maintained document links/whitespace.

Keep each candidate independently reproducible and never stack the variants.
The owner prohibits any further V2 runs. The queued routing/CE control repeats
were cancelled before launch; input ran candidate-first against its fresh
control, and routing/CE reused the same completed input-repeat control. These
time/order limits and exact gaps are recorded in the receipts and report.
The owner's subsequent instruction replaces the compiler-expression approach to
input preparation/routing: reuse existing handwritten kernels first, adapt them
where needed, and document concrete gaps before writing missing kernels.
Wrapping tensor expressions in `torch.compile` does not satisfy this assignment.
Existing V2 model/loss compilation remains enabled.
No finite acceptance item remains open. No further V2 runs or training-source
changes are part of this completed comparison. The owner's subsequent
"commit and push all" request authorizes publishing the completed four-fusion
changes to `origin/main`; it authorizes no further compute.

Publication: the subsequent user request authorizes committing the completed V2
implementation, runners, tests and recorded results, then pushing to
`origin/main`. Earlier no-publication statements and receipt flags describe the
measurement assignments before this authorization. No new compute is requested.

## Accepted optimization: V2 CPU prefetch, depth 32

Owner decision, 13 November 2026: V2 is accepted for the current long-epoch
100B-crop workload based on the matched +2.1181% input TPS result and the
[MCP Perfetto comparison](../runs/v2-prefetch-baseline/report.md#mcp-perfetto-comparison-and-acceptance).
Input-to-first-F/B falls by 667.337 ms on rank 0; the full three-microbatch
device span remains 1294.519 / 1293.713 ms with no material sampled execution
regression. The accepted variant uses an explicit prefetch depth of 32; the
canonical depth-0 control and loader API default remain available unchanged.
The one-pair magnitude remains provisional; no general production-scale claim
or additional compute follows from this decision. The raw historical plans
below retain their original nonacceptance statements as history.

## Completed assignment: V2 long logical epoch comparison

The follow-up goal authorizes task-local benchmark configuration, runner and
targeted test changes, matched measurements and bounded diagnostics on the
existing eight-H100 allocation. Reuse the completed V2 correctness and review
evidence below. Preserve the short-epoch pair as a separate experiment.

Finite acceptance checklist:

- [x] Apply native Mosaic `dataloader.dataset.epoch_size: 51200` identically to
  both variants; verify 100 updates / 3200 microbatches per rank per logical
  epoch and the affected runner/config path. This repeats the existing fixture
  logically; it does not enlarge or diversify the physical corpus.
- [x] Verify current task processes, ownership, all eight GPUs and execution
  interfaces; run depth 0 then depth 32 serially for 60 optimizer updates each,
  with updates 1–20 warmup and exactly updates 21–60 measured. Preserve the
  workload, accumulation 32, LR horizon/warmup 65000/2000 and native timer bounds.
- [x] Verify matching resolved A/B configuration except queue depth and output
  paths, 83,886,080 physical measured input tokens, no epoch transition after
  initial entry through update 60, actual V2 queue use and finite native logging.
- [x] Capture four separate rank-0/1 traces of the input prefix, valid-token
  reduction and first three complete microbatches of update 21 after warmup 20;
  all eight ranks continue full accumulation outside capture. Verify coverage,
  identities, lossless compression and hashes; compare input/queue/overlap.
- [x] Record exact commands, identities, update series, TPS, memory, limitations
  and failed/skipped checks in the existing [report](../runs/v2-prefetch-baseline/report.md)
  and results with distinct run IDs; verify owned-process exit and GPU release.

Initial reconciliation: execution is already inside the authorized allocation
`lm-mpi-job-8dff1e19-674b-42ed-96a9-bcd042dfefe5`, with the task directory on
verified SR008 NFS and the established `/opt/torchtitan-venv/bin/python3.12`.
All eight H100s are idle with no compute applications; no prior training
controller/rank is live. Existing Perfetto analysis services are preserved.
The measurements required no installation or new allocation. Publication is
covered by the subsequent user authorization above.

Both new TPS runs passed all 60 updates on all eight ranks. The exact 40-update
windows and token totals passed; the populated native epoch receipts show no
rollover and V2 has a live producer with 32 ready batches at the timing boundary.
The provisional new pair is 138,009.86 versus 140,933.05 global physical input
tokens/s (+2.12% for V2). Four update-21 traces passed native coverage and lossless
gzip verification. Observed input-to-first-F/B spans are about 676–678 ms for A
and 8–9 ms for B; prior-producer overlap remains unobserved. All owned runner /
torchrun groups exited and all eight GPUs are idle. The
[report](../runs/v2-prefetch-baseline/report.md#long-logical-epoch-comparison--complete)
and [new pair receipt](../runs/v2-prefetch-baseline/pair-long-epoch-001.json)
own exact results, commands, identities, full series, memory and failed/skipped
checks. The subsequent owner decision above accepts V2 for this workload after
the MCP trace comparison. General speedup and repeatability are not established
by this single pair. No further run follows; publication is authorized.

## Completed assignment: V2 CPU prefetch above the monitoring-free baseline

The 13 November 2026 goal reopens only V2 and authorizes task-local source,
tests, runner/config repairs, the existing eight-H100 allocation, loader-only
checkpoints, matched TPS runs and bounded diagnostic traces. Other cancelled
candidates remain out of scope. Starting HEAD is `3b6b8928`; the earlier baseline
execution pin is `16b13c25`, and the preserved V2 implementation is `21c1891d`.
The existing loader-check helper is preserved. Publication is now covered by
the subsequent user request above.

Finite acceptance checklist:

- [x] Port only the required V2 changes onto current source; default depth 0,
  candidate depth 32, CPU FIFO, consumer-owned epoch entry and scalar committed
  checkpoint cursor; preserve accumulation, normalization and transfers.
- [x] Pass affected CPU/config/lifecycle checks and scoped Ruff; close real-MDS
  eight-owner Gloo/DCP gates with two workers in both persistence modes, skewed
  rollover, queue-ahead save and older-cursor restore. Resolve one independent
  final integration/runtime-cost review, reusing the prior review.
- [x] Verify current authorized allocation, all eight GPUs, runtime and paths;
  complete native forward/backward and optimizer updates with finite logging.
- [x] Measure fresh matched depth-0/depth-32 runs, each 10 warmup + 10 measured
  full updates of 32 microbatches, with the fixed baseline and monitoring off.
  Report input/valid-target TPS, update series/spread, paired memory and uncertainty;
  repeat only if the provisional result needs resolution.
- [x] Capture comparable baseline/candidate traces after warmup; all eight ranks
  train, only ranks 0/1 profile the input prefix, token reduction and at most three
  complete microbatches. Bound collection itself and verify parsed coverage.
- [x] Record source/config/data/runtime identities, commands, logs, artifact
  hashes and mechanism findings in the [single experiment report](../runs/v2-prefetch-baseline/report.md)
  and compact results; confirm owned-process exit and GPU release.

Before implementation, reuse the pinned V2 design/source review. Expected costs
are one producer/rank, bounded queue push/pop and scalar snapshots per batch,
Python/GIL contention, synchronous epoch priming and thread start/join per epoch.
Depth 32 holds 5 MiB/rank of ready tensors, excluding workers, active reads,
trainer-held batches and retained generator storage. Pause/close can wait on I/O;
enabled distributed restore adds two consumer Gloo barriers. No extra GPU work,
H2D transfers, per-batch collectives or model arithmetic is intended. Performance
and actual storage lifetime remain measured gates. The 56-local-batch fixture
prevents a depth-32 queue from hiding all consumer-owned epoch-boundary work.

Assignment complete. Both fresh TPS runs passed; depth 32 measured 2.08% lower
input TPS in this provisional pair. Four native traces passed actual bounded
collection, all-rank training, lossless compression and Perfetto MCP import
reconciliation. Queue activation is observed, with synchronous consumer epoch
entry still exposed. The [report](../runs/v2-prefetch-baseline/report.md) and
[results](../runs/v2-prefetch-baseline/results.json) own exact values, uncertainty,
commands, identities, failures and final checks. Owned training exited and all
eight GPUs were idle at release. No repeat or other optimization follows this
assignment; publication is now authorized above.

The corrected final Gloo gate passed both persistence modes; all diagnostic
groups stopped about 52 seconds before baseline measured update 11. The affected
CPU/config suite passed 245 checks, scoped Ruff and whitespace checks passed,
and independent source/runner review found no unresolved material defect. The
[source manifest](../runs/v2-prefetch-baseline/source-manifest.json) fixes the
18-file V2 source/test/golden scope and complete patch identity. The requested
pull found current `origin/main` at `3b6b8928`. Canonical experiment/model/
distributed source and `train_step` remain unchanged; default depth is zero.
Three-microbatch trace support is confined to the task runner. Further execution
requires a separately authorized assignment.

## Completed assignment: read-only update-11 candidate analysis

The [analysis](../runs/overnight-streaming-perf/report.md#update-11-candidate-analysis-with-perfetto-mcp)
completed workload/coverage verification, Perfetto import reconciliation,
candidate cost/wait measurements and evidence-based dispositions. SQL and compact
results are retained with the existing run. No new GPU run or training-source
change was made. Future comparisons remain separately authorized; streaming
prefetch is the first directly exposed opportunity, followed by linear/CE and
the specific MoE synchronization path. No candidate is accepted.

The sections below preserve the earlier assignments and original V2 plan.
Their V2 open-gate and future-execution statements are historical; the
completed results above own the current experiment status.

## Completed assignment: monitoring-free baseline and fresh traces

The [execution plan](overnight-streaming-perf.md) records completion: one valid
10-warmup/10-measured unprofiled baseline and exactly two verified, losslessly
compressed traces of optimizer update 11 on ranks 0/1. The [report](../runs/overnight-streaming-perf/report.md)
and [results](../runs/overnight-streaming-perf/results.json) own exact TPS,
configuration/source identities, artifact hashes and limits. Owned training
processes exited, GPUs were released and obsolete task caches/temporary copies
were removed. MDS and useful compact evidence are preserved. Results/documentation
are published through the requested normal `origin/main` commit/push. Stop here;
candidates and full-model numerical validation remain cancelled.

## Completed assignment: cleanup and protocol revision

The [cleanup plan](overnight-streaming-perf.md) contains the finite acceptance
checklist and ledger. The [launch prompt](overnight-agent-prompt.md) explicitly
stops after cleanup publication. The original overnight G0–G7 campaign is
cancelled/incomplete. No training, benchmarks, new traces, block comparisons or
candidate continuation may run during this assignment.

| Item | Status | Next action |
| --- | --- | --- |
| Full-model numerical machinery | Removed; scoped checks and independent review passed | Keep compact C1/C2 history; no replacement framework |
| Performance config and runner | Verified by 101 scoped CPU/config checks | Monitoring callbacks/hooks disabled; ordinary 2 chunks; 10 + 10 optimizer updates; schedule 65,000/2,000 and accumulation 32 preserved |
| Owned jobs/heavy numerical payloads | Complete; no owned numerical processes or reference blobs remain | Retain [cleanup receipt](../runs/overnight-streaming-perf/cancellation-cleanup.json) and compact evidence |
| Revised documentation | Complete; link/fragment, JSON and whitespace checks passed | Preserve the revised policy and historical evidence |
| Publication | Complete: `aa45e3cd` pushed to `origin/main` | Stop; do not continue the campaign |

## First future experiment: V2 input prefetch

The [baseline trace inspection](../runs/overnight-streaming-perf/report.md#baseline-input-preparation-analysis)
establishes exposed input preparation and a real opportunity to overlap it with
previous GPU work. Start with preserved **V2**, revision
`21c1891dda531a1222e6c350802f2765758794f9`, setting
`num_prefetch_batches: 32` in the candidate override; retain default 0 elsewhere.
This selects a ready queue for one full 32-microbatch fetch burst, not a proven
optimal depth. Follow the preserved [queue and checkpoint contract](../runs/overnight-streaming-perf/reference-coverage.md#queue-depth-and-what-the-trace-can-establish)
and [open V2 acceptance gates](../runs/overnight-streaming-perf/v2-prefetch-review.md).
Epoch entry remains synchronous; do not promise removal of the entire input gap.

Keep the [baseline configuration](../docs/baseline.md) and original TPS procedure:
10 warmup plus 10 measured full optimizer updates, with 32 microbatches per update.
Diagnostic traces capture at most 3 complete microbatches per rank, as specified
in [validation](validation.md#trace-capture-window). Verify activation, exposed
waits and queue memory in the matched comparison; no V2 gain is accepted yet.

## Preserved future questions — no execution authorization

The validated MDS fixture, historical HF/MDS smokes, monitoring-inclusive G2
controls, G3 Perfetto analysis and candidate sources remain useful evidence.
No candidate is accepted. V2 source is preserved at `21c1891d`; actual Gloo/DCP
and performance gates were not executed/completed. Future authorized work starts
with a fresh short **monitoring-off** baseline/candidate comparison, then local
numerical assessment of a candidate worth retaining. Historical monitoring-on TPS
cannot serve as its baseline. Single-pair effects remain provisional; repeat only
when the decision needs more evidence. CUDA Graphs remain off.

For a future approved experiment, use one report and compact machine-readable
result under `runs/<experiment>/`: question/decision, fixed source/config/data,
one candidate change, runtime/topology, activation evidence, optimizer-update
warmup/window, timer/token/memory scopes, step series, provisional/repeated
outcome and local numerical findings. Record failed/skipped checks and remaining
uncertainty. Planning an item does not authorize compute or source changes.
