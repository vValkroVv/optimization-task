# Performance-only execution review

## Verdict and finite acceptance

The five-run design is valid for a bounded performance assessment, subject to
the timer labeling and phase-memory entry correction below. Run serially:
off-AA1, off-AA2, on-B1, off-A2, on-B2. AA1/AA2 estimates identical-control
variation; AA2/B1/A2/B2 supplies alternating off/on repeats. Reusing AA2 is
legitimate, but makes these summaries dependent: do not describe the noise pair
and A/B estimates as independent replications. Two pairs and twenty updates
provide descriptive evidence, not a precise significance or tail estimate.

Acceptance is limited to the intended workload/options, actual optimizer and
microbatch completion, useful-token accounting, finite native metrics, intact
logs, applicable replay/recapture evidence, matched integrated throughput and
labeled memory costs. No numerical tensor recording, checkpoint/RNG/full-state
restoration, detailed gradient/update comparison, sustained-run gate or new fault
framework belongs to this phase. This review adds no production-readiness claim.

## Fixed measurement window

Updates 21–40, closed at update 41's first batch-start before its fetch, are a
reasonable predeclared twenty-update window. A008 directly demonstrates settled
loss replay at 21–23 on all eight ranks, so the choice has empirical support.
That three-update proof does not establish the full later window. Preserve the
complete startup and step series. Actual compilation, capture, recapture or an
unexplained settling trend inside 21–40 makes stable interpretation inconclusive;
do not shorten to a favorable suffix or extend without concrete warmup evidence
and a recorded decision. A twenty-update window can miss uncommon events and
cannot establish long-run reliability. The prior native CUDA fault at update 48
remains unresolved; finishing at 41 does not demonstrate its repair.

Each run needs 41 native optimizer updates and 164 forward/backward microbatches
per rank, plus twenty correctly ordered measured token records. Under the
fixed-length contract these are 1,048,576 already-global useful tokens/update,
20,971,520 per window, without summing the duplicated global scalar across ranks.
Stop the chain on the first CUDA, logging or validity failure, retain every
attempt, and report measurement blocked rather than starting another fault study.

## Timer and memory findings

Inspected `CampaignTimingObserver.batch_start`, `_capture_boundary` and
`_finish_window` in `gift/torchtitan/experiments/cuda_graph_campaign/observer.py`:
the primary timer runs from the opening CPU clock after device drain/rank barrier
through the closing device drain. It includes native fetching, model/loss,
backward, accumulation, clipping, optimizer, scheduler and recurring native
metrics work inside those boundaries; final validity/time aggregation and JSON
persistence are outside. Throughput is window tokens divided by the maximum
completed-window wall time across ranks. Report start-to-first-update and
start-to-window setup separately with explicit launch/native-start definitions.

Per-update host boundary spans are submission/loop spans. Current-stream CUDA
events can omit independent side streams. Neither is complete-update latency or
pure device compute. Report their p50/spread with these names; completed-window
wall/20 is a completed mean, not a median. Publish complete-update p50 only if an
existing native timer proves that scope. With twenty observations, prefer max
and spread; any empirical p99 must be labeled descriptive and poorly resolved.
Reuse traced host/device/copy/communication mechanism scopes; do not infer wait
time by subtracting incomparable timers or add overlapping activity totals.

**Preparation correction:** the current `train_phase_memory.main` passes
`DEFAULT_PHASE_BOUNDARIES` with `expected_end_step=500`; `_document` therefore
marks a successful 41-update run incomplete. The task-owned entry must pass the
accepted installer an expected final *metrics* step consistent with the actual
cadence, while preserving native Trainer and all existing calls. At log frequency
10 that normally means step 40, not optimizer count 41. Verify the actual final
metrics producer; do not override invalid status after execution.

Reuse accepted phase-memory wrappers unchanged. Native reset after 20 followed
by samples/resets at 30 and 40 supports intervals 21–30 and 31–40; their maximum
is the corresponding allocator envelope. Verify actual origins and required
counter/status completeness on all ranks. Opening/closing allocator samples
are separate observations and must not be mistaken for reset-aligned peaks.
Device-level sampled maxima are lower bounds on true maxima; sampled free
headroom is an upper bound on worst-case free headroom. Join actual UUID/rank and
UTC brackets, report sample gaps, and do not claim capture-only or whole-process
exact device peaks. Adapter host bookkeeping is matched but unmeasured; it adds
no per-update device synchronization, allocator query, copy or collective.

## Frozen active-path checks

Before launch, the execution owner should verify the frozen imports/bytes and
resolved rank-zero configurations, not infer behavior from the new snapshot ID:

- Native `Trainer`; timing and graph-step callbacks only; CP/numerical/recorder
  callbacks absent. Preserve inactive stopped edits without completing them.
- Exact 3B MoE, HybridEP, FP8 recipe, FlexAttention, eighteen selective-AC and
  compiled blocks, compiled chunked loss, FSDP2/EP8, local8/global256/seq4096/
  accumulation4, fused AdamW, scheduler horizon, seed/data and native cadence.
- Both sides retain scan/backend and dense-GEMM graph-safety prerequisites,
  `mix_order_reduction=false`, partitioning true and all eighteen model graph
  exclusions. Only agreed compiler graph activation changes, including mask.
- Compare actual active compiler/model/loss/optimizer/dispatch, callbacks,
  phase-memory adapter and entry against Source009/006 as applicable. Record
  the new full manifest and explicit active equality; freeze throughout all runs.
- Preserve compiler receipts on all ranks, unprofiled configuration, process
  group/device mapping, complete rank logs, exit statuses and native counters.

## Reused mechanism, cost and provenance

[A008 mechanism review](../../../LANE-A/analysis/008/mechanism-review.md) proves
32 actual CE-forward and 32 CE-backward graph launches/update/rank, eight reused
IDs per phase, four mask replays, zero model graph launches and zero fresh
capture/instantiate at updates 21–23 on all ranks. Its SHA256 is
`0f2fb5ecaea7ddc33089dedb16a6c71ee7070cd34769bea63c16737feaa5d0be`.

[Copy review](../../../LANE-A/analysis/008-copy-cost/report.md), SHA256
`7c762bc03d9311d37f319e14a1b69352e186e60dfa19dfd41bb3a7ee3eb947d4`,
establishes 25 GiB BF16-logits staging plus 1 MiB labels per rank/update;
17.605–17.641 ms copy activity is not exposed wall-time tax or peak residency.
Persistent pools/input staging can increase memory; matched measurements decide.
Model, head GEMM, optimizer and communication remain outside graphs and inside
the integrated timer. No new copy/pool workaround is justified.

Source006 manifest SHA256 is
`3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603`;
[Source009 manifest](../../../SOURCE-SNAPSHOT/009/manifest.json) is
`cf95f726518062714913b8155dad878b73442fc5a0d68d993b7abdf6a55cf3f8`.
Pinned Torch is `2.15.0.dev20260822+cu130`, git
`8b353065ce260336c5fdfa74d754e8bd9b91b062`; CUDAGraphTrees SHA256 is
`7cab9ba4338305b10c792ffc7389898b46623a8e87052928158581b33e413e76`.
Reuse the [accepted memory review](../../../PHASE-MEMORY-REVIEW/001/REVIEW.md)
and retain historical lane dispositions as superseded where their numerical/
continuation/500-update gates exceed this explicitly narrowed phase.

Reviewed relevant perf-experiments, code-rules and titan-train-job skills.
No source edit, GPU query/run, install, process operation or new trace analysis
was performed. Only this evidence report was written; it has no training runtime
impact. Remaining decisions are preparation correction and actual matched results.

## Replay claim clarification

Reuse A008's all-rank proof with active source/runtime/options equality; leave
replay and recapture during the unprofiled 21–40 window explicitly unobserved.
Compiler-entry receipts record options, not actual launches. Read-only inspection
of the installed CUDAGraphTrees file confirms its hash above. Its existing
`cudagraph_recorded_non_static_inputs` counter counts input slots when nodes are
constructed, not replay or every capture; manager `num_rerecord` counts unexpected
rerecord paths, not every new capture. Neither supplies existing per-window
replay/recapture evidence. No new instrumentation or trace run is required.

Accurate result wording: “The candidate previously demonstrated persistent actual
CE forward/backward replay with zero fresh capture at updates 21–23 on all eight
ranks. These matched unprofiled runs measure the integrated effect with the same
active path; replay and recapture over updates 21–40 were not directly observed.”
