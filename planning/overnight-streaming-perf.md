# Monitoring-free baseline measurement and trace capture

Status: RUNNING. The 13 November 2026 user request authorizes this bounded
baseline measurement and fresh trace capture on the existing eight-H100
`sr008-gpu` allocation. Starting source is
`eadcc6ac2ca5e7a24389e03416327fa8712e44da`; the exact execution revision will be
recorded after the necessary timing-receipt repair below. The original candidate campaign remains
cancelled. The completed cleanup record below describes the previous assignment.

## Current scope and acceptance

1. Verify current source, owned processes, all eight GPUs, runtime and existing
   MDS paths. Preserve the pre-existing untracked V2 loader-check helper.
2. Run one unprofiled baseline with 10 warmup and 10 measured optimizer updates,
   preserving all workload settings and the 65,000/2,000 LR schedule. Detailed
   monitoring/hooks, validation, checkpoints, numerical capture, memory snapshots
   and CUDA Graphs remain off. Include normal input loading and transfers.
3. Use the actual update series and logs to assess settling. Extend warmup only
   if real compilation or continuing settling affects the window; no exhaustive
   cache audit or mandatory control matrix. Treat one run as provisional.
4. In a separate run, use the supported runner/profile schedule to capture one
   complete optimizer update after that observed warmup. All eight ranks train;
   only ranks 0 and 1 construct a profiler. Require one active update, 32
   microbatches and the optimizer, with shapes/stacks/profiler memory off.
5. Parse both traces, verify update identity and coverage, retain paths/hashes,
   resolved configurations and source/runtime/data identities. Compress losslessly
   with verified round trips and retain exactly one canonical copy per rank.
6. Record complete-window global/per-GPU input TPS, update median/range and
   defined memory observations in the existing report/results and current state.
   Historical monitoring-inclusive results remain historical and are not a
   candidate optimization comparison. No new bottleneck analysis is required.
7. After consumers stop, clean exact task-owned redundant traces, temporary
   files and obsolete caches. Verify process exit and released GPUs, then commit
   and push results/documentation using the existing identity. Stop there.

No candidate experiments, full-model numerical comparisons, new allocation or
unrelated implementation work is authorized. Use the reviewed runner and existing
runtime; record a concrete failure before any repair needed for this measurement.

## Execution ledger

| Item | Current evidence / next action |
| --- | --- |
| Initial checkout/process inspection | HEAD `eadcc6ac`; only the existing untracked V2 helper; no training process at initial inspection |
| Unprofiled measurement | Held before launch: the default observer-off receipt omitted wall timestamps required by the parser; `window_wall_receipt_fix` owns the minimal serialization repair and actual-output regression |
| Profile schedule | Source contract verified: for observed warmup W, capture update W+1 with one Kineto warmup update W; only ranks 0/1 construct profilers; all eight train |
| Results, cleanup and publication | Pending the two run receipts; coordinating owner integrates and publishes |

---

# Completed cleanup and revised protocol

Status: CLEANUP_COMPLETE. The original G0–G7 overnight optimization campaign is
cancelled and incomplete. This assignment changes tooling and instructions;
it does not resume experiments. All new communication and deliverables use English.

## Completed cleanup assignment and finite acceptance

Work in `/workspace-SR008.nfs2/vkropotin/optimization-task`. The last campaign
execution source was `e9048f65f6a00b87c97b1e7e2cfb31fa66f56414`.
The revised user objective supersedes the old campaign's long-window, monitoring
and full-model numerical prerequisites. Complete only these acceptance items:

1. Stop exact task-owned C1/C2/C50 processes before changing their files. Remove
   campaign `numerics.py`, `numerics_payload.py`, exclusive tests/configuration,
   and active runner/observer/parser integrations. Preserve unrelated training.
2. Disable detailed monitoring callbacks and hooks in the resolved performance
   configuration, using ordinary two-chunk loss. Retain lightweight native loss,
   step-time/TPS and error logging, EMA and training losses.
3. Accept 10 warmup plus 10 measured **optimizer updates**, preserving accumulation
   32 and LR horizon 65,000/LR warmup 2,000. Disable validation, checkpoint saves,
   memory snapshots, numerical capture and CUDA Graphs.
4. Record the future performance-first/local-block protocol below. Remove old
   active numerical and long-measurement prerequisites; do not build another
   numerical framework or run local block comparisons now.
5. After owned jobs stop, delete cancelled task-owned reference tensors,
   temporary checkpoints and unnecessary heavy dumps. Preserve MDS preparation,
   candidate source, compact historical evidence, Perfetto SQL/results and the
   canonical compressed trace pair. Record actual cleanup counts only from receipts.
6. Use scoped CPU/config checks to verify imports/parsing, no active numerical
   entry points, short windows, disabled monitoring and preserved training
   settings. Check affected document links. Record checks, cleanup status and
   unresolved issues in the existing [report](../runs/overnight-streaming-perf/report.md)
   and [results](../runs/overnight-streaming-perf/results.json).
7. The coordinating owner reviews the integrated changes, commits and pushes with
   the existing repository-local identity under the user's authorization, then stops.

No GPU training, benchmark, candidate experiment, trace collection or numerical
block execution is authorized by this cleanup assignment. Do not allocate jobs,
install packages or continue the former campaign.

### Completed cleanup ledger

| Item | State / evidence |
| --- | --- |
| Numerical source removal and runner/config verification | PASS: 101 scoped CPU/config checks; independent integration/runtime-cost review; see [results](../runs/overnight-streaming-perf/results.json) |
| Exact owned-job stop and heavy-payload cleanup | PASS: no owned numerical processes or reference blobs remain; 40,650 files / 381,915,112,855 logical bytes removed; [receipt](../runs/overnight-streaming-perf/cancellation-cleanup.json) |
| Revised instructions and historical reconciliation | PASS: this plan, [launch prompt](overnight-agent-prompt.md), [validation](validation.md), [current state](../docs/status.md); 156 local links/16 documents and results JSON checks pass |
| Commit/push | Cleanup commit `aa45e3cd8d0c8856d383e72a9ad5f5d60c0530ae` pushed to `origin/main`; this receipt update records completion |

All cleanup acceptance items passed and the cleanup commit was pushed. Stop here;
publication does not authorize training or reopen the original G0–G7 gates.

## Fixed workload for future authorized comparisons

Keep the validated 896-row full-length MDS fixture and its packing/partitioning.
The [baseline](../docs/baseline.md) owns all defaults: D4096, eight trunk blocks,
E128/H1536/top8/shared1, MTP1, B2/S4096/global512/accumulation 32, EP8/GFSDP2 and
PP/TP/CP1; dense+sparse FP8, CuTe, model/loss compile, fused AdamW, adaptive
routing, production SAC, EMA, capacity 50,000 and seed 2048. Training losses and
EMA are required behavior, not diagnostic monitoring.

Detailed monitoring callbacks/hooks are disabled, rather than delayed with a
larger interval. Ordinary loss uses two chunks. Lightweight loss/time/TPS/errors
remain. Validation, checkpoint saving, memory snapshots, numerical capture,
profiling and CUDA Graphs are off for unprofiled performance measurements.
A short stop does not compress the LR schedule: horizon 65,000/warmup 2,000 remain.

## Future short comparison protocol — not an instruction to run now

Start with a short matched baseline/candidate pair: 10 warmup optimizer updates
and 10 measured optimizer updates (updates 11–20), each with 32 microbatches.
Freeze source, initialization/data order, workload, topology, timer boundaries,
cache policy and the one candidate change; inspect actual resolved configuration
and candidate activation before interpreting results. Keep physical input and
valid-target TPS distinct, including global/per-GPU units, elapsed scope, step
series, median/spread and memory scope. Report excluded lifecycle costs.

Extend warmup only when actual compilation or continuing runtime settling affects
the measurement window. A new internal compiler-cache entry alone does not justify
extension or invalidate an otherwise useful window. No mandatory 110 warmup,
50-update window, exhaustive compiler-cache audit, large A/A or A/B/A/B matrix
precedes an initial candidate comparison. A single pair is provisional; repeat
promising or ambiguous effects as needed to distinguish them from noise and drift.
Use traces only for an unresolved mechanism question under separate authorization;
there is no mandatory new trace pair per candidate.

### Future command example — do not execute during this assignment

From the repository root, after separate authorization for a future measurement:

```bash
/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py \
  --run-id future-baseline-001 --steps 20 --warmup 10 --window 10
```

The current runner defaults are 20 total, 10 warmup and 10 measured optimizer
updates. There is no monitoring-interval flag: detailed callbacks/hooks are
removed from the resolved performance configuration. Compiler-cache observation
is optional and diagnostic, not a mandatory acceptance gate. This example has
not been executed and does not authorize a run during cleanup.

### Local numerical assessment follows performance

For a candidate worth retaining after its performance measurement, choose the
smallest meaningful affected block. Capture representative boundary inputs and
only weights/state needed to replay that block. Run original and changed blocks
on identical inputs and compare their outputs. If backward changes, replay the
affected backward block with matched activations and incoming gradients.

Interpret differences using accumulation precision, casts, rounding, reduction
order and fusion. The baseline is not assumed mathematically exact; use a
higher-precision local reference when needed to judge which implementation is
more accurate. Justify tolerances for the block/dtypes and observed behavior;
there is no universal tolerance or assumed bitwise equality. Keep unexplained
differences open. Do not collect whole-model gradients, parameters or AdamW state.
Do not implement or execute these checks during this cleanup assignment.

## Preserved history and cancelled gates

| Original gate | Historical disposition |
| --- | --- |
| G0 entry/source/MCP access | Completed evidence retained |
| G1 full-length MDS preparation/native smoke | Completed evidence retained; 10 updates/320 microbatches per rank |
| G2 monitoring-inclusive controls | Historical measurements retained; new monitoring-off baseline required for future comparisons |
| G3 two-rank update 111 capture/Perfetto analysis | Accepted post hoc after category-aware parser repair; SQL/results/compressed traces retained |
| G4 reference reconciliation | Useful source findings retained; full campaign coverage incomplete |
| G5 candidates | Cancelled/incomplete; no candidate accepted |
| G6 combined result | Not executed; no qualifying improvement |
| G7 original campaign handoff | Original campaign incomplete/cancelled; cleanup publication has its own finite acceptance above |

Historical monitoring-inclusive G2 median input TPS was 124454.81175967702 global
and 15556.851469959627 per GPU, over updates 111–160 including monitor 150. These
values are **not directly comparable** to future monitoring-off measurements.
The previous W60/cache-state rejection and W110 selection explain historical
choices only; they impose no new gate on the revised protocol.

C1 B2 passed and compact evidence is retained. C2 A reached a terminal export;
C2 B and C50 were cancelled. Full-model numerical validation was abandoned,
not completed or replaced with a new full-model gate. The prior failed C1 B
remains failed in history; its tuple/list metadata diagnosis and successful
retry are retained without requiring another run.

V2 source variant `21c1891dda531a1222e6c350802f2765758794f9`, branch
`vkropotin-v2-prefetch`, is preserved with default 0 and its
[source/CPU review](../runs/overnight-streaming-perf/v2-prefetch-review.md).
Actual eight-owner Gloo/DCP and native/performance acceptance were not executed
or completed. No candidate gain, convergence, full-model continuation or
production-scale result is established. See the [coverage record](../runs/overnight-streaming-perf/reference-coverage.md)
and [historical report](../runs/overnight-streaming-perf/report.md).
