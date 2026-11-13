# Cleanup-only agent launch prompt

Status: completed; reuse the verified receipts in `planning/overnight-streaming-perf.md`.
Do not restart the cancelled campaign or run the future measurement examples.

Work in `/workspace-SR008.nfs2/vkropotin/optimization-task` and complete only the
cleanup and protocol revision in [the active plan](overnight-streaming-perf.md).
Use English. Read [current state](../docs/status.md), [baseline](../docs/baseline.md),
[validation](validation.md) and applicable source rules before relevant work.
The original G0–G7 optimization campaign is cancelled and incomplete.

Stop exact task-owned C1/C2/C50 processes before changing their files. Remove
campaign full-model numerical modules, exclusive tests/config and active
runner/observer/parser integrations. Remove cancelled owned reference tensors,
temporary checkpoints and unnecessary heavy dumps after jobs stop. Preserve MDS,
candidate sources, compact historical results, Perfetto SQL/results and the
canonical compressed trace pair. Preserve other contributors' changes.

Disable detailed monitoring callbacks and hooks in the performance configuration;
a longer monitoring interval is insufficient. Retain native lightweight loss,
step-time/TPS and errors; use ordinary two-chunk loss. Preserve model/data, FP8,
compile, CuTe, SAC, MTP, EMA, optimizer, routing, batch/accumulation 32 and schedule 65,000/2,000.
Keep validation, checkpoint saves, memory snapshots, numerical capture and CUDA
Graphs off. Support 10 warmup plus 10 measured optimizer updates without compressing
LR scheduling. Verify imports/parsing and resolved configuration with scoped CPU
checks, then reconcile docs, historical status and affected links.

Record future policy without executing it: begin with a short matched baseline/
candidate pair; a single pair is provisional, and promising or ambiguous effects
may need repeats. Extend warmup only for actual compilation/runtime settling in
the measured window, not a cache entry alone. No mandatory 110/50 or large A/A/
A/B/A/B matrix. Measure performance first; for a candidate worth retaining, then
replay the smallest affected block with matched inputs and needed weights/state.
Compare outputs and affected backward with matched incoming gradients when it
changes; analyze precision, casts, rounding, reductions and fusion, using a
higher-precision local reference when needed. Do not assume universal tolerances,
bitwise equality or that baseline output is exact. Do not collect full-model
parameter/gradient/Adam state or implement a replacement numerical framework.

Do not launch training, benchmarks, traces, candidate experiments or block checks.
The monitoring-inclusive historical TPS is not comparable to monitoring-off.
Record actual process/cleanup and check receipts, unresolved issues and the
resulting commit. Commit and push under the user's existing authorization and
repository-local identity after integrated review. Stop after cleanup publication;
do not continue the campaign or execute the future measurement example.
