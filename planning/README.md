# Work plan

[Current state](../docs/status.md) owns results and unknowns;
[candidates](../docs/candidates.md) owns mechanisms/source pointers.
Follow [workflow](workflow.md) and [validation](validation.md).

Publication: the subsequent user request authorizes committing the completed V2
implementation, runners, tests and recorded results, then pushing to
`origin/main`. Earlier no-publication statements and receipt flags describe the
measurement assignments before this authorization. No new compute is requested.

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
checks. No general speedup or adoption is established by this single pair;
default depth remains zero. No further run follows; publication is authorized.

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

## Completed assignment: monitoring-free baseline and fresh traces

The [execution plan](overnight-streaming-perf.md) records completion: one valid
10-warmup/10-measured unprofiled baseline and exactly two verified, losslessly
compressed traces of optimizer update 11 on ranks 0/1. The [report](../runs/overnight-streaming-perf/report.md)
and [results](../runs/overnight-streaming-perf/results.json) own exact TPS,
configuration/source identities, artifact hashes and limits. Owned training
processes exited, GPUs were released and obsolete task caches/temporary copies
were removed. MDS and useful compact evidence are preserved. Results/documentation
are published through the requested normal `origin/main` commit/push. Stop here;
candidates other than the separately reopened V2 assignment above and full-model
numerical validation remain cancelled.

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

## Preserved future questions — no execution authorization

The validated MDS fixture, historical HF/MDS smokes, monitoring-inclusive G2
controls, G3 Perfetto analysis and candidate sources remain useful evidence.
No general candidate performance gain is accepted. V2 source was preserved at
`21c1891d`; its current gates are owned by the completed V2 assignment above. Future authorized work starts
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
