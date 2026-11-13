# Work plan

[Current state](../docs/status.md) owns results and unknowns;
[candidates](../docs/candidates.md) owns mechanisms/source pointers.
Follow [workflow](workflow.md) and [validation](validation.md).

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
