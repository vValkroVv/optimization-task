# Compact historical CUDA Graph evidence

Read [the current conclusions](../../docs/cuda-graphs.md). This evidence concerns
the historical 3B workload, not the current crop. The [manifest](manifest.json)
pins retained bytes and their original paths at the archived task revision.

| Evidence | Purpose |
| --- | --- |
| [Dense replay](dense-replay-counts.json), [phase counts](dense-phase-counts.json) | A007 actual launches and recurring recording |
| [Loss replay](loss-replay-counts.json), [phase counts](loss-phase-counts.json) | A008 persistent loss forward/backward replay on all ranks |
| [Loss copies](loss-copy-cost.json) | Input-staging traffic and measured copy activity |
| [Trace assessment](trace-assessment.md) | Later graph-off critical-path analysis, timing definitions and limits |
| [Trace acceptance](trace-acceptance.json) | Capture acceptance and source/trace identities |
| [Runtime excerpts](loss-runtime-source.txt) | Inspected graph input-copy/pool implementation |

JSON fields retain historical paths and scope; they are evidence, not current
launch commands. Original analysis scripts, complete reports, configs, source
snapshots and failed attempts are recoverable using
[the archive instructions](../../references/README.md). Raw profiler payloads
remain outside Git. No new measurement was made during this consolidation.
