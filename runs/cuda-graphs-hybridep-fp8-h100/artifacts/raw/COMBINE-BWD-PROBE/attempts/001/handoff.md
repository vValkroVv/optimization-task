# COMBINE-BWD-PROBE / attempts/001

## Result

The single native60 run launched at 2026-11-12 03:58:32.684068 UTC under controller PID 591470 and exited 1 at 04:00:54.625620 UTC. It reached update 21, not update 60. All eight ranks completed 20 optimizer updates and 80 forward/backward microbatches before the active window. Each rank then completed microbatch 0 of update 21, for 81 successful microbatches total.

The active callback records show 24 completed checkpoints per rank, ordinals 1361–1384, followed by an unmatched enter at ordinal 1385 in microbatch 1. Rank 6 recorded CUDA error 715 at event.synchronize in _call_adapter, then logged fwd_bwd_error at 04:00:33.744 UTC and step_error at 04:00:33.745 UTC. All ranks entered ordinal 1385; rank 6 alone recorded event_wait_error before torchrun terminated its peers. The full rank 6 stack is retained in the elastic error file linked by the extraction.

The callback records have no timestamps or wait-duration fields. The only nearby timed scope is rank 6 microbatch 1 forward/backward: start 04:00:33.168 UTC, error 04:00:33.744 UTC, reported span 575.82 ms. That is the whole forward/backward span, not the event wait. The exact 1384-to-1385 time gap and event wait duration are unavailable.

Updates 1–20 logged 68 selected calls per update, 17 per microbatch, while the observer was unarmed. Those are selection counts, not active host completion checkpoints. This diagnostic adds per-call synchronization and changes peer progress; it is not a performance measurement or repair. The run does not establish the originating kernel or that this shares a cause with earlier faults.

## Provenance and release

The one-shot binary monitor started at 04:01:02.740 UTC, after the run exited. The rank PIDs had already disappeared, so the external helper was not called on dead PIDs and no provenance-only rerun was made. Same-run quantize/GEMM hashes and comparison to A006 remain open. This is missing provenance, not a negative binary finding.

Exact release passed at 04:08:34 UTC: controller, runner, torchrun agent, sampler and all eight rank PIDs were absent; the same eight H100 UUIDs were idle with zero memory/utilization and no compute apps. The single post-release ECC/PAGE_RETIREMENT/ROW_REMAPPER query succeeded: numeric ECC and row-remapper counters were zero on all eight GPUs; retired-page fields were N/A. These counters do not clear transient hardware, driver or kernel causes. The raw sampled-memory CSV is retained; no throughput or memory-performance claim is made.

No earlier device-side printf/assert/timeout marker preceded 715. The supervisor log does contain host-side Quack cache-lock timeout fallback warnings and process-group timeout configuration, not a device timeout. The nvidia-smi stderr file is empty.

## Evidence

- [Verified post-run extraction](postrun-extraction-verified.json)
- [Binary provenance status](binary-provenance-status.json)
- [Complete process/GPU release check](gpu-release-check-complete.json)
- [Raw ECC/page-retirement/row-remapper query](hardware-error-counters.json)
- [Combined supervisor log](supervisor.log)
- [Rank callback JSONL files](native/combine_bwd_completion/)
- [Rank structured JSONL files](native/structured_logs/)
- [Sampled device memory](device-memory.csv)
- [Launch receipt](launch-invocation.json)

Artifact notes: two initial preflight files preserve topology-parser false negatives; preflight-final.json plus coordinator topology review passed. The first post-run extraction has an invalid trailing literal escape and is preserved; the verified extraction above supersedes it. The initial gpu-release-check.json omitted runner/torchrun PID rows; gpu-release-check-complete.json is the complete check. The late monitor summary misattributes the missed helper to the later step-21 error; binary-provenance-status.json records the monitor timing failure correctly.
