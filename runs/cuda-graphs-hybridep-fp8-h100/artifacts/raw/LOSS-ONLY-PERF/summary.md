# Loss-only unprofiled matrix status

## Outcome

The four-run matrix stopped at the first invalid attempt, C0a/off. Rank 3 failed during step 48 with a CUDA illegal-instruction error observed in the compiled loss backward path. The run completed 47 optimizer updates and 188 successful forward/backward microbatches per rank, then failed before update 48.

The required timing window (steps 21–70, 50 updates) is incomplete: only steps 21–47 completed, and native/timing_observer contains no per-rank records. No primary throughput, paired A/A noise, or graph-on/off comparison is reported.

| Attempt | Variant | Status |
|---|---|---|
| 001 | C0a / graphs off | Invalid; CUDA illegal instruction on rank 3 at step 48; exit 1 |
| 002 | C1a / graphs on | Prepared, not launched |
| 003 | C0b / graphs off | Prepared, not launched |
| 004 | C1b / graphs on | Prepared, not launched |

The protocol requires stopping at the first fault. There was no retry, and attempts 002–004 remain unlaunched.

## Evidence

- Counts, rank checks, config/options and fault status: attempts/001/attempt-result.json.
- Fault detail: attempts/001/failure-diagnosis.md.
- Structured event series: attempts/001/structured-step-events.csv; JSONL checks: attempts/001/structured-log-integrity.json.
- Sampled device-memory maxima: attempts/001/gpu-sampled-memory-summary.json. These are 500-ms sampled maxima only; exact phase allocator/native peaks remain unverified.
- Process/GPU release: attempts/001/gpu-release-check.json (pass).
- Raw traceback: attempts/001/supervisor.log, attempts/001/torchrun/*/attempt_0/3/error.json and rank 3 structured/stderr logs.
- Prepared protocol: preparation/001/protocol.json. Source snapshot manifest SHA-256: 3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603.

No numerical loss/gradient comparison is made. Earlier C1 storage evidence remains intact; this C0a run showed no ENOSPC/no-space message and its structured log sequence was intact.
