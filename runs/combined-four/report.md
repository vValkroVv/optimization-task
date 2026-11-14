# Combined four improvements

Status: baseline, candidate, and diagnostic passed on eight H100 ranks. The
update-21 trace confirms all four mechanisms in the combined path, within the
observability limits in [trace analysis](trace-analysis.md). Exact commands and
receipts are in [protocol.json](protocol.json).

Both timing runs used task source HEAD `6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507`,
logical epoch 102,400, seed 2048, microbatch 2, sequence length 4096, global
batch 512, accumulation 32, and the agreed FP8, compile, SAC, EMA, MTP and
optimizer settings. Each completed 140 updates: warmup 1–40, measured 41–140
(209,715,200 physical and 209,664,000 supervised global tokens). The candidate
changes prefetch depth (0→32), BF16 expert-weight reuse (false→true), and the two
frozen routing/CE activations. Compiler-cache policy also differs, as described
below. GPU input preparation and Linear+CE were
excluded. Candidate source patches have since been reversed; canonical defaults
remain unchanged.

## Measured pair

| Metric | Baseline `combined-four-baseline-003` | Candidate `combined-four-candidate` | Candidate change |
| --- | ---: | ---: | ---: |
| Global physical input tokens/s | 149,386.56 | 156,094.27 | +4.490% |
| Global supervised tokens/s | 149,350.08 | 156,056.16 | +4.490% |
| Eight-rank elapsed window | 1,403.843 s | 1,343.516 s | −4.297% |
| Maximum allocated memory | 69.989 GiB | 68.188 GiB | −1.802 GiB |
| Maximum reserved memory | 72.057 GiB | 69.879 GiB | −2.178 GiB |
| Allocator retry / OOM deltas | 0 / 0 on all ranks | 0 / 0 on all ranks | unchanged |

The elapsed envelope is `max(rank end monotonic) − min(rank start monotonic)`
with CUDA synchronization at window boundaries. It includes loading/waits/transfers,
training and optimizer, ordinary logs, periodic GC, and adapter accounting;
startup/warmup, validation, checkpoints, detailed monitoring, profiling and
memory snapshots are excluded. Cadence is host timing, not a GPU-only timer;
component durations cannot be separated.

This single sequential A/B gives a provisional fixed-window difference, not a
steady-state or general speedup. Rank-0 mean host cadence continued to fall:
15.0430→13.4995 s from the first to last 20 measured updates for baseline and
14.3415→12.9129 s for candidate. The equal-work difference narrows from +4.891%
to +4.543%; no cause for the continuing settling is established. This was one
sequential pair without an A/A estimate, repeat, convergence check, or production
validation.

The baseline used fresh task-owned caches. Candidate and diagnostic passed
`--compiler-cache-root /workspace-SR008.nfs2/vkropotin/optimization-task/runs/overnight-streaming-perf/artifacts/raw/combined-four-baseline-003/cache`
(the absolute task path); Triton group records contain absolute child paths, so
all eight rank caches were reused in place. Startup showed 39 DeepGEMM kernels
planned, 39 cached and zero compiled. Timing runs kept warmup 40; diagnostic kept
20. The observer was disabled, so this does not prove the measured window was
compilation-free.

## Diagnostic and verification

`combined-four-diagnostic` passed 21 updates on all ranks, with all 32
microbatches and the optimizer. Update-21 F/B traces on ranks 0/1 cover the first
three microbatches; separate queue observations cover all input collection before
F/B on those ranks (32 pops each, minimum four ready, 0.370/0.402 ms host spans).
Per eligible BF16 group on every rank, each update adds refresh 1, hits 63,
direct uses 64, invalidation 1 and generation 1 (cumulative after update 21:
21/1323/1344/21/21); traced ranks show seven casts
and zero singleton copies, versus historical 21/42. All 78 FP8 quantizations
remain. Routing has 39 kernels/rank (21 forward, 18 SAC recompute); compiled
stable-CE ancestry joins 12 forward and 12 backward kernels/rank to two-chunk
main/MTP loss. Perfetto imported 9,445/9,445 events per trace, with no errors or
ambiguous links. NCCL counts (40 all-reduces, 69 all-gathers, 36 reduce-scatters)
and 42 capacity predicates are unchanged. Some local CE spans shrink while one
rank-0 head/loss guard grows 94.537→100.838 ms. The historical control differs
in source, job and epoch, so traces establish mechanisms only.

Exact device-wait attribution and a full-update critical path are unavailable:
the profiler omits event/stream handles and captures only three of 32
microbatches. Overlapping local durations are not additive. Trace analysis does
not establish memory savings; allocator peaks above come from the unprofiled
pair. Nonfatal finalizer `Errno 16` warnings for busy NFS `.nfs*` files followed
the diagnostic; exit was 0, the group stopped and scratch was removed.

Focused composition (3 passed, 4 deselected, 0 skipped), scoped Ruff, the
independent runtime review, and corrected CPU cache checks passed. NFS-lock
checks and the two excluded failures are detailed in [native execution](native-execution.md).
All final groups exited without timeout or orphan; source/config and scratch
receipts passed.

The [compact results](results.json), [full cadence series](update-series.json),
[source identities](source-identities.json), and [native execution record](native-execution.md)
retain hashes and artifacts. No repeat or additional run is authorized; no
candidate default was promoted.
