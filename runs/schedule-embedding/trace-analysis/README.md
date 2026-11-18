# D0 control operation matching

This bounded deliverable establishes the existing current-D control and reusable
matching queries. It does not measure candidate behavior or speed.

Acceptance completed: preserve original traces; verify only permitted derivative
changes; reconcile MCP event/flow counts and import errors; attribute scheduling
and embedding operations semantically; retain interval-union metrics per rank
and microbatch with exact payloads; document candidate inputs and limitations.

## Evidence and reproduction

- [identities.json](identities.json) pins original and new derivative hashes.
- [imports.json](imports.json) pins original import receipts and new MCP results.
  Original imports reported 40/39 overlapping-complete-event drops on ranks 0/1.
- New imports each retain 8,230 kernels, 202 copies, 885 memsets and 330 GPU
  annotations. All 9,317 device flow endpoints match runtime launch correlation;
  no nonzero import error/drop statistic remains.
- [prepare.py](prepare.py) adapts the previous campaign's greedy lane coloring.
  It changes only overlapping GPU display tids and matching finish-flow tids,
  adding `args.stream` where absent from GPU events using the original GPU tid.
  All original fields, events, timestamps and durations are verified unchanged.
  Synthetic lanes are **not CUDA streams**. The original stream stays in args.
- [queries.json](queries.json) reuses the actual previous MCP flow-query method.
  `device` joins GPU events through `flow` to runtime/driver launch and CPU
  ancestors; GPU annotations remain available independently.
- [analyze.py](analyze.py) reads saved actual MCP results. `control.json` contains
  compact results; the ignored `artifacts/raw/trace-analysis/control-detail.json`
  contains operation-level rows, exact names, launch ancestors and overlap rows.
  The compact result pins that detail file. This is offline JSON processing,
  without training-source or runtime impact.

Executed commands: `python runs/schedule-embedding/trace-analysis/prepare.py`,
MCP `load_trace` on both derivatives, MCP `execute_sql` using census/errors/flows/
scopes/owner_host_waits/gpu_annotations and device pages at offsets 0/4000/8000,
then `python runs/schedule-embedding/trace-analysis/analyze.py`.
Saved MCP responses are in the ignored raw directory.

## Control schedule

Times below are ms relative to each original trace's earliest timestamp.
Microbatches are ordered by the CPU `task.profile_update.21.microbatch.rank*`
annotations; device ownership follows launch time in that CPU scope, including
device tails beyond the host scope. Collective identity requires the exact
GFSDP module annotation, runtime launch flow, NCCL family, count/dtype/group
metadata and sequence. IDs are local joins only, never cross-trace identities.

| Rank | Microbatch | Head→FFN stream-7 gap | FFN RS end→parent RS start | Parent RS∩local kernels | Parent RS∩layers.7 compiled compute |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 7.319755 | 10.918897 | 1.112866 | 0 |
| 0 | 1 | 5.134313 | 11.401042 | 1.158436 | 0 |
| 0 | 2 | 9.341519 | 12.648371 | 1.263779 | 0 |
| 1 | 0 | 7.493793 | 11.961203 | 1.248933 | 0 |
| 1 | 1 | 7.003615 | 11.302353 | 1.140485 | 0 |
| 1 | 2 | 9.411752 | 10.442476 | 1.041638 | 0 |

Rank-0 first microbatch reproduces the objective exactly: lm_head RS
171.710666–179.576214; available suffix ends 173.002828; MTP FFN AG
179.578358–180.320279; stream-7 work resumes 180.322583. Its MTP parent RS is
200.498903–202.813306; norm gradient packing runs 202.817466–202.837050;
layers.7 dense AG runs 202.816026–203.475611; compiled layer computation begins
203.481947. Thus norm packing and layer computation start after the parent RS.
The 1.112866 ms local-kernel overlap is earlier norm/auxiliary backward work,
including copies/fills and arithmetic. It is not layers.7 computation overlap.
All-stream-7 device-event overlap is separately 1.114370 ms; annotation occupancy
is separately recorded and must not be substituted for actual kernel activity.

Exact group payloads are unchanged-control references: head RS 525,336,576 FP32
input elements; parent RS 99,791,360; FFN RS 134,479,872; norm RS 135,168.
Parent/FFN AG outputs contain 99,791,360/134,479,872 BF16 elements. Final dense
AG output contains 85,230,080 BF16 elements. The detailed records also retain
layers.7 expert, layers.6 and layers.5 collectives to prevent misidentifying a
nearby layer reduction or gather as the MTP parent operation.

`control.json` records each final-layer compute span, layer-6 compute start,
layer-6 gather, final dense reduction and the next-microbatch device boundary.
The two rank-0 next-microbatch gaps are 0.407361/0.451808 ms; rank 1 has
0.439554/0.416290 ms. A negative final-dense-RS-end→layer6-compute delta denotes
overlap, not a negative stall. No third next boundary is captured.

## Embedding control

Attribution is through `_OwnerLookup` and `_OwnerLookupBackward` launch ancestry.
It excludes neighboring small kernels owned by other paths.

| Rank | Kernel window | Owner forward spans | Owner backward spans | Combined spans |
| --- | --- | --- | --- | --- |
| 0 | 1172.896806 | 4.261384 | 3.829574 | 8.090958 |
| 1 | 1173.298156 | 4.266259 | 3.235886 | 7.502145 |

Each microbatch has 31 local kernels + 3 NCCL forward, and 34 local kernels +
1 NCCL backward. Rank-0 totals reproduce the objective: local kernels
3.786534 ms; NCCL 1.126498 ms; copy/memset 0.071327 ms; between-operation gaps
3.106599 ms. Forward gaps are 2.819586 ms, including 2.112008 ms before the next
host launch starts. Rank-1 forward gaps are 2.733831 ms, including 2.068814 ms
before the next host launch. These are attribution intervals, not removable
latency predictions.

Each forward has two device→host copies (unique count and exchanged split
metadata), one device→device copy, and two host `cudaStreamSynchronize` API
calls. Host API spans, device copy durations and device gaps remain separate.
There are three `cudaMemcpyAsync` host calls per forward and no backward host
synchronize in the captured owner scopes. Host stream-wait API calls enqueue
dependencies; their CPU durations do not measure the GPU wait.

## Reuse for S and SE

Expected inputs: original complete rank-0/1 JSON traces from the same diagnostic
route (20 warmup updates, three F+B microbatches of update 21 with GPU tails),
resolved source/configuration and activation receipts. Call `prepare(source,
destination)` for each original; retain original hashes; load derivatives through
MCP; execute the saved queries and store responses in the same query envelope
as `d0-rank*.mcp.json`. Page device results below the 5,000-row cap and reject
truncated/error responses. Reconcile original GPU categories, annotation count,
flow/correlation coverage and import statistics before comparisons.

`analyze.summarize(rank, tag, raw_path)` accepts a new saved capture tag and raw
source path. It deliberately asserts one required backward collective per
module/microbatch. Backward AG matching excludes forward-prefetch ancestry and
therefore permits either backward-prefetch or post-backward-prefetch ownership.
Inspect changed annotations rather than weakening attribution if a candidate
fails that assertion. Actual compiled blocks are associated with their module's
pre-backward annotation and local `CompiledFunctionBackward` scope; revisit that
association if a candidate changes compile boundaries. Embedding rows retain
exact operator ancestry for comparing removed copies, coalescing kernels,
ordered owner additions, routing kernels and retained communication.

No profile establishes throughput, convergence, peak memory or full-update
critical-path effects. Control-only traces cannot establish that the repaired
schedule has no later stall or that embedding fusion improves performance.
Candidate acceptance requires new S/SE measurements and matched MCP evidence.
