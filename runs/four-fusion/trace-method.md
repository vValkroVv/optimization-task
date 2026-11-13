# Matched native trace method

Prepared 2026-11-13 before the four-fusion traces arrive. This is a method,
not a measurement or candidate acceptance. No training source, configuration,
runner or canonical trace was changed.

## Finite acceptance for the analysis

1. Verify matched run/config/activation receipts and ranks 0/1, with exactly
   three complete F/B microbatches after ten warmup updates; all eight ranks
   still execute accumulation 32 and optimizer. Inspect each receipt's actual
   update number rather than assuming the historical update 21.
2. Import every recorded device event without changing its time or CUDA stream;
   verify counts, duration sums, flow coverage and importer errors.
3. Report equivalent enclosing-region elapsed time, host time, device interval
   union, kernel-duration sum, overlap and downstream waiting separately, per
   rank and microbatch. Include loss forward AND backward in both variants.
4. Retain exact queries/results, normalization receipts and unresolved attribution.
   A trace establishes mechanism, not a general speedup or numerical parity.

## Preparation: reuse the existing repair

From repository root, for every canonical **gzip** rank trace:

```bash
python3 runs/overnight-streaming-perf/monitorfree-perfetto-prepare.py \
  /absolute/canonical/rank0_trace.json.gz \
  /absolute/task-artifacts/control-rank0.perfetto.json.gz
```

Use distinct output names for control/candidate and rank. Inputs must be gzip:
the helper always calls `gzip.open`, irrespective of filename. Preserve canonical
bytes; if only plain JSON is available, make a gzip copy in task artifacts first.
No timestamp rebasing, duration conversion, stream filtering or GPU-time clipping
is needed. Chrome JSON times are microseconds; Perfetto SQL times are nanoseconds
and `/1e6` converts SQL duration to milliseconds.

The helper removes `gpu_user_annotation` projections, partitions each original
device `(pid,tid)` into nonoverlapping display lanes, and moves matching `ph=f`
flow endpoints using original `(pid,tid,ts)`. It retains device arguments,
including `args.stream`, timestamps, durations, names, CPU events and metadata.
Synthetic display tids are NOT CUDA streams. Do not use thread/track ids as
stream identity after this repair. Use `(device args.device, args.stream)` and
retain raw pid/tid where needed to distinguish devices/processes.

The historical long-epoch baseline receipt recovered 8,795 kernels, 274 copies,
861 memsets; 51 device events/flow endpoints moved. V2 had 59 moves. These are
historical checks, not expected counts for a new capture. Original imports lost
overlapping events; uncorrected imports are unsuitable for exact comparisons.

The helper's assumptions must be checked, not silently generalized: numeric tids;
synthetic tids do not collide with existing tracks; a moved original
`(pid,tid,ts)` identifies one event; each moved endpoint exists and is unambiguous.
An assertion, count mismatch, duplicate endpoint or missing linkage blocks exact
attribution until resolved. Do not change training instrumentation to repair import.

## Perfetto tools and import gate

Call actual `mcp__perfetto__load_trace` with `paths` containing absolute prepared
paths (or one `path`), save returned trace handles, and pass the corresponding
`trace_id` explicitly on every subsequent `execute_sql`. Tool metadata confirms
`slices.with_context` is the generic module; Chrome browser task modules are not
appropriate for Kineto. Results are capped at 5,000 rows: aggregate or partition
queries and save full names when needed. Do not infer missing events from a
truncated result. Table creation is supported; Perfetto tables are immutable,
so use separate CREATE tables rather than INSERT. On schema errors inspect
`list_table_structure` immediately rather than guessing.

Run these as separate SQL calls in order on each trace (fresh table names/session
on rerun). They reuse the successful joins in the historical SQL evidence:

```sql
INCLUDE PERFETTO MODULE slices.with_context;
CREATE PERFETTO TABLE fusion_device AS
SELECT * FROM thread_slice
WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>=0;
```

```sql
CREATE PERFETTO TABLE fusion_link AS
SELECT DISTINCT d.id device_id,h.id host_id,
  h.ts host_ts,h.dur host_dur,h.name host_name,h.utid host_utid
FROM fusion_device d JOIN flow f ON f.slice_in=d.id
JOIN thread_slice h ON h.id=f.slice_out
WHERE h.category IN ('cuda_runtime','cuda_driver');
```

```sql
SELECT category,count(*) n,sum(dur)/1e3 sum_us FROM fusion_device GROUP BY category;
```

```sql
SELECT (SELECT count(*) FROM fusion_device) device_events,
 (SELECT count(DISTINCT device_id) FROM fusion_link) linked_events,
 (SELECT count(*) FROM stats WHERE severity='error' AND value>0) importer_errors;
```

Compare category counts exactly and duration sums at source timestamp precision
against the helper receipt. Query positive error stats individually. Check link
multiplicity with `GROUP BY device_id HAVING count(*)<>1`; do not multiply kernel
durations by multiple flow rows. Zero missing links was achieved historically;
new missing links require raw correlation investigation before exact tail/owner
claims. Driver launches (`cuLaunchKernel*`, `cuLaunchKernelEx`) and runtime
launches both count. Raw fallback joins use correlation plus process/device
scope and a bounded temporal match, with ambiguity reported; correlation alone
is not a globally unique key. Flow arrows represent host/device launch
correlation, not arbitrary CUDA data dependencies.

## Common windows and joins

First enumerate CPU `user_annotation` names/counts/durations for
`task.profile_update.*`, `task.token_count*`, `task.prefetch.*`; separately retain
the original GPU projections as coverage evidence before normalization. Match
`task.profile_update.<update>.microbatch.rank<rank>` exactly and number by ts.
Require three complete CPU markers. Input prefix begins at the train-step CPU
marker and ends at microbatch 1 start. Token reduction must be complete before
that first microbatch. The bounded train-step marker is NOT the full update.

For each microbatch CPU `[lo,hi)`, assign launches by `lo<=host_ts<hi` and their
linked device events, then calculate `max(device.ts+device.dur)` as its device
tail. Also report CPU `hi-lo`, GPU first-start to final-end span and
`max(hi,device_tail)-lo`; these are distinct quantities. Repeat for the entire
three-F/B region. Inter-microbatch overlap means the individual spans need not
sum to the enclosing elapsed time. Report unattributed/late worker-thread
launches if the launching thread is not enclosed by a microbatch marker.

For a known CPU/custom-op root `r`, select host launches on the SAME `utid`
with `h.ts>=r.ts AND h.ts<r.ts+r.dur`, then follow `fusion_link` to device events.
When nested operator ownership is needed, use `ancestor_slice(host_id)` or
the shortest enclosing CPU/custom-op slice on that utid; do not sum inclusive
parent and child durations. Preserve the enclosing compiled-region name too.
GPU temporal overlap with a CPU marker is not operator ownership.

For every device window clip intervals to `[lo,hi)`, discard empty intervals,
and compute unions with the maximum end of ALL previous intervals, not merely
`lag(end)` (nested overlaps would be wrong):

```sql
WITH clipped AS (
 SELECT id,MAX(ts,${LO_NS}) b,MIN(ts+dur,${HI_NS}) e
 FROM fusion_device WHERE ts<${HI_NS} AND ts+dur>${LO_NS}
), prior AS (
 SELECT *,MAX(e) OVER(ORDER BY b,e,id
 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) previous_end FROM clipped
)
SELECT count(*) n,sum(e-b)/1e6 kernel_and_copy_sum_ms,
 sum(MAX(0,e-MAX(b,coalesce(previous_end,b))))/1e6 device_busy_union_ms
FROM prior;
```

Replace the literal `${LO_NS}`/`${HI_NS}` placeholders with integer bounds.
Partition the window expression by actual stream or family for those unions.
Compute family intersections from merged intervals (or inclusion-exclusion of
two unions); pairwise raw kernel overlap sums overcount. Retain copies, memset,
NCCL, HybridEP and other kernels separately. Classify explicit EP names before
broad `*nccl*` matching: EP mangled names can contain `nccl`. Communication and
spin kernels include waiting; `other_kernel` does not prove useful computation.

## Minimum equivalent regions per candidate

| Candidate | Candidate anchors | Control region and required surrounding work |
| --- | --- | --- |
| Input preparation | `four_fusion::prepare_input_tensors` or `_scalar`; `_prepare_kernel` | Microbatch entry through first model launch, covering `Trainer.post_dataloading_process`, local valid-token count, aux share update and `loss_fn.pack_targets`. Include masks/positions, packing/copies and aux buffer copy in both. Keep input-prefix/token-reduction costs separate: this fusion runs inside each F/B, not the prefetch queue. |
| Routing metadata | `four_fusion::routing_metadata_cuda`; `_routing_metadata_kernel` | Per `hybridep::dispatch` occurrence: retained expert IDs/weights to dispatch consumption. Control `indices_to_map` builds offsets, valid mask, indices, routing/probability storage and scatters. Include dispatch-ready gap/wait and enclosing dispatch elapsed; report SAC recompute occurrences separately from original forward. Compare metadata backward/gather if visible; do not assume the entire dispatch kernel changed. |
| Linear CE | `four_fusion::native_linear_ce`; tiled `forward_kernel` and `backward_kernel` plus GEMMs | Entire `ChunkedCELoss._chunked_loss_and_grad`: head gather/setup, both native sequence chunks, main+MTP heads folded into batch, head GEMM/CE, immediate inner backward, gradient accumulation, head sync/reshard. Candidate computes dinput/dweight INSIDE the custom op forward; timing only its autograd backward falsely reports elimination. Include tiled dinput/dweight GEMMs, fp32 dweight initialization/accumulation/cast and terms reduction. |
| Stable CE | `four_fusion::stable_ce_forward`, `four_fusion::stable_ce_backward`; `forward_kernel`, `backward_kernel` | Matched CE and CE backward per chunk/head (including cast/contiguous/reductions), plus enclosing head+loss F/B as a guard against moved cost. Control compiled CE may not expose an `aten::cross_entropy_loss` marker. Head GEMM unchanged still belongs in the enclosing comparison. |

These four modules have no dedicated `record_function` boundary markers.
Custom-op names and kernels establish candidate activation, but do not create
an equivalent control boundary. `forward_kernel`/`backward_kernel` names are
shared by the two loss candidates; attribute by enclosing custom op and run
identity. Do not combine the independently patched candidates.

Use actual trace CPU/custom-op/compiled-region names and launch ownership to
choose paired boundaries. If baseline compilation hides the minimum block,
compare the smallest common visible enclosing region and report that exact
block attribution remains unavailable. Do not invent source-level markers or
estimate baseline region time by subtracting a kernel histogram. No change to
the frozen source/runner is authorized by this note.

## Downstream waits and artifacts

For each anchor retain its producing launch, first consuming GPU launch, actual
stream, intervening event record/wait/synchronize APIs and concurrent copy/EP/
NCCL work. Report launch API duration, launch-start-to-device delay and gap on
the consuming stream separately. `cudaEventRecord`, `cudaStreamWaitEvent`,
`cudaEventSynchronize`, `cudaStreamSynchronize` and driver counterparts are
leads; only establish a dependency when raw args/source identifies the event
and producer/consumer. Flow alone or adjacent timestamps cannot establish a
critical-path wait. Intersect downstream gaps with all-device idle and other
stream activity; distinguish exposed idle from compute-stream waiting while
communications run. Name missing event identities or stacks as unknowns.

Preserve canonical gzip + SHA256; prepared gzip + receipt + helper SHA256;
execution/source/config/patch pins; resolved rank-0 settings and activation logs;
all-rank completion receipts; per-rank query text/results/import statistics;
chosen region IDs/names/bounds and launch/stream tables; classifications and
ambiguous/unlinked events. Large files stay in task artifacts, outside tracked
evidence. Do not align independent rank clocks without explicit clock evidence.
Three sampled microbatches omit the remaining 29 and optimizer; no extrapolated
full-update time, speedup, memory saving, loss parity or convergence conclusion.

## Basis and verification of this method

Inspected the existing preparation helper, successful SQL/results and import
receipts in `runs/v2-prefetch-baseline/trace-analysis-long-epoch-perfetto.json`,
current Perfetto tool metadata, four explicit modules and integration patches,
`Trainer.forward_backward_step`, HybridEP metadata helper and native chunked
loss loop. Existing evidence verifies the preparation and flow join on this
trace family. New templates have not been executed on new traces, which are not
yet available; new import/schema/attribution gates remain with their analyst.
All five SQL blocks parsed under SQLite using synthetic versions of the stated
tables; nested intervals `[0,10)`, `[2,5)`, `[9,14)` produced duration sum 18
and union 14. This checks SQL structure/union arithmetic, not Perfetto import
or schema. Markdown fences and scoped whitespace checks passed.
This note has no runtime cost: all source/config/runner bytes remain unchanged.
