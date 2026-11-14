# Timing drift during training

Updated 14 November 2026. This retrospective note records the timing variation visible
within three completed 140-update runs and separates measured observations from possible
causes. The companion [comparison JSON](historical-drift-comparison.json) contains
all-rank window medians and sidecar hashes, rank-zero ten-update bins, and the recorded
garbage-collection lines.

## Observed host cadence

For each run, rank-zero median cadence was lower in updates 121–140 than in updates
41–60. The historical baseline `combined-four-baseline-003` changed from 15.0376687075
seconds to 13.485153973 seconds, a 10.32417168% decrease. The historical candidate
`combined-four-candidate` changed from 14.3497127895 seconds to 12.932786025 seconds, a
9.87425174% decrease. The current control `fqla-native-fla-001` changed from 14.35722263
seconds to 12.8648394995 seconds, a 10.39465062% decrease. All eight ranks show the same
direction in all three runs.

The rank-zero ten-update bins are nonmonotonic. From updates 11–20 to 31–40 and then
131–140, the historical baseline medians are 13.564546544, 15.5186635585, and
13.468053907 seconds; the historical candidate medians are 12.950386093, 14.878632388,
and 12.862225862 seconds; and the current control medians are 12.9539951575,
14.8559093035, and 12.8449245175 seconds. Each curve first rises and then recovers.

These values are host cadence intervals recorded after `Profiler.step` returns. They are
not device-only durations. The windows show that update timing changes during training,
but they do not establish stationary throughput, a general speedup, or the cause of the
change.

## Data loading and cache hypothesis

Data-cache effects remain possible. The selected MDS fixture is 14,702,684 bytes, about
14.7 MB, with 896 physical rows on NFS. The run uses local stream mode, shuffle
disabled, and two persistent workers. Its configured logical epoch is 102,400 units,
while 140 × 512 is 71,680 slots, so the run does not cross a logical-epoch boundary. The
original task copy's `runs/flash-qla/artifacts/raw/mds-copy.json` already verifies the
fixture-copy identity; that receipt is not mirrored in the local task copy, and this
note does not recompute shard hashes.

The small repeated fixture could be affected by filesystem or host cache state, but no
per-window I/O-wait or cache-miss telemetry was recorded. The observations therefore do
not rule out data-cache effects. The initial rise followed by recovery also means a
simple monotonic warm-up story is not sufficient by itself.

## MoE imbalance working hypothesis

Changing MoE routing and load imbalance is the main working hypothesis for the timing
hump, but it is unproven. The relevant source was inspected in the frozen remote
`/workspace-SR008.nfs2/vkropotin/opt-fq` tree at revision
`0f32623d5e0e7544b20c1f047d94236cb6697758`; that checkout is not mirrored in the local
task.

`load_balance.py` under `gift/torchtitan/models/common/moe/` (lines 22–40 and 133–140)
updates adaptive bias by `0.001 * sign(mean-load) * clamp(1 + max_vio, 1, 5)` from an
optimizer pre-hook on every update. `router.py` in the same directory (lines 158–167)
adds that bias before top-8 selection. `hybridep.py` under
`gift/torchtitan/distributed/deepep/` (lines 270–275, 318–339, and 378–379) has blocking
receives based on actual padded row counts. `gmm.py` under
`gift/torchtitan/models/common/moe/experts/` (lines 242–322) groups the current routed
work for grouped FP8 execution. These paths make changing per-expert and per-destination
work a plausible source of changing execution time; they do not show that such a change
occurred in these windows.

Each B2/S4096 microbatch has 8,192 local tokens per rank. With EP8 and top-8 routing,
that gives 524,288 useful routes per microbatch. Each optimizer update accumulates 32
microbatches. The capacity function in
`gift/torchtitan/models/common/moe/dispatch/capacity.py:100-142` takes scores shaped
`[tokens, top_k]`, then counts routes and computes the group budget for that call. This
means the route, padding, and group-capacity comparisons here are per microbatch. The
recorded global padding bound is at most 16,256 rows per microbatch, less than 3.101% of
the 524,288 routes. Padding volume alone is not enough to explain a roughly 10% whole-step
cadence decrease. A transient routing imbalance could fit the early hump and
later recovery, but no expert or destination counts, padded-row series, or
maximum-violation series were recorded across updates 41–140.

## Other evidence and limits

A separate combined-four diagnostic at update 21 sampled ranks 0 and 1 across all 32
batches before forward/backward. It reported readiness 32 with a minimum of 4 and total
pop spans of 0.370 ms and 0.402 ms for the two ranks. The remote task's
`runs/combined-four/trace-analysis.md` (not mirrored in the local workspace) records
that result. It does not establish queue readiness or receive timing during updates
41–140.

Periodic garbage collection was recorded at updates 50 and 100: 0.81/0.61 seconds for
the historical baseline, 0.67/0.61 seconds for the historical candidate, and 0.65/0.57
seconds for the current control. These are isolated events and do not establish the
cause of a smooth cadence trend. The older candidate also showed the same hump with 39
cached and 0 compiled DeepGEMM startup entries. No post-startup recompile was logged,
but the compiler observer was disabled, so the logs cannot prove that recompilation
never occurred.

Token dropping cannot trigger for any microbatch under the recorded EP8 capacity
arithmetic: `50000 // 8` gives 6,250 tokens per expert. The group-overload threshold
is 6,250 × 128 = 800,000 per microbatch, above the maximum 524,288 routes in that
microbatch. The [overnight report](../overnight-streaming-perf/report.md) records this
bound at lines 308–315. This rules out token dropping as an explanation for these runs
under that configuration; it does not identify the source of the cadence drift.

## Useful future discriminator

If a later task separately authorizes a diagnostic, the smallest useful discriminator is
to record per-expert and per-destination route counts, padded rows, maximum violation,
and receive readiness on all eight ranks around updates 30–40 and 120–140. Per-update
input-wait and cache counters over the same windows would help separate data-path
effects from routing effects. This note does not authorize another run. No source,
configuration, or runner files changed; this documentation has no training runtime
effect.
