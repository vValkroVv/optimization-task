# Native profiler and measurement inspection

Status: ARCHIVED historical evidence from the cancelled/incomplete original
campaign. Old numerical-harness, monitoring-inclusive, long-window/cache and
pre-performance numerical instructions below are retired; they describe the
method reviewed at that time, not active requirements. The [revised plan](../../planning/overnight-streaming-perf.md)
and [validation](../../planning/validation.md) own the cleanup-only assignment
and future 10+10 monitoring-off/performance-first local-block policy. No new run
or trace is authorized. Referenced removed modules are historical symbols.

Read-only design inspection at source `a2a59e14545b55ca777f94a28e332759e7c06087`.
No training, benchmark, source modification or runtime correctness test was
performed by this investigation. The coordinating agent owns implementation and
its independent review.

Finite acceptance: establish native trace boundaries and defaults; identify the
smallest opt-in rank/shape change; establish physical/supervised counters and
timer scopes; define a complete 50-update measurement and its focused checks.
These inspection items are complete; implementation/runtime gates remain open.

## Selected design

`gift/torchtitan/config/schema/tools/profiler.py::ProfilerConfig` lacks rank and
shape controls. `gift/torchtitan/tools/profiler.py::Profiler.build_torch_profiler`
constructs a profiler on every rank and hardcodes `record_shapes=True`.
Add generic typed `profile_ranks: list[int] | None = None` and
`record_shapes: bool = True`. Validate nonnegative, unique rank IDs; fail clearly
for IDs outside the initialized world rather than silently capturing no trace.
Return `None` on nonselected ranks before directory creation or profiler
construction. Pass `record_shapes=cfg.record_shapes`. Keep defaults unchanged.
Stacks and profiler memory recording are already false by native PyTorch defaults;
passing false explicitly documents the campaign requirement. Independently keep
`enable_memory_snapshot=False` to prevent `_record_memory_history`.

This is preferable to copying the native builder or temporarily monkeypatching
the global torch profiler constructor. The task-only measurement adapter belongs
outside the training tree and delegates to unchanged `torchtitan.train.main()`.
PyTorch's [official profiler API](https://docs.pytorch.org/docs/2.15/profiler.html)
documents the native schedule and warns that shape capture retains references
and can add overhead. Installed dev-build schedule behavior must still be checked.

`gift/torchtitan/trainer.py::Trainer.train` calls `profiler.step()` once AFTER
`train_step`, checkpoint handling and validation. `train_step` performs all 32
microbatches, optimizer, scheduler, EMA/other callbacks, monitoring emission and
core metric output. No changes to training participation or collective order are
required. For a fresh process targeting update N, use `profile_freq=2`,
`profiler_warmup=1`, `profiler_active=1`, `profiler_repeat=1`,
`profiler_skip_first=N-2`, `profiler_skip_first_wait=0`. Update N-1 warms the
profiler, N records, and end-of-update export produces
`iteration_N/rank0_trace.json` and `rank1_trace.json`. Select N after settled
warmup and require N modulo monitoring interval != 0. This formula assumes fresh
step zero: native overwrites `step_num` after `__enter__`; resumed scheduling is
not established by inspection and is not used by this campaign.

## Counters and timer findings

- `Trainer.train_step` increments
  `metrics_processor.ntokens_since_last_log` by post-mutator `labels.numel()`.
  `MetricsProcessor.interval_metrics` divides these PHYSICAL local tokens by
  elapsed host time and `non_data_parallel_size`; with PP/TP/CP1 its console TPS
  is per-GPU input TPS when verified input/label geometry agrees. It does not
  count only supervised labels or multiply by MTP depth.
- `global_valid_tokens` sums `(labels != IGNORE_INDEX)` across all microbatches
  and the batch mesh, with local clamp to one. The full-row fixture has nonzero
  counts, so the clamp is inactive. `Trainer.ntokens_seen` is int64 and adds that
  global count ONCE per optimizer update. Read its synced window endpoint delta
  for supervised TPS; do not multiply by MTP. Expected full fixture numerators
  are 2,097,152 physical and 2,096,640 supervised tokens per update, subject to
  actual row/mask checks.
- `TrainerSource.after_optimizer_step` snapshots `ntokens_seen` BEFORE the
  trainer updates it. Its throughput token deltas lag one update, including an
  initial zero delta. Do not use this series as primary campaign TPS.
- `TrainerSource.batch_total` ends before complete monitoring reduction/emission
  and core metric output. `MetricsProcessor.interval_metrics` resets its timer
  before logger emission, attributing output cost to the next interval; its
  `time/train_step` always divides by log_freq, including the first single step.
  Neither gives the required individual complete iteration series directly.
- `Trainer.batch_generator` times only `next(data_iterator)`: its data-loading
  metric excludes subsequent device transfers and model work.

## Task-only timing contract

Intercept return from native `Profiler.step` in every process. Boundaries after
updates W and W+50 include exactly updates W+1..W+50, including input fetch,
GC, transfers, 32 forward/backward microbatches, optimizer/scheduler, monitoring
and core metric output. Validation, saving and memory snapshots remain disabled
by the authorized campaign configuration. Unprofiled runs require profiler off.

Synchronize CUDA once before recording the W start timestamp and once before
recording the W+50 final timestamp. Add no per-update synchronization, barrier,
collective or device scalar read. Buffer local monotonic timestamps and write
JSON after the measured window. Record individual differences as host completion
cadence, with the final interval including final completion drain; they are not
device kernel time. Complete-window elapsed throughput is the primary metric.
Use all-rank elapsed series for skew; on this single node, monotonic clocks share
the kernel clock. Report per-rank spans/spreads and aggregate rate using the
slowest local complete-window elapsed, with its stated boundary convention.

To reuse physical counters without per-microbatch overhead, intercept
`MetricsProcessor.interval_metrics` before it clears the existing physical-token
counter and maintain a Python-int flushed count. At either window endpoint,
total physical tokens equal flushed count plus current live counter. Assert
input/label geometry in the data-contract check; label storage is not universally
an input-token counter. Synced endpoint reads of `ntokens_seen` give supervised
count. Keep native metrics cadence and monitoring interval unchanged.

## Memory and runtime costs

`tools/utils.py::device_capacity_report` reports active/reserved allocator peaks
and current non-PyTorch memory. Its console total is reserved peak plus
report-time non-PyTorch usage, not a simultaneous physical-device peak.
`MetricsProcessor.log` resets allocator peaks after native logging. Therefore
collect `memory_stats()` at iteration boundaries AND immediately before each
native reset (`components.metrics.reset_peak_stats` is an imported alias).
Fold allocated/active/reserved peaks into task-owned maxima across the 50 updates.
Retain start/end retry and OOM counters and their deltas. Do not reset native
counters or redefine their lifecycle. Sample `mem_get_info` only at boundaries
if required, clearly labeling non-PyTorch measurements as endpoint samples.

Added costs are O(1) timestamps/bookkeeping and allocator counter reads per
update/rank, and two window-boundary CUDA synchronizations. No model arithmetic,
tensor allocation, extra device kernel, communication or checkpoint state is
needed. The same instrumentation must run for every control/candidate. Profiling
adds substantial selected-rank host/device overhead and export stalls peers at
their next collective; its timing cannot enter unprofiled TPS comparisons.

## Required focused checks and review risks

1. Typed config decoding accepts controls, rejects invalid rank IDs and unknown
   names, preserves defaults, remains torch-free and updates any affected golden
   fingerprints intentionally.
2. Behavior test mocks profile construction: selected global ranks enter once;
   nonselected ranks never construct, enter or export; shape capture is false
   only on opt-in. Memory history constructor is never invoked when disabled.
3. CPU native profiler test with named update/microbatch markers verifies one
   exported active update, all 32 markers and optimizer marker, export numbering,
   repeat termination and no extra trace on normal context exit. Compare schedule
   actions against the installed torch dev build. CPU success does not prove CUDA.
4. Fake trainer/profiler/clock integration exercises W..W+50, one monitoring event,
   core metric-output inclusion, exactly two completion syncs, callback exceptions,
   no partial-window success, buffered output and per-rank identity. Test flushed
   physical counters across a native metric reset and memory maxima across reset.
5. Eight-GPU native trace acceptance: all eight ranks complete every update; only
   ranks 0/1 instantiate profilers; exactly two valid traces; target update is
   ordinary; 32 microbatches and optimizer appear; no shape/stack/history capture;
   hashes/config/source and Perfetto queries retained. Reuse authorized integration
   rather than launching an unrelated miniature training study.

Independent review must check timer boundary off-by-one, default behavior,
reset aliases, final GPU completion, all-rank lifecycle and failure propagation,
counter double counting, resumed-schedule assumptions, and instrumentation costs.
No numerical change is intended. Native loss/grad/update behavior must remain
unchanged; CPU mocks cannot establish GPU/NCCL timing or production acceptance.
