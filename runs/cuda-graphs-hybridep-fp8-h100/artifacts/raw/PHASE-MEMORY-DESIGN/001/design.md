# OPT-004 allocator phase observability: bounded design

Status: design only; no source modification, tests, installation or GPU execution.
Inspected task repository revision: `446431dfa191ef0300f116257ae918bc85e38c06`.
The active Source006 matrix remains frozen. The supplied workload contract and
A008 replay evidence are inputs; this document does not revalidate those runs.

## Finite acceptance checklist

1. Locate actual allocator producer, reset, callback and lifecycle boundaries.
2. Select one opt-in task-local observation route that preserves native metrics,
   reset cadence and computational behavior without new per-update GPU work.
3. Define exact supported phase semantics, all-rank records and honest limits.
4. Specify focused behavior tests, GPU gates, cleanup and runtime costs.
5. Explain reuse of Source006 evidence and the remaining 500-update memory gate.

Items 1–5 are addressed below as a design. Implementation/runtime acceptance
remains open.

## Established source behavior

All paths below are relative to the repository root.

- `gift/torchtitan/trainer.py:719`, `Trainer.train_step`: `batch_start` precedes
  zero-grad, data transfer, four-microbatch forward/backward and optimizer update;
  `batch_end` precedes native metric collection. Native logging is gated by
  `MetricsProcessor.should_log` at update 1 and multiples of `log_freq` (10).
- `gift/torchtitan/components/metrics.py:487`, `MetricsProcessor.log`: every
  training rank computes `device_capacity_report`, passes metrics to its local
  logger, then resets peaks at line 515. Logger rank filtering does not filter
  the capacity computation. `__init__` also resets; validation resets separately
  in `log_validation`. Preserve all of these calls and their ordering.
- `gift/torchtitan/tools/utils.py:286`, `device_capacity_report`: reads the raw
  `device_module.memory_stats()` dictionary, passes it to `device_memory_usage`,
  then publishes **active**, not allocated, peak percentages. Therefore copying
  only its returned report or console line cannot recover allocated peak.
  `device_memory_usage` calls `mem_get_info` and estimates total memory as
  reserved peak plus **current** non-Torch memory. That sum is not an observed
  device high-water mark: the terms may occur at different times.
- `gift/torchtitan/callbacks/builtin/system_monitor.py`,
  `SystemMetricsMonitor.get_memory_report`: correctly maps allocated, active,
  reserved current/peak counters, but makes an additional `memory_stats` and
  `mem_get_info` call, rounds byte quantities to five significant figures in GiB,
  and routes output through the callback metric emitter. Enabling this callback
  is not necessary for the proposed native-stat reuse and would add work.
- `gift/torchtitan/callbacks/base.py`, `ObserverState`: exposes neither trainer
  nor metrics processor. There is no post-native-metrics/pre-reset event.
  `batch_end` alone cannot receive the dictionary computed later by native log.
- `gift/torchtitan/experiments/cuda_graph_campaign/observer.py`,
  `CampaignTimingObserver.batch_start`, `_finish_window`, `_memory_boundary`:
  samples current/peak values only at opening/closing, with existing opening and
  closing synchronizations. Intervening native resets erase phase maxima.
  `_finish_window` completes/synchronizes at the next batch start, before its
  all-rank MAX aggregation and persistence. Preserve these timing definitions.
- `gift/torchtitan/trainer.py:901`, `Trainer.train`: checkpoint and validation
  follow native log; normal `fit_end` occurs after the loop and a rank-zero
  sleep. `Trainer.close` closes callbacks before checkpointer/metrics.
  `gift/torchtitan/train.py:57` closes a constructed trainer on ordinary
  exceptions; an outer task entry `finally` is needed for BaseException and
  construction-failure restoration/partial flush.
- `gift/torchtitan/ops/warmup/deep_gemm_warmup.py:447` contains a direct CUDA
  peak reset in a different warmup utility. Trainer actually imports
  `deep_gemm_fast_warmup.setup_deep_gemm_cache`; do not count all repository
  reset sites as active. Any future active direct reset not covered below must
  invalidate an affected interval rather than silently stitch it together.

Official API cross-check: [PyTorch 2.15 memory_stats](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.memory.memory_stats.html)
distinguishes `allocated_bytes`, `active_bytes`, and `reserved_bytes`, and current
versus peak statistics. API pages describe the release family; Source006 runtime
is recorded as `2.15.0.dev20260822+cu130` in the existing A008 runtime-source
excerpts. Pin/check the actual API implementation identity during implementation;
release documentation alone does not establish the development build's behavior.
[NVIDIA's CUDA Graph memory explanation](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/memory-issues.html)
also explains why active and allocated can differ across streams.

## Selected route: native-stat tap in an opt-in task entry

Use one small adapter outside the training source, loaded explicitly by the task
entry before native `main`. Do not change core/framework files, torch APIs,
native logger configuration, or `save_for_all_ranks`. An ordinary native entry
never imports/installs it. Proposed interface:

```
restore = install_phase_memory_observer(output_dir, phase_boundaries)
try:
    native_main(existing_argv)
finally:
    observer.finish_and_flush(primary_exception)
    restore()
```

The adapter holds only original callable identities, rank-local Python scalar
records, and a small active-call context. Install before Trainer construction:

1. Wrap `MetricsProcessor.log` and `log_validation` with `*args, **kwargs` to set
   `{step, train_or_validation}` for the duration of the **original** call.
   Restore the prior context in `finally`; no early return or duplicate calls.
2. Wrap the `torchtitan.tools.utils.device_memory_usage` module binding. The
   original `device_capacity_report` resolves this binding at runtime and
   already supplies the raw stats. While inside a native log context, copy only
   the required Python integers into a record, then return the original helper's
   result unchanged. No additional per-log `memory_stats`, `mem_get_info`, scalar tensor
   extraction, CUDA event, synchronization, or collective. Calls without an
   active native-log context pass through without recording. A configured
   SystemMetricsMonitor imports its own helper binding and remains unchanged.
3. Wrap **the imported** `torchtitan.components.metrics.reset_peak_stats`
   binding, not only `utils.reset_peak_stats`: invoke the original exactly once,
   record reset completion time, context and generation. This catches processor
   construction and native train/validation resets. Detect nesting, a reset
   without its expected native snapshot, or a snapshot without a reset. Do not
   turn these into valid contiguous intervals. Restoring utils alone would miss
   metrics' imported binding.
4. Add only boundary timestamp capture to the existing timing observer through
   a task-local installed method wrapper (same identity-checked restoration).
   At opening and closing, use the observer's existing `perf_counter` values and
   take bracketed `time.time_ns()`/`perf_counter_ns()` anchors. Record the timing
   observer's true completed stop before aggregation/I/O. Do not substitute a
   second callback that runs before the original closing synchronization and
   then call its timestamp a completed boundary. Phase labels can be assigned
   offline from update/reset IDs; no per-update allocator read is needed.

Do not rewrite the metric report implementation or duplicate its formulas. Fail
installation before execution if expected bindings/signatures differ. Require a
single install, restore exact borrowed bindings in reverse order, and refuse to
restore over a third party's replacement. Tests must exercise native call paths.

## Phase scope and what can be exact

A peak is the allocator counter maximum **since its actual last reset**, observed
at the native snapshot. Record both reset completion and snapshot timestamps;
there is a small snapshot-to-reset gap containing native metric logging. It is
not sampled. Do not call a record a peak through reset completion, or pretend the
gap is zero. With disabled external metric backends, no training GPU allocation
occurs in this gap by inspected call order, but this assumption must be checked
against resolved configuration and the actual logger. Future loggers that
allocate GPU storage invalidate complete phase coverage under this route.

The first native reset is `MetricsProcessor.__init__` at metrics.py:316, built
from Trainer.__init__ at trainer.py:188. It follows `setup_deep_gemm_cache` and
meta-model construction, and precedes model parameter materialization. Thus the
first reused native snapshot covers this constructor reset through update1;
subsequent records cover reset1→snapshot10 and reset10→snapshot20. Their maximum
is the **post-metrics-construction startup/warmup/capture-settling envelope**.
Record that origin explicitly; it is not a whole-process bootstrap peak.
The earlier direct-reset utility is not imported by Trainer. The actual fast
warmup entry has no direct reset call; it may launch separate warmup work, whose
process/device semantics are not part of this training-rank allocator envelope.

If whole-process allocator bootstrap memory is wanted, the same adapter can make
one host `memory_stats` read immediately BEFORE the constructor's native reset,
retain it as `pre_metrics_constructor_bootstrap`, then execute the native reset
unchanged. This is a once-only startup read, not a per-update query or reset.
Keep its original counter origin explicit (process allocator initialization unless
an earlier active reset is demonstrated). Without this read, earlier bootstrap
allocated/reserved peak is unavailable. The existing whole-run device sampler
still covers that lifecycle with its stated resolution.

The profiled updates21–23 in A008 have ZERO captures and persistent loss replay;
they are not a capture phase. Define settled steady memory as native reset20→
snapshot30, reset30→snapshot40, and so on through70 or500. Aggregate the maximum
allocated/reserved peak across these intervals on each of eight ranks. Preserve
startup and steady records separately. Verify no later recapture contradicts the
settled classification; if it occurs, annotate/reclassify the affected interval.
These reset-aligned scopes establish phase-envelope allocator peaks, not exact
memory cost of individual capture API calls. No narrower capture-only acceptance
requirement is introduced.

The existing CPU throughput window begins batch_start21 and ends completed
batch_start71 (or the later configured boundary). Its exact endpoints differ
slightly from native reset20→snapshot70. Persist both definitions and their
actual UTC/monotonic times; do not call them identical. Add the existing closing
memory sample for any residual lifecycle tail, but do not conflate its since-reset
peak with a previous native interval. If the tail could allocate GPU memory,
report its coverage explicitly. Do not add resets to manufacture alignment.

## Records, device samples and headroom

Every rank writes its own file after measurement. Required fields:

- schema/run/source/config identities, rank/local rank/world size, device UUID or
  a verified index mapping, allocator backend and byte capacity;
- event kind, step, context, reset generation, previous-reset completion times,
  snapshot bracket times, native reset completion, coverage/status/reasons;
- integer `allocated_bytes.all.current/peak`,
  `reserved_bytes.all.current/peak`, optional separately named active values,
  retry/OOM counters; missing keys are null with reason, never zero;
- rank-local measured-window opening and completed-close UTC/monotonic anchors,
  bracket widths, phase envelopes and actual interval memberships;
- device sample source/cadence/count/first/last UTC, maximum observed used bytes,
  maximum observed sample gap, phase and boundary exclusions, device capacity.

The existing attempt004 launcher already gives NVML/nvidia-smi samples absolute
UTC (`TZ=UTC`, `timestamp,index,memory.used`, `--loop-ms=500`). The missing link is
absolute timing/phase windows in rank records, not absent sampler timestamps.
Offline join samples to each rank's device and recorded window/envelope; exclude
ambiguous boundary samples using clock-anchor bracket uncertainty, state sample
resolution and any wall-clock discontinuity. Require UUID/index mapping proof;
rank number is not inherently device index. An empty phase sample set is
unavailable. Report whole-run and per-envelope values separately.

A 500 ms sampled device maximum is a lower bound on true device peak. Its
corresponding `capacity - sampled_max_used` is an **upper bound** on worst-case
free headroom, not guaranteed minimum headroom. `capacity - reserved_peak` is
allocator-only headroom and excludes non-Torch allocations. Native
`reserved_peak + current_non_torch` remains an estimate; do not relabel it an
exact device peak. No exact whole-device phase high-water mark is established by
the inspected APIs. Preserve that explicit open metric rather than inventing a
zero or an exact result. The inspected generic NVML API URL failed to fetch in
this design pass; no claim of a globally unsupported NVML feature is made.

## Lifecycle and bounded cost review

Buffer only fixed scalar fields, roughly 52 native records per rank for 500
updates plus boundaries. Do not retain raw dictionaries, tensors, models or
tracebacks. Flush once outside the completed measured window or after native
main returns; no per-update JSON, file flush, fsync, background thread or remote
write. On ordinary/BaseException failure, flush existing partial records with
status incomplete, no collectives/device reads and no attempt to close a hung
rank. Flush failure must not replace the primary exception; attach a note and
log a concise secondary error. Restoration must run even when persistence fails.
SIGKILL/node loss can lose the RAM buffer; missing rank files fail completeness.
Optional recovery does not justify per-step I/O. Output collision fails before
training; do not overwrite an earlier run.

Cost before implementation: additional Python calls and a handful of integer
copies/timer reads only at native log/reset cadence, plus four UTC clock anchors
at timing-window boundaries. RAM is O(native logs), not O(tokens/model size).
The optional bootstrap read is once at construction. No new device allocation, precision conversion, computation, compilation region,
communication, checkpoint state, or autograd/RNG mutation. No per-step host-value
reads. Native `mem_get_info` and metrics syncs remain inherited costs.
Timer/callback/copy costs are unmeasured, not claimed zero. A mocked 500-step CPU
call-path check can count operations and time the observer's own calls without
GPU; actual GPU runtime magnitude remains an on/off observation gate only if
unexpected perturbation is material. No extra training benchmark is warranted
merely to produce another number. Reconcile this cost list against the final
implemented adapter once.

## Focused verification and remaining gate

Proposed tests (not executed here):

1. Native MetricsProcessor log with mocked device helpers and logger: allocated
   peak differs from active, early high peak disappears after reset, later
   closing sample is smaller. Assert observer retains the high allocated and
   reserved values, native metrics/results and reset counts/order are unchanged,
   all-rank BaseLogger paths still record, and no new per-log GPU helper invocation; optional constructor read occurs once.
   This regression must fail with the current boundary-only observer.
2. Native train/validation/constructor contexts: known reset generations,
   validation contamination flagged, gaps/missing keys/duplicate install/output
   collision rejected, no false claim that profiled21–23 captures or reset scopes equal CPU scopes. Offline phase max
   and sample-window/device mapping include boundary uncertainty and empty sets.
3. Adapter lifecycle: native exception, construction failure, BaseException,
   persistence failure, exact restoration/idempotent close, no primary exception
   masking. Default native entry has no adapter import/calls/files.
4. Existing observer opening/closing tests: retain original timers/events,
   synchronization/reduction counts and partial-window behavior; anchor brackets
   refer to actual opening/completed-close instants.

GPU acceptance: on the authorized matched 500-update baseline and candidate,
apply the same opt-in observer to both, preserve exact Source006 compute/compiler
policy, prove eight complete rank files and native cadence, inspect resolved
allocator/logger settings, correlate actual capture settling, and report all
phase-envelope allocated/reserved maxima plus sampled device peak/headroom
bounds. Any direct reset, missing record, logger device allocation, OOM or
recapture changes coverage/verdict and must be explained. Exact individual-API-capture-only peaks and exact whole-device phase peaks remain
unavailable. Native reset-aligned startup/settling and steady envelopes are the
reproducible phase definitions; no new owner approval or impossible narrow-phase
requirement is introduced.

The Source006 A/B/A/B matrix answers integrated performance with its original
instrumentation. This observation-only addition does not invalidate its completed
mechanism/numerical evidence or require repeating that matrix. Keep a distinct
adapter hash and instrumentation manifest for the 500 pair; compare that pair to
each other, and do not pool differently instrumented timing series as identical.
If measured activation/configuration or adapter behavior reveals a material
computational change or timing perturbation, repeat only the affected comparison.
No such finding is established by this design.
