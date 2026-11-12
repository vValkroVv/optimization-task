# CUDA Graph campaign callbacks

The native GIFT trainer can build `campaign_timing_observer` as an optional
callback. It is off when absent. Configure one callback for a run:

```yaml
callbacks:
  campaign_timing:
    type: campaign_timing_observer
    start_step: 101
    update_count: 50
    output_dir: cuda_graph_timing
```

`start_step` is the first measured optimizer update. `update_count` is the number
of measured updates. `output_dir` is relative to the run's `dump_folder` unless
it is absolute. Each rank creates one `rank-NNNNN.jsonl` file containing one
activation record and one timing-window record; existing rank files are never
overwritten. Records stay in memory during the window and are written only when
the window completes or the run ends.

## Optimizer-update graph boundary

The native trainer can build `campaign_cudagraph_step_boundary` as an optional
callback. It is off when absent and has no enable flag or trainer-selection
requirement:

```yaml
callbacks:
  campaign_cudagraph_step_boundary:
    type: campaign_cudagraph_step_boundary
```

At each native `batch_start`, the observer calls the public
`torch.compiler.cudagraph_mark_step_begin()` API once. This event precedes
zero-grad, batch fetching, compiled mask creation, and every forward/backward
microbatch in the optimizer update. Keep it after `campaign_timing_observer` in
the callback mapping so the timing observer records that boundary first. The
observer adds no tensor work, synchronization, copies, or communication; it
changes compiler iteration ownership. It rejects pipeline parallelism because
the selected boundary assumes the native non-PP update schedule.

Enabling the callback creates a CUDA Graph Trees lifetime candidate. Its
configuration does not establish graph replay, numerical equivalence,
continuation, or a speedup. Independent-run loss/norm bitwise differences remain
diagnostic and are not an eligibility gate. Target capture, changing-mask
lifetime, numerical, continuation, memory, and integrated performance checks
remain separate campaign gates.

The window opens at `batch_start(start_step)` and closes at
`batch_start(start_step + update_count)`. The training run must reach and execute
that following update to emit the closing boundary; it is unmeasured. A run that
ends before the boundary writes an `incomplete` record and raises. Invalid token
counts or missing boundaries never produce a throughput value.

The host submission spans use successive `batch_start` callback timestamps.
They include native work between them, including post-update metrics and any
checkpoint, validation, or profiler work at its normal cadence, and stop at the
closing callback before outstanding device work is drained. The primary
completed-window timer starts after a one-time pre-window device drain and rank
barrier, then stops after the closing boundary's device drain. It includes the
remaining asynchronous device tail but does not claim device-compute time. The
elapsed/validity MAX reduction and file persistence happen after this stop
timestamp and are excluded from the timer. Throughput is the sum of the
already-global Python valid-token count once per measured update divided by the
maximum completed-window elapsed time across the initialized world group. Token
counts are never reduced across ranks. This avoids extra per-update scalar
reads or collectives.

The CUDA events are recorded asynchronously on the current CUDA stream at the
same boundaries. Their spans are reported separately from both CPU-clock
measurements; they are not device-compute time and may omit independent
side-stream work.

For `hf_text` with `c4_test`, the source path shifts each fixed-length token row
into labels without padding masks. The observer records local/global batch size,
accumulation, sequence length, and expected full-length token count. Each
measured native `batch_end` Python scalar must match that expected count. A
tensor-valued count is rejected without calling `.item()`.

The activation record separates requested config from constructed markers. It
lists compiled model callables, selective checkpoint wrappers, dense FP8 linears,
grouped FP8 experts, HybridEP dispatchers, the actual HybridEP communication
stream setting, process-group rank membership, imported source paths, and
runtime versions. Constructed markers do not prove a compiled specialization,
FP8 custom-op call, or CUDA Graph replay executed. Compiled loss callable state
is not exposed through public `ObserverState`; use the native loss-compile
startup marker and runtime traces for that gate. The observer records allocator
current and peak counters at the window boundaries without resetting peaks;
these high-water counters are neither window-only peaks nor NVML device peaks.

This adds one host timestamp and one asynchronous CUDA event at each boundary
inside the selected window, two memory-counter samples, one pre-window CUDA
drain, one rank-alignment barrier, one post-window CUDA drain, one
elapsed/validity MAX collective after the stop timestamp, and a small JSONL
write per rank after completion. There is no per-update file flush or extra
per-update token collective. CPU double tests check callback behavior and
accounting; they do not establish CUDA, NCCL, event timing, allocator, or
production throughput behavior.

## HybridEP combine backward completion diagnostic

The opt-in `campaign_combine_bwd_completion` callback must be paired with
`torchtitan.experiments.cuda_graph_campaign.train_combine_bwd_completion`. That
entrypoint installs the campaign scan correction first and then wraps the
pinned module-global `_run_on_comm_stream` helper. Its callback rejects runs
outside the fresh, eight-rank EP8/PP1 graph-off 3B target. It writes one
rank-local JSONL file under `dump_folder/combine_bwd_completion` by default,
including update and microbatch boundaries and every selected invocation in
the diagnostic window. A zero-call microbatch inside that window invalidates
activation evidence, without assuming a fixed call count for observed batches.

On updates 21 through 60, after the native communication call has joined the
current compute stream, the adapter records one no-timing event on that stream
and waits for that event. This localizes the executed prefix across
`combine_bwd`; it can change overlap and temporary-object lifetime, may mask a
timing race, and does not certify peer progress or identify the originating
kernel. The ordinary `train` and `train_scan_fix` entrypoints do not install the
adapter. This diagnostic is not a performance comparison or a fault repair.

## Native phase-memory capture

Use `torchtitan.experiments.cuda_graph_campaign.train_phase_memory` only for an
authorized allocator observation run. It retains the HybridEP scan correction
and installs a task-local adapter before native Trainer construction. The
ordinary `train` and `train_scan_fix` entrypoints do not import or install the
adapter.

Set these environment variables to the same values on every rank before
launching the opt-in entrypoint:

| Variable | Purpose |
| --- | --- |
| `TORCHTITAN_PHASE_MEMORY_OUTPUT_DIR` | New shared directory for rank-local JSON records. |
| `TORCHTITAN_PHASE_MEMORY_RUN_ID` | Unique run or attempt identifier. |
| `TORCHTITAN_PHASE_MEMORY_SOURCE_ID` | Exact source snapshot or revision identifier. |
| `TORCHTITAN_PHASE_MEMORY_CONFIG_ID` | Resolved configuration identity, preferably its recorded hash. |

The output directory must not already contain `rank-*.json` records. Each rank
writes `rank-NNNNN.json` once after the native entry returns or raises. The
default phase boundaries expect native training metrics through update 500 and
group native snapshots through update 20 under `initial_through_20`; later
snapshots keep their actual reset-aligned window labels, such as
`steady_21_to_30`. A shorter run emits an `incomplete` record. A missing
capture, unexpected reset, missing required counter, changed native binding, or
metrics exception invalidates the affected coverage.

The adapter copies integer counters from the raw dictionary already passed to
`device_memory_usage`. The original helper, metrics methods, and peak resets run
once with their existing results and order. The adapter adds no `memory_stats`,
`mem_get_info`, synchronization, CUDA event, tensor-value read, or collective.
Validation records stay separate from training phase envelopes. Updates 21–23
are labeled as steady replay windows; they are not represented as capture-only
peaks. A small copy-time bracket follows the native stats return, so it does
not timestamp the allocator API itself. The native stats-to-reset logging gap
remains unsampled, and no pre-constructor allocator snapshot is taken.

If `campaign_timing_observer` is enabled and reaches its configured window, the
adapter brackets its existing opening timer and completed-close timer with UTC
and monotonic clocks. It copies the observer's existing opening and closing
memory samples without making new device reads. The closing sample is reported
separately with the allocator reset generation in effect at that boundary; it
is not merged into a native interval peak. Clock brackets expose uncertainty
for offline joining with device samples. Rank-local files do not establish
eight-rank completeness or GPU correctness until checked against the run's
source, configuration, device mapping, and sampler evidence.

The entrypoint also requires the existing `GIFT_HYBRID_EP_SCAN_RUNTIME_DIR`
setting and a fresh scan-fix attempt root; follow the
[scan-fix setup](backend_scan_fix.md). The rank JSON records the external
sampler source and expected cadence. Sample count, first/last sample times,
observed used-memory maximum, sample gap, and phase/boundary exclusions remain
null until the external CSV is joined offline.

## Direct numerical reference recorder

`campaign_numerical_probe` is an opt-in callback on `CampaignTrainer`. It writes
one native S0 inventory, then records either one (`updates: 1`, the default) or
ten uninterrupted native updates. The supported modes are:

| Mode | Required native state | Result |
| --- | --- | --- |
| `create_s0` | Fresh run; one real update; synchronous full step-1 checkpoint | Rank-local parameter, optimizer, buffer, router, RNG, and trainer-state inventory beside the native checkpoint. `output_dir` must be `<checkpoint.folder>/numerical_probe`. |
| `observe` | Fresh output run; full load of that same step-1 checkpoint with `load_only: true`; absolute stop step `1 + updates` | Verified loaded S0, then step-2 onward payloads for exactly `updates` native optimizer steps. |

The observer keeps the native clipping function, optimizer container, fused
AdamW instances, scheduler, model defaults, and checkpoint format in control.
It captures accumulated preclip gradients at `before_optimizer_step`, actual
postclip gradients in each inner AdamW pre-hook, and post-update parameters and
AdamW `exp_avg`, `exp_avg_sq`, and step tensors. It also records actual loss,
token, input, router, buffer, LR, and continuation metadata at their native
boundaries. It does not reconstruct CUDA optimizer arithmetic.

The native full checkpoint does not own the plain `aux_loss_scale` attribute
created by `wire_aux_loss`. The S0 and loaded inventories retain its identity,
layout, and hashes, while load comparison permits only this explicitly marked
`AuxLossTracker.weigh` value to be reconstructed. During observation the probe
wraps the native `weigh` call and records the actual per-part scale after it
returns and before the forward; declared splice modules must still reference
that native tracker tensor. Other direct attributes remain exact inventory
checks.

Each rank writes one compact tensor pack per update. Logical tensors are copied
to CPU one at a time, hashed with identity, dtype, shape, and phase, and read
back on CPU before the step report is marked complete. `None` gradients and
missing optimizer state remain distinct from zero tensors. The prepared-input
capture supports the pinned FlexAttention `BlockMask` fields; callable code and
closure observations are metadata only and do not establish reconstructibility.
An unsupported input or state shape fails the opt-in probe.

The shared live-file ceiling is 700 GiB with an additional 32 GiB free-space
reserve. Admission checks inspect the mount, inodes, free bytes, and a small
write/fsync before tensor payloads. Every rank reservation, input, tensor pack,
partial file, state file, and report contributes to the cap. The recorder leaves
failed partials for diagnosis and never prunes checkpoints or other output.
Native S0 admission runs after the native checkpoint save. A launch-side
preflight must account for the projected checkpoint size and reserve before
allowing that save; the observer does not establish this preflight.
Retain the native S0 and reference only for their named consumers, then follow
the task's recorded cleanup authorization.

Enabled observation is expensive: it persists five complete FP32 tensor
families per update, about 60.84 GiB globally, then reads that payload once on
CPU for integrity verification. The preclip inventory also hashes the unchanged
parameter and two AdamW moments, so the large-family device-to-host path is
about eight families (`8F`, roughly 97.35 GiB globally) per update before input,
buffer, scalar, and state reads. Ten updates write about 608.4 GiB and move
about 973.5 GiB through those large-family copies, before checkpoint and
metadata overhead. Blocking CUDA-to-CPU copies, the preclip/post-update device
drains, filesystem writes/fsyncs, and CPU readback are diagnostic overhead. The
direct recorder adds no H2D transfer, optimizer arithmetic, or comparison
tolerance. With the callback absent, the campaign trainer installs no numerical
hooks and performs no numerical-probe tensor copies or device synchronization.
These runs provide no throughput evidence.

The one-update run is the required target readiness probe. CPU boundary tests
exercise native CPU AdamW, accumulated backward, clipping, callback ordering,
payload integrity, the ten-update stop bound, cap refusal, and hook cleanup.
They do not establish FSDP2/NCCL, FP8 custom-op, compiled loss/model, HybridEP,
CUDA Graph, native checkpoint restore, or GPU readback behavior. Production
ten-update use stays gated on the target one-update probe, inventory and
readback review, measured storage/host peaks, and independent runtime-cost
review. Production A/A tolerances and graph-on numerical interpretation stay
unset; consult [continuation behavior](continuation.md) for the separate
checkpoint-continuation contract.
