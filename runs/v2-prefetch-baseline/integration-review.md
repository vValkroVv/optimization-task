# V2 final integration and runtime-cost review

Review scope: current working delta above `3b6b8928ec4b15a529d07ee99e672f2fc75ef73a`,
reusing the [prior review](../overnight-streaming-perf/v2-prefetch-review.md)
and [V2 contract](../overnight-streaming-perf/reference-coverage.md#v2-lifetime-and-state-findings).
No source edits, probes, training or publication were performed by this reviewer.

Finite acceptance: inspect the scoped loader/lifecycle integration and inherited
consumers; review bounded profiling and timing costs; reconcile the CPU timeout
with its controlled diagnosis; record distributed and production evidence gates.
This review does not extend to unrelated model code.

## Findings and dispositions

**No new material source or runner correctness defect established by inspection.**
The targeted CPU worker replay now passes in the native-matched harness. The
broader affected suite and distributed/production evidence remain separate gates.

- **CPU timeout: targeted disposition resolved; broader suite pending.** The initial focused suite
  stopped after six passes. The targeted `worker-1-false-deadline.log` stack
  blocks in `test_streaming_prefetch.py:176`, `control_first = next(iter(control))`,
  through `_iter_default` and Torch's worker-result queue. This precedes creation
  of the candidate iterator/producer at lines 177–178. The fork warning alone
  therefore does not establish a V2 producer deadlock. Preserve both this log
  and the initial `focused-pytest.log` under
  `artifacts/raw/cpu-checks/`; resolve the default-path/harness cause and rerun
  the affected cases before closing the CPU gate. Moving expected-loader creation
  alone would not explain this observed blocked frame.
  The diagnostic shell's96/96 Torch thread settings differ from native OMP1,
  but that does not establish causality. Separately, the real-MDS Gloo helper's
  world1 diagnostic exposed a tensor-IPC AF_UNIX path-length failure under its
  deeply nested task-local `TMPDIR`. The owners are checking a short task-local
  temporary path with native thread settings. The exact `[1-False]` case then
  passed in9.93s with OMP/MKL/OPENBLAS=1 and short task-local `TMPDIR`. It includes
  default0 loading followed by expected-loader construction/consumption while
  the candidate producer lives, resolving that concrete fork-interaction concern
  for the intended harness. Both thread settings and temporary path changed;
  the prior CPU log contains no AF_UNIX/OSError, so its root cause remains
  unassigned. Do not describe the failed runs as passes or infer a source defect.
  Passing evidence: `artifacts/raw/cpu-checks/worker-1-false-native-thread.log`;
  environment evidence: `native-thread-preflight.log` beside it. No producer or
  other training-source change was needed for this disposition.
- **Prior cleanup finding: resolved and retained.** `Trainer.close()` invokes
  loader shutdown in `finally`, including earlier callback/checkpointer/metrics
  failure. Wrapper cleanup is owner-only. Validation and batch generators close
  their iteration on early exit/error. The prior review's failure and targeted
  passing regressions remain applicable to the mechanically reused code.
- **Probe receipt wording: resolved.** Tiny-probe values 64
  forward/backward calls and two optimizer calls describe each rank's two
  updates, rather than the sum across eight ranks. Each target update still has
  32 calls and one optimizer call. The coordinator added `probe_count_scope`
  to the runner receipt and explicit per-rank wording to the experiment report.
  This changes no probe verdict.

## Integration conclusions

`num_prefetch_batches` is typed, rejects negatives and defaults to zero. The
disabled branch preserves synchronous iteration/state behavior and creates no
producer, ready queue, scalar snapshot per batch or restore barriers. Configuration
goldens gain only the zero field; the canonical workload configuration and
`Trainer.train_step` are unchanged.

The consumer constructs and primes each finite raw epoch. The producer receives
only that epoch's remainder under copied Streaming ContextVars; it cannot advance
the epoch or initiate its collective work. FIFO delivery commits the associated
scalar cursor before yielding. Saving pauses in-flight source advancement and
copies committed state under the existing `dp_rank_0` key. Loading joins the old
producer before mutation, invalidates old iterators, retains explicit DataWorld
barriers, and resets the shared older-epoch cursor only on local rank zero.
Timeout keeps producer ownership rather than claiming successful shutdown.

Unchanged text collation/FIM and Trainer consumers retain batch contents/dtypes,
32-batch collection, callbacks, int64 valid-token counting, global normalization,
device transfers, forward/backward, optimizer and EMA behavior. No model,
precision, SAC, compilation, communication-wrapper or loss change is included.
Actual distributed replay remains an execution gate; this source reasoning does
not replace it.

The bounded profiler uses the existing native lifecycle. On selected ranks 0/1,
the third complete forward/backward returns before a CUDA drain and closing of
the prefix scope; native profiler exit/export then clears `torch_profiler` to
prevent duplicate exit/export. The remaining 29 microbatches and optimizer run
normally. Other ranks train normally without constructing a profiler. The parser
checks the complete prefix, token reduction, three CPU/GPU microbatch projections
and backward coverage. The
[runner receipt](runner-verification.json) records normal and injected-error
eight-rank Linear/NCCL probes, preserved failures and focused checks. These are
runner diagnostics, not GIFT performance or numerical evidence.

## Runtime costs and limits

| Path | Added work and consequence |
| --- | --- |
| Enabled ordinary input | One producer/rank; Condition push/pop and notifications; one Mosaic scalar-state snapshot/source batch, including shared-memory reads and JSON/Python work. GIL, IPC/storage lifetime and CPU contention can offset hidden input wait. |
| CPU memory | Depth32 retains at most32 ready batches, 5 MiB/rank for this collator. This excludes active source/consumer items, the outer generator's retained first item, Torch IPC/worker queues and the Trainer's existing32-batch accumulation list. It is not a total RSS bound. |
| Epoch boundary | Consumer-owned raw iterator reset/first batch remains synchronous. Producer start/join occurs each finite epoch; the56-local-batch fixture prevents continuous cross-epoch filling. |
| Checkpoint/lifecycle | Pause waits for an active read; load/close joins with the existing30-second producer deadline. Load adds two owner-group Gloo barriers. Scalar-only checkpoint bytes are preserved; there is no model/optimizer export. |
| Device and communication | No added model kernels, H2D copies, device-value reads, per-batch collectives or compile boundaries. Existing transfers and global token reduction remain on the consumer. |
| Timed runner | No new diagnostic wrappers or device synchronization within the timed loop. Host process-tree RSS/HWM reads occur before the start timestamp and after the end timestamp; inherited boundary CUDA drains remain. RSS sums can double-count shared pages; HWM is process lifetime and two snapshots do not establish window peak. |
| Trace runner | Profile-only queue snapshots/scopes, Python/GIL activity, third-microbatch drain and export perturb this diagnostic run. Do not derive TPS from it. |

Host sampling before the timing boundary also gives a live producer a short
extra filling interval. Retain matched sampling and the full measured update
window when interpreting the provisional comparison. No speedup or memory-saving
claim follows from source inspection or the tiny probes.

## Acceptance gates

At this review point, affected CPU-suite completion and
the actual eight-owner real-MDS Gloo/DCP gate are pending. Native GIFT training,
matched unprofiled depth0/depth32 TPS and paired memory observations, comparable
bounded production traces, and final owned-process/GPU release are also open.
The [experiment report](report.md) owns those executed results and identities.
Do not mark V2 production accepted or a performance gain established until the
required evidence is complete.

**Review verdict:** no demonstrated source/runtime blocker remains in this
scoped inspection and controlled-case reconciliation. Source/cost review is
complete. The coordinator owns reconciliation of the still-open execution gates
in the experiment report before overall acceptance; this verdict grants neither
production acceptance nor a speedup claim. A later material source correction
requires only review of that correction and its affected behavior.
