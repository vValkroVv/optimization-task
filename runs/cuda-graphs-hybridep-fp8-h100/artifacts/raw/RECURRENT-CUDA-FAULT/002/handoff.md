# Recurrent CUDA 715 at update 48: next diagnostic decision

## Decision and finite acceptance

Retain the existing post-`combine_bwd` completion-prefix experiment as the single
next diagnostic, extended to updates 21–60 with native stop 60. It can distinguish
a fault reported by a joined-prefix checkpoint from a fault reported between
successful checkpoints. It cannot identify an originating kernel. **It is not
currently executable as a configuration-only extension:** the implemented code
fixes the window at 21–40 and explicitly rejects `training.steps != 40`.

This delivery completes four read-only/documentation items: reconcile the new
fault, inspect the adapter's actual eligibility, define one bounded protocol,
and specify the missing quantize-binary provenance. No source change, GPU query,
GPU run, installation, cleanup, or root-document edit occurred. The coordinator
has confirmed source-edit scope for task-local `gift` campaign
diagnostics and will assign that work separately; execution still needs its
explicit release. The existing 40-update diagnostic could still test the earlier
22/33 failure horizons, but would not meet the requested step-48 coverage.

## Reconciled evidence and affected contract

The execution owner's supplied result is reused: graph-off C0a failed with 715
in update 48's first microbatch after all eight ranks completed 47 optimizer
updates / 188 forward-backward calls. Rank 3 first recorded `fwd_bwd_error` at
03:03:53.706 UTC; its later watchdog event-query report is propagation evidence.
The primary extraction owner owns complete counts, log integrity and cleanup.

The targeted rank-3 elastic `error.json` was inspected. Its stack is
`loss.backward` → AOTAutograd/Inductor → cached
`cb3phbwhwqlg3hsapbyxedotxx7g54mlajsocxjdeioufefknbd7.py:4345` →
`torch.ops.cudacherry.quantize_blk128x128.default(primals_28, None, True, False,
False, 0.0)`. The cache callable shows shape `(2048, 1536)` and this quantize
inside recomputation before its `hybridep.dispatch_bwd` at line 4529 and
`hybridep.combine_bwd` at line 4680. It is therefore **not** the same callable's
post-combine Flex/RoPE interval. Asynchronous reporting does not establish that
quantize originated the fault, nor that the historical and new failures share
one cause. Successful earlier-layer checkpoints in that microbatch may still
bound a prior executed prefix; a failure before its first checkpoint cannot.

Together with original graph-off update 22 and corrected graph-off update 33,
this establishes recurrence at different horizons, ranks and observation sites.
The scan-bound correction remains a justified correction with verified prior
SASS provenance, without demonstrated causal repair. The new result supplies
no reason to remove FP8, SAC, model/loss compilation, unsafe-GEMM path, marker,
all-18 graph exclusions, or HybridEP. Graph profitability and baseline numerical
variability are separate questions.

## Exact source/configuration blocker

At inspected HEAD `446431dfa191ef0300f116257ae918bc85e38c06`, branch `main`:

- `gift/torchtitan/experiments/cuda_graph_campaign/combine_bwd_completion.py`
  sets `_FIRST_UPDATE = 21`, `_LAST_UPDATE = 40`; `_validate_target_runtime`
  additionally requires `int(training.steps) == 40`.
- Its callback schema exposes only `output_dir`. Neither YAML nor CLI has a
  supported window/stop override. A 60/61-step launch fails `fit_start`; changing
  only `_LAST_UPDATE` also leaves that guard wrong. Do not monkeypatch constants,
  spoof native step/configuration, or weaken source/selector guards in a launcher.
- The implemented registered-body selector remains module + qualified lambda
  name + physical source path + line 651. HybridEP SHA-256 remains pinned to
  `12b1921372f265ede5cca4974f5867f954f84862cc257ff8e02bb7ecafa0814d`.

The smallest proposed source adjustment is a fixed diagnostic update range
21–60 and fixed native stop 60, keeping the existing callback schema/entrypoint
and exact HybridEP selector. Make the stop guard use the canonical last-update
constant, update maintained diagnostic documentation, and adjust the affected
window/target-runtime/complete-coverage tests to 20/21/60/61 and 160 active
microbatches. Demonstrate that 60 is accepted and 40/61 are rejected for this
fixed diagnostic. Reuse unchanged CPU interception/liveness/error evidence;
run the affected guarded CPU module and scoped lint, then review this narrow
delta and freeze a new integrated snapshot. This is proposed work, not performed
by this delivery; the coordinator owns the separately authorized source work. Do not claim the prior 38-pass review covers
newly changed bytes.

Exact affected source symbols are `_LAST_UPDATE` and
`_validate_target_runtime`; `_FIRST_UPDATE` stays 21. In
`gift/tests/unit_tests/test_campaign_combine_bwd_completion.py`, update
`_valid_observer_state`'s fixed stop and these four affected tests:
`test_observer_records_fixed_boundaries_and_waits_only_in_window`,
`test_observer_requires_runtime_call_activation_for_complete_result`,
`test_observer_invalidates_active_microbatch_without_selected_calls`, and
`test_observer_completes_when_every_active_microbatch_is_observed`.
The last test's one-selected-call-per-microbatch double must now produce exactly
160 calls/completions; this is a fixture count, not a predicted GPU layer count.
Add the direct fixed-native-stop acceptance/rejection behavior to this module.
Update only the diagnostic section of
`gift/torchtitan/experiments/cuda_graph_campaign/README.md`; no schema,
entrypoint, HybridEP body or production-preset change is needed.

## Single conditional probe protocol

After that scoped source gate and owner release, use exactly one fresh attempt
with no restart/retry (`torchrun --max-restarts=0`), the reviewed detached
launcher/preflight/timeout and precise task-process cleanup. No matrix, profiler,
launch-blocking override, fault sweep or separate provenance GPU run.

1. Preserve the exact C0a production/runtime inputs: 3B MoE/FlexAttention,
   HybridEP EP8/FSDP2 on the same verified one-node eight-H100 NVLink topology,
   FP8, selective AC, model/loss Inductor compilation, batch 8/256, S4096,
   accumulation 4, seed 42, existing data/cache/allocator route, marker,
   unsafe-GEMM adapter, corrected isolated per-rank scan runtime, partition true,
   `TORCHINDUCTOR_CUDAGRAPHS=0`, `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` and all
   18 blocks graph-excluded. Keep scheduler horizon 30000 and checkpoint false,
   fresh native step zero. No continuation claim is made.
2. Retain the task-local compiler-entry assertions/records for commonmix false,
   cudagraph false and partition true; change its delegated main from
   `train_scan_fix` to
   `torchtitan.experiments.cuda_graph_campaign.train_combine_bwd_completion.main`.
   That entry installs scan correction first, then the adapter, runs native main,
   and restores the adapter in `finally`.
3. Import the same pinned preset; retain `campaign_step_boundary` and the exact
   all-18 exclusion list. Replace the `campaign_timing` callback with exactly one
   `combine_bwd_completion: {type: campaign_combine_bwd_completion,
   output_dir: combine_bwd_completion}`. Retain profiling/snapshot false. The old
   start-21/count-50 timing callback needs its update-71 boundary and must be absent;
   its timings would also be invalid under per-call synchronization.
4. Launch native `--training.steps 60 --lr_scheduler.total_steps 30000
   --debug.seed 42 --checkpoint.enable false` to a new dump folder. Stop on first
   fault or native completion of update 60; use the inherited bounded wall timeout
   for a hang. Stop 61 is unnecessary for this completion observer and unsupported
   by the proposed fixed-60 contract. No later performance-matrix variant is released.
5. Preserve rank-zero resolved configuration/diff, all-rank compiler options,
   scan/source markers, physical topology/groups, structured logs, first elastic
   exception, callback JSONL, external memory series and precise exit/cleanup
   receipts. Missing activation, malformed/truncated logs, OOM, timeout/hang or
   instrumentation error makes the diagnostic inconclusive rather than a kernel
   attribution. Do not query CUDA inside sticky-error cleanup.

On a successful full attempt each of eight rank logs must independently show:
activation with window 21–60; updates 21–60 started and optimizer callbacks
completed; four native microbatches per active update (160 total), each with
positive selected-call count; no zero-call microbatch; strictly increasing local
ordinals; one enter and one complete per selected active call; aggregate
`active_completed_checkpoints == active_selected_calls > 0`; and complete
diagnostic result. Export each rank's actual per-update/per-microbatch counts and
totals. Do not assume 18 calls from the decoder count or require ordinal equality
across ranks without evidence. For a fault, preserve partial counters, unmatched
enter/error stage, and last completed ordinal on every rank; do not require a
successful fit-end record after a fault.

## Minimal quantize provenance in the same attempt

GEMM `.so` provenance alone does not identify the quantize provider. Inspected
installed `cudacherry/quantization/blockwise.py` calls `load_family("quantize")`
before dispatching the quantize op. `_loader.py` chooses a prebuilt module or JIT
fallback and caches it; the local installation contains
`quantization/_C_quantize.cpython-312-x86_64-linux-gnu.so`. These are static
observations, not proof of what failed rank 3 loaded.

Use the task-owned external monitor once after successful native update 1 is
observed in contiguous structured logs. Obtain the eight exact rank PIDs from
this attempt's torchrun/structured identity, then save rank/PID/time/update,
matching `/proc/<PID>/maps` rows (path/device/inode), and SHA-256/size/stat identity
for every unique actual mapped cudacherry quantize/GEMM binary. Include prebuilt
and JIT paths; match the quantize and GEMM family names rather than assuming the
A006 GEMM `.so` also provides quantize. Retain the installed loader/wrapper path
and hash as static provenance. Confirm each inspected PID still belongs to the
same rank/attempt when collecting the receipt. These are CPU reads by the
external monitor, with no training-source/package change or added in-process
callback, barrier, CUDA wait, extension import/reload, forced JIT or tensor read.

All eight ranks need mapping receipts, including explicit absent/unreadable or
deleted files. Bind each file hash to mapping/stat identity; flag a changed
file/inode or inaccessible mapping instead of claiming binary certainty. Record
that no live `sys.modules` snapshot was obtained if the current runner does not
already expose one: external maps plus matched binary hashes establish loaded
file provenance without inventing a module receipt. An optional existing
`sys.modules` receipt can corroborate package/JIT module identity; do not add a
new source hook solely to obtain it. No separate GPU run is needed.

Inspect the monitor's generated PID-to-rank mapping and receipt parser on a
representative existing mapping record before release. The one-time CPU maps
read, matched binary hash and small receipt write are the intended added costs.
They do not locate a CUDA fault or prove an extension's registered operator body.
Compare receipts to A006's recorded actual GEMM identity after the run, without
presuming a bad binary or unsupported stack.

## Outcomes, runtime costs and remaining limits

| Outcome | Decision changed by this attempt |
| --- | --- |
| Native/helper, event-record or event-wait error | Investigate the prefix since the last successful boundary, including prior operations, lifetimes and traps. It does not isolate combine as cause. |
| Successful checkpoint followed by quantize/other failure before the next checkpoint | Prioritize the intervening submitted path and retained/reused inputs. Use actual cached-call order; do not relabel this as post-combine Flex/RoPE automatically. Earlier silent corruption and unrelated streams remain possible. |
| Failure before any selected call in the failing microbatch | No intra-microbatch combine boundary was established; retain only the previous prefix and the current recomputation observation. The same-call combine is not implicated by that order. |
| All-rank active coverage and no fault through 60 | Nonreproduction under synchronization/lifetime perturbation; no repair or causal exclusion. |
| Missing hits, source mismatch, hook error, OOM or hang | Instrumentation/progress evidence is insufficient; preserve first failure and stop. No automatic follow-up. |

The adapter calls the original helper once, after which the inherited compute
stream joins communication. It retains callable/closure/inputs/output until its
no-timing event finishes. It adds no tensor compute/copy/conversion/collective,
but adds a Python selector/counters on helper calls, one event record and host
wait plus flushed logs per active selected call. This reduces overlap and changes
allocator reuse and buffer/handle lifetime, potentially masking races. Native
peer progress is source-supported for matching same-node calls, not certified
for the closed extension. Extending 20 active updates to 40 doubles active
diagnostic call opportunities at equal per-update work; measured cost is unknown.
Retain external memory usage and any existing allocated/reserved peak reports
with scope/units; missing allocator peaks remain unavailable. Do not estimate
device, host, wait, copy/communication or integrated timing by subtraction, or
publish throughput/peak-memory improvements from this diagnostic.

Open gates are the authorized narrow extension implementation/check/review,
new integrated source pin, external provenance-monitor preparation, owner
compute release and preflight, actual all-eight-rank target activation/progress
and outcome evidence.
Originating PC/kernel, whether the failures share a cause, GPU numerical effects
of altered scheduling, sustained production correctness and fault repair remain
unknown. The prior 38 guarded CPU passes and independent integration review are
reused for unchanged behavior; this delivery adds no GPU evidence.
