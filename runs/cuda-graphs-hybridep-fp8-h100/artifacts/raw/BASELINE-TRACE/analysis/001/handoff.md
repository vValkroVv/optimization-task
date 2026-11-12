# Baseline trace analysis, attempt 002

## Result

All eight completed traces show the native forward/backward and optimizer path,
compiled regions, HybridEP forward/backward, FP8-related kernels, attention and
chunked CE kernels. No `cudaGraphLaunch` or `cuGraphLaunch` API event occurs on
any rank. This establishes replay absence in this baseline only.

Native updates **22 and 23** are the two complete update windows used below.
Each rank/update has four native `fwd_bwd` spans, one `optim` span and 68 calls
each to `hybridep::dispatch`, `combine`, `dispatch_bwd`, and `combine_bwd`
(17 sparse blocks × four microbatches). Native update 24 is also recorded by
the profiler; its native full-step span includes trace export, so it is excluded
from full-step timing. Rank0's structured spans independently show four
forward/backward calls and one optimizer call for update24.

## Evidence and reproduction

- Input: `attempts/002/native/profiling/traces/iteration_24/rank{0..7}_trace.json`.
- `summary.json` contains every original JSON SHA256, all-rank event/feature
  inventories and the native update summaries; `rank{0..7}.json` are its slices.
- `compute-gaps-rank0.json` contains 35 representative largest compute gaps,
  correlated next launches, enclosing operators and native span timestamps.
- Commands from repository root:

  ```sh
  nice -n 10 python runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-TRACE/analysis/001/analyze.py runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-TRACE/attempts/002/native runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-TRACE/analysis/001
  nice -n 10 python runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-TRACE/analysis/001/inspect_compute_gaps.py runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-TRACE/attempts/002/native runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-TRACE/analysis/001/compute-gaps-rank0.json
  ```

The first parser pass was stopped after finding duplicate CPU/GPU
`ProfilerStep` markers; it produced preliminary rank slices only. The completed
pass replaced those slices, uses native epoch timestamps and dominant marker
overlap, and hashes original trace bytes while reading. `ProfilerStep#21` maps
to native update22, not update21. Original traces remain unchanged.

Verification passed: interval union/intersection examples; eight ranks;
native IDs 22/23; four microbatches and one native optimizer per selected
rank/update; four HybridEP operator counts of 68; required feature inventories;
zero graph launches. Two complete updates meet the requested 2–3-update scope.

## Timer definitions and observations

Native full-step wall clock comes from paired structured `step_start/end`
timestamps, converted to Kineto time using `baseTimeNanoseconds`.
GPU busy is the interval union of kernels, copies and memset across all streams,
clipped to that window. Host API, launch and synchronization/query times are
separate interval unions. They overlap device execution and each other; their
totals must not be added. Kernel-family sums are inventories, not elapsed time.
An executing HybridEP/NCCL kernel can include peer waits or spins; transfer and
wait fractions are not separable from its name and span alone.

Across 16 selected rank/update windows:

| Profile-only scope | Range, ms |
| --- | ---: |
| Native full-step window | 3283.130–3331.349 |
| All-stream device idle | 56.267–70.887 |
| Host runtime/driver API union | 2142.190–2232.660 |
| Host launch API union | 913.940–970.543 |
| Host synchronization/query union | 1177.726–1210.648 |
| Synchronization inside forward `hybridep::dispatch` | 366.439–387.780 |
| The same dispatch synchronization overlapping all-stream idle | 8.512–9.132 |
| Device copy/memset union | 10.805–11.492 |

These are mechanism observations under an invasive CPU/CUDA profiler, not
speed measurements, removable costs or speedup bounds. In particular,
all-stream idle cannot bound graph benefit: persistent communication can mask
compute gaps. Rank0/update22 has 1108.768 ms between noncommunication kernels
across compute-bearing streams, of which 1047.323 ms overlaps communication.
This classification conservatively calls unknown kernels compute; it does not
prove that every classified kernel does useful arithmetic.

## Representative dependencies and lane decisions

The largest rank0/update22 compute gaps are communication-associated:

| Compute gap, ms | Communication overlap, ms | Next kernel launch before gap start, ms |
| ---: | ---: | ---: |
| 15.554 | 15.539 | 82.638 |
| 15.497 | 15.485 | 87.330 |
| 15.393 | 15.380 | 81.747 |

Each ends at the backward compiled gather/scatter kernel
`triton_poi_fused_clamp_min_gather_lt_masked_fill_mul_new_zeros_scatter_add_view_8`
on stream7, with its correlated `cuLaunchKernel` already queued. These gaps
cannot be attributed to a late host submission of that kernel. The trace slice
does not identify the precise peer/event dependency, so it does not claim that
shortening one overlapping collective would shorten the update.

The largest all-stream host-late gaps on rank0 are submillisecond launches of
label comparison and BF16/fp32 parameter-copy kernels. This does not establish
a material uncovered attention/shared-FFN/chunked-loss capture region. The
largest compute-gap slice likewise supplies no such region. **Lane B has no
positive trace-supported target from this baseline**; this is not a universal
incompatibility finding and does not replace observing Lane A's actual coverage.

**Lane A remains the next finite capture question:** does toggling the verified
compiler setting create replay in the unchanged native path, and which regions
does it cover or skip? Baseline absence alone does not answer compatibility.

**Lane C's host-count barrier is active, but benefit is unproven.** Forward
dispatch contains hundreds of milliseconds of host synchronization per profiled
update, mostly while device work continues. The active source explicitly
requires blocking host count discovery to size the exact result. A legal
H100-safe count/capacity route and material remaining critical-path blocker
after A are separate gates. The gross wait is not sufficient justification
for implementing a prerequisite or estimating its benefit.

## Source ties and limitations

- `gift/torchtitan/trainer.py:612–701`: native forward, loss, backward;
  `:735–791`: four-microbatch consumer; `:793–815`: clipping, native optimizer
  and scheduler; `:933–974`: native step, checkpoint/validation and profiler.
- `gift/torchtitan/tools/profiler.py:240–281`: native schedule and CPU/CUDA
  recording; only `record_shapes=True`, without Python stacks.
- `gift/torchtitan/observability/structured_logger/structured_logging.py:292–405`:
  native wall-clock spans are structured records, not Kineto annotations.
- `gift/torchtitan/distributed/deepep/hybridep.py:268–275`: exact-count blocking
  contract; `:321–341`: dispatch and join; `:362–378`: pinned host counts and
  device return; `:556–638`: genuine backward custom operators.
- `gift/torchtitan/models/common/moe/dispatch/hybridep.py:123–141`: dispatcher
  passes live per-step routes and configured nonblocking factor to that path.

Sparse FP8 is directly evidenced by `deep_gemm::sm90_fp8_gemm_*` kernels;
FP8 transpose/scaling operators and dense `nvjet_sm90_*` kernels are present.
Dense FP8 dtype activation should be combined with the campaign's resolved
configuration/backend source proof, rather than inferred from `nvjet` alone.
Compiled attention forward/backward and fused chunked log-softmax/NLL backward
kernels are present on every rank. The trace has no explicit SAC annotation;
retain source/config activation proof instead of interpreting marker absence
as SAC absence. No Python-stack or GPU peer-wait decomposition is available.
No production source, GPU process or original trace was changed by analysis.
