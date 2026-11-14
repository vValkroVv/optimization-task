# BF16 expert cache: bounded trace mechanism result

## Result and acceptance

**PASS for the affected mechanism in the eight-GPU crop.** Both ranks show one
FP32-to-BF16 preparation per expert group in microbatch 1, no repeated cast in
microbatches 2/3, and no size-one input-to-output local copy. Existing expert
FP8 weight quantization still executes at the same observed frequency and grids.
All eight ranks' diagnostic counters establish reuse throughout all 32
microbatches and across all 21 optimizer boundaries.

Finite acceptance was: verify capture/config/activation; import complete device
events; attribute every eligible expert group's casts and local copies; inspect
consumer waits, overlap and FP8 quantization; reconcile full-accumulation counters;
retain identities, commands, queries and limits. These items are complete within
the trace's stated observability. Exact device wait attribution remains unavailable
because this profiler omits event handles; this is not a measured zero wait.

This report establishes execution mechanism. It does not convert device work
removed into optimizer-step savings or establish numerical equivalence.

## Identity and coverage

The control is the saved accepted V2 `v2-long-epoch-candidate-trace-001`, not a
fresh run. The candidate is `bf16-cache-trace-001`: 21 total optimizer updates,
20 warmup, update 21 first three complete F/B microbatches on ranks 0/1.
Rank sidecars confirm collection stopped at three while all eight ranks completed
32 microbatches and the optimizer. Diagnostic counters were enabled here and
are absent from the timing run. The saved control has the same bounded prefix,
update number, warmup and V2 prefetch/epoch configuration.

The candidate's actual native `resolved_config.yaml` confirms prefetch 32,
epoch 51,200, microbatch 2, global batch 512, sequence 4096 and
`parallelism.gfsdp_reuse_bf16_expert_weights: true`; FP8, compile, selective AC
and MTP remain in the resolved configuration. Complete launch argv and hashes
are retained in [the machine-readable result](trace-analysis.json).
The native owner separately verifies broader run/config acceptance.

Candidate source HEAD is `657e77a4f8ddb116b75925e5671b00642a0e4d32` with frozen
tracked-plus-untracked digest
`3dfde3ee2b0d9b207d042522bbc4882d4bcbaec6d07250efdde24ae6dc24e42c`.
The historical control source identity and exact compatible performance window
are owned by [execution readiness](execution-readiness.md#saved-v2-control-and-diagnostic-evidence).
It has a different source/runtime identity; these captures are not contemporary
paired performance trials.

Canonical paths share these prefixes:

- Control: `runs/v2-prefetch-baseline/artifacts/raw/v2-long-epoch-candidate-trace-001/native/profiling/traces/iteration_21/`.
- Candidate: `runs/overnight-streaming-perf/artifacts/raw/bf16-cache-trace-001/native/profiling/traces/iteration_21/`.

| Canonical file | SHA256 |
| --- | --- |
| Control `rank0_trace.json.gz` | `196806cee1fe857e97e4d80f3b7aa5961e323860f5fa1274d3e4b59a6f522c94` |
| Control `rank1_trace.json.gz` | `6a8bcd4624f06459656db535618ddc2ae0c21f6167e2d74f5276c044de08cfc7` |
| Candidate `rank0_trace.json` | `fdcf72769a9ddbad51db80b0bf7663784f7bbf4c72971e3ee8d7925cfc800203` |
| Candidate `rank1_trace.json` | `3594219f6ea8282cd85e963ea7d5db2b51a2dfc52732e91de08f59fa3b419902` |

## Import and attribution

Actual MCP Perfetto imported repaired display copies using the existing
`runs/overnight-streaming-perf/monitorfree-perfetto-prepare.py`. The helper
preserves timestamps, durations, names and `args.stream`; synthetic display tids
are not CUDA streams. Canonical bytes remain unchanged. Candidate plain JSON was
losslessly gzip-copied into this analysis's ignored artifact directory before
running the gzip-only helper. Preparation receipts retain both hashes/census.

| Variant / rank | Kernels | Memcopies | Memsets | Linked device events | Import errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control 0 | 8,795 | 274 | 861 | 9,930 / 9,930 | 0 |
| Control 1 | 8,795 | 274 | 861 | 9,930 / 9,930 | 0 |
| Candidate 0 | 8,781 | 232 | 861 | 9,874 / 9,874 | 0 |
| Candidate 1 | 8,781 | 232 | 861 | 9,874 / 9,874 | 0 |

Imported counts and duration sums match the raw census. Flow links join device
events to `cuda_runtime` and `cuda_driver` submissions, then to same-thread
`GFSDP::all_gather (...)` regions. Independent raw correlation attribution has
exactly one host match for every device event and agrees with MCP's per-group
counts. The raw interval calculations retain source microsecond precision;
sub-microsecond union/sum rounding differences are expected.

All seven actual enabled FQNs are `layers.1[efsdp]` through `layers.7[efsdp]`.
The initial pre-lazy-init counter labels `layers.N[group_1]` resolve to those FQNs
on the first update; group counts were derived from sidecars and trace regions.
Every one of these groups has six gather regions: forward and backward for each
of the three microbatches. Per group, on **both** ranks:

| Observable | Control | Candidate |
| --- | ---: | ---: |
| FP32-to-BF16 cast kernels | 3: one per microbatch | 1: first microbatch only |
| Size-one DtoD copies | 6: one per gather | 0 |
| Casts in microbatches 2/3 | 2 | 0 |

All affected operations execute on actual CUDA stream 24. Each control gather's
`record_param_comms` identifies `mesh_efsdp`, BFloat16, group size 1 and
301,989,888 input/output elements (603,979,776 bytes per local copy). Attribution
uses these regions and launch links, rather than a global copy histogram.
No other device operation appears inside the affected gather regions.

## Timing scopes and overlap

All times below are milliseconds, independently measured per rank over the
sampled prefix. Host gather spans include Python/dispatcher/submission work and
scheduling, not CPU utilization. Device sums and interval unions are distinct;
these particular affected events serialize on stream 24, so their union equals
their sum. Whole-model kernel/copy/EP/NCCL family sums and unions, per-microbatch
spans and region/launch details are retained in the JSON result.

| Scope | Control rank 0 | Candidate rank 0 | Control rank 1 | Candidate rank 1 |
| --- | ---: | ---: | ---: | ---: |
| Affected gather host-span sum | 6.432 | 2.048 | 6.127 | 2.061 |
| Expert cast device sum | 12.350 | 4.118 | 12.338 | 4.126 |
| Expert localcopy device sum | 17.005 | 0 | 17.005 | 0 |
| Affected device interval union | 29.355 | 4.118 | 29.343 | 4.126 |
| Affected union overlapping other kernels | 11.356 | 0.179 | 11.124 | 0.178 |
| Affected union overlapping EP/NCCL | 0.799 | 0 | 1.588 | 0 |
| Three-F/B device first-start to final-end | 1293.499 | 1269.508 | 1293.678 | 1269.687 |
| Three-F/B CPU-start to final CPU/device completion | 1293.713 | 1269.710 | 1293.873 | 1269.897 |

Other-kernel overlap excludes the expert preparation events themselves and all
named EP/NCCL/copy/memset events. It may still include unclassified device waits.
EP/NCCL kernel time can contain peer/spin waiting. Overlap classes can coincide;
their durations are not added or subtracted to claim critical-path savings.
The enclosing span includes unrelated model computation and communication and
reflects profiling/host scheduling. These independent historical captures cannot
attribute its entire change to preparation.

### Consumer waits

All 42 expert pre-forward/pre-backward regions retain `cudaStreamWaitEvent`:
42 submissions on each rank in each variant. Host API sums are 0.04018/0.04862 ms
(control/candidate rank 0) and 0.04085/0.04875 ms (rank 1). No host
`cudaEventSynchronize`, `cudaStreamSynchronize` or `cudaEventQuery` appears inside
those expert consumer regions. This does not describe the rest of the program.

The preparation regions add explicit cache lifetime ordering: wait submissions
increase from 42 to 98 per rank. Their host sums are 0.03229/0.07046 ms (rank 0)
and 0.03242/0.07101 ms (rank 1). Gather launch API time and cast queue delays are
recorded separately in raw device/launch rows. No per-consumer wait handle is
exported, so exact producer-event joins, device wait duration and exposed
consumer-stream idle attributable to a particular refresh are unavailable.

The inspected source provides the ordering contract:
`GFSDPParamGroup.unshard` waits for preceding readers/writes before refresh,
records the cache-ready event, and `AllGatherResult.wait` orders consumer use.
`reshard` records reader completion. Existing shared-buffer release events remain.
The trace demonstrates retained wait submissions and stream overlap; correctness
of the lifetime contract also depends on the separately executed focused checks.

## Complete accumulation and FP8 consumers

All eight `measurement/rank-XX.json` sidecars independently pass the same counter
assertions at **every optimizer boundary 1–21**, for every enabled group:

- Per update: refreshes +1, hits +63, direct uses +64, invalidations +1,
  generation +1. Two gathers per microbatch account for 64 uses over 32 microbatches.
- At update 21: refreshes 21, hits 1323, direct uses 1344, invalidations 21,
  generation 21. The first consumer of the next generation must refresh.
- Every noneligible group's cache counters and generation remain zero.

These counters cover all groups/updates outside the three-microbatch trace cap;
they supplement the device trace rather than extrapolate its first three batches.
The diagnostic profiling wrapper leaves generic `microbatch_calls_by_step` empty: it
calls `note_profile_microbatch_complete` instead of the generic optional counting
hook. `profile_capture.microbatch_calls=32` and `optimizer_step_calls=1` directly
verify update 21 on every rank. The earlier updates have per-group 64-use
forward/backward gather totals and one refresh per optimizer boundary, consistent
with the unchanged 32-microbatch native accumulation path; they do not provide
independent generic per-microbatch hook records for updates 1–20. This retained
instrumentation limitation does not invalidate update-21 coverage or cache counters.

The active FP8 consumer is `hybridep.py` dispatch's `quantize_expert_weight`
route, transported through `FP8DispatchState.fp8_expert_weights` to
`gmm.py::_fp8_experts_forward`. GMM's `_quantize_local_expert_weights` is the
fallback when no prequantized weights are supplied. The dispatch, GMM and
`quantize_expert_weights.py` bytes have no task diff; their SHA256s are recorded.

All four traces contain exactly **78 `_block_quantization_kernel` launches**
inside **39 `hybridep::dispatch` calls**, two weight tensors per call:

| Phase | Grid | Control count per rank | Candidate count per rank |
| --- | --- | ---: | ---: |
| Original forward | `[384,32,1]` | 21 | 21 |
| Original forward | `[512,12,1]` | 21 | 21 |
| Backward/SAC recomputation | `[384,32,1]` | 18 | 18 |
| Backward/SAC recomputation | `[512,12,1]` | 18 | 18 |

Forward/recompute attribution follows the enclosing `CompiledFunction` versus
`CompiledFunctionBackward`, not an assumed universal recompute count. There is
no observed skipped expert quantization or changed launch grid. Launch counts
cannot prove equality of values, scale metadata or gradients; those belong to
the focused correctness evidence.

## Reproduction and retained evidence

Ignored analysis artifacts reside in
`runs/bf16-expert-cache/artifacts/raw/trace-analysis/`. The directory contains
prepared gzip files/receipts, `control{0,1}-mcp.json`, `candidate-mcp.json`, and
four raw-analysis JSON files. MCP records retain exact executed SQL/results and
trace handles. `analyze.py` implements bounded raw launch/region attribution and
interval unions/intersections; `summarize.py` adds quantization ancestry,
all-rank counter assertions, source/config identities and the maintained JSON.
No new dependency, GPU run or source change was made for analysis.

```bash
python3 runs/overnight-streaming-perf/monitorfree-perfetto-prepare.py CANONICAL_GZIP PREPARED_GZIP
python3 runs/bf16-expert-cache/artifacts/raw/trace-analysis/analyze.py CANONICAL_TRACE RAW_RESULT_JSON
python3 runs/bf16-expert-cache/artifacts/raw/trace-analysis/summarize.py
```

MCP `load_trace` uses each prepared path; SQL includes `slices.with_context`,
constructs `cache_device` and launch-flow `cache_link`, validates counts/errors,
and joins launches within the same-thread expert gather regions. The existing
[trace method](../four-fusion/trace-method.md) defines the preserved import and
interval rules. Reproduction of trace analysis does not launch training.

The prefix omits the remaining 29 microbatches and optimizer device trace.
There is no rank-clock alignment, full-update critical-path estimate, repeated
noise estimate, significance claim or PP4/EP32 conclusion. The experiment's
unprofiled TPS and memory results are owned by the parent report, not inferred
from these profiled timings. No further comparison is required by this analysis.
