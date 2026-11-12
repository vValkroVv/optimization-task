# Native short performance assessment

## Verdict: measurement blocked

The second graphs-off control,007, failed with an actual CUDA illegal instruction during
update34. The fixed21–40 window did not complete. The matrix stopped immediately;
008/on-B1,009/off-A2 and010/on-B2 were not released or run. A/A variation,
paired off/on throughput and graph memory cost are **unavailable**. No shortened
window or speedup/no-benefit inference is reported.

The observed report is rank1's NCCL watchdog,2026-11-12 10:04:19 UTC:
`CUDA error: an illegal instruction was encountered`.
See [rank1 stderr](../../attempts/007/torchrun/2dacdb64-4018-4100-9392-464f8f41ef2c_hlqm0hj5/attempt_0/1/stderr.log)
line19. CUDA reporting can be asynchronous; this location does not establish the
originating operation or implicate NCCL. This run had compiler graph activation
false. No fault diagnosis, retry, new trace or instrumentation was performed.

## Runs and native completion

| Attempt | Variant | Optimizer updates/rank | Fwd/bwd microbatches/rank | Disposition |
| --- | --- | --- | --- | --- |
|001|off-AA1 original|0|0|Preserved pre-native missing-identity failure; superseded by corrected preparation002|
|002–005|original remaining matrix|—|—|Not run; preserved|
|006|off-AA1|41 on all8|164 on all8|Complete valid native window|
|007|off-AA2|33 on all8|135 rank1;134 other ranks|CUDA failure during34; no complete timing/phase-memory output|
|008|on-B1|—|—|Not run after007 failure|
|009|off-A2|—|—|Not run after007 failure|
|010|on-B2|—|—|Not run after007 failure|

All eight006 and007 structured logs parse and have strictly increasing unique
sequence IDs.007's last events reach34, but only33 `optim_end` events establish
completed optimizer updates. Native rank-zero loss/norm records are finite at
1/10/20/30 for007 and additionally40 for006; this is the configured native
rank-zero metrics output, not elementwise numerical or convergence evidence.
Counts, last events and available metrics are in [007-failure.json](007-failure.json).
The failure prevented normal phase-memory flushing; missing outputs are retained
as missing, not reconstructed or declared complete.

## Single valid baseline006

| Metric | Result | Scope |
| --- | --- | --- |
|Useful throughput|316975.7 tokens/s|20,971,520 already-global useful tokens / maximum completed-window wall across8 ranks|
|Completed window|66.161293s|Updates21–40; closes before update41 fetch after device drain|
|Completed mean update|3.308065s|Completed wall /20; not a median|
|Host submission span min/median/max|3.132306 /3.299407 /3.561982s|160 per-rank loop/submission spans; not complete-update latency|
|Current-stream boundary span min/median/max|3.123445 /3.321089 /3.410040s|160 CUDA-event spans; independent side streams can be omitted|
|Allocator allocated peak envelope|28.089726GiB|Maximum all-rank native reset-aligned intervals21–30 and31–40|
|Allocator reserved peak envelope|33.625000GiB|Same intervals; no allocation retries or OOM counters|
|Sampled device window peak|36.032227GiB|500ms NVML; lower bound on true peak|
|Sampled whole-process peak|36.032227GiB|Setup and training samples; lower bound|
|Maximum sampling gap|545.000ms|Actual timestamps inside each rank's window|

Allocator opening/closing samples are separately retained in [006-memory.json](006-memory.json)
and [006-nvml-setup.json](006-nvml-setup.json). The closing sample follows the native40 reset;
it is a boundary tail, not the reset-aligned21–40 peak. The NVML join uses actual
per-rank UTC opening/closing brackets and verified UUID→physical-index mapping;
only samples strictly inside the bracket are selected. Per-rank gaps and
excluded boundary tails are reported. Device samples include non-allocator memory.
Sampled minimum free memory is an upper bound on true minimum headroom.

Maximum per-rank launch→native training-start22.231671s,
launch→first completed update63.661568s, launch→window opening127.047080s;
native training-start→window opening105.236614s. Setup through20 is excluded
from the throughput window. These labels do not split compilation, capture,
autotuning or warmup into unsupported independent phase costs.

The integrated timer includes native fetching, model/loss, backward,
accumulation, clipping, optimizer, scheduler and recurring native metrics;
it excludes final timing aggregation/persistence. Device-compute-only time,
host wait/idle attribution and separate communication/copy elapsed costs are
unavailable in these unprofiled runs; overlapping prior trace activity is not summed.

## Workload, freeze and inherited cost

Actual006 rank-zero resolved configuration was inspected: native Trainer;
only timing and graph-step callbacks; exact3B MoE,18 blocks, HybridEP32-expert/top4,
EP8/FSDP2, FlexAttention, FP8 dense/shared quantizer and sparse grouped GEMM,
selective AC, compiled model+chunked8 loss,18 model graph exclusions,
local8/global256/seq4096/accumulation4, BF16 parameters/FP32 reduction,
fused AdamW, scheduler horizon30000, seed42, c4_test, native metrics cadence10.
Checkpoint/evaluation/profiler/numerical recording are off as accepted for this phase.
All8 activation/compiler/environment receipts agree with the intended constructed
path and graph-off options; both controls retain partitioning=true and
mix_order_reduction=false. Physical topology is the existing single8-H100 NVLink
allocation; actual group memberships are preserved in [006-activation.json](006-activation.json).

Before007 launch, all98 Source010 working files and every frozen preparation/attempt
asset matched their manifests; the final006 extraction revalidated the same freeze.
Source manifest SHA256:
`b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20`.
Frozen runnable manifest SHA256:
`e440255d61acb668e000e532b94432a80952959acc9adcba0b07bdffbdc10c05`.
Observer YAML SHA256:
`01675fbc967e2e0a7e3798d333edc117d18bf5ad3746d999a86f566a3ff554f8`.
Runtime: Torch2.15.0.dev20260822+cu130, CUDA13.0, NCCL2.30.7,
Triton3.8.0+git3f6e4113, DeepEP1.2.1+1b8f467, DeepGEMM2.3.0+f093a4c.
Source010 base commit446431dfa191ef0300f116257ae918bc85e38c06;
full source/runtime/seed/config and exact argv remain in existing receipts.

Reuse [accepted review](../../review/001/REVIEW.md) and
[active-path equality](../../../PERFORMANCE-SCOPE/001/source-active-equality.json).
Matched observer bookkeeping/sampler host costs are unmeasured; no new per-update
instrumentation was added. A008 previously demonstrated persistent actual CE
forward/backward+mask replay at21–23 on all8 ranks with zero fresh capture.
Replay/recapture21–40 was not observed here; compiler receipts prove options only.
A008's25GiB/rank/update staging copy volume is not exposed wall tax or residency.
The earlier fault at48 remains unresolved, and this independent graph-off failure
adds no causal attribution. No production, convergence or continuation claim follows.

## Release, checks and artifacts

The accepted owner reports all724007 process identities absent; controller674544
and owner674545 were separately confirmed absent by PID/start ticks. The single
completion monitor exited. At10:05:23 UTC the same8 UUIDs showed zero utilization,
zero used memory and no GPU applications; see
[post-failure availability](../../attempts/007/post-failure-availability.json).
No further process termination or resource mutation was needed.

[Commands](COMMANDS.md), [machine summary](summary.json),
[006 extraction](006-extraction.json), [006 series](006-step-series.csv),
[007 partial series](007-step-series.csv), all raw attempts and failed stderr are
preserved. A reader check initially assumed native metrics printed on every rank;
it was corrected to the actual `save_for_all_ranks=false` producer and passed.
That offline assumption failure was not a training harness failure. Source,
launcher and running files were not edited. No packages, allocation, remote work,
commits, profiling, CUDA retries or additional acceptance gates were introduced.
