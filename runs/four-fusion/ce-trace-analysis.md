# Stable cap-zero CE: matched native trace

Stable CE executes and reduces local CE device work in every matched chunk.
The complete head/loss guard also has a shorter final device tail in all six
samples, but the next decoder backward starts earlier in four of six samples.
This trace supports a local improvement; it does not establish a general TPS
gain, numerical acceptance or memory saving.

## Capture and activation

Compared accepted V2 `four-fusion-control-matched-baseline-trace-001` against
`four-fusion-ce-ce-trace-001`, update 11 after ten full warmup updates. Both
capture ranks 0/1, three complete F/B microbatches, two outer sequence chunks,
and both main/MTP heads in every chunk. All eight ranks complete accumulation
32 and an optimizer step. The trace omits the remaining 29 microbatches and
optimizer device work.

Resolved configuration preserves B2/S4096, global batch 512, FP8 trunk/BF16
head, FP32 reduction, dense FSDP8/expert EP8, MTP1 weight 0.3, compile model/loss,
selective activation checkpointing, prefetch32, epoch51200 and original LR
schedule. Seed is 2048. Monitoring, evaluation, checkpoint saving and memory
snapshots are disabled. Runtime is Torch `2.15.0.dev20260822+cu130`, CUDA13,
Triton3.8 on eight H100 80GB GPUs. Recorded source head is `33b336a592746340d327d86e34149a23dc8688ee`;
the run snapshot and patch/config hashes, rather than that commit alone, pin
the executed dirty snapshot.

Each candidate rank contains **12 `forward_kernel` and 12 `backward_kernel`**
launches, owned through actual launch flows by compiled CE forward/backward
graphs. Rank-specific generated Inductor modules call
`torch.ops.four_fusion.stable_ce_forward.default` and
`stable_ce_backward.default` at `[4096,128256]`; the original Inductor CE
kernels disappear at all matched sites. `Torch-Compiled Region: 1/0` remains
around forward, and `CompiledFunctionBackward` remains around backward.
The profiler omits standalone CPU custom-op labels, so activation relies on
the generated op calls plus matched GPU kernels and compiled ownership.
This is an active CE capture, unlike the earlier FLCE fallback capture.

Actual Perfetto MCP imports preserve 8,795 kernels, 274 copies and 861 memsets
per rank: all 9,930 events have exactly one linked launch and zero importer
errors. Category duration sums match normalization receipts. Actual CUDA
streams are 7 and 24 in the loss guard; synthetic display tids are unused.

## Matched CE forward and backward

All times are milliseconds; arrows show V2 → stable CE. Forward includes the
FP32 scalar SUM in both variants. Each head still has three logical CE kernels:
baseline online statistics + chosen-label/SUM + backward; candidate row CE +
row-loss SUM + backward. Total captured kernel counts are unchanged.
Wider CE includes weights, normalization, scalar accumulation and
`SplitBackward0` concatenation, through the same source boundary.

| Rank/micro/chunk | Forward kernel sum | Backward kernel sum | Wider CE device union | Wider CE device elapsed |
| --- | ---: | ---: | ---: | ---: |
| 0/1/1 | 1.226→0.709 | 1.528→1.464 | 4.208→3.634 | 4.250→3.677 |
| 0/1/2 | 1.833→0.891 | 2.028→1.708 | 5.429→4.192 | 5.481→4.245 |
| 0/2/1 | 1.333→0.702 | 1.552→1.474 | 4.341→3.637 | 4.383→3.680 |
| 0/2/2 | 1.891→0.856 | 1.732→1.540 | 5.089→3.863 | 5.144→3.916 |
| 0/3/1 | 1.224→0.711 | 1.530→1.466 | 4.212→3.631 | 4.255→3.674 |
| 0/3/2 | 1.733→0.844 | 1.963→1.628 | 5.290→4.198 | 5.340→4.244 |
| 1/1/1 | 1.228→0.698 | 1.536→1.467 | 4.222→3.626 | 4.262→3.667 |
| 1/1/2 | 1.743→0.912 | **1.636→1.694** | 4.842→4.137 | 4.889→4.189 |
| 1/2/1 | 1.228→0.699 | 1.532→1.480 | 4.217→3.636 | 4.259→3.679 |
| 1/2/2 | 1.551→0.931 | **1.601→1.714** | 4.616→4.176 | 4.662→4.228 |
| 1/3/1 | 1.226→0.703 | 1.529→1.478 | 4.221→3.642 | 4.261→3.684 |
| 1/3/2 | 1.735→0.885 | 1.948→1.574 | 5.234→3.927 | 5.281→3.975 |

Wider CE **elapsed** improves in all 12 chunks by 0.434–1.306 ms/chunk.
Elapsed starts at the first owned CE device event and ends at the final owned
CE event, covering both heads, forward/backward, FP32 SUM, scaling, scalar
work and backward concatenation. Saved complete device windows establish a
consecutive CE block on actual stream7: there are no interleaved head GEMMs
or other unowned stream7 events. The elapsed interval retains all internal
launch gaps and any in-window device waits (0.041–0.055 ms total internal
gaps in V2; 0.041–0.054 ms in CE). Concurrent communication on other streams
can still affect kernel durations; causal wait identities remain unavailable.
It excludes head GEMMs before/after the block and is separate from the full
head guard below.

Wider CE union improves by 0.440–1.307 ms/chunk. Forward accounts for most of this;
backward is slower in the two bold rank1/chunk2 samples. The JSON retains
per-chunk CPU elapsed, kernel sum, interval union, span, device tail, original
stream, root IDs and all selected device IDs. Graph hashes identify recorded
compiled regions only; ownership is re-established from kernels and flows on
each rank.

### Mechanism grounded in generated code

Pinned baseline generated forward scans BF16 logits once, retaining vector
online-softmax accumulators. Its installed `online_softmax_combine` performs
two `libdevice.exp` evaluations per lane/iteration; `online_softmax_reduce`
then combines lanes. A second kernel gathers chosen BF16 logits and reduces
the loss. Only two FP32 row-stat vectors and the scalar loss are allocated.

The candidate scans the same BF16 logits once using 4096-column tiles, reducing
tile max/SUM before updating scalar row maximum/denominator. It uses `tl.exp`
for each tile plus scalar rescaling, gathers the chosen value in that kernel,
stores two FP32 statistics and row losses, then executes a separate FP32 SUM.
Ignored rows skip the vocabulary scan. Different recurrence, exponential
implementation and tiling provide a source-supported explanation for changed
forward cost; this trace does not isolate their individual contributions.
It removes neither a full-vocabulary FP32 buffer nor a vocabulary read pass
from this compiled baseline.

Baseline generated backward first reduces a vocabulary-wide one-hot scaled
gradient, then reads BF16 logits, reconstructs probabilities and writes BF16
gradients. Candidate backward uses the known scalar incoming scale directly
and performs one logits read/gradient write loop without that one-hot reduction.
FP32 target subtraction and incoming scaling precede the BF16 store. Despite
less reduction arithmetic, backward duration is not uniformly lower.

The [design](loss-design.md) and [implementation](loss-implementation.md) own
the centered-stat numerical correction and existing tests; no new numerical
claim follows from this trace.

## Complete head/loss guard and downstream work

The guard begins after `GFSDP::post_forward (mtp.layers.0)` and ends at the
decoder-gradient bridge's CPU entry. It includes both chunks, the head forward
GEMM and both head-gradient GEMMs per chunk, folding/casts/copies, callbacks,
gradient accumulation and resharding. Device ownership follows launches within
these CPU bounds, including the inner-backward worker thread.

| Rank/micro | CPU submission elapsed | Owned device union | CPU start→final device tail | CPU start→next backward stream7 start | Whole head stream7 elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0/1 | 9.844→9.801 | 89.912→87.815 | 94.854→94.178 | 85.545→86.099 | 80.719→79.850 |
| 0/2 | 7.096→8.606 | 90.058→89.464 | 97.117→95.991 | 89.181→87.245 | 82.268→80.868 |
| 0/3 | 6.964→7.893 | 90.813→89.085 | 97.768→95.781 | 88.272→86.692 | 81.454→80.218 |
| 1/1 | 7.706→9.247 | 89.752→87.759 | 94.798→94.498 | 88.781→86.270 | 83.893→79.641 |
| 1/2 | 6.501→8.186 | 89.948→89.466 | 96.655→96.631 | 89.051→87.165 | 82.521→80.140 |
| 1/3 | 6.406→8.310 | 90.790→88.973 | 97.413→95.433 | 86.969→87.883 | 80.546→81.543 |

CPU submission grows in five samples, while GPU work is queued far ahead.
The JSON additionally records the whole head/loss stream7 first-owned-device
start→last-owned-completion elapsed and its exact bounds/IDs. This device
anchor excludes CPU-to-GPU backlog; it includes both chunks and both head
gradients, but excludes stream24 communication extending beyond stream7.
This whole-head device elapsed improves in five samples and regresses by
0.997 ms at rank1/micro3. Thus the local CE elapsed improvement does not imply
a uniform improvement of complete head computation in this single trace.
Both variants retain six head `aten::mm` roots per microbatch; linked head
GEMM/support sums vary 66.445–70.111 ms in V2 and 67.201–69.319 ms in CE.
Those sums are not whole loss elapsed time.

One head gather per sampled microbatch is prefetched during MTP forward and
recorded separately; gather-to-bridge elapsed therefore includes remaining MTP
work. The guard itself contains two NCCL kernels on stream24 from subsequent
prefetch/callback activity, not just head CE. Their duration sums vary
10.843–22.210 ms in V2 and 12.324–24.223 ms in CE. Consequently summed guard
kernel time can increase even when its device union/tail decreases. Final
head-gradient reduction at accumulation32 is outside this bounded capture.

Guard copies remain two events and about 0.046–0.048 ms per microbatch.
Within the guard's device span, all-device idle is 0.118–0.178 ms in V2 and
0.114–0.137 ms in CE. The last loss-owned stream7 cast to the first decoder
backward launch has only 2.1–5.4 µs of stream7 gap; concurrent device work
covers every such gap. No removed exposed launch bubble is demonstrated.
The next backward launch is already queued 75.7–82.5 ms before its GPU start
in V2 and 76.3–79.5 ms in CE. These delays measure backlog, not causal CE waits.

The broad final tail improves in every sample, but stream7 downstream start
is later by 0.554 ms at rank0/micro1 and 0.914 ms at rank1/micro3 relative to
loss CPU entry. Timing/flow alone cannot identify a critical CUDA dependency:
event record/wait APIs in canonical traces lack usable event identities.
Independent rank clocks are not aligned.

## Verdict and evidence

Local CE is faster in this matched trace. The complete guard shows no broad
tail regression, with extra CPU submission cost and mixed downstream results.
The [unprofiled pair](ce-pair-001.json) remains a separate provisional result
(about +0.303% input TPS); this trace cannot establish its statistical size.
Peak allocated memory increased slightly in that pair, and no memory saving
is claimed. A retained candidate still needs the coordinator's representative
local numerical gate and repeated unprofiled comparison.

[Compact manifest](ce-trace-analysis.json) pins canonical/prepared hashes,
normalization receipts, source/config/runtime records, generated rank0/rank1
code, selected roots/IDs, stream/flow evidence and all metrics.
Exact executed SQL and complete results are retained in ignored
`artifacts/raw/ce-trace-queries-complete.json`; detailed CPU/flow rows and device
windows are in `ce-trace-query-data.json` and `ce-trace-window-data.json` there.
One query failed after a cached processor evicted derived tables; schema
inspection and table recreation resolved it, and the failed query is retained.

No GPU command, test, install or training-source/config/runner write was made
for this analysis. JSON, recorded file hashes and maintained local links pass
the scoped document checks. Source pointers and timings were reviewed once
against the final artifact.
