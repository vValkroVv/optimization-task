# Hopper kernel fusion comparison

Each candidate is compared independently with the selected compiled S plus GPU
selector baseline at `abbbcaf0311409a817b16d794af79e3f3c106864`. The workload
uses eight H100s on `sr008-gpu1`, the real production-tokenizer MDS input,
EP8, accumulation of 32 microbatches, FP8, selective activation checkpointing,
FlashQLA and EMA. The original capacity policy still reduces 128 counters.

The acceptance order is focused component parity and timing, then a native
10-warmup/10-measured run, followed by a separate update-21 profile of three
microbatches. Every native candidate run enables one switch. Its trace must
confirm execution of the affected path. Record global physical tokens per
second, measured-window cadence, compilation activity and the maximum
allocated and reserved peaks across ranks. Profiles supply device evidence
for the change. They cover a fraction of an optimizer update.

## Designs under test

| Variant | Mechanism and preserved behavior | Reference |
| --- | --- | --- |
| MLA RoPE | Rotate Q and shared K in FP32, then write directly to the strided BF16 layout consumed by cuDNN. Backward retains Inductor's FP32 K-head reduction before rotating the gradient. | [H100 reference component](../../artifacts/mla-rope-component/attempt-004/results.json), [cuDNN RoPE](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/RoPE.html) |
| Dense weight tail | Let CudaCherry's 128×128 quantizer zero the partial BF16 tile during each orientation's existing quantization. GFSDP's pre-backward hook must restore the original full weight view before dgrad. | [FP8Gemm](../../gift/torchtitan/ops/float8/cuda_gemm.py), [H100 reference component](../../artifacts/dense-pad-component/attempt-002/results.jsonl) |
| GDN short-conv backward | Recreate the width-four convolution and SiLU/L2 gradients in the tile. Write activation gradients and the existing FP32 weight partials. | [QKV producer](../../gift/torchtitan/models/common/attention/qkv.py) |
| GDN output norm and FP8 | Quantize each 128-channel head in the output norm epilogue. Preserve the BF16 norm and output multiplier roundings, FP32 scale and projection backward boundary. | [GDN projection](../../gift/torchtitan/models/common/attention/linear/gdn.py) |
| MoE dual-layout SwiGLU backward | Produce both E4M3 layouts in a 128-row expert tile. Column scales and values use the rounded rowwise E4M3 values and their scales. Check the earlier column allocation against full-step memory. | [Grouped GEMM](../../gift/torchtitan/ops/float8/grouped_gemm.py), [Transformer Engine FP8 block scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html) |
| CuTe residual | Load the residual through the norm's existing TMA ring and apply the add after BF16 norm rounding. Earlier BF16 residual staging uses shared memory and may reduce the number of pipeline stages. Match all four norm gradients and the identity residual gradient. | [CuTe forward](../../gift/torchtitan/models/common/norms/cute_gated_norm/forward.py), [CUTLASS epilogue fusion](https://docs.nvidia.com/cutlass/4.8.0/media/docs/operators/tutorials/001_gemm_with_fused_epilogue.html) |
| MLA gate and FP8 | Fuse sigmoid and attention layout conversion with rowwise quantization using the installed CudaCherry operand layout and FP32 power-of-two scale rule. Preserve the BF16 boundary and verify the projection backward. | [MLA](../../gift/torchtitan/models/common/attention/softmax/mla.py) |

The task runner and candidate switches are in [run.py](run.py).

## Matched control

The fresh `fusion-control-001-timing` run processed 20,971,520 physical
tokens in 121.598 seconds across measured updates 11–20. Global throughput
was **172,466 tokens/s**. The last five updates' median host cadence was 0.45%
longer than the first five. Across eight ranks, peak allocated and reserved
were 65.890 and 67.580 GiB, respectively. There were no allocator retries or
OOMs. The boundary compiler-cache observer reported stable metadata during
the window. [Raw timing summary](../../artifacts/kernel-fusion/control/timing-summary.json)

The matched update-21 profile spans **1145.316 ms** from the first microbatch
marker to the final GPU activity. Rank 0 records 8269 kernels attributed to the
three microbatches. Perfetto independently shows 18 causal convolution backward
kernels at 6.413 ms total and 21 SwiGLU backward kernels at 6.668 ms total.
[Rank-0 control trace](../../artifacts/kernel-fusion/control/rank0_trace.json)

## Hopper component results

| Variant | Compiled component result | Precision and integration checks |
| --- | --- | --- |
| MLA RoPE | 3.390 → 3.061 ms for the cuDNN/SAC attention tail. 6.484 → 6.214 ms through the actual compiled MLA dispatcher | Output and the K and V gradients match bitwise in the attention tail. Four Q-gradient elements differ at the baseline A/A scale. The full MLA output is exact and its input-gradient error matches A/A. [Measurement receipt](../../artifacts/kernel-fusion/mla-rope/component/verification.json) |
| Dense weight tail | Median paired forward and backward latency −4.90% without checkpointing and −1.50% with SAC | Outputs, input gradients and all weight gradients match bitwise. The two-rank GFSDP check also matched both local gradient shards after the first and third modules reused one all-gather buffer. [Component measurements](../../artifacts/kernel-fusion/dense-pad/component/dense-pad.json), [Rank 0](../../artifacts/kernel-fusion/dense-pad/component/dense-gfsdp.rank0.json), [Rank 1](../../artifacts/kernel-fusion/dense-pad/component/dense-gfsdp.rank1.json) |
| GDN short-conv backward | 722.725 → 646.520 µs for the QKV convolution VJP | Q, K and V match bitwise with SAC. dX and dW relative L2 errors are 2.0e-6 and 2.6e-5. [Source and measurement receipt](gdn-short-conv-component.json) |
| GDN output norm and FP8 | 151.706 → 63.822 µs for the forward norm-to-FP8 chain | E4M3 bytes, FP32 scales, projection output and gradients match bitwise at the fixed native shape with SAC. The Triton Gluon layout preserves FLA's FP32 reduction order. Mixed-shape compilation hit a Dynamo SymInt error. [Source and measurement receipt](gdn-output-norm-component.json) |
| MoE dual-layout SwiGLU backward | 492.00 → 615.74 µs for SwiGLU backward plus both FP8 layouts at 63,872 rows | Both layouts, scales, grouped loss, and grouped gradients match bitwise under compile and SAC. Tuning removed the first prototype's local stack spills. Compiled grouped backward peak allocated increased 6.81 MiB. [Source and measurement receipt](../../artifacts/kernel-fusion/moe-component/moe-component-003/result.json) |
| CuTe residual | SAC forward 292.98 → 241.94 µs. SAC forward and backward 764.82 → 770.78 µs | Output and all gradients match bitwise. SAC peak allocated for the forward and backward is unchanged in this component. [Measurement receipt](../../artifacts/kernel-fusion/control/cute-residual-component.json) |
| MLA gate and FP8 | 1183.47 → 1250.03 µs for compiled gate-through-projection forward and backward | E4M3 bytes, FP32 scales, output and all gradients match bitwise. Peak additional allocation fell 65 MiB. [Measurement receipt](../../artifacts/kernel-fusion/mla-gate-fp8/component/verification.json) |

## Native 10+10 measurements

Each arm has a worktree containing only that variant's source patch. The
original checkout supplies the control, canonical YAML and input receipt. All
ranks use the same production software and data fingerprints and reuse the
verified compiler cache. Candidate preflight checks the imported GIFT path
against the variant's worktree.

| Arm | Global physical tokens/s | Delta vs control | Peak allocated / reserved GiB | Update-21 evidence |
| --- | ---: | ---: | ---: | --- |
| Compiled S + GPU selector | 172,466 | — | 65.890 / 67.580 | [Rank 0](../../artifacts/kernel-fusion/control/rank0_trace.json) |
| CuTe residual | 172,915 | +0.26% | 65.960 / 67.775 | [Rank 0](../../artifacts/kernel-fusion/cute-residual/rank0_trace.json), [Comparison](../../artifacts/kernel-fusion/cute-residual/comparison.json) |
| MLA RoPE | 173,083 | +0.36% | 65.892 / 67.561 | [Rank 0](../../artifacts/kernel-fusion/mla-rope/rank0_trace.json), [Comparison](../../artifacts/kernel-fusion/mla-rope/comparison.json) |
| GDN output norm and FP8 | 173,117 | +0.38% | 65.888 / 67.678 | [Rank 0](../../artifacts/kernel-fusion/gdn-output-fp8/rank0_trace.json), [Comparison](../../artifacts/kernel-fusion/gdn-output-fp8/comparison.json) |
| Dense weight tail | 172,854 | +0.22% | 65.897 / 67.561 | [Rank 0](../../artifacts/kernel-fusion/dense-pad/rank0_trace.json), [Comparison](../../artifacts/kernel-fusion/dense-pad/comparison.json) |
| GDN short-conv backward | 172,893 | +0.25% | 65.890 / 67.561 | [Rank 0](../../artifacts/kernel-fusion/gdn-short-conv/rank0_trace.json), [Comparison](../../artifacts/kernel-fusion/gdn-short-conv/comparison.json) |
| MLA gate and FP8 | 172,513 | +0.03% | 65.891 / 67.580 | [Rank 0](../../artifacts/kernel-fusion/mla-gate-fp8/rank0_trace.json), [Comparison](../../artifacts/kernel-fusion/mla-gate-fp8/comparison.json) |
| MoE dual-layout SwiGLU backward | 172,253 | −0.12% | 65.897 / 67.580 | [Rank 0](../../artifacts/kernel-fusion/moe-dual-fp8/rank0_trace.json), [Comparison](../../artifacts/kernel-fusion/moe-dual-fp8/comparison.json) |
| Same control after sweep | 172,593 | +0.074% | 65.898 / 67.561 | [Unprofiled timing](../../artifacts/kernel-fusion/control/bracket-timing-summary.json) |

The repeat control processed the same 20,971,520 physical tokens in
121.508 seconds. The resolved config matches the initial control apart
from the output directory. Source revision, input fingerprint, seed,
runtime and cache policy also match. The two controls differ by 8.95 ms
per update. Relative to the repeat, the candidates are GDN output norm
and FP8 +0.303%, MLA RoPE +0.284%, CuTe residual +0.187%, GDN short-conv
backward +0.173%, dense weight tail +0.151%, MLA gate and FP8 −0.046%,
and MoE dual-layout SwiGLU backward −0.197%. This bracket measures the
change in control timing over this sweep. Each candidate has one native
timing window. Across all timing runs, every rank reported zero
allocator retries and OOMs, and the cache observer reported stable
boundary metadata. [Repeat config](../../artifacts/kernel-fusion/control/bracket-timing-resolved-config.yaml)
· [Repeat metadata](../../artifacts/kernel-fusion/control/bracket-timing-metadata.json)

The CuTe arm timed 121.282 s over ten optimizer updates. Its last five
updates' median host cadence was 0.39% longer than the first five. The
boundary compiler-cache observer reported stable metadata. The throughput delta
comes from one matched pair. [Raw timing summary](../../artifacts/kernel-fusion/cute-residual/timing-summary.json)

CuTe's update-21 trace contains 108 launches owned by the fused residual op
and 129 ordinary width-4096 norm output launches. The 48 separately launched
residual adds identified immediately after the baseline norm disappear.
Summed device time for the width-4096 norm outputs and those adds moves from
19.715 to 18.735 ms. The three-microbatch window moves from 1145.316 to
1143.547 ms. Perfetto links the fused launches to the new CPU op by external
ID and shows 206,848 bytes of shared memory for those launches. The resolved
timing configuration differs from the control only in the output directory.

MLA RoPE replaces 126 complex-input preparation, multiplication and cuDNN
input-packing launches with 18 fused pack and nine unpack launches. Their
summed device time falls from 4.157 to 3.570 ms. Perfetto shows 27 cuDNN SDPA
kernels in both traces, at 30.137 and 30.234 ms. The three-microbatch window
moves from 1145.316 to 1140.681 ms. The native 10+10 window is 121.164 s. The
compiler observer reported stable metadata, and all eight ranks reported zero
allocator retries and OOMs. [Raw timing summary](../../artifacts/kernel-fusion/mla-rope/timing-summary.json)

GDN output norm and FP8 replaces 36 three-kernel chains with 36 fused
launches. Summed device time for these launches falls from 5.826 to 2.514
ms. Perfetto attributes the remaining 18 FLA norm forward launches to the
backward `torchtitan::fla_vjp` scope. The three-microbatch window is 1145.793
ms. The independent 10+10 measures 121.141 s with a stable boundary compiler
observer and zero allocator retries or OOMs. [Raw timing summary](../../artifacts/kernel-fusion/gdn-output-fp8/timing-summary.json)

The dense weight-tail trace drops all 36 BF16 weight-padding launches at
the selected shape. Both traces have 48 CudaCherry weight-quantization
launches. Summed GPU time for pads plus quantization falls from 4.956 to
2.564 ms, and the three-microbatch window is 1143.897 ms. Peak allocated
in the native run is 6.56 MiB above the control. The lower temporary cost
of the individual weight quantizer therefore did not reduce the full-step
peak. [Raw timing summary](../../artifacts/kernel-fusion/dense-pad/timing-summary.json)

GDN short-conv backward fuses packing, convolution recomputation and
backward. Matching CUDA launches by their enclosing operator on the CPU
finds 18 complete backward chains in each trace. The baseline
`torchtitan::fla_vjp` includes 3.039 ms of QKV unpack and forward replay
that the earlier comparison omitted. Both arms also cast the FP32
weight gradient after the partial reduction. With matching boundaries,
the device sum falls from **13.390 to 12.110 ms** (−9.56%). The median
first-to-last device span falls from 756 to 676 µs (−10.52%), which
agrees with the isolated component result. The full sampled rank-0
window is still 3.30 ms longer in this trace pair.
[Corrected trace comparison](../../artifacts/kernel-fusion/gdn-short-conv/comparison.json)
· [Raw timing summary](../../artifacts/kernel-fusion/gdn-short-conv/timing-summary.json)

### MLA gate audit

Perfetto matches launches on stream 7, using CUDA correlation IDs to
distinguish ordinary forward from SAC forward recomputation. The three
sites per microbatch are layers 3 and 7 and the MTP block. The analysis
matches the first output-projection GEMM after each gate input on that
stream. The elapsed rows include any launches and gaps between the gate
start and the end of the projection.

| Work across three microbatches | Control, ms | Fused, ms |
| --- | ---: | ---: |
| Gate and activation quantization, nine forward sites | 0.887 | 0.520 |
| Gate and activation quantization, nine SAC recomputations | 1.073 | 0.582 |
| Gate through output GEMM, nine forward sites, elapsed | 3.052 | 2.685 |
| Gate through output GEMM, nine SAC recomputations, elapsed | 3.656 | 3.177 |
| Output-projection dgrad GEMMs, nine sites | 2.069 | 2.161 |
| Output-projection wgrad GEMMs, nine sites | 2.694 | 2.776 |
| Compiled sigmoid gradient and cuDNN backward preparation, nine sites | 0.976 | 0.984 |

The nine backward sites issue the same weight and incoming-gradient
quantization, dgrad, two FP8 transposes, wgrad, and sigmoid/attention
gradient preparation. Total time for these launches is 6.819 versus
7.010 ms. The sigmoid preparation has the same 106.593 µs median,
32 registers per thread and no shared memory in both traces. Excluding
the one MTP site with NCCL overlap in each arm gives 0.851 versus
0.852 ms for the remaining eight preparations. The relevant launches
were queued at least 3.16 ms before the predecessor finished. The MTP
outlier in each trace occurs in a different microbatch under different
collective overlap. The difference between those two sites accounts for
0.096 ms of the 0.191 ms net increase in the matched backward launches.
The separate component's combined forward and
backward was 66.6 µs slower at the median, while its FP8 payload,
FP32 scales, projection output and gradients matched bitwise.
[Component data](../../artifacts/kernel-fusion/mla-gate-fp8/component/verification.json)

The 0.858 ms device saving includes both forward and SAC replay
across three microbatches. At 32 microbatches, it would save about
9.15 ms per update if representative and fully exposed. The mean
control update is 12.160 s, so this would move throughput about
0.075%, or 130 tokens/s. The two native timing windows processed the
same 20,971,520 tokens and measured 121.598 versus 121.565 seconds.
That is 3.33 ms per update and +0.027% throughput. Both resolved
configurations match apart from the output directory. The repeat control
was 0.074% faster than the initial control, and the fusion was 0.046%
slower than the repeat. The gate's native result falls between the
two control measurements.
[Control timing](../../artifacts/kernel-fusion/control/timing-summary.json)
· [Fused timing](../../artifacts/kernel-fusion/mla-gate-fp8/timing-summary.json)

In the profiled pair the three-microbatch rank-0 window is
1145.316 versus 1150.695 ms. All-stream GPU idle rises from 27.403
to 33.961 ms while all-stream GPU busy time falls from 1117.613 to
1116.503 ms. The extra idle includes gaps at unrelated routing and
input-preparation launches. The trace establishes the gate savings
through the output GEMM and shows where the sampled step spent more
time. The timings do not establish a repeatable training speedup or
a slowdown caused by the gate backward.
[Control profile](../../artifacts/kernel-fusion/control/rank0_trace.json)
· [Fused profile](../../artifacts/kernel-fusion/mla-gate-fp8/rank0_trace.json)
· [Comparison](../../artifacts/kernel-fusion/mla-gate-fp8/comparison.json)

Rank 1 gives the same local result. Gate and activation-quantization
device time is 1.889 versus 1.064 ms. The elapsed time through the output
GEMM is 6.525 versus 5.636 ms. The nine sigmoid-gradient and cuDNN
preparation launches total 0.955 versus 0.956 ms, with medians of 106.081
and 106.209 µs. The whole rank-1 window is 5.285 ms longer, with 2.089 ms
more idle and 3.197 ms more device busy time across all streams.
[Control profile](../../artifacts/kernel-fusion/control/rank1_trace.json)
· [Fused profile](../../artifacts/kernel-fusion/mla-gate-fp8/rank1_trace.json)

### MoE dual-layout SwiGLU backward

The trace confirms removal of all 21 standalone `w13` gradient
requantization launches, which take 3.247 ms in the control. The
21 SwiGLU backward kernels rise from 6.668 to 12.737 ms, so the
targeted device time rises from **9.915 to 12.737 ms** (+28.5%).
The 63 other grouped requantization launches stay at 10.074 versus
10.072 ms. Native throughput is 172,253 tokens/s, and the native trace
agrees with the slower isolated compiled chain.

The emitted SwiGLU kernel uses 512 threads and 128 registers per thread.
The trace's register-based residency limit is one block per SM versus
four 128-thread blocks in the control. Both have the same theoretical
resident warp count of 16 warps per SM. These are launch properties.
They do not isolate the share of kernel latency due to the added
column-layout work, shared-memory traffic or scheduling.
[Raw timing summary](../../artifacts/kernel-fusion/moe-dual-fp8/timing-summary.json)
· [Rank-0 trace](../../artifacts/kernel-fusion/moe-dual-fp8/rank0_trace.json)

## Result

GDN output norm and FP8 and MLA RoPE combine the strongest native
throughput comparisons with matching improvements in their targeted
trace chains. Dense weight tail, CuTe residual and GDN short-conv
backward also shorten their chains and are faster than both controls
in this sweep. CuTe residual raises peak reserved memory by 200 MiB
against the initial control. The MLA gate saves work on both profiled
ranks, with native throughput between the two controls. The targeted
MoE dual-layout SwiGLU chain regresses. All seven switches are opt-in.

## Combined five at 40+100

The 40+100 comparison uses the same compiled S plus GPU capacity-selector
recipe on the eight H100s of `sr008-gpu1`. The control uses
`abbbcaf0311409a817b16d794af79e3f3c106864`. The candidate uses the
same revision plus the source for GDN output norm → FP8, MLA RoPE and
Q/K packing, CuTe norm and residual add, GDN short-conv backward, and
the dense FP8 weight tail. The source files match local `main` commit
`fe5950b56`. The candidate's patch SHA256 is
`d388a664f9965b4afbfefcd3da692415ffcb456000b664bd7b030b2f53b20990`.
[Source manifest](../../artifacts/kernel-fusion/combined-five-140/combined-source.sha256)

The [runner](run-140.sh) launches the control and combined arm serially as
`fusion-140-pair-001-baseline` and `fusion-140-pair-001-combined-five`.
Each warms up on updates 1–40 and times complete updates 41–140,
including communication, optimizer and EMA. The two arms use the
same real data order, production tokenizer, compiler-cache policy,
32 microbatches per update and the existing 65,000-update LR schedule.
The timing starts and ends at synchronized update boundaries across
all eight ranks. Per-update host cadence is recorded separately.

## Implementation audit

The timings in this audit refer to the existing single-feature rank-0
captures of three microbatches. The proposed follow-ups need their own
component checks and native measurements.

| Variant | Remaining work or cost | Next targeted check |
| --- | --- | --- |
| GDN output norm → FP8 | The forward computes FP32 inverse RMS and discards it. Backward launches 18 FLA norm replays for 1.262 ms. Each materializes a 64 MiB BF16 norm output. The separate multiply of the incoming gradient takes another 0.723 ms across the 18 sites. | Verify FP32 inverse RMS against FLA, then call FLA backward with that statistic under the existing SAC policy. The statistic is 1 MiB per site at the tested shape. If keeping it raises the live-memory peak, measure a statistics-only replay. Preserve the FP32 reduction order and the BF16 rounding before quantization and backward. [Forward and backward](../../gift/torchtitan/models/common/attention/linear/gdn_output_fp8.py) |
| GDN short-conv backward | The complete chain improves 13.390 → 12.110 ms. The fused kernel uses 255 registers/thread and 32 KiB shared memory per CTA. It keeps large FP32 tiles live while gathering shifted values. | Try a narrower V tile, whose channels are independent, or a Gluon layout that avoids the six full-tile shifted gathers. Keep the width-four tap order, BF16 rounding and FP32 weight-partial grouping. Check register/spill counters and replay the existing gradient test before interpreting timing. [Kernel](../../gift/torchtitan/models/common/attention/linear/gdn_short_conv.py) |
| CuTe norm and residual | The norm output stage moves from 160 to 202 KiB shared memory per CTA at 168 registers/thread in both arms. Among 108 fused sites, 54 are SAC recomputations. At 24 matched replay sites the next device kernel is the same norm backward in both arms, and the output stage alone rises from 1.468 to 1.939 ms. The custom op still loads and adds the full residual at these sites. | Confirm which replay graphs consume the residual output. Produce just the saved statistics at sites whose activation output is dead. Separately test moving the residual load to the epilogue, since the current TMA ring includes its 8 KiB tile in every stage's barrier. Keep both BF16 rounding boundaries. [TMA ring and epilogue](../../gift/torchtitan/models/common/norms/cute_gated_norm/forward.py) · [Saved tensors](../../gift/torchtitan/models/common/norms/cute_gated_norm/op.py:248) |
| Dense FP8 weight tail | Thirty-six explicit BF16 pad launches disappear. The 18 transposed dgrad quantizers still recompute the same block scales for 1.078 ms. Six output-gradient trim copies also remain and take 0.370 ms. | Test CudaCherry's existing given-scale transposed quantizer with the forward FP32 scale plane. Retain the restored BF16 GFSDP weight view and verify FP8 bytes, gradients and scale lifetime under SAC. The scale plane is about 11 KiB per weight invocation. [Dense forward and backward](../../gift/torchtitan/ops/float8/cuda_gemm.py) |
| MLA RoPE and Q/K packing | The new compiled path has no extra Q/K contiguous copies at the nine audited backward sites. The pack describes shared K across two 16-head CTA tiles. The remaining K/V gradient assembly also exists in the control and takes 0.772 ms in the fused arm. | Keep the Q tile and test one rotation of shared K per token with broadcast into the existing cuDNN stores. A general 32-head tile was slower in the component. Keep the K-head gradient reduction and BF16 storage boundary. [RoPE kernel](../../gift/torchtitan/models/common/attention/softmax/mla_rope_pack.py) |
| MoE dual-layout SwiGLU backward | Using one kernel for both layouts grows the target chain from 9.915 to 12.737 ms. The staged path still launches 21 DeepGEMM transposes of the w13 rowwise scale plane for 0.159 ms. The installed CudaCherry row-to-column consumer also converts that scale plane to its preferred physical stride. | Keep staged SwiGLU and test writing the rowwise FP32 scale plane directly in the physical stride required by both consumers. Keep the logical shape, padded expert boundaries and quantization bits. Verify removal of the copies with compile and SAC. [Scale allocation and FP8 output wrapper](../../gift/torchtitan/ops/float8/triton_kernels/fused_swiglu_quantization.py) |
| MLA gate → FP8 | The native forward chain is faster on two ranks. Its native throughput falls between the controls. The separately tested projection forward and backward is slower at the component median. | Defer this arm. A tiling-only experiment could group adjacent heads while keeping the BF16 and FP8 boundaries. Keeping the full FP32 sigmoid would retain 128 MiB per call at this shape. [Native gate audit](#mla-gate-audit) |
