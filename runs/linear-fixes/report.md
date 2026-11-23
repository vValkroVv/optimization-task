# Linear attention and MLA component validation

The `linear-fixes` worktree starts from original A, `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`.
The isolated comparisons ran on separate H100 80 GB GPUs on `sr008-gpu` in
ML Space job `84fd3579-7bd2-4969-ae47-b7af17b73ca2` with the original A
Python 3.12 runtime, PyTorch `2.15.0.dev20260822+cu130`, CUDA 13.0, and
Triton 3.8.0. Device indices were 0 for MLA, 1 for direct GDN backward,
2 for GDN short convolution, and 3 for GDN output norm and FP8 projection.
Each comparison ran both variants on its assigned device. Native FLA
`1d2281ff1b7988d8070f00dba5ffa9747ae3a60e` supplied the native FLA control.

## Previous kernels on the 10+10 crop

The two resolved configurations from the +16.4% 10+10 comparison,
`original-a-vs-cute-001-a2` and `original-a-vs-cute-001-prod`, agree on
`gift_3_5/100b_crop8` with model width 4096 and a rank-local batch of 2
sequences of 4096 tokens. GDN uses 16 Q/K and 32 V heads, both with width
128, a four-tap convolution and a norm multiplier of 2.0. MLA uses 32
heads, 128 non-rotary + 64 rotary Q/K channels and 128 V channels. Q-LoRA
is disabled and KV-LoRA rank is 512. The resolved configuration hashes are
`5e389eebaf3f09d0c9d40232afee8f0ef52ce40530ce2a3dc3510f009ba2d6aa`
for A and
`71dda3f90c66e36735a53b96d50663d2c5d624cfc860fe8858a9df0cb9986003`
for production.

The previous MLA RoPE, GDN short-conv and GDN norm-to-FP8 kernel files
were introduced at `fe5950b56de5b2cec7ff4cd063f7338c98200c45`. All three
match the kernel files in the measured production revision
`2c67c182e8de6145f3796d62203714ab1b815446` byte for byte. Both arms in
each component replay use the same fixture, device, compile settings and
checkpoint policy. FlashQLA stays disabled in these runs. The previous
output-skipping scan implementation requires FlashQLA `external_gate`, so
the scan row uses the previous native FLA recompute path.

| Component, previous → generalized | Synchronized wall | CUDA timing | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: |
| MLA forward + backward | 9.954 → 9.975 ms, within control variation | 9.937 → 9.958 ms event span | 1865.74 → 1865.74 MiB | 2046 → 2046 MiB |
| GDN scan forward + backward | 3.134 → 2.715 ms (−13.4%) | 2.524 → 2.147 ms graph (−14.9%) | 1508.56 → 1444.56 MiB | 1512 → 1448 MiB |
| GDN short-conv forward + backward | 1.209 → 1.106 ms, inconclusive | 796.97 → 733.07 µs graph (−8.0%) | 656.25 → 656.25 MiB | 658 → 658 MiB |
| GDN output norm → FP8 packing only | 139.44 → 145.66 µs | 63.84 → 67.97 µs graph (+6.5%) | 353.00 → 353.00 MiB | 418 → 418 MiB |
| GDN output norm → FP8 projection forward + backward | 1.450 → 1.456 ms, within control variation | 1.443 → 1.448 ms event span | 673.07 → 673.07 MiB | 804 → 804 MiB |

The table reports medians of alternating, warmed, unprofiled samples and
latency changes relative to the previous implementation. CUDA Graph timing
excludes Python/autograd submission and capture-time allocation; event timing
can include device idle gaps during host submission. Memory peaks exclude
CUDA Graph capture and include the resident fixture and weights.

- **MLA.** The median of the 24 paired wall differences was −1.28 µs
  (interquartile range −27.27 to +54.42 µs), against a +10.56 µs median
  difference between identical flag-off controls. Both kernel traces
  show two pack launches and one unpack launch. The combined durations
  of those three kernels were 348.51 → 341.18 µs in the separate profiles.
  The direct comparison gave identical outputs and maximum relative L2
  gradient difference of `9.57e-6` for `wq.weight`. [Timing](evidence/mla-4k-crop-followup/timing.json) · [Parity](evidence/mla-4k-crop-followup/check.json) · [Traces](evidence/mla-4k-crop-followup/trace-comparison.json)
- **GDN direct backward.** The output and all seven gradients matched exactly.
  The trace showed two → one output scans and three → two W/U launches.
  The wall median across 24 identical-control calls was 3.185 ms, and its
  CUDA Graph median was 2.524 ms. Separate trace kernel durations summed to
  2524.23 → 2146.45 µs. [Receipt](evidence/gdn-direct-10plus10/result.json)
- **Short-conv.** Q/K/V, dQKV and convolution dW matched elementwise. Both
  traces followed the fused backward path. The first run and the 24-sample
  repeat each reduced graph replay by 8.0%. In the repeat, paired graph
  differences ranged from −8.13% to −7.87%, versus −0.44% to +0.06% for the
  identical control. The measured wall medians fell 2.2% and 8.5% in the two
  groups, with substantial overlap between paired differences and the
  identical control. [First receipt](evidence/gdn-short-conv-specialized/result.json) · [Repeat](evidence/gdn-short-conv-specialized/repeat-24.json)
- **GDN output norm and FP8.** The packed FP8 bytes, scales, projection
  output and all four gradients matched bitwise. The old and generalized
  traces each show a single fused pack in the pack-only scope and two
  packs with forward/backward. Across 24 graph pairs the generalized pack
  was 6.08% to 6.83% slower, versus −0.58% to +0.48% for the identical
  control. With the projection included, paired wall medians changed
  +0.33% for the candidate and −0.38% for the identical control, with
  overlapping sample distributions. [Receipt](evidence/gdn-fp8-specialized-vs-generalized/result.json)

| Component | Host call median, µs | Trailing event record/wait median, µs | Profiled kernel sum, µs | Profiled device gaps, µs | Profiled copy/memset, µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLA | 2067.0 → 2079.5 | 7891.8 → 7875.8 | 8969.86 → 8982.95 | 126.24 → 131.13 | 15.97 → 14.88 |
| GDN scan | 2256.9 → 1843.9 | 891.7 → 856.8 | 2524.23 → 2146.45 | 2383.73 → 413.89 | 0.77 → 0.77 |
| GDN short-conv, repeat | 669.8 → 626.5 | 541.5 → 476.3 | 801.12 → 738.85 | 2540.65 → 339.46 | 0 → 0 |
| GDN FP8 packing only | 87.5 → 89.1 | 52.0 → 56.4 | 63.20 → 69.09 | 0 → 0 | 0 → 0 |
| GDN FP8 projection forward + backward | 1232.0 → 1240.8 | 215.4 → 215.8 | 1181.75 → 1189.34 | 269.06 → 258.43 | 2.34 → 2.31 |

The profile columns describe separate, single-stream calls. Profiler
overhead can affect device gaps. Host submission and GPU work can overlap.
All measurements are single-device replays without collectives. These
benchmarks measure component costs at the 10+10 tensor shapes. The 10+10
training-window throughput remains a separate measurement.

The first MLA and FP8 launches stopped before GPU work because the SSH
shell selected the host CUDA driver API. The successful launches sourced
`/etc/profile.d/wm.sh` to initialize the training image environment. The
failed launch logs are preserved alongside the raw evidence.

## Earlier unfused-control timings

| Component forward and backward | Wall latency A → B | Wall latency reduction | CUDA timing A → B | Peak allocated A → B | Peak allocated reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLA, B2/S4096/H32 | 10.296 → 9.981 ms | 3.1% | 10.278 → 9.963 ms event span | 1867.74 → 1865.74 MiB | 2 MiB |
| MLA, B2/S8192/H64 | 46.273 → 45.951 ms | 0.7% | 46.254 → 45.932 ms event span | 5088.27 → 5084.27 MiB | 4 MiB |
| GDN scan, B2/S4096/H16/HV32, K128/V128 | 3.074 → 2.636 ms | 14.2% | 2.532 → 2.153 ms graph replay | 1508.56 → 1444.56 MiB | 64 MiB |
| GDN short convolution, B2/S4096/H16/HV32, K128/V128, W4 | 1.261 → 1.073 ms | 14.9% | 0.883 → 0.737 ms graph replay | 1041.25 → 656.25 MiB | 385 MiB |
| GDN output norm and FP8 projection, B2/S4096/HV32, V128 | 1.623 → 1.451 ms | 10.6% | 1.616 → 1.444 ms event span | 673.07 → 673.07 MiB | 0 MiB |

Wall times are medians from warmed, interleaved, unprofiled runs. The percentage
is the latency saved divided by the original latency. Peak allocated totals
include each process's resident inputs and weights. The MLA CUDA event includes
any device gaps caused by host submission. The GDN graph timing replays
captured device work with fixed inputs and excludes Python/autograd submission
and allocations performed at capture time. MLA used 10 warmup calls and 10
measured calls, with 24 measured calls in the 8K repeat. The scan, convolution,
and FP8 payload harness warmed each callable five times before capture and used
20 graph replays per sample. The FP8 projection harness used 10 warmup calls
before its synchronized timing. Each timing group includes an
identical baseline control. Control wall medians were 10.281 ms and 46.520 ms
for the MLA cases, and 3.069 ms and 1.285 ms for GDN scan and short convolution.
The 4K historical crop kernel ran at 9.950 ms wall with the same peak allocated
memory as the generalized kernel. Its source SHA-256 is recorded in the MLA
receipts. An independent 24-sample repetition at 8K gave 46.658 → 45.407 ms
wall (2.7% lower), with a 46.521 ms identical control and the same peak
allocation in both variants as the first run. The two 8K measurements
support a small improvement whose size varies between runs.
[First 8K timing](evidence/mla-8k-timing.json) · [8K repetition](evidence/mla-8k-repeat.json)

The FP8 forward epilogue alone, from RMSNorm through the quantized bytes
and scales, went from 152.0 to 69.0 µs in CUDA Graph replay (54.6% lower).
Peak allocated memory went from 417 to 353 MiB. Both variants peaked at
673.07 MiB with the projection GEMM and backward included. Peak reserved
memory for that call went from 772 to 804 MiB. The
identical control was 151.5 µs for forward graph replay and 1.628 ms wall
for the projection forward and backward. [Initial timing](evidence/gdn-fp8-benchmark/result.json) · [Final timing](evidence/gdn-fp8-benchmark-final/result.json)

| CUDA diagnostics, separate profiled call | Kernel durations A → B | Device gaps A → B | Copy/memset A → B | Host call A → B | Host record/wait A → B |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLA 4K | 9264.9 → 8983.0 µs | 886.2 → 131.1 µs | 15.2 → 14.9 µs | — | — |
| MLA 8K | 42711.5 → 41724.3 µs | 647.3 → 281.8 µs | 20.4 → 19.0 µs | — | — |
| GDN scan | 2532.1 → 2156.9 µs | 2257.9 → 420.6 µs | 0.8 → 0.8 µs | 2201.1 → 1784.2 µs | 864.8 → 862.5 µs |
| GDN short convolution | 899.4 → 740.1 µs | 2392.3 → 251.8 µs | 0 → 0 µs | 904.9 → 589.8 µs | 337.0 → 481.1 µs |
| GDN norm/FP8 forward epilogue | 146.6 → 69.6 µs | 56.0 → 0 µs | 0 → 0 µs | 188.6 → 110.9 µs | 98.4 → 51.6 µs |
| GDN norm/FP8 with projection forward and backward | 1362.1 → 1202.9 µs | 275.7 → 234.0 µs | 2.3 → 2.3 µs | 1394.9 → 1230.2 µs | 226.2 → 224.5 µs |

Each trace uses one CUDA stream on one GPU. The diagnostic gaps include
instrumentation effects. Separate host call and wait timings are measured by
the GDN unprofiled harnesses, and are unavailable for the MLA runs. The two
timing scopes overlap. There are no distributed collectives in these component
calls. GDN scan peak reserved memory was 1512 → 1448 MiB. Short-conv peak
reserved was 1042 → 658 MiB.

## Numerical and execution checks

- **MLA.** Seven CUDA test cases passed, covering actual compiled pack/unpack
  at both representative shapes, odd dimensions, strided projections and
  gradients, and implicit and indexed positions. The initial CUDA run caught
  a Triton branch-local variable shape mismatch and an invalid FP32 reference
  view. Both were repaired and the full unit file passed. The actual MLA
  module under Inductor and GIFT selective activation checkpointing passed
  output, input-gradient, and projection-gradient comparisons. The maximum
  relative L2 difference on any compared tensor was 0.0011% at 4K and 0.0480%
  at 8K. The corresponding identical controls reached 0.00053% and 0.0462%.
  Each candidate trace shows two fused forward launches including the SAC
  recompute and one fused backward launch. Multi-rank CP was outside the
  single-device fixture. [4K parity](evidence/mla-4k-check.json) · [8K parity](evidence/mla-8k-check.json) · [4K trace receipt](evidence/mla-4k-profile.json) · [8K trace receipt](evidence/mla-8k-profile.json)
- **Direct GDN backward.** Dense and packed BF16, FP16, and FP32 passed
  exact output and all seven gradient comparisons. The dense component trace
  confirms two scan-output launches reduced to one and three W/U launches
  reduced to two. An independent review found that matching FLA's explicit
  `cu_seqlens_cpu=None` argument enables reuse of forward's cached packed
  chunk indices. All five targeted packed and metadata checks passed after the
  change. The dense timing path is unaffected by that change.
  [Benchmark receipt](evidence/gdn-direct-benchmark/result.json)
- **GDN short convolution.** The native capture's Q/K/V matched exactly.
  Input gradients differed at 2 of 67,108,864 elements with a maximum
  absolute difference of `3.64e-12`. Weight gradients differed at 7 of
  32,768 elements with a maximum absolute difference of `1.19e-7`.
  Five CUDA boundary cases passed on the first run. The synthetic crop
  had four near-cancellation elements outside the strict elementwise
  tolerance. Replaying the contributing head and taps in FP64 with the
  native BF16 rounding points bounded both paths' errors. The compiled
  crop rerun passed. The candidate trace confirms the fused gradient tile
  and removal of the replayed FLA gradient pack. [Crop check](evidence/gdn-short-conv-crop.xml) · [Benchmark receipt](evidence/gdn-short-conv-benchmark/result.json)
- **GDN output norm and FP8 projection.** Bytes, scales, projection output
  and gradients matched exactly in the isolated crop and the targeted
  head-width, multiplier, padding and dtype tests. The compiled projection
  accepted changing sequence lengths in one process. The test also covered
  strided activation and weight views with an FP32 gate. The actual GDN
  module passed exact output and gradient checks at S67 and S149 with both
  arms under the same dynamic Inductor and selective checkpointing. The
  first integration attempt compared compiled and eager arms and failed
  exact comparison. The matched rerun confirmed the crossing-block kernel
  in the trace and checked the fallback through projection and norm hooks.
  The crop trace shows the fused forward epilogue replacing separate norm,
  scale and quantization work. Backward uses FLA's original norm VJP. The installed CudaCherry
  0.1.0 row quantizer source SHA-256 was
  `e62efe4691fc3bd3fdea79443ca1150a0928c0994d6a4dfda1cb9abae26ecbbd`.
  [Module check](evidence/gdn-fp8-module.xml) · [Benchmark receipt](evidence/gdn-fp8-benchmark-final/result.json)

Focused Ruff and whitespace checks passed. The local macOS suite reported
105 passes and eight CUDA skips. The historical FlashQLA tests in that
suite could not initialize FLA under the repository's macOS Triton stub.
FlashQLA was excluded from the H100 comparisons.

## Precision diagnosis against the original baseline

This follow-up compares with the original unfused paths on the same H100
allocation, using component replay and fixed inputs and cotangents for each
comparison. FlashQLA stayed disabled. The numerical scope is one GPU per
component or module. Distributed training and long-run convergence remain
separate acceptance work.

### All three changes in the GDN module

The actual GDN module ran at the resolved 10+10 crop shape under Inductor
fullgraph and the crop's layer-0 selective activation checkpointing policy.
The replay uses captured production-data hidden states, QKV projection and
short-conv weights. It checks that the FP8 QKV projection reproduces the
captured QKV exactly. The remaining projections are seeded and the captured
value gradient supplies a fixed output cotangent. The control has all three
switches off. The candidate profiler confirmed all three custom ops ran.

The baseline repeat was bitwise exact. With all three changes enabled, the
output and six of eight parameter gradients were bitwise exact. All compared
tensors were finite. The input, QKV projection weight and short-conv weight
gradients differed.
The all-on and short-conv-only variants matched bitwise for every compared
tensor, which localizes the differences to short-conv in this replay.

| Tensor | Changed elements | Max absolute difference | Relative L2 |
| --- | ---: | ---: | ---: |
| Input gradient | 108 / 33,554,432 | `5.82e-11` | `6.79e-6` |
| QKV projection weight gradient | 253 / 33,554,432 | `7.45e-9` | `6.00e-6` |
| Short-conv weight gradient | 6 / 32,768 | `1.19e-7` | `3.16e-5` |

[Module receipt](evidence/gdn-numerics/result.json) · [Replay harness](gdn_numerics.py)

### GDN direct backward

The original native FLA replay and the direct solver had bitwise-identical
outputs and all seven gradients at the 10+10 crop. The SR008 H100 rerun of
the direct-backward suite passed all 12 cases, including compiled packed
sequences. [GPU test receipt](evidence/gdn-numerics/direct-scan.xml)

### GDN short-conv rounding and reduction

The stage probe's final input and weight gradients matched the uninstrumented
fused kernel for the real crop and the seeded synthetic crop. On both inputs
the fused main tile matched native FLA through BF16 preactivation, rounded
Q/K, FP32 inverse norm, packed BF16 gradient and FP32 post-SiLU derivative.
Feeding that derivative into native convolution backward produced the native
gradients exactly.

- **Input gradient.** The real crop differed at 2 of 67,108,864 elements
  (maximum absolute difference `3.64e-12`). Both differences, and the four
  previously identified synthetic cancellation outliers, came from the
  halo recomputation at the tile edge. The halo preserved the same rounded
  BF16 preactivation. Triton used a different FP32 addition tree for the
  halo's sum of squares. The resulting inverse norm differed by one or two
  representable steps in the contributing rows, shifting a packed BF16
  gradient to an adjacent value. Replaying those tap contributions in FP64
  reproduced the two paths' rounded input gradients and showed cancellation
  between opposite-sign taps.
- **Weight gradient.** For each of the seven real differences, both paths
  used identical BF16 inputs and FP32 post-SiLU derivatives at every tile.
  Triton used different time-axis reduction layouts. The resulting FP32
  partials differed by at most `2.84e-13`. The shared
  FP64 sum was near the midpoint between the rounded BF16 results. The
  largest final absolute difference was `1.19e-7`. In the synthetic case
  seven of eight weight differences followed the same pattern and one also
  received a changed halo derivative.

The fused path keeps FLA's BF16 cast points. The measured residuals arise
from FP32 arithmetic in the recomputed halo or the weight reduction, with
final rounding at the existing dtype boundaries.
[Stage and partial evidence](evidence/gdn-short-conv-numeric/partials.json)

### GDN norm and FP8 rounding

At the 10+10 crop the unfused and fused paths matched bitwise for all
33,554,432 FP8 payload bytes, 262,144 FP32 scales and 33,554,432 BF16
cotangent elements at the input to FLA's norm backward. Projection output
and all four gradients matched bitwise. The fused kernel preserves the
baseline BF16 store after norm and the BF16 product after the multiplier.
Both paths give the same scaled cotangent to the original FLA VJP.

A separate padded B2/S67/HV3/V96 test with an FP32 gate and a 1.375 multiplier
matched bitwise at the same boundaries. A control that omits the BF16 norm
store changed 813 FP8 bytes. Omitting the BF16 rounding of the backward
multiplier changed 34,697 VJP input elements. These controls show the parity
checks detect the rounding stages on inputs where they matter.
[Precision receipt](evidence/gdn-fp8-precision/result.json)

### MLA RoPE rounding and reduction

The original unfused PyTorch path was compiled with Inductor. Both variants
received the same Q/K projections and exactly the same gradients captured
at the cuDNN/SAC tail. At B2/S4096/H32 and B2/S8192/H64 the packed Q/K,
query gradient and nonrotary key gradient matched bitwise. A repeat of each
boundary variant with fixed operands was bitwise exact at both shapes.

The probe captured the FP32 shared-key head sum from Inductor's generated
reduction and the corresponding sum from the fused backward. At 4K the
sums differed at 52,827 of 524,288 elements, with maximum difference
`4.44e-16`. Applying the same FP32 inverse rotation and final BF16 cast
reconstructed both paths bitwise. Exactly seven differences survived the
cast, with maximum absolute difference `5.68e-14` and relative L2
`5.88e-8`. At 8K, 979 of 1,048,576 sums differed before the cast and the
entire shared-key gradient matched bitwise after it. Both compiled paths
keep the shared-key sum in FP32 through inverse rotation.

A separate attention-tail repeat fed identical packed tensors to the same
cuDNN path. It already produced query-gradient variation at both shapes.
The fixed-gradient RoPE replay separates that variation from the measured
shared-key reduction effect.
[4K receipt](evidence/mla-rope-numerics/confirmed/4096.json) · [8K receipt](evidence/mla-rope-numerics/confirmed/8192.json)

## Direct backward for the FP8 output norm

Enable `GIFT_GDN_DIRECT_OUTPUT_NORM_BWD=1` together with
`GIFT_GDN_FUSE_OUTPUT_NORM_FP8=1`. The comparison keeps the current fused
FP8 forward in both variants. The previous backward multiplies the incoming
gradient in PyTorch, replays FLA's gated-norm forward, and calls its backward.
The candidate reads the same saved `x`, `gate`, and `weight` plus the FP8
projection's raw dgrad. It recomputes inverse RMS in FP32 in registers and
stores `dX`, `dGate`, and one FP32 `dWeight` partial per CTA. A second
kernel reduces the partials without atomics. The launcher derives the tile,
grid, and addresses from the feature width, rows, tensor strides and SM count.

The production crop replay used `[2, 4096, 32, 128]` with the resolved
layer-0 SAC policy, Inductor fullgraph, native FLA, and CudaCherry FP8 GEMMs.
The inputs came from the production-tokenizer capture. The harness replayed
the scan, seeded the gate and output projection weights, and used captured
`dV` as the fixed output cotangent. The FP8 payload, scales, projection output,
and all four gradients matched bitwise. The isolated backward also matched
bitwise for every gradient. Both A/A checks were exact.

| Crop scope | Previous wall time | Direct wall time | Median paired change / A/A | Peak allocated, previous → direct | Peak reserved, previous → direct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compiled norm backward | 471.2 µs | 318.7 µs | −32.45% / +2.09% | 641.4 → 512.6 MiB | 740 → 610 MiB |
| Compiled FP8 tail with SAC | 1,464.5 µs | 1,371.2 µs | −6.02% / +0.36% | 737.4 → 672.6 MiB | 870 → 806 MiB |

The timing uses 24 paired calls after ten warmups per variant and a repeated
previous implementation as control. CUDA Graphs were disabled. Wall time is
synchronized component latency. Separate timings report host call and end-event
wait. The isolated host call changed from 446.3 to 219.0 µs and the wait from
25.0 to 99.6 µs. In the full tail they changed from 1,280.1 to 1,054.1 µs and
185.4 to 321.8 µs. The host wait includes pending GPU work at submission return.

Separate warmed traces show kernel duration sums of 246.4 → 146.5 µs for
the isolated backward and 1,198.7 → 1,094.4 µs for the full tail. They show
the separate multiplier, FLA norm replay, and native gradient passes give
way to the two Gluon kernels, while the FP8 norm quantize still runs in the
forward and SAC replay. The isolated traces have no copies or memsets.
Full-tail memsets total 2.27 and 2.34 µs. These replays have no cross-GPU
communication. Weight-gradient scratch rose from 66 to 256 KiB, while the
multiplied gradient, replayed norm output, and FP32 norm statistics no longer
need global buffers. Forward still saves only `x`, `gate`, and `weight`.

Padded width 128, strided width 127 with FP32 incoming gradient, and wide
width 513 with FP16 incoming gradient matched bitwise. A width-96 case with
FP32 gate and weight, Swish, and multiplier 1.7 had bitwise `dX` and `dGate`.
Its FP32 `dWeight` differed in 59 of 96 elements, with max absolute
difference `3.81e-6` and relative L2 `9.56e-8`. The previous and direct kernels
had similar relative L2 errors to the FP64 reference, `1.23e-7` and
`1.25e-7`. The different FP32 summation order can change results in heavily
cancelling inputs. The fused multiplier retains both existing casts, first to
the incoming gradient dtype and then to the norm output dtype. A control
that bypassed the first cast changed `dX` and `dGate` in the width-96 case.
Both kernels kept FLA's `dGate` cast through the norm input dtype.

Two additional compiled BF16 cases used three rows at widths 1 and 32,768,
the latter at FLA's 64 KiB feature boundary. Width 1 matched bitwise. At
width 32,768 both activation gradients matched bitwise. One of 32,768
weight-gradient elements differed by `0.00390625`, with relative L2
`1.66e-5`. Every compared tensor was finite.
[Width-bound receipt](evidence/gdn-output-norm-backward/width-bounds.json)

All 14 FP8 output-norm regression cases passed on H100, including the
new compiled padded/strided cases for BF16, FP16, and FP32 cotangents.
The pytest result is in the XML. The shell exit-code trap failed after
pytest completed. [GPU test receipt](evidence/gdn-output-norm-backward/unit.xml)

The crop CUBINs report 89 registers per thread for native FLA backward
and 168 for the Gluon kernel, with zero stack and local bytes for both.
The runs use 512 and 128 threads per CTA, respectively. Nsight Compute
returned `ERR_NVGPUCTRPERM`, so achieved occupancy and measured HBM
bandwidth were unavailable. Kernel durations above come from the separate
PyTorch device traces. [CUBIN resources](evidence/gdn-output-norm-backward/crop-cubin-resources.txt) · [Nsight error](evidence/gdn-output-norm-backward/ncu-candidate.csv)

The initial harness run rejected a valid layer-0 SAC configuration before
timing. The next run resolved SAC through GIFT's own policy and found the
cross-dtype cotangent discrepancy. The final run includes the autograd cast
that resolved it. These measurements establish a component gain on the SR008
H100 and keep the switch opt-in. Distributed training throughput and long-run
stability require a separate acceptance run.

[Final A/B receipt](evidence/gdn-output-norm-backward/measured-v3/result.json) · [Harness](gdn_output_norm_backward_bench.py) · [SR008 launcher](evidence/gdn-output-norm-backward/run.sh)

### Output norm backward at other GDN shapes

Four additional shape pairs ran on SR008 H100s in job
`84fd3579-7bd2-4969-ae47-b7af17b73ca2`. Both variants used the existing FP8
forward, native FLA input replay, Inductor fullgraph, and the resolved crop
layer-0 SAC policy for the FP8 tail. Each pair used identical operands. The
rank-0 capture supplied batch row 0 for B1, the two batches joined as
`[0|1, 1|0]` for S8192, duplicated value heads for H64, and the first 96
channels of each value head for D96. The harness rebuilt a seeded native
FP8 projection when its input dimension changed. The isolated backward
used the resulting actual BF16 projection dgrad. These are component
replays of adapted captures.

| `[B, S, H, D]` | Compiled scope | Median wall time, previous → direct | Paired latency change / A/A | Peak allocated, previous → direct | Peak reserved, previous → direct |
| --- | --- | ---: | ---: | ---: | ---: |
| `[1, 4096, 32, 128]` | Norm backward | 451.8 → 244.3 µs | −46.2% / +3.6% | 384.6 → 320.3 MiB | 450 → 386 MiB |
| `[1, 4096, 32, 128]` | FP8 tail with SAC | 1,263.7 → 1,048.3 µs | −17.3% / +0.4% | 448.6 → 416.3 MiB | 530 → 498 MiB |
| `[2, 8192, 32, 128]` | Norm backward | 606.3 → 453.0 µs | −25.6% / −1.0% | 1,154.1 → 896.3 MiB | 1,250 → 994 MiB |
| `[2, 8192, 32, 128]` | FP8 tail with SAC | 2,559.2 → 2,369.2 µs | −7.5% / −0.02% | 1,314.1 → 1,184.3 MiB | 1,570 → 1,442 MiB |
| `[2, 4096, 64, 128]` | Norm backward | 573.0 → 457.5 µs | −19.9% / +1.5% | 1,122.1 → 864.3 MiB | 1,186 → 930 MiB |
| `[2, 4096, 64, 128]` | FP8 tail with SAC | 2,508.1 → 2,328.7 µs | −7.2% / +0.2% | 1,250.1 → 1,120.3 MiB | 1,476 → 1,348 MiB |
| `[2, 4096, 32, 96]` | Norm backward | 639.6 → 328.2 µs | −48.8% / −1.7% | 521.1 → 424.2 MiB | 690 → 642 MiB |
| `[2, 4096, 32, 96]` | FP8 tail with SAC | 1,479.6 → 1,274.6 µs | −13.6% / +0.9% | 609.1 → 560.2 MiB | 740 → 692 MiB |

Each comparison used 24 alternating synchronized calls after ten warmups
per variant, with CUDA Graphs disabled. Paired changes are medians of
per-pair differences relative to the previous backward. Both variants
started from the same allocated and reserved memory in each scope.
The receipts contain separate CUDA event spans, host submission, and end
event waits. Device profiles were collected separately. Isolated backward
kernel duration sums changed from 131.8 to 75.4 µs for B1, 472.4 to
285.3 µs for S8192, 473.9 to 285.2 µs for H64, and 342.1 to 131.0 µs
for D96. The D96 trace uses FLA's generic backward launch, which took
241 µs versus 125 µs for the specialized D128 launch in the crop trace.
The candidate used 512 first-stage CTAs in all four shapes, with more
rows per CTA for S8192 and H64. The traces show the two Gluon kernels
replacing the multiplier, FLA norm replay and old gradient passes in each
case. The new backward and the FP8 forward/SAC replay were active in every
shape. The isolated traces contain no copies or memsets and the full-tail
traces record 2.1–2.8 µs of copies or memsets. There is no distributed
communication in either component.

The FP8 payload and scales matched bitwise in every A/B pair. The full-tail
output and all compared gradients were finite and bitwise equal in every
A/A and A/B pair. These are component latency and PyTorch allocator peaks.
Distributed training TPS and GIFT's native total memory remain separate
measurements.

[B1 receipt](evidence/gdn-output-norm-backward-scaling/b1-s4096-h32-d128/metrics/result.json) · [S8192 receipt](evidence/gdn-output-norm-backward-scaling/b2-s8192-h32-d128/metrics/result.json) · [H64 receipt](evidence/gdn-output-norm-backward-scaling/b2-s4096-h64-d128/metrics/result.json) · [D96 receipt](evidence/gdn-output-norm-backward-scaling/b2-s4096-h32-d96/metrics/result.json) · [Harness](gdn_output_norm_backward_scaling.py) · [Launcher](evidence/gdn-output-norm-backward-scaling/run.sh)

## Replay inputs and artifacts

The MLA harness uses captured production-tokenizer MTP hidden inputs and
upstream gradients with matched seeded MLA weights and causal positions. The
8K case joins rank 0 and rank 1 sequences and exercises 64 heads and Q-LoRA
rank 1536 at model dimension 4096. It isolates the MLA module from the rest
of the 700B model. The two GDN scan and convolution comparisons use the same
captured QKV projection and convolution weight. Scan setup uses matched
initialized gate projection and decay parameters and replays the captured
value gradient as the incoming scan gradient.
The FP8 benchmark reuses the scan output and real activations with seeded FP8
gate and output projection weights, replaying the captured value gradient as
the projection cotangent. That replay setup occurs before timing.

The scripts and compact receipts are next to this report. Raw logs, exit codes
and Chrome traces are under
`/workspace-SR008.nfs2/vkropotin/optimization-task/artifacts/linear-fixes/results`.
The benchmark repository on SR008 is the `linear-fixes` worktree under
`artifacts/linear-fixes/source`. The receipts contain fixture hashes, source
hashes, software versions and measured samples. Component results give the
isolated cost change. Training-step TPS and memory require their own run.
