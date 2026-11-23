# FLA and MLA group review

Updated 21 November 2026. The five changes were developed on the GIFT branch
`linear-fixes`, checked out at `optimization-task/.worktrees/linear-fixes/gift`.
The branch is based on original A at `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`.
Single-GPU checks, matched training measurements and update-11 profiles are
complete for the tested crop. The changes are uncommitted and final source
review remains open. [Worktree benchmark report](../../.worktrees/linear-fixes/runs/linear-fixes/report.md)

## Changes under review

All five switches default off. Output-norm backward requires the fused FP8
forward. The other four switches can be enabled independently. This group
uses native FLA throughout.

| Change and switch | Original behavior and previous optimization | Generalized implementation |
| --- | --- | --- |
| **GDN direct backward** · `GIFT_GDN_DIRECT_FLA_BACKWARD` | The opaque native FLA scan reconstructed autograd by replaying the forward, including an attention output the backward discarded. The historical output-skipping implementation used FlashQLA `external_gate`. | Rebuild the gate cumsum and KKT solve, then call the pinned native FLA backward. The replay skips the forward recurrence, discarded output and duplicate W/U computation. Forward and backward share FLA's chunk size and packed-index cache key. The stateless GIFT path inherits FLA limits such as key width ≤256 and V-head count divisible by Q/K-head count. [Implementation](../../.worktrees/linear-fixes/gift/torchtitan/models/common/attention/linear/gdn_fla_backward.py) |
| **GDN short-conv backward** · `GIFT_GDN_FUSE_SHORT_CONV_BWD` | Native FLA replayed the convolution forward, packed an intermediate gradient and ran the causal-conv backward. The previous fusion specialized the gradient tile for BF16, four taps and 128-wide Q/K and V heads. | Recompute convolution and SiLU/L2 derivatives in the gradient tile using the configured tap count and separate Q/K and V widths. Preserve FLA's normalization casts, packed boundaries and FP32 weight partials. Above 64 taps, reuse FLA's causal backward accumulation to bound tile overlap. The native FLA QKV forward supports head widths up to 256. [Implementation](../../.worktrees/linear-fixes/gift/torchtitan/models/common/attention/linear/gdn_short_conv.py) · [FLA forward](../../.worktrees/linear-fixes/gift/contrib/flash-linear-attention/fla/ops/common/fused_conv_qkv.py) |
| **GDN output norm → FP8** · `GIFT_GDN_FUSE_OUTPUT_NORM_FP8` | Gated RMSNorm wrote a BF16 activation, GDN multiplied it, then the FP8 projection quantized it. The previous fusion required V-head width 128, multiplier 2.0 and aligned token counts. | Write the CudaCherry payload and scales directly. Preserve the BF16 rounding points and use the configured multiplier. Keep a single-kernel path at width 128, handle quantization blocks that cross head boundaries, and use the projection's padding contract. With the direct-backward switch off, use FLA's original norm VJP. Ineligible projections and observable module hooks use the ordinary path. [Implementation](../../.worktrees/linear-fixes/gift/torchtitan/models/common/attention/linear/gdn_output_fp8.py) |
| **MLA RoPE + Q/K packing** · `GIFT_MLA_FUSED_ROPE` | PyTorch rotated Q and the shared K rotary slice, then concatenated them into the attention layout. The previous fusion required Q shape `[2, 4096, 32, 192]`, exact crop strides, and explicit crop-shaped positions. | Rotate directly into packed Q/K using the actual dimensions and strides. Support implicit, shared and per-batch positions. Backward reads strided gradients and reduces the shared K gradient in FP32 before inverse rotation. The compiled CUDA path checks the rotary-cache contract. Eager and DTensor inputs use the PyTorch path. [Implementation](../../.worktrees/linear-fixes/gift/torchtitan/models/common/attention/softmax/mla_rope_pack.py) |
| **GDN output-norm backward** · `GIFT_GDN_DIRECT_OUTPUT_NORM_BWD` | With fused FP8 forward enabled, backward scaled the projection gradient, replayed FLA gated RMSNorm, and ran the native norm backward. | Fuse incoming gradient scaling with FP32 inverse RMS recomputation and the per-tile gradients. Reduce FP32 weight partials in a second kernel. Preserve the incoming and activation dtype casts and derive launch geometry from shape and strides. [Implementation](../../.worktrees/linear-fixes/gift/torchtitan/models/common/attention/linear/gdn_output_norm_backward.py) |

## Measured component cost on the 10+10 crop

The resolved `gift_3_5/100b_crop8` configurations from original A and
historical production use model width 4096, local batch 2 and sequence length
4096. GDN uses 16 Q/K heads and 32 V heads, both width 128, with four conv taps
and norm multiplier 2.0. MLA uses 32 heads, Q/K width 128 + 64 rotary, V width
128, Q-LoRA disabled and KV-LoRA rank 512. [10+10 run identities](../../runs/fusion-followup/results/original-a-vs-production.json)

Each result shows **wall / CUDA latency in ms · peak allocated / reserved
memory in MiB**. Timings are medians of warmed, unprofiled component calls.
Graph replay times device work on fixed inputs. Event spans can include gaps
during host submission. Allocator peaks include resident inputs and weights
and exclude CUDA Graph capture. Each component ran on one H100 80 GB, with
the comparisons split across GPUs 0–3 in job
`84fd3579-7bd2-4969-ae47-b7af17b73ca2`. The runtime used PyTorch
`2.15.0.dev20260822+cu130`, CUDA 13.0, Triton 3.8.0, CudaCherry 0.1.0
and FLA `1d2281ff1b7988d8070f00dba5ffa9747ae3a60e`.

| Component | CUDA timer | Original baseline | Previous implementation | Generalized implementation |
| --- | --- | ---: | ---: | ---: |
| MLA forward + backward | Event | 10.323 / 10.307 · 1,868.24 / 2,078 | 9.954 / 9.937 · 1,865.74 / 2,046 | 9.975 / 9.958 · 1,865.74 / 2,046 |
| GDN scan forward + backward | Graph | 3.134 / 2.524 · 1,508.56 / 1,512 | Same native FLA recompute control | 2.715 / 2.147 · 1,444.56 / 1,448 |
| GDN short-conv forward + backward | Graph | 1.261 / 0.883 · 1,041.25 / 1,042 | 1.209 / 0.797 · 656.25 / 658 | 1.106 / 0.733 · 656.25 / 658 |
| GDN norm → FP8 packing only | Graph | 0.2854 / 0.1520 · 417.00 / 482 | 0.1394 / 0.0638 · 353.00 / 418 | 0.1457 / 0.0680 · 353.00 / 418 |
| GDN norm → FP8 with projection forward + backward | Event | 1.623 / 1.616 · 673.07 / 772 | 1.450 / 1.443 · 673.07 / 804 | 1.456 / 1.448 · 673.07 / 804 |

MLA measured all three implementations in the same run. Short-conv and FP8
use earlier unfused controls from the same fixtures and harnesses, and the
previous and generalized columns are paired directly. Those rows summarize
two comparisons. The GDN scan starts with prepared Q/K/V and measures the
direct backward change alone. The historical FlashQLA scan was outside the
comparison. The three other historical kernels were byte-identical between
their donor revision `fe5950b56de5b2cec7ff4cd063f7338c98200c45` and production
`2c67c182e8de6145f3796d62203714ab1b815446`. [Controls, sample series, source hashes and profiles](../../.worktrees/linear-fixes/runs/linear-fixes/report.md#previous-kernels-on-the-1010-crop)

- **MLA.** The previous and generalized versions have equal memory peaks.
  The paired wall median favored the generalized kernel by 1.28 µs, within
  control variation. Both traces show two forward packs including SAC replay,
  and one backward unpack. Their fused kernel sums were 348.51 → 341.18 µs
  in the separate traces. [Timing and trace evidence](../../.worktrees/linear-fixes/runs/linear-fixes/evidence/mla-4k-crop-followup/timing.json) · [Trace comparison](../../.worktrees/linear-fixes/runs/linear-fixes/evidence/mla-4k-crop-followup/trace-comparison.json)
- **GDN scan.** Direct backward reduced wall time by 13.4%, graph time by
  14.9% and both memory peaks by 64 MiB. The earlier pair measured a 14.2%
  wall reduction. Traces show output scan launches going from two to one
  and W/U launches from three to two. [Receipt](../../.worktrees/linear-fixes/runs/linear-fixes/evidence/gdn-direct-10plus10/result.json)
- **GDN short-conv.** Both fused implementations saved 385 MiB of peak allocated
  memory versus the original path. The new path reduced graph replay time by
  8.0% versus the previous fusion in both runs. The wall difference overlapped
  identical-control variation. [Original control pair](../../.worktrees/linear-fixes/runs/linear-fixes/evidence/gdn-short-conv-benchmark/result.json) · [24-sample specialized pair](../../.worktrees/linear-fixes/runs/linear-fixes/evidence/gdn-short-conv-specialized/repeat-24.json)
- **GDN norm → FP8.** Generalized packing took 4.1 µs longer in graph replay
  than the previous fusion (+6.5%). Both saved 64 MiB of peak allocated memory
  in the packing-only scope. The full projection wall difference was within
  identical-control variation and both fused implementations had equal
  memory peaks. [Original control pair](../../.worktrees/linear-fixes/runs/linear-fixes/evidence/gdn-fp8-benchmark-final/result.json) · [Specialized pair](../../.worktrees/linear-fixes/runs/linear-fixes/evidence/gdn-fp8-specialized-vs-generalized/result.json)

Historical standalone 10+10 runs of the specialized kernels on compiled S +
GPU selector measured +0.36% TPS for MLA, +0.25% for GDN short-conv and +0.38%
for GDN norm → FP8 versus the first control. Against the repeat control the
differences were +0.284%, +0.173% and +0.303%. These are one-window results on
the old implementations. The historical +16.40% 10+10 result measures the
whole production recipe against original A, from 154,919 to 180,325 tokens/s.
[Historical standalone sweep](../../runs/kernel-fusion/report.md#native-1010-measurements) · [Full recipe comparison](../../runs/fusion-followup/report.md#original-a-versus-promoted-production)

The output-norm backward component used `[2, 4096, 32, 128]` with the fused
FP8 forward enabled in both arms. Compiled backward wall time fell from
471.2 to 318.7 µs (−32.45% paired median), with allocated and reserved
peaks falling from 641.4 to 512.6 MiB and 740 to 610 MiB. The compiled FP8
tail with SAC took 1,464.5 → 1,371.2 µs (−6.02%). Four additional shapes
reduced isolated backward wall time by 19.9–48.8% and the compiled tail by
5.9–17.3%. [Crop result](../../.worktrees/linear-fixes/runs/linear-fixes/report.md#direct-backward-for-the-fp8-output-norm) · [Shape results](../../.worktrees/linear-fixes/runs/linear-fixes/report.md#output-norm-backward-at-other-gdn-shapes)

## Matched training results against original A

Both campaigns used the same eight H100 80 GB GPUs, production-tokenizer
`data-002`, seed 2048, local batch 2, sequence length 4096, global batch 512,
32 microbatches per update, dense FSDP8 and TP/PP/CP1. They preserved FP8,
compile, SAC, MTP and EMA, with native FLA and zero batch prefetch. The sparse
meshes were EP8/eFSDP1 and EP4/eFSDP2. Baseline A used all five switches off.
`combined` enabled all five. TPS uses global physical input tokens divided
by elapsed time across the measured window. Memory is the maximum native
GIFT `memory: … GiB` across all ranks and measured updates. All arms passed
the resolved-config and route checks. The measured windows had stable compiler
observations and zero allocator retries and OOMs.

### 30+30 per-fix matrix

Each arm ran 30 warmup updates and measured updates 31–60 without profiling.
The TPS delta uses the baseline on the same topology.

| Arm | EP8 TPS | EP8 TPS Δ | EP8 memory, GiB | EP4 TPS | EP4 TPS Δ | EP4 memory, GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline A | 144,474.58 | — | 72.31 | 144,171.70 | — | 73.69 |
| GDN scan backward | 144,380.62 | −0.07% | 72.35 | 144,833.79 | +0.46% | 73.77 |
| GDN short-conv backward | 144,048.38 | −0.29% | 72.27 | 144,693.53 | +0.36% | 73.61 |
| GDN norm → FP8 | 143,799.56 | −0.47% | 72.27 | 144,608.14 | +0.30% | 73.56 |
| MLA RoPE + packing | 143,910.20 | −0.39% | 72.29 | 144,697.08 | +0.36% | 73.77 |
| GDN norm → FP8 + norm backward | 144,406.91 | −0.05% | 72.39 | 144,968.59 | +0.55% | 73.56 |
| Combined | 145,016.18 | +0.37% | 72.24 | 146,342.98 | +1.51% | 73.53 |

Adding norm backward to the FP8 forward arm changed TPS by +0.42% on EP8
and +0.25% on EP4. Its peak memory changed by +0.12 GiB and 0.00 GiB,
respectively. Step 50 was the sole Tukey outlier in each arm. Scheduled
Python GC ran on every rank, and TPS fell 4.0–6.8% below the mean of the two
neighboring steps. Excluding step 50 in a diagnostic recalculation from the
step logs gave combined gains of +0.37% on EP8 and +1.53% on EP4.

### Repeat 10+10

Four fresh unprofiled runs measured updates 11–20. EP8 ran baseline then
combined. EP4 reversed that order.

| Topology | Arm | Global TPS | TPS Δ | Peak total memory, GiB | Memory Δ, GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| EP8 | Baseline A | 155,744.76 | — | 70.34 | — |
| EP8 | Combined | 157,734.66 | +1.28% | 70.24 | −0.10 |
| EP4 | Baseline A | 146,575.86 | — | 73.38 | — |
| EP4 | Combined | 147,941.54 | +0.93% | 73.34 | −0.04 |

EP8's middle half of measured steps spanned about 0.17% of median TPS in
both arms. EP4 baseline spanned 0.38%, and EP4 combined spanned 1.35%.
EP4 combined reached 143,032 TPS on step 12 against a median of 149,068 TPS.
Excluding that step from both EP4 series for diagnosis gave +1.30% by the
harmonic mean of step TPS. The table includes all ten steps. Combined gained
TPS on both topologies in both campaigns. The larger gain switched from EP4
in 30+30 to EP8 in 10+10. Each campaign has one run per arm, leaving the size
of the topology effect and the small standalone TPS losses unresolved.
[10+10 result and step series](../../.worktrees/linear-fixes/runs/linear-fixes/artifacts/timing-10x10-001.json)

### Matched update-11 profiles

Four 10+1 captures recorded all 32 microbatches and the optimizer on every
rank. The table reports per-rank GPU kernel duration sums over the update,
as medians across eight ranks. Percentages are medians of paired rank changes.

| Matched region | EP8 baseline → combined, ms | EP4 baseline → combined, ms |
| --- | ---: | ---: |
| GDN scan backward | 403.890 → 330.310 (−18.2%) | 402.895 → 330.205 (−18.1%) |
| GDN short-conv backward | 143.096 → 114.283 (−20.0%) | 142.762 → 114.123 (−20.0%) |
| GDN output norm → FP8 | 67.759 → 33.465 (−50.6%) | 75.489 → 35.608 (−52.7%) |
| MLA RoPE + Q/K packing, enclosing MLA regions | 401.610 → 358.496 (−10.6%) | 399.743 → 364.331 (−8.7%) |
| GDN output-norm backward | 46.722 → 27.694 (−40.7%) | 46.732 → 27.793 (−40.5%) |

The backward analysis separates the three original `fla_vjp` call types by
correlated kernel launches. Each row matches 192 calls per rank. The norm
backward row includes the original multiplier launch outside `fla_vjp`.
The norm → FP8 row includes 192 forward calls and 192 SAC replays per rank
and ends before the projection weight pack and GEMM. MLA uses the same
projection and preparation boundaries around cuDNN attention in both arms.
Its inverse RoPE and gradient packing subset took approximately 18.3 → 8.7 ms
on both topologies. [Backward matches](../../.worktrees/linear-fixes/runs/linear-fixes/artifacts/profile-analysis/gdn-backward-matched.json) · [FP8 forward matches](../../.worktrees/linear-fixes/runs/linear-fixes/artifacts/profile-analysis/gdn-fp8-forward-matched.json) · [MLA boundaries and matches](../../.worktrees/linear-fixes/runs/linear-fixes/artifacts/profile-analysis/mla-matched.json)

On the profiled update, combined shortened the full step by 249 ms on EP8
and 91 ms on EP4. Communication without compute overlap changed by −26 ms
and +54 ms, respectively. Dense FSDP reduce-scatter kernel sums changed by
−58 ms and +132 ms. The scopes can overlap. These observations describe
update 11, and the direction of the longer 30+30 topology gap remains
unexplained. [All-rank trace breakdown](../../.worktrees/linear-fixes/runs/linear-fixes/artifacts/profile-analysis/paired-all-rank.json)

## Completed correctness checks

| Change | Observed results |
| --- | --- |
| GDN scan | Exact output and seven-gradient parity for dense and packed BF16, FP16 and FP32. Five targeted packed and metadata checks passed after matching FLA's chunk-index cache key. |
| GDN short-conv | The crop replay matched the previous specialization elementwise for Q/K/V, dQKV and conv dW. Versus native FLA, Q/K/V were exact and maximum absolute differences were `3.64e-12` for dQKV and `1.19e-7` for dW. Five boundary cases passed. An FP64 local reference bounded the synthetic crop's near-cancellation differences, and the compiled crop rerun passed. |
| GDN norm → FP8 | FP8 bytes, scales, projection output and four gradients matched bitwise on the crop. Targeted tests covered widths, multipliers, padding and a strided FP32 gate. The compiled GDN module passed at S67 and S149 in the same process with matched Inductor and SAC, including hooks and fallback. |
| MLA | Seven focused checks passed for compiled dispatch and GPU RoPE, including strided inputs, signed positions and different dimensions and dtypes. At 4K the previous and generalized MLA outputs matched bitwise. The largest relative L2 gradient difference was `9.57e-6`. At B2/S8192/H64 with Q-LoRA rank 1536, the generalized version passed the actual compiled MLA replay versus the unfused path. Two runs measured 0.7% and 2.7% lower wall time and 4 MiB less peak allocated memory. Multi-rank MLA CP remains unverified. |
| GDN output-norm backward | The crop's FP8 payload, scales, projection output and four gradients matched bitwise against native FLA backward with the same fused FP8 forward. All four scaling shapes were finite and bitwise equal in the measured comparisons. Width-96 and width-32,768 edge cases had exact activation gradients and relative L2 `dWeight` differences of `9.56e-8` and `1.66e-5`. [Norm-backward checks](../../.worktrees/linear-fixes/runs/linear-fixes/report.md#direct-backward-for-the-fp8-output-norm) |

Focused Ruff and whitespace checks passed. The local suite recorded 105 passes
and eight CUDA skips. Historical FlashQLA cases could not initialize FLA under
the macOS Triton stub. The worktree report records failed attempts and the
corresponding successful GPU reruns where applicable. [Numerical checks and receipts](../../.worktrees/linear-fixes/runs/linear-fixes/report.md#numerical-and-execution-checks)

The subsequent H100 precision diagnosis used the unfused baseline and fixed
incoming gradients. For short-conv, both real input-gradient differences
came from a different FP32 sum in the halo Q/K normalization, followed by
the original BF16 cast. All seven real weight-gradient differences first
appeared in FP32 tile reductions with identical operands. For MLA, applying
the same FP32 inverse rotation and BF16 cast to the two captured FP32
shared-key sums reproduced all seven 4K gradient differences (maximum
absolute difference `5.68e-14`). At 8K, all isolated RoPE outputs and
gradients matched bitwise. FP8 matched the original BF16 rounding stages,
bytes, scales and norm-backward input. With all three GDN switches enabled,
the crop module output remained bitwise exact and short-conv alone
reproduced every gradient difference. Every compared tensor was finite.
The H100 direct-backward regression rerun passed all 12 cases.
[Precision diagnosis and receipts](../../.worktrees/linear-fixes/runs/linear-fixes/report.md#precision-diagnosis-against-the-original-baseline)

## Remaining work

- [ ] **Final code review.** Review the uncommitted `linear-fixes` patch
  against original A. Verify native FLA solver and metadata assumptions,
  gradient accumulation and rounding, CudaCherry payload and projection
  lifetime, norm-backward dispatch with fused FP8 forward, and MLA dispatch,
  fallback and SAC contracts. Review the complete working tree diff,
  including untracked kernel and test files, and compare it with the tested
  source fingerprint below. [Measurement and approval protocol](../../planning/validation.md#memory-and-final-approval)

## Evidence locations

The training and profiling campaigns ran on SR008 job
`84fd3579-7bd2-4969-ae47-b7af17b73ca2`. The frozen GIFT source SHA-256 was
`dea5cba1edce08c8e77725a486dd9149bfd5a21a5a31bc873efec0ea9254841f`.
The raw campaign root is
`/workspace-SR008.nfs2/vkropotin/linear-accept-001/runs/linear-fixes/artifacts/raw`.
It contains the `acceptance-30x30-001`, `timing-10x10-001` and
`profile-10x1-mb32-001` campaigns and their per-arm siblings. The archived
`acceptance-30x30-001/matrix-results.json` has SHA-256
`8cd23515bdf7bedfdccecf48405a3caf6374a9879d36ec90e2108d98a69ed32a`.
The profile files are under each arm's `native/profiling/traces/iteration_11`.

The older component artifacts on SR008 use
`/workspace-SR008.nfs2/vkropotin/optimization-task/artifacts/linear-fixes`,
with the synchronized component checkout in `source` and raw data in `results`.
