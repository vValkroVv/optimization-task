# 100B C8 isolated-head phase traces

At 128k, the corrected candidate reduced elapsed time from first GPU activity to FP32 reduce-scatter entry from 1326.01 to 1234.17 ms for the median of seven GPUs other than physical GPU 6. GPU 6 took 48.62 ms longer than in the baseline. It entered reduce-scatter 181.41 ms after the first candidate rank and spent 6.74 ms in the collective. The other seven ranks spent 166.61–188.14 ms in that collective, and all eight finished within 0.064 ms. Adjacent unprofiled rank-max medians were 1379.01 ms baseline and 1375.82 ms candidate.

The large slowdown did not recur in the reversed-order pair with physical GPUs 0 and 6 swapped. The candidate reached 1256.01 ms against 1360.43 ms baseline in adjacent unprofiled samples, an 8.31% throughput gain for this pair. CUDA UUIDs verified physical GPU 6 at logical rank 0 in both arms. It entered reduce-scatter at 1248.01 ms in the candidate, 12.00 ms after the median of the other seven GPUs and 3.40 ms before the last arrival from physical GPU 1. The large original slowdown was absent, so the swap provides no observation of its rank assignment when the slowdown recurs. [Original all-rank receipts](phase-main.json) · [Swap all-rank receipts](phase-swap.json)

## The slowdown followed physical GPU 6

A separate candidate-only arm kept the swapped mapping for 10 warmups and 32 measured calls. Adjacent unprofiled rank-max latency rose from a median of 1255.92 ms over calls 1–5 to 1396.95 ms over calls 27–32. Sample 26 was traced between 13:12:10.873 and 13:12:12.269 UTC and measured 1395.96 ms.

Physical GPU 6, now logical rank 0, entered reduce-scatter at 1389.02 ms after its first GPU activity. The other seven had a median of 1242.87 ms. The entry spread reached 160.67 ms, versus 0.076 ms for all-gather exits. GPU 6 spent 6.34 ms in reduce-scatter and the other GPUs' median was 152.53 ms. Exits were within 0.059 ms. Physical GPU 6 had 0.38 ms of pre-collective internal GPU gaps, versus the other ranks' median of 0.36 ms.

The late trace repeated the kernel-duration skew under the new rank mapping. GPU 6 versus the other-rank median took 435.36 vs 384.05 ms in projection, 463.11 vs 394.03 ms in dHidden, 421.03 vs 393.65 ms in dW, and 59.02 vs 54.24 ms in CE. GPU 6's thermal clock-event flag appeared in 6 of 42 observations during the first five calls and 54 of 56 during calls 27–32. The sample-26 bracket added 101.6 ms to its NVML thermal-policy counter. Physical GPU 0, which took logical rank 6, showed zero thermal flags in the corresponding 287 observations. [Extended swapped run](phase-late.json) · [Hardware telemetry](../telemetry/README.md)

## Corrected 128k phase breakdown

The typical column is the median of the seven GPUs other than physical GPU 6 for the row shown. The activity rows give GPU duration sums in milliseconds. The elapsed row includes all work and gaps between the rank's first GPU activity and its reduce-scatter entry.

| Activity | Baseline typical | Candidate typical | Baseline GPU 6 | Candidate GPU 6 |
| --- | ---: | ---: | ---: | ---: |
| Projection | 385.80 | 382.09 | 397.74 | 434.51 |
| dHidden GEMM | 387.14 | 392.14 | 397.45 | 466.03 |
| dW GEMM | 366.36 | 393.47 | 372.10 | 434.32 |
| Baseline FP32 dW staging | 17.94 | 0 | 18.02 | 0 |
| CE kernels and gradient join | 145.40 | 54.78 | 150.24 | 59.91 |
| First GPU activity to reduce-scatter entry | 1326.01 | 1234.17 | 1356.72 | 1405.34 |

Both arms used the same eight projection GEMM launches and eight dHidden GEMM launches, with matching names and launch grids. On GPU 6 the extra candidate time accumulated across those same kernels, dW and CE. Inside the pre-collective device span, total uncovered gaps were 0.33–0.37 ms across candidate ranks. All-gather exits were within 0.06 ms baseline and 0.12 ms candidate. Local GPU work accounts for the slow arrival. The trace measures device work and collective arrival. Device power and thermal behavior require the accompanying hardware samples.

## Timing across lengths

Each unprofiled median is a median of nine per-sample maxima across the eight ranks. The traced figure uses the same timer for sample 6. The memory column is peak native GIFT total across unprofiled ranks and samples. The historical counterbalanced full-head grid measured the speedup separately. [Final grid](../../full-head/final-summary.json) · [Corrected 128k grid](../../full-head/corrected-100b-c8-128k-summary.json)

| Length | Pair order | Unprofiled A / candidate, ms | Traced A / candidate, ms | Peak A / candidate, GiB |
| --- | --- | ---: | ---: | ---: |
| 4k | A then candidate | 65.15 / 50.82 | 64.96 / 49.98 | 7.62 / 5.71 |
| 64k | Candidate then A | 720.81 / 686.37 | 672.68 / 637.62 | 24.50 / 12.80 |
| 128k | A then candidate | 1379.01 / 1375.82 | 1371.86 / 1413.28 | 42.76 / 20.36 |
| 128k, GPUs 0 and 6 swapped | Candidate then A | 1360.43 / 1256.01 | 1356.29 / 1257.91 | 42.76 / 20.36 |
| 256k | Candidate then A | 2760.79 / 2772.92 | 2747.76 / 2763.01 | 78.27 / 35.44 |

At 64k, the traced call was 6–7% shorter than the adjacent unprofiled median in both arms. In the initial 128k pair, the unprofiled median shifted from 1364.54 to 1435.90 ms in the baseline and from 1337.45 to 1431.20 ms in the candidate between the samples before and after the trace. These traces support kernel and dependency attribution. The unprofiled pairs measure isolated-head elapsed time under their respective conditions. The 256k baseline logged one allocation retry per rank before sample 1 and no further retries during measurement. All other arms had zero allocation retries.

The 256k candidate showed a similar wait to the initial 128k pair. Median elapsed time to reduce-scatter across GPUs other than GPU 6 fell from 2703.71 to 2453.46 ms, while physical GPU 6 took 2754.92 ms in the candidate. Candidate ranks entered reduce-scatter across 317.47 ms, with all-gather exits within 0.06 ms.

## CE, copies and gradient GEMMs

All durations are medians across eight ranks. Each category is aggregated independently.

| Length | Folded rows | Folded logits elements | Baseline CE forward / backward / join, ms | Candidate CE, ms | Baseline dW GEMM + staging, ms | Candidate dW GEMM, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4k | 1,024 | 131,334,144 | 1.47 / 1.83 / 1.46 | 1.64 | 11.30 + 16.26 | 16.33 |
| 64k | 16,384 | 2,101,346,304 | 22.60 / 25.07 / 23.56 | 26.28 | 178.87 + 17.15 | 188.22 |
| 128k | 32,768 | 4,202,692,608 | 49.23 / 51.67 / 44.47 | 54.81 | 367.88 + 17.95 | 393.89 |
| 256k | 65,536 | 8,405,385,216 | 92.11 / 99.93 / 88.65 | 111.02 | 756.39 + 18.51 | 769.14 |

At 4k, materializing the FP32 weight gradient directly removes 16.26 ms of baseline GFSDP copy/add staging for the fixed 128256-by-4096 weight matrix and eight chunks. The direct FP32 dW GEMM is 5.03 ms slower, leaving about 11.23 ms saved on dW and its staging. CE saves 3.12 ms in this trace. Projection changes from 11.24 to 10.94 ms and dHidden from 10.21 to 10.27 ms. Baseline dW staging stays within 16.26–18.51 ms over the tested lengths while the three GEMMs and CE scale with folded rows. The fixed staging saving has a larger share of head time at short lengths. At longer lengths the absolute CE saving grows, and longer GEMMs plus group waiting determine the observed share. [4k all-rank receipts](phase-short.json)

At 4k and 64k, baseline SplitBackward0 → aten::cat used eight CatArrayBatchedCopy_vectorized launches. At 128k and 256k, it used sixteen device-to-device copies of 4,202,692,608 and 8,405,385,216 bytes each, respectively. The pinned ATen CUDA cat checks the signed-32-bit index range of the output before selecting the batched copy and otherwise copies each tensor into its slice. Observed join duration grew from 23.56 to 44.47 to 88.65 ms across 64k, 128k and 256k. The baseline compiled forward used 32 Triton launches at 64k and 48 at 128k and 256k. Its backward used 16 launches at all three lengths. Candidate CE used eight exact_ce<8, 256, 1, 32, 0, true, true, true, false> launches and sixteen small reductions at each length. The main grid scaled from [65536, 2, 1] to [131072, 2, 1] to [262144, 2, 1]. The observed CE dispatch and launch count remained the same across the 2³² boundary. [ATen CUDA cat](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/Shape.cu#L521-L618) · [Signed index limit](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/CanUse32BitIndexMath.h#L10)

Projection performs [R, 4096] × [4096, 128256], dHidden uses [R, 128256] × [128256, 4096], and dW contracts R into [128256, 4096]. R is the folded row count per chunk. Each arm ran eight kernels per GEMM phase at the tested lengths. Candidate projection at 64k used a tuned descriptor with a 168×128 tile. Baseline projection used a 192×192 tile, and both arms used the same nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN projection at 128k and 256k. Both arms' dHidden changed from 320×128 NNT at 64k to 192×192 NNN at 128k and 256k. Candidate dW used beta zero for chunk one and beta one thereafter. At 4k a tuned 128×160 tile wrote the FP32 destination. At 256k the pinned tune used a 256×128 beta-one tile. Names, launch grids and runtime algorithm descriptors are in the receipts.

## Capture and verification

- Each arm ran the isolated eight-rank GFSDP head using the pinned policy, source and tiled 100B fixture. Checks examined each rank's token counts, loss components, source hash, active CE, BF16 all-gather, FP32 SUM reduce-scatter, gradient paths and manifest digests.
- The PyTorch CPU/CUDA profiler started after sample 5 and the original barrier and device sync followed. A CPU annotation brackets the original measured call, including its ending CUDA sync. The profiler stopped after the original timer read the clock. Shape and stack collection were off. The compact receipts retain each raw trace path, SHA-256 and UTC boundaries.
- CUDA runtime and driver correlation IDs link kernel activities to host operators. Duration sums can overlap across streams. GPU span, busy interval union, internal gaps and synchronization events are recorded per rank. Host submit and wait refer to measured wall intervals on the controlling rank. NCCL durations include peer waits and may overlap other work. The trace does not resolve all CPU time to individual Python frames.
- The first 128k attempt failed before CUDA because the wrapper could not import the preserved rank launcher. The corrected path passed the import check and both 128k arms produced eight traces. The first failure remains under remote runs/profile-128k. No training-source files changed.
- This replay covers only the isolated head, including hidden z-loss and native GFSDP. It does not measure whole-model training throughput.

[Launch driver](run_profile.py) · [Rank wrapper](profile_rank.py) · [Trace analysis](analyze_phase.py) · [Pinned grid policy](../../full-head/final-policy.json)
