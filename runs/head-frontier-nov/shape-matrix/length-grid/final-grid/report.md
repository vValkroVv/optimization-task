# Isolated LM head on the full sequence grid after plan calibration

Compare compiled Baseline A at revision `e9b7cdbb` with the tuned consuming CUDA CE and cuBLASLt head. Each cell uses one local sequence, main plus MTP1, and eight H100s on SR008. The three configurations are 100B at two and eight chunks and 700B at eight chunks, with lengths from 4,096 to 262,144 positions in powers of two. The matrices have vocabulary 128,256 and hidden widths 4,096 and 7,168. The production chunk shape has `2 × sequence / chunks` folded rows. The fixture repeats the stored 4,096-position tile, using seeded activations and weights.

The timer includes the ordinary BF16 weight all-gather, head projection, weighted CE, dHidden, FP32 weight-gradient accumulation/reduction, and the FP32 reduce-scatter. The 100B fixture includes hidden z-loss. Steady-state wall time and peak GIFT total refer to this isolated head. Whole-model TPS and memory are outside this benchmark.

At a fixed sequence length and chunk count, throughput gain is `100 × (baseline_ms / tuned_ms − 1)`. The memory change is `tuned_GiB − baseline_GiB`; its percentage uses the baseline peak as denominator. A CUDA OOM has no successful steady-state time or peak to use in either ratio.

[Previous full-head sweep](../report.md) · [Existing long-context calibrations](../region-tuning/report.md)

The calibration covered all 21 cells. For the weight gradient, the benchmark selected ten plans against their current controls, retained three previously calibrated routes and kept the native generic route in eight cases. The final C2 and C8 catalogs are separate because an overwrite followed by one update and an overwrite followed by seven updates weight the two GEMMs differently. At 4,096 folded rows, the eight ranks actually dispatched the selected dW plan for 100B C2/4k and the native generic route for 100B C8/16k. Projection and dHidden use a shared descriptor-based catalog. Seven more projection plans cleared the component timing, memory and full-output bitwise checks, in addition to the retained earlier calibrations. The tuned consuming CE remains selected throughout. [Weight-gradient calibration](dw/README.md) · [Projection, dHidden and CE](linear/README.md)

Each table uses the two fresh eight-rank repeats per variant. The 100B C8/128k row uses the corrected policy retest, and the other 20 rows use the full sweep. The second pair reverses the first pair's order. Each repeat has allocator priming, ten warmups and ten measured iterations. Each iteration takes the synchronized wall time of its slowest rank. The time shown is the median of the two repeat medians. Memory is the highest GIFT total across the eight ranks, measured iterations and both repeats, in GiB per rank. The initial launcher attempt stopped before measurement; the working run initialized the image's CUDA compatibility libraries through its existing shell profile. [Eight-rank protocol](full-head/README.md)

### 100B C2

| Sequence | Baseline A + compile (ms/GiB) | Our CE + cuBLASLt (ms/GiB) | Throughput gain | Memory change |
|---:|---:|---:|---:|---:|
| 4k | 51.76 / 8.626 | 46.38 / 6.477 | +11.6% | -24.9% (-2.148 GiB) |
| 8k | 92.51 / 11.946 | 83.41 / 7.727 | +10.9% | -35.3% (-4.219 GiB) |
| 16k | 176.75 / 18.626 | 162.59 / 10.149 | +8.7% | -45.5% (-8.477 GiB) |
| 32k | 349.59 / 32.495 | 324.38 / 15.073 | +7.8% | -53.6% (-17.422 GiB) |
| 64k | 715.92 / 59.704 | 670.49 / 24.880 | +6.8% | -58.3% (-34.824 GiB) |
| 128k | OOM | 1235.21 / 44.569 | — | — |
| 256k | OOM | OOM | — | — |

### 100B C8

| Sequence | Baseline A + compile (ms/GiB) | Our CE + cuBLASLt (ms/GiB) | Throughput gain | Memory change |
|---:|---:|---:|---:|---:|
| 4k | 65.04 / 7.610 | 51.09 / 5.696 | +27.3% | -25.2% (-1.914 GiB) |
| 8k | 104.38 / 8.548 | 87.02 / 6.165 | +20.0% | -27.9% (-2.383 GiB) |
| 16k | 190.27 / 10.794 | 163.67 / 7.122 | +16.2% | -34.0% (-3.672 GiB) |
| 32k | 353.06 / 16.323 | 311.94 / 9.018 | +13.2% | -44.8% (-7.305 GiB) |
| 64k | 770.67 / 24.489 | 669.61 / 12.790 | +15.1% | -47.8% (-11.699 GiB) |
| 128k | 1362.37 / 42.751 | 1358.65 / 20.349 | +0.3% | -52.4% (-22.402 GiB) |
| 256k | 2762.55 / 78.261 | 2593.43 / 35.429 | +6.5% | -54.7% (-42.832 GiB) |

### 700B C8

| Sequence | Baseline A + compile (ms/GiB) | Our CE + cuBLASLt (ms/GiB) | Throughput gain | Memory change |
|---:|---:|---:|---:|---:|
| 4k | 112.27 / 11.067 | 90.21 / 8.489 | +24.5% | -23.3% (-2.578 GiB) |
| 8k | 185.62 / 12.454 | 152.42 / 9.095 | +21.8% | -27.0% (-3.359 GiB) |
| 16k | 320.84 / 14.759 | 284.35 / 10.247 | +12.8% | -30.6% (-4.512 GiB) |
| 32k | 611.01 / 18.042 | 570.28 / 12.593 | +7.1% | -30.2% (-5.449 GiB) |
| 64k | 1230.82 / 28.415 | 1126.86 / 17.321 | +9.2% | -39.0% (-11.094 GiB) |
| 128k | 2289.76 / 48.923 | 2136.85 / 26.696 | +7.2% | -45.4% (-22.227 GiB) |
| 256k | OOM | 4222.16 / 45.466 | — | — |

[All 84 full-head results and per-order gain](full-head/final-summary.json) · [Rank receipts](full-head/results/final)

Across the 20 shapes where the optimized head completed, its median within-shape time change versus the previous sweep was −1.13%. Among the 16 with different effective GEMM choices, 13 completed faster than in that earlier sweep. The largest adverse change after accounting for the historical baseline was 100B C8/128k. Before the scoped projection rollback, its new baseline times were 1,395.80 and 1,436.06 ms, and tuned times were 1,351.37 and 1,466.09 ms. The historical medians were 1,477.49 and 1,253.80 ms. Host submission stayed near 10 ms for the tuned path. Most of the wall-time difference showed up waiting for CUDA completion. The receipts identify the projection plan as the only deliberately changed route for this shape; they do not break out time by GPU kernel. Both fresh pairs had identical 42.751 and 20.349 GiB peaks for baseline and tuned, respectively.

## Direct optimized-head comparison at 100B C8/128k

On the same eight H100s, the two heads loaded identical tuned CUDA CE, generic direct FP32 dW and ATen dHidden paths. The prior head used ATen projection, while the candidate ran 184 calls per rank with the selected cuBLASLt projection. In the prior-first pair, the prior and candidate took 1,257.80 and 1,323.88 ms, respectively. In the reverse-order pair, the prior and candidate took 1,255.76 and 1,251.37 ms. All four arms measured 20.34851 GiB and zero allocation retries. The prior head reproduced its historical time within 0.32% on both runs. The explicit plan lost 66.07 ms in one pair and saved 4.39 ms in the other. It showed no reliable integrated benefit despite the isolated GEMM win.

During the slower candidate repeat, median and slowest-rank wall times agreed within 0.04 ms, and time waiting for CUDA rose by about 66 ms. Sampled clocks do not establish a cause. The comparison supports using ATen projection specifically for 100B C8/128k. The 100B C2/32k configuration shares the projection descriptor and retains its calibrated route. [Direct A/B receipts](full-head/projection-128k-summary.json)


The final benchmark policy restores ATen projection only for 100B C8/128k. Its fresh baseline-first repeat measured 1,371.51 → 1,359.11 ms (+0.91% throughput). Its candidate-first repeat measured 1,353.22 → 1,358.19 ms (−0.37%). Both measured the same peak reduction of 22.402 GiB per rank. The headline row above uses these corrected repeats. The gain changes sign, so these measurements establish a memory win and do not establish a speed win at this operating point. Even with the restored projection, the corrected 1,358.65 ms is 8.36% above the historical 1,253.80 ms, while the direct old-versus-new comparison reproduced the old time when paired with the other optimized head. The available measurements do not identify the cause of the remaining elapsed-time variation.

[Final per-cell policy](full-head/final-policy.json) · [Final runner](full-head/run_final_grid.py) · [Corrected cell receipts](full-head/corrected-100b-c8-128k-summary.json)

### Subsequent 100B C8 timing diagnosis

A follow-up traced all eight ranks and sampled the driver. On the original-order 128k trace, the optimized path saved about 92 ms of work up to reduce-scatter on the median of the other seven devices, but physical GPU 6 arrived 181 ms behind the first peer. After swapping physical GPUs 0 and 6, a sustained repeat reproduced the lag on logical rank 0, still physical GPU 6, when the driver reported thermal clock limits. Median unprofiled latency went from 1,255.92 to 1,396.95 ms within the same process. In a separate matched pair without sampled thermal limiting on the optimized arm, that arm took 1,256.01 ms versus 1,360.43 ms for baseline. The published 0.3% does not establish the intrinsic performance of this LM-head shape on an unthrottled node. The earlier acceptance runs did not record sufficient telemetry to reconstruct what fraction of each interval was lost to this mechanism. [Per-device thermal evidence](c8-diagnosis/telemetry/README.md) · [Short-context traces](c8-diagnosis/phase/phase-short.json) · [Long-context traces](c8-diagnosis/phase/phase-main.json)
