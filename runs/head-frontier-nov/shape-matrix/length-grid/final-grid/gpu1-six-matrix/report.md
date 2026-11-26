# Full isolated LM-head on six H100 GPUs

All 21 configurations ran on physical GPUs 0–5 of `sr008-gpu1` with local batch 1, main plus MTP1, and the same sequence length and chunk count within each comparison. The timed full head includes the BF16 weight all-gather and FP32 gradient reduce-scatter. The candidate uses the final shape-specific CE and GEMM plans, including the corrected ATen projection route for 100B C8 at 128k.

Each configuration ran twice per variant in opposite orders, with 10 warmup calls and 10 timed calls per process. Times are the median of the two run medians. Each timed call uses the slowest of six ranks. Peak memory is the maximum per-GPU GIFT total across both runs and all timed calls. GIFT adds peak PyTorch reserved memory to current non-PyTorch usage. Throughput gain is 100 × (baseline time / candidate time − 1). Each OOM occurred in both runs at the initial head call before warmup. The matrix contains 76 successful runs and eight CUDA OOMs.


### 100B · C2

| Sequence | Baseline A + compile, ms / GiB | Our CE + cuBLASLt, ms / GiB | Throughput gain | Peak memory change |
|---:|---:|---:|---:|---:|
| 4k | 51.31 / 8.819 | 46.00 / 6.651 | +11.6% | -24.6% (-2.168 GiB) |
| 8k | 91.89 / 12.140 | 82.90 / 7.901 | +10.8% | -34.9% (-4.238 GiB) |
| 16k | 175.02 / 18.800 | 162.73 / 10.343 | +7.6% | -45.0% (-8.457 GiB) |
| 32k | 342.41 / 32.688 | 319.59 / 15.247 | +7.1% | -53.4% (-17.441 GiB) |
| 64k | 694.30 / 59.917 | 634.41 / 25.093 | +9.4% | -58.1% (-34.824 GiB) |
| 128k | OOM | 1223.43 / 44.763 | — | — |
| 256k | OOM | OOM | — | — |

### 100B · C8

| Sequence | Baseline A + compile, ms / GiB | Our CE + cuBLASLt, ms / GiB | Throughput gain | Peak memory change |
|---:|---:|---:|---:|---:|
| 4k | 64.69 / 7.667 | 50.87 / 6.026 | +27.2% | -21.4% (-1.641 GiB) |
| 8k | 104.61 / 8.702 | 86.57 / 6.358 | +20.8% | -26.9% (-2.344 GiB) |
| 16k | 185.85 / 10.030 | 163.08 / 7.354 | +14.0% | -26.7% (-2.676 GiB) |
| 32k | 350.94 / 15.071 | 309.50 / 9.212 | +13.4% | -38.9% (-5.859 GiB) |
| 64k | 682.08 / 24.682 | 623.40 / 12.983 | +9.4% | -47.4% (-11.699 GiB) |
| 128k | 1345.24 / 45.854 | 1251.16 / 20.542 | +7.5% | -55.2% (-25.312 GiB) |
| 256k | 2751.71 / 78.474 | 2497.93 / 35.622 | +10.2% | -54.6% (-42.852 GiB) |

### 700B · C8

| Sequence | Baseline A + compile, ms / GiB | Our CE + cuBLASLt, ms / GiB | Throughput gain | Peak memory change |
|---:|---:|---:|---:|---:|
| 4k | 112.03 / 11.378 | 89.25 / 8.839 | +25.5% | -22.3% (-2.539 GiB) |
| 8k | 185.38 / 12.804 | 151.99 / 9.425 | +22.0% | -26.4% (-3.379 GiB) |
| 16k | 319.43 / 15.128 | 277.65 / 10.616 | +15.0% | -29.8% (-4.512 GiB) |
| 32k | 593.11 / 20.325 | 537.00 / 12.923 | +10.4% | -36.4% (-7.402 GiB) |
| 64k | 1150.16 / 28.784 | 1074.06 / 17.671 | +7.1% | -38.6% (-11.113 GiB) |
| 128k | 2284.18 / 49.272 | 2123.82 / 27.065 | +7.6% | -45.1% (-22.207 GiB) |
| 256k | OOM | 4190.98 / 45.835 | — | — |

## Temperatures

All 4,560 measured rank calls had thermal counter samples bracketing the call. No thermal counter increased and no thermal slowdown or power-brake flag was sampled. Peak core and HBM temperatures were 79 °C and 78 °C. The software power cap was active in 16,368 of 16,415 in-call GPU samples. Physical GPUs 6 and 7 had zero sampled utilization throughout. The six-rank experiment does not establish the cause of the earlier eight-rank timing anomaly.

[Per-rank measurements and launch summaries](replay/summary.json) · [Thermal evidence](thermal/verdicts.json) · [Telemetry methodology](thermal/README.md)
