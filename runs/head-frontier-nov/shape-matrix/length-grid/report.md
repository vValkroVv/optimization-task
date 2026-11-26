# Isolated eight-H100 LM-head sweep at local batch one

On 26 November 2026, both the compiled Baseline A (`e9b7cdbb4cd6`) and the tuned CUDA CE + cuBLASLt package ran from the 4,096-token point on the replacement SR008 node (ML Space job `60634e19-cb86-4e8b-a27c-60078635cd4b`). This series uses one physical sequence per rank and one microbatch per timed operation. Main plus MTP1 produce `2 × sequence / chunks` logits rows in each folded projection. The 100B head uses width 4,096, the 700B GIFT 3.5 linear head uses width 7,168, and both use vocabulary 128,256.

Each implementation starts in a fresh eight-rank process group. Each successful arm has ten measured samples after ten warmups and an allocator priming step. Wall time is the median of the slowest synchronized rank per sample. GIFT total is the highest peak PyTorch reserved memory plus sampled non-PyTorch usage across all ranks and samples. CUDA OOM requires an exception from an actual attempted launch. Percentages in the table use baseline time divided by tuned time minus one.

The timer covers the ordinary BF16 weight all-gather, projection, weighted main and MTP CE, both gradient GEMMs over all chunks and the FP32 SUM reduce-scatter. The 100B run includes hidden z-loss. The inputs use the same seed weights and a 4,096-position tile from the stored fixture. Longer sequences repeat that tile with its packed-label boundaries. The runtime tensor has the full stated sequence length. The decoder and optimizer are outside scope.

All successful tuned runs use the same measured CUDA binary and cuBLASLt catalog. The length sweep does not recalibrate the catalog. The 100B C2/8k, C8/8k and C8/32k points used calibrated dW and projection plans. Other successful shapes used the generic direct FP32 cuBLASLt dW writer with the normal ATen projection and dHidden. The prior 700B C8 calibration used 3,072 folded rows per chunk, a shape absent from this B1 powers-of-two grid.

[Original tuning](../tuning/report.md) · [Launcher](launch.py) · [Aggregation](summarize.py) · [Fresh per-rank measurements and OOM receipts](results/server-60634e19) · [Machine-readable summary](server-60634e19-summary.json)

### 100B C2

| Local sequence | Baseline A + compile, ms / GiB | Tuned CE + cuBLASLt, ms / GiB | Head throughput gain |
|---:|---:|---:|---:|
| 4,096 | 51.48 / 8.626 | 46.99 / 6.477 | +9.5% |
| 8,192 | 91.29 / 11.946 | 83.26 / 7.727 | +9.6% |
| 16,384 | 176.69 / 18.626 | 163.89 / 10.149 | +7.8% |
| 32,768 | 349.24 / 32.495 | 326.18 / 15.073 | +7.1% |
| 65,536 | 692.99 / 59.704 | 634.83 / 24.880 | +9.2% |
| 131,072 | CUDA OOM | 1267.86 / 44.569 | — |
| 262,144 | CUDA OOM | CUDA OOM | — |

### 100B C8

| Local sequence | Baseline A + compile, ms / GiB | Tuned CE + cuBLASLt, ms / GiB | Head throughput gain |
|---:|---:|---:|---:|
| 4,096 | 65.16 / 7.610 | 52.93 / 5.696 | +23.1% |
| 8,192 | 104.93 / 8.548 | 86.70 / 6.165 | +21.0% |
| 16,384 | 189.67 / 10.794 | 166.84 / 7.122 | +13.7% |
| 32,768 | 364.84 / 16.323 | 324.51 / 9.018 | +12.4% |
| 65,536 | 684.61 / 24.489 | 630.81 / 12.790 | +8.5% |
| 131,072 | 1477.49 / 42.751 | 1253.80 / 20.349 | +17.8% |
| 262,144 | 2751.12 / 78.261 | 2599.48 / 35.429 | +5.8% |

### 700B C8

| Local sequence | Baseline A + compile, ms / GiB | Tuned CE + cuBLASLt, ms / GiB | Head throughput gain |
|---:|---:|---:|---:|
| 4,096 | 112.61 / 11.067 | 94.46 / 8.489 | +19.2% |
| 8,192 | 185.22 / 12.454 | 159.55 / 9.095 | +16.1% |
| 16,384 | 321.87 / 14.759 | 287.08 / 10.247 | +12.1% |
| 32,768 | 639.09 / 18.042 | 582.91 / 12.593 | +9.6% |
| 65,536 | 1159.91 / 28.415 | 1106.29 / 17.321 | +4.8% |
| 131,072 | 2314.05 / 48.923 | 2186.38 / 26.696 | +5.8% |
| 262,144 | CUDA OOM | 4461.72 / 45.466 | — |
