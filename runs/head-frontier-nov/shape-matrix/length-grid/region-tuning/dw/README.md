# C8 dW tuning at local batch one

Five isolated dW plans beat the active length-grid writer in discovery and confirmation. The 700B head improved at all four lengths, with confirmation wall reductions of 5.2–7.9% against the control median. The 100B head improved at 262,144 tokens by 1.4%. All five selected plans matched every FP32 output bit after each of the eight writes. Each selected plan used the same measured GIFT total as its control. The other three 100B lengths retain their current writers. [Selected plan catalog](selected-catalog.json)

## Confirmation results

The table reports medians from twelve samples per arm on one H100. Paired differences are medians of within-round differences. A/A noise is the median absolute difference between the two identical control arms. GIFT total is the isolated per-arm maximum. Elapsed-time reductions divide the difference of the control and finalist medians by the control median.

| Model | Local tokens | K | Control, ms | Repeat, ms | Finalist, ms | Paired Δ vs control and repeat, ms | A/A noise, ms | GIFT control / finalist, GiB | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100B | [32,768](results/100b-c8-b1-s32768.json) | 8,192 | 94.410 | 94.967 | 96.939 | +2.033 / -0.870 | 2.458 | 4.749 / 4.749 | Keep control |
| 100B | [65,536](results/100b-c8-b1-s65536.json) | 16,384 | 196.538 | 194.503 | 196.558 | +0.061 / +1.460 | 2.092 | 6.761 / 6.761 | Keep control |
| 100B | [131,072](results/100b-c8-b1-s131072.json) | 32,768 | 387.800 | 389.362 | 394.821 | +7.007 / +5.777 | 2.252 | 10.804 / 10.804 | Keep control |
| 100B | [262,144](results/100b-c8-b1-s262144.json) | 65,536 | 775.796 | 771.411 | 764.985 | -8.758 / -9.643 | 6.178 | 18.909 / 18.909 | Select |
| 700B | [32,768](results/700b-c8-b1-s32768.json) | 8,192 | 179.310 | 178.331 | 167.964 | -9.963 / -9.555 | 1.549 | 6.272 / 6.272 | Select |
| 700B | [65,536](results/700b-c8-b1-s65536.json) | 16,384 | 352.886 | 352.060 | 333.779 | -19.045 / -17.408 | 2.069 | 8.362 / 8.362 | Select |
| 700B | [131,072](results/700b-c8-b1-s131072.json) | 32,768 | 704.061 | 703.129 | 667.734 | -35.314 / -36.499 | 1.793 | 12.464 / 12.464 | Select |
| 700B | [262,144](results/700b-c8-b1-s262144.json) | 65,536 | 1477.833 | 1478.314 | 1360.956 | -120.610 / -115.300 | 25.952 | 20.765 / 20.765 | Select |

| Model | Local tokens | Graph control → finalist, ms | Submit control → finalist, ms | Final wait control → finalist, ms |
|---|---:|---:|---:|---:|
| 100B | 32,768 | 96.359 → 97.837 | 0.130 → 0.132 | 94.276 → 96.805 |
| 100B | 65,536 | 194.872 → 196.748 | 0.140 → 0.128 | 196.393 → 196.431 |
| 100B | 131,072 | 389.215 → 395.304 | 0.156 → 0.152 | 387.644 → 394.678 |
| 100B | 262,144 | 781.344 → 768.393 | 0.209 → 0.241 | 775.537 → 764.782 |
| 700B | 32,768 | 179.416 → 168.340 | 0.147 → 0.141 | 179.145 → 167.827 |
| 700B | 65,536 | 353.878 → 335.324 | 0.200 → 0.171 | 352.652 → 333.593 |
| 700B | 131,072 | 703.612 → 667.559 | 0.177 → 0.158 | 703.856 → 667.583 |
| 700B | 262,144 | 1456.552 → 1346.186 | 0.185 → 0.175 | 1477.631 → 1360.784 |

## Selected configurations

All selected writes use cuBLASLt algorithm 66, one K partition, zero reduction scheme, four workspace bytes and FP32 accumulation. The catalog contains the five new shape entries.

| Model | Local tokens | β = 0 layout / tile / custom / cluster | β = 1 layout / tile / custom / cluster |
|---|---:|---|---|
| 100B | [262,144](results/100b-c8-b1-s262144-plan.json) | column / 248 / 5 / 5 | row / 24 / 7 / 5 |
| 700B | [32,768](results/700b-c8-b1-s32768-plan.json) | row / 24 / 7 / 5 | column / 24 / 7 / 5 |
| 700B | [65,536](results/700b-c8-b1-s65536-plan.json) | column / 209 / 5 / 3 | column / 24 / 7 / 5 |
| 700B | [131,072](results/700b-c8-b1-s131072-plan.json) | row / 24 / 7 / 5 | row / 24 / 7 / 5 |
| 700B | [262,144](results/700b-c8-b1-s262144-plan.json) | column / 24 / 7 / 5 | row / 24 / 7 / 5 |

## Method

The control uses the live [length-grid catalog](../../../tuning/lt/results/calibrated-catalog.json). Its sole matching calibrated shape is the 100B head at 32,768 tokens. The remaining seven shapes activate the native generic direct cuBLASLt writer. The eight-write chain has one overwrite and seven in-place accumulations. Each multiplication uses BF16 inputs, FP32 compute and an FP32 output slot of shape `[128256, width]`. The width is 4,096 for 100B and 7,168 for 700B. The reduction dimension is `S / 4` for every write. [Length grid](../../report.md) · [Prior dW calibration](../../../tuning/lt/README.md)

The length grid uses real production-tokenizer targets and packed positions plus seeded BF16 hidden activations and weights. The run replayed the full grid's label resize, boundary masks and MTP shift. It compared the main labels, MTP labels, positions and hidden source indices for all eight chunks. All agreed in each cell. The harness then generated one full-size hidden and consuming CUDA CE `dLogits` pair using the active projection route and global denominator. Each arm reused the same pair at the same addresses for eight writes. The pair ranged from about 2 GiB to 17 GiB, so the whole pair cannot fit in H100 L2, but reuse can affect cache behavior and allocator lifetimes. GIFT total here covers one retained input pair, the output slot, writer workspace and process/library memory on one H100. The full head and training have different live tensors and execution. [Grid fixture and dispatch](../../replay.py) · [Search harness](search.py)

The search requested up to 32 heuristic entries per layout at each of the 32 MiB and zero-byte workspace caps. It added up to 24 configurations per layout by varying tile, custom option and CTA cluster from heuristic anchors and applying `cublasLtMatmulAlgoCheck`. It timed each valid plan independently at β = 0 and β = 1, then screened the entire FP32 output for bitwise parity at each beta. A shortlist of fastest exact plans across and within layouts, plus single-beta alternatives, ran the full chain. The finalist's entire FP32 slot was also checked after every write. All 64 screened candidates at each shape passed both per-beta bitwise comparisons. The persisted receipts bind the serialized algorithms to CUDA device, library, shape and alignment. [Prior neighborhood search](../../../tuning/lt_extended/README.md) · [NVIDIA cuBLASLt algorithms](https://docs.nvidia.com/cuda/archive/13.1.2/cublas/index.html#cublasltmatmulalgocheck)

Each chain window alternated forward and reverse arm order. The discovery window had eight samples per arm. The confirmation window had twelve and constructed fresh writers and CUDA graphs. To win, in both windows the finalist had to beat both identical control arms in at least 75% of rounds and have a paired median wall-time improvement over each that exceeded the median absolute paired A/A difference. The synchronized wall timer includes host submission and the final stream wait. CUDA Graph event replay measures the device chain span including any gaps. The JSON records host submission, final wait and graph samples independently. Kernel-only time and separate wait attribution were unavailable. Projection, CE preparation, explicit copies and all collectives are outside the timed region.

## Reproduce

The runs used three H100 80GB HBM3 GPUs on the SR008 node, PyTorch `2.15.0.dev20260822+cu130`, CUDA `13.0`, cuBLASLt 13.1.1 and driver `570.133.20`. The [launcher](run.sh) checks the supplied GPU's idleness, pins its UUID and runs the selected cells in the installed PyTorch environment. The source and catalog hashes, device UUID, exact candidate options, parity counts, data provenance, memory and raw timings are in each result.

```bash
root=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125
bash "$root/length-grid/region-tuning/dw/run.sh" "$root" 0 100b 32768
```
