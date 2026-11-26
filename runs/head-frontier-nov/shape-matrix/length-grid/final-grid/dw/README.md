# Folded dW plan selection at local batch one

The [cell manifest](manifest.json) maps each grid cell to its active dW route, exact evidence receipt and catalog entry. The [C2 catalog](dw-plans.selected.c2.json) and [C8 catalog](dw-plans.selected.c8.json) use the existing ReplayWriter format. The folded GEMM has shape `[K, V]ᵀ @ [K, H] → [V, H]`, with `K = 2S / C`, vocabulary 128,256, width 4,096 for 100B and width 7,168 for 700B. The two catalogs allow a shape shared by C2 and C8 to choose independently after one FP32 overwrite and one or seven in-place FP32 accumulations.

## Measured selections

Across the 21 cells, ten selected a faster isolated dW pair, three kept their calibrated control and eight kept the native generic control. The ten selected pairs used four workspace bytes and matched the control GIFT total.

| Model | Chunks | Local tokens | Wall control → selected, ms | Wall reduction | GIFT total, GiB |
|---|---:|---:|---:|---:|---:|
| 100B | 2 | [4,096](results/100b-c2-b1-s4096.json) | 13.104 → 11.805 | 9.9% | 4.749 |
| 100B | 8 | [4,096](results/100b-c8-b1-s4096.json) | 18.615 → 16.451 | 11.6% | 4.768 |
| 100B | 8 | [262,144](../../region-tuning/dw/results/100b-c8-b1-s262144.json) | 775.796 → 764.985 | 1.4% | 18.909 |
| 700B | 8 | [4,096](results/700b-c8-b1-s4096.json) | 34.243 → 29.141 | 14.9% | 6.272 |
| 700B | 8 | [8,192](results/700b-c8-b1-s8192.json) | 51.511 → 44.872 | 12.9% | 6.292 |
| 700B | 8 | [16,384](results/700b-c8-b1-s16384.json) | 93.387 → 89.659 | 4.0% | 6.272 |
| 700B | 8 | [32,768](../../region-tuning/dw/results/700b-c8-b1-s32768.json) | 179.310 → 167.964 | 6.3% | 6.272 |
| 700B | 8 | [65,536](../../region-tuning/dw/results/700b-c8-b1-s65536.json) | 352.886 → 333.779 | 5.4% | 8.362 |
| 700B | 8 | [131,072](../../region-tuning/dw/results/700b-c8-b1-s131072.json) | 704.061 → 667.734 | 5.2% | 12.464 |
| 700B | 8 | [262,144](../../region-tuning/dw/results/700b-c8-b1-s262144.json) | 1477.833 → 1360.956 | 7.9% | 20.765 |

Elapsed reductions divide the difference of the confirmation medians by the control median. The 100B C2/65,536 comparison included the selected C8/262,144 pair for its identical GEMM descriptor and retained the generic C2 control. The 100B C2/262,144 isolated chain fit on one H100. Its finalist was bitwise exact but beat both wall controls in seven of twelve confirmation rounds, so the catalog retains the native generic control. [C2/65,536 receipt](results/100b-c2-b1-s65536.json) · [C2/262,144 receipt](results/100b-c2-b1-s262144.json)

## Controls and admission

The control is the [existing length-grid catalog](../../../tuning/lt/results/calibrated-catalog.json) where the descriptor matches, followed by the active native generic cuBLASLt writer. The [prior C8 calibration](../../region-tuning/dw/README.md) supplies eight measurements at 32,768–262,144 tokens. The other thirteen measurements use the same neighborhood search and the actual C2 or short C8 write chain.

The [calibration script](calibrate.py) uses the original length replay's fixture resize, packed positions and main/MTP1 labels, including its denominator and MTP shift. It runs the active projection route and the consuming CUDA CE backward to generate the BF16 dLogits. Each distinct chunk has its own hidden/dLogits pair. Chunks with identical hidden source indices, labels and positions share storage, and every arm uses the same addresses. The script checks candidate descriptors on the installed cuBLASLt and screens at each beta. Admitted plans preserve BF16 operands, FP32 compute and output, and bitwise equality over every FP32 element after each write. [cuBLASLt algorithm checks](https://docs.nvidia.com/cuda/archive/13.1.2/cublas/index.html#cublasltmatmulalgocheck)

The discovery window has eight samples per arm. Confirmation uses twelve samples, fresh writers and CUDA graphs. Both windows reverse arm order every round. In both windows the candidate must beat both identical controls in at least 75% of rounds and show a paired median wall reduction larger than the median absolute A/A difference. The measured per-arm GIFT total may not exceed the control. This memory metric sums peak PyTorch reserved bytes and current non-PyTorch device bytes. The arms share identical operand and output lifetimes during the timed writer chain. The manifest records peak reserved, peak allocated, current external memory and allocator retries separately.

The synchronized wall timer includes host submission and the final stream wait. CUDA Graph events measure the device span of the dW chain. Kernel-only durations and attribution within the final wait are unavailable. Projection, CE preparation, explicit copies, collectives, full-head memory and training are outside this calibration. The original full-head replay reported CUDA OOM at 100B C2/262,144 for both arms, at C2/131,072 for its baseline and at 700B C8/262,144 for its baseline. An isolated dW result is compatible with those OOM receipts because the full head has additional live tensors. [Original full-head sweep](../../report.md)

## Reproduce

The new runs used four H100 80GB HBM3 GPUs on SR008 job `60634e19-cb86-4e8b-a27c-60078635cd4b`, PyTorch `2.15.0.dev20260822+cu130`, CUDA 13.0, cuBLASLt 13.1.1 and driver 570.133.20. The [launcher](run.sh) checks the assigned GPU is idle and exposes its UUID as the sole device. Case receipts record the fixture identity, activation, exact candidates, raw timings, numerical comparisons and source hashes. The first attempt failed during receipt setup because the previous local C8 catalog had no remote copy. The launcher reran that cell after the catalog was copied into this evidence directory. [First attempt](first-attempt/gpu-0-pilot.log)

```bash
root=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125
bash "$root/length-grid/final-grid/dw/run.sh" "$root" 0 100b:2:4096
python3 length-grid/final-grid/dw/assemble.py
```

Run the assembler from the checked-out shape-matrix directory with the raw results and prior calibration receipts present. It checks all 21 descriptors, selected route activation, per-write bitwise parity, paired timing thresholds and measured memory before producing the two catalogs.
