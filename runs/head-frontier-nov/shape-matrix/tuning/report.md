# Hopper CE and cuBLASLt tuning on the isolated GIFT head

This study compares the accepted consuming CUDA CE and direct FP32 weight gradient with a calibrated alternative on the same SR008 node. The 100B controls use the historical best 9.14 / 5.5 plans. The 700B control uses the previously measured generic first-compatible heuristic. All variants retain the ordinary BF16 weight all-gather, native GFSDP FP32 SUM reduce-scatter, and separate main and MTP losses. The 100B replay includes hidden z-loss.

## Eight-rank head results

Each time pair is the median for run 1 / run 2 (ms). The time change compares the arithmetic mean of the two run medians on each side. The schedule followed control A, candidate A, control B, candidate B on the same node for each preset.

| Preset | Current, ms | Calibrated, ms | Time saved, ms | Time reduction | Peak GIFT total, current → calibrated, GiB/GPU |
|---|---:|---:|---:|---:|---:|
| 100B C2 | 84.95 / 85.19 | 83.09 / 84.05 | 1.50 | 1.76% | 7.727 → 7.727 |
| 100B C8 | 87.59 / 87.73 | 87.25 / 86.92 | 0.57 | 0.66% | 6.165 → 6.165 |
| 700B C8 | 225.22 / 224.63 | 216.17 / 217.02 | 8.33 | 3.70% | 9.661 → 9.661 |

Each candidate run beat its paired fresh control. Across the two control runs the medians differed by 0.24, 0.14 and 0.59 ms, respectively. Across the two candidate runs they differed by 0.96, 0.33 and 0.85 ms. On 100B C8 both old and new implementations are around 87 ms. The earlier published bests were 83.80, 87.12 and 224.33 ms; improvements over the old 100B records are considerably smaller than the head-to-head deltas. The 700B result also beats the historical best by over 7 ms in both repeats. [Per-step times, host submission, final waits, and precision checks](head-summary.json)

In all six candidate replays, every rank matched the control for the main loss, MTP1 loss, total loss, entire BF16 hidden gradient and entire reduced FP32 weight-gradient shard. The FP64 error metrics were unchanged. The preflight in every arm observed one ordinary BF16 all-gather and one FP32 SUM reduce-scatter per microbatch. These timed samples do not decompose communication and compute or measure an end-to-end training speedup. [Rank receipts](results)

## Single-GPU regions

Each cell gives current → calibrated synchronized wall time in ms / isolated GIFT total in GiB. The control and candidate peaks are equal in each displayed cell. Projection, CE and dHidden each time one chunk. The dW row times **all** loss chunks in sequence, including the beta transition. Region peaks have different live inputs. Adding region peaks or multiplying every region by C would give a misleading head estimate.

| Region | 100B C2 | 100B C8 | 700B C8 |
|---|---:|---:|---:|
| Projection GEMM | 11.931 → 11.603 / 3.893 | 3.135 → 3.129 / 2.429 | 7.393 → 7.616 / 3.815 |
| Weighted CE and gradient | 1.763 → 1.719 / 4.638 | 0.568 → 0.554 / 1.727 | 0.768 → 0.750 / 2.196 |
| dHidden GEMM | 11.160 → 11.342 / 4.020 | 2.828 → 3.010 / 2.516 | 7.871 → 7.502 / 3.923 |
| dW GEMMs, all chunks | 23.999 → 24.266 / 6.776 | 25.197 → 24.982 / 6.796 | 71.694 → 66.858 / 10.407 |

700B dW also improved in a preallocated CUDA Graph, from 69.92 to 65.17 ms across eight chunks. The dHidden and dW arms use BF16 gradients produced by the control CE; warm eager calls include Python submission, while graph spans exclude it. The 700B projection and both 100B dHidden alternatives lost time in eager execution and were left on ATen in the integrated candidate. The 100B C8 projection showed a graph gain and effectively tied in regional eager time. Integrated dispatch caches matched descriptors to reduce host overhead. Direct strided dHidden saved 42 MiB allocated on the 700B component but requires a change to the gradient accumulator contract and was not used in the full-head tests. [CE measurements](ce/README.md) · [Projection and dHidden, corrected memory and timing](proj/README.md) · [dW measurements](lt/README.md)

## Choosing the kernels

- **CE.** The selected CUDA kernel computes the maximum on packed BF16 values and prepares FP32 softmax rescaling factors across warp lanes. It retains the original FP32 recurrence and ordering for the denominator, the FP32 per-head scale, and the in-place BF16 gradient. The kernel reads the number of rows, vocabulary and head count at runtime. A separate check exercised eleven heads with an odd, unaligned vocabulary. The benchmark covers both head sums and the weighted loss. [CUDA changes, ABI, and build](ce/README.md) · [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- **Weight gradient.** The offline search times cuBLASLt row- and column-major descriptors with separate candidates for the overwrite and accumulation. It checks the actual workspace requirement and compares all FP32 output bits. At 700B the winning plans use tile 29 in two layouts and request four workspace bytes. The first chunk writes the native slot with beta zero. The other seven update the same slot with beta one. The 100B search recovered the performance of the hand-selected plans in this isolated region. [Search and numerical checks](lt/README.md) · [NVIDIA autotuning example](https://github.com/NVIDIA/CUDALibrarySamples/blob/a2eddd862e886c0ea5827013919801a1ddc25430/cuBLASLt/LtSgemmSimpleAutoTuning/sample_cublasLt_LtSgemmSimpleAutoTuning.cu)
- **Projection and hidden gradient.** The calibrated policy tries a measured 2D cuBLASLt descriptor for the 100B projection and the 700B hidden gradient. All use BF16 operands and BF16 output. The calibrated GEMMs use FP32 compute, one K partition and the current PyTorch workspace. Other operations use the existing ATen path. The 100B projection passes a 2D view of the destination to cuBLASLt and returns its owning 3D tensor to autograd. The first integration attempt returned a view, which the consuming CE could not legally overwrite. The corrected path allocates a single logits buffer. [Replay integration](head_run.py) · [Measured linear plans](linear-plans.json) · [First failed integration](failed-projection-view.log)

A single CE artifact and the same two calibration files serve all three scenarios. cuBLASLt plan data are specific to the measured descriptors and library. The runtime selects by tensor dimensions, strides, alignment and device information within the existing contiguous Hopper BF16 contract. Uncalibrated GEMM shapes take the native path. The replay has no target model-name or chunk-count branches in the CUDA kernel or plan lookup. First-use enumeration still has a host cost, and the generalization outside this contract remains unmeasured. The calibration occurs outside the timed run. First-use plan construction and compilation are outside the steady-state results. [Linear dispatch](linear_policy.py) · [Weight-gradient catalog](lt/results/calibrated-catalog.json) · [Exact installed NVIDIA API](https://docs.nvidia.com/cuda/archive/13.1.2/cublas/index.html#cublasltmatmulalgocheck)

## Measurement scope

The inputs are seeded BF16 hidden states and weights with real packed targets and positions. All ranks replay the same local input. The head uses 100B H=4,096 or 700B H=7,168, V=128,256, sequence length 4,096, main plus MTP1, and local batch two or three respectively. All three comparisons hold the loss chunk count fixed. The CUDA setup is one eight-H100-80GB NVLink node, PyTorch 2.15.0.dev20260822+cu130 and cuBLASLt 130101. [Fixture and timer boundaries](../report.md)

For each full-head arm we launch eight fresh ranks, discard ten warmups, prime the allocator, and collect twenty synchronized wall samples. Each sample uses the slowest rank. Memory is the maximum GIFT total (peak reserved plus non-PyTorch device usage at readout) over all samples and ranks. Each arm also performs an untimed clean backward and compares dHidden, each rank's reduced dW shard and per-head loss to the shared FP64 oracle. The oracle starts from the baseline BF16 projection. BF16 dHidden and FP32 shard digests compare the full tensors to the current control. This measures one head microbatch, without decoder or optimizer. [Launch settings](launch.py) · [Timed region](../head_bench.py) · [FP64 oracle](../head_oracle.py)

## Additional C8 search and reproduction

An extended 100B C8 search admitted another 96 cuBLASLt configurations for dW with split-K of one. In discovery the finalist improved the complete chain by 0.56 ms in paired eager wall time. In confirmation it took 0.16 ms longer than plan 5.5. Its paired CUDA Graph delta also reversed, from −0.315 to +0.115 ms. The candidate failed the repeatability criterion and was excluded from the head policy. [Extended search receipt](lt_extended/results/100b-c8-extended.json)

The dedicated remote benchmark directory is `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125` on `sr008-gpu`. Its `tuning/head_bench.py` is the [recorded replay](../head_bench.py), SHA-256 `484bee2f2affee0999268804f314219d6e69829a75bbd38ab4d83607eff220ed`. The original CE DSO SHA-256 is `17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8` and the measured new DSO SHA-256 is `d3343cd518a05a6d5bea58b0493fa33373e040f1ce3dd5864b14c03033528929`. Each completed arm records its invocation, environment, library digest, plan activation, statistics and numerical outputs. The launcher resumes by skipping successful receipts. Use fresh run labels for an independent new measurement.

```bash
source /etc/profile.d/wm.sh
/opt/torchtitan-venv/bin/python3.12 -Bu tuning/launch.py tuning/controls.json
/opt/torchtitan-venv/bin/python3.12 -Bu tuning/launch.py tuning/finalists.json
python3 tuning/summarize.py tuning/results
```

Full-head kernel-level times and communication durations were not collected in these replays. Projection and dHidden screening included separate profiles. Generalizing beyond the existing Hopper, contiguous BF16, vocabulary and GFSDP lease constraints requires checking those contracts independently. No production training-source checkout was modified.
