# cuBLASLt plans for direct FP32 head dW

The shape-general cuBLASLt search found a 700B C8 plan pair that reduced the preallocated dW graph span from 69.915 to 65.043 ms, a 6.97% reduction relative to the first generic control. A second run measured 69.919 to 65.167 ms, a 6.80% reduction. The timed pair used 4 bytes of workspace and matched the generic writer bitwise for β = 0, β = 1, and the full eight-chunk chain. Both 100B searches produced bitwise matches within the variation of their existing fixed plans.

## Timings and memory

Each group ran on one H100 80GB HBM3. Ten graph samples per arm use one replay of the complete preallocated dW chain after capture and two warm replays. The second control is the same writer and algorithm, captured separately. Arm order reverses on alternate rounds. The timed path excludes input preparation, output allocation and tuning. Same-stream eager measurements record submission, the final host wait and synchronized wall separately. The graph span includes all chain kernels and GPU scheduling gaps. It cannot attribute a gap to a kernel or identify the effect on head or training latency. The graph and eager spans contain no explicit copies or collective communication.

| Case | Graph control / repeat / candidate, ms | Eager wall control / candidate, ms | Host submit control / candidate, ms | Final host wait control / candidate, ms | Isolated GIFT total control / candidate, GiB |
|---|---:|---:|---:|---:|---:|
| 100B C2, fixed control | 24.099 / 24.983 / 24.122 | 23.999 / 24.266 | 0.068 / 0.082 | 23.932 / 24.185 | 6.776 / 6.776 |
| 100B C8, fixed control | 24.752 / 26.411 / 24.824 | 25.197 / 24.982 | 0.125 / 0.167 | 25.044 / 24.815 | 6.796 / 6.796 |
| 700B C8, generic control, initial | 69.915 / 69.926 / 65.043 | 70.093 / 66.466 | 0.170 / 0.217 | 69.779 / 66.215 | — |
| 700B C8, generic control, verification | 69.919 / 72.924 / 65.167 | 71.694 / 66.858 | 0.193 / 0.223 | 71.349 / 66.638 | 10.407 / 10.407 |

The 700B paired graph deltas versus the first control are −4.668 and −4.341 ms, using the median of paired sample differences. The identical-control deltas are +0.182 and +2.902 ms. The 100B C2 and C8 candidate deltas are +0.178 and +0.121 ms, with identical-control deltas +1.051 and +1.702 ms. Relative to the weaker generic 100B C8 control of 29.872 ms, both the fixed plan and shape-general timed candidate were about 5 ms faster. The first row-major heuristic by itself measured 25.167, 30.227 and 71.645 ms across the three respective cases.

Isolated GIFT total uses peak reserved bytes plus current external device bytes, including all resident dW operands and the preallocated FP32 slot. Each method is measured alone after allocator cleanup, including writer setup. Peak allocated / reserved GiB were 5.996 / 6.037, 5.996 / 6.057 and 9.655 / 9.670 for both the selected candidate and control at the three shapes. Allocation retries were zero. Initial 700B measured 10.409 GiB with all arms resident.

## Operands and precision

The [shape-matrix fixtures](../../matrix_fixture.py) use real production-tokenizer targets and packed positions from data-002, with BF16 hidden states and weights generated at seed 2048. The existing CUDA-13 folded CE generates the actual contiguous BF16 dLogits with the weighted head scales and denominator. A single FP32 slot receives one overwrite and C−1 accumulation writes. The 100B tests multiply K × V × H with K = 8,192 or 2,048, V = 128,256 and H = 4,096. The 700B test uses K = 3,072, V = 128,256 and H = 7,168. Logical chain work is 17.214 TFLOP for either 100B test and 45.187 TFLOP for 700B. Logical BF16 input reads are 4.039, 4.039 and 6.199 GiB. FP32 slot writes plus β = 1 reads are 5.871, 29.355 and 51.372 GiB. Actual kernel traffic can differ.

The FP64 oracle computes over the same BF16 operands and every output element. The β = 0 bit check starts from NaNs. The standalone β = 1 check seeds the slot with the control’s first-chunk contribution and uses the actual final chunk. The full-chain check compares the raw INT32 view of the final FP32 slot against the control. Every selected plan recorded zero bit mismatches in all three checks. The respective full-chain relative L2 errors against FP64 were 1.2322e-5, 3.1310e-6 and 4.8366e-6. Maximum absolute errors were 5.2333e-10, 1.1969e-10 and 1.3964e-10. The corresponding control metrics were identical.

## Search and integration

The search uses the CUDA 13.1.1 cuBLASLt shipped on the host. It applies the live descriptor dimensions and alignments to both row-major and transposed column-major descriptions of the same storage. The native heuristic considers column-major first and takes the first compatible result. The benchmark measures all successful candidates from up to 32 heuristic entries per layout and workspace cap, with 4 shuffled CUDA Graph samples at each beta. It then replays the fastest per-beta pair, the fastest per-beta bitwise pair, heuristic arms, and pair variants using the runner-up for one beta. All three shapes returned 16 candidates after filtering and deduplication. Some zero-byte heuristic returns required 4 bytes under `cublasLtMatmulAlgoCheck`. The search skips those entries for the zero-byte cap and retains the validated configuration from the 32 MiB query. The [initial failed attempt](700b-c8-first.log) records how this discrepancy was discovered.

The selected 700B pair uses cuBLASLt algorithm 66, tile 29, one K partition and 4 bytes workspace. β = 0 uses the column-major layout and β = 1 uses the row-major layout. The generic uses algorithm 66 tile 209. Both 700B timed labels in the verification run resolve to the same algorithm pair. The selection mechanism follows NVIDIA’s bounded heuristic-and-timing example and queries the official API for actual workspace support. Both layouts use `CUBLAS_COMPUTE_32F`, BF16 A/B and FP32 C/D in place. [NVIDIA autotuning example at a pinned revision](https://github.com/NVIDIA/CUDALibrarySamples/blob/a2eddd862e886c0ea5827013919801a1ddc25430/cuBLASLt/LtSgemmSimpleAutoTuning/sample_cublasLt_LtSgemmSimpleAutoTuning.cu) · [NVIDIA cuBLASLt API](https://docs.nvidia.com/cuda/archive/13.1.2/cublas/index.html#cublasltmatmulalgocheck)

```python
from tuning.lt.lt_writer import make_dw_writer

writer = make_dw_writer(receipt=root / "tuning/lt/results/calibrated-catalog.json")
writer(dlogits, hidden, fp32_slot, accumulate, stream)
activation = writer.receipt()
```

The [shared calibrated catalog](results/calibrated-catalog.json) covers all three measured shapes. The replay factory also accepts an individual receipt and caches the selected plan by shape and buffer alignment on the given stream. It checks the device model, capability and cuBLASLt version before restoring the benchmark receipt. Unmatched metadata uses the native generic writer. A CPU dispatch check with the measured descriptors selected all three calibrated pairs, confirmed format-1 compatibility, and verified native generic fallback for an unknown shape, insufficient alignment and a cuBLASLt version mismatch. Replay adds Python metadata validation, cache lookup and beta counters. It makes no tuning launches, copies, additional GEMMs, or synchronization calls on the warmed path, and keeps one scratch allocation of 4 requested bytes for the measured shape. No training-source files are changed. Full-head integration and native collectives are outside these measurements.

## Reproduction

The remote root is `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125`. [The launcher](run.sh) initializes the existing toolchain, confirms the assigned device is idle and maps its UUID as the only visible GPU.

```bash
bash "$root/tuning/lt/run.sh" "$root" 700b-c8 0
bash "$root/tuning/lt/run.sh" "$root" 100b-c2 1
bash "$root/tuning/lt/run.sh" "$root" 100b-c8 2
bash "$root/tuning/lt/run.sh" "$root" 700b-c8 0 \
    --reuse-screen "$root/tuning/lt/results/700b-c8-screen-initial.json"
```

PyTorch `2.15.0.dev20260822+cu130`, build CUDA 13.0, driver 570.133.20, cuBLASLt 130101. The receipts identify the device UUID, native writer and CE artifact hashes, source fixture path and row hashes. Scheduler job ID was unavailable inside the process.

- [700B initial samples](results/700b-c8-screen-initial.json) · [Verification samples, memory and oracle](results/700b-c8.json) · [Bitwise plan](results/700b-c8-plan-bitwise.json) · [Verification log](700b-c8-verification.log)
- [100B C2 samples, memory and oracle](results/100b-c2.json) · [Bitwise plan](results/100b-c2-plan-bitwise.json) · [Run log](100b-c2.log)
- [100B C8 samples, memory and oracle](results/100b-c8.json) · [Bitwise plan](results/100b-c8-plan-bitwise.json) · [Run log](100b-c8.log)
