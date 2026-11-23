# Fused centered cross-entropy

An eight-CTA CUDA kernel reads each valid BF16 logit once and writes its final
scaled BF16 gradient back to the same allocation. It matched native FP32
per-row loss and BF16 dLogits throughout the directly checked C2 and C8 chunks.
Complete isolated replay digests matched for weighted loss, BF16 dHidden,
FP32 dW, hidden z-loss, and its BF16 gradient. The packed-BF16-max variant
saved 0.59 ms at C2 and 1.33 ms at C8 in the regular-gather replay.
With aliased gather, it saved 1.05 ms at C2 and 1.31 ms at C8 at the same
GIFT total peaks of 10.681 and 9.177 GiB, respectively.

## Complete isolated replay

All arms used the same real capture, one H100 per shape, Inductor-compiled
CE, direct BF16 dHidden, early release of logits, the existing tuned FP32
cuBLASLt dW writer, the native GFSDP callbacks and z-loss, and expandable
allocator segments. A control already overwrites its BF16 logits in CE
backward. The experiment moves the write into the forward kernel.

The table reports the median of process medians. Each process took ten
synchronized wall-time measurements after ten warmups and a priming call
with the accumulated gradient retained. The memory columns are peak GiB
over all timed invocations for allocated, reserved, and native GIFT total
memory. Allocated and reserved show the allocator state behind the total.

| Chunks | Gather | Control → fused, wall ms | Wall change | Allocated GiB | Reserved GiB | GIFT total GiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | Regular, cuda-13, four pairs | 76.153 → 75.569 | −0.77% | 10.34824 | 10.35352 | 11.63757 |
| 8 | Regular, cuda-13, five pairs | 78.506 → 77.709 | −1.01% | 8.83359 | 8.84961 | 10.13367 |
| 2 | Aliased, cuda-13 | 76.464 → 75.667 | −1.04% | 9.36972 | 9.39648 | 10.68054 |
| 8 | Aliased, cuda-13 | 78.830 → 77.942 | −1.13% | 7.85508 | 7.89258 | 9.17664 |
| 2 | Regular, cuda-16 | 75.900 → 75.314 | −0.77% | 10.34824 | 10.35352 | 11.63757 |
| 8 | Regular, cuda-16 | 78.487 → 77.160 | −1.69% | 8.83359 | 8.84961 | 10.13367 |
| 2 | Aliased, cuda-16 | 76.331 → 75.280 | −1.38% | 9.36972 | 9.39648 | 10.68054 |
| 8 | Aliased, cuda-16 | 78.767 → 77.454 | −1.67% | 7.85508 | 7.89258 | 9.17664 |

For cuda-13, the median adjacent-pair change was −0.634 ms at C2 and
−0.894 ms at C8.
All four regular C2 pairs favored fusion, including one near tie (−0.005 ms).
Four of five regular C8 pairs favored fusion, with individual changes from
−1.046 to +0.336 ms. Both aliased-gather pairs favored fusion for each
chunk count. The two regular-gather cuda-16 pairs changed by −0.402 and
−0.769 ms at C2, and −1.232 and −1.422 ms at C8. The aliased-gather
cuda-16 pairs changed by −1.528 and −0.574 ms at C2, and −1.382 and
−1.244 ms at C8.

The fused route added 2 KiB to PyTorch peak allocation in each arm. Peak
reserve and the GIFT total estimate were exactly equal. The aliased-gather
arms retained all 24 native singleton all-gathers and verified that the
send and receive slice addresses match. The regular gather follows the
path in the most recent eight-rank training candidate. Aliased gather
represents the earlier minimum-memory isolated frontier.
[Per-process samples, pair deltas, and digest checks](summary.json)

## CE device timing and arithmetic

| CUDA configuration | C2 ms/chunk | C8 ms/chunk | FP32 row loss and BF16 dLogits |
| --- | ---: | ---: | --- |
| Native forward and consuming backward | 2.188 | 0.582 | Control |
| Four-CTA global-max reduction, cuda-4 | 2.082 | 0.557 | Changed 57,378 / 13,902 BF16 values |
| Four-CTA native-order reduction, cuda-11 | 1.743 | 0.454 | Bitwise match |
| Eight-CTA native-order reduction, cuda-13 | 1.657 | 0.432 | Bitwise match |
| Eight-CTA native-order reduction with packed BF16 max, cuda-16 | 1.518 | 0.390 | Bitwise match |

The CE-only CUDA event spans start after restoring the original logits and
end after writing BF16 dLogits. They include queue gaps between the four
native launches and exclude projection, reductions over per-row losses,
packing, GEMMs, and the input restoration. Multiplying paired per-chunk
savings by chunk count gives about 1.06 ms at C2 and 1.20 ms at C8 for cuda-13. The same calculation
gives 1.34 ms and 1.49 ms for cuda-16. CUDA Graph replay on the final microbatch chunk removed
Python launch spacing from the device measurement and measured
2.182 → 1.652 ms at C2 and 0.578 → 0.430 ms at C8. It compared the raw
bytes of the full loss and dLogits tensors and matched each value.
The integrated wall figures include host submission and final synchronization.
The replay does not attribute their costs individually or separate the
copies, launch gaps, and native singleton communication.

The selected CUDA configurations use eight CTAs per row and 256 threads per
CTA. Each CTA stages up to 32 KiB of packed BF16 logits in shared memory. It computes each 4,096-value tile maximum in parallel, derives
running maxima using the same tile order, and reproduces the native FP32
pairwise lane tree, warp shuffle order, and sequential denominator merge.
The FP32 logarithm uses CUDA libdevice. The gradient subtracts the maximum,
then the log denominator, applies the exponent, subtracts one at the target,
multiplies by the upstream head scale and rounds once to BF16. The first
implementation used a parallel global-max sum and an approximate base-2
logarithm. It had similar FP64 error on the sampled rows, and it changed
the full gradient digests. Two Triton CTA-cluster schedules and the
register-retained CUDA schedules were slower than the native path. CUDA
packed BF16 pairwise maxima retained the native maxima in the measured
inputs. Special-casing the target inside the vector store remained
bitwise-equal but slowed the kernel. Quack uses a related row-retention
schedule with different loss and gradient arithmetic.
[Pinned Quack CE](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/quack/cross_entropy.py)

The direct comparison covered 1,050,673,152 BF16 entries and 8,192 FP32 row
losses at C2, and 262,668,288 entries and 2,048 row losses at C8, before the
same per-head FP32 loss sums. It includes ignored targets, packed shifting
for MTP, and injected targets at vocabulary indices 0 and 128,255. Ignored
gradients were zero. The exact route matched the native FP64 error on six
selected rows. The actual incoming replay scales matched the precomputed
FP32 scales for both heads, 4.769536303683708e-7 and 1.4308609763702407e-7.
The replay read the captured host denominator 2,096,640 and native weights
1.0 and 0.3. [Direct C2 data](raw/check-c2-exact.json)
[Direct C8 data](raw/check-c8-exact.json)
[Reordered C2 data](raw/check-c2-v1.json)
[Reordered C8 data](raw/check-c8-v1.json)
[Graph replay C2](raw/graph-c2-exact.json) [Graph replay C8](raw/graph-c8-exact.json)
[Packed variants C2](raw/check-c2-opt.json) [Packed variants C8](raw/check-c8-opt.json)

The selected CUDA SASS contains eight 128-bit global loads per thread in
the shared-memory staging phase, with eight corresponding vector stores to
shared memory and zero compiler-reported register spills. The remaining
scalar global loads fetch labels and scales. Inductor calls the mutating
CE op with the original logits tensor. Native CE issues another full
logical logits read in backward, about 3.91 GiB per microbatch at this
shape. The fused kernel eliminates that read for valid rows. Physical HBM
bytes were unmeasured because hardware counters on this allocation require
privileges unavailable to the experiment user. The amount served from
cache in either arm remains unknown.
[Cuda-13 SASS instructions](raw/cuda-13-sass-instructions.txt)
[Cuda-16 SASS instructions](raw/cuda-16-sass-instructions.txt)
[Compiler resource usage](raw/build-exact-log.log)
[Packed kernel compiler report](raw/build-opt.log)
[Device and counter permissions](raw/gpu-environment.txt)
[Inductor call boundary](raw/inductor-consume-boundary.txt)

## Reproduction and scope

- SR008 job 84fd3579-7bd2-4969-ae47-b7af17b73ca2, H100 80 GB HBM3, GPU 4
  for C2 and GPU 5 for C8. Driver 570.133.20, PyTorch
  `2.15.0.dev20260822+cu130`, Triton 3.8.0, CUDA runtime 13.0 and CUDA
  compiler 13.2. Receipts record UUID, compilation counters, FP32 dW
  algorithm pair, activation calls, and allocator retries.
- Integrated harness `optimization-task/.worktrees/lm-head-mtp-fix/runs/lm-head-mtp`
  at `2c67c182e8de6145f3796d62203714ab1b815446`. Standalone checks import
  kernels from `lm10-c1/gift`. Both sources have identical native CE
  `forward_kernel`, `backward_kernel`, and `_launch_backward`.
  [Kernel identity checks](raw/native-ce-kernels.txt)
- Real capture `runs/kernel-fusion/artifacts/raw/backward-inputs-001/component-inputs/rank0.pt`,
  SHA-256 `15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727`.
  BF16 projection with V128256, D4096, B2, S4096, and one MTP head.
  Independent captured z-loss input SHA-256
  `dced479c2bea4ff5cc5ade9dad04531d075d74f4bbc5c79e3f069c049d8863a1`.
  The integrated replay fixes seed 2048 and disables TF32.
- Build with `/usr/local/cuda-13.2/bin/nvcc -O3 -shared -Xcompiler=-fPIC -lineinfo -Xptxas=-v -arch=sm_90a --fmad=false --cudart=static centered.cu -o centered.so`.
  Launch with `bash paired.sh 4 2 cuda-13 SERIES control fused fused control`
  and the corresponding GPU 5 C8 command. Select `cuda-16` for packed
  BF16 maxima. Set `FUSED_CE_ALIAS_GATHER=1`
  for the aliased-gather series. [Launcher](paired.sh)
  [Integration](integrated.py) [CUDA launch](centered.cu)
  [Ordered kernel](exact.cuh) [Direct check](check.py)
  [Receipt verifier](summarize.py)
- The experiment exercises the immediate first-order training VJP of the
  two-head local BF16 folded loss with known normalization. Bitwise parity
  depends on the native 4,096-element reduction layout and libdevice
  arithmetic used on this stack. Evaluation,
  higher-order gradients, TP/DTensor, other dimensions and dtypes, and
  full training or multi-rank timing are outside this measured scope.
