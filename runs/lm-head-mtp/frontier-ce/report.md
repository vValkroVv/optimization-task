# Stable CE vocabulary tiling

The tiled backward computes the same BF16 logit gradients. The head and loss
replay showed a lower median at each chunk count, with changes comparable to
process variation and unchanged peak total GPU memory.

| Fixed chunks | Native wall ms | Tiled wall ms | Native process spread, ms | Tiled process spread, ms | Peak total GPU, GiB, both |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 78.114 | 77.619 | 1.426 | 1.370 | 14.194 |
| 4 | 80.324 | 79.997 | 0.733 | 2.138 | 12.237 |
| 8 | 83.256 | 82.846 | 0.473 | 1.410 | 11.261 |
| 16 | 93.814 | 93.435 | 0.279 | 0.613 | 10.280 |

The source revision is `2c67c182e8de6145f3796d62203714ab1b815446`.
Each wall cell is the median of two process medians from a same-GPU
ABBA or BAAB series. Each process warmed up nine calls and took 14
synchronized host-wall samples. Chunks 2 and 4 ran on H100 GPU 2 and
chunks 8 and 16 on H100 GPU 3. Both arms used compiled Folded Stable CE,
BF16 dHidden, fused hidden z-loss, the cuBLAS FP32 GFSDP gradient writer,
and singleton SUM. Both retained the primed FP32 gradient across timed
calls. The receipts recorded zero allocator retries and OOMs. Total
memory uses the replay's native GIFT estimator (peak allocator reserved
bytes plus external device use at observation). Integrated latency includes
host dispatch, CUDA synchronization, native target packing, loss bookkeeping,
head forward and backward, BF16 hidden grad, FP32 dW, and the z-loss.
Host dispatch and wait, copies, and communication were not separately
attributed. [Replay methodology](../report.md) [2/4 receipts](matched-2-4)
[8/16 receipts](matched-8-16)

The first A/B pair at each fixed count recorded matching SHA-256 digests of
the complete CE loss, BF16 dHidden, FP32 dW, z-loss, and BF16 z-loss
gradient. The `.pt` digest receipts contain JSON metadata and hashes,
rather than raw tensors. In a separate activation check, all five tested
tiles returned elementwise-equal BF16 dLogits at all four shapes. That
check used captured projected logits, native packed MTP targets, both
head scales, an explicit ignored row, and targets at vocabulary columns
0 and 128255. The ignored gradients were exactly zero. Selected-row
relative L2 versus a FP64 softmax gradient was 0.0011116–0.0011128 in
both arms. The activated Triton kernel was generated in the integrated
smoke's fresh cache. [Kernel receipt](kernel/result.json)
[Integrated smoke](smoke)

## Device measurements

A standalone native backward uses two head kernels, one CTA per row, with
32 vocabulary blocks visited in sequence. The tiled candidate uses one
head-aware kernel, one CTA per row and 8192-column tile, with eight warps.
It reuses the same FP32 maximum, log denominator, and scalar scale for each
tile. It evaluates the same centered exponential and subtraction before a
single BF16 store to a contiguous output. Ignored tiles skip the logits
load and store zero. There is no new scratch allocation proportional to
logits, and native GEMMs consume the same contiguous dLogits shape.

| Fixed chunks | Folded rows per chunk | Native backward, ms | Tiled backward, ms |
| ---: | ---: | ---: | ---: |
| 2 | 8192 | 1.5006 | 1.4376 |
| 4 | 4096 | 0.7680 | 0.7205 |
| 8 | 2048 | 0.3989 | 0.3612 |
| 16 | 1024 | 0.2009 | 0.1824 |

These medians cover device execution of preallocated kernels on one CUDA
stream. Each arm ran six CUDA-event windows of ten warm-cache CUDA Graph
replays, with alternating ordering. Allocation, host launching of the CE
operator, and surrounding GEMMs are outside these windows. The measured
kernel-span change corresponds to approximately 0.13–0.30 ms when
multiplied by the number of outer chunks in this replay.

## Forward alternatives

A single launch of the native row kernel across both heads was bitwise
equal for all checked forward tensors. Its preallocated CUDA Graph span
was 0.6692 ms versus 0.6924 ms at 8192 folded rows and 0.0924 ms versus
0.1135 ms at 1024 rows. The per-head FP32 sum remains separate.

A two-stage FP32 tile/merge reduction and a Triton CTA-cluster reduction
were also compared. The tile/merge route was 0.6914 ms at 8192 rows and
0.0888 ms at 1024 rows in the fastest configurations at each shape.
It needs temporary FP32 tile maxima and sums, and changed up to 88
selected BF16 gradient elements at 8192 rows. The maximum selected-row
CE loss error versus FP64 was 6.68e-7 for native and at most 5.52e-7 for
tile/merge. FP64 dLogits relative L2 was approximately 0.000726 in both.
CTA-cluster variants took 1.657–1.855 ms at 8192 rows and 0.213–0.236 ms
at 1024 rows. The two-stage and clustered variants were kept as standalone
forward diagnostics. [Forward receipts](forward/result.json)
[Pinned Quack vocabulary reduction reference](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/quack/cross_entropy.py)

At 16 fixed chunks, the compiled replay combining the single native
forward launch with tiled backward measured 93.501 ms versus 93.882 ms
for native in fresh GPU 3 BAAB processes. Each process had seven warmup
calls and 12 synchronized host-wall samples. Both arms recorded 10.280 GiB
peak total GPU memory, one captured graph, zero allocator retries and
OOMs, and matching hashes for the complete loss and gradients. The
0.381 ms difference is smaller than the native process spread of
1.135 ms. [Consolidated replay receipts](matched-consolidated-16)

## Reproduction

Run the [launcher](../frontier_ce_run.sh) from the authorized worktree
using an idle GPU. It records the command arguments, environment, and
individual samples. Pass arms as mode:chunks:tile:warps:warmup:samples
and append :yes to request full-output digests. The integrated launcher
patches the selected custom operator in-process in the
[experiment wrapper](../frontier_ce.py). It leaves the production checkout
and the original replay script untouched. The standalone
[backward check](../frontier_ce_check.py) and [forward check](../frontier_ce_forward_check.py)
use the same interpreter and CUDA visibility from the launcher.
