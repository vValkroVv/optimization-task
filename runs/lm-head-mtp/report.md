# LM-head and MTP loss experiments

## Original A training at 10+10 on eight H100s

All six runs completed on sr008-gpu. The selected original A source was
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`. The runs use the same eight-layer
100B crop, B2/S4096, global batch 512, accumulation 32, EP8/eFSDP1,
dense FSDP8, model and loss compilation, per-layer SAC, MTP1, and the
same real data-002 and production tokenizer. All use the inherited
`expandable_segments:True`. Native FLA, original routing, and the rest of
the resolved recipe match across arms. Only loss chunk count and the
listed LM-head/loss changes vary across the measured configurations.

Original A uses its original CE. The previous candidate enables Stable
and Folded CE, direct BF16 dHidden, fused hidden z-loss, and direct FP32
dW via cublasGemmEx. The new candidate adds early logits release, in-place
CE backward into the BF16 logits allocation, and tuned cublasLtMatmul
for dW. Both candidate arms use native GFSDP reduce-scatter and the
original separate BF16 all-gather input and output.

| Loss chunks | Variant | Median physical global TPS | Peak total GPU, GiB | TPS vs same-count A | Peak vs same-count A, GiB | Raw run |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | Original A | 157,072 | 72.38 | +0.00% | +0.00 | [lh10-a2-001](../../../../../a10-cute-001/runs/production-data-abcd/artifacts/raw/lh10-a2-001) |
| 2 | Previous LM-head | 158,553 | 70.27 | +0.94% | -2.11 | [lh10-p2-002](../../../../../lm10-c1/runs/production-data-abcd/artifacts/raw/lh10-p2-002) |
| 2 | New LM-head, regular all-gather | 158,800 | 68.27 | +1.10% | -4.11 | [lh10-n2-002](../../../../../lm10-c1/runs/production-data-abcd/artifacts/raw/lh10-n2-002) |
| 8 | Original A | 152,209 | 68.86 | +0.00% | +0.00 | [lh10-a8-001](../../../../../a10-cute-001/runs/production-data-abcd/artifacts/raw/lh10-a8-001) |
| 8 | Previous LM-head | 154,605 | 68.28 | +1.57% | -0.58 | [lh10-p8-002](../../../../../lm10-c1/runs/production-data-abcd/artifacts/raw/lh10-p8-002) |
| 8 | New LM-head, regular all-gather | 158,454 | 68.30 | +4.10% | -0.56 | [lh10-n8-002](../../../../../lm10-c1/runs/production-data-abcd/artifacts/raw/lh10-n8-002) |

The new 2-chunk arm saved 2.00 GiB versus the
previous 2-chunk arm, with a +0.16% observed TPS change.
At 8 chunks it changed TPS by +2.49% and total peak by
+0.02 GiB versus the previous 8-chunk arm.
Each cell is one run. Small differences remain sensitive to run-to-run
variation.

TPS is the median of 2,097,152 physical tokens divided by the slowest
rank's update interval for each of updates 11–20. Total GPU memory is
the maximum native GIFT console estimate across all eight ranks and
the same ten updates. Compiler observers were stable in all six measured
windows. All eight ranks logged the intended candidate dW writer,
compiled Folded CE mode and fused z-loss. The new candidate compiles
CE forward statistics and uses the Triton in-place backward via eager
autograd.

The first candidate attempt stopped before training because the detached
worktree lacked an empty raw-artifacts directory expected by the runner.
The completed retry used the same candidate source and configuration.
[Verified matrix and step series](../../../../../lm10-01/matrix.json)
[Launchers and all-rank summaries](../../../../../lm10-01)

## Result

At each chunk count, original compiled GIFT is paired with the current path
on the same H100. The current path applies Stable cap-zero CE, folded CE,
direct BF16 dHidden, and fused hidden z-loss. It uses `cublasGemmEx` to
accumulate dW directly into the FP32 gradient buffer through the GFSDP lease.

| Loss chunks | Original wall, ms | Current wall, ms | Wall change | Original total GPU, GiB | Current total GPU, GiB | Memory change, GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 83.71 | 76.83 | -8.22% | 16.151 | 14.194 | -1.957 |
| 4 | 89.10 | 79.22 | -11.08% | 14.196 | 12.237 | -1.959 |
| 8 | 99.74 | 83.16 | -16.62% | 14.194 | 11.261 | -2.934 |
| 16 | 116.81 | 93.72 | -19.77% | 11.751 | 10.280 | -1.471 |

Each row used two processes per arm in ABBA or BAAB order, with ten warmup
and ten timed calls per process. Wall time is the median of the two process
medians. Total GPU memory is the larger of the two native peak estimates.
The two process medians within an arm differed by 0.03 to
1.72 ms across the matrix. All runs used singleton SUM
and recorded zero allocator retries and OOMs. Each row ran on a different
H100. [Verification summary](artifacts/matched-ancestral-004/verified-summary.json)
[Run receipts and launcher](artifacts/matched-ancestral-004)

## Memory-priority BF16 replay

### Full memory stack versus current GemmEx

The full stack lowers peak total GPU memory by 1.377–3.514 GiB at fixed
loss chunk counts. The 8-chunk configuration uses 9.177 GiB at 78.82 ms.
The lowest measured peak is 8.903 GiB at 16 chunks.

| Fixed loss chunks | GPU | GemmEx → memory stack wall, ms | Wall change | GemmEx → memory stack total GPU, GiB | Memory saved, GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3 | 77.411 → 76.615 | −1.03% | 14.194 → 10.681 | 3.514 |
| 4 | 3 | 78.929 → 77.780 | −1.46% | 12.237 → 9.684 | 2.553 |
| 8 | 4 | 81.564 → 78.820 | −3.36% | 11.261 → 9.177 | 2.084 |
| 16 | 5 | 92.012 → 89.296 | −2.95% | 10.280 → 8.903 | 1.377 |

The control uses the four existing fixes, the compiled loss path and the
native FP32 GemmEx writer. The candidate releases logits after each chunk,
overwrites them with BF16 dLogits in backward, tunes the FP32 dW writer with
cuBLASLt, enables expandable allocator segments, and aliases the BF16
all-gather send buffer with its output slice. Both arms retain all 24
native all-gathers in the singleton replay.

Each row compares two fresh processes per arm on the same physical H100
using ABBA or BAAB order, ten warmup calls and ten synchronized timed
calls per process. Wall time is the median of the two process medians.
Total GPU memory is the maximum native GIFT peak over timed invocations
and replicates. It includes allocator reserve and sampled GPU memory
outside the allocator. Within-arm process spread was 0.025–0.968 ms.
The full FP32 loss, BF16 dHidden, FP32 dW, FP32 z-loss and BF16 z-loss
gradient matched bitwise within every row. The head inputs and gradients
follow the same precision boundaries. Absolute wall times across rows
on different GPUs are descriptive. A separate two-rank GFSDP replay
at eight chunks isolates the all-gather change with actual inter-GPU
collectives. Peak total GPU memory went from 8.016 to 7.509 GiB per rank.
[Two-rank isolation](#two-rank-gfsdp-replay-at-eight-chunks)

[Direct matched results](artifacts/frontier-best-final-001/verified-summary.json)
[Launch logs and per-process receipts](artifacts/frontier-best-final-001)
[Full stack wrapper](frontier_best.py)
[Launcher](frontier_best_lane.sh)
[Result verifier](frontier_best_verify.py)

### Default-allocator stack comparison

The early-release and in-place CE combination lowered the total GPU peak by
1.957 GiB at 2 chunks and about 0.98 GiB at 4 and 8 chunks. Adding the tuned
cuBLASLt dW writer also lowered replay wall time by 3.33% at 8 chunks and
3.02% at 16 chunks. The 8- and 16-chunk candidates both reached 10.280 GiB
with the default allocator. Eight chunks therefore gives the lower measured
wall time at that memory floor in this sweep. Each fixed-count comparison
used a single H100. The two chunk counts used different H100s, so their
wall-time difference is descriptive.

| Fixed chunks | Folded rows per chunk | GPU | Current → candidate wall, ms | Wall change | Process spread, ms, current / candidate | Current → candidate total GPU, GiB | Memory change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 8192 | 5 | 76.775 → 76.401 | −0.49% | 0.017 / 0.810 | 14.194 → 12.237 | −1.957 GiB (−13.79%) |
| 4 | 4096 | 3 | 78.557 → 78.307 | −0.32% | 0.111 / 0.288 | 12.237 → 11.259 | −0.979 GiB (−8.00%) |
| 8 | 2048 | 4 | 81.614 → 78.897 | −3.33% | 0.645 / 0.889 | 11.261 → 10.280 | −0.980 GiB (−8.71%) |
| 16 | 1024 | 2 | 92.680 → 89.883 | −3.02% | 1.371 / 0.513 | 10.280 → 10.280 | 0.000 GiB (0.00%) |

Both arms use the captured main/MTP targets and head weights, bias-free BF16
projection, Folded Stable cap-zero CE, BF16 dHidden, fused hidden z-loss
and FP32 dW in the native GFSDP slot. The current replay uses
`cublasGemmEx`. The candidate drops the logits reference after each chunk,
overwrites logits with dLogits in CE backward and uses a shape-tuned
`cublasLtMatmul` writer with 4 bytes of scratch. Both writers consume the
same BF16 operands and use FP32 accumulation and output. The writer uses
beta zero on the first chunk and beta one thereafter.

Each row uses two fresh processes per arm in ABBA or BAAB order, with ten
warmup and ten timed invocations per process. Wall time is the median of
the two process medians. It includes host dispatch, label packing, loss
bookkeeping, head forward/backward, the independent z-loss and CUDA
synchronization. Process spread is the absolute difference between the
two medians. The 2- and 4-chunk wall changes are within the larger repeat
spread. Total GPU peak is the maximum native estimate across timed
invocations and both replicates (allocator peak reserved plus observed
external GPU use). Separate peak allocated bytes and attribution to device
compute, host work, host waits, copies and communication are unavailable
for this series. Changes are relative to the same-GPU current arm. All
receipts show one captured Inductor graph, CUDA Graph replay disabled,
singleton SUM and zero allocator retries or OOMs. GPUs 6 and 7 were
reserved. [Verified results](artifacts/frontier-stack-final-001/verified-summary.json)
[Launch logs and per-process receipts](artifacts/frontier-stack-final-001)
[Writer selection and NVIDIA API](frontier-dw/report.md)

At each fixed chunk count, all four output digests match for the FP32
loss, BF16 dHidden, FP32 GFSDP dW, FP32 z-loss and BF16 z-loss gradient.
The digests cover the full outputs of a post-timing invocation with the
weight gradient reset. The independent dW oracle compares each writer
with a full FP64 gradient computed on its actual BF16 callback operands.
Both writers have identical relative L2 and maximum absolute error at
each count. Relative L2 goes from 1.295e−5 at 2 chunks to 2.164e−6 at 16
chunks. The CE storage check compares BF16 dLogits elementwise against
Stable CE, including zero dLogits on ignored rows. These comparisons
cover this fixture and precision contract. [Writer oracle](frontier-dw/verified-summary.json)
[CE storage oracle](artifacts/ce-storage-check-004/result.json)

### Logits lifetime and arithmetic

The 2×2 screen kept GemmEx fixed. One process per arm shows that both
lifetime changes were needed to lower the reported peak at 2, 4 and 8
chunks. Early release removes the reference that otherwise survives into
the next projection. In-place backward removes the separate BF16
dLogits allocation. FlashAttention exposes in-place CE backward. Liger
uses a reusable chunk buffer and explicitly releases aliases, with a
different whole-op gradient contract. Apple CCE avoids materializing the
global logits matrix and remains a distinct approach. The local whole-op
comparisons appear below. [Pinned FlashAttention CE](https://github.com/Dao-AILab/flash-attention/blob/dbabcc788590e793eee11566d7253906187e612e/flash_attn/losses/cross_entropy.py)
[Pinned Liger Linear+CE](https://github.com/linkedin/Liger-Kernel/blob/85a9c40f1f5f9bb39cf74e70ccbc3d416479ee42/src/liger_kernel/ops/fused_linear_cross_entropy.py)
[Apple CCE paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/aa963ac256590bb7ad5fc26c68229a3a-Abstract-Conference.html)

| Fixed chunks | Native peak, GiB | Early release only, GiB | In-place CE only, GiB | Both, GiB | Repeated native → both wall, ms | Repeated native / both spread, ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 14.194 | 14.194 | 14.194 | 12.237 | 76.337 → 76.576 | 0.572 / 0.619 |
| 4 | 12.237 | 12.237 | 12.237 | 11.259 | 79.172 → 78.456 | 0.562 / 0.727 |
| 8 | 11.261 | 11.261 | 11.261 | 10.280 | 82.011 → 81.873 | 0.308 / 0.679 |
| 16 | 10.280 | 10.280 | 10.280 | 10.280 | 93.576 → 92.768 | 0.294 / 0.616 |

The repeated native/combined comparison reproduced the peak in each row,
with matching full-output digests. Timing effects from the lifetime
changes alone are small. The separate dW-writer comparison with native
storage reproduced most of the stack's timing benefit at 8 and 16 chunks
(−3.21% and −3.72% wall, equal peak within each pair). These separate
experiments used their own matched controls. [Lifetime 2×2 receipts](artifacts/frontier-lifetime-screen-001)
[Repeated lifetime receipts](artifacts/frontier-lifetime-final-002)
[Repeated writer receipts and device timing](frontier-dw/report.md)

- Vocabulary-tiled Stable CE backward gave elementwise-equal BF16
  dLogits. The isolated CUDA Graph kernel savings corresponded to roughly
  0.13–0.30 ms per replay. Whole-replay differences were comparable to
  process spread and the total peak was unchanged at all counts.
  [CE tiling measurements](frontier-ce/report.md)
- At 8 chunks, tuned projection improved a preallocated warm-first-chunk
  CUDA Graph GEMM from 3.336 to 3.159 ms. In the complete native-storage
  replay the median moved from 81.559 to 80.552 ms, within the 2.237 ms
  native spread, and the peak stayed at 11.261 GiB. The first replay pair's
  output digests matched. The projection and dHidden screen used BF16
  outputs with reduced-precision BF16 reduction disabled in both arms.
  Tuning dHidden showed no clear device-time gain. [Projection and dHidden screen](artifacts/frontier-projection-001/screen.json)
  [Projection replay receipts](artifacts/frontier-projection-001)
- Pairing adjacent chunks only for dW shortened the standalone CUDA-event
  span from 27.795 to 25.334 ms at 8 chunks and from 38.485 to 27.940 ms at
  16. Those spans start with preassembled BF16 operands. Full-gradient
  relative L2 versus FP64 rose from 3.823e−6 to 5.418e−6 at 8 chunks and
  from 2.164e−6 to 2.958e−6 at 16. Grouping changes the FP32 accumulation
  order even with the same BF16 inputs. The faster grouped arm fails the
  preserve-precision criterion and would need extra operand lifetime in a
  replay. [Grouped dW receipts](artifacts/frontier-grouped-dw-001)

### Allocator and peak attribution

Expandable segments lowered the candidate's reported total peak at all
four chunk counts. At 16 chunks the peak fell to 9.880 GiB in this
comparison. The 8-chunk run reached 10.134 GiB and had the lower
observed wall time. Each allocator pair is matched within a chunk count.
Cross-count timing is descriptive because the H100s differ.

| Fixed chunks | GPU | Default → expandable wall, ms | Process spread, ms, default / expandable | Default → expandable total GPU, GiB | Memory change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 76.374 → 77.061 | 0.422 / 0.776 | 12.237 → 11.638 | −0.600 GiB (−4.90%) |
| 4 | 3 | 78.328 → 77.978 | 1.490 / 0.546 | 11.259 → 10.641 | −0.617 GiB (−5.48%) |
| 8 | 5 | 79.199 → 78.109 | 0.446 / 0.343 | 10.280 → 10.134 | −0.146 GiB (−1.42%) |
| 16 | 2 | 90.650 → 89.417 | 0.856 / 0.925 | 10.280 → 9.880 | −0.400 GiB (−3.89%) |

These comparisons hold the early-release, in-place CE and cuBLASLt
candidate fixed. They used two fresh processes per arm in ABBA or BAAB
order, ten warmup and ten timed invocations, and the same integrated
timing and memory scopes as above. All full-output digests agree within
each row. There were zero allocator retries or OOMs. The setting affects
the whole allocator process. Its memory effect with decoder allocations,
multiple ranks and production communication remains open.
[Allocator receipts at 2, 4, and 8](artifacts/frontier-allocator-final-002)
[Allocator receipts at 16](artifacts/frontier-allocator-screen-001)
[PyTorch allocator semantics](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-memory-management)

| Replay snapshot | Peak allocated, MiB | Peak reserved, MiB |
| --- | ---: | ---: |
| Current, 2 chunks, default | 12600.598 | 13220 |
| Candidate, 2 chunks, default | 10596.598 | 11216 |
| Candidate, 8 chunks, default → expandable | 9045.598 → 9045.598 | 9212 → 9062 |
| Candidate, 16 chunks, default → expandable | 8787.098 → 8787.098 | 9212 → 8802 |

The separate snapshots follow allocation history through the first timed
invocation. The analyzer reconstructs the allocated peak and checks it
against PyTorch allocator counters. At 2 chunks the candidate drops
2004 MiB from both allocated and reserved peaks, the size of one
8192×128256 BF16 tile. At 8 and 16 chunks the runtime snapshots confirm
the allocator setting and show the same live peak in each pair.
Expandable segments lowered reserve by 150 MiB at 8 chunks and 410 MiB
at 16 chunks. With the default allocator, both counts reserved 9212 MiB
even though the 16-chunk live peak was 258.500 MiB lower.
[Snapshot receipts](artifacts/frontier-memory-snapshot-002)
[Snapshot analyzer](frontier_memory.py)

### In-place BF16 all-gather

The private head prototype casts the FP32 parameter shard directly into
its rank slice of the BF16 all-gather output. NCCL explicitly supports
this pointer layout. The same GIFT API, communicator, CUDA streams and
gradient reduction run in both arms. The alias removes a separate
BF16 send allocation while retaining the BF16 weight read by the head.
The prototype invalidates the borrowed input before each real gather,
because another GFSDP group could overwrite the shared output between
gathers. This raises the observed master-to-BF16 cast count from 1 to 24.
The native all-gather API still executes 24 times in both singleton arms.
[NCCL in-place layout](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather)
[GFSDP alias experiment](frontier_gather.py)

| Fixed chunks | Memory stack → with alias wall, ms | Wall change | Stack → with alias total GPU, GiB |
| ---: | ---: | ---: | ---: |
| 2 | 75.901 → 76.389 | +0.64% | 11.638 → 10.681 |
| 4 | 77.322 → 77.884 | +0.73% | 10.641 → 9.684 |
| 8 | 78.916 → 78.248 | −0.85% | 10.134 → 9.177 |
| 16 | 89.424 → 89.257 | −0.19% | 9.880 → 8.903 |

Both arms have early release, in-place CE, tuned cuBLASLt and expandable
segments. All rows ran on physical GPU 4, with two fresh processes per
arm and ten warmup and ten timed invocations each. Every row saved
exactly 1002 MiB at the allocated peak. Differences in allocator reserve
account for the varying total savings. The full fresh-gradient dW digest
and the loss, dHidden, z-loss and z-loss gradient digests matched within
every row. Allocator retries and OOMs remained zero.
[Verified alias ablation and method](artifacts/frontier-gather-repeat-001)

#### Two-rank GFSDP replay at eight chunks

This isolated test used the native GFSDP head on two NVLinked H100s.
Each rank held a disjoint parameter shard and a disjoint FP32 gradient
shard. Both ranks replayed the same captured rank-0 batch. The control
and candidate both use early logits release, in-place CE, tuned cuBLASLt
and expandable segments. They differ only in whether the BF16 local
send shard has its own allocation or shares its slice of the
all-gather output. Each process and rank executed 24 BF16 all-gathers
and 23 FP32 SUM reduce-scatters over the run.

| Peak memory per rank | Separate BF16 shard, GiB | In-place BF16 shard, GiB | Saved, MiB |
| --- | ---: | ---: | ---: |
| PyTorch allocated | 6.387 | 5.898 | 501 |
| PyTorch reserved | 6.428 | 5.920 | 520 |
| GIFT total GPU | 8.016 | 7.509 | 520 |

Both ranks repeated the same peaks in both pairs. Wall time is the median
of the slower rank for each invocation. It moved 85.306 → 87.048 ms and
86.009 → 85.956 ms in the two pairs, which showed no consistent time change.
Full output digests and local FP32 dW shard digests matched on each rank.
A separate primitive check with changed BF16 contents verified the exact
rank-local NCCL alias in synchronous and asynchronous modes.
[Two-rank head receipts](artifacts/frontier-dual-001/summary.json)
[Two-rank NCCL primitive](artifacts/frontier-collective-alias-001)

For dense FSDP8, the plain BF16 shard removed by this allocation change
would be 125.25 MiB per rank before allocator effects. The larger
singleton and two-rank memory savings scale with their shard sizes.
Dense FSDP8 timing, shared-buffer invalidation across modules, YCCL and
optimizer integration remain unverified.

### Remaining live buffers and lower-memory designs

The memory diagnostic checked the active allocation addresses at the
16-chunk peak. Three FP32 buffers of 2004 MiB each held the rank-local master
parameter, the rank-local accumulated gradient and the unsharded
reduce-scatter input. The BF16 all-gather input and output held 1002 MiB each.
These five allocations accounted for about 91% of the 8787.098 MiB live
peak in the singleton replay. The BF16 logits tile was 250.5 MiB.
[Address-attributed snapshot](artifacts/frontier-memory-roles-004/stack-16.memory.json)

A single FP32 slot for local accumulated dW and unsharded dW would remove
one 2004 MiB buffer in the singleton experiment. Direct beta-one
accumulation across invocations changes the order of the FP32 additions.
With identical captured BF16 operands, the 16-chunk, 32-accumulation
FP64-relative L2 error increased from 2.178e−6 to 2.345e−6 (+7.69%).
This excludes the simple one-slot schedule under the precision requirement.
The diagnostic keeps the original per-invocation sum in one arm and varies
the individual dW contributions in a second, signed-scale case.
[One-slot precision receipts](artifacts/frontier-accumulation-001/precision.json)

An explicit reusable BF16 projection output also left the total GPU
peak unchanged at 10.280 GiB for both 8 and 16 chunks. This single-pair
screen used the default-allocator stack in both arms and verified
workspace reuse and bitwise equality of the full output digests.
[Projection workspace receipts](../../artifacts/frontier-workspace/initial)

Two designs could change the memory limit more substantially. A
vocabulary-sharded head, as used in Megatron, keeps logits and dW local
to each vocabulary partition. It needs a new TP topology, distributed
centered CE statistics and a higher-precision dHidden reduction before
the existing BF16 boundary. A vocabulary-panel scratch could keep the
current DP topology and replace the 2004 MiB FP32 reduction input with
smaller panels. Releasing a panel requires it to contain all outer-chunk
contributions, so a panel-major schedule must retain or recompute dLogits,
or trade one full reduction for more frequent panel reductions. A
quarter-sized panel is 501 MiB before its supporting buffers. The CCE
paper's rematerialization illustrates the extra projection work. The local
unfiltered CCE replay was too slow to use its implementation directly.
These designs have no local performance claim.
[Megatron vocab-parallel CE](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.20.0/megatron/core/tensor_parallel/cross_entropy.py)
[Cut Cross-Entropy paper](https://arxiv.org/abs/2411.09009)
[Megatron's gradient-buffer lifecycle](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/generalized_tensor_parallel.html#buffer-memory-management)

### Device trace

Separate Nsight Systems traces captured one post-warmup invocation per
arm at 2 and 16 chunks with the default allocator. The NVTX marker
includes synchronous host-wall time through the device synchronization.
GPU activity is the union of kernel, copy and memset intervals across
streams. dW is the summed kernel duration of the cuBLAS dW calls.
The remaining portion of the marker contains no recorded GPU activity.
These metrics overlap. Pure host dispatch time is unavailable. No NCCL
kernel appears inside these singleton markers.

| Fixed chunks | Current → candidate marker, ms | GPU activity, ms | Outside GPU activity, ms | Host device-sync API, ms | Device copies, ms | dW GEMMs, ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 73.97 → 71.66 | 72.18 → 70.19 | 1.79 → 1.47 | 66.99 → 64.80 | 0.689 → 0.692 | 22.17 → 21.21 |
| 16 | 91.27 → 85.97 | 87.67 → 83.56 | 3.60 → 2.41 | 57.14 → 61.92 | 0.688 → 0.690 | 37.08 → 32.98 |

At 16 chunks the trace changes the dW kernel from the 320×128, 3-stage
NVJET family to the 128×160, 5-stage family. Projection and dHidden
kernel totals each changed by less than 0.06 ms in that trace. The
profiled replay receipt's wall timer also covers profiler stop overhead.
Use the repeated unprofiled comparison for effect size. The 8- and
16-chunk wall changes also keep their direction when trimming the first
one to four timed samples from every process. [Nsight reports and database exports](artifacts/frontier-stack-profile-001)
[Capture boundary](frontier_profile.py)

### Integration boundary

The memory result makes early release plus in-place Stable CE the first
candidate for production integration. The prototype compiles the Stable
CE forward statistics inside an eager first-order autograd shell. The
current arm compiles the native CE route. In both arms the outer loop,
projection and GFSDP callbacks run eagerly. Matching captured-graph counts
establish execution with Inductor enabled at those boundaries. A later
integration must model logits mutation and aliases accurately for the
compiler, ensure backward has exclusive consumption of the logits
(including saved-tensor hooks), run any MTP argmax reader before mutation,
and preserve validation and the PP metadata probe path.
[Autograd storage shell](ce_storage.py) [Lifetime prototype](frontier_lifetime.py)

The four-count matrices use a world-size-one GFSDP mesh with rank-local weights
and exercises the direct-write slot, final callback, singleton SUM and
steady gradient accumulation. The replay adapts the native two-chunk
eligibility check for equal-shaped chunks, and retains its exclusions for
shared weights, DTensor weights, YCCL and supported parallelized-forward
checks. The focused two-rank alias check covers the NCCL gradient path. Production
still needs to establish dense FSDP8 slot ownership and reuse across
streams, collectives and overlap with the decoder, TP/CP/PP
layout handling, and end-to-end peak memory and throughput. Optimizer
updates, accumulation boundaries, saved checkpoint state and resume also
need validation. [Direct-write lease prototype](direct_dw.py)
[Replay source](replay.py)

Reproduce the four rows on the recorded idle GPUs from the worktree root
using the [stack launcher](frontier_stack_lane.sh), [wrapper](frontier_stack.py)
and runtime and fixtures under [Scope and reproduction](#scope-and-reproduction).
Create a distinct series directory under `runs/lm-head-mtp/artifacts`,
then pass the physical GPU, loss chunk count, series name and arm order.
For example, the 8-chunk lane used `4 8 SERIES stack control control stack`.
The launcher uses per-lane Inductor and Triton caches. Run
`frontier_verify.py` with the completed series directory to validate all
four lanes and regenerate the summary. Preserve the pinned tuning receipts
for cuBLASLt 13.1.1. [Verifier](frontier_verify.py) [Tuning receipts](frontier-dw)

## Two-chunk package and writer comparison

| Matched replay | Wall ms | Time change | Total GPU GiB | Memory change, GiB |
| --- | ---: | ---: | ---: | ---: |
| Initial GIFT → current five-fix package | 83.46 → 77.04 | -7.69% | 16.151 → 14.194 | -1.957 |
| Current source with all five off → all five on | 83.30 → 77.04 | -7.51% | 16.151 → 14.194 | -1.957 |
| Current two-chunk staging → two-chunk cuBLAS | 77.04 → 75.63 | -1.83% | 14.194 → 14.194 | +0.000 |
| Initial GIFT → fastest, two-chunk cuBLAS | 83.46 → 75.63 | -9.38% | 16.151 → 14.194 | -1.957 |

These four comparisons share a single H100 with alternated candidates.
The staged-to-cuBLAS improvement in the two alternating pairs was 0.73 and
2.09 ms. All final paired runs used singleton SUM and recorded zero allocator
retries and OOMs. [Speed and package receipts](artifacts/finalists-003/lane-0)
[Earlier writer and memory receipts](artifacts/memory-pair-003)

## Five separate changes at two chunks

| Change | Wall ms, control → enabled | Time change | Total GPU GiB | Memory change, GiB |
| --- | ---: | ---: | ---: | ---: |
| Stable cap-zero CE, with folding off | 81.03 → 79.48 | -1.92% | 16.151 → 16.151 | +0.000 |
| Folded CE, with stable CE active | 79.48 → 76.68 | -3.53% | 16.151 → 14.194 | -1.957 |
| Direct BF16 dHidden | 77.95 → 76.67 | -1.64% | 14.194 → 14.194 | +0.000 |
| Staged LM-head dW | 79.95 → 76.67 | -4.10% | 15.173 → 14.194 | -0.979 |
| Fused hidden z-loss | 78.10 → 76.67 | -1.83% | 14.194 → 14.194 | +0.000 |

Folded CE gave the largest memory reduction among the five. Staged dW gave
the largest solo time reduction using the full ten-sample window. Folded CE
and staged dW exchanged rank when the first few samples were trimmed. All
five individual timings retained their direction when trimming up to four
samples. The direct hidden gradient and fused z-loss reduced wall time
with the same total peak. The first two rows share the unfolded Stable CE
configuration. The matched package comparison measures the combined effect.
[CE ablations](artifacts/ablation-ce-002) [Gradient and z-loss ablations](artifacts/ablation-grad-002)

## Weight-gradient writers

Each latency column covers the complete replay. Each pair used its own
same-GPU ordinary control. The TMA lane used the pinned Quack source;
its interpreter reported the same PyTorch/CUDA versions as the cuBLAS lane.
The additional staged arm at two chunks is in the final comparison above.

| Loss chunks | Folded rows/chunk | Ordinary → cuBLAS, ms | Ordinary → TMA, ms | Ordinary → direct total GiB | FP64 relative L2, ordinary → cuBLAS → TMA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16384 | 77.50 → 77.43 | 78.53 → 87.73 | 19.087 → 18.108 | 0.00167 → 2.46e-05 → 2.46e-05 |
| 2 | 8192 | 80.03 → 77.06 | 79.47 → 83.86 | 15.173 → 14.194 | 0.00135 → 1.29e-05 → 1.29e-05 |
| 4 | 4096 | 83.93 → 80.76 | 83.86 → 83.97 | 13.216 → 12.237 | 0.00113 → 6.97e-06 → 6.97e-06 |
| 8 | 2048 | 93.28 → 82.56 | 92.84 → 84.23 | 12.727 → 11.261 | 0.000984 → 3.82e-06 → 3.82e-06 |
| 16 | 1024 | 110.82 → 94.60 | 110.79 → 96.71 | 11.259 → 10.280 | 0.000906 → 2.16e-06 → 2.16e-06 |

The captured BF16 hidden/weight/target tensors were used to reconstruct
identical BF16 gradient GEMM operands for all writers at each chunk count.
The FP64 reference accumulates products of those same operands. Ordinary
and staged produced the same relative L2 at two chunks. The direct writers
were finite in every cell and eliminated per-chunk BF16 dW rounding.
[cuBLAS grid](artifacts/cublas-grid-002) [TMA grid](artifacts/tma-grid-002)
[FP64 and dW-only CUDA-event receipts](artifacts/dw-check-002)

| Loss chunks | Ordinary dW, ms | cuBLAS dW, ms | Staged dW, ms |
| ---: | ---: | ---: | ---: |
| 1 | 23.23 | 24.39 | — |
| 2 | 25.52 | 24.69 | 23.24 |
| 4 | 29.52 | 25.30 | — |
| 8 | 38.59 | 27.20 | — |
| 16 | 56.45 | 38.02 | — |

## Reusing logits storage and trying full Linear+CE

The storage arms shared an eager first-order autograd shell and compiled
Stable CE forward statistics. The shell selected a distinct dLogits buffer
or consumed the logits buffer after forward. Each pair held the writer
constant. Here the stated tile is one folded sequence chunk.

| Writer | Folded rows | Separate → reuse, ms | Total GPU GiB in both |
| --- | ---: | ---: | ---: |
| cublas | 8192 | 76.84 → 77.01 | 14.194 |
| cublas | 4096 | 78.94 → 78.61 | 12.237 |
| cublas | 2048 | 82.28 → 80.72 | 11.261 |
| tma | 8192 | 83.78 → 83.82 | 14.194 |
| tma | 4096 | 83.44 → 83.64 | 12.237 |
| tma | 2048 | 83.37 → 84.11 | 11.261 |

Reuse removes one CE-local BF16 allocation. This earlier comparison kept
the previous logits reference live during the next projection. The later
2×2 experiment measured both lifetime changes independently and together.
[Lifetime comparison](#logits-lifetime-and-arithmetic)

At 2, 4, and 8 chunks, Folded, separate, and reused dLogits matched
Stable CE elementwise in BF16. All ignored rows
had zero gradients.
[cuBLAS storage](artifacts/cublas-storage-002) [TMA storage](artifacts/tma-storage-002)
[FP64 CE and ignore-mask checks at all measured tile sizes](artifacts/ce-storage-check-004/result.json)

| Alternative | Same-GPU current, ms/GiB | Candidate, ms/GiB | dHidden relative L2 vs FP32 | dW relative L2 vs FP32 |
| --- | ---: | ---: | ---: | ---: |
| Quack Scaled Exp Q4096 | 77.47 / 14.194 | 77.33 / 15.110 | 0.00286 | 0.002509 |
| Quack Scaled Exp Q8192 | 77.47 / 14.194 | 77.24 / 16.089 | 0.00286 | 0.00251 |
| Liger SM90 with FP32 dW accumulation | 77.47 / 14.194 | 122.63 / 13.153 | 0.00178 | 0.002103 |
| Historical FLA-style 1,024-row kernel | 76.84 / 14.194 | 97.05 / 13.153 | 0.00178 | 0.001846 |
| Apple CCE, no gradient filtering | 76.84 / 14.194 | 408.63 / 13.153 | 0.00178 | 0.002112 |

Quack latency was within the repeated native variation. Both Quack whole-op
arms used more total memory and had higher gradient errors in this replay.
The other external kernels reduced memory versus their local current
controls but were slower. Quack exposes a compiled custom-op boundary.
Liger and Apple CCE captured multiple graphs through their Python wrappers.
Liger used its enabled host-side target validation.
The prototype retains the native loss loop and real singleton GFSDP hooks
and reports each candidate's actual outer chunk setting. The Quack, Liger
and CCE arms use one outer chunk. FLA and the local current controls use two.
The third-party projection-fused APIs expose loss, dHidden and dW for
comparison. The dLogits comparison above covers the local CE routes.

Against streamed FP32 projection/CE using the same source tensor values,
the current two-chunk replay measured dHidden relative L2
0.00178 and dW relative L2
0.001851. The fastest cuBLAS arm
measured 0.00178 and
0.001272, respectively. The sixteen-chunk
cuBLAS arm measured 0.00178 and
0.001265. Both two-chunk cuBLAS gradient dumps matched the current loss,
dHidden, z-loss, and z-loss gradient bitwise. The original and current
z-loss and z-loss gradient also matched bitwise on the captured final-norm
input. The shared z-loss gradient's relative L2 against a CPU FP64 reference
was 0.001672. The whole-component FP32 oracle used a freshly reset gradient
for comparison, while timing used the steady accumulation path.
[Vendor timing](artifacts/vendor-runs-002)
[Vendor numerics](artifacts/vendor-numerics-003/result.json)
[Final numerics](artifacts/final-numerics-003/result.json)

## Scope and reproduction

- Host `sr008-gpu`, worktree branch `lm-head-mtp-fix`, eight H100 80 GB HBM3
  cards used as independent lanes. CUDA 13.0, PyTorch
  `2.15.0.dev20260822+cu130`, Triton 3.8.0, driver 570.133.20.
- Initial GIFT is root commit `e3871b36c0be1cb5c027b811d1684558fb11b02b`.
  Current GIFT is `2c67c182e8de6145f3796d62203714ab1b815446`. Both run the
  native Inductor-compiled CE boundary, with native eager GFSDP projection
  and callbacks. The current-source all-five-off comparison holds the
  intervening source changes fixed. Current launch receipts identify the
  active CE route and writer and preserve every individual sample.
- Real production-tokenizer capture, main and one MTP head with weights
  1.0 and 0.3, shared denominator, original packed labels and boundaries,
  B2 S4096, D4096, V128256, bias-free BF16 projection. Head capture SHA-256
  `15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727`.
  Independent captured final-norm input for hidden z-loss SHA-256
  `dced479c2bea4ff5cc5ade9dad04531d075d74f4bbc5c79e3f069c049d8863a1`.
  Its forward retains state across head backward and its VJP is timed
  with the replay. This isolates the head and losses from decoder layers
  and optimizer work.
- Each timed arm starts a fresh process and warms compilation. The timed
  window retains a primed sharded gradient across ten invocations.
  Time is median synchronized host-wall milliseconds per invocation.
  Repeated arms are summarized with the median of their medians and the
  maximum of their total-memory peaks. Percentage changes use the same
  GPU's control as denominator. Whole-component timings include native
  label packing, loss accounting and the singleton GFSDP gradient path.
- Memory is the maximum native GIFT total estimate over the ten timed
  invocations, including weight, FP32 state and the external CUDA runtime.
  It adds allocator peak reserved bytes to device-wide non-PyTorch use
  at report time. The device is otherwise idle. Brief external peaks
  between samples can escape this estimate.
- The pinned third-party source trees were read through the task's Python
  paths. Quack `bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8` uses the
  preserved scale-axis patch. Liger
  `85a9c40f1f5f9bb39cf74e70ccbc3d416479ee42` and Apple CCE
  `3de376c106a1916bc5e1b619f9c77c87a461ee1c` use their pinned source archives.
- Runner [replay.py](replay.py), [lane.sh](lane.sh), writer
  [direct_dw.py](direct_dw.py), storage [ce_storage.py](ce_storage.py),
  vendor [vendor_ce.py](vendor_ce.py), and numerical checks
  [dw_check.py](dw_check.py) and [numerics.py](numerics.py).
  For the final native series, source `/etc/profile.d/wm.sh`, set
  `CUDA_VISIBLE_DEVICES` to one idle H100 UUID, and invoke
  `/opt/torchtitan-venv/bin/python3.12 -Bu runs/lm-head-mtp/replay.py current RESULT.json --sum-reduction --chunks 2 --writer cublas`
  from the worktree root. Use `original` and `--writer native` for the
  initial control. Provide distinct output names and a task-local cache.
- The first smoke reached timing, then its GIFT RaggedShardDTensor check
  attempted `to_local()` in grad mode. The receipt check was moved to
  `no_grad` and the matched measurements were repeated. The diagnostic
  failure remains in [smoke-001](artifacts/smoke-001).
