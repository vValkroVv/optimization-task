# Eight-rank isolated LM-head gather comparison

## Results

The in-place gather lowered peak GIFT total memory in every matched block. It saved 140 MiB per rank at C2 and 120 MiB at C8. The latency differences changed direction across repeated runs at both chunk counts.

| Outer chunks | Original compiled baseline, ms / GiB | Exact CE, regular gather, ms / GiB | Same exact CE, in-place gather, ms / GiB |
|---:|---:|---:|---:|
| 2 | 92.096 / 14.016 | 83.390 / 7.708 | 83.442 / 7.571 |
| 8 | 105.242 / 9.614 | 86.839 / 6.165 | 86.815 / 6.048 |

| Chunks | Baseline run medians, min–max ms | Regular run medians, min–max ms | In-place run medians, min–max ms | Paired in-place minus regular, median [min, max] ms |
|---:|---:|---:|---:|---:|
| 2 | 91.196–93.196 | 83.179–83.841 | 82.893–83.829 | −0.093 [−0.439, +0.651] |
| 8 | 104.752–105.489 | 86.445–87.247 | 86.557–87.052 | +0.051 [−0.547, +0.600] |

Positive paired changes indicate slower in-place runs. In-place ran faster in four of six C2 blocks and three of six C8 blocks. Paired differences use matched blocks; subtraction of the displayed arm medians is a separate statistic. Relative to the complete baseline, the in-place candidate's median latency fell 9.40% at C2 and 17.51% at C8. Peak total fell 6.445 and 3.566 GiB, respectively.

## What ran and what matched

All 36 groups finished. Each of the 288 rank receipts verified the native BF16 all-gather and FP32 SUM reduce-scatter on all eight ranks. Each process performed 24 gathers and 23 reduce-scatters including setup and the reset-output check. Regular gather allocated one 125.25 MiB local send shard and reused its BF16 cast in this fixed-weight replay. In-place used the rank's exact slice of the full receive allocation and cast on all 24 gathers. Actual GIFT total savings track allocator reservation and external usage, so they can differ from the payload size. All six group peak values were identical within each matrix cell; no timed allocator retries or OOMs occurred.

At each chunk count all 48 corresponding block/rank comparisons between regular and in-place matched bitwise for weighted loss, BF16 dHidden, FP32 dW shard, z-loss and BF16 z-gradient. Against the original compiled baseline, the optimized computation changes dHidden and dW digests at both chunk counts and the weighted loss digest at C2. These digest comparisons establish equality or difference, not the size of numerical differences or training convergence.

[Verified matrix and per-group statistics](matrix-summary.json) · [Eight-rank raw receipts and launch order](matrix-01/) · [Pilot receipts](pilot-01/)

## Setup

All eight H100 80 GB GPUs on the same SR008 NVLink node form one real GFSDP shard group. Each GPU holds a distinct weight shard and receives its own FP32 dW shard through NCCL SUM reduce-scatter. Every rank replays the same captured rank-zero main/MTP batch and captured z-loss input used in the earlier isolated head study. The shape is B2/S4096, hidden width 4096, vocabulary 128256, main + one MTP head weighted 1.0/0.3. C2 and C8 count outer sequence chunks. This is a component measurement with real communication; decoder and optimizer costs are outside the timer.

The three arms use the original compiled GIFT baseline (e3871b36, default allocator), or the current lm10-c1 candidate (working source hashes in the receipts, expandable segments) with exact ordered fused CE cuda-13. Both candidate arms use early logits release, in-place BF16 CE gradients, BF16 dHidden, the same direct FP32 dW plans, and the same fused z-loss. Their only functional difference is how GFSDP stages the local BF16 weight shard. The regular gather can reuse a valid BF16 send buffer. The in-place gather recasts the local FP32 shard into the rank's receive slice before each gather. On eight ranks that slice is 131,334,144 bytes (125.25 MiB) per rank. The baseline comparison includes the full earlier head optimization package and the allocator change.

## Measurement

Each chunk count uses all six permutations of the three arms. There are six fresh eight-rank process groups per cell. Each group performs ten warmup calls, primes the retained sharded gradient, and measures ten complete isolated microbatches. An eight-rank start barrier precedes every timed call. Timing includes head forward and backward, both losses, input packing, collectives and final CUDA synchronization. For each aligned iteration we take the slowest rank, take the median of its ten group timings, then report the median over the six groups. Peak memory is the highest native GIFT total observed over all eight ranks and all 60 timed calls in that cell. Each observation adds peak allocator reservation to sampled external GPU use. These totals describe one GPU at the peak.

Compact raw measurements include per-rank samples, device and compute identities, actual NCCL group sizes, storage addresses, cast counts and reset-output digests. The separate C8 2+2 pilot checked both collectives and the timing/verification path before launching the matrix.

[Launch and order](campaign8.py) · [Eight-rank replay](replay8.py) · [All-gather storage and cast checks](gather8.py) · [Verification and aggregation](summarize8.py)
