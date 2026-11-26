# Isolated cross-entropy and LM-head comparison on eight H100s

25 November 2026. Each table entry reports synchronized wall time in milliseconds and peak GIFT total memory in GiB per GPU. Results apply to one replayed LM-head microbatch. Training throughput and peak memory of the decoder lie outside both timers.

## Inputs and measurement boundaries

| Preset | Local batch | Sequence | Hidden | Vocabulary | Loss chunks | Folded rows per chunk | Main / MTP1 weights |
|---|---:|---:|---:|---:|---:|---:|---|
| 100B | 2 | 4,096 | 4,096 | 128,256 | 2 | 8,192 | 1 / 0.3 |
| 100B | 2 | 4,096 | 4,096 | 128,256 | 8 | 2,048 | 1 / 0.3 |
| 700B (3.5, linear output head) | 3 | 4,096 | 7,168 | 128,256 | 8 | 3,072 | 1 / 0.1 |

The three cases use seeded synthetic BF16 hidden states and weights, plus real packed token IDs and derived targets from `data-002`. Each rank replays the same local fixture. The denominator is the number of valid main tokens times the eight ranks and 32 accumulation steps. This preserves the normalization used by the isolated replay. Only one microbatch is timed. The checks compare these tensors and do not evaluate convergence on a training trajectory. [Fixture](matrix_fixture.py)

- **CE on an existing logits buffer.** One H100 measures the final chunk. Inputs are prepared before the timer. Both raw FP32 head sums, the normalized weighted loss and the contiguous weighted BF16 gradient must be ready at the end. The original input buffer stays live for restoration before each sample. We report the median of 32 timed repetitions after 10 warmups, including host dispatch and synchronization. Separate graph-replay and CUDA event spans are in the raw results. [CE measurement](ce_bench.py) · [CE results](ce-summary.json)
- **Whole isolated head.** Eight H100s execute one native GFSDP group, one ordinary BF16 weight all-gather, chunk projections and CE, dHidden, dW, and one FP32 SUM reduce-scatter with gradient divisor 1. The 100B preset also executes the hidden z-loss on its separate pre-norm input. Each arm starts in a fresh set of processes. The primary measurement takes 10 samples after warmup. Each sample uses the longest synchronized elapsed time among eight ranks. The reported time is the median of those maxima. The total memory is the maximum over ranks and samples. [Head measurement](head_bench.py) · [Head launcher](head_matrix.py)
- **GIFT total.** The allocator uses `expandable_segments:True`. At readout GIFT adds peak PyTorch reserved bytes to the current device usage outside PyTorch's reserve. NCCL and other device allocations therefore count. The CE and full-head measurements have different live inputs, and their absolute peaks must be interpreted separately. [Memory definition](../../../planning/validation.md)
- **Precision.** The CE oracle evaluates all scored entries on the measured chunk. The full-head oracle takes the same BF16 projection as its starting point and computes losses, dHidden, and dW in FP64. It uses the matching rank slice of the SUM-reduced dW. Both oracles run outside the timed region. “No regression” in the head comparison means that both maximum absolute error and relative L2 error are no greater than the selected control on every rank for every reported component. Two different gradient tensors can have the same error metrics. [Head oracle](head_oracle.py)

## Implementations and inner chunking

The default GIFT comparisons use one folded projection per external chunk, combining main and MTP1. The generic kernel receives the actual row count, vocabulary and number of heads. cuBLASLt selects and caches a plan for the actual dimensions and alignment, with direct FP32 output in the GFSDP slot. The first chunk overwrites that slot and subsequent chunks add to it. The original fixed plans are retained as 100B controls. The compiled CE artifact used for this replay has SHA-256 `17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8`. Its fast path targets Hopper, contiguous BF16 logits, and vocabularies up to 131,072. A separate GPU check covers main plus ten MTP heads on a smaller odd vocabulary. The performance matrix uses main plus MTP1. [CUDA checks](checks/native-ce/smoke.log) · [Writer checks](checks/writer-check/result.json)

| Projection used by a Linear+CE implementation | 100B C2 | 100B C8 | 700B C8 |
|---|---:|---:|---:|
| GIFT or Liger Linear with aligned inner chunking | 1 × 8,192 | 1 × 2,048 | 1 × 3,072 |
| Liger Linear with its default inner chunking | 32 × 256 | 32 × 64 | 12 × 256 |
| FLA Linear with one inner chunk per loss head | 2 × 4,096 | 2 × 1,024 | 2 × 1,536 |
| FLA Linear with its default eight inner chunks per head | 16 × 512 | 16 × 128 | 12 × 256 |

Counts describe projection rows per external chunk in a forward pass. Liger uses a rowwise output for the two loss sums and recomputes its logits in backward. FLA's stock Linear implementation returns one scalar per call, so main and MTP1 run separately. The adapted FLA Linear kernels on the GIFT head again receive one folded projection and use the incoming head scales before storing dLogits. Apple processes the vocab in tiles and does not form a full external chunk of logits. Apple runs with gradient filtering disabled. The separate Apple adaptation supplies the GFSDP FP32 slot to its weight-gradient kernel.

The baseline GIFT checkout is `e9b7cdbb`. The historical Folded package is based on `2c67c182`. The generalized implementation is a working-tree change on `3783bf33e`. Library snapshots are pinned. [Liger](https://github.com/linkedin/Liger-Kernel/tree/42dc23c41dcaf1f95d858837aed5301fd4faa27a) · [FLA](https://github.com/fla-org/flash-linear-attention/tree/c6e4f00d92fa2e3ed7232159d7be6fbbcff5c69f) · [Apple CCE](https://github.com/apple-aiml-research/ml-cross-entropy/tree/3de376c106a1916bc5e1b619f9c77c87a461ee1c)

## CE on the last logits chunk

The last column lists cases passing the strict Folded CE accuracy check. “All” covers the three columns. Masked negative zero is recorded separately in the baseline receipts and does not represent an absolute gradient error.

| Route | 100B C2, ms / GiB | 100B C8, ms / GiB | 700B C8, ms / GiB | Strict check |
|---|---:|---:|---:|---|
| Baseline A + compile | 4.382 / 8.561 | 1.243 / 2.682 | 1.753 / 3.679 | None |
| Previous Folded CE | 2.406 / 6.589 | 0.911 / 2.194 | 1.167 / 2.936 | All |
| Generalized cuda-13 | 1.761 / 4.636 | 0.563 / 1.725 | 0.761 / 2.194 | All |
| Liger Triton | 6.194 / 4.636 | 1.644 / 1.725 | 2.404 / 2.194 | None |
| Liger Triton with FP32 scratch | 11.543 / 8.542 | 2.956 / 2.702 | 4.387 / 3.679 | All |
| Liger Triton with early FP32 scale | 2.040 / 4.636 | 0.597 / 1.725 | 0.843 / 2.194 | 100B C2 |
| Liger CuTe | 6.512 / 4.636 | 1.763 / 1.725 | 2.552 / 2.194 | None |
| Liger CuTe with FP32 scratch | 11.583 / 8.542 | 2.965 / 2.702 | 4.396 / 3.679 | 100B C8, 700B C8 |
| FLA standalone | 2.452 / 4.636 | 0.821 / 1.725 | 1.028 / 2.194 | None |
| FLA centered, FP32 scratch | 14.633 / 10.495 | 3.732 / 3.173 | 5.551 / 4.421 | None |

The new kernel matched Folded CE bit for bit for both FP32 head sums, weighted loss and BF16 dLogits on the three sampled chunks. The Liger early-scale adaptation avoided the FP32 logits-sized scratch. On the two C8 chunks its MTP relative-L2 error exceeded Folded by at most 7.16e-17. The error criterion uses the literal inequalities recorded in the receipts.

## Whole head, eight H100s

Each cell is synchronized ms / peak total GiB per GPU. The last two columns count shapes passing the literal full-head FP64 rule relative to each control. Per-rank, per-component errors, individual steps and comparison deltas are recorded in the [head summary](head-summary.json).

| Route | 100B C2 | 100B C8 | 700B C8 | No worse than final | No worse than baseline |
|---|---:|---:|---:|---:|---:|
| Baseline A + compile | 91.60 / 11.946 | 104.54 / 8.548 | 252.27 / 13.157 | 0/3 | 3/3 |
| Baseline A + compile with FLA standalone CE and ordinary baseline dW | 91.04 / 11.028 | 102.86 / 8.059 | 249.94 / 12.454 | 0/3 | 0/3 |
| Previous Folded package | 86.21 / 9.681 | 101.43 / 7.630 | 247.39 / 12.122 | 0/3 | 0/3 |
| Folded CE + generic cuBLASLt | 86.06 / 7.727 | 91.30 / 6.165 | 227.06 / 9.661 | 3/3 | 2/3 |
| cuda-13 + generic cuBLASLt | 84.17 / 7.727 | 89.61 / 6.165 | 224.33 / 9.661 | 3/3 | 2/3 |
| cuda-13 + fixed 100B cuBLASLt plan | 83.80 / 7.727 | 87.12 / 6.165 | — | 2/2 | 2/2 |
| Liger Triton on GIFT + Lt | 93.94 / 7.727 | 97.43 / 6.165 | 236.37 / 9.661 | 0/3 | 2/3 |
| Liger early FP32 scale on GIFT + Lt | 86.07 / 7.727 | 90.03 / 6.165 | 224.21 / 9.661 | 0/3 | 1/3 |
| Liger Triton FP32 scratch on GIFT + Lt | 102.27 / 11.634 | 106.26 / 7.141 | 249.70 / 11.106 | 0/3 | 1/3 |
| Liger CuTe on GIFT + Lt | 94.38 / 7.727 | 96.98 / 6.165 | 236.44 / 9.661 | 0/3 | 1/3 |
| Liger CuTe FP32 scratch on GIFT + Lt | 102.94 / 11.634 | 106.68 / 7.141 | 250.01 / 11.106 | 0/3 | 1/3 |
| FLA standalone on GIFT + Lt | 86.55 / 7.727 | 90.52 / 6.165 | 225.71 / 9.661 | 0/3 | 0/3 |
| FLA Linear CE kernels on GIFT + Lt | 85.43 / 7.727 | 90.41 / 6.165 | 227.56 / 9.661 | 0/3 | 0/3 |
| Liger Fused Linear CE, aligned | 122.53 / 11.634 | 148.08 / 9.095 | 348.12 / 14.075 | 0/3 | 0/3 |
| Liger Fused Linear CE, default | 241.99 / 8.821 | 885.90 / 8.665 | 645.76 / 14.231 | 0/3 | 0/3 |
| FLA Fused Linear CE, aligned per head | 410.58 / 11.610 | 465.61 / 9.774 | 1119.21 / 16.141 | 0/3 | 0/3 |
| FLA Fused Linear CE, default | 442.68 / 10.048 | 706.75 / 9.620 | 1337.99 / 15.927 | 0/3 | 0/3 |
| Apple CCE exact | 421.11 / 8.641 | 447.02 / 8.544 | 1139.27 / 14.013 | 0/3 | 1/3 |
| Apple CCE exact with direct FP32 dW | 418.24 / 5.790 | 418.18 / 5.614 | 1094.41 / 8.915 | 0/3 | 1/3 |

The only strict comparison against Baseline A that the generalized cuda-13 path misses is 700B dHidden relative L2. It increases from 0.00224366798282 to 0.00224366826121. Its maximum absolute error is unchanged. The owner accepted this 2.78e-10 increase for the comparison. The 700B dW worst-shard relative L2 decreases from 0.00240615 to 0.00173393. The raw pass/fail columns keep the original threshold.

The follow-up FLA standalone run on the baseline checkout changed only the compiled per-head CE callable. It used the published `FusedCrossEntropyLoss` on logits with sum reduction and in-place backward. Baseline A issued one CE call per head, kept its ordinary projection backward, and performed one BF16 gather and one FP32 SUM reduce-scatter per rank. All three cells used 10 warmups and 10 timed repetitions on eight ranks. Compared with the original Baseline A series, the medians decreased by 0.56, 1.68 and 2.33 ms, and the measured peak fell by 0.918, 0.488 and 0.703 GiB. No gradient maximum absolute error increased on any rank. Some rank-level relative L2 errors increased, by up to 1.27e-8 for dW. [Follow-up measurements and FP64 comparisons](head-baseline-fla/summary.json)

| Generalized cuda-13 relative to Baseline A | Wall time difference | Wall time reduction | Peak total saved |
|---|---:|---:|---:|
| 100B-C2 | -7.43 ms | 8.1% | 4.219 GiB |
| 100B-C8 | -14.93 ms | 14.3% | 2.383 GiB |
| 700B-C8 | -27.94 ms | 11.1% | 3.496 GiB |

FLA Linear with the configured FP32 gradient accumulator still uses private weight-gradient storage, produces a BF16 gradient at the weight boundary and computes separate head losses. The GIFT adaptation changes projection grouping, where scaling is applied, and the dW writer together. The difference between those two timings includes all of these changes.

Apple direct has the lowest primary memory peak on all three geometries. Relative to cuda-13 it saves 1.938, 0.551 and 0.746 GiB respectively. It takes 4.97×, 4.67× and 4.88× as long. It also misses the strict full-head check against cuda-13. The rank-level error fields specify which component differs.

## Direct FP32 dW on the original compiled baseline

This follow-up reran both Baseline A and Baseline A with the cuBLASLt writer on the same eight H100s. The isolated overlay on `e9b7cdbb` retains its original compiled CE, logits lifetime, FP32 hidden-gradient buffer, z-loss and ordinary gather. It connects the writer to the baseline head callback and uses the same 100B fixed plans and 700B generic selection as the final comparison. There were 10 warmups and 20 timed samples per cell. The deltas use the controls from this follow-up. [Overlay](checks/baseline-dw-overlay.patch) · [Per-rank measurements and FP64 checks](head-dw-only/summary.json)

| Shape | Baseline A, ms / total GiB | Baseline A + direct dW, ms / total GiB | Time saved | Peak total saved |
|---|---:|---:|---:|---:|
| 100B C2 | 93.16 / 11.946 | 90.61 / 11.927 | 2.55 ms (2.7%) | 0.020 GiB |
| 100B C8 | 106.26 / 8.548 | 93.12 / 7.532 | 13.14 ms (12.4%) | 1.016 GiB |
| 700B C8 | 253.72 / 13.157 | 233.81 / 11.419 | 19.91 ms (7.8%) | 1.738 GiB |

Loss, per-head components and the complete BF16 dHidden digest matched across both arms on every rank. All shapes retained one gather and one FP32 SUM reduce-scatter. The maximum relative L2 error over FP32 dW shards fell from 0.001833 to 0.000726 at 100B C2, from 0.001819 to 0.000724 at 100B C8, and from 0.002406 to 0.001734 at 700B C8. The C2 peak changed by just 20 MiB. This run measured total peaks and did not capture allocation timelines.

## Repeated finalists

The primary series had 10 measured repetitions. Each following series used fresh processes, 10 warmups and 20 measured repetitions. The cells give median ms / peak GiB from each series in chronological order.

| Route | 100B C2 | 100B C8 | 700B C8 |
|---|---|---|---|
| cuda13 | 84.17 / 7.727 · 84.45 / 7.727 · 85.35 / 7.727 | 89.61 / 6.165 · 90.30 / 6.165 · 90.34 / 6.165 | 224.33 / 9.661 · 224.17 / 9.661 · 225.77 / 9.661 |
| cuda13-tuned | 83.80 / 7.727 · 84.05 / 7.727 | 87.12 / 6.165 · 88.09 / 6.165 | — |
| folded-lt | 86.06 / 7.727 · 86.09 / 7.727 | 91.30 / 6.165 · 90.55 / 6.165 | 227.06 / 9.661 · 226.78 / 9.661 |
| standalone-fla | 86.55 / 7.727 · 86.11 / 7.727 | 90.52 / 6.165 · 90.60 / 6.165 | 225.71 / 9.661 · 225.94 / 9.661 |

For CE alone, the three low-memory finalists were also rotated on the same card for each shape, with 10 warmups and 32 measurements per route. Every timed invocation tracked synchronized host wall time and GIFT total.

| CE route, interleaved | 100B C2, ms / GiB | 100B C8, ms / GiB | 700B C8, ms / GiB |
|---|---:|---:|---:|
| cuda13 | 1.783 / 4.638 | 0.584 / 1.727 | 0.783 / 2.196 |
| liger-early | 2.049 / 4.638 | 0.608 / 1.727 | 0.849 / 2.196 |
| fla | 2.448 / 4.638 | 0.802 / 1.727 | 1.010 / 2.196 |

## Generalization and verification

An additional native, eight-rank compiled replay exercised main plus ten MTP heads, unequal weights including zero, two different microbatches with distinct rank inputs, and three unequal chunks of 154, 154 and 110 folded rows. Every rank executed the consuming CE, one ordinary gather and one FP32 SUM reduce-scatter per microbatch. The first writer call overwrote a deliberately NaN-filled slot and the second microbatch accumulated into the rank shard. Against the eager oracle, maximum absolute errors were 6.08e-8 for loss, 2.38e-7 for BF16 dHidden and 7.54e-9 for the accumulated FP32 dW shard. The shape-general result concerns this isolated head. The separate full-recipe MTP transition prefetch still has a single-MTP-layer guard. [Test](multi_mtp_smoke.py) · [Passing run](checks/multi-mtp/multi-mtp-smoke-event.log) · [Unit results](checks/output/unit-tests.log)

One earlier harness used the default GFSDP gradient divisor and exposed a factor-of-eight discrepancy against the SUM oracle. The primary and repeated results above all assert divisor 1 and collect data from the corrected run. The initial small multi-MTP smoke also exposed that the isolated caller could read a shard before completion of the reduction. It now waits on the native post-reduce event. Full-head timed repetitions synchronize the device before stopping the clock. All numerical comparisons run outside that timer. [Recorded measurement harness](checks/head-measured.py) · [Summarizer](head_results.py)

The generic path is a reasonable common implementation across these geometries. At fixed memory, the old shape-specific 100B C8 plan measured 87.12 and 88.09 ms, ahead of the generic selection's 89.61–90.34 ms. Same-card interleaving reproduced the standalone CE ordering on all three shapes. None of the tested full-head alternatives delivers another 5% time reduction at the current total-memory budget and matched precision.

The subsequent descriptor search and CUDA CE kernel comparison includes repeated full-head measurements against these controls. [Hopper CE and cuBLASLt tuning](tuning/report.md)
