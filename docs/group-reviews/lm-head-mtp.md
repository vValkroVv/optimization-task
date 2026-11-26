# LM-head and MTP loss group review

Updated 26 November 2026. The current production recipe enables the five
changes described below. The final fused CE acceptance is recorded separately
in this review. The historical measurements used an eight-layer crop on eight
H100s with local B2/S4096, hidden width 4096, vocabulary 128256, MTP1 and two
sequence chunks. In this setup the main loss and one MTP loss share one LM-head
projection. They occupy separate rows of the folded batch. Each sequence
chunk includes rows for both loss heads. [Production recipe](../baseline.md#current-final-task-baseline)

CE backward produces `dLogits` for each chunk. The LM-head backward uses it
in two GEMMs, `dHidden = dLogits @ weight` and
`dW = dLogits.T @ hidden`, with batch and sequence flattened for the GEMMs.
`dHidden` goes back to the decoder and MTP branches in disjoint sequence
slices. Every chunk contributes to the same `dW` matrix, so GFSDP sums those
contributions before reduction. [Chunked loss and decoder bridge](../../gift/torchtitan/components/loss.py#L675)

## Six-rank sequence-length grid, 26 November

The full isolated head ran on physical GPUs 0–5 of `sr008-gpu1`, with local
batch 1 and main plus MTP1. Each comparison kept sequence length and chunk
count fixed. The timed head includes BF16 weight all-gather and FP32 gradient
reduce-scatter. The candidate uses the final shape-specific CE and GEMM plans,
including the corrected ATen projection route for 100B C8 at 128k. Each
configuration ran twice per variant in opposite orders, with 10 warmups and
10 timed calls per process. Times are the median of the two run medians, using
the slowest of six ranks per call. Peak memory is the maximum per-GPU GIFT
total across both runs and all timed calls; GIFT adds peak PyTorch reserved
memory to current non-PyTorch usage. Throughput gain is
`100 × (baseline time / candidate time − 1)`. OOMs occurred in both runs at
the initial head call before warmup. There were 76 successful runs and eight
CUDA OOMs.

### 100B · C2

| Sequence | Baseline A + compile, ms / GiB | Our CE + cuBLASLt, ms / GiB | Throughput gain | Peak memory change |
|---:|---:|---:|---:|---:|
| 4k | 51.31 / 8.819 | 46.00 / 6.651 | +11.6% | -24.6% (-2.168 GiB) |
| 8k | 91.89 / 12.140 | 82.90 / 7.901 | +10.8% | -34.9% (-4.238 GiB) |
| 16k | 175.02 / 18.800 | 162.73 / 10.343 | +7.6% | -45.0% (-8.457 GiB) |
| 32k | 342.41 / 32.688 | 319.59 / 15.247 | +7.1% | -53.4% (-17.441 GiB) |
| 64k | 694.30 / 59.917 | 634.41 / 25.093 | +9.4% | -58.1% (-34.824 GiB) |
| 128k | OOM | 1223.43 / 44.763 | — | — |
| 256k | OOM | OOM | — | — |

### 100B · C8

| Sequence | Baseline A + compile, ms / GiB | Our CE + cuBLASLt, ms / GiB | Throughput gain | Peak memory change |
|---:|---:|---:|---:|---:|
| 4k | 64.69 / 7.667 | 50.87 / 6.026 | +27.2% | -21.4% (-1.641 GiB) |
| 8k | 104.61 / 8.702 | 86.57 / 6.358 | +20.8% | -26.9% (-2.344 GiB) |
| 16k | 185.85 / 10.030 | 163.08 / 7.354 | +14.0% | -26.7% (-2.676 GiB) |
| 32k | 350.94 / 15.071 | 309.50 / 9.212 | +13.4% | -38.9% (-5.859 GiB) |
| 64k | 682.08 / 24.682 | 623.40 / 12.983 | +9.4% | -47.4% (-11.699 GiB) |
| 128k | 1345.24 / 45.854 | 1251.16 / 20.542 | +7.5% | -55.2% (-25.312 GiB) |
| 256k | 2751.71 / 78.474 | 2497.93 / 35.622 | +10.2% | -54.6% (-42.852 GiB) |

### 700B · C8

| Sequence | Baseline A + compile, ms / GiB | Our CE + cuBLASLt, ms / GiB | Throughput gain | Peak memory change |
|---:|---:|---:|---:|---:|
| 4k | 112.03 / 11.378 | 89.25 / 8.839 | +25.5% | -22.3% (-2.539 GiB) |
| 8k | 185.38 / 12.804 | 151.99 / 9.425 | +22.0% | -26.4% (-3.379 GiB) |
| 16k | 319.43 / 15.128 | 277.65 / 10.616 | +15.0% | -29.8% (-4.512 GiB) |
| 32k | 593.11 / 20.325 | 537.00 / 12.923 | +10.4% | -36.4% (-7.402 GiB) |
| 64k | 1150.16 / 28.784 | 1074.06 / 17.671 | +7.1% | -38.6% (-11.113 GiB) |
| 128k | 2284.18 / 49.272 | 2123.82 / 27.065 | +7.6% | -45.1% (-22.207 GiB) |
| 256k | OOM | 4190.98 / 45.835 | — | — |

All 4,560 measured rank calls had thermal counter samples bracketing each
call. No thermal counter increased, and no thermal slowdown or power-brake
flag was sampled. Peak core and HBM temperatures were 79 °C and 78 °C. The
software power cap was active in 16,368 of 16,415 in-call GPU samples. Physical
GPUs 6 and 7 had zero sampled utilization; this six-rank experiment does not
establish the cause of the earlier eight-rank timing anomaly.
[Full six-rank report](../../runs/head-frontier-nov/shape-matrix/length-grid/final-grid/gpu1-six-matrix/report.md) ·
[Per-rank measurements and launch summaries](../../runs/head-frontier-nov/shape-matrix/length-grid/final-grid/gpu1-six-matrix/replay/summary.json) ·
[Thermal evidence](../../runs/head-frontier-nov/shape-matrix/length-grid/final-grid/gpu1-six-matrix/thermal/verdicts.json) ·
[Telemetry methodology](../../runs/head-frontier-nov/shape-matrix/length-grid/final-grid/gpu1-six-matrix/thermal/README.md)

## Generalized CE and cuBLASLt comparison, 25 November

The isolated study adds 700B (GIFT 3.5) at C8 to the 100B C2/C8 comparisons.
The experimental kernel takes the runtime head count, row count and vocabulary.
The direct FP32 writer selects its cuBLASLt plan from operand dimensions and
alignment. The tables report elapsed milliseconds and GIFT total GiB per GPU
for both the standalone CE and the eight-rank head, with FP64 comparisons on
the same fixtures. The production acceptance below still refers to the November
21 code. [Shapes, results and precision checks](../../runs/head-frontier-nov/shape-matrix/report.md)

### Tuned CUDA CE and cuBLASLt versus Baseline A + compile

The timer covers one full LM-head microbatch on eight H100s, including the weight all-gather, all loss chunks, backward and the gradient reduce-scatter. Each row compares the same model shape and loss chunk count. Baseline A uses the median from the earlier compiled baseline run. The tuned result is the mean of two independent run medians, each measured over 20 microbatches after 10 warmups. [Baseline measurements](../../runs/head-frontier-nov/shape-matrix/report.md#whole-head-eight-h100s) · [Tuned runs and timing method](../../runs/head-frontier-nov/shape-matrix/tuning/report.md#eight-rank-head-results)

| Preset | Baseline A + compile, ms | Tuned CUDA CE + cuBLASLt, ms | Time reduction | Head throughput gain | Peak GIFT total per GPU, baseline → tuned (GiB) |
|---|---:|---:|---:|---:|---:|
| 100B C2 | 91.60 | 83.57 | 8.8% | 9.6% | 11.946 → 7.727 |
| 100B C8 | 104.54 | 87.08 | 16.7% | 20.1% | 8.548 → 6.165 |
| 700B C8 | 252.27 | 216.60 | 14.1% | 16.5% | 13.157 → 9.661 |

Head throughput gain is `baseline time / tuned time − 1` for the same token count. These percentages apply to the isolated head. On 700B C8, the full package saved 35.67 ms and 3.496 GiB per GPU versus Baseline A. The latest tuning alone saved 8.33 ms versus the previous `cuda-13` + cuBLASLt head in the paired runs, with both peaking at 9.661 GiB per GPU. [Paired run receipts](../../runs/head-frontier-nov/shape-matrix/tuning/head-summary.json)

## Final fused CE acceptance

The LM-head work used the `lm-head-mtp-fix` development branch on `sr008-gpu`,
based on `2c67c182`. The final acceptance runs used the separate `lm10-c1`
checkout at detached `e9b7cdbb` with the LM-head working-tree changes and the
pinned `cuda-13` hook.
[Final run receipts](../../runs/head-frontier-nov/train-ep8/ab01-summary.json)

On 21 November 2026, the owner accepted the final fused CE implementation
based on the full-training and isolated LM-head measurements below. **Code
review remains open.** The accepted variant uses exact folded CE `cuda-13`,
reuses BF16 logits for `dLogits`, writes `dHidden` in BF16, accumulates
`dW` directly into the FP32 GFSDP slot, and uses fused hidden z-loss.
It uses regular BF16 all-gather. Both compared arms use
`expandable_segments:True`. The results measure the full LM-head and loss
package against Baseline A with compile. The exact CE kernel's incremental
effect was measured separately against the same optimized head.
[Exact CE kernel comparison](../../runs/head-frontier-nov/native-ce-finalist/report.md)

### CE-only comparison with Liger and FLA

`cuda-13` had the lowest latency in both comparisons. We replayed the same
BF16 logits from the last chunk of a captured production-tokenizer microbatch
on one H100. Each route produced FP32 loss per row and wrote BF16 `dLogits`
into the logits buffer. CUDA Graph device events measured the CE region.
Projection, input restoration, per-head loss sums, and the LM-head gradient
GEMMs were outside the timed region. The Liger comparator used its pinned
Triton kernels. Both FLA variants used the bundled source.
[Liger source](https://github.com/linkedin/Liger-Kernel/blob/42dc23c41dcaf1f95d858837aed5301fd4faa27a/src/liger_kernel/ops/cross_entropy.py) · [Capture provenance](../../runs/head-frontier-nov/layout-order/report.md#writer-launch-order-and-numerical-verification)

**Unit upstream scale**, median of 32 interleaved repetitions per cell.

| CE implementation | C2 · 8,192 rows, ms | C8 · 2,048 rows, ms |
|---|---:|---:|
| **GIFT `cuda-13`** | **1.655** | **0.430** |
| Liger CE, Triton | 1.848 | 0.469 |
| FLA standalone CE | 2.368 | 0.626 |
| FLA Linear CE, CE kernels only | 13.813 | 3.455 |

Against Liger, `cuda-13` took 10.4% less time at C2 and 8.4% less at C8.
The Linear CE row runs the bundled FLA CE kernels on the same input shapes.
The earlier training port used a different CE implementation.
[Unit-scale measurements and kernel profiles](../../runs/head-frontier-nov/fused-ce/raw/ce-vendors-unit.json) · [Earlier port](../../runs/four-fusion/loss-implementation.md#source-and-integration)

**Production upstream scales**, median of 36 interleaved repetitions on the
second C2 chunk, with logits shaped `[8,192 × 128,256]` and scales
`1/N` for main and `0.3/N` for MTP1.

| CE implementation | C2, ms | Time relative to `cuda-13` |
|---|---:|---:|
| **GIFT `cuda-13`** | **1.654** | **1.00×** |
| Liger CE, Triton, with in-place row scaling | 3.282 | 1.98× |
| FLA standalone CE | 2.396 | 1.45× |

With scaling included, `cuda-13` took 49.6% less time than Liger and 31.0%
less than standalone FLA. The Liger profile measured 1.84 ms for CE and
1.43 ms for a separate pass that scales the saved BF16 gradient in place.
FLA applies the row scale in its CE backward kernel before storing BF16.
`cuda-13` applies it in FP32 within the single CE kernel before writing
`dLogits`. These are isolated CE timings.
[Production-scale measurements and kernel profiles](../../runs/head-frontier-nov/fused-ce/raw/ce-vendors-weighted-c2.json)

### Open code review for the integrated LM head

The source review below covers `final-all-fix@3783bf33e`, where the final CE
(`cuda-13`) has a native loss wrapper. The standalone acceptance above used
the benchmark hook. The integrated results cover EP8/eFSDP1 and EP4/eFSDP2
at C2 and C8, all with dense FSDP8. The standalone EP8 acceptance also covers
C32. [Combined results](all-fixes.md#1010-results) ·
[Integrated run and trace evidence](../../runs/final-all-fix/artifact-index.json)

#### Standardize cuBLASLt plan selection

`_staged_head_dw_writer.py` selects serialized descriptors by chunk count and
requires cuBLASLt 13.1.1. The labels are candidate indices from the tuner,
ordered by the first (`beta=0`) and subsequent (`beta=1`) writes. Both pairs
use cuBLASLt algorithm 66. C2 uses row-major descriptors with tiles 24 and
495. C8 and C32 use equivalent column-major descriptors with tile 29 for
both writes. All selected plans request four bytes of workspace and use
split-K 1. The selected writer also requires 256-byte-aligned operands and
reuses its descriptors and workspace on the original stream. See
`lm-head-mtp-fix@1d3bc5ca9`, `runs/lm-head-mtp/frontier-dw/tune-{2,8}.json`.
In the separate Lt versus GemmEx replay on that branch, the C2 wall-time
difference was within control spread. C8 improved by 3.21% at equal peak
total memory. See `runs/lm-head-mtp/frontier-dw/report.md` at the same revision.

Use one FP32 `dW` writer with a per-device plan cache keyed by actual GEMM
dimensions, strides and layout, dtype and compute type, pointer alignment,
and beta mode. Target the validated H100 Docker image and its cuBLASLt build.
Establish a bounded workspace budget. Query
`cublasLtMatmulAlgoGetHeuristic` at setup or on the first use of a descriptor
and validate the candidate and actual alignment. Keep the current plans as
measured regression controls. If no compatible Lt plan fits the budget,
test the existing `cublasGemmEx` writer with BF16 inputs, FP32 compute and
the same FP32 destination. Select that path before the first write. Compare
the entire `dW` chain at fixed chunk counts before switching the default.
[NVIDIA heuristics cache](https://docs.nvidia.com/cuda/archive/13.1.1/cublas/index.html#heuristics-cache) ·
[NVIDIA descriptor validation](https://docs.nvidia.com/cuda/archive/13.1.1/cublas/index.html#cublasltmatmulalgocheck)

#### Required support for chunk counts, MTP depth and model dimensions

The final CE and `dW` writer must work with C4, C16 and the effective chunk
count on monitoring steps, including switching back on the following ordinary
step. The trainer already changes and restores `loss_fn.num_chunks` for this.
Derive each call's dimensions from the actual chunk and support a shorter
last chunk. Keep batch, sequence length, hidden width and vocabulary out of
the preset guards. CUDA needs real row strides and safe vector tails and
vocabulary tiling within shared-memory limits. GEMM planning and the FP32
slot dimensions follow the actual operands.

Use the runtime MTP head count and validated loss weights, with MTP10 as a
validation case. The regular `MTPLoss` already folds a runtime count. The
current CUDA path returns two FP32 sums, passes `rows / 2` to CUDA and selects
between two scales. The `dW` selector separately fixes 16,384 folded rows.
Pass `rows_per_head` to CUDA, index `scales[row / rows_per_head]` and return
one FP32 sum per head. Preserve per-head weights, the shared main-token
denominator, label shifts, `-100` masks, BF16 `logits` reuse and `dHidden`
slice assembly. At C8 with B2/S4096, MTP1
to MTP3 raises the folded row count from 2,048 to 4,096 and doubles the
logits buffer from 501 to 1,002 MiB. The shared `dW` shape stays `[V,D]`.
[Acceptance wrapper](../../runs/head-frontier-nov/train-ep8/final_child.py#L124) ·
[CUDA launcher](../../runs/head-frontier-nov/fused-ce/centered.cu#L208) ·
[Kernel scale selection](../../runs/head-frontier-nov/fused-ce/exact.cuh#L164)

The combined recipe also enables `gfsdp_mtp_transition_prefetch`. Its
separate startup guard requires one MTP layer and its buffer transition is
defined for `mtp.layers.0`. This is an integration gate for multi-head runs
with the combined preset. See `models/gift_moe/_base/parallelize.py`,
`_validate_mtp_transition_request` and `set_buffer_region_transition`.

#### Parallelism and the loss contract

| Area | Current behavior and required follow-up |
|---|---|
| CP | Each rank sees fewer sequence positions. CE can operate on local token rows with a full vocabulary. The legacy dense GFSDP mesh includes the CP ranks and can sum their `dW` contributions. The existing MTP code shifts targets and input embeddings after CP partitions the sequence. Matching CP1 requires preserving both shifts across shard and load-balancer seams. See `trainer.py`, `distributed/parallel_dims.py`, `MTPLoss._shift_labels` and `models/common/decoder_blocks/mtp.py`. |
| PP and deferred sync | The last stage runs the head at the pipeline microbatch size. The schedule also makes a shape probe that must leave weight gradients untouched. Under PP, GFSDP allocates a private gradient buffer per group through the no-reduction policy, while the selector checks a head-specific flag set only under PP1. More importantly, the writer expects a clean slot and one reduce-scatter at the last loss chunk. PP can hold the unreduced gradient across loss calls. Direct `dW` must preserve that contribution and obey the schedule's reduction boundary. Also restore the caller's reshard policy. |
| Tied input and output weights | For shared `W`, the projection and the input lookup both contribute to `W.grad`. The lookup contribution arrives during model backward, after the current head lease has handed off `dW`. A tied implementation must combine the contributions under the shared parameter's ownership, either before reduction or in aligned sharded gradients. The current GIFT `gfsdp2` setup rejects weight tying and row-owner embeddings require untied weights. The CE calculation itself uses logits, labels and scales. |
| CE and writer selection | The final CE is currently conditional on the direct `dW` lease. Give the two paths separate eligibility checks. The CE needs a complete vocabulary per row and exclusive use of logits through the overwrite. The direct writer needs an owned FP32 slot and the correct reduction boundary. |
| GIFT 700B dimensions | The bundled 700B presets use local B3/S4096, hidden width 7168 and vocabulary 128256. GIFT 4 also uses `Unembedding` with softcap 30.0 and needs its derivative. The fused hidden z-loss currently uses the regular implementation at width 7168. See `configs/presets/train/gift_700b.yaml` and the GIFT 3.5 and GIFT 4 model presets. |

The combined preset's row-owner embeddings separately require TP/PP/CP1.
Their scratch path can borrow the head gradient buffer after its release.
Any tied design that holds this buffer through model backward must account
for that borrowing if it retains row-owner routing.

The non-PP trainer calls `loss = loss_fn(pred, labels, global_valid_tokens)`
and then `loss.backward()`. Inside `loss_fn`, each `chunk_loss.backward()`
has already produced `dW` and the BF16 `dHidden` slice with the per-head
weights and token normalization. The outer backward feeds the assembled
`dHidden` to the decoder with upstream scale 1. A factor applied after the
loss call cannot change the already accumulated `dW`; the inherited decoder
bridge also ignores the outer `grad_output`. Replaying backward on the same
loss cannot replay the per-chunk head work. These are limits of the chunked
loss interface. Existing in-loss weights and trainer accumulation work within
its normal single-backward contract.

Verify with compiled isolated replays for C4 and C16 with the existing C2/C8/C32
controls, a monitoring chunk switch, representative alternate B/S/D/V shapes,
and MTP1/2/3/10. Include unequal and zero weights, an entirely masked head,
and a shorter last chunk. Compare per-head losses, BF16 `dLogits` and `dHidden`,
reduced FP32 `dW`, and total GPU memory at fixed chunk count.
Check the fallback and the GFSDP slot lifecycle separately, including
distinct inputs across ranks and microbatches. `beta=0` starts a new full
slot each microbatch, then native reduce-scatter adds that microbatch to the
persistent sharded gradient. The saved isolated replay compares a reset
gradient and uses the same rank-zero input on all ranks.
[Isolated receipts](../../runs/head-frontier-nov/isolated-ep8/iso01-summary.json)

### Full-training acceptance runs

Six unprofiled runs used the same eight H100s on `sr008-gpu`, the eight-layer
100B-dev crop, EP8/eFSDP1 and dense FSDP8. Each run used the same real data,
production tokenizer, seed, EMA, SAC, Inductor model and loss, MTP1,
local B2/S4096 and 32 accumulation microbatches. Baseline A used
`e9b7cdbb` with its original compiled CE. The final arm used the same base
revision with the LM-head changes and the pinned fused CE hook. At C32,
the replay and trainer used the C8 cuBLASLt plan 5.5, verified for the
512-row folded chunk and both FP32 accumulation beta settings.
[All six run receipts and resolved configuration checks](../../runs/head-frontier-nov/train-ep8/ab01-summary.json)

| Loss chunks | Baseline A, tokens/s | Final CE, tokens/s | TPS change | Update time, baseline → final, s | Peak GIFT total, baseline → final, GiB/GPU | Peak change, GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 156,411 | 158,920 | +1.60% | 13.408 → 13.196 | 70.34 → 66.33 | −4.01 |
| 8 | 151,626 | 157,486 | +3.86% | 13.831 → 13.316 | 66.92 → 66.29 | −0.63 |
| 32 | 134,049 | 145,149 | +8.28% | 15.645 → 14.448 | 66.37 → 66.29 | −0.08 |

Each cell contains one run of 10 warmup and 10 measured optimizer updates.
Time and throughput are medians over updates 11–20, using the slowest rank's
elapsed time per update. Throughput counts global physical input tokens.
Memory is the maximum native GIFT total across all eight ranks and those
updates. The final CE was active on all ranks. C2 had the highest training
throughput of the three final-CE configurations.
[Memory and measurement protocol](../../planning/validation.md#memory-and-final-approval)

### Isolated LM-head and loss comparisons

The same node ran one 10+10 replay per cell with a real eight-rank GFSDP
group, regular BF16 all-gather and FP32 SUM reduce-scatter. Every rank used
the same captured rank-zero input. Each rank owned its own weight shard
and received its own reduced `dW` shard. The timer covers the isolated
LM-head and main/MTP loss forward and backward, hidden z-loss, and final
device synchronization. For each aligned replay, timing uses the slowest
rank. The table reports the median of ten timings and the maximum GIFT
total across the same replays and ranks. The TPS column describes the
isolated replays.
[All six replay receipts](../../runs/head-frontier-nov/isolated-ep8/iso01-summary.json)

| Loss chunks | Head input tokens/s, baseline → final | Head TPS change | Wall time, baseline → final, ms | Peak GIFT total, baseline → final, GiB/GPU | Peak change, GiB |
|---:|---:|---:|---:|---:|---:|
| 2 | 720,807 → 781,843 | +8.47% | 90.920 → 83.822 | 11.927 → 7.708 | −4.219 |
| 8 | 626,342 → 760,653 | +21.44% | 104.634 → 86.158 | 8.489 → 6.165 | −2.324 |
| 32 | 404,008 → 521,480 | +29.08% | 162.215 → 125.673 | 7.454 → 5.794 | −1.660 |

Historical C2 baseline peaks were 16.151 GiB on singleton GFSDP and
14.016 GiB on GFSDP8, both using `e3871b36` and the standard allocator.
The matched acceptance baseline uses `e9b7cdbb`, GFSDP8 and expandable
segments, which yielded 11.927 GiB. The GFSDP degree, revision, and allocator
vary across these historical comparisons. The final CE with regular
gather reproduced the earlier eight-rank peaks of 7.708 GiB at C2 and
6.165 GiB at C8.
[Singleton baseline](#direct-fp32-dw-replay-and-remaining-production-gates) · [Earlier eight-rank comparison](../../runs/head-frontier-nov/gather-repeat/report.md)

### Full-step memory headroom

At C8 and C32, the final variant lowered isolated peak memory versus
Baseline A by 2.324 and 1.660 GiB, respectively. Increasing the final
variant from C8 to C32 reduced its isolated peak by another 0.371 GiB
and added 39.516 ms per microbatch. Both full-training runs peaked at
66.29 GiB per GPU.

In the historical C8 update-11 trace, rank 0's LM-head phase peaked at
61.359 GiB of live PyTorch allocations. The rank's step peaked at
63.012 GiB when the decoder's gated delta rule backward allocated a 32 MiB
buffer in `prepare_wy_repr_bwd`. CE therefore had headroom below the phase
that set the observed step peak. Lowering the other peak-setting phases
could let the head's local savings reduce full-training memory. The resulting
GIFT total also depends on allocator reservation and sampled non-PyTorch
usage. The fresh runs establish the total peaks, and the precise peak phase
at C32 remains unverified. The earlier profile is `lh11-n8-001` update 11
in `lm11-01/phase-peak.jsonl` and `lm11-01/new-memory.jsonl` on the SR008
task share.
[Full-training GIFT totals](../../runs/head-frontier-nov/train-ep8/ab01-summary.json) · [Isolated GIFT totals](../../runs/head-frontier-nov/isolated-ep8/iso01-summary.json)

## Five changes in the current recipe

| Change | Previous behavior and cost | Current behavior and intended saving |
| --- | --- | --- |
| **1. Stable cap-zero cross-entropy** | The compiled PyTorch path used vector online-softmax statistics and a separate chosen-label reduction. Its backward performed a vocabulary-wide reduction for the scaled one-hot target gradient. | A tiled CE kernel computes centered FP32 statistics and gathers the target in the same row pass. Backward uses the scalar scale directly, subtracts the target in FP32, and stores `dLogits` in the logits dtype. This reduces CE work. The compiled control already scanned logits once in forward and kept two FP32 row-stat vectors. [Implementation](../../gift/torchtitan/experiments/four_fusion/stable_ce.py#L16) · [Generated-code comparison](../../runs/four-fusion/ce-trace-analysis.md#mechanism-grounded-in-generated-code) |
| **2. Folded CE backward** · `GIFT_STABLE_CE_FOLDED_BACKWARD` | Each loss head allocated its own `dLogits`. `SplitBackward0` then copied them into one folded matrix for the two LM-head backward GEMMs. The per-head buffers and folded destination were live during that copy. | Allocate the folded `dLogits` once. Each CE backward launch writes into its head's slice. Keep the per-head FP32 loss sums, loss weights, shared denominator, ignore mask and existing GEMMs. The implementation derives the number of loss heads at runtime. Main plus three MTP heads means four output slices and four CE backward launches. At a fixed token count per head, the eliminated copy grows with the number of head rows. The full-shape GPU receipt covers main plus one MTP head. [Implementation](../../gift/torchtitan/experiments/four_fusion/stable_ce.py#L179) · [Folded loss caller](../../gift/torchtitan/components/loss.py#L1089) |
| **3. Direct BF16 `dHidden`** · `GIFT_DIRECT_BF16_DHIDDEN` | Allocate a zero-filled FP32 destination, convert each BF16 hidden-gradient chunk to FP32 and copy it into its sequence slice. Cast the entire buffer back to BF16 for decoder backward. | Allocate the BF16 destination with `empty_like` and copy each chunk directly into its slice. These slices disjointly cover the sequence. The offset advances by the actual chunk width and the result checks coverage. The path handles 4/8/16 chunks under the existing plain BF16 guard. DTensor keeps its placement-aware path. The BF16 stores round the GEMM result as before. [Gradient assembly](../../gift/torchtitan/components/loss.py#L379) |
| **4. Staged LM-head `dW`** · `GIFT_STAGED_HEAD_DW` | `torch.mm` allocated a BF16 weight-gradient matrix for each chunk. GFSDP copied the first result to its FP32 slot and added subsequent results there. | With two eligible chunks, each native GEMM writes BF16 directly into one half of each row of the FP32 slot. After both writes, a Triton kernel loads both halves and stores their FP32 sum in the same bytes. This saves a full-size BF16 temporary while retaining the per-chunk BF16 rounding. The slot has space for exactly two BF16 matrices. Other chunk counts use the ordinary GFSDP path. [Staging and eligibility](../../gift/torchtitan/distributed/gfsdp/_staged_head_dw.py) · [Native GFSDP accumulation](../../gift/torchtitan/distributed/gfsdp/_param_group.py#L1266) |
| **5. Fused hidden z-loss** · `GIFT_FUSED_HIDDEN_Z_LOSS` | The final-norm input regularizer materialized whole-activation FP32 intermediates for magnitude and log-sum-exp. | The kernel reads BF16 hidden states and computes magnitude, maximum and shifted exponential in FP32. It retains ATen's FP32 row sum and final scalar arithmetic. Backward computes the elementwise derivative and stores BF16 once. The current guard selects contiguous BF16 hidden states of width 4096. [Regularizer](../../gift/torchtitan/models/common/regularization.py#L227) · [Kernel and replay](../../runs/fusion-pack/zloss-component.md#arithmetic-and-storage) |

### Gradient buffer changes in code

The snippets show the allocation and write paths. In folded CE the unchanged
forward splits logits by loss head. The backward change moves the split to
views of one preallocated gradient destination.

```python
# Before: per-head backward output, then SplitBackward0 assembles the result.
head_grads = [ce_backward(head_inputs) for head_inputs in inputs_by_head]
dlogits = torch.cat(head_grads, dim=0)

# After: each CE backward launch receives its own output view.
dlogits = torch.empty_like(logits)
for head_inputs, output in zip(inputs_by_head, dlogits.chunk(num_heads)):
    launch_ce_backward(*head_inputs, output)
```

For `dHidden`, the sequence slices and GEMMs stay the same. The destination
changes from zero-filled FP32 to BF16 storage that is fully overwritten.

```python
# Before, for a BF16 input and BF16 chunk gradient.
buffer = torch.zeros_like(hidden_states, dtype=torch.float32)
buffer[:, start:end, :] = chunk_grad.to(torch.float32)
# After all chunks:
dhidden = buffer.to(hidden_states.dtype)

# After, on the direct BF16 path.
buffer = torch.empty_like(hidden_states)
buffer[:, start:end, :] = chunk_grad
# After all chunks and the coverage check:
dhidden = buffer
```

## Measured cost in the historical comparisons

Training memory below is the historical maximum-rank PyTorch allocator peak,
shown as allocated/reserved. Component measurements use one H100. Each row
cites its own control because the five changes were introduced in different
comparisons.

| Change | Standalone 10+10 TPS and memory | Component or trace evidence |
| --- | --- | --- |
| Stable CE | **+0.3034%**, +0.0194 GiB allocated in the initial pair. The repeat comparison found **+0.0560%**, −0.0074 GiB allocated. The training effect was inconclusive. | Wider CE trace elapsed fell by 0.434–1.306 ms per chunk. Whole-head timing was mixed. The local synthetic replay had equal total weighted loss and at most one BF16 ULP difference across the checked gradients. [Historical comparison](../../runs/four-fusion/report.md#comparison) |
| Folded CE | **+0.72%**, −0.923/−0.977 GiB. | Full two-chunk head replay took 81.571 → 79.426 ms by CUDA events. Losses, `dLogits`, `dHidden` and accumulated head `dW` matched bitwise. Native traces showed six target gradient concatenations per rank before and zero after over three microbatches. [Component and training evidence](../../runs/backward-components/report.md) |
| Direct BF16 `dHidden` | Standalone training result open. | CUDA-event span 0.665 → 0.193 ms per two-chunk call. Peak allocated fell 256 MiB, reserved was equal, and the assembled gradient matched bitwise. [Component results](../../runs/fusion-pack/dhidden-component.json) |
| Staged LM-head `dW` | **+0.543%**, −1003/−1000 MiB. | Both BF16 chunk results and the merged FP32 `dW` matched bitwise. The one-rank integration replay also covered native GFSDP callbacks and slot reuse. [Component and training evidence](../../runs/fusion-pack/report.md) |
| Fused hidden z-loss | Standalone training result open. | Eager forward plus backward fell from about 1.160 to 0.299 ms. Both allocator peaks fell by about 512 MiB. The loss, row statistics and hidden gradient matched bitwise on the captured input. [Component results](../../runs/fusion-pack/zloss-component.md) |

The historical three-fix package combined direct BF16 `dHidden`, fused hidden
z-loss and direct GDN backward. It measured **+0.681% TPS** and **−382/−240 MiB**
allocated/reserved. That pair does not isolate either LM-head-group change.
There is no matched five-change-only comparison for this group.
[Package comparison](../../runs/fusion-pack/report.md#native-1010-results)

## Direct FP32 `dW` replay and remaining production gates

The two singleton replay series used a fixed chunk count and measured the
isolated LM-head forward and backward, main/MTP CE, hidden z-loss and GFSDP
path. Each arm ran
in two fresh processes on the same H100, with 10 warmup and 10 timed
iterations. The first series compared the original compiled GIFT path with
four fixes plus direct FP32 `dW` accumulation replacing staged `dW`. The fresh
matched series compared four fixes plus compiled GemmEx with the full memory
stack, which also adds early logits release, BF16 in-place `dLogits`, tuned
FP32 cuBLASLt `dW`, the expandable-segments allocator, and aliasing the BF16
NCCL all-gather send shard with its local receive slice.

| Sequence chunks | Historical replay: compiled GIFT → four fixes + direct FP32 GemmEx (ms / total GiB) | Fresh matched replay: four fixes + compiled GemmEx → full memory stack (ms / total GiB) |
| ---: | ---: | ---: |
| 2 | 83.71 → 76.83 / 16.151 → 14.194 | 77.411 → 76.615 / 14.194 → 10.681 |
| 4 | 89.10 → 79.22 / 14.196 → 12.237 | 78.929 → 77.780 / 12.237 → 9.684 |
| 8 | 99.74 → 83.16 / 14.194 → 11.261 | 81.564 → 78.820 / 11.261 → 9.177 |
| 16 | 116.81 → 93.72 / 11.751 → 10.280 | 92.012 → 89.296 / 10.280 → 8.903 |

In the fresh matched replay, full loss, `dHidden`, `dW`, z-loss and
z-gradient digests matched bitwise between each pair.

### Native two-rank all-gather alias replay at 8 chunks

This test used NVLinked H100s, replicated the captured rank-0 batch on both
ranks, kept parameter and `dW` shards distinct, and isolated the all-gather
alias on the same allocator. Both paths executed 24 BF16 all-gathers and 23
FP32 SUM reduce-scatters per process and rank. Peak total memory was the same
on both ranks and fell from **8.016 to 7.509 GiB** (**520 MiB per rank**).
Peak allocated memory fell by **501 MiB per rank**. The synchronized
maximum-rank medians were
**85.306 → 87.048 ms** and **86.009 → 85.956 ms** across the two paired
comparisons, so timing did not show a consistent speedup. Full output digests,
including each rank's local FP32 `dW`, matched exactly. The BF16 input shard
is 125.25 MiB per rank at FSDP8. The subsequent eight-rank replay measured
140 MiB of total savings at C2 and 120 MiB at C8 with aliasing.
[Eight-rank gather comparison](../../runs/head-frontier-nov/gather-repeat/report.md)

The canonical report is `runs/lm-head-mtp/report.md` in the `lm-head-mtp-fix`
worktree on `sr008-gpu`. See **Two-rank GFSDP replay at eight chunks**.
The adjacent `artifacts/` directory contains
`frontier-best-final-001/verified-summary.json` and
`frontier-dual-001/summary.json`.

### FP64 `dW` reference

On identical BF16 operands, the relative L2 error to FP64 for ordinary GFSDP
versus direct GemmEx was **1.35e−3 → 1.29e−5** at two chunks,
**1.13e−3 → 6.97e−6** at four, **9.84e−4 → 3.82e−6** at eight, and
**9.06e−4 → 2.16e−6** at sixteen. At two chunks, ordinary and staged `dW`
had the same relative L2 error. The fresh matched replay compared the
subsequent changes against direct GemmEx with full-output digests.

The six original-A training runs are recorded in **Original A training at
10+10 on eight H100s** in the canonical report above. They cover EP8,
dense FSDP8 and two or eight loss chunks. The matrix and all-rank summaries
are in `lm10-01` on `sr008-gpu`. The promoted production recipe and longer
convergence validation remain separate acceptance gates.
[Measurement protocol](../../planning/validation.md#memory-and-final-approval)

The original implementation proposal was to accumulate each BF16-input GEMM
directly into the existing FP32 GFSDP slot with `CUBLAS_COMPUTE_32F`,
`alpha=1`, `beta=0` for the first write and `beta=1` thereafter, then pass the
slot to the existing reduction on the final sync callback. Against ordinary
GFSDP this removes the separate **1002 MiB** BF16 `dW` temporary and its
copy/add; it uses the existing **2004 MiB** FP32 slot. At two chunks, staging
already avoids that temporary. Later GEMMs still read and write the slot, and
cuBLAS may use algorithm-dependent workspace. The earlier cuBLASLt sweep kept
BF16 output and tested pitched staging.
[Apex implementation](https://github.com/NVIDIA/apex/blob/master/csrc/megatron/fused_weight_gradient_dense_cuda.cu#L33) · [Megatron caller](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.20.0/megatron/core/tensor_parallel/layers.py#L741)
[Captured GEMM shapes and cuBLASLt confirmation](../../runs/fusion-followup/head-lt.md#balanced-confirmation)

**Precision hypothesis.** The staged path computes
`sum_fp32(bf16_round(dW_i))`; direct accumulation computes
`sum_fp32(dW_i)`. Direct accumulation removes one rounding of each chunk
result while both inputs remain BF16. For example, partial contributions
`1.001953125` and `−1` sum to `0.001953125`, while rounding the first to BF16
makes the sum `0`. That example illustrates the arithmetic change; the
reported relative-L2 results above measure its error against FP64 on the
captured operands.
