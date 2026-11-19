# Folded CE and expert wgrad conversion on Hopper

The fixed production recipe uses S, the GPU capacity selector, five Hopper fusions, FP8 expert weights, direct expert dW and tail retention. All training arms use EP8/eFSDP1, dense FSDP8, TP/PP/CP1, B2/S4096, accumulation 32, MTP1 with weight 0.3, SAC, compile, EMA, the same production tokenizer, data-002, seed 2048 and the saved D compiler cache. The two candidates affect different backward paths. [Recipe](../../docs/baseline.md)

## Native input capture

A one-update run at `8c223c720` passed with all three GFSDP flags on eight H100 80 GB GPUs on `sr008-gpu1`. Rank zero captured both heads, the full BF16 output weight, labels, positions, normalization and all four rowwise expert operands at the first microbatch. The original byte/scale layouts and direct FP32 slot offsets are in the fixture. Torch was `2.15.0.dev20260822+cu130`, Triton 3.8.0 and DeepGEMM `2.3.0+f093a4c`. The 116-SM writer build manifest SHA-256 is `caf632bcf954b8e036cbeaebde66fbdde728cc9b3ef7b9476222d28c0188d53d`.

The server task root is `/workspace-SR008.nfs2/vkropotin/optimization-task`. The fixture is `runs/kernel-fusion/artifacts/raw/backward-inputs-001/component-inputs/rank0.pt`, SHA-256 `15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727`. Its resolved config SHA-256 is `459c283ec8daebb6493198fca4cba6e23e168324a097f79e948ea72ba5b2ed3d`. Raw component timings, per-expert bit checks and Kineto traces are under `runs/backward-components/artifacts/raw/components-001/`. Both component experiments ran at `01418da86`. [Capture](capture.py) · [Campaign commands](campaign.sh)

## Folded CE

Each existing stable CE backward launch writes its head's rows into the final BF16 logits gradient. The two native chunk callbacks, head GEMMs, global denominator, head weights and FP32 row statistics are preserved. The experimental switch is `GIFT_STABLE_CE_FOLDED_BACKWARD`, default off. The native launcher exposes it through `--backward-optimization folded-ce`. FlashAttention demonstrates directing CE backward into a selected destination. The task implementation retains both the saved logits and GIFT's established rounding. [FlashAttention at 060c9188](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/flash_attn/ops/triton/cross_entropy.py) · [GIFT loss](../../gift/torchtitan/components/loss.py) · [GIFT stable CE](../../gift/torchtitan/experiments/four_fusion/stable_ce.py)

| One-H100 component, full two-chunk forward and backward | Control | Folded CE |
| --- | ---: | ---: |
| Median CUDA-event span, ms | 81.571 | 79.426 |
| Median wall time, ms | 81.627 | 79.453 |
| Peak allocated, GiB | 9.370 | 8.454 |
| Peak reserved, GiB | 9.943 | 8.455 |
| Target gradient concatenations per three microbatches | 6 | 0 |

Fresh processes each completed ten warmups and thirty observations. CUDA events include submission gaps. The standalone replay explicitly compiled the real Linear head and loss. The captured native GFSDPLinear had an eager module call boundary and compiled loss. Full training below uses its original boundaries. The component includes a local post-accumulate hook that collects both BF16 chunk gradients into the FP32 weight accumulator. [Full-shape replay](folded_ce.py)

All 2,101,346,304 BF16 gradient elements across the two chunks matched bitwise. Both per-head FP32 loss sums, the weighted loss, metrics, acceptance counters, both hidden-gradient chunks and the accumulated FP32 head gradient matched. The six ignored rows had zero gradients and both cross-chunk MTP targets matched the independent shift check. The opt-in CPU fallback test also passed in the SR008 environment.

The initial trace checker required Python custom-op labels that Inductor inlined. Its raw reports retain the failed label check. The corrected checker correlates the actual CE launches, split-owned copies and both consumer GEMMs. Both saved traces passed offline. Each had twelve CE backward kernels and twelve matching head backward GEMMs. The six control copies totaled 8.814 ms. The same-stream CE-to-final-GEMM spans summed to 150.041 ms in control and 145.210 ms with folded CE. The checker found no equivalent copy in those spans. [Compiled trace check](component_trace.py)

## Expert wgrad row2col

The existing CudaCherry and Triton converters both reconstruct per-token FP8 values in FP32 and requantize into the contiguous, expert-grouped layout consumed by the Hopper wgrad GEMM. CudaCherry 0.1.0's installed Python wrapper hash matched the local donor at `585bd50a641857ccae8b5ec691bb15791c9529e7`. Both candidates used the same existing rounded FP32 direct writer and 116-SM budget. [Replay](row2col_replay.py) · [DeepGEMM Hopper scale layout](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/docs/scaling-factor-format.md)

| Scope | CudaCherry median, ms | Triton median, ms |
| --- | ---: | ---: |
| w13 X, 4096 | 0.2344 | 0.4226 |
| w2 dY, 4096 | 0.2348 | 0.4215 |
| w13 dY, 3072 | 0.1789 | 0.3272 |
| w2 X, 1536 | 0.1019 | 0.1875 |
| w13, both conversions and direct GEMM | 2.3231 | 2.6958 |
| w2, both conversions and direct GEMM | 1.2937 | 1.5603 |

Each scope used six warmups and 18 timed trials with interleaved matched order and a second CudaCherry control. Both complete projection paths had identical measured allocator peaks across backends. All consumed E4M3 bytes, FP32 scales and both direct FP32 dW slots agreed bitwise. The real activation had 63,872 aligned rows across 16 experts. A separate fixture built from captured values verified empty experts, partially padded groups, and an unused capacity tail.

Both width-4096 roles needed one scale-layout copy. Their copies cost approximately 20 µs on each backend. The w13 X conversion kernel itself took 204 µs with CudaCherry and 375 µs with Triton. The corresponding w2 dY durations were 208 and 374 µs. The shared copy accounts for little of the difference. Keep CudaCherry in production. The Triton arm failed the standalone speed gate for both projections and did not qualify for a training run. A producer-stride optimization would affect both backends and requires its own producer-plus-consumer measurement.

## Native 10+10 comparison

| One eight-H100 training run per arm | Global physical tokens/s | Window, s | Median max-rank update interval, s | Max-rank allocated peak, GiB | Max-rank reserved peak, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current production | 177,001.55 | 118.482 | 11.843 | 62.577 | 64.314 |
| Folded CE only | 178,274.40 | 117.636 | 11.754 | 61.654 | 63.338 |

Folded CE gained 1,273 tokens/s (+0.72%) in this pair. Allocated and reserved peaks fell by 0.923 GiB and 0.977 GiB. Both runs processed 20,971,520 physical input tokens in updates 11–20. Their normalized resolved configs, GIFT source fingerprint and direct-dW build matched. Each measured window had zero allocator retries and OOMs on every rank, and the cache observer reported stable state. A single timing pair measures the difference on this allocation.

The native campaign ran at `01418da868916ac73564aaa4c04feae1a267ec3a`. Receipts and traces are under `runs/kernel-fusion/artifacts/raw/backward-native-001-{control,folded-ce}-{timing,trace}/` on the server. The run IDs encode the arm and measurement. The corresponding command was `bash runs/backward-components/campaign.sh native backward-native-001 control folded-ce`. Timing and profiling ran in separate processes. Each trace captured the first three microbatches of update 21 on ranks 0 and 1.

| Profiled rank | Control target copies | Folded target copies | Control copy time, ms | CE start through both backward GEMMs, control → folded, ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6 | 0 | 8.837 | 154.236 → 147.536 |
| 1 | 6 | 0 | 8.830 | 153.921 → 145.396 |

The spans sum six chunks per rank. The CE and GEMM kernel names, grids and block dimensions matched across arms. The only intervening device operations in the folded chains were GEMM memsets. Both ranks retained 84 CudaCherry conversions and 42 rounded FP32 expert wgrad launches per capture. The expert weight cache refreshed 14 times on the first microbatch and zero on the next two, with 12 logical expert unshards on each microbatch. All eight timing ranks logged the expected arm. [Native trace analysis](trace.py) · [Measurement values and source, configuration and trace hashes](results.json)
