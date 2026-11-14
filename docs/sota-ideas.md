# SOTA runtime ideas after hypothetical GFSDP completion

**Survey date:** 14 November 2026; **research window:** 14 November 2025–14 November 2026, with older predecessors where relevant.
**Status:** Read-only research and source screening. No source or configuration change, training run, benchmark, or candidate acceptance follows from this survey.

## Purpose and evidence boundary

This document adapts the owner's accepted SOTA survey into English. It records runtime-optimization directions to consider only after a hypothetical completion of the GFSDP work. It does not change the accepted workload, reopen completed comparisons, or authorize implementation or execution. The [work plan](../planning/README.md) owns future-work authorization; [candidates](candidates.md) and [profile notes](profile-my-ideas.md) retain earlier inventories and hypotheses.

The reference workload is the agreed eight-H100, NVLink-connected crop: EP8, dense FSDP8, expert FSDP1, PP/TP/CP1; dense and sparse FP8; model and loss compilation; the current SAC policy; MTP1; microbatch 2, sequence length 4096, and accumulation 32. See the [baseline contract](baseline.md) for the complete configuration.

The profile anchors below come from the existing V2 rank-0 capture of update 21's first three microbatches. Its integrated three-microbatch device span is 1293.713 ms. It is neither a whole optimizer update nor an all-rank comparison, and it does not represent a hypothetical post-GFSDP-optimization implementation. The recorded all-device-idle union is 42.899 ms (3.32% of that span); that small measured idle share does not bound the gains of changes to algorithms, dataflow, or work that is currently overlapped. Kernel sums and opportunity anchors can overlap other streams and are not removable latency or speedup estimates. The [machine-readable screening record](../runs/v2-prefetch-baseline/v2-candidate-screening-perfetto.json) and [V2 report](../runs/v2-prefetch-baseline/report.md) retain the trace context and method.

The order below is a research priority, not a claim that ten ready-to-run optimizations exist. It preserves the accepted prioritization and its compatibility gaps.

## Ranked research queue

| Rank | Direction | Existing trace anchor | Main gate |
| --- | --- | ---: | --- |
| 1 | Pipeline dispatch, expert GEMM, and combine within each MoE layer | EP union 157.627 ms; actual FP8 expert-GEMM kernel sum 168.435 ms | Selected scope: fine-grained kernel overlap; preserve blockscaled FP8 and full training semantics; substantial kernel/backend development is required |
| 2 | FlashQLA for GDN | Forward + backward scan kernels 58.979 ms | Validate CP warmup, compilation, SAC, and numerical behavior |
| 3 | Distribution-aware grouped-GEMM scheduling | FP8 grouped-GEMM kernel sum 169.217 ms | Measure real expert shapes and the current scheduler before judging the 116-SM preset |
| 4 | Backward dispatch, recomputation, and combine/dW overlap | No directly isolated removable-delay estimate | Reschedule against current SAC, gradient, and reduction dependencies |
| 5 | Router-preserving dynamic expert copies | Rank-0 EP evidence is insufficient to establish skew | First establish all-rank load skew and copy/replica costs on H100 FP8 |
| 6 | Co-selection-aware expert placement | No placement cost measured in the crop | Preserve global expert IDs, capacity/drop behavior, and the current HybridEP contract |
| 7 | GPU-resident metadata and larger CUDA Graph regions | 42.899 ms all-device idle; only 6.319 ms of capacity host-sync intersects it | Verify exposed launch/control cost in the actual path; CUDA Graphs remain off in the accepted baseline |
| 8 | Producer-side normalization/quantization fusion | Requantize 14.861 ms; vector quantize 9.846 ms; scaling-aware transpose 7.241 ms | Target only boundaries not already fused in current source; cited TE MLP fusion is SM100-only |
| 9 | Native FP8 parameter storage/gather | No post-GFSDP benefit measured | Preserve FP32 masters/state and gate FP32→FP8 against FP32→BF16→FP8 rounding |
| 10 | FP8 weight cache across 32 microbatches | Dense weight block quantization 9.246 ms | Refresh scales/layouts after optimizer updates; account for memory and separate expert-weight lifecycle |

### 1. Pipeline dispatch, expert GEMM, and combine within each MoE layer

The selected scope is fine-grained, kernel-level overlap of dispatch, expert GEMMs, and combine. The owner rejected shared-expert-only overlap and coarse token-chunk pipelines as implementation directions. The rank-0 capture records 157.627 ms of EP-family kernels and 168.435 ms of actual FP8 expert GEMMs, all on stream 7 with no overlap between those families. The earlier 169.217 ms grouped-path aggregate, retained as a survey anchor elsewhere below, includes 0.783 ms of transpose helpers. These budgets are not measured removable step latency.

See the [detailed MoE overlap analysis](sota-ideas/overlap-moe.md) for the Hopper benchmarks, trace-derived scenarios, exact FP8 incompatibilities, and accepted engineering scope. The existing DeepGEMM public API does not expose the readiness and output-publication protocol needed for this overlap. Either extend DeepGEMM kernels with communication or implement the current blockscaled FP8 recipe inside a donor's fused GEMM. Preserving the FP8 numerical recipe does not require preserving the DeepGEMM library.

The owner survey grouped DeepSeek MegaMoE, ByteDance MegaScale-MoE, FLUX, COMET, FlashComm, MiniMax, and MAI references under this direction. Relevant linked reports include [DeepSeek V4](https://arxiv.org/html/2606.19348v1), [ByteDance MegaScale-MoE](https://chaojin0310.github.io/files/papers/EuroSys26-MegaScale-MoE.pdf), [MiniMax-01](https://arxiv.org/html/2501.08313v1), and [MAI Base 2](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/). The screening record also pins FLUX and MegaMoE DeepGEMM, and the inspected [FlashComm implementation](https://github.com/ByteDance-Seed/Triton-distributed/tree/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/FlashComm). That pin combines dispatch GEMM and combine on SM90 but is BF16/FP16; full training integration is not established. The MegaMoE DeepGEMM pin (`5081f8513af57e61696de4f4f04b0e707ffb503e`) is SM100 forward-only, so it is not evidence for H100 training. Neither establishes the required end-to-end FP8 forward/backward path.

### 2. FlashQLA for GDN

The [Qwen 3.8 Next report](https://arxiv.org/html/2608.30320v1) motivates examining FlashQLA with the GDN layers. The inspected [FlashQLA implementation](https://github.com/QwenLM/FlashQLA/tree/a14dbf3474514e1674526431e960325f72081e9f) supports SM90 forward/backward, BF16/FP16, FP32 state accumulation, and K/V dimension 128. Its GVA structure is compatible with this model's 16 key heads and 32 value heads; the contract is also recorded in the JSON screening record.

The current scan-kernel anchor is 30.245 ms forward plus 28.734 ms backward, or 58.979 ms total. The 8.868 ms extra VJP-forward kernels are already inside the forward total and must not be added again. QKV convolution/unpack is another 17.013 ms anchor; changing the scan does not automatically remove it.

First assess the inspected `auto_cp=False` path. CP warmup uses a decay threshold of −10 and needs a numerical review. Compilation, SAC interaction, and numerical parity are untested. These conditions keep FlashQLA high in the research queue, but do not establish a port.

For the detailed source contract, benchmark comparison, conditional savings arithmetic, and gate-integration options, see the [FlashQLA analysis](sota-ideas/flash-qla.md).

### 3. Distribution-aware grouped-GEMM scheduling

The grouped-GEMM kernel sum is 169.217 ms, but the current preset's 116-SM budget does not prove the active DeepGEMM scheduler is poorly matched. Compare actual expert-token shapes and the current scheduler/code path before changing SM allocation or scheduling. Use [Kimi K3](https://arxiv.org/html/2607.24653v1) as a research reference, not a claim that its setup transfers directly to this crop. Preserve current routing, expert shape, FP8 format, SAC, and compilation during any later authorized comparison.

### 4. Backward dispatch, recomputation, and combine/dW overlap

Study the two reported overlaps separately: initial backward dispatch with recomputation, and final collect with weight-gradient work. [MAI Base 2](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/) is a relevant systems reference. The compiled V2 schedule already executes both expert dgrads and gradient collection before the wgrads; simply changing the Python backward order is not a new optimization. These operations still do not overlap on the GPU. The crop already repeats work through SAC, so a useful schedule must fit the existing recomputation policy and correct gradient/reduction dependencies. Cross-microbatch EP overlap in the inspected Megatron reference conflicts with shared-expert overlap and selected recompute modes; its PP>1 setup also requires virtual pipeline parallelism. No official YOLO implementation pin was established in the screening, and the available GB200 evidence does not establish H100 behavior.

### 5. Router-preserving dynamic expert copies

MoonEP and FEPLB suggest dynamic expert replication/balancing while preserving the router's selected experts. Before investigating a copy policy, establish all-rank skew and the communication, storage, and replica-gradient costs. A rank-0 trace cannot establish an imbalance benefit. The inspected [MoonEP implementation](https://github.com/MoonshotAI/MoonEP/tree/d86fda40329de4002e0cade6817df7b4b5512ee5) demonstrates BF16 training with dynamic redundancy on H20 at EP8; H100 FP8 remains untested, and its zero-copy lifetime constraints matter. FEPLB has H100 per-layer evidence, but no established end-to-end FP8, FSDP, compile, and SAC path for this workload. See the [FEPLB report](https://arxiv.org/html/2604.19654v1); implementation pins and gaps are recorded in the JSON.

### 6. Co-selection-aware expert placement

[Expert Coupling](https://arxiv.org/html/2610.09372v1) studies equal-expert-count graph partitioning based on co-selection. The potential fit here is to reduce the number of recipient GPUs without changing which experts the router selects. The placement portion is relevant to the current TP1 layout; strategies that change token ownership through TP/SP imply a different topology.

Before considering a placement change, reconcile it with HybridEP's existing deduplication, global expert IDs, capacity checks, and token-drop semantics. No official H100 FP8 implementation pin or crop placement measurement was established. [PlaceMoE](https://arxiv.org/html/2610.14798v1) is additional related research context, not a validated local implementation.

### 7. GPU-resident metadata and larger CUDA Graph regions

The trace shows host synchronization around capacity decisions and many short launches, but the total 652.941 ms of capacity host-sync time is not GPU idle time: only 6.319 ms intersects all-device-idle intervals. The all-device-idle union is 42.899 ms. This intersection is a useful exposure bound for that trace, not a bound on every possible dataflow change.

Investigate whether metadata/control can remain device-resident and whether a larger graph region addresses measured launch gaps while preserving the accepted compile and SAC behavior. The baseline leaves CUDA Graphs disabled, and this survey does not enable them. The inspected [Transformer Engine Hopper device-offset GEMM](https://github.com/NVIDIA/TransformerEngine/blob/7457e47d26d9d02dfb783007335dfbe75e1b90a8/transformer_engine/pytorch/ops/basic/grouped_linear.py#L76) has cuBLASLt version gates: 13.4 for BF16, 13.5 for current FP8, and 13.6 for FP8 block-128 alignment; the current environment was not verified against those requirements. The [Transformer Engine optimization documentation](https://docs.nvidia.com/deeplearning/transformer-engine/examples/advanced_optimizations.html) is a reference, not compatibility proof.

### 8. Producer-side normalization and quantization fusion

The trace records 14.861 ms in grouped requantization, 9.846 ms in vector quantization, and 7.241 ms in scaling-aware transpose. These operation budgets may overlap work on other streams and cannot be converted directly into removable step latency. Current source already has SwiGLU quantization and scaling-aware transpose, so those mechanisms should not be proposed again as missing work. Examine only remaining producer/consumer boundaries.

The inspected [Linghe implementation](https://github.com/inclusionAI/linghe/tree/adaa17798a84ff6ff353ca8f713d7a06b1093638) has H800 training evidence for normalization and quantization, but its scale-recipe compatibility is unknown. [Transformer Engine's fused MLP implementation](https://github.com/NVIDIA/TransformerEngine/blob/7457e47d26d9d02dfb783007335dfbe75e1b90a8/transformer_engine/pytorch/ops/fused/grouped_mlp.py#L1204) is SM100-only and does not establish a Hopper implementation for this workload. No fusion result is accepted here.

### 9. Native FP8 parameter storage and gather

The [Nemotron Nano 2 report](https://arxiv.org/abs/2508.14444) and Megatron's inspected `fp8_param_gather` path motivate evaluating native FP8 parameter storage/gather only after the hypothetical GFSDP completion. Keep FP32 master parameters and optimizer state. Gate any representation change on the actual FP32→FP8 versus FP32→BF16→FP8 rounding and its effect on training. This is not a proposal to lower FP32 gradient-reduction precision. The incremental benefit after GFSDP is unknown, and a 2025 fully-FP8 training reference retained a BF16 final projection.

### 10. FP8 weight cache across 32 microbatches

The first-three-microbatch trace records 9.246 ms of dense weight block quantization, about 0.7% of the 1293.713 ms measured window. This is an anchor, not guaranteed recoverable time. A cache would need to refresh quantized weights, scales, and layouts after each optimizer update; expert-weight caching has a separate lifecycle and cost. Account for extra memory. At the inspected Transformer Engine revision, its named FP8 weight cache is disabled for FSDP2, which is not proof of compatibility with GFSDP. This remains the tenth-ranked candidate pending a post-GFSDP cost estimate.

## Screened directions and current-source dispositions

These notes prevent future work from treating already-present mechanisms or rejected evidence as new opportunities.

- **Four fusion experiments:** all four explicit-kernel candidates were measured in their assigned scope; none was accepted or enabled. GPU input preparation and routing metadata improved their local regions, but their training-throughput comparisons were inconclusive. Cap-zero CE improved local CE timing, but the whole-training throughput result remained inconclusive. The active 1,024-token-tile linear-CE port regressed the whole head and training throughput and was rejected. The [four-fusion report](../runs/four-fusion/report.md) owns all results and caveats.
- **SonicMoE:** its inspected implementation is an H100 BF16 training/I/O reference, not an established FP8 EP path. Current task source already avoids saving expert outputs solely for router gradients: probability is computed inside SwiGLU and `fused_swiglu_quantization` exposes `grad_probs`. Token rounding changes router behavior and is excluded. The task HEAD (`657e77a4f8ddb116b75925e5671b00642a0e4d32`) is not proof that the V2 execution used identical source.
- **FUSCO:** its permutation–communication fusion concept is relevant, but HybridEP already has an optional `fuse_permute` path. The inspected [FUSCO source pin](https://github.com/infinigence/FUSCO/tree/dd3f46113c1cf40dbebd426f2f3557e31a16b7fe) has no public CUDA runtime source; `libfusco.so` is required, float8 is rejected, and the weighted-combine probability-gradient wrapper has a gap. The paper is [here](https://arxiv.org/html/2512.22036v1).
- **ReWY:** the inspected equal-head API does not fit this model's 16/32 key/value head dimensions, and distributed compilation is unvalidated. The [paper](https://arxiv.org/abs/2611.01940) does not remove those implementation gates.
- **FP8FlowMoE:** mechanisms attributed to this work are already active in the current trace/source. It is not a new independent candidate without a specific missing mechanism and measured boundary; see the [report](https://arxiv.org/html/2511.02302).
- **Chunked output projection/CE:** the current workload already uses two loss chunks. The prior linear-CE port regressed and was rejected. The eager BF16 LM head remains a large 203.602 ms forward/backward GEMM anchor, but this survey establishes no new compatible accelerator or CE substitution. The [GLM 5 report](https://arxiv.org/html/2602.15763v1) is architecture/research context, not a validated alternative implementation.
- **MLA backend:** the active cuDNN flash SDPA path has 29.969 ms of kernels; this does not establish a faster fallback or an MLA replacement's end-to-end value. Source inspection found the current MLA scaling, mask, and boundary contracts consistent with the active row-wise causal path, but full numerical forward/backward parity remains untested. FlashMLA's current code removed Hopper support, and newer inference MLA mechanisms are not training drop-ins.
- **Memory offload / allocator-directed movement:** ERNIE VMM allocator-offload ideas require evidence of relevant memory pressure. The V2 capture does not provide that evidence, so the direction is not prioritized from this trace. See the [ERNIE 5 report](https://arxiv.org/html/2602.04705v1).
- **Architecture, objective, optimizer, and precision changes:** LatentMoE, TST/Token Superposition, KDA and sparse-attention designs change model architecture or attention behavior; activation changes are a separate category. Token rounding, Muon, FP4/FP8 output-head changes, and changed quantization recipes are also not drop-in runtime substitutions. These changes require a separate quality/numerical evaluation. Timing alone cannot establish equal-quality efficiency.
- **Shared-expert overlap correction:** the general MoE interface supports deferred combine completion, but the active HybridEP path joins immediately, before shared-expert compute. The earlier claim that this overlap was already active was incorrect for V2. The owner nevertheless rejected shared-expert-only overlap as a direction for this task; see the [accepted scope](sota-ideas/overlap-moe.md#decision).

## Research coverage

The accepted screening covered relevant sections of recent model, pretraining, MoE-system, and kernel reports. It is a bounded literature scan, not a claim that every public paper or report was read in full. Links below are the sources preserved in the screening record when available.

| Area | Covered reports and papers |
| --- | --- |
| DeepSeek | V3/V3.2/V4/V4.1 ([V4](https://arxiv.org/html/2606.19348v1), [V4.1](https://arxiv.org/html/2609.19969v1)). |
| GLM | 4.5/5/5.3 ([GLM 5](https://arxiv.org/html/2602.15763v1)). |
| Kimi | K2/K2.5/K3/K3.1 release ([K3](https://arxiv.org/html/2607.24653v1)). |
| Qwen | 3/Next/3.5–3.8 Next and applicable multimodal/coder reports ([Qwen 3.8 Next](https://arxiv.org/html/2608.30320v1)). |
| MiniMax | 01/M1/M2 series/M3 sparse ([MiniMax-01](https://arxiv.org/html/2501.08313v1), [M2 series](https://arxiv.org/html/2605.26494v2)). |
| Nemotron | Nemotron-H, Nano 2, 3 Nano/Super/Ultra, and 3.5 ([Nano 2](https://arxiv.org/abs/2508.14444)). |
| Microsoft MAI | Thinking 1/Base 1/Base 2 ([MAI announcement](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/), [Thinking 1/Base 1 report](https://microsoft.ai/pdf/mai-thinking-1.pdf)). |
| Additional model reports | MiMo V2.6 ([report](https://arxiv.org/html/2610.11959v1)); Ling/Ring 2.6 ([report](https://arxiv.org/html/2606.15079v1)); LongCat 2 ([report](https://tech.meituan.com/2026/06/30/LongCat2.0.html)); ERNIE 5 ([report](https://arxiv.org/html/2602.04705v1)); Hunyuan A13B ([report](https://arxiv.org/html/2609.27284v1)). |
| MoE systems, kernels, and balancing | ByteDance MegaScale-MoE ([paper](https://chaojin0310.github.io/files/papers/EuroSys26-MegaScale-MoE.pdf)); SonicMoE ([paper](https://arxiv.org/html/2512.14080)); FP8FlowMoE ([paper](https://arxiv.org/html/2511.02302)); ReWY ([paper](https://arxiv.org/abs/2611.01940)); FUSCO ([paper](https://arxiv.org/html/2512.22036v1)); FEPLB ([paper](https://arxiv.org/html/2604.19654v1)); Expert Coupling ([paper](https://arxiv.org/html/2610.09372v1)); PlaceMoE ([paper](https://arxiv.org/html/2610.14798v1)). |

## Pinned implementations and limits

The [machine-readable screening record](../runs/v2-prefetch-baseline/v2-candidate-screening-perfetto.json) is the pin ledger for the inspected FlashQLA, MoonEP, SonicMoE, MegaMoE DeepGEMM, FlashComm, FLUX, Transformer Engine 2.21.0, Linghe, FUSCO, and ReWY revisions. It also records the exact contract and compatibility limit for each pin. No official implementation pin was established there for MAI Base 2, FEPLB, Expert Coupling, or PlaceMoE.

Claims remain bounded by the inspected source revision, hardware, precision, and stated validation. Unverified compilation, SAC behavior, numerical parity, distributed integration, environment requirements, and all-rank skew remain open wherever listed above or in the pin ledger. Trace sums must not be added into a predicted speedup. If future work is authorized, the first source/trace comparisons should focus on FlashQLA, intra-layer MoE overlap, and expert-GEMM scheduling, preserving the accepted baseline settings and validating their full affected forward/backward paths.
