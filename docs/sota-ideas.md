# SOTA runtime ideas: current priorities and historical survey

**Current update:** 16 November 2026; **historical survey date:** 14 November 2026; **research window:** 14 November 2025–14 November 2026, with older predecessors where relevant.
**Status:** This document links the selected task baseline, integrates later measurements, and states candidate directions. It does not authorize source/configuration changes or new execution. The original 14 November survey was read-only at the time it was written.

## Purpose and evidence boundary

This document adapts the owner's accepted SOTA survey into English, then adds current priorities against the selected task recipe. It does not change the accepted workload, reopen completed comparisons, or authorize implementation or execution. The [work plan](../planning/README.md) owns future-work authorization; [candidates](candidates.md) and [profile notes](profile-my-ideas.md) retain earlier inventories and hypotheses.

The reference workload is the agreed eight-H100, NVLink-connected crop: EP8, dense FSDP8, expert FSDP1, PP/TP/CP1; dense and sparse FP8; model and loss compilation; the current SAC policy; MTP1; microbatch 2, sequence length 4096, and accumulation 32. See the [baseline contract](baseline.md) for the complete configuration.

The original survey anchors below come from a V2 rank-0 capture of update 21's first three microbatches, using the old test tokenizer. Its 1293.713 ms device span and 42.899 ms all-device-idle union are historical screening evidence only; they are not current-input performance or an all-rank comparison. Kernel sums and opportunity anchors can overlap other streams and are not removable latency or speedup estimates. The [machine-readable screening record](../runs/v2-prefetch-baseline/v2-candidate-screening-perfetto.json) and [V2 report](../runs/v2-prefetch-baseline/report.md) retain that historical trace context. The accepted baseline and production-data rerun are documented in the [current baseline](baseline.md) and [production-data report](../runs/production-data-abcd/report.md).

The sections below separate the current priority order from the preserved 14 November survey queue. The historical queue's ranks and V2 anchors are not today's performance ranking.

## Current priorities (16 November 2026)

| Priority | Direction | Why it leads | Main evidence gate |
| --- | --- | --- | --- |
| 1 | Retain useful forward results to reduce selective activation checkpointing (SAC) replay; consider packed FP4 activation storage as an extension | Current measurements show memory headroom, while SAC replays expensive forward work | Measure peak memory, cache lifetimes, all-rank routing skew, and the full training path |
| 2 | Overlap dispatch, expert compute, and combine at fine granularity within each MoE layer | The current path has sequential device work in these families | Preserve FP8 forward/backward, router gradients, SAC, and measure full MoE resource contention |
| 3 | Evaluate a Hadamard codec at combine | It could reduce a confirmed communication bottleneck | Confirm actual communication budget and validate numerics, router gradients, and reductions |

### 1. Memory headroom to reduce SAC recomputation

**[IMPORTANT] Memory optimization to retain useful forward results and remove expensive SAC recomputation is a particularly promising direction and the first priority.**

Selectively retain expensive forward results while keeping checkpointing where it remains useful. The active performance path includes DeepGEMM expert kernels, CuTe normalization kernels, and the FlashQLA/TileLang scan; current evidence does not identify generic `torch.compile` kernel quality as the main bottleneck.

#### Measured headroom and Quack component evidence

In the SE 140-update timing run, peak allocated/reserved memory was 66.924/68.334 GiB, versus 70.387/71.670 GiB for the A reference: 3.463 GiB less allocated and 3.336 GiB less reserved. These are run-level allocator measurements, not a guarantee of reusable memory at every phase.

The isolated [Quack LM-head report](../runs/quack-lmhead/report.md) measured the pinned Quack `bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8` on BF16, D=4096, V=128256, B=2, S=4096 main+MTP. Patched Q4096 reduced component peak allocated memory from 9.370 to 6.251 GiB (3.119 GiB, 33.3%) and reserved memory from 9.943 to 7.109 GiB. It showed no robust direct latency gain. Q8192 had about 1% lower paired mean latency, still provisional. Both variants had larger local gradient error than native against FP32; neither case measures full-training memory or convergence. The Q4096 task-local scale-axis fix changed metadata interpretation only, not arithmetic. Inputs came from a real capture, but the norm/MTP/head modules were canonically reinitialized; this is not an original trained-checkpoint result.

As a conditional budget illustration, if roughly 2 GiB of Quack's component allocation reduction transferred to the run and added cleanly to the 3.463 GiB already freed versus A, the total would be about 5.46 GiB. That additivity and global-peak reduction have not been measured.

#### Selective SAC proposal and memory estimate

The current policy saves `experts_up` on layers 2–4 and 6–7, and `experts_up` plus `comm` on layer 5; layers 0–1 recompute. A proposed policy is shown for discussion only and has not been applied:

```yaml
module_modes:
  - 'layers.0=selective'
  - 'layers.[12456]=save:experts_up+comm'
  - 'layers.[37]=save:experts_up+comm+attn'
  - 'mtp.layers.0=save:attn'
```

These rules use the existing [SAC categories and first-match policy](../gift/torchtitan/distributed/activation_checkpoint.py).

Under balanced receiver-local R=65,536–67,456, the estimated additional retained storage is:

| Retained results | Estimated additional storage | Scope |
| --- | ---: | --- |
| `comm` on layers 1, 2, 3, 4, 6, 7 | 3.61–3.66 GiB | Cache FP8 tokens/scales/probabilities, quantized expert weights, and BF16 combine values; do not double-count a logical expanded BF16 dispatch matrix |
| Layer 1 `experts_up` | 0.375–0.386 GiB | Layer 1 is not currently saved |
| MLA attention results on layers 3 and 7 plus MTP | about 0.19 GiB | Retain only these attention results |
| Full proposed set | 4.18–4.23 GiB | Estimated retained tensor sizes, not measured CUDA peak |

A first subset of four `comm` results plus layer 1 `experts_up` and the listed attention results is estimated at 2.97–3.01 GiB. That is below the observed 3.463 GiB allocated reduction versus A, so it is a plausible first measurement, not proof that global peak memory will fall by the same amount. The estimates assume balanced receiver-local counts; measure cache lifetime and routing skew across all eight ranks. R is local to a receiving rank, not the global token count, and accumulation 32 does not multiply saved-activation memory by 32.

Layer 5 already saves `experts_up` and `comm`, so its outer replay has no dispatch, combine, or down work; the other six MoE layers still replay those regions. This suggests a saved combined output could avoid regenerating the down path, but that causal explanation is not a controlled policy A/B. Mandatory gradient communication remains. Saving GDN attention is estimated at about 0.1875 GiB per layer against an outer-only replay target of 1.6–1.85 ms per three microbatches; FLA's internal VJP reconstruction remains. `save:mm` does not catch the active `fp8_dense_gemm` path and needs a new category mapping. `save:experts` would also retain down outputs (about 0.5 GiB per layer); its benefit is not additive with `comm`, so it is not the first choice.

#### SAC-removal trace: conditional arithmetic, not measured speedup

The [reproducible estimator](../runs/schedule-embedding/trace-analysis/sac-removal-estimate.py) and [machine-readable result](../runs/schedule-embedding/trace-analysis/sac-removal-estimate.json) use update 21, ranks 0 and 1, and the first three of 32 microbatches. The selected core-MoE set has 115 device events per microbatch. Across the three sampled microbatches, its union and exclusive time are both 59.174388 ms on rank 0 and 58.829261 ms on rank 1; no other device event overlaps these selected intervals in this capture. Adding MLA attention for layers 3 and 7 and MTP gives 121 selected events per microbatch, union 65.733410/65.382568 ms and exclusive time 63.646110/62.099197 ms on ranks 0/1. Exclusive time subtracts all unselected kernel, copy, memset, and NCCL device events.

The estimator uses baseline 12.8479317002 s/update and 2,097,152 physical tokens/update. For each rank, it computes `removal_s = exclusive_ms_for_3_microbatches / 3 * 32 / 1000`, then subtracts that from the baseline and divides tokens by the conditional update time:

| Fixed-timeline removal set | Conditional removal | Conditional update | Conditional TPS | Change from baseline |
| --- | ---: | ---: | ---: | ---: |
| Core MoE | 0.6275–0.6312 s | 12.2167–12.2204 s | 171,610–171,662 | +5.13–5.17% |
| Core MoE plus MLA attention | 0.6624–0.6789 s | 12.1690–12.1855 s | 172,102–172,335 | +5.44–5.58% |

This is fixed-timeline interval removal, not a measured speedup or exact critical path. Complete host/CUDA dependencies and peer waits were not resimulated; removing work can change scheduling, contention, and waiting. The extrapolation assumes all 32 microbatches and other ranks match the two-rank sample. The optimizer is outside the capture, while the baseline update time includes it. These are conditional estimates for prioritization; they are not mathematical bounds on a modified run or an established 6% gain.

#### Packed FP4 activation storage as a memory-direction extension

The [MXFP4-Hopper paper](https://arxiv.org/html/2603.02731v1) reports 671B training on 256 Hopper GPUs:

| SAC setting | FP8 tokens/GPU/s | MXFP4 tokens/GPU/s |
| --- | ---: | ---: |
| Same aggressive recomputation | 1,157 | 1,156 |
| Reduced recomputation | OOM | 1,302 |

The roughly 12.5% advantage is a memory-to-less-recomputation comparison. GEMMs remain FP8; packed FP4 is used for activation storage/dispatch. Its 16B-model, 160B-token quality evidence is separate from the 671B systems benchmark.

The public [MXFP4-Hopper implementation pin](https://github.com/anonymous-git-2026/MXFP4-Hopper/tree/fede2e1594fe0302570189e877bfa40fd1c5119c) is incomplete as a training wrapper and retains BF16 materializations. A local implementation needs integration with backward, router gradients, and SAC, plus numerical validation and convergence evidence.

### 2. Fine-grained dispatch/compute/combine overlap

The accepted direction is a full-streaming MoE path: publish readiness as input arrives, overlap dispatch with FC1/SwiGLU/FC2, then publish outputs for combine. The detailed [MoE overlap analysis](sota-ideas/overlap-moe.md) already selects this kernel-level scope. Coarse Python token chunking and shared-expert-only overlap remain outside the selected direction. The pinned [FlashComm implementation](https://github.com/ByteDance-Seed/Triton-distributed/tree/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/FlashComm) is an SM90 BF16/FP16 donor, not a current blockscaled-FP8 full-training backend. Any implementation must preserve the existing FP8 recipe, router gradients, SAC behavior, and full forward/backward semantics. Communication and compute compete for SM and HBM resources, so evaluate the whole MoE forward/backward path under SAC.

### 3. Hadamard codec at combine, after confirming budget

[End-to-end 4-bit training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/) motivates rotating before quantization. A candidate combine codec would rotate, quantize to INT8/FP8, transmit, decode into FP32 accumulation, then apply the inverse Hadamard transform; the ordering and numerical behavior need validation. The current dispatch already uses FP8 and combine uses BF16. The [TACO paper](https://arxiv.org/abs/2604.24088) and pinned [COCCL implementation](https://github.com/hpdps-group/COCCL/tree/aa10c6bb802771fa79d5f15ea0cac85cdd9616d2) describe adaptive-scale Hadamard dual-scale FP8 and demonstrate TP training on H100. That is not a direct fit for this TP1 HybridEP path, which bypasses NCCL; the potential EP32 inter-node benefit is also not established for current EP8. The Blackwell S2F6/BBBK implementation does not establish an H100 codec or training path. Gate further work on measured communication budget, code fit, numerical correctness, router/probability gradients, and reduction-scale evidence.

## Historical 14 November research queue

The ranks and trace anchors in this section preserve the original bounded survey. Its V2 performance anchors use the old test tokenizer and must not be treated as current-input performance or the present priority order.

### Ranked research queue

| Rank | Direction | Existing trace anchor | Main gate |
| --- | --- | ---: | --- |
| 1 | Pipeline dispatch, expert GEMM, and combine within each MoE layer | EP union 157.627 ms; actual FP8 expert-GEMM kernel sum 168.435 ms | Selected scope: fine-grained kernel overlap; preserve blockscaled FP8 and full training semantics; substantial kernel/backend development is required |
| 2 | FlashQLA for GDN | Historical V2 scan-kernel anchor: 58.979 ms | `external_gate` is selected in the current task recipe and has local task-copy evidence; CP warmup, convergence, checkpoint/resume, and production PP4/EP32 remain open |
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

The historical V2 scan-kernel anchor is 30.245 ms forward plus 28.734 ms backward, or 58.979 ms total. The 8.868 ms extra VJP-forward kernels are already inside the forward total and must not be added again. QKV convolution/unpack is another 17.013 ms anchor; changing the scan does not automatically remove it. These are old-tokenizer screening anchors, not current-input performance.

The owner has since selected `external_gate` for the current task recipe, and the task-copy implementation has local component and eight-rank training evidence. The selection is recipe-scoped and does not establish production validation or a general throughput gain; see the [FlashQLA report](../runs/flash-qla/report.md) and [current baseline](baseline.md). CP warmup, long-run convergence, checkpoint/resume, EMA updates, and production PP4/EP32 remain open gates. Earlier old-tokenizer performance values do not replace the current production-data rerun.

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

The [Nemotron Nano 2 report](https://arxiv.org/abs/2508.14444) and Megatron's inspected `fp8_param_gather` path motivate evaluating native FP8 parameter storage/gather only after the selected GFSDP path is measured. Keep FP32 master parameters and optimizer state. Gate any representation change on the actual FP32→FP8 versus FP32→BF16→FP8 rounding and its effect on training. This is not a proposal to lower FP32 gradient-reduction precision. The incremental benefit in the current recipe is unknown, and a 2025 fully-FP8 training reference retained a BF16 final projection.

### 10. FP8 weight cache across 32 microbatches

The historical first-three-microbatch V2 trace records 9.246 ms of dense weight block quantization, about 0.7% of its 1293.713 ms window. This is an old-tokenizer anchor, not guaranteed recoverable time. A cache would need to refresh quantized weights, scales, and layouts after each optimizer update; expert-weight caching has a separate lifecycle and cost. Account for extra memory. At the inspected Transformer Engine revision, its named FP8 weight cache is disabled for FSDP2, which is not proof of compatibility with the selected GFSDP recipe. This remains a historical tenth-ranked candidate pending a current cost estimate.

## Screened directions and current-source dispositions

These notes prevent future work from treating already-present mechanisms or rejected evidence as new opportunities.

**Clarification (16 November 2026):** the four explicit-kernel fusion dispositions below are historical results for those candidate experiments. The current selected combined-four foundation separately includes BF16 expert-weight reuse, the explicit routing-metadata kernel, and stable cap-zero CE through the existing `CrossEntropyLoss`/`ChunkedCELoss` path, alongside the accepted V2 CPU-prefetch setting. See the [current baseline](baseline.md#current-final-task-baseline) and [combined-four report](../runs/combined-four/report.md). This distinction preserves the original fusion results without implying that the current foundation is unselected.

- **Four fusion experiments:** all four explicit-kernel candidates were measured in their assigned scope; none was accepted or enabled. GPU input preparation and routing metadata improved their local regions, but their training-throughput comparisons were inconclusive. Cap-zero CE improved local CE timing, but the whole-training throughput result remained inconclusive. The active 1,024-token-tile linear-CE port regressed the whole head and training throughput and was rejected. The [four-fusion report](../runs/four-fusion/report.md) owns all results and caveats.
- **SonicMoE:** its inspected implementation is an H100 BF16 training/I/O reference, not an established FP8 EP path. Current task source already avoids saving expert outputs solely for router gradients: probability is computed inside SwiGLU and `fused_swiglu_quantization` exposes `grad_probs`. Token rounding changes router behavior and is excluded. The task HEAD (`657e77a4f8ddb116b75925e5671b00642a0e4d32`) is not proof that the V2 execution used identical source.
- **FUSCO:** its permutation–communication fusion concept is relevant, but HybridEP already has an optional `fuse_permute` path. The inspected [FUSCO source pin](https://github.com/infinigence/FUSCO/tree/dd3f46113c1cf40dbebd426f2f3557e31a16b7fe) has no public CUDA runtime source; `libfusco.so` is required, float8 is rejected, and the weighted-combine probability-gradient wrapper has a gap. The paper is [here](https://arxiv.org/html/2512.22036v1).
- **ReWY:** the inspected equal-head API does not fit this model's 16/32 key/value head dimensions, and distributed compilation is unvalidated. The [paper](https://arxiv.org/abs/2611.01940) does not remove those implementation gates.
- **FP8FlowMoE:** mechanisms attributed to this work are already active in the current trace/source. It is not a new independent candidate without a specific missing mechanism and measured boundary; see the [report](https://arxiv.org/html/2511.02302).
- **Chunked output projection/CE:** the current workload already uses two loss chunks. The prior linear-CE port regressed and was rejected. The eager BF16 LM head remains a large 203.602 ms forward/backward GEMM anchor, but the historical survey established no compatible accelerator or CE substitution. The later isolated [Quack component study](../runs/quack-lmhead/report.md) reduced allocated memory without robust latency gain; it does not establish an accepted whole-training CE replacement. The [GLM 5 report](https://arxiv.org/html/2602.15763v1) is architecture/research context, not a validated alternative implementation.
- **MLA backend:** the active cuDNN flash SDPA path has 29.969 ms of kernels; this does not establish a faster fallback or an MLA replacement's end-to-end value. Source inspection found the current MLA scaling, mask, and boundary contracts consistent with the active row-wise causal path, but full numerical forward/backward parity remains untested. FlashMLA's current code removed Hopper support, and newer inference MLA mechanisms are not training drop-ins.
- **Memory offload / allocator-directed movement:** ERNIE VMM allocator-offload ideas require evidence of relevant memory pressure. The V2 capture does not provide that evidence, so the direction is not prioritized from this trace. See the [ERNIE 5 report](https://arxiv.org/html/2602.04705v1).
- **Architecture, objective, optimizer, and precision changes:** LatentMoE, TST/Token Superposition, KDA and sparse-attention designs change model architecture or attention behavior; activation changes are a separate category. Token rounding, Muon, FP4/FP8 output-head changes, and changed quantization recipes are also not drop-in runtime substitutions. These changes require a separate quality/numerical evaluation. Timing alone cannot establish equal-quality efficiency.
- **Shared-expert overlap correction:** the general MoE interface supports deferred combine completion, but the active HybridEP path joins immediately, before shared-expert compute. The earlier claim that this overlap was already active was incorrect for V2. The owner nevertheless rejected shared-expert-only overlap as a direction for this task; see the [accepted scope](sota-ideas/overlap-moe.md#decision).

## Research coverage

The 14 November screening covered relevant sections of recent model, pretraining, MoE-system, and kernel reports. It is a bounded literature scan, not a claim that every public paper or report was read in full. Links below are the sources preserved in the screening record when available; the 16 November references are listed separately below.

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
| 16 November memory/communication addendum | MXFP4-Hopper ([paper](https://arxiv.org/html/2603.02731v1), [pinned code](https://github.com/anonymous-git-2026/MXFP4-Hopper/tree/fede2e1594fe0302570189e877bfa40fd1c5119c)); TACO ([paper](https://arxiv.org/abs/2604.24088), [pinned COCCL code](https://github.com/hpdps-group/COCCL/tree/aa10c6bb802771fa79d5f15ea0cac85cdd9616d2)); Hadamard-rotation motivation ([article](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)). |

## Pinned implementations and limits

The [machine-readable screening record](../runs/v2-prefetch-baseline/v2-candidate-screening-perfetto.json) is the historical pin ledger for the inspected FlashQLA, MoonEP, SonicMoE, MegaMoE DeepGEMM, FlashComm, FLUX, Transformer Engine 2.21.0, Linghe, FUSCO, and ReWY revisions. It also records the exact contract and compatibility limit for each pin. No official implementation pin was established there for MAI Base 2, FEPLB, Expert Coupling, or PlaceMoE. The 16 November section above owns its added MXFP4-Hopper and COCCL pins; they are not entries in that earlier ledger.

Claims remain bounded by the inspected source revision, hardware, precision, and stated validation. Unverified compilation, SAC behavior, numerical parity, distributed integration, environment requirements, and all-rank skew remain open wherever listed above or in the pin ledger. Trace sums must not be added into a predicted speedup. As of 16 November, the priority order is memory-saving selective SAC, fine-grained intra-layer MoE overlap, then a Hadamard combine codec after a substantial communication cost is confirmed. FlashQLA `external_gate` is already selected and locally tested; its production validation gates remain open. Any authorized work must preserve the accepted baseline settings and validate the full affected forward/backward paths.
