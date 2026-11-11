# GIFT 4 implemented optimization inventory

- **Source snapshot:** `f840cf25e4294f93a5c91a20707c30b97e3c1dbf`
- **Task revision:** `c6eb0f83fdf535c206840b116053161771c3fca7`
- **Inventory date:** 2026-10-27

This is a source-grounded list of existing GIFT options and automatic kernels that can inform tests for the current target: GFSDP, FP8, model and loss compilation, and SAC/full activation checkpointing. Small models on one or two nodes are the initial focus; PP is deprioritized.

“Ready to test” means the option is implemented and exposed or its supported path is clear in this source snapshot. It does not mean the option has been run on GPU, is numerically equivalent, or improves end-to-end throughput. No performance claims are made here.

The existing torch-free configuration check covered 29 cases: 26 accepted and 3 expected rejections. It checks schema decoding, materialization, and round trips only; it does not check installed dependencies, kernel compatibility, runtime, or speed. The rejected combinations include Flex and sparse FP8 with non-blocking capacity or without fused permutation. See [the validation record](evidence/gift-4-optimization-config-checks.json) for commands and cases.

## Stock configuration status

These are the checked-in starting points, not performance results. The 3B HybridEP example shows an architecture/config fragment and is not a complete training recipe.

| Configuration | Existing settings | Reference |
|---|---|---|
| GIFT 4 debug | FSDP2; model compilation off; FP8 off; SAC. | [gift_4_debug.yaml](../gift/configs/experiments/gift_4_debug.yaml) |
| GIFT 4 700B | FSDP2 with CPU offload; EP=32 with DeepEP; full activation checkpointing; loss-only compilation; dense and sparse FP8 enabled via `fp8_full`; shared dense quantizer and both fused SwiGLU/quant flags enabled; CuTe norm off. | [experiment](../gift/configs/experiments/gift_4_700b.yaml), [GIFT 700B training preset](../gift/configs/presets/train/gift_700b.yaml), [full FP8 preset](../gift/configs/presets/train/fp8_full.yaml) |
| 3B HybridEP example | MLA-only architecture example; not a full GIFT 4 recipe. | [example config](../gift/configs/experiments/gift_moe_3b_hybridep_dense_fp8_compile_non_blocking.yaml) |

## Candidate options

These candidates retain the existing model, loss, and data contract unless a risk below says otherwise. Keep model and loss compilation and the selected FP8 and SAC/full settings in comparisons.

| Candidate | Setting and current default | Fit and cost to examine | Source |
|---|---|---|---|
| CuTe self-gated RMSNorm | `norm_backend: cute`; default is `compile` (native PyTorch, eager unless the enclosing model is compiled). Set `model.config.norm` and the applicable block norms. | Hopper SM90, BF16 activations, gate rank 16, and widths divisible by 128 are supported; GIFT widths fit. CuTe preserves the mathematical objective but changes floating-point rounding through BF16 gate math versus native FP32, so compare numerical behavior and convergence. Optional CUTLASS/TVM dependencies and SAC recomputation may affect fit and cost. Details below. | [schema](../gift/torchtitan/config/schema/models/common/norms/gated_norm.py), [CuTe op](../gift/torchtitan/models/common/norms/cute_gated_norm/op.py) |
| MLA FlashAttention backend | `model.config.blocks.{mla_moe,mtp}.attention.inner_attention.type: flash_fa2`, `flash_fa3`, or `flash_fa4`; default SDPA selects an optimized CUDA backend at runtime. | FA3 targets Hopper; FA4 needs the optional CuTe package. GIFT MLA Q/K width 192 and V width 128 are not covered by the existing equal-width 64 tests. Check actual wrapper support, compilation, SAC, and memory before measuring. | [FlashAttention wrapper](../gift/torchtitan/models/common/attention/softmax/kernels/flash_base.py), [kernel schema](../gift/torchtitan/config/schema/models/common/attention/softmax/kernels/) |
| FP8 dense quantizer sharing | `fp8.dense_use_shared_quantizer: true`; default `false`. | Reuses the quantized input for dense SwiGLU `w1` and `w3`. Verify activation and numerical effects under the selected FP8 recipe. | [quantization utilities](../gift/torchtitan/components/quantization/utils.py), [FFN](../gift/torchtitan/models/common/ffn/__init__.py) |
| FP8 fused SwiGLU/quantization | `fp8.dense_fused_swiglu_quant: true` and `fp8.sparse_fused_swiglu_quant: true`. | Compare fused and unfused paths with the same recipe. Intermediate rounding differs. Keep fused behavior disabled if hidden gate/up `zloss` is required. Sparse FP8 requires fused permutation in HybridEP. | [FFN](../gift/torchtitan/models/common/ffn/__init__.py), [MoE experts](../gift/torchtitan/models/common/moe/experts/gmm.py), [FP8 grouped GEMM](../gift/torchtitan/ops/float8/grouped_gemm.py) |
| HybridEP dispatch/combine overlap | `model.config.blocks.{gdn_moe,mla_moe}.moe.experts.token_dispatcher.comm_stream: true`; default `false`. | Can overlap dispatch with expert-weight quantization. Measure communication and compute overlap on the target topology. `fuse_permute: true` folds permutation/unpermutation and is required for sparse FP8. | [HybridEP schema](../gift/torchtitan/config/schema/models/common/moe/dispatch/hybridep.py), [dispatcher](../gift/torchtitan/distributed/deepep/hybridep.py), [overlap tests](../gift/tests/unit_tests/test_ep_overlap.py) |
| HybridEP SM counts and padding | `num_sms_dispatch` and `num_sms_combine` default to 16; `fp8.sparse_deep_gemm_num_sms` defaults to 116 (valid range 1–132); `pad_multiple` defaults to `null`. | Tune dispatch/combine SM counts jointly with the sparse DeepGEMM SM budget and measure the token-count distribution. Sparse FP8 requires padding to a multiple of 128. Reserve HybridEP's maximum-token buffer before training. | [HybridEP schema](../gift/torchtitan/config/schema/models/common/moe/dispatch/hybridep.py), [dispatcher](../gift/torchtitan/distributed/deepep/hybridep.py) |
| HybridEP capacity factor | `non_blocking_capacity_factor: 1.0`; default `null`. | Avoids a CPU token-count synchronization with a conservative no-drop capacity bound for BF16 experts. Sparse FP8 rejects this combination because of its NaN tail; dense FP8 can remain enabled. Values below 1.0 drop routes and change behavior. The 3B example is an MLA-only architecture, not a complete training recipe or GIFT 4 evidence. | [HybridEP schema](../gift/torchtitan/config/schema/models/common/moe/dispatch/hybridep.py), [dispatcher](../gift/torchtitan/distributed/deepep/hybridep.py), [example config](../gift/configs/experiments/gift_moe_3b_hybridep_dense_fp8_compile_non_blocking.yaml) |
| Chunked cross entropy | `loss.num_chunks`; default 8. Try 4 and 16. | Trades peak memory against launch/loop overhead. It preserves the objective and MTP head count; GIFT already batches MTP heads and shares the LM head once per chunk. Preserve GFSDP flags. | [loss](../gift/torchtitan/components/loss.py), [loss schema](../gift/torchtitan/config/schema/components/loss.py), [tests](../gift/tests/unit_tests/test_chunked_loss_fsdp_protocol.py) |
| GFSDP parameter resharding | `parallelism.fsdp_reshard_after_forward: never`; without PP, the default reshard-after-forward behavior applies. | Can avoid backward weight regathers, at the cost of retaining gathered weights and increasing memory pressure. Measure at the same memory lifecycle point. | [GIFT parallelization](../gift/torchtitan/models/gift_moe/_base/parallelize.py), [FSDP wrapper](../gift/torchtitan/distributed/fsdp.py), [GFSDP state](../gift/torchtitan/distributed/gfsdp/_state.py) |
| BF16 Adam states | `optimizer.implementation: fused_opt_states_bf16`; default `fused`. | Stores Adam moments in BF16 while parameters and gradients remain FP32. This changes numerical behavior; run as a separate quality and checkpoint/resume experiment, not as a presumed-equivalent speed switch. Combined GFSDP runtime and resume behavior are not established by the config check. | [optimizer docs](../gift/docs/bf16_optimizer_states.md), [optimizer](../gift/torchtitan/components/optimizer.py), [optimizer schema](../gift/torchtitan/config/schema/components/optimizer.py) |
| Activation checkpoint budget mode | `activation_checkpoint.mode: memory_budget`; `activation_checkpoint.memory_budget` defaults to 0.5. Requires model compilation. `early_stop` defaults to `false`. | Existing full/selective checkpointing is supported. Budget mode changes the recomputation schedule and must not silently replace the agreed full/SAC baseline. Check replay side effects before enabling early stop. | [activation checkpointing](../gift/torchtitan/distributed/activation_checkpoint.py), [tests](../gift/tests/unit_tests/test_activation_checkpoint.py) |
| Data-loader workers and prefetch | `dataloader.num_workers` defaults to 0; `persistent_workers` and `pin_memory` default to `false`; `prefetch_factor` defaults to `null`. | These are ready loader controls to measure against host input wait and memory use. Do not assume asynchronous H2D overlap in the GFSDP path. Pretokenized streaming MDS avoids online tokenization; its predownload/cache controls address input I/O. Preserve corpus identity, tokenization, and ordering when comparing loaders. | [dataloader schema](../gift/torchtitan/config/schema/components/dataloader.py), [HF text dataset](../gift/torchtitan/datasets/hf/text_datasets.py), [streaming text loader](../gift/torchtitan/datasets/streaming/text/dataloader.py) |

### CuTe norm coverage

Set `norm_backend: cute` on the top-level `model.config.norm` and, where present, `attention_norm`, `ffn_norm`, `post_attn_norm`, and `post_ffn_norm` in `gdn_dense`, `gdn_moe`, `mla_moe`, and `mtp` blocks. Set `attention.kv_norm` and `attention.q_norm` for MLA/MTP where configured; `q_norm` applies only to 3B and 700B variants. GDN `attention.o_norm` is part of a different fused FLA path.

The implementation targets SM90 and BF16, gate rank 16, and widths divisible by 128. It preserves the mathematical objective and the `zero_centered`, epsilon, initialization, and gate settings; verify the resolved config. The native norm computes gate math in FP32, while CuTe uses BF16, so rounding changes. The SAC `MUST_SAVE` list does not include the CuTe norm custom op, so checkpoint replay may recompute it. See [norm config](../gift/torchtitan/config/schema/models/common/norms/gated_norm.py), [CuTe implementation](../gift/torchtitan/models/common/norms/cute_gated_norm/op.py), and [unit tests](../gift/tests/unit_tests/test_cute_gated_norm.py).

### Attention backend coverage

The option is under `model.config.blocks.{mla_moe,mtp}.attention.inner_attention`. SDPA is already a CUDA-optimized runtime choice. FA2/FA3 are saved by SAC; FA4's custom-op checkpoint handling is marked TODO, so its SAC compatibility is unconfirmed. Existing FlashAttention tests use equal Q/K/V widths of 64, unlike GIFT MLA's Q/K 192 and V 128. Verify the actual external wrapper and installed package for the GIFT shape before any runtime claim. Packed backends include `varlen`, `flash_varlen_fa2`, `flash_varlen_fa3`, and `flash_varlen_fa4`; a variable-length config exists at [gift_3_5_debug_varlen.yaml](../gift/configs/experiments/gift_3_5_debug_varlen.yaml). Switching from dense causal to block causal attention changes document boundaries; compare only with matching masks and data. See [FlashAttention wrapper](../gift/torchtitan/models/common/attention/softmax/kernels/flash_base.py) and the [kernel schema](../gift/torchtitan/config/schema/models/common/attention/softmax/kernels/).

## Other implemented runtime controls

| Area | Existing controls | Notes |
|---|---|---|
| DeepGEMM warmup/cache | `fp8.deep_gemm_warmup_on_start: true`; `DG_JIT_SHARED_CACHE_DIR`, `PROPAGATE_DG_CACHE`, `PROPAGATE_DG_CACHE_DIR`. | Can manage startup compilation/cache costs. Warmup budget must match SM count. Report startup separately; this is not evidence of faster steady-state steps. See [DeepGEMM warmup](../gift/torchtitan/ops/warmup/deep_gemm_fast_warmup.py) and [cache manager](../gift/torchtitan/ops/warmup/cache_manager.py). |
| Logging and monitoring | `debug.enable_structured_logging`; `metrics.log_freq` (default 10); `callbacks.<monitoring-entry>.log_interval` for hook-based sources. | Preserve required observability. When monitoring uses module hooks, logged steps run the model blocks eagerly. With `log_interval: 1`, every step runs uncompiled; sources without module hooks do not open this eager window. See [monitoring callback](../gift/torchtitan/callbacks/builtin/monitoring.py). |
| FLA TileLang backend | The default enables TileLang on NVIDIA Hopper with Triton ≥3.4 when available; an explicit `FLA_TILELANG` value other than `0` enables it. | There is a known GVA backward issue on Hopper with Triton versions `[3.4, 3.7.1)` without TileLang. Do not recommend disabling it blindly. See [backend selection](../gift/contrib/flash-linear-attention/fla/ops/common/backends/tilelang/__init__.py) and [linear-attention tests](../gift/tests/unit_tests/test_linear_attention.py). |

The following lower-level kernel arguments exist but are not wired as GIFT YAML/environment controls: CuTe gated norm forward accepts internal `num_sms`, `n_hc`, and `f2_pdl` kwargs, but the public op does not pass them; the `row2col` Triton alternative is a method argument only. Treat these as source-level implementation details, not ready model-config switches. The active 64-token FLA chunk path also bypasses `FLA_USE_TMA` and `FLA_TRIL_PRECISION`; `FLA_FLASH_QLA` rejects mandatory `use_gate_in_kernel`.

## Already automatic: preserve and account for these

- **GDN / FLA:** packed QKV projection; fused causal convolution, SiLU, unpack, and QK norm; fused gate and inscan; fused KKT and triangular solve at chunk size 64; fused FLA output norm. There is no public chunk-size control. The GDN backward wrapper recomputes inner FLA forward; SAC saves convolution and scan outputs. See [FLA custom ops](../gift/torchtitan/models/common/attention/linear/fla_custom_ops.py).
- **MoE:** BF16 experts pack `w1`/`w3` and use one grouped GEMM. The BF16 path named `fused_swiglu` is PyTorch code, not a custom kernel. DeepEP combine overlaps shared experts. Quantization before dispatch is implemented; overlapping that quantization with dispatch remains TODO. Router top-k uses `torch.topk`; it is not a special Triton kernel, and SAC saves its choices. See [expert implementation](../gift/torchtitan/models/common/moe/experts/gmm.py), [grouped FP8 GEMM](../gift/torchtitan/ops/float8/grouped_gemm.py), and [HybridEP dispatcher](../gift/torchtitan/distributed/deepep/hybridep.py).
- **Distributed/runtime:** GFSDP buffer reuse and EP prefetch are automatic; there is no YAML prefetch-distance knob. AG and RS share one stream; concurrent AG/RS execution is not implemented through separate streams. HybridEP reserves its maximum-token buffer before training. See [GFSDP resharding](../gift/torchtitan/distributed/fsdp.py) and [GFSDP state](../gift/torchtitan/distributed/gfsdp/_state.py).
- **Other:** fused AdamW is the default implementation; complex RoPE caching is automatic and there is no fused-rotary option.

## Behavior-changing controls and out-of-scope paths

- MoE token-drop policies and group-limited routing change which routes or experts are eligible. Do not treat them as speed-only candidates. `branchless_capacity_drop: true` is an implemented Triton selector for the same configured token-drop policy and defaults to `false`; confirm policy and drop counts. See [token-drop implementation](../gift/torchtitan/ops/token_drop.py) and [MoE config](../gift/torchtitan/config/moe.py).
- Skipping weight-gradient computation only applies when the model is already frozen.
- GraphTrainer/CUDA graphs, offload, and bucketing are limited to graph model families and do not cover GIFT 4. Flex block mask is incompatible with GDN attention modes `NONE` and `VARLEN`. See [config builder](../gift/torchtitan/config/build.py) and [decoder schema](../gift/torchtitan/config/schema/models/common/decoder_blocks/decoder.py).
- GFSDP2 rejects CPU offload, symmetric memory, TP, asynchronous checkpointing, and Hugging Face checkpoint I/O. See [sharding backend schema](../gift/torchtitan/config/schema/distributed/sharding_backend.py).
- Optimizer-in-backward rejects EP > 1 and PP > 1, and changes gradient clipping/accumulation behavior. It is not a drop-in candidate for the target setup.
- HybridEP sparse FP8 requires compatible widths: 256 and 1280 are not divisible by 512; 700B width 7168 is. DeepEP rejects model compilation. GQA fused QKV is the wrong architecture for GIFT 4.
- Inference context parallelism and YCCL have no training-time configuration activation. PP-only controls remain deprioritized for this task.

## Suggested first measurements

Within the chosen small-model preset and fixed topology, a compact first pass is:

1. CuTe norm, only after checking all applicable norm fields and accepting an explicit numerical comparison.
2. Explicit FA3 on Hopper, only after confirming the external wrapper supports GIFT MLA's Q/K 192 and V 128 shape under SAC and compile.
3. HybridEP `comm_stream` and SM-budget tuning for a HybridEP-compatible setup; test sparse FP8 only with fused permutation and 128-token padding.
4. `loss.num_chunks` at 4, 8, and 16; then `fsdp_reshard_after_forward: never` if memory permits.
5. FP8 shared quantizer and fused SwiGLU/quantization as separate switches, holding the FP8 recipe fixed.
6. BF16 optimizer states as a separate numerical and resume-compatibility study.

These are test candidates, not a ranked speedup forecast. Resolve the target preset, exact model size, hardware, and production FP8 recipe before interpreting any result. A 700B GIFT 4 configuration adapted to GFSDP2, CPU offload off, HybridEP 128, and model+loss compilation is only a config-validation specimen; it is not the small-model production baseline or evidence of a hardware allocation.
