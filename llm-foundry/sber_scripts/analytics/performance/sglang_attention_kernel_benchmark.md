# SGLang attention kernel benchmark

Файлы:

- `run_sglang_attention_kernel_benchmark.sh` - единственная поддерживаемая точка входа для job-запуска. Это конфигурационный файл: размеры, TP, image, queue и наборы cases правятся прямо в нем.
- `sglang_attention_kernel_benchmark.py` - Python benchmark, который запускается внутри job image. Он импортирует `sglang`/`sgl_kernel` из активного окружения image и не загружает веса модели.
- `sglang_attention_kernel_job.py` - submit/monitor wrapper для `InfiniteJob`.

Benchmark измеряет attention-layer-like path на случайных tensors. Это не полный inference server: нет загрузки весов, tokenizer/scheduler, embedding, MLP/MoE и logits. E2E здесь означает sequence внутри attention слоя: chunked prefill/extend, запись KV/state, затем последовательный decode.

## Запуск

По умолчанию entrypoint запускает все основные suite:

```bash
bash sber_benchmarks/run_sglang_attention_kernel_benchmark.sh
```

Это эквивалентно case `all`. Он добавляет восемь benchmark-команд в один job:

- `all-generic-sparse`: generic sparse FlashAttention probes.
- `all-linear`: Qwen3.5 Gated Delta Net probes and E2E.
- `all-dsv4-pro-kernels`: DeepSeek V4 standalone sparse/FlashMLA/compression probes.
- `all-dsv4-pro-none-e2e`: DeepSeek V4 sparse/SWA-only E2E without CSA/HCA compression.
- `all-dsv4-pro-csa-e2e`: DeepSeek V4 DSA E2E with CSA/C4 compression.
- `all-dsv4-pro-hca-e2e`: DeepSeek V4 DSA E2E with HCA/C128 compression.
- `all-deepseek-v3-mla-e2e`: DeepSeek V3 latent MLA E2E.
- `all-glm-dsa-e2e`: GLM-5.2 DSA E2E with production indexer/top-k sparsity.

Доступные cases:

- `all` или `full-all`
- `list`
- `e2e-all`
- `normalized-e2e-all`
- `linear-qwen35-e2e`
- `linear-qwen35-all`
- `deepseek-v3-mla-e2e`
- `glm-dsa-e2e` или `glm5_2-dsa-e2e`
- `dsv4-pro-e2e`
- `dsv4-pro-none-e2e`
- `dsv4-pro-csa-e2e`
- `dsv4-pro-hca-e2e`
- `dsv4-pro-all`
- `dsv4-flash-e2e`
- `custom`

Отдельных shape-only и коротких sanity cases нет. Если нужен маленький performance run, передайте обычные runtime параметры через `custom` или нужный case, например `--input-len`, `--output-len`, `--iters`, `--warmup-iters`.

## Normalized suite

`normalized-e2e-all` запускает только E2E attention-layer paths на синтетических normalized-конфигах из `sber_benchmarks/sglang_attention_normalized_configs/`:

- `normalized-gdn`: extrapolated Qwen-family Gated Delta Net.
- `normalized-dsv4-none`, `normalized-dsv4-csa`, `normalized-dsv4-hca`: DeepSeek V4 DSA/SWA, CSA/C4 and HCA/C128.
- `normalized-deepseek-v3-mla`: DeepSeek V3 latent MLA with the same synthetic DSV4-width heads/ranks.
- `normalized-glm-dsa`: GLM-5.2 DSA with production indexer/top-k sparsity.

DSA/MLA normalized target is DeepSeek V4 Pro attention width: `hidden_size=8192`, `num_attention_heads=128`, `qk_nope_head_dim=448`, `qk_rope_head_dim=64`, `v_head_dim=512`, `kv_lora_rank=512`, `q_lora_rank=1024`, and DSA indexer width `64 x 128`. Top-k is not forced to one common value because production SGLang constraints differ: DeepSeek V4 accepts `c4_sparse_topk` only in `{512, 1024}` and the normalized DSV4 config uses `1024`; GLM-5.2 fused/tilelang sparse path requires `topk == 2048`, so normalized GLM uses `2048`.

The normalized DeepSeek V3 MLA shape has a synthetic 512-wide MHA head dimension. On H100, FA3 rejects this shape, so the launch script defaults only this normalized MLA run to `NORMALIZED_MLA_ATTENTION_BACKEND=triton`. Regular non-normalized MLA runs still follow `MLA_ATTENTION_BACKEND` and server defaults.

The normalized GDN config is intentionally extrapolated rather than width-matched one-to-one: Qwen3 Next 80B -> Qwen3.5 397B doubles hidden size and value heads while keeping key heads and key/value head dims fixed. The normalized config repeats that scaling step to `hidden_size=16384`, `linear_num_value_heads=128`, `linear_num_key_heads=16`, `linear_key_head_dim=128`, `linear_value_head_dim=128`.

Example context sweep used for normalized comparison:

```bash
for input_len in 65536 131072 262144 524288 1048576; do
  INPUT_LEN=${input_len} OUTPUT_LEN=32768 ENABLE_DP_ATTENTION=1 MONITOR=0 bash sber_benchmarks/run_sglang_attention_kernel_benchmark.sh normalized-e2e-all
  sleep 1
done
```

## TP и DP-attention

Отдельного DP параметра в launch script нет.

- `TP_SIZE` задает число GPU и tensor parallel size.
- При `TP_SIZE > 1` shell запускает Python через `python -m torch.distributed.run --standalone --nproc_per_node ${TP_SIZE}`.
- Python требует, чтобы `WORLD_SIZE == --tp-size`; иначе benchmark падает до замеров.
- Внутри `torchrun` benchmark поднимает SGLang distributed/model-parallel groups через `init_distributed_environment`, `initialize_model_parallel` и production `initialize_dp_attention`; отдельный inference server для NCCL-коммуникации не нужен.
- По умолчанию `ENABLE_DP_ATTENTION=0`, то есть измеряется обычный TP: `attention_dp_size=1`, `attn_tp_size=tp_size`.
- Если в shell поставить `ENABLE_DP_ATTENTION=1` или передать Python flag `--enable-dp-attention`, benchmark выставляет `attention_dp_size=tp_size`, effective `attn_tp_size=1`, а SGLang global `is_dp_attention_enabled()` тоже становится true.
- В DP-attention режиме `batch_size` в JSON означает локальный batch на attention-DP rank. Effective global batch записывается отдельно как `effective_global_batch_size = batch_size * attention_dp_size`.
- KV/cache в DP-attention режиме локален для attention-DP rank, как в SGLang serving: один запрос не TP-шардится по 8 attention ranks, потому что effective `attn_tp_size=1`. JSON пишет `kv_cache_owner_scope=attention_dp_rank_local` и `kv_cache_sharded_across_attention_tp=false`.

В обычном TP benchmark использует SGLang collectives. Generic sparse and GDN layer-boundary communication идут через `attention_tensor_model_parallel_all_reduce`. DSV4 E2E делает reduction в `wo_b`-аналоге через `tensor_model_parallel_all_reduce`, как production `RowParallelLinear(reduce_results=attn_tp_size > 1)`. DeepSeek V3 MLA and GLM DSA E2E инстанцируют SGLang `DeepseekV2AttentionMLA` с `reduce_results=(attn_tp_size > 1)`, поэтому output projection делает production TP all-reduce внутри модуля.

В DP-attention режиме benchmark добавляет measured `all_gather_into_tensor` плюс local copy, чтобы учесть production DP-attention gather/scatter boundary из `dp_gather_partial`/`dp_scatter`. Это не отдельный DP>1 sweep: DP-attention включается только флагом и использует те же `TP_SIZE` GPU. JSON пишет `sglang_dp_attention_initialized=true`, когда production DP-attention globals были инициализированы. Для SGLang module E2E (`DeepseekV2AttentionMLA`/GLM DSA) synthetic `ForwardBatch` также выставляет DP-attention buffer metadata (`DpPaddingMode`, global/local DP buffer lengths), потому что production `RowParallelLinear` читает этот state внутри `o_proj` даже до возможного all-reduce.

Rank 0 печатает records и пишет JSON. После каждого kernel record rank 0 собирает per-rank statuses через distributed object gather; job command остается successful в non-strict режиме, но failed records явно печатаются в console summary и попадают в JSON. Если весь benchmark command всё же завершился nonzero, job script печатает exit code команды, продолжает остальные benchmark runs и завершает job с exit 0 by design.

## Decode и cache lifecycle

E2E пути моделируют рост cache как в autoregressive decode:

- Prefill идет chunk schedule по `CHUNKED_PREFILL_SIZE`.
- E2E paths require more than one prefill chunk; a single-chunk E2E run fails explicitly because it is not a valid chunked-prefill comparison.
- Каждый prefill chunk сначала пишет KV/state, потом вызывает attention/update на этом chunk.
- Decode идет циклом `step in range(output_len)`.
- На каждом decode step используется отдельный per-step random hidden tensor.
- Для DSV4 позиция равна `input_len + step`, SWA KV cache append происходит на один токен, а sparse metadata строится для `input_len + step + 1`. В режимах CSA/HCA дополнительно обновляется production extra KV cache через compressor.
- Для GDN используется тот же causal-conv state и recurrent/SSM state, обновляемые последовательными prefill chunks и decode вызовами.
- Для DeepSeek V3 MLA и GLM DSA benchmark создает полный prefill hidden tensor и отдельный hidden tensor на каждый decode step; `ReqToTokenPool`/`TokenToKVPool` получают последовательные token slots, поэтому cache length растет как `input_len + step + 1`.
- Для GLM DSA sparse часть обязательна: слой по умолчанию `GLM_DSA_LAYER_ID=2` вычисляет production indexer/top-k и должен вернуть `topk_indices` для следующего shared-topk слоя. Если indexer не создан или top-k не возвращается в этом режиме, benchmark падает.

Random tensors and synthetic weights are preallocated before timing, чтобы замерять layer kernels and communication, not random generation.

E2E timing is split into phases:

- `timing.prefill`: measures only full chunked prefill over `input_len` tokens.
- `timing.decode`: before each measured sample, primes cache/state by running full prefill outside the measured CUDA-event region, then measures only the sequential `output_len` decode loop.
- Top-level aggregate `tokens_per_second_p50` is intentionally not used for E2E comparison; use `prefill_tokens_per_second_p50` and `decode_tokens_per_second_p50`.

## DeepSeek V4 path

Target path: `DeepseekV4AttnBackend`, not generic sparse-flash and not DSA backend.

Important production-aligned details in this benchmark:

- Preset `deepseek_v4_pro`: `num_attention_heads=128`, `num_key_value_heads=1`, `head_dim=512`, `qk_nope_head_dim=448`, `qk_rope_head_dim=64`, `kv_lora_rank=512`, `q_lora_rank=1024`, `index_n_heads=64`, `index_head_dim=128`, `index_topk=1024`, `o_groups=16`.
- DSV4 attention query dimension is `qk_nope_head_dim + qk_rope_head_dim = 512`. `kv_lora_rank=512` is the value/output rank. Production FP8 KV storage is packed as `448` FP8 nope bytes + `64` BF16 rope values + scale bytes, i.e. `584` bytes/token before page padding.
- Pure TP with `tp_size=8` has local query heads `128 / 8 = 16` before production FlashMLA padding.
- With DP attention disabled and effective `attn_tp_size > 1`, current SGLang `DeepseekV4AttentionMLA.forward` pads per-rank Q to `64` heads when `n_local_heads <= 64`, otherwise to full `n_heads`, then slices output back to the rank-local heads. For DSV4 Pro TP=8 this means `local_q_heads=16`, `flashmla_q_heads=64`, not 128. The benchmark mirrors this production workaround and emits warning messages for DP-off DSV4 runs, because it is not the preferred TP=8 sparse-attention layout.
- With `--enable-dp-attention` and `tp_size=8`, effective `attn_tp_size=1`, so DSV4 uses all `128` query heads on each attention-DP rank and does not take the TP padding path. This is the mode used for the new comparison sweeps.
- `flashmla-kvcache-e2e` starts from synthetic hidden states, applies synthetic `wq_a/q_norm/wq_b`, synthetic `wkv/kv_norm`, RoPE, production packed SWA KV cache writes, attention, inverse RoPE, grouped `wo_a`, `wo_b`, and distributed boundary communication.
- DSV4 E2E compression is enabled by default. The shell default is `DSV4_COMPRESSION=csa`, which maps to `--dsv4-compression csa` and ratio `4`.
- SWA-only, CSA, and HCA are separate E2E runs in `all`, `e2e-all`, and `dsv4-pro-all`: `none` uses ratio `0`, `csa` uses ratio `4`, `hca` uses ratio `128`.
- SWA-only can be run directly with `--dsv4-compression none` or case `dsv4-pro-none-e2e`.
- The E2E path calls SGLang `flash_mla_with_kvcache_entrypoint`, the same wrapper used by `DeepseekV4AttnBackend.forward`. When compression is `csa`/`hca`, the benchmark computes compressed KV through SGLang compressor kernels, writes production packed extra KV cache, and passes `extra_k_cache`, `extra_indices_in_kvcache`, and `extra_topk_length` into that wrapper.
- CSA/C4 E2E includes the production C4 indexer work: synthetic `q_lora` goes through `wq_b`, synthetic hidden state goes through `weights_proj`, the C4 indexer compressor updates the indexer KV cache via `linear_bf16_fp32`, `compress_forward`, `compress_fused_norm_rope_inplace`, `rotate_activation`, `act_quant`, and the production index buffer store layout, then DeepGEMM `fp8_paged_mqa_logits` and SGLang `topk_transform_512*` produce `c4_sparse_page_indices`.
- HCA/C128 E2E does not call the C4 indexer in SGLang production either. Its `extra_indices_in_kvcache` come from SGLang `init_compression_metadata` as padded `c128_page_indices`, and `extra_topk_length` comes from `c128_topk_lengths_clamp1`.
- `attn_sink` and `topk_length` are enabled for DSV4 paths where the installed wrapper supports them; unsupported shapes become `status=unsupported` or `status=error` records, not silent fallbacks.

DSV4 E2E kernel selector:

- Shell config `DSV4_ATTENTION_KERNEL` and Python flag `--dsv4-attention-kernel` choose the SGLang DSV4 kvcache attention entrypoint.
- Accepted benchmark values are `triton`, `flashmla`, and `tilelang`.
- `triton` maps to SGLang wrapper backend `triton`. This is the default for H100 and is validated with compressed CSA/C4 and HCA/C128 full 8-GPU jobs.
- `flashmla` maps to SGLang wrapper backend `kernel`, i.e. the installed `sgl_kernel.flash_mla` implementation. In the current H100 image it is not usable for compressed DSV4 E2E with the packed FP8 cache layout: full 8-GPU CSA and HCA probes record `status=error`, `kv must have shape (num_blocks, page_block_size, h_kv, bytes_per_token)`.
- `tilelang` maps to SGLang wrapper backend `tilelang`. In the current H100 image full 8-GPU CSA and HCA probes record `status=error`, `Check failed: src_info.order < dst_info.order (6 vs. 4)`.
- SGLang also has `torch` and `comparison` branches inside the wrapper; those are reference/debug paths and are intentionally not exposed as production benchmark choices.
- SGLang generic `attention_backend=dsa` choices such as `flashmla_sparse`, `flashmla_kv`, `flashmla_auto`, `fa3`, `tilelang`, `aiter`, and `trtllm` belong to `DeepseekSparseAttnBackend`, not the DeepSeek V4 `DeepseekV4AttnBackend` path benchmarked here.

Current validation note:

- Older aggregate E2E numbers in this file were removed from the methodology because they mixed prefill and decode into one throughput and predated the hidden-state DSV4/GDN fixes. Use only phase metrics from fresh jobs after this version.

E2E output path included in this benchmark:

- DSV4 E2E slices FlashMLA output back to TP-local heads, applies inverse RoPE on the rope tail, runs grouped `wo_a`, then `wo_b`. Pure TP reduction is inside the `wo_b` analogue; DP-attention boundary communication happens after projection.
- GDN E2E runs `RMSNormGated(core_attn_out, z)` and a row-parallel `out_proj` analogue before layer-boundary communication.

Remaining differences from a full SGLang server:

- The benchmark still does not load real checkpoint weights; DSV4/GDN projection and output weights are synthetic benchmark tensors.
- It does not run embedding, MLP/MoE, scheduler, or token sampling.
- DSV4 request/page metadata is synthetic and benchmark-local, but C4 and C128 sparse metadata uses the same SGLang kernels/helpers as the production backend. It is not produced by a live request scheduler.
- DP-attention communication is measured with SGLang group collectives, but without constructing a full scheduler. For DeepSeek V3 MLA and GLM DSA the benchmark does construct SGLang `ForwardBatch`/`RadixAttention` metadata; for the custom DSV4/GDN E2E paths it models the same boundary collective directly.

## DeepSeek V3 MLA path

Target path: SGLang `DeepseekV2AttentionMLA` + selected MLA attention backend. Это full attention-module E2E, но не полный server/model block.

Default preset `deepseek_v3_mla` follows the largest DeepSeek V3 MLA geometry implemented by SGLang: `hidden_size=7168`, `num_attention_heads=128`, `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`. The benchmark extends `max_position_embeddings` when needed so the default 262144+32768 token run has valid positions.

The E2E path includes dense input projections inside `DeepseekV2AttentionMLA`, RoPE, KV-pool writes, `RadixAttention`, `w_vc`, and `o_proj`. With TP>1 the SGLang row-parallel `o_proj` performs the output all-reduce. Backend selection is controlled by shell `MLA_ATTENTION_BACKEND` / Python `--mla-attention-backend`; accepted values are `auto`, `fa3`, `flashinfer`, `flashmla`, `cutlass_mla`, `trtllm_mla`, `tokenspeed_mla`, `cutedsl_mla`. On H100, `auto` resolves to `fa3`.

Current validation note: older aggregate MLA numbers are stale after phase timing; use fresh `timing.prefill` and `timing.decode` records.

## GLM-5.2 DSA path

Target path: SGLang `DeepseekV2AttentionMLA` in DSA mode, `DSATokenToKVPool`, production indexer/top-k, and selected DSA prefill/decode backends. This is not DSV4 CSA/HCA compression. GLM DSA sparsity is the production indexer-selected top-k KV/page subset: the model computes index logits, transforms them into `topk_indices`, limits DSA cache lengths to `min(seq_len, index_topk)`, and passes those sparse indices into DSA attention kernels.

Default preset `glm5_2_dsa` follows the largest GLM-5.2 config available in SGLang-compatible HF format: `hidden_size=6144`, `num_attention_heads=64`, `q_lora_rank=2048`, `kv_lora_rank=512`, `qk_nope_head_dim=192`, `qk_rope_head_dim=64`, `v_head_dim=256`, `index_n_heads=32`, `index_head_dim=128`, `index_topk=2048`, `index_topk_freq=4`, `index_skip_topk_offset=3`, `rope_theta=8000000`, `max_position_embeddings=1048576`.

The default benchmark layer is `--glm-dsa-layer-id 2`: it computes indexer/top-k itself and, under GLM5.2 sharing rules, returns `topk_indices` for the next shared-topk layer. Layers that would skip top-k are rejected, because measuring sparse attention without sparse indices would be invalid. During long DSA steps the benchmark also rejects dispatches that do not go through `attention_backend=dsa`/`AttnForwardMethod.MLA`, and records observed sparse dispatch/top-k-return counters in JSON. A full multi-layer GLM server alternates compute-topk and reuse-topk layers; this benchmark measures one production compute-topk attention layer, not the whole transformer stack.

Backend selection follows SGLang server-style flags: shell `DSA_PREFILL_BACKEND`, `DSA_DECODE_BACKEND`, `DSA_TOPK_BACKEND`, `DSA_KV_CACHE_DTYPE`; Python `--dsa-prefill-backend`, `--dsa-decode-backend`, `--dsa-topk-backend`, `--dsa-kv-cache-dtype`. Accepted DSA backend names are `auto`, `flashmla_sparse`, `flashmla_kv`, `flashmla_auto`, `fa3`, `tilelang`, `aiter`, and `trtllm`; top-k backends are `sgl-kernel`, `torch`, and `flashinfer`. On H100, SGLang auto/BF16 DSA defaults are prefill `flashmla_sparse` and decode `fa3`; with `fp8_e4m3`, auto uses `flashmla_kv` for both.

SGLang also has `--enable-hisparse` for DSA/DeepSeek V4 serving, but it is not just another attention kernel selector. In production it swaps in HiSparse KV pool allocators, a `HiSparseCoordinator`, host/device KV staging, and scheduler-specific prefill-to-decode transitions. This standalone attention-module E2E intentionally does not expose `ENABLE_HISPARSE`, because setting only `hisparse_coordinator` or only backend names would not be a faithful server path. GLM DSA sparsity measured here is the base production indexer/top-k sparse attention path, not dense attention and not DSV4 CSA/HCA compression.

The current job image has SGLang 0.5.13. That release already contains `GlmMoeDsaForCausalLM`, DSA indexer/top-k, `DSATokenToKVPool`, DSA backend sparse-index calls, and the needed server args. SGLang 0.5.14 adds helper/refactor code around the same semantics; the benchmark keeps the skip-topk check locally so it can run on 0.5.13.

Current validation note: older aggregate GLM DSA numbers are stale after phase timing; use fresh `timing.prefill` and `timing.decode` records.

## Qwen3.5 GDN path

Target path: SGLang Qwen3.5 Gated Delta Net kernels.

Production-related imports/paths:

- `sglang.srt.layers.attention.linear.kernels.gdn_triton.TritonGDNKernel`
- `sglang.srt.layers.attention.linear.kernels.gdn_flashinfer.FlashInferGDNKernel`
- `sglang.srt.layers.attention.linear.kernels.gdn_cutedsl.CuteDSLGDNKernel`
- `sglang.srt.layers.attention.fla.fused_gdn_gating.fused_gdn_gating`
- `sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_fn`
- `sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_update`

`linear-gdn-e2e` starts from synthetic hidden states, applies synthetic `in_proj_qkvz` and `in_proj_ba`, then uses the production fused `fused_qkvzba_split_reshape_cat_contiguous` path where the Qwen3.5 head ratio permits it. Prefill slices hidden states by token offset, runs production causal-conv prefill/update against a persistent conv state, then runs production GDN gating and recurrent/SSM state update. Decode uses a distinct hidden tensor for each step, calls `causal_conv1d_update`, then GDN packed decode when the selected backend supports it. The path finishes with `RMSNormGated + out_proj` and layer-boundary communication. It skips non-attention model blocks. Standalone linear kernels measure one selected backend/helper directly.

Current validation note: older aggregate GDN numbers are stale after hidden-projection and phase-timing fixes; use fresh `timing.prefill` and `timing.decode` records.

## Important CLI flags

Python benchmark flags normally passed by the shell:

- `--kernel`: kernel or group; repeatable.
- `--model-preset`: `qwen3_5_397b_a17b`, `qwen3_next_80b_a3b`, `deepseek_v4_pro`, `deepseek_v4_flash`, `deepseek_v3_mla`, `glm5_2_dsa`, or `custom`.
- `--tp-size`: tensor parallel size; must equal distributed world size when launched with torchrun.
- `--enable-dp-attention`: compute `attention_dp_size=tp_size`, `attn_tp_size=1`.
- `--attn-tp-size`: explicit specialization probe. Do not use it for production-layout results unless you intentionally want a kernel-shape probe.
- `--input-len`, `--output-len`, `--seq-len`, `--batch-size`, `--decode-batch-size`.
- `--chunked-prefill-size`: shared E2E prefill chunk size for DSV4/GDN.
- `--dsv4-compression`: `none`, `csa`, `hca`, or `custom`; default is `csa`.
- `--dsv4-compress-ratio`: required only with `--dsv4-compression custom`; accepts `0`, `4`, or `128`.
- `--dsv4-attention-kernel`: DSV4 E2E kvcache attention implementation: `triton` by default, or explicit `flashmla`/`tilelang` probes.
- `--mla-attention-backend`, `--mla-kv-cache-dtype`: DeepSeek V3 MLA E2E backend and KV cache storage selectors.
- `--dsa-prefill-backend`, `--dsa-decode-backend`, `--dsa-topk-backend`, `--dsa-kv-cache-dtype`, `--glm-dsa-layer-id`: GLM DSA E2E backend/indexer selectors.
- `--dtype`: requested target dtype; FP8 request still uses BF16 activation tensors where SGLang kernels require BF16 and FP8 for KV/cache/indexer storage.
- `--state-dtype`: GDN recurrent state dtype.
- `--strict`: raise on first kernel error instead of recording an error record and continuing.
- `--output-json`: write a JSON list of records.

## Output records

Each record includes:

- launch/model fields: `argv`, `launch_args`, `model`, `model_preset`.
- parallel layout: `tp_size`, `attention_dp_size`, `attn_tp_size`, `enable_dp_attention`, `distributed_world_size`, `distributed_rank`, `local_rank`, `tp_communication_mode`, `batch_size_scope`, `effective_global_batch_size`, `effective_global_decode_batch_size`, `kv_cache_owner_scope`, `kv_cache_sharded_across_attention_tp`.
- shapes and memory estimates: `shapes`, `estimated_live_bytes`, `estimated_live_gib`, plus persistent/transient split where useful.
- DSV4 fields: `dsv4_compression`, `dsv4_compress_ratio`, `dsv4_compress_ratio_requested`, `dsv4_compress_ratio_overridden`, `dsv4_attention_kernel`, `dsv4_attention_kernel_entrypoint`, `local_q_heads`, `flashmla_q_heads`, `q_padded_for_flashmla`, `q_padded_to_full_heads`, `production_output_slice_heads`, `dsv4_c4_indexer_enabled`, `q_lora_rank`, `index_n_heads`, `index_head_dim`, `c4_page_size`, `c4_max_seq_len`, and `c128_indices_width`.
- GDN E2E fields: `linear_conv_prefill_impl`, `linear_conv_decode_impl`, `linear_conv_dim`, `linear_conv_kernel_dim`, `gdn_prefill_uses_unique_chunk_tensors`, `gdn_conv_state_enabled`, and `gdn_ssm_state_enabled`.
- MLA/DSA E2E fields: `production_module`, `production_forward_includes_input_projections`, `production_forward_includes_rope`, `production_forward_includes_kv_cache_writes`, `production_forward_includes_output_projection`, `production_forward_uses_indexer`, `dsa_sparse_indices_required`, `dsa_sparse_topk`, `dsa_observed_sparse_dispatches`, `dsa_observed_mha_one_shot_dispatches`, `dsa_observed_topk_returns`, `mla_attention_backend_effective`, `dsa_prefill_backend_effective`, `dsa_decode_backend_effective`, `dsa_topk_backend`, `dsa_layer_id`, and `dsa_layer_skips_topk`.
- runtime: `timing.prefill`, `timing.decode`, phase latency percentiles, phase throughput, CUDA peak memory.
- status: `ok`, `unsupported`, or `error`.
- diagnostics: `warning_messages`, `model_parallel_notes`, `distributed_rank_statuses`, `error_message`, `traceback`.

Non-ok records are printed in the console summary even though the overall job command exits successfully in non-strict mode. Error records are best-effort enriched with `shape_summary` fields so failed DSV4 CSA/HCA runs still show the intended compression mode, ratio, and cache geometry in JSON.
