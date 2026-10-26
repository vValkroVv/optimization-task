# Float8 training

The repository contains two independent float8 training schemes: quantization of **dense** layers (regular `nn.Linear` modules) and **sparse** layers (MoE experts). They are enabled separately and do not depend on each other.

⚠️ Training is supported only on Hopper GPUs (H100): cudacherry and DeepGEMM are built with `TORCH_CUDA_ARCH_LIST="9.0"` (see [gift_docker/Dockerfile](../../../gift_docker/Dockerfile)).

Everything is configured through the `fp8:` section of an experiment YAML, which decodes into the [FP8Config](../../config/configs.py) dataclass:

```yaml
fp8:
  dense_include_linear_layers: '.*\.(w1|w2|w3|wo)$'  # dense: Linear modules to replace
  dense_use_shared_quantizer: true                   # dense: shared FFN quantizer
  dense_fused_swiglu_quant: true                     # dense: fuse SwiGLU with w2 input quantization
  sparse_use_grouped_gemm: true                      # sparse: FP8 grouped GEMM for experts
  sparse_deep_gemm_num_sms: 116                      # sparse: SM budget for DeepGEMM
  sparse_fused_swiglu_quant: true                    # sparse: fuse SwiGLU with w2 input quantization
```

- `dense_fused_swiglu_quant=False` and `sparse_fused_swiglu_quant=False` replace fused SwiGLU plus quantization with a bf16 SwiGLU between the FP8 GEMMs (the second GEMM quantizes its input internally).
- Hidden z-loss on the gate/up projections of routed experts (`layers.*.moe.experts.w1` / `.w3` in `model.config.hidden_z_loss_patterns`) requires `sparse_fused_swiglu_quant=False` for FP8 experts: the fused kernel's backward passes an FP8 gradient to `w13`, and the z-loss gradient cannot be added to it.
- Likewise, hidden z-loss on the gate/up projections of a dense FFN (`layers.*.feed_forward.w1` / `.w3` and `layers.*.moe.shared_experts.w1` / `.w3`) requires `dense_fused_swiglu_quant=False` for an FP8 FFN: the fused kernel's backward passes an FP8 gradient to `w1`/`w3`.

The `dense_` and `sparse_` prefixes separate the two schemes: `dense_` fields do not affect experts, while `sparse_` fields do not affect regular linear layers.

---

## Sparse MoE FP8

### Quick start

One flag in the `fp8:` section and a compatible token dispatcher in the experiment's `model.config` are sufficient:

```yaml
# e.g. configs/experiments/my_gift_3_5_3b_fp8.yaml
config_version: 1
hf_assets_path: ./assets/hf/gift_3_5-3b
loss: {type: cross_entropy}
dataloader: {type: hf_text, dataset: c4}
imports:
  - ../presets/models/gift_moe/gift_3_5/3b.yaml
model:
  config:
    moe:
      experts:
        token_dispatcher: {type: deepep}
fp8:
  sparse_use_grouped_gemm: true
activation_checkpoint:
  mode: selective
parallelism:
  expert_parallel_degree: 8
```

Ready-to-use experiments in `configs/experiments/`:

| Experiment                 | Purpose                                                                 |
| -------------------------- | ----------------------------------------------------------------------- |
| `gift_3_5_700b.yaml`       | GIFT 3.5 700B, production: dense FP8 (`dense_include_linear_layers`) + grouped GEMM over DeepEP |
| `gift_4_700b.yaml`         | The same with the soft-capped LM head                                   |
| `llama4_debug_fp8.yaml`    | Smoke run: FP8 grouped-GEMM experts over DeepEP                         |
| `llama3_debug_float8.yaml` | Smoke run: dense Float8                                                 |
| `gift_moe_debug_deepep_fp8.yaml` | Smoke run: dense + grouped FP8 over DeepEP (preset `debugmodel_fp8`, EP 4) |
| `gift_moe_debug_hybridep_fp8.yaml`, `..._compile.yaml` | Smoke run: dense + grouped FP8 over HybridEP (EP 2), without and with `torch.compile` |
| `gift_moe_3b_hybridep_fp8_compile*.yaml` | 3B, FP8 over HybridEP + compile: EP 8, `_pp`, `_hsdp`, `_tp` (sparse FP8 only) |
| `gift_moe_3b_hybridep_dense_fp8_compile_non_blocking.yaml` | 3B, compiled dense FP8 with BF16 experts and fixed-capacity non-blocking HybridEP |

The shared dense + sparse FP8 block is the mixin `configs/presets/train/fp8_full.yaml`.

Run (the experiment from the example above):

```bash
NGPU=8 ./run_train.sh configs/experiments/my_gift_3_5_3b_fp8.yaml
```

### Requirements and validation

All of the following is validated by [enable_fp8_grouped_experts()](../../components/quantization/utils.py). Invalid configurations fail at startup with a clear `ValueError`, before model allocation:

- `deepep` **or** `hybridep`. Regular all-to-all and EP=1 are unsupported: DeepGEMM requires expert-aligned token groups. `hybridep` additionally requires `dim % 512 == 0`, a `pad_multiple` divisible by 128, `fuse_permute=True`, and an unset `non_blocking_capacity_factor`.
- `score_before_experts=False`. Routing weights are applied to the experts' bf16 output during the combine stage, rather than before the FP8 experts.
- A `param_init` containing `w13_E2FD`. Both bf16 and FP8 experts store the gate and up projections in a single `w13_E2FD` parameter of shape `(E, 2, F, D)` (slot 0 is gate and slot 1 is up). The initializer for each half is the spec `FusedW13Init(gate=, up=)`; it is no longer inferred from the `w1_EFD` / `w3_EFD` pair.

---

## DeepGEMM and the kernel cache

DeepGEMM implements efficient float8 grouped matrix multiplication between expert groups and token groups. It uses JIT compilation to select and compile the best kernel configuration for specific matrix sizes, SM count, and other parameters.

At the start of training, this can cause hours of idle time as ranks wait for compilation on other ranks. The cache should therefore be warmed in advance and propagated to every rank.

### How the cache reaches a training run

The logic lives in [cache_manager.py](../warmup/cache_manager.py) and is called from [trainer.py](../../trainer.py) when `fp8.sparse_use_grouped_gemm` is set.

**The cache is supplied externally:**

```bash
PROPAGATE_DG_CACHE=1 \
PROPAGATE_DG_CACHE_DIR=/path/to/warmed/cache \
NGPU=8 ./run_train.sh configs/experiments/my_gift_3_5_3b_fp8.yaml
```

Rank 0 copies the specified directory to `/temp/.cache/rank_{r}/deepgemm` for every rank. If the path does not exist, an assertion fails at startup instead of silently falling back to JIT compilation.

The log will contain `DeepGEMM: transferred N file(s) from ...`. If `N=0`, **the cache was not loaded**, and kernels will be compiled at runtime.

### Running without a warmed cache

By default, a missing kernel raises an **error**:

```console
RuntimeError: Assertion error (.../jit/compiler.hpp): false and "CACHE MISS: set
DG_CACHE_FORCE_SKIP=1 to allow runtime compilation, or warm up the cache before running"
```

If warming the cache is undesirable, set `DG_CACHE_FORCE_SKIP=1` in the environment. Missing kernels will then compile at runtime without raising an error, but the first steps will be very slow.

### Warming the cache

1. Fill in a warmup configuration under [gift_configs/warmup/](../../../gift_configs/warmup/).
2. Set `USER_LOGIN` and `WARMUP_CONFIG_PATH` in [gift_scripts/submit_warmup.py](../../../gift_scripts/submit_warmup.py), then run the script.
3. Save the warmed cache and provide it through `PROPAGATE_DG_CACHE_DIR`, or add it to the image. The S3 path is defined in the **CI-CD** repository in `scripts/download_deep_gemm_cache_s3.sh`.

Example configuration:

```yaml
deep_gemm:
  cache_dir: "/path/to/cache/folder"
  gemm_groups:
    - "w1_m_grouped"
    - "w1_m_grouped_nt_for_dgrad_transposed"
    - "w1_k_grouped"
    - "w2_m_grouped"
    - "w2_m_grouped_nt_for_dgrad_transposed"
    - "w2_k_grouped"
  hid_dim: 1280          # experts.dim
  int_dim: 896           # experts.hidden_dim (moe_hidden_dim)
  num_groups: 16         # num_routed_experts // ep_size
  num_sms: 116           # = FP8Config.sparse_deep_gemm_num_sms
  min_size: 128          # minimum number of tokens per expert
  max_size: 64000        # maximum number of tokens per expert (expert capacity)
```

The cache is shape-specific. Changing `dim`, `moe_hidden_dim`, `expert_parallel_degree`, or `num_sms` invalidates the old cache. Check `n_deepgemm_files` at startup.

### SM budget

`sparse_deep_gemm_num_sms` shares the GPU with the EP backend's own SM budget
(`num_sms` on `DeepEPTokenDispatcher.Config`; when omitted, DeepEP picks its
default). An H100 SXM has 132 SMs in total; the configured default of 116 for
DeepGEMM leaves 16 SMs for communication. Coordinate both budgets in the
experiment YAML: `fp8.sparse_deep_gemm_num_sms` and
`model.config.blocks.<block>.moe.experts.token_dispatcher.num_sms` on every
MoE block that uses DeepEP, including any selected MoE MTP blocks. For the
GIFT 3.5 presets, the trunk blocks are `gdn_moe` and `mla_moe`:

```yaml
fp8:
  sparse_deep_gemm_num_sms: 116
model:
  config:
    blocks:
      gdn_moe:
        moe:
          experts:
            token_dispatcher: {type: deepep, num_sms: 16}
      mla_moe:
        moe:
          experts:
            token_dispatcher: {type: deepep, num_sms: 16}
```

---

## Dense FP8

Dense FP8 and LoRA may target disjoint projections. Overlapping selections are rejected before training; the same projection cannot use both converters.

The dense path is independent of MoE and does not use Transformer Engine. `apply_float8()` walks the model configuration tree and replaces every `Linear.Config` whose FQN matches `dense_include_linear_layers` with [FP8Linear.Config](./cuda_gemm.py).

### Quick start

```yaml
fp8:
  dense_include_linear_layers: '.*\.(w1|w2|w3|wo)$'
  dense_use_shared_quantizer: true
```

- `dense_include_linear_layers` is an FQN regex. The pattern above covers only `w1/w2/w3/wo`, so `lm_head`, `router.gate`, and `ba_proj` remain in bf16.
- `dense_use_shared_quantizer` enables a shared input quantizer on every `FeedForward` ([ffn.py](../../models/common/ffn/__init__.py)). This is useful when several linear layers consume the same activation (`w1` and `w3` in an FFN): the activation is quantized once instead of once per layer.

Dense only: import `configs/presets/train/fp8_full.yaml` and set `fp8.sparse_use_grouped_gemm: false`.

### No cache warmup required

Dense FP8 requires no warmup: its forward and backward passes use cudacherry's FP8 GEMM rather than shape-specific DeepGEMM JIT templates. Set the `fp8:` section and start training.

The only restriction is `bias=False` on replaced `Linear` modules: `FP8Linear` does not support bias, and validates this when it is constructed.
