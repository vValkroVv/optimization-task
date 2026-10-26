# gift_moe

Two model types cover a DeepSeek-V3-style MoE and the GIFT 3.5/4 hybrids
built on top of it. All of them are MoE decoders with routed + shared
experts, optional multi-token-prediction (MTP) heads, and a DeepSeek-V3-shaped
HuggingFace checkpoint layout.

| `model.config.type` | attention | LM head | presets (`configs/presets/models/gift_moe/`) |
| --- | --- | --- | --- |
| `gift_moe` | MLA everywhere | plain | `debugmodel`, `3b`, `16b`, `236b`, `671b` |
| `gift_3_5` | GDN linear attention with MLA every few layers | plain | `gift_3_5/{debugmodel,3b,700b}` |
| GIFT 4 (`gift_3_5` type) | same as 3.5, preset for preset; plain (not zero-centered) gated norms (each block's norms layered with `zero_centered: false` over the gift_3_5 preset), `post_attn_norm` weights start at zero | logits soft-capped to ±30 | `gift_4/{debugmodel,3b,700b}` |

The debug experiments import tiny `debugmodel` presets and train on the bundled `c4_test`
corpus and test tokenizer (`tests/assets/tokenizer`) — no downloads, no HF assets.

## Layout

| path | what |
| --- | --- |
| `torchtitan/config/schema/models/gift_moe/` | the model Configs (`GiftMoEModelConfig` for `gift_moe`, `GiftHybridModelConfig` for `gift_3_5`): runtime metadata and pure validation. GIFT 4 uses the `gift_3_5` type; structure and initialization are YAML presets |
| `configs/presets/models/gift_moe/`, `configs/presets/init/` | model structure (`blocks` / `layers`) and initialization policies; see [docs/config.md](/docs/config.md#the-model-config-convention) |
| `versions.py` | the runtime half (`BASE`, `HYBRID`): parallelize, pipelining, state-dict adapter |
| `const.py` | the weight-init conventions all types share |
| `_base/` | the MLA MoE model, its parallelize/sharding, and the DeepSeek-V3 state-dict adapter |
| `gift_3_5/` | the hybrid: GDN layers, gated RMSNorms, row-renormalized router gates, and the adapter additions they need |
| `train.sh` | local wrapper around `run_train.sh` (not used by CI) |

`gift_3_5` reuses `_base`'s parallelize function and adapter through its
`_runtime_spec`; both types have `family: gift_moe` (`ModelSpec.name`). GIFT 4
is not a type: its presets (`gift_4/*.yaml`, labels `gift_4/<size>`) import the
GIFT 3.5 ones and set plain norms, a zero-initialized post-attention norm and
`lm_head: {type: unembedding, logit_softcap: 30.0}`. The init preset
(`configs/presets/init/gift_hybrid.yaml`) is shared: a gated norm's weight
init is `norm_unit_scale`, which follows the norm's `zero_centered` (zeros
when zero-centered, ones when plain), so the norm alone says which it is.

## Quick start

```bash
# Tiny debug run on the bundled data (uses all visible GPUs)
torchtitan/models/gift_moe/train.sh

# Pick GPU count / experiment / pass through any torchtitan override
NGPU=2 torchtitan/models/gift_moe/train.sh configs/experiments/gift_3_5_debug.yaml --training.steps 20
```

| env | default | meaning |
| --- | --- | --- |
| `CONFIG_FILE` | `configs/experiments/gift_moe_debug.yaml` | experiment YAML; the first positional argument works too |
| `NGPU` | all detected GPUs | number of GPUs |

Anything after the experiment path is forwarded to `torchtitan.train` as a dot-path
override (e.g. `--parallelism.tensor_parallel_degree 2`). Validate a file before
launching with `python scripts/config_check.py <yaml>`.

## Experiments

The runs under `configs/experiments/`. Each imports a model preset from the table
above and adds the training sections; open the file for the exact tree.

| experiment | shape | notes |
| --- | --- | --- |
| `gift_moe_debug.yaml` | 6 layers, MLA, 2 MTP heads | easiest to run; CI adds `--parallelism.expert_parallel_degree 2` |
| `gift_moe_debug_deepep.yaml` | same | EP=2 via DeepEP; needs a DeepEP build (`gift_docker/Dockerfile`) and 2 GPUs |
| `gift_moe_debug_deepep_fp8.yaml`, `gift_moe_debug_hybridep_fp8{,_compile}.yaml` | same, at `debugmodel_fp8` width (2048) | dense + grouped fp8 over DeepEP (EP 4) or HybridEP (EP 2); `DG_CACHE_FORCE_SKIP=1` without a warmed DeepGEMM cache |
| `gift_moe_3b_hybridep_fp8_compile{,_pp,_hsdp,_tp}.yaml` | 18 layers, MLA, flex | fp8 over HybridEP, compiled; one node, EP 8 or the named parallelism variant |
| `gift_moe_3b_hybridep_dense_fp8_compile_non_blocking.yaml` | 18 layers, MLA, flex | compiled dense FP8, BF16 experts, fixed-capacity non-blocking HybridEP |
| `gift_moe_debug_mds.yaml` | same | reads the bundled MDS corpus (`presets/datasets/c4_test_mds.yaml`) instead of the HF loader |
| `gift_moe_debug_monitored.yaml` | same | every callback of `presets/callbacks/monitored.yaml` |
| `gift_moe_streaming_debug.yaml` | 6 layers, no MTP | streaming dataloader over a real corpus; a 4-GPU CI case |
| `gift_3_5_debug.yaml` | 6 layers (GDN + MLA), 2 MTP | hybrid debug |
| `gift_3_5_debug_mds.yaml`, `gift_3_5_debug_varlen.yaml` | same | MDS corpus; document-packed sequences with the varlen kernels |
| `gift_4_debug.yaml` | as `gift_3_5_debug` | plus the logit soft cap, plain (not zero-centered) gated norms, zero-initialized post-attention norms and the projection z-loss preset |
| `gift_3_5_700b.yaml` | 64 layers (48 GDN + 16 MLA), 1 MTP | production; DeepEP, fp8, EP 32; needs real assets |
| `gift_4_700b.yaml` | as `gift_3_5_700b` | plus the logit soft cap, plain gated norms and zero-initialized post-attention norms |

Other sizes (`16b`, `236b`, `671b`, and `3b` beyond its fp8 experiments) exist as model presets only. To train one,
write an experiment that imports it:

```yaml
# GIFT 3.5 3B on the bundled corpus, with DeepEP on two GPUs.
# torchrun --nproc_per_node=2 -m torchtitan.train configs/experiments/gift_3_5_3b_deepep.yaml
# Save as configs/experiments/gift_3_5_3b_deepep.yaml.
config_version: 1
imports:
  - ../presets/models/gift_moe/gift_3_5/3b.yaml
  - ../presets/train/debug.yaml
model:
  config:
    blocks:
      gdn_moe:
        moe:
          experts:
            token_dispatcher: {type: deepep}
      mla_moe:
        moe:
          experts:
            token_dispatcher: {type: deepep}
loss: {type: cross_entropy}
parallelism:
  expert_parallel_degree: 2
```

Validate this file with
`python scripts/config_check.py configs/experiments/gift_3_5_3b_deepep.yaml --world-size 2`.
Training requires the GDN and DeepEP runtime dependencies as well as two GPUs.

Overrides address each named block. For example, FlexAttention on `gift_moe`
belongs in both `blocks.mla_dense.attention.inner_attention` and
`blocks.mla_moe.attention.inner_attention`. To change the dispatcher of the
GIFT 3.5 example above from DeepEP to HybridEP:

```bash
NGPU=2 ./run_train.sh configs/experiments/gift_3_5_3b_deepep.yaml \
    --model.config.blocks.gdn_moe.moe.experts.token_dispatcher.type hybridep \
    --model.config.blocks.mla_moe.moe.experts.token_dispatcher.type hybridep
```

An override of one block does not change another. If a selected MTP head
contains MoE, configure its named block's dispatcher too.

## Linear attention (GDN)

The GDN layers call `fla` kernels, imported lazily — a pure-MLA preset runs without
`fla` installed. The chunked scan and its gated norm come from `fla`, and the training
backward has one environment constraint: on Hopper with Triton in [3.4.0, 3.7.1) the
Triton kernel is numerically broken (fla #640), so `fla` routes through its tilelang
backend and hard-raises when neither `tilelang` nor Triton >= 3.7.1 is present
(`pip install tilelang`).

## HF checkpoint conversion

`_base/state_dict_adapter.py` handles the DeepSeek-V3 layout (MLA, MoE experts, dense
MLP, MTP heads, FP8 dequant on import); `gift_3_5/state_dict_adapter.py` adds the
gated norms and the fused GDN `in_proj_qkvz` / `conv1d` surgery. Both work offline
(plain tensors, the `convert_{to,from}_hf.py` CLI, which take `--config <yaml>`;
pass the run's `<dump_folder>/config/resolved_config.yaml`) and online (sharded
`DTensor`s).

Attention keys are derived from the attention config of each layer, so a hybrid
converts whatever mix it actually has — MLA, GQA (both the separate `q/k/v` form HF
stores and a fused titan `wqkv`, split on export and fused on import), and GDN/KDA,
including on MTP heads. One mix is refused rather than mis-converted: MLA *without*
Q-LoRA together with GQA, because both want HF `self_attn.q_proj` for a different
tensor and the key maps are templated over the layer index, not per layer.
