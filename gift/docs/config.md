# YAML experiment configs

YAML is the user interface for experiments: an experiment file plus small
reusable presets fully describes a run, the model architecture included.
Experiment recipes and model sizes live in YAML. Python registers component
and trainer types; it no longer defines recipe functions or model-shape tables.

```bash
# validate before submitting (works in a minimal env, without torch)
python scripts/config_check.py configs/experiments/llama3_debug.yaml

# train
CONFIG_FILE=configs/experiments/llama3_debug.yaml NGPU=1 ./run_train.sh
NGPU=8 ./run_train.sh configs/experiments/gift_3_5_debug.yaml --training.steps 100

# or directly
torchrun --nproc_per_node=8 -m torchtitan.train \
    configs/experiments/gift_3_5_debug.yaml --training.steps 100
```

For a first run, start with an existing experiment. To change settings, use
[CLI overrides](#cli-overrides) or [create a small experiment](#create-an-experiment)
that imports it. To extend the code, start with [Adding things](#adding-things).

Guide map:

- [File layout](#where-things-live) and [typed config fields](#the-schema-is-the-runtime-config)
- [Model structure](#the-model-is-fully-described-in-yaml) and [model conventions](#the-model-config-convention)
- [Trainer types](#trainer-type), [imports](#imports-and-presets) and [callbacks](#choose-only-the-callbacks-a-run-needs)
- [CLI overrides](#cli-overrides) and [validation stages](#validation-stages)
- [Schema/runtime boundaries](#schema-and-runtime) and [module map](#module-map)
- [Run bundles](#reproducibility) and [golden snapshots](#golden-snapshots)
- [Extension checklist](#adding-things) and [programmatic loading](#programmatic-loading)
- [Recipe migration](#recipe-migration-coverage) and [GIFT norms, packing and MTP](#norms-packing-and-mtp-heads-in-gift-presets)

The GIFT 3.5/4 700B training preset enables checkpointing every 2000 steps.
Use `--checkpoint.enable false` only when intentionally running without saves.
Optimizer and monitoring stat names are checked before submission; router
`route_scale` must be finite. A dense FP8 selector that matches no Linear
layers emits a warning when converters are applied.

## Where things live

| path | what |
| --- | --- |
| `configs/experiments/*.yaml` | one file per run; the only thing a launcher takes. Every file starts with a comment stating what it is and the `torchrun` line that runs it |
| `configs/presets/models/<family>/[<type>/]<size>.yaml` | canonical architecture presets: the `model:` section only. The single place a named architecture's hyperparameters live |
| `configs/presets/train/*.yaml` | training mixins (`debug.yaml`: tiny bundled corpus, 10 steps; `graph_trainer.yaml`: the graph trainer) |
| `configs/presets/datasets/*.yaml` | dataloader sections (the bundled MDS copy of `c4_test`, the `cc12m_test` image-caption corpus) |
| `configs/presets/callbacks/*.yaml`, `configs/presets/regularization/*.yaml` | callback sets and regularization blocks shared by several experiments |

Launchers (`run_train.sh`, `torchtitan/models/gift_moe/train.sh`,
`torchtitan/models/flux/run_infer.sh`, `scripts/generate/run_llama_generate.sh`,
the integration suites, `gift_scripts/`, the CI submitter) take an experiment
YAML as `CONFIG_FILE=` or the first positional argument, followed by dot-path
overrides.

## The schema is the runtime config

A YAML document decodes into typed schemas that runtime owners expose as
`Config`. Trainer sections decode field-for-field into `Trainer.Config`:
section names are its field names
(`training.steps`, `optimizer.lr`, `parallelism.tensor_parallel_degree`, ...),
so the same path works in YAML, in CLI overrides, and in logs. Decoding is
strict: unknown keys, wrong types and violated `__post_init__` invariants fail
with the full config path (`optimizer.weigth_decay: unknown key. Did you mean
'weight_decay'?`). Strict means no coercion beyond `int` -> `float` and a lone
scalar standing for a one-element list: a fixed-length tuple such as
`window_size: [left, right]` needs exactly that many items, a `dict` keyed by
`MeshAxisName` rejects an unknown axis, and a field the code sets itself
(`model_spec`, declared `init=False`) is not a YAML key at all.

Two extra top-level keys exist only in YAML:

* `config_version: 1` — required in every experiment file (not via imports).
* `model:` — replaces `model_spec`, which is always built from it (below).

The base trainer requires both `dataloader` and `loss`. Import a complete
experiment or provide these sections explicitly: a model preset plus the
`train/debug.yaml` mixin supplies the model and dataloader, but not the loss.

Swappable components are chosen by a stable `type` name, declared on the
config class itself (`@config(type="hf_text")` under `schema/`; the family is
the nearest `@config(family=...)` base in its MRO):

```yaml
dataloader: {type: hf_text, dataset: c4}     # or streaming_text, hf_chat, ...
loss:       {type: mtp, loss_weights: [0.1]}
```

A mapping without `type` keeps the class of the field's default (falling back
to the family's default type), and its keys are layered on that default
instance; see "Partial override" under [CLI overrides](#cli-overrides).

Plain dataclass values use the same rule where a choice exists: a rope's
`scaling` is `{type: llama}` / `{type: yarn, rope_factor: 32.0, ...}` (`none`
when absent), and a module config's `param_init` maps parameter names to init
specs picked by `type`
(`{weight: {type: normal, std: 0.02}, bias: {type: zeros}}`), and a sharding
placement per mesh axis is picked by `type` with no default
(`{tp: {type: shard, dim: 0}, cp: {type: replicate}}`; `{}` is an error, not
`Replicate()`). A model
Config's `materialize()` fills `param_init` on the configs it derives; on a
config it passes through unchanged (`rope`, the inner-attention kernels, flux
`autoencoder`) a YAML `param_init` reaches the module. Explicit initializer
names are checked against the module's registered parameters when it is
built; a misspelled name raises `ValueError` instead of being ignored.

Where a component needs code the user chooses, that code is itself a typed
component with its own `type` table rather than an import path. The chat
loader's sample processor is the case in point:

```yaml
dataloader:
  type: hf_chat
  dataset_path: openai/gsm8k
  load_dataset_kwargs: {name: main, split: train}
  sample_processor: {type: split_reasoning, separator: "####"}   # or question_answer
```

## The model is fully described in YAML

`model.config` is the model's module tree, written with the fields of the
modules' own Config classes. Repeated layers are named once under `blocks` and
placed by `layers`:

```yaml
model:
  label: 3b               # display label (ModelSpec.flavor)
  config:                 # the model's own Config; `type` picks the class
    type: gift_3_5
    dim: 1280             # every block, the embedding and the norms take it
    vocab_size: 128256    # the embedding and LM head take it
    norm: {type: gated_rmsnorm, eps: 1.0e-6, zero_centered: true}
    rope: {type: complex, dim: 64, max_seq_len: 4096}
    blocks:               # name -> TransformerBlock.Config, field for field
      gdn_moe:
        attention: {type: gdn, num_k_heads: 8, num_v_heads: 16, ...}
        moe: {experts: {hidden_dim: 896, num_experts: 64}, router: {top_k: 6}, ...}
        attention_norm: {type: gated_rmsnorm, eps: 1.0e-6, zero_centered: true}
        ...
      mla_moe: {...}
    layers:               # the stack, front to back
      - {block: gdn_dense}
      - {block: gdn_moe, repeat: 2}
      - {block: mla_moe}
      - {cycle: [gdn_moe, gdn_moe, gdn_moe, mla_moe], repeat: 2}
    mtp:                  # optional MTP heads, placed the same way
      layers: [{block: mtp}]
    converters:           # optional model-config converters (FP8 is the `fp8:` section)
      - {type: lora, rank: 8}
    hidden_z_loss_patterns:   # module FQN globs, or <fqn>.<projection> where a
      "layers.*.attention.q_latent": 0.0   # module exposes projection coefs
```

A block is written with the fields its Config class has: a field added to
`ResidualTransformerBlock.Config` (or to any module under it) is a YAML key at
the same path, with no model-family code in between. `layers` entries are
`{block: <name>}`, `{block: <name>, repeat: N}` or `{cycle: [<names>],
repeat: N}`; an entry may also be a block config written in place.

`model.config` decodes straight into the model's `Config` class: the base
`BaseModelConfig` is the `model` component family, so `type` picks the class
like any other component (`llama3`, `qwen3`, `qwen3_vl`, `deepseek_v3`,
`llama4`, `gpt_oss`, `flux`, `gift_moe`, `gift_3_5`, and the experiment
types `ft/llama3`, `graph_trainer/<model>`). A model type exists for each
runtime model class and nothing else: dense and MoE stacks, or GIFT 3.5 and
GIFT 4, are the same type with different presets. The classes live in
`torchtitan/config/schema/models/<family>/model.py`; a type's `family`
(`gift_moe` for every GIFT type) is `ModelSpec.name`, which backends and logs
key on, and `ModelSpec.flavor` is the preset's `label`, which names a
variant where there is one (`gift_4/700b`, `qwen3_moe/30B-A3B`).

`materialize()` turns the written config into the tree the runtime builds,
the same way for every model: each `layers` position gets an independent copy
of its named block, the decoder's `dim` and its index; the block completes its
sublayers' widths; the embedding, final norm, LM head and MTP heads are
completed from `dim` and `vocab_size`; linear attention without a mask type
takes the one the other layers share; the init policy fills every module
without a `param_init`; the rope contract is pushed to the attentions; and
`hidden_z_loss_patterns` apply to the placed tree's FQNs. Materialization is
pure and torch-free. `converters` are applied separately by the runtime
builder. `build_model_spec` materializes once and `ModelSpec.model` is that
tree; it carries a `materialized` flag, is internal (not serialized as YAML),
and is what runtime constructors, `prepare_config` and validation consume.

Component completion leaves shared input configs unchanged. Edit the partial,
written config before materialization to derive widths and masks from new inputs.
Python `_copy` retains values already computed; reset dependent fields to `None`
when they should be inferred again. Explicit values remain authoritative and
incompatible shapes are rejected.
Frozen/LoRA wrappers, initialization and runtime precision metadata survive copies.
Serializing a completed component writes its resolved values: after reload those
values are explicit, so later copies keep them unless reset to `None`.
A converter's single `validate_pipeline` hook receives the pristine tree,
preceding converters and optional trainer options, without importing its
runtime. To change a block, edit it under `blocks` before materialization; to
change one placed layer, edit `ModelSpec.model.layers`.

To freeze existing modules, select their runtime module FQNs:

```yaml
model:
  config:
    converters:
      # Freeze both layers without changing checkpoint parameter names.
      - type: freeze
        selectors: [layers.0, layers.1]
```

Selectors support globs such as `layers.*.attention`; each selector must match
at least one module. The whole model root cannot be selected. A selected module
and all its descendants have `requires_grad=False`; forward execution is
unchanged. Overlapping selectors and repeated application are harmless.
Converters run in list order: freezing a layer also freezes any LoRA adapters
inside it, whichever of those two converters runs first. See
[`gift_moe_debug_frozen.yaml`](../configs/experiments/gift_moe_debug_frozen.yaml).

LoRA requires a positive integer `rank` and finite `alpha`; zero and negative
`alpha` remain valid scales. RoPE requires finite `theta > 1`. Llama scaling
factors must be finite and positive, with `high_freq_factor >= low_freq_factor`
(equal bounds remain supported for Llama4). YaRN requires finite `rope_factor > 1`
and finite `beta_fast > beta_slow > 0`. Invalid values fail during YAML decoding,
before submission or run-bundle creation.

Only one LoRA converter is supported. Dense FP8 and LoRA must select disjoint
linear projections; their overlap is rejected before submission. Sparse-only
FP8 and freezing remain compatible with LoRA. The runtime also rejects repeated
LoRA conversion and dense FP8 replacement of specialized linear schemas rather
than discarding their adapter fields.
Active HF checkpoint export also rejects unmerged LoRA adapters. Use the native
checkpoint format for adapter training; an empty LoRA target list only freezes
base projections and remains compatible with HF export.

FP8 settings belong exclusively in the top-level `fp8:` section. These module
flags are derived runtime state and are rejected in YAML, including `null` resets:

| Module field | YAML setting |
|---|---|
| FFN `use_shared_quantizer` | `fp8.dense_use_shared_quantizer` |
| FFN `dense_fused_swiglu_quant` | `fp8.dense_fused_swiglu_quant` |
| Experts `use_float8_grouped_gemm` | `fp8.sparse_use_grouped_gemm` |
| Experts `deep_gemm_num_sms` | `fp8.sparse_deep_gemm_num_sms` |
| Experts `sparse_fused_swiglu_quant` | `fp8.sparse_fused_swiglu_quant` |

**Bundle compatibility:** earlier `resolved_config.yaml` files included these
module keys even at their defaults. They now fail validation. Remove the five
module fields, configuring any desired overrides under `fp8:`, or regenerate the
bundle from the experiment YAML. New bundles serialize only that FP8
settings surface. Runtime application overwrites derived flags on every call;
changing the dense projection selection requires a fresh materialized tree.

Runtime `prepare_config` and `validate_typed_config` require a materialized
Config. The written form
is what `resolved_config.yaml` stores and what pre-submit decodes (and
materializes with torch blocked). A schema's `_runtime_spec` is an import
identifier for an `ArchRuntime` constant in
`torchtitan/models/<family>/versions.py`. Only the runtime builder imports it
to obtain parallelize / pipelining functions and the state-dict adapter.
The runtime owner implements `prepare_config` (parallelism, sharding and FP8)
and `get_nparams_and_flops`; these methods do not live in the schema.

Models declare `supported_sharding_backends` as class metadata. The default is
`("fsdp2",)`; a model advertises another backend only after its runtime
parallelization hook implements it. Both pre-submit and runtime backend
construction check that declaration. `build_sharding_backend` accepts the model
config (`ModelSpec.model`), rather than a family-name string. Parallel-degree
geometry is likewise shared by the schema and runtime `ParallelDims`; invalid
degrees or world-size combinations raise `ValueError` before creating a mesh.

Canonical size presets live in `configs/presets/models/` and are the only
place a named architecture's hyperparameters live. An experiment imports one
and overrides what differs:

```yaml
imports:
  - ../presets/models/gift_moe/gift_3_5/700b.yaml
model:
  config:
    blocks:              # the preset's block names
      mla_moe: {moe: {experts: {token_dispatcher: {type: deepep}}}}
      gdn_moe: {moe: {experts: {token_dispatcher: {type: deepep}}}}
    mtp:
      layers: [{block: mtp}]
```

An override reaches every block it means by name: a preset's block names are
part of its interface, and a block the override does not name keeps its own
settings.

## The model config convention

How a model Config and its presets are shaped. Three properties, in priority
order: unambiguous (no key means two things, no invalid model is
expressible), readable (a preset shows the model's structure), uniform (one
family's preset teaches you every other's).

> **The YAML is the module tree. A Config class exists for a module the runtime
> builds, or for a value it derives; there are no family templates, layout
> flags or default-only subclasses.**

Every decoder (`DecoderConfig`,
`schema/models/common/decoder_blocks/decoder.py`) has `dim`, `vocab_size`,
`rope` (one positional scheme shared by every layer), `norm` (the final norm,
also the MTP heads'), `init` (the init policy, from an init preset),
`blocks`, `layers`, the optional `tok_embeddings`, `lm_head` and `mtp`, and
`hidden_z_loss_patterns` and `converters`. A model class adds only what its
runtime class has (`enable_weight_tying` for the tied decoders,
`vision_encoder` and `mrope_section` for qwen3_vl) and its validation. The
stack's structure -- which layers are dense or MoE, where full attention or
NoPE layers stand, sliding windows -- is the `layers` list, not a flag:

| model | blocks in its presets |
| --- | --- |
| `llama3`, `qwen3` | `dense` (or `moe`) |
| `qwen3_vl` | as qwen3, plus the `vision_encoder`'s own `blocks` / `layers` |
| `deepseek_v3`, `gift_moe` | `mla_dense`, `mla_moe` |
| `gift_3_5` (GIFT 3.5 and 4 presets) | `gdn_dense`, `gdn_moe`, `mla_moe`, and the MTP head `mtp` |
| `llama4` | `dense`, `moe`, `nope_dense`, `nope_moe` (`type: llama4_chunked` blocks) |
| `gpt_oss` | `moe`, `sliding_moe` (`type: gpt_oss_sliding` blocks, `gpt_oss_sink` attention, `gpt_oss_biased` MoE) |
| `flux` | `double` (`type: double_stream`), `single` (`type: single_stream`), placed by `double_blocks` / `single_blocks` |

```yaml
# qwen3 8B
config:
  type: qwen3
  dim: 4096
  vocab_size: 151936
  norm: {type: rmsnorm, eps: 1.0e-06}
  rope: {type: cos_sin, dim: 128, max_seq_len: 4096, theta: 1000000.0}
  blocks:
    dense:
      attention: {type: gqa, n_heads: 32, n_kv_heads: 8, head_dim: 128, qk_norm: {eps: 1.0e-06}}
      feed_forward: {hidden_dim: 12288}
      attention_norm: {eps: 1.0e-06}
      ffn_norm: {eps: 1.0e-06}
  layers:
    - {block: dense, repeat: 36}
```

The optional `lm_head` selects a typed output projection. Omit it to keep the
model's plain linear head, or select a soft-capped head explicitly:

```yaml
model:
  config:
    lm_head:
      # Dimensions come from dim and vocab_size; conflicting widths fail.
      type: unembedding
      logit_softcap: 30.0
```

`type: linear` selects the ordinary projection. Both types support the existing
linear configuration fields, including `bias` and `param_init`. The default
initialization role remains `lm_head`. GIFT4 requires an explicit unembedding
head with a positive `logit_softcap` and plain (not zero-centered) norms.

**Component settings.** A block's fields are its sublayers' component configs
(`attention:` a `GQAttentionConfig`, `MLAConfig`, `GatedDeltaNetConfig`, ...,
`moe:` a `MoEConfig`), written without the widths the owner supplies (`dim`,
a norm's `normalized_shape`, a router's `gate`) and the init. Attention is a
component family (`gqa`, `mla`, `gdn`, `kda`, `gpt_oss_sink`), so a block switches
kernel in YAML (`attention: {type: kda, conv_size: 4, ...}`). A preset nests
what the runtime nests -- `moe: {experts: {hidden_dim: 2048, num_experts:
256}, router: {top_k: 8, score_func: sigmoid}, shared_experts: {hidden_dim:
2048}}` -- and any component setting is reachable with no extra field or
builder argument for it: `attention: {o_norm: {eps: 1.0e-5}}`, `attention:
{inner_attention: {type: varlen}}`, `attention: {qkv_linear: {type: fused}}`,
`moe: {experts: {token_dispatcher: {type: deepep}}}`. Settings that differ
per layer are different blocks.

**What a `None` means.** A field's annotation says what its `None` stands for
(`torchtitan/config/configurable.py`):

- `OwnerFilled[T]` -- a width or position the owning config fills (`dim`,
  `layer_id`, `normalized_shape`, `num_embeddings`, a Linear's `in_features` /
  `out_features`, ...). YAML may write one and the owner checks it agrees, but
  it remains available in suggestions and is labeled "filled by owner" in
  "Valid keys". `resolve_child` fills only these (`dims=`); a value the owner
  merely requires, such as a `bias` bool that has no unset, goes in `require=`
  and is checked, never filled.
- `Derived[T]` -- "compute it": a projection, norm or kernel the config
  completes when unset (`wo`, `kv_norm`, `qkv_linear`, `token_dispatcher`,
  ...). YAML may set it.
- no marker -- `None` is a value: the module is absent (`gate`, `mtp`, ...),
  or the config itself picks which of its fields a configuration needs (MLA's
  `wq` versus `wq_a` / `wq_b` / `q_norm`, by `q_lora_rank`), or the consumer
  applies a documented default (GQA's `head_dim` / `n_kv_heads`, RoPE's
  `cache_len`). Such defaults need not be stored in the completed config.

After `materialize()` every field marked `OwnerFilled` / `Derived` must be
set; the shared check in `validate_materialized` lists all missing fields in
schema order. A role is the outermost annotation, defaults to `None`, and belongs
to a `ModuleConfig`; `OwnerFilled[T] | None` is invalid because the alias already
includes `None`.
Unmarked fields, the conditional ones included, are not checked by it. Code
that inspects a field's type unwraps `Annotated` first (`decode`,
`schema_paths.concrete_schema`).

**Capabilities.** Restrictions that follow from a module are declared on its
Config class and `DecoderConfig.validate_parallelism` checks them over the
tree for every model: `supports_tensor_parallel` / `supports_context_parallel`
on attentions (linear attention supports neither) and blocks (`llama4_chunked`
has no CP path), an inner kernel's `supports_context_parallel`, a softmax
attention's `context_parallel_kernels` (MLA: SDPA only), and MTP heads, which
exclude sequence parallelism and CP. A model class adds only restrictions of
its own runtime; HF export constraints are covered under "Validation stages".

MTP with TP, PP or EP requires a model runtime that implements head sharding;
currently only the GIFT runtimes do. Serial and data-parallel MTP remain available
elsewhere, subject to the shared SP/CP restrictions. GPT-OSS sink attention
requires a Flex kernel. These contracts are checked before training.

**Init.** Every module config carries `param_init`, parameter name -> init
spec (`{weight: {type: trunc_normal, std: 0.02}, bias: {type: zeros}}`; the
`param_init` family: `normal`, `trunc_normal`, `zeros`, `ones`,
`xavier_uniform`, `kaiming_uniform`, `skip`, `depth_scaled` and `dim_scaled`
with a `base` spec, `norm_unit_scale`, and the families' own). A parameter
left uninitialized is `{type: skip}`; an entry of a default table cannot be
removed with `null` (see the `null` rule below).
Normal initializers require a finite mean and finite, nonnegative standard
deviation; truncated normal requires a positive standard deviation and finite
ordered bounds. Normalization epsilon must be finite and positive.
`norm_unit_scale` is a norm weight that starts the norm at scale one: zeros
for a zero-centered norm (scale `1 + weight`), ones for a plain one, read off
the norm's own `zero_centered`, so flipping that needs no init change. A
decoder Config holds the model's init policy, `init:`, one table per role
(`linear`, `linear_out`, `norm`, `gated_norm`, `embedding`, `lm_head`,
`router_gate`, `experts`, `conv`, `o_norm`, `gated_delta`, `sinks`); each
module says which role it plays (`init_role`: unset, the owner's choice for a
position such as a router's gate, else the class default; a projection
restated in YAML keeps the owner's choice unless it names a role), and
`materialize()` gives every module without a `param_init` the policy's table
for its role. A nonempty module table overlays named entries on that role
table. An empty `param_init: {}` is invalid; use an explicit initializer such
as `param_init: {weight: {type: skip}}` for each parameter to leave unchanged.
Omitting `param_init` or resetting it with `null` keeps role/owner inheritance.
Resolve `depth_scaled` with the
layer index and `dim_scaled` with the model width. Policies are presets under
`configs/presets/init/` (`depth_scaled` is the upstream torchtitan scheme the
llama / qwen3 / deepseek presets import; `gift_base`, `gift_hybrid`,
`gpt_oss`), imported by the model presets. A preset or experiment overrides
the policy by role (`init: {router_gate: {weight: {type: renorm_keep_std,
init: {type: normal, std: 0.006}}}}`, layered on the imported policy: the
`bias` entry stays) or one module in a block (`blocks: {dense: {attention: {wo: {param_init: {weight:
{type: zeros}}}}}}`).

Flux and Qwen-VL projections inherit their owner's initializer entries, with
explicit `param_init` entries taking precedence per entry: `param_init:
{weight: ...}` replaces only the weight initializer, and the bias, if the
projection has one, keeps the owner's. Supplied children must agree with their
owner's widths, `bias` and stream mode. A written `Linear` defaults to
`bias: false`, so restating a projection whose default has a bias, even only
its `param_init`, needs `bias: true` ("requires bias: true" otherwise).

Module children may be nested in lists and dictionaries. Traversal, validation,
copying and converters follow the same container paths. Tuples are reserved for
plain values such as dimensions; use lists for module children so converters can
replace positions. Converter templates are not part of the placed module tree.

## Trainer type

`config_version` must be the integer `1`; booleans and floating-point values
are not versions. `trainer` must be a mapping. If no section is present, the
default trainer is used. An empty `{}` is valid and preserves an imported
trainer; `null` removes the earlier section and restores the default.

The model's `trainer_type: ClassVar[str]` declares its runtime contract; the
trainer registry's `model_trainer_type` declares the accepted contract. Decode
checks the pair in both directions before importing runtime modules. The base
and RL policy-actor paths accept ordinary models; FT, graph and Flux require
their corresponding model variants.

`trainer: {type: ...}` selects which trainer runs the job; omitted, it is the
base `Trainer`. Each type maps to a schema import spec in
`registry.TRAINER_TYPES`. Accepted root sections are discovered from that
schema's init fields, excluding `model_spec`; no separate section table is
maintained. Most schemas extend `TrainerConfig`; RL has its own root schema.
`ft` (torchft fault tolerance) accepts a
`fault_tolerance:` section and pairs with the `ft/llama3` model type; its
`optimizer` and `checkpoint` are the ft ones (families of their own, `type: ft`
is the only and the default type; the base trainer cannot select them); see
`configs/experiments/ft_llama3_debug.yaml`. `flux` adds `encoder:` and
`inference:`. `graph_trainer` (graph-captured training) adds no section but
retypes `compile` (`compile.mode`, `compile.passes`, ...) and pairs with the
`graph_trainer/<model>` model types; layer
`configs/presets/train/graph_trainer.yaml` over a base experiment, as
`configs/experiments/graph_llama3_debug.yaml` does. `rl` (the GRPO controller in
`experiments/rl`) is not a `Trainer` subclass: it replaces the root section
set wholesale (`num_steps`, `env`, `validation_env`, `batcher`,
`policy_trainer`, `generator`, ...; `registry.TRAINER_TYPES["rl"].sections`), takes the
policy loss as `policy_trainer.loss: {type: grpo}` and the environments as
`{type: sum_digits, ...}`; see `configs/experiments/rl_grpo_qwen3_0_6b.yaml`.
Its runtime (Monarch, vLLM) ships in no image yet; pre-submit decodes and
validates those documents torch-free like any other, only building the
`ModelSpec` needs the runtime.

## Imports and presets

Every imported file has the shape of an experiment file: root sections only.
`imports:` takes one of two forms, and one file uses one form:

```yaml
# Full-config imports: a path or an ordered list of paths, merged at the root.
config_version: 1
imports:
  - ../presets/models/gift_moe/gift_3_5/700b.yaml
  - ../presets/train/debug.yaml
```

```yaml
# Section imports: a mapping section -> file. Only the named section of each
# file is taken; its other sections are ignored. Here: the optimizer of the
# 3.5 run, and the validator block of the llama3 debug experiment.
config_version: 1
imports:
  optimizer: gift_3_5_700b.yaml
  validator: llama3_debug.yaml
```

* Later imports override earlier ones (list order, or mapping order); the
  importing file overrides all of them. The whole import tree is flattened
  into one ordered list of files (depth-first, a file's imports before its own
  values) and merged in a single left-to-right pass, so a `null` or a
  wholesale replacement in an imported file acts on everything merged before
  it, including files imported earlier by the files above it.
* Mappings deep-merge; scalars and lists are replaced wholesale.
* `null` clears a value back to the schema default, wherever it appears (also
  in a section no import defined, and in a mapping inside a list, e.g.
  `converters: [{type: lora, rank: null}]` drops `rank`). A `null` is never
  stored: a mapping of only nulls under an absent key is not created
  (`gate: {gate_proj: null}` with no `gate` set earlier leaves the gate off),
  while an explicit `{}`, a `!replace` mapping, or one replacing a scalar
  becomes `{}`. A bare `null` list
  item (`keep: [1, null]`) is data and stays. A schema rule keeps that
  unambiguous: a field whose type admits `None` defaults to `None`
  (`tests/unit_tests/test_config_schema_rules.py`), so for such a field `null`
  means the default, which is `None`. A field whose "off" state is not the
  default spells it as a value instead (e.g.
  `parallelism.context_parallel_load_balancer: disable`).
* The same reset rule applies to open mappings (`dict[str, ConfigValue]` or
  `dict[str, Any]`), including external kwargs. For example,
  `--dataloader.load_dataset_kwargs.split null` removes an inherited `split`
  argument. Mapping values cannot use YAML `null` to pass Python `None` to an
  external API. Loading and merging do not depend on a field's schema or the
  component type selected by a later layer.
* In a mapping field with default entries (the init tables), a `null` entry
  set by a file earlier in the merge is removed with it, and the default's
  entry applies; a `null` on an entry only the schema default has is an
  error (`model.config.init.linear.bias: null cannot remove an entry of the
  default mapping`): the run's table could not be written to
  `resolved_config.yaml`. Turn an entry off by its value, e.g.
  `bias: {type: skip}`.
* A `null` that deletes nothing (no earlier file set the key) must name a
  field: `training: {stpes: null}` is an error, `training.stpes: unknown key
  (reset with null). Did you mean 'steps'?`. The loader reports such nulls
  (`RawDocument.missing_cleared_paths`, CLI overrides included) and the strict
  decode checks each against the class it decodes at the parent path, so a
  component's fields are those of its final `type` (the family default when
  `type` is reset). Under a mapping the data leaves out, the class is the
  field default's, else `X` of `X | None`, else the family default; under an
  absent entry of a closed mapping the reset is an error (`no entry to
  reset`). A reset of an ancestor supersedes the resets recorded below it.
  Mapping keys containing dots remain whole keys: a reset inside
  `hidden_z_loss_patterns: {"layers.*.attention.wo": {on: null}}` checks
  `on` in that pattern's value, including when no earlier layer set `on`.
  A `type: null` always passes. Keys of open mappings
  (`load_dataset_kwargs`, any `dict[str, Any]`) are not checked. A `null`
  that pops an inherited key is not checked either, which leaves one accepted
  gap: a key misspelled in a base file and cleared with the same misspelling
  in a child passes (the base on its own fails pre-submit, and the pair has no
  effect on the run).
* Paths resolve relative to the importing file; an imported file may itself
  import. A file reached twice through different full imports (a preset
  layered on a preset the experiment also imports, as `gift_4_debug.yaml`
  does with the gift_3_5 debug preset) is allowed while merging it again
  changes nothing. If a file merged between the two occurrences set one of
  its keys, the repeat would silently undo that: it is an error naming the
  file, both importers and the keys (import the file once, or set the keys
  after its last import, e.g. in the entry file). The same file named for two
  section imports is not a repeat. Cycles and missing files are errors, as is any file that cannot be
  read or parsed (a directory, invalid UTF-8, a non-mapping or non-string
  root key, an alias that makes the structure recursive). The error names the
  file, and for an import also the file that declares it.
* A section import of a file that lacks that section is an error naming both
  files. A file that sets the section to `null` does have it: importing that
  section imports the reset. A list item that is a mapping (`- optimizer: file.yaml`) is an error:
  to combine full imports with a section import, put the section import in a
  small experiment-shaped preset and import that fully.
* Duplicate explicit YAML mapping keys are errors at any nesting level,
  including imported files. Errors identify the key and both source locations.
  YAML merge keys (`<<`) may still supply defaults overridden by explicit keys.
* Values are read by the YAML 1.2 core schema, on purpose unlike PyYAML's
  default (YAML 1.1): only `true` / `false` are booleans (`yes`, `on` are
  strings), `010` is 10 and `0o10` is 8, `1e-4` is a float; `12:30`, dates
  and `1_000` are strings. Quote a string that YAML would read otherwise
  (`description: '2024'`); the decode error for a non-string in a `str`
  field says so.
* One file per section in the mapping form; to layer two presets into one
  section, let one preset import the other.
* No expression language and no interpolation — if a structure needs heavy
  editing, make a preset.

## Choose only the callbacks a run needs

`callbacks` is an ordered mapping of user-chosen names to typed configs. It is
empty by default; the detailed `monitored.yaml` preset is an optional debug
configuration, not a required catalogue to copy into every experiment.

For a small reusable set, import `../presets/callbacks/basic.yaml` from an
experiment. It includes system metrics, timers, trainer/data statistics and
two tensor statistics. Override `callbacks.monitoring.log_interval` to change
the logging interval; set `callbacks.monitoring.sources.optimizer: null` to
remove optimizer diagnostics. Lists replace as a whole, so a custom
`callbacks.monitoring.sources.modelling.metrics` list selects exactly the
statistics that run needs.

```yaml
# Keep a small callback set and rely on each schema's defaults.
callbacks: !replace
  system: {type: system_monitor, memory_log_interval: 10}
  timers: {type: timers_monitor, interval: 10}
```

`!replace` also lets an experiment replace an imported callback set with this
short mapping in the desired order. To keep the set and change one field, use
`--callbacks.system.memory_log_interval 20`; to remove an imported callback,
set its named entry to `null`. Dependencies must precede the callbacks that
consume them. Within the mutator and observer groups, insertion order controls
execution; keep the timers observer last when it reads other callbacks' timers.

Pre-submit checks the dynamic LR rescaler's dependency by callback type: exactly
one monitoring callback must precede it, regardless of the entry's name.
Final HF saves require `checkpoint.last_save_model_only: true`; final EMA saves
require exactly one EMA callback. These final-save checks are inactive when
checkpointing is disabled or `checkpoint.load_only: true`.

The EMA callback's `use_ema` flag (default `true`) selects averaged weights for
validation, HF export and Autometrics. Before the average starts, live weights
are used. Set `callbacks.<name>.use_ema: false` to keep publishing live weights.

Autometrics checks its own task list and inference resource settings before
submission. Requirements involving checkpoint settings, the backend, local
files or the submission service are still checked at runtime. The validator's
`freq` must be a positive integer. Modelling metrics validate the executed
plan (`effective_steps()`); a reduce stage with no group or `op: none` does
not satisfy the reduction requirement.

Autometrics tasks accept `num_fewshot: 0` or `num_fewshot: [0, 10]`.
Additional PAM serving flags go in `inference_options`, for example
`inference_options: {mem-fraction-static: 0.7, chunked-prefill-size: 4096}`.
These options cannot override the dedicated topology or request-limit fields;
PAM validates the backend-specific flags.

Pre-submit also checks global batch divisibility by local batch times DP degree
(when DP is known), active PP microbatch divisibility, and the model's sequence
partitioning requirements. Unknown `fim_rates` keys are rejected both globally
and in individual streams.

## CLI overrides

```bash
python -m torchtitan.train exp.yaml --training.steps 2000 --optimizer.lr 5e-5
```

Each `--a.b value` is one more YAML layer, `{a: {b: value}}`, merged after
the experiment file and its imports by the same fold, in command-line order.
So `--x v` means what `x: v` means in a file imported last: any field is
reachable (`--model.config.dim 512`, `--dataloader.type streaming_text`), in
any order, since nothing is typed before the strict decode, and a later
override of the same path wins.

| Value | Reads as |
| --- | --- |
| `10`, `5e-5`, `true`, `c4_test`, `./out` | YAML: int, float, bool, string, string |
| `null` / `~` | resets the key to its default, as a YAML `null` does |
| `a,b` (a bare comma) | the list `[a, b]`; spaces around the items are dropped |
| `[a, "b,c"]`, `{steps: null, seq_len: 64}` | a YAML flow list / mapping; a mapping deep-merges into the section and a `null` inside resets only its key |
| `"'2024'"`, `"'a,b'"`, `"''"` | a quoted YAML string (outer quotes for the shell, inner for YAML), needed where YAML reads the text as something else: `2024`, `true`, `null`, `0x10`, a comma, a leading `#` |
| a bare `--x.flag` | `true` |

The decode judges the value as it judges a file value: `--dump_folder 2024`
fails with `dump_folder: expected a string, got 2024; quote it ('...') if it
is meant as text`, and an unknown path or a `--x.typo null` fails with the
full path and a suggestion. An empty value, a value YAML reads as a comment
(`#tag`) and a path with a list index (`model.config.converters[0].rank`)
are errors. A dot-path cannot address a list item or a key that itself
contains dots (the `model.config.hidden_z_loss_patterns` globs): those, and
other structured edits, belong in YAML.

**Partial override of a section the document does not spell out.** A
mapping is layered on the field's Python default, the way a YAML layer is
layered on the layers below it: the keys it leaves out keep the default's
values, and a component field without `type` keeps the default's class. The
validator's dataloader defaults to
`HuggingFaceTextDataLoader.Config(dataset="c4_validation", infinite=False)`, so
`--validator.dataloader.num_workers 2` (or with `--validator.dataloader.type
hf_text`) keeps that dataset and `infinite: false`. A `type` naming another
class uses that class's defaults for omitted fields; imported YAML keys still
deep-merge across a type switch. Switching to a sibling component emits a
structured warning with retained keys and source files; switching to the same
class or a subclass does not. Warnings appear in the validation report, warning
count, and container logger.
If a retained field fails decoding, the error itself includes its type-switch
and replacement guidance, including in programmatic loaders. Unrelated errors
and explicitly supplied new fields retain their ordinary diagnostics.

**Replace a mapping in the same file.** `!replace` discards earlier YAML values
at that path. Schema defaults still apply to fields omitted from the replacement.
Later layers can extend the resulting mapping normally.

```yaml
# Replace the imported dataloader instead of inheriting its dataset/options.
dataloader: !replace
  type: hf_chat
  dataset_path: json
  load_dataset_kwargs: {data_files: tests/assets/sft_test/data.json, split: train}
  sample_processor: {type: question_answer}
```

The marker also works on nested mappings and CLI values:
`--dataloader '!replace {type: hf_chat, dataset_path: json, sample_processor: {type: question_answer}}'`.
The existing two-step CLI reset remains supported:
`--dataloader null --dataloader '{type: hf_chat, ...}'`.
A `null` resets a section to the field's default, not the class defaults.


To inspect the selected attention, mixer and norms on each materialized decoder
layer without building a model, use `config-check <experiment.yaml>
--print-layers`. This reports actual component choices after applying the
model's `layers`.

## Validation stages

Pre-submit also verifies the resolved YAML round trip without building runtime
objects. Supply `config-check config.yaml --world-size N` (or the `world_size`
keyword of `run_pre_submit`) to check parallel degrees against the planned job.
Without that argument the world size is unknown; the local shell's process count
is not assumed to describe the future job.

Llama 3 and Qwen 3 runtimes currently prepare GQA sharding; other attention
families require a runtime that implements their sharding contract. Qwen-VL's
vision tower currently excludes context parallelism. Dense FP8 selection rejects
specialized projections with fields that its replacement cannot preserve.

An HF export rejects what the target format cannot hold instead of dropping
it, in two layers that run only when checkpointing is enabled, `load_only` is
false, and `save_in_hf`/`last_save_in_hf` is set (or in `convert_to_hf.py`).
Inherited save flags have no effect in load-only mode; checks for an actual
HF load remain separate. The state dict adapter is the source of
truth for state: the checkpointer round-trips the model's keys and shapes
through it at construction, so Llama 3 QK norms, LoRA adapters, MTP heads or
Qwen MoE expert-bias balancing (disable with `load_balance_coeff: 0`) fail there.
Before submission, the model's `validate_hf_export` checks from the config
alone what that round-trip cannot see: decoders require affine RMSNorms, one
attention/FFN/MoE width and MoE routing for all layers, and no residual scale
below 1 at an HF save (the first HF save must come after the warmup;
`convert_to_hf.py` cannot know the step and only warns), and GIFT validates its
`config.json` contract. Loading from HF accepts such a model; entries missing
from the HF format keep their init, with a warning.

Checkpoint intervals must be positive, and retention is either 0 (keep all) or
at least 2. Scheduler warmup is nonnegative and an explicit horizon is positive;
after that horizon, decay stays at its final factor rather than becoming negative
or starting another cosine cycle.

Model diagnostics name the original YAML field when its dependency
is known, for example `model.config.blocks.dense.attention.n_heads` rather
than a placed layer's path. The report's `source` identifies the file or CLI
override that won the merge; `runtime_path` retains the materialized location.
Source metadata is not written into `resolved_config.yaml`. Reloading that
file makes it the new source of its values. Metadata is also excluded from
`dataclasses.fields()` so runtime converters see only configuration data.
Repeated diagnostics for one shared template field are reported once. Invalid
FFN widths and implicit attention head widths fail at their original YAML field
before runtime modules are built.

One engine, two stages (`validation.py`):

* **pre-submit** (`config-check` / `scripts/config_check.py`): YAML syntax and
  imports, then the strict typed decode of the document (`config_version`,
  trainer type, unknown top-level sections, a known model `type`) and of
  every root section into the torch-free schema tree
  (`torchtitan/config/schema/`: unknown keys, types, component `type`
  dispatch, `__post_init__` invariants), the materialization of
  `model.config` into the full model config tree (every
  `ModuleConfig.__post_init__`, e.g. the QKV projection shapes; converters
  are not run, but their pure validation hooks check constraints such as freeze
  selector matches), and the typed `validate(ctx)` pass over both. Runs
  with only Python + PyYAML, whether or not torch is installed; only the `ModelSpec`'s
  parallelize / pipelining callables wait for the container. `--print` (an alias
  of `--print-merged`) dumps the merged document; `--print-resolved` includes
  typed defaults. The CI runs it over every experiment YAML
  (`lint:config_check`), and `gift_scripts` / `job_submitter` run it before
  submitting a job.
  Pre-submit rejects nonpositive router top-k, top-k above the expert count,
  non-finite or negative z-loss coefficients, scheduler ratios outside `[0, 1]`,
  and nonpositive metric log intervals. Model owners also check dimensions before
  arithmetic: FLUX heads and positional axes, its packed VAE channel widths, and
  Qwen3-VL vision heads, position grid, MRoPE slices and vision-to-text output width.
  Shared experts use an explicit hidden width and must consume the routed experts'
  input width; their hidden width need not be a multiple of a routed expert's width
  for training. HF export requires that multiple because `n_shared_experts` is an
  integer count, and rejects a width it would otherwise round down.

* **in-container** (`torchtitan.train`, before anything is built): the
  strict decode including the model config and its materialization (with
  converters), then environment checks on the decoded config
  (`preflight.py`: writable `dump_folder`, `hf_assets_path` present, visible
  accelerators, backend packages for the dispatchers in the model tree), and
  finally `validate_typed_config`
  with the runtime (`WORLD_SIZE` vs parallelism degrees in
  `TrainerConfig.validate`), the typed pass pre-submit runs too: every
  Config node's `validate(ctx)` over the decoded tree and the materialized
  model tree (under `model.`), then the trainer Config's `validate_model(tree)`
  (heads vs TP, experts and the DeepEP/HybridEP dispatcher vs EP, MTP heads
  vs loss weights, seq_len vs rope, vision patches vs dataloader), issues
  accumulated, not raised one by one. Both
  stages report the same issues at the same paths, except what only the
  runtime knows (e.g. flex `BACKEND=FLASH` on a pre-Hopper GPU) and what the
  converters change, which only the in-container stage sees.
  The runtime owner's `prepare_config` reuses the same pure model checks
  (`schema/models/common/cross_section.py`) before applying runtime settings.

A Config keeps its invariants in two places: `__post_init__` for what one node
decides on its own (raise `ValueError`, the decoder adds the path) and
`validate(ctx)` for what needs the stage, the runtime or a sibling section
(return `Issue`s with paths relative to the node).

## Schema and runtime

Buildable `Configurable.Config` subclasses live in `torchtitan/config/schema/`,
a torch-free tree that mirrors `torchtitan/` module for module (`schema/models/llama3/model.py`
holds the configs of `torchtitan/models/llama3/model.py`). The rules, each
checked by `tests/unit_tests/test_config_torch_free.py`: everything under
`torchtitan/config` imports with torch blocked; every registry entry points
into `schema/`; no `@config` class outside `schema/`; every `__init__.py`
under `schema/` is empty; every `Configurable.Config` subclass is declared with
`@config` (`kw_only` + `slots`), never as a bare dataclass; every experiment and preset
materializes with torch blocked.

Shared value dataclasses without a runtime owner, such as `TrainingConfig`,
live in `configs.py`. They hold data and do not build runtime objects.

A class that builds a runtime object declares its config in the schema module
mirroring its own, named `<Owner>Config`, with `_owner_spec =
"module:Owner"`; the owner keeps `Config: TypeAlias = <Owner>Config` so
`Owner.Config(...)` and `config.build()` read as before, and the owner module
is imported on the first `build()`, never by decoding. Every buildable config
class must declare its own `_owner_spec` or have its runtime `Config` alias
bind that exact class. A new subclass never silently builds the parent's owner.
An intentional schema-only variant can explicitly reuse the runtime:
`_owner_spec: ClassVar[str] = ParentConfig._owner_spec`. Values are data, not
torch objects: parameter initializers are `InitSpec`s (`TruncNormal`,
`Normal`, `Zeros`, `Ones`, `Skip`, ...; a family-specific spec registers its
resolver in the runtime module mirroring the spec's), sharding placements
are `Shard(dim)` / `Replicate()` / `Partial()` specs resolved to DTensor
placements at parallelize time, and a z-loss is its coefficient. A model
family splits the same way: its model Config class (runtime fields, init
policy default, validation) in `schema/models/<family>/model.py`, `_runtime_spec` names the runtime metadata in
`torchtitan/models/<family>/versions.py`, and runtime methods live on the
model owner in `torchtitan/models/<family>/model.py`.

## Reproducibility

Rank 0 of every `torchtitan.train` run writes a bundle:

* `resolved_config.yaml` -- the full typed config the run used: every section
  and every field with its effective value, defaults included, component
  `type`s named, `model.config` decoded (every field of the model Config as
  written, `blocks` and `layers` included, not the materialized tree). Rerunning it does not depend on
  presets or non-`None` schema defaults staying as they were; it is checked,
  before writing, to decode back to the same config. Fields at `None` are left
  out (on load a `null` is a reset to the default, so writing one would change
  nothing): if such a field's default changes later, a rerun takes the new
  default.
* `source_config.yaml` -- the merged document as the files and the command
  line wrote it (only the keys they set), imports resolved and resets
  applied: the minimal config of the run.
* `overrides.yaml`, `imports.yaml`, `preflight_report.yaml`,
  `metadata.yaml` (git commit, actual process argv, selected launch environment,
  package versions, world size).

The model is recorded as written (`model.config` in both config files): the
placed layer stack is derived from it by `materialize()`.

A field or component `type` renamed or removed later makes an old
`resolved_config.yaml` fail loudly (unknown key or type, with a suggestion)
instead of silently taking a new default. Rerunning a flux bundle warns that
`training.seq_len` is ignored: every path is spelled out.

Each launch gets its own
`<dump_folder>/config/launches/<YYYYmmdd-HHMMSS>-<host>-<random>/`, so a
relaunch or resume into the same `dump_folder` keeps the earlier launches'
provenance; `<dump_folder>/config/` itself mirrors the latest launch. Files are
written atomically. All ranks validate serialization before runtime setup; rank
zero writes the bundle. A failure exits the worker, and the launcher terminates
its siblings. Under torchft every replica group has its own rank 0 and
writes its own launch directory; the top-level mirror is then last writer
wins per file (the copies differ only in `replica_id`, host and pid).

The checkpoint conversion scripts take `--config <yaml>`; pass the run's
`resolved_config.yaml` so the rebuilt model matches the checkpoint exactly.

## Golden snapshots

`tests/unit_tests/golden_configs/` contains readable YAML for every experiment
and model preset. Each snapshot records the written model and its materialized
module tree; experiments also record the typed trainer sections. Unset fields
remain present as `null`, so adding or removing an optional field produces a
visible diff. These checks run without importing the training runtime. Runtime
bindings and converters have separate constructor tests.

Snapshots use standard YAML anchors and aliases to store exactly identical
subtrees once per file. There are no templates, merge defaults, custom tags,
or cross-file references. Loading a snapshot with `yaml.safe_load` restores
the complete structural value, including every explicit `null` and list entry.
Anchor names include a readable type label and a content digest to avoid
renumbering unrelated aliases when a value changes; the values remain visible.
Tests compare expanded JSON and show expanded diffs on failure. The diagnostic
command below also keeps its full JSON output.

A changed snapshot shows which defaults, architecture, or decoded sections
changed. New recipes require new snapshots; deleted recipes must remove theirs.
Review snapshot diffs before accepting an update, and explain intentional
behavior changes in the commit message. A snapshot update is not numerical proof.

```bash
# see what changed: dump the tree on both checkouts and diff
python -m tests.unit_tests.config_golden experiments/llama3_debug.yaml > before.json
# after reviewing an intended change, update the readable snapshots
TORCHTITAN_UPDATE_GOLDEN=1 python -m pytest tests/unit_tests/test_experiment_parity.py \
    tests/unit_tests/test_model_arch_presets.py
```

A GPU-level proof for a change that should not alter numerics is an A/A run
before vs after with `--debug.deterministic --debug.seed=42`, comparing the
TensorBoard scalars (`scripts/loss_compare.py`).

## Module map

All under `torchtitan/config/`.

| module | role | torch |
| --- | --- | --- |
| `configs.py` | the shared config dataclasses (`TrainingConfig`, `ParallelismConfig`, `CompileConfig`, `FP8Config`, ...) | free |
| `configurable.py`, `function.py` | `Configurable` (a class whose `Config`, declared with `@config`, `build()`s it), `Config.owner()` / `validate(ctx)`, and `Function` (a callable-valued config) | free |
| `schema/` | the torch-free typed tree, mirroring `torchtitan/` module for module: every config class, the model Configs, the value specs | free |
| `sections.py` | top-level section names, `ModelSection` (`label` + the typed `config`), `CONFIG_VERSION` | free |
| `loader.py` | YAML + imports + deep merge | free |
| `overrides.py` | dot-path CLI overrides on the raw document | free |
| `decode.py` | strict dict -> dataclass decoding, `type` dispatch | free* |
| `registry.py` | `import_schema` (one walk registers every `type`, model types included), `family_for_base`; `TRAINER_TYPES`, `ArchRuntime` | free |
| `validation.py` | issues, reports, stage context; `validate_config_tree` / `validate_typed_config` over the decoded tree | free |
| `errors.py` | `ConfigError`, `ConfigFileError`, `ConfigSchemaError` | free |
| `pre_submit.py` | `config-check` entrypoint | free* |
| `preflight.py` | in-container environment checks on the decoded config | free* |
| `save.py` | resolved-config bundle | free |
| `build.py` | decode into `Trainer.Config`, build `ModelSpec` (materializes the model Config) | free to import; model construction needs torch |
| `cli.py` | `load_document`: torch-free loading; `load_trainer_config`: runtime decode and preflight | free to import; runtime loading needs torch |

`*` — imports nothing heavy itself; escalates when the decoded classes do.
The package `__init__` imports the torch-free CLI document loader, but never
imports `build`; a unit test
(`test_package_import_is_torch_free`) enforces it, because pre-submit must
run in a minimal environment.

## Adding things

* **An experiment**: a file under `configs/experiments/` with
  `config_version: 1`, a header comment (what it is, the `torchrun` line),
  imports of the presets it builds on and the sections that differ. Run
  `scripts/config_check.py` on it, then pin it (above). Tests load experiments
  through `tests/unit_tests/config_helpers.py::load_experiment_config`.
* **A config class**: declare it with `@config` in the schema module that
  mirrors the owner's (`torchtitan/x/y.py` -> `torchtitan/config/schema/x/y.py`),
  named `<Owner>Config` with `_owner_spec = "torchtitan.x.y:Owner"`; give the
  owner `Config: TypeAlias = <Owner>Config`. Local invariants go in
  `__post_init__`, cross-section ones in `validate(ctx)`; values are specs, not
  torch objects. The torch-free tests fail on a config declared anywhere else.
* **A component type**: `@config(type="<name>")` on the class; its family is
  the nearest `@config(family=...)` base. Add the name to the pinned set in
  `tests/unit_tests/test_config_torch_free.py`.
* **A model size preset**: add a YAML under `configs/presets/models/<family>/`
  (the `model:` section only, `label` set) and pin it
  (`TORCHTITAN_UPDATE_GOLDEN=1 pytest tests/unit_tests/test_model_arch_presets.py`).
* **A trainer type**: add an entry to `registry.TRAINER_TYPES` (Config import
  spec, plus `model_trainer_type` if the trainer requires a distinct model
  runtime contract). Root sections are discovered from the schema's init
  fields. See [extension.md](extension.md#extending-trainerconfig).
* **A model family**: write its Config in
  `torchtitan/config/schema/models/<family>/model.py`: a `DecoderConfig`
  (or `TiedDecoderConfig`) subclass declared `@config(type="<name>")` with
  `_owner_spec`, `family`, `_runtime_spec` (an `ArchRuntime` constant in
  `torchtitan/models/<family>/versions.py`), the fields its runtime class
  adds and restrictions of its own runtime. The structure goes in
  the presets' `blocks` / `layers`. The runtime owner implements
  `prepare_config` and `get_nparams_and_flops`. Add the type to the pinned
  set in `tests/unit_tests/test_config_torch_free.py`, add presets and pin
  them. See [extension.md](extension.md#modelspec) and
  [torchtitan/models/README.md](../torchtitan/models/README.md).
* **A field on a module** (a block, an attention, a norm, ...): add it to that
  module's schema Config class, validate it and consume it in the runtime
  owner. It becomes a YAML key at the module's path in `blocks`; no extra
  model-family forwarding field is needed. Update affected golden snapshots
  and test the behavior, including the omitted-field default. See the
  [callback example](../torchtitan/callbacks/README.md#adding-a-callback) for
  a complete schema, runtime, YAML and CLI example of a new field.

### Create an experiment

Save this as `configs/experiments/llama3_custom.yaml`. Importing a complete
experiment retains its model, dataloader and loss; this file changes only the
training settings:

```yaml
# Custom Llama 3 debug run.
# torchrun --nproc_per_node=1 -m torchtitan.train configs/experiments/llama3_custom.yaml
config_version: 1
imports:
  - llama3_debug.yaml

training:
  steps: 20
optimizer:
  lr: 1e-4
```

```bash
python scripts/config_check.py configs/experiments/llama3_custom.yaml --world-size 1
NGPU=1 ./run_train.sh configs/experiments/llama3_custom.yaml
```

Paths in `imports` are relative to this new file. When combining a model
preset with training mixins instead, supply every required section yourself.
Before committing an experiment, add its [golden snapshot](#golden-snapshots).

### Inspecting merged and resolved configs

`config-check experiment.yaml --print-merged` prints the imported YAML and CLI
layers before defaults are applied (`--print` remains an alias).
`--print-resolved` prints the typed config with defaults, using the same
representation as `resolved_config.yaml`; it does not build runtime objects.

### Inspecting the placed tree

`config-check experiment.yaml --print-layers` lists the component types on
every placed layer. The materialized tree is internal and is not saved as
model YAML. Without an installed console script, use
`python scripts/config_check.py experiment.yaml` with the same options.

## Programmatic loading

`load_config` returns the decoded trainer config, resolved source and structured
validation report. It raises `ConfigError`; logging and `SystemExit` belong to
the CLI wrapper. This is a runtime API: it prepares the model spec and checks
the environment like the training entrypoint, so it requires torch and the
selected runtime dependencies. For torch-free inspection, use `run_pre_submit`
or `scripts/config_check.py` instead.

```python
# Nested dictionaries are YAML layers; CLI token blocks may appear between them.
from pathlib import Path
from torchtitan.config import load_config

loaded = load_config(
    Path("configs/experiments/llama3_debug.yaml"),
    overrides=["--optimizer.lr", "1e-4", {"training": {"steps": 10}}],
)
# Caller-owned mappings are preserved, and overrides apply in their given order.
config = loaded.trainer_config
warnings = loaded.preflight_report.warnings
```

## Recipe migration coverage

The [recipe migration inventory](config_recipe_migration.md) lists the historical
Python recipes that have a YAML experiment, those with only a model preset,
and the remaining gaps. File coverage does not establish numerical parity.

## Norms, packing and MTP heads in GIFT presets

Norm settings are explicit at each site: every block states its norms (a
file-local YAML anchor shares them), and the GIFT 4 presets layer plain
norms, a zero-initialized post-attention norm and a softcapped head over the
GIFT 3.5 presets they import.

The HF export contract requires one norm class, gating weight and eps across
the RMS/gated norms; the fused linear-output norm has a separate HF eps field.

When checkpointing is enabled and `checkpoint.save_in_hf` or
`checkpoint.last_save_in_hf` is requested,
pre-submit also checks the GIFT model against the same pure HF configuration
contract used by the exporter: supported norm classes and gate ranks, uniform
norm kind/gating weight and sandwich layout, and representable MLA/GDN fields.
These export-only restrictions do not apply to training without HF saves.
Both GIFT runtimes share the module-based sharding dispatcher and HF adapter,
including plain, gated and sandwich norms and GDN layers. Plain GIFT exports
retain their existing tensor keys and values.
MTP heads share the trunk's HF dimensions, router fields, SwiGLU limit and norm
settings. Dense heads must match `intermediate_size`; sparse heads must match
the trunk MoE contract. Heads use MLA with the same attention fields as the
trunk MLA layers; HF has no per-head attention-type selector. A training-only
head can differ, but requesting HF export rejects an unrepresentable head.
The `deepseek_v3` format only supports sparse MTP heads; dense heads need the
`gigachat3_5` format with its `nextn_is_sparse` selector.

Document packing follows the softmax attention's kernel: selecting a VARLEN
kernel in every full-attention block also selects block-causal masking for the
linear attention layers, which take the stack's shared mask type. All layers
must agree; a mix is rejected.

MTP heads are placed from named blocks like the stack. The GIFT 3.5 / 4
presets name the architecture's head block `mtp`; an experiment adds heads
with `mtp: {layers: [{block: mtp, repeat: 2}]}`, or names any other block, or
writes a head block in place.
