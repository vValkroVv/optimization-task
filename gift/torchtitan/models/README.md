This note outlines the process of adding a new model in the `torchtitan` repo. In most cases, new models should be added first under the `torchtitan/experiments` folder. For criteria of contributions, please see the [Contributing Guidelines](/torchtitan/experiments/README.md) therein. In general, please adhere to the [Guiding Principles](/README.md#overview) of `torchtitan`.

For offline explorations, we recommend the same steps, unless otherwise noted.

## Adding the model

Please refer to the [Llama 3 folder](llama3) as an example.

The folder should be organized as follows
- `model.py`
  - NOTE: Please adhere to the guiding principles and write single-device model code.
  - Reuse shared components from `torchtitan/models/common/`; keep model-specific architecture in its model folder.
  - Define a Model class inheriting from a base model (e.g. `Decoder` from `torchtitan/models/common/decoder_blocks/decoder.py`).
  - Declare the lightweight config class in `torchtitan/config/schema/models/<family>/model.py` and expose it on the runtime owner as `Config: TypeAlias = <Owner>Config`.
    - The Config is the module tree the YAML writes (`blocks` placed by `layers`); `materialize()` places and completes it. There is one Config per runtime model class.
    - The runtime owner’s `get_nparams_and_flops()` is used to understand model size and compute throughput.
    - The runtime owner’s `prepare_config()` prepares the materialized config from training settings (seq_len, sharding and FP8). Schema classes contain data and pure validation.
  - `__init__()` accepts the written or the materialized config and builds modules from the materialized tree. Preparation, parallelization and adapters consume that tree.
  - Parameter initialization is handled by the `param_init` system on each module's `Config`. Set `param_init` (a `Mapping[str, InitSpec]` mapping parameter names to initializer specs) on every sub-config the model Config's `materialize()` derives (the init policy fills what has none). `init_states()` auto-recurses into all submodules, so manual recursive calls are not needed. Override `_init_self_buffers()` for device-aware buffer initialization (e.g., RoPE, MoE).
  - Add additional files to reduce the complexity of `model.py` if it grows too large or complex, e.g. moe.py to host the `MoE`, `Router`, and `GroupedExperts` modules.
- `state_dict_adapter.py`
  - Inherit [`BaseStateDictAdapter`](/torchtitan/protocols/state_dict_adapter.py) to implement state dict mappings between `torchtitan` model definition and other model definitions (e.g. from HuggingFace so that we can save / load model checkpoints in HF formats).
  - There are multiple ways such adapters could be used
    - Checkpoint conversion scripts in `scripts/checkpoint_conversion/` will use them to adapt state dicts containing non-sharded `torch.Tensor` on CPU.
    - During training, [`CheckpointManager`](/torchtitan/components/checkpoint.py) will use them to adapt state dicts containing (potentially sharded) `DTensor` on GPUs to save / load checkpoints in HF format.
    - In post-training, `to_hf()` helps convert a torchtitan model to HF model, which can be used for inference by other frameworks.
  - This is optional for offline exploration.
- `sharding.py`
  - Define `set_<model>_sharding_config(config, *, loss_parallel, enable_sp, ...)` that populates `sharding_config` on each `Module.Config` in the model config (embeddings, norms, attention, feed-forward, output). TP, SP, and inner-attention `LocalMapConfig` placements are expressed declaratively via `ShardingConfig` instead of a runtime `parallelize_module` plan.
  - Call the helper from the runtime owner’s `prepare_config()` so placements depend on the trainer's `parallelism` settings.
  - Reuse shared helpers from `torchtitan/models/common/decoder_blocks/sharding.py` (`set_decoder_sharding_config`, `set_dense_ffn_sharding`, `set_gqa_attention_sharding`, `norm_config`, `dense_param_placement`, `dense_activation_placement`) where possible.
  - Under `--parallelism.full_dtensor`, declare the mesh axes in canonical outer-to-inner SPMD order: `(dp_replicate, dp_shard, cp, tp)` for dense (attention/MLP/norm/embed/lm_head) and `(dp_replicate, efsdp, ep)` for sparse (MoE expert weights). `Module.parallelize` resolves the mesh by the declared order and validates it matches one of the SPMD meshes; declaring axes out of order raises `ValueError`.
- `parallelize.py`
  - Declare `supported_sharding_backends` on the schema model class. The default is `("fsdp2",)`; add a backend only when this runtime hook implements it. Shared pre-submit/runtime validation refuses undeclared backends.
  - apply training techniques in the following order
    - `model.parallelize(parallel_dims)` — auto-recursive declarative sharding driven by `sharding_config` (TP, SP, attention `local_map`). Replaces per-model `parallelize_module` plan dicts.
    - (MoE models) `apply_moe_ep_tp` for expert-parallel + TP on MoE experts (not yet config-based).
    - activation checkpointing
    - `torch.compile`
    - FSDP /  HSDP
    - NOTE: currently CP support for language models is enabled via a context manager in `torchtitan/train.py`. Ideally no extra work is needed to enable CP.
- `pipeline.py` (optional if model size is small)
  - apply PP
- `versions.py`
  - The runtime half of the family's model types: named `ArchRuntime` constants (parallelizing / pipelining functions, state dict adapter) named by the schema Config's `_runtime_spec` import identifier; only the runtime builder imports them.
- `torchtitan/config/schema/models/<family>/model.py` (the torch-free schema mirror of `model.py`)
  - The model Config the YAML `model.config` decodes into, declared `@config(type="<name>")` with its `family`: the fields its runtime class adds, restrictions of its own runtime and `_runtime_spec`; the init policy is an init preset under `configs/presets/init/` the size presets import (see the [model config convention](/docs/config.md#the-model-config-convention)). The layer structure (dense/MoE split, attention schedule) is the presets' `blocks` / `layers`; `DecoderConfig.materialize()` places and completes it the same way for every model. Experiment types register the same way, under names like `graph_trainer/<model>`.
  - `torchtitan.config.build` turns an experiment's `model:` section into the [`ModelSpec`](/torchtitan/protocols/model_spec.py) (family and label, the materialized model config, parallelizing and pipelining functions, state dict adapter) from these.
  - Read [more](/docs/extension.md#modelspec) on `ModelSpec`.
- `__init__.py`
  - Exports the model class and the parallelize function; no config tables.
- `configs/presets/models/<family>/*.yaml` (outside the model folder)
  - One canonical model preset per size (e.g. `debugmodel.yaml`, `8b.yaml`, `70b.yaml`): the `model:` section only. `tests/unit_tests/test_model_arch_presets.py` compares each preset against its readable compact and materialized snapshot under `tests/unit_tests/golden_configs/presets/`.
  - Training runs are experiment YAMLs under `configs/experiments/` that import a preset and add the training sections (see [docs/config.md](/docs/config.md)). They are selected at runtime by path: `./run_train.sh configs/experiments/<experiment>.yaml`.
- `README.md`
  - Include [instructions](/README.md#downloading-a-tokenizer) to download tokenizers / encoders.
  - Include instructions to download model checkpoints for continued pretraining or post training.
  - Update the current status of development, including the supported features and coming features.
  - This is optional for offline exploration.

## Testing and Benchmarking
- Numerics testing
  - One way of doing this E2E is to load the same model checkpoint into the `torchtitan` model and the HF model, and compare the model output given the same input. This assumes
    - HF implementation is correct.
    - The correctness of a `torchtitan` model and the corresponding state dict adapter together indicates the correctness of both.
- Loss converging
  - If there is a verified baseline, compare the loss curves with the baseline.
  - For comparisons within `torchtitan`, see the [guidelines](/docs/converging.md).
- Performance benchmarking
  - Please refer to the [benchmarks](/benchmarks/) folder.
- CI tests
  - Including unit tests and integration tests, see [examples](/tests/).
  - If the model folder is under the experiments folder, put the tests under the model folder. Otherwise, put the tests under the `/tests` folder.
  - Add necessary GitHub [workflows](/.github/workflows/).
