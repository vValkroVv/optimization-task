To support rapid experimentation with torchtitan, we provide several extension points. The principle for adding these extension points is to support various use cases with flexible component swapping and reuse, while trying to keep the code clean and minimal.

The extension points and protocols mentioned in this note are subject to change.


### `ModelSpec`

[`ModelSpec`](../torchtitan/protocols/model_spec.py) supports configuring high-level components in model training, including
- definitions of model config and model class
- model parallelization functions
- state-dict adapters

The coarse level abstraction tries to hit a balance between flexible component swapping and a straightforward train script ([train.py](../torchtitan/train.py)).

A `ModelSpec` is never written by hand: `torchtitan.config.build` builds it from the experiment YAML's `model:` section, whose `config` decodes into the model's own Config (picked by `type`) and is materialized once into the module tree. The YAML is that tree written with the modules' Config fields; repeated layers are named under `blocks` and placed by `layers` (see [the model config convention](config.md#the-model-config-convention)).

To register a model family, write in the torch-free `torchtitan/config/schema/models/<family>/model.py` a `DecoderConfig` (or `TiedDecoderConfig`) subclass declared `@config(type="<name>")` with

- `_owner_spec`, the runtime model class, and `family`;
- `_runtime_spec`, an import identifier for an `ArchRuntime` constant in `torchtitan/models/<family>/versions.py`, naming the parallelizing / pipelining functions and state-dict adapter; only the runtime builder imports it;
- any field its runtime class adds, and validation of restrictions specific to its own runtime (module restrictions are capability flags on the module's Config, checked for every model).

The structure -- which blocks, in which order -- and the init policy (an init preset under `configs/presets/init/`) go in the size presets under `configs/presets/models/<family>/` (pinned by `tests/unit_tests/test_model_arch_presets.py`), not in Python. A new module or a new field on one is added to that module's Config class and is usable in every model's YAML at once. The runtime model owner implements `prepare_config` and `get_nparams_and_flops`. Then add the type to the pinned set in `tests/unit_tests/test_config_torch_free.py`. See [torchtitan/models/llama3](../torchtitan/models/llama3/) for an example and [torchtitan/models/README.md](../torchtitan/models/README.md) for the full checklist.


### Train script

To perform various tasks, from adding a new model (possibly with a new modality), to trying out a new training paradigm (e.g. async training), a single train script cannot handle all the cases, unless customization points are inserted everywhere to make it less readable. Instead of always starting and maintaining a standalone train script, we group code in [train.py](../torchtitan/train.py) into functions to allow for reuse.

This is an ongoing effort, and the level of grouping is subject to change.


### Extending `Trainer.Config`

To add a root config section for an experiment, subclass `TrainerConfig`
(`torchtitan/config/schema/trainer.py`) in the schema module mirroring your
trainer's (see [Schema and runtime](config.md#schema-and-runtime)). Add typed
fields there, expose that schema as the runtime trainer's `Config` alias, and
register it as a trainer type selected by `trainer: {type: ...}`.
Subclassing `Trainer` alone does not add YAML fields. If an existing component
owns the setting, extend its schema instead; use a callback for behavior that
fits an existing trainer event (see [Adding a Callback](../torchtitan/callbacks/README.md#adding-a-callback)).

`@config` uses `dataclass(slots=True)`, which replaces the class object.
Inside overridden methods, call the parent explicitly:
`TrainerConfig.validate(self, ctx)` or `ParentConfig.__post_init__(self)`.
Zero-argument `super()` can raise `TypeError` here. Preserve the parent's
checks when adding your own validation.

#### Example

To add a custom config section for an experiment:

```python
# Lightweight schema: no runtime imports or torch.
# torchtitan/config/schema/experiments/your_folder/trainer.py
from dataclasses import dataclass, field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.trainer import TrainerConfig

@dataclass(kw_only=True)
class CustomConfig:
    how_is_your_day: str = "good"

@config
class CustomTrainerConfig(TrainerConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.your_folder.trainer:CustomTrainer"
    )
    custom_config: CustomConfig = field(default_factory=CustomConfig)
```

The runtime owner exposes that schema:

```python
# torchtitan/experiments/your_folder/trainer.py
from typing import TypeAlias

from torchtitan.config.schema.experiments.your_folder.trainer import CustomTrainerConfig
from torchtitan.trainer import Trainer

class CustomTrainer(Trainer):
    Config: TypeAlias = CustomTrainerConfig
```

Register the type in [`torchtitan/config/registry.py`](../torchtitan/config/registry.py)
by adding an entry to `TRAINER_TYPES`. Its accepted sections are discovered from
the schema's init fields; no second section-name table needs updating:

```python
"my_trainer": TrainerType(
    spec="torchtitan.config.schema.experiments.your_folder.trainer:CustomTrainerConfig",
),
```

Then describe the run in an experiment YAML:

```yaml
# Custom trainer on the bundled Llama 3 debug model and corpus.
# torchrun --nproc_per_node=1 -m torchtitan.train configs/experiments/my_experiment_debug.yaml
# configs/experiments/my_experiment_debug.yaml
config_version: 1
imports:
  - ../presets/models/llama3/debugmodel.yaml
  - ../presets/train/debug.yaml

trainer:
  type: my_trainer

loss: {type: cross_entropy}

custom_config:
  how_is_your_day: great

training:
  steps: 100
```

The debug training mixin supplies a dataloader, but the loss must be selected
explicitly. After adding the schema, runtime owner and registry entry, validate
the experiment before launching it:

```bash
python scripts/config_check.py configs/experiments/my_experiment_debug.yaml --world-size 1
NGPU=1 ./run_train.sh configs/experiments/my_experiment_debug.yaml
```

Every file under `configs/experiments/` is decoded by `tests/unit_tests/test_experiment_parity.py`, which requires a readable golden snapshot for it (`TORCHTITAN_UPDATE_GOLDEN=1`, see [config.md](config.md#golden-snapshots)). Section discovery also applies to roots with a different shape, such as the RL controller. `configs/experiments/ft_llama3_debug.yaml` and `flux_debug.yaml` are the in-tree examples of typed trainers, and [config.md](config.md#trainer-type) lists every type.

### Source paths for derived fields

Decoded configs carry internal source metadata through `deepcopy` and
`_copy(**changes)`. Use `_copy` when changing a config: plain
`dataclasses.replace` resets non-init metadata. Source metadata is excluded
from YAML, equality and golden snapshots.

When a builder renames a compact field, use `derive` to state the dependency
at its construction point. Matching values cannot tell whether a generated
embedding dimension came from `dim` or `vocab_size`.

```python
from torchtitan.config.configurable import config, Configurable
from torchtitan.config.schema.models.common.embeddings.token_embeddings import (
    EmbeddingConfig,
)

# Relevant part of a compact Config; model-specific methods are omitted.
@config
class EmbeddingOwnerConfig(Configurable.Config):
    dim: int
    vocab_size: int

    def _embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig.derive(
            self,
            field_sources={
                # Runtime field -> compact field, never a guessed value.
                "num_embeddings": "vocab_size",
                "embedding_dim": "dim",
            },
        )
```

`derive` copies the referenced semantic values and carries their source paths
without retaining the compact config object. Additional constructor values
can be passed normally. A validation `Issue` can then name the original YAML
field and winning file or override, while `runtime_path` records its location
in the built tree. Configs created directly from code keep ordinary runtime
diagnostics when no YAML origin is known.
