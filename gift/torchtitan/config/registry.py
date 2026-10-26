"""Component families and ``trainer.type`` names.

Component ``type`` names register when their ``@config`` schema module is
imported, so lookups import the whole torch-free ``schema`` package first.
Type names are a public API, pinned in
``tests/unit_tests/test_config_torch_free.py``.
"""

import functools
import importlib
import pkgutil
import sys
from dataclasses import dataclass, field, fields
from typing import Any

from .configurable import Family, family_of, import_spec


@functools.cache
def import_schema() -> None:
    """Import every module under ``torchtitan.config.schema`` once."""
    import torchtitan.config.schema as pkg

    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(info.name)


def family_for_base(cls: type) -> Family | None:
    """The component family whose base is *cls*, or None for a plain dataclass."""
    import_schema()
    return family_of(cls)


def families() -> dict[str, Family]:
    """Every component family by name (imports the schema package first)."""
    import_schema()
    found: dict[str, Family] = {}
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("torchtitan.config.schema"):
            continue
        for obj in list(vars(module).values()):
            if not (isinstance(obj, type) and obj.__module__ == module_name):
                continue
            family = family_of(obj)
            if family is None:
                continue
            if found.setdefault(family.name, family) is not family:
                raise TypeError(f"component family {family.name!r} declared twice")
    return found


# ---------------------------------------------------------------------------
# trainer types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TrainerType:
    """One selectable trainer: the import spec of its ``Config`` class."""

    spec: str
    """``"module:Qual.Name"`` of the trainer's ``Config`` class."""

    model_trainer_type: str = "default"
    """Model runtime contract accepted by this trainer."""

    @property
    def sections(self) -> frozenset[str]:
        """Root sections the document may set, excluding runtime metadata."""
        return _sections_of(self.spec)


@functools.lru_cache(maxsize=None)
def _sections_of(spec: str) -> frozenset[str]:
    return frozenset(f.name for f in fields(import_spec(spec)) if f.init) - {
        "model_spec"
    }


# YAML ``trainer.type`` -> trainer, imported lazily.
TRAINER_TYPES: dict[str, TrainerType] = {
    "default": TrainerType(spec="torchtitan.config.schema.trainer:TrainerConfig"),
    "ft": TrainerType(
        spec="torchtitan.config.schema.experiments.ft.trainer:FaultTolerantTrainerConfig",
        model_trainer_type="ft",
    ),
    # Text-to-image rectified flow: T5+CLIP tokenizer container, frozen
    # encoders and an autoencoder (`encoder`), image sampling (`inference`).
    "flux": TrainerType(
        spec="torchtitan.config.schema.models.flux.trainer:FluxTrainerConfig",
        model_trainer_type="flux",
    ),
    # Graph-captured training. No new sections: it retypes `compile` to
    # GraphTrainerCompileConfig. Pairs with the `graph_trainer/<family>` models.
    "graph_trainer": TrainerType(
        spec="torchtitan.config.schema.experiments.graph_trainer.trainer:GraphTrainerConfig",
        model_trainer_type="graph_trainer",
    ),
    # GRPO controller on Monarch (rollouts through vLLM, a PolicyTrainer
    # actor); its own root sections.
    "rl": TrainerType(
        spec="torchtitan.config.schema.experiments.rl.grpo:RLTrainerConfig",
    ),
}


@dataclass(frozen=True, kw_only=True)
class ArchRuntime:
    """Runtime bundle selected by a model schema's ``_runtime_spec``.

    The referenced constant lives in the family's runtime ``versions.py``.
    """

    parallelize_fn: Any
    pipelining_fn: Any = None
    state_dict_adapter: type | None = None
    spec_cls: type | None = None
    """``ModelSpec`` subclass to build; None means ``ModelSpec`` itself."""
    spec_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra ``spec_cls`` fields (e.g. a fault-tolerance ``fragment_fn``)."""
