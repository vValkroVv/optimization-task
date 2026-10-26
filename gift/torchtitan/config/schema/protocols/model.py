from dataclasses import fields
from typing import TYPE_CHECKING, ClassVar, TypeVar

from torchtitan.config.configurable import Configurable, config
from torchtitan.config.schema.protocols.module import ModuleConfig

if TYPE_CHECKING:
    from torchtitan.config.validation import Issue


_ModelConfigT = TypeVar("_ModelConfigT", bound="BaseModelConfig")


@config(family="converter")
class ModelConfigConverterConfig(Configurable.Config):
    _owner_spec: ClassVar[str] = "torchtitan.protocols.model:ModelConfigConverter"

    def validate_pipeline(
        self,
        model_config: Configurable.Config,
        previous: list["ModelConfigConverterConfig"],
        trainer_config=None,
    ) -> None:
        """Check the unconverted tree, preceding converters and optional run options.

        Model-only builders omit trainer options; no runtime modules are needed.
        """


@config(family="model")
class BaseModelConfig(ModuleConfig):
    """Base config for all models; YAML picks the concrete one by ``type``.

    Subclasses define model-specific hyperparameters.
    """

    supported_sharding_backends: ClassVar[tuple[str, ...]] = ("fsdp2",)
    """Backends implemented by this model's runtime parallelization hook."""

    runtime_manages_sharding: ClassVar[bool] = False

    trainer_type: ClassVar[str] = "default"
    """Trainer contract required by this model runtime."""

    family: ClassVar[str] = ""
    """``ModelSpec.name`` shared by several types (``gift_3_5`` and ``gift_4``
    are both ``gift_moe``)."""

    _runtime_spec: ClassVar[str] = ""
    """Import spec of the family's runtime bundle, resolved by the builder."""

    def validate_parallelism(self, parallelism) -> "list[Issue]":
        """Pure model-specific restrictions, shared by pre-submit and runtime."""
        return []

    def sequence_length_divisor(self, parallelism) -> int:
        return 1

    def validate_hf_export(self, *, step: float | None = None) -> "list[Issue]":
        """Pure representability checks when a run requests HF checkpoints.

        ``step``: the 0-based schedule step of the exported weights, ``None``
        when HF saves can happen at any step.

        Models with a schema-owned export contract override this hook. Returning
        no issues does not imply that a runtime adapter implements HF export.
        """
        return []

    @property
    def materialized(self) -> bool:
        """Whether this is the tree ``materialize()`` returned: complete,
        converted in place by the builder, and no longer YAML."""
        return self._get_metadata("materialized", False)

    def validate_materialized(self) -> None:
        """Check completed-tree invariants without changing the tree.

        Called at the boundaries a tree crosses -- ``materialize()``, after the
        builder's converters, and where a runtime constructor takes it -- not
        on every copy: converters and ``prepare_config`` edit trees in place.
        Model owners override it when completion supplies fields that written,
        partially specified configs may omit.
        """
        from torchtitan.config.configurable import (
            DERIVED,
            fields_with_role,
            OWNER_FILLED,
        )
        from torchtitan.config.schema.models.common.moe.load_balance import (
            validate_batched_load_balancing,
        )
        from torchtitan.config.schema.protocols.module import iter_module_configs

        missing = []
        for path, module in [("", self), *iter_module_configs(self)]:
            cls = type(module)
            required = fields_with_role(cls, OWNER_FILLED) | fields_with_role(
                cls, DERIVED
            )
            for field in fields(module):
                if field.name in required and getattr(module, field.name) is None:
                    where = f"{path}.{field.name}" if path else field.name
                    missing.append(f"{where}: still unset after materialize")
        if missing:
            raise ValueError(
                "; ".join(missing) + ". Complete these child settings explicitly "
                "or through their owning config before building."
            )
        validate_batched_load_balancing(self)

    def materialize(self) -> "BaseModelConfig":
        """The complete module tree to build, independent of this config."""
        from torchtitan.config.configurable import independent_config_positions

        tree = independent_config_positions(self)
        tree._set_metadata("materialized", True)
        tree.validate_materialized()
        return tree

    def layer_summary(self) -> list[dict[str, str | int]] | None:
        """Describe actual layers without constructing modules, when supported."""
        return None
