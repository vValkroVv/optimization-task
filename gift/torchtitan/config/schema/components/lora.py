import math
import re
from dataclasses import fields
from typing import ClassVar, TypeVar

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.schema.protocols.model import ModelConfigConverterConfig

_FrozenConfigT = TypeVar("_FrozenConfigT", bound="FrozenConfig")


@config
class FrozenConfig(Configurable.Config):
    """Config wrapper that freezes all parameters at build time.

    Works with any ``Module.Config`` — used by ``LoRAConverter`` to freeze
    non-target Linear modules.
    """

    _owner_spec: ClassVar[str] = "torchtitan.components.lora:build_frozen"
    _yaml_serialization_error: ClassVar[str | None] = (
        "Frozen.inner has no typed YAML module-reference contract; save the compact "
        "model config with its converters to recreate the runtime tree"
    )

    inner: Configurable.Config

    def _copy(self: _FrozenConfigT, **changes) -> _FrozenConfigT:
        """Copy forwarded fields on the inner config, retaining the freeze policy."""
        wrapper_fields = {field.name for field in fields(self)}
        inner_changes = {
            name: value for name, value in changes.items() if name not in wrapper_fields
        }
        if inner_changes:
            inner = changes.get("inner", self.inner)._copy(**inner_changes)
            changes = {
                name: value
                for name, value in changes.items()
                if name not in inner_changes
            }
            changes["inner"] = inner
        return Configurable.Config._copy(self, **changes)

    def traverse(self, config_cls, *, _prefix: str = ""):
        """The wrapper builds the inner module at the same runtime FQN."""
        if isinstance(self.inner, config_cls):
            yield _prefix, self.inner, self, "inner"
        else:
            yield from self.inner.traverse(config_cls, _prefix=_prefix)

    def __getattr__(self, name: str):
        try:
            inner = object.__getattribute__(self, "inner")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(inner, name)

    def __setattr__(self, name: str, value) -> None:
        # Dataclass initialization also assigns fields inherited from Config.
        # Those belong to this wrapper before its inner config exists.
        # Slots and properties declared by the wrapper (including inherited
        # service metadata) remain its own state; undeclared names delegate.
        descriptor = getattr(type(self), name, None)
        if name in type(self).__dataclass_fields__ or hasattr(descriptor, "__set__"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)


def unwrap_frozen(value: object) -> object:
    """Read a config's concrete schema for dispatch, without changing its owner.

    Keep the original wrapper when storing or building the config: its build
    policy freezes parameters without introducing another runtime module.
    """
    while isinstance(value, FrozenConfig):
        value = value.inner
    return value


@config(type="lora")
class LoRAConverterConfig(ModelConfigConverterConfig):
    _owner_spec: ClassVar[str] = "torchtitan.components.lora:LoRAConverter"

    rank: int = 8

    alpha: float = 16.0
    """Scaling factor. Output is scaled by alpha/rank."""

    target_modules: list[str] | None = None
    """Module names to apply LoRA to (matched against the last segment of the FQN).
    None means all Linear layers. An empty list means no layers."""

    def __post_init__(self) -> None:
        from torchtitan.config.errors import ConfigFieldError

        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank <= 0
        ):
            raise ConfigFieldError("rank", "LoRA rank must be a positive integer")
        # Zero and negative scales are defined by alpha/rank; only non-finite
        # values poison otherwise valid adapter outputs.
        if not math.isfinite(self.alpha):
            raise ConfigFieldError("alpha", "LoRA alpha must be finite")

    def validate_pipeline(
        self,
        model_config,
        previous: list[ModelConfigConverterConfig],
        trainer_config=None,
    ) -> None:
        if any(isinstance(converter, LoRAConverterConfig) for converter in previous):
            raise ValueError(
                "repeated LoRA converters are unsupported; configure one adapter per model"
            )

        if self.target_modules == []:
            return
        checkpoint = getattr(trainer_config, "checkpoint", None)
        if checkpoint is not None and checkpoint.hf_save_enabled:
            raise ValueError(
                "HF checkpoint export cannot save unmerged LoRA adapters; "
                "disable checkpoint.save_in_hf and checkpoint.last_save_in_hf"
            )
        fp8 = getattr(trainer_config, "fp8", None)
        pattern = getattr(fp8, "dense_include_linear_layers", "")
        if not pattern:
            return
        from torchtitan.config.schema.models.common.nn_modules import LinearConfig

        dense_pattern = re.compile(pattern)
        for fqn, _, _, _ in model_config.traverse(LinearConfig):
            if (
                self.target_modules is None
                or fqn.rsplit(".", 1)[-1] in self.target_modules
            ) and dense_pattern.search(fqn):
                raise ValueError(
                    f"LoRA and dense FP8 both select {fqn!r}; this composition is unsupported. "
                    "Use disjoint projections or disable dense FP8"
                )
