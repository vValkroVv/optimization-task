import dataclasses
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import ClassVar

from torchtitan.config.configurable import (
    component,
    component_family,
    config,
    Configurable,
    iter_container_leaves,
    to_plain,
)
from torchtitan.config.schema.models.common.regularization import HiddenZLoss
from torchtitan.config.schema.protocols.sharding import ShardingConfig


@component_family("param_init")
@dataclass(frozen=True, slots=True)
class InitSpec:
    """How one parameter is initialized, picked in YAML by ``type``.

    ``torchtitan.protocols.module.resolve_param_init`` maps a spec to its
    ``torch.nn.init`` call; a model family can register resolvers for its own
    specs. ``@component`` goes above ``@dataclass`` because ``slots=True``
    returns a new class.
    """

    entries_removable: ClassVar[bool] = False
    """A ``null`` may not drop an init table's entry: a parameter without one
    fails to initialize. ``{type: skip}`` leaves it uninitialized."""

    def to_dict(self) -> dict:
        """Serialize with the type name; ``Zeros()`` would dump as ``{}``."""
        return to_plain(self)


@component("trunc_normal")
@dataclass(frozen=True, slots=True)
class TruncNormal(InitSpec):
    std: float
    mean: float = 0.0
    a: float = -2.0
    b: float = 2.0

    def __post_init__(self) -> None:
        _validate_normal(self.mean, self.std)
        if self.std == 0:
            raise ValueError("trunc_normal initializer std must be positive")
        if not math.isfinite(self.a) or not math.isfinite(self.b) or self.a >= self.b:
            raise ValueError("trunc_normal requires finite bounds with a < b")


@component("normal")
@dataclass(frozen=True, slots=True)
class Normal(InitSpec):
    std: float
    mean: float = 0.0

    def __post_init__(self) -> None:
        _validate_normal(self.mean, self.std)


def _validate_normal(mean: float, std: float) -> None:
    if not math.isfinite(mean):
        raise ValueError("normal initializer mean must be finite")
    if not math.isfinite(std) or std < 0:
        raise ValueError("normal initializer std must be finite and nonnegative")


@component("zeros")
@dataclass(frozen=True, slots=True)
class Zeros(InitSpec):
    pass


@component("ones")
@dataclass(frozen=True, slots=True)
class Ones(InitSpec):
    pass


@component("xavier_uniform")
@dataclass(frozen=True, slots=True)
class XavierUniform(InitSpec):
    pass


@component("kaiming_uniform")
@dataclass(frozen=True, slots=True)
class KaimingUniform(InitSpec):
    a: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.a):
            raise ValueError("kaiming_uniform initializer a must be finite")


@component("skip")
@dataclass(frozen=True, slots=True)
class Skip(InitSpec):
    """Leave the parameter as built, e.g. because it is tied to another."""


ParamInit = Mapping[str, InitSpec]
"""Per-parameter initializers by local parameter name.

``Mapping``, not ``dict``: ``dict`` is invariant, and init-table constants are
inferred with the concrete spec type.

Explicit module initializer names are checked against the built module's
declared parameters before initialization."""


@config
class ModuleConfig(Configurable.Config):
    param_init: ParamInit | None = None
    init_role: str | None = None
    """Init policy entry (``linear``, ``norm``, ...) used when ``param_init`` is
    unset. ``None`` defers to the owner, then to ``default_init_role``."""
    default_init_role: ClassVar[str] = ""
    """Empty for a module with no parameters of its own (a block, a MoE)."""
    sharding_config: ShardingConfig | None = None
    hidden_z_loss: HiddenZLoss | None = None

    def fwd_flops_per_token(self, seq_len: int) -> int:
        missed_class = (
            self.__class__._owner.__name__
            if self.__class__._owner is not None
            else self.__class__._owner
        )
        raise NotImplementedError(f"Implement flops counter for class {missed_class}")

    def build(self, **kwargs):
        # slots=True prevents super().build() from working; call explicitly.
        return self._attach_config(Configurable.Config.build(self, **kwargs))

    def _attach_config(self, instance):
        """Copy config-carried state onto a freshly built module.

        Separate from ``build`` for the Config that constructs its instance
        another way -- ``HFTransformerModel.Config``, which cannot use
        ``dataclasses.replace``. Call this rather than copying fields by
        hand, or the next field added here goes missing there. Not in
        ``Module.__init__`` because no common ``__init__`` runs for all
        subclasses.
        """
        if self.param_init is not None:
            from torchtitan.protocols.module import validate_param_init_names

            names = self._explicit_param_init_names
            if names is None:
                names = frozenset(self.param_init)
            validate_param_init_names(instance, names)
            instance._param_init = self.param_init
        if self.sharding_config is not None:
            instance._sharding_config = self.sharding_config
        if self.hidden_z_loss is not None:
            from torchtitan.models.common.regularization import attach_hidden_z_loss

            attach_hidden_z_loss(self.hidden_z_loss, instance)
        return instance


def iter_module_configs(
    config, prefix: str = ""
) -> Iterator[tuple[str, "ModuleConfig"]]:
    """Every nested module config, with the FQN ``build()`` will give its module.

    ``Configurable.Config.traverse`` stops at the first match on each branch; this
    descends all the way down. Use ``traverse`` when the outermost match is what
    you want, this when you need every level.
    """
    from torchtitan.config.schema.components.lora import FrozenConfig

    if isinstance(config, FrozenConfig):
        if isinstance(config.inner, ModuleConfig):
            yield prefix, config.inner
        yield from iter_module_configs(config.inner, prefix)
        return
    tree_fields = getattr(config, "tree_fields", None)
    for field in tree_fields() if tree_fields else dataclasses.fields(config):
        value = getattr(config, field.name)
        fqn = f"{prefix}.{field.name}" if prefix else field.name
        for item_fqn, item, _, _ in iter_container_leaves(
            value, config, field.name, fqn
        ):
            if isinstance(item, ModuleConfig):
                yield item_fqn, item
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                yield from iter_module_configs(item, item_fqn)
