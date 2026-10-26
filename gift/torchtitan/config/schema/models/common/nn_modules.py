from typing import ClassVar

from torchtitan.config.configurable import config, OwnerFilled
from torchtitan.config.schema.protocols.module import ModuleConfig


@config
class Conv1dConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.nn_modules:Conv1d"
    default_init_role: ClassVar[str] = "conv"

    # Filled by the owning attention.
    in_channels: OwnerFilled[int] = None
    out_channels: OwnerFilled[int] = None
    kernel_size: OwnerFilled[int] = None
    stride: int = 1
    padding: OwnerFilled[int] = None
    groups: OwnerFilled[int] = None
    bias: bool = True

    def fwd_flops_per_token(self, seq_len: int) -> int:
        """Forward FLOPs per output position."""
        return (
            2 * self.out_channels * (self.in_channels // self.groups) * self.kernel_size
        )


@config
class Conv2dConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.nn_modules:Conv2d"
    default_init_role: ClassVar[str] = "conv"

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    # ``nn.Conv2d``'s default; ``LinearConfig.bias`` defaults to False.
    bias: bool = True

    def fwd_flops_per_token(self, seq_len: int) -> int:
        """Forward FLOPs per output spatial position."""
        return (
            2
            * self.out_channels
            * self.in_channels
            * self.kernel_size
            * self.kernel_size
        )


@config
class GELUConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.nn_modules:GELU"

    approximate: str = "none"


@config
class IdentityConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.nn_modules:Identity"


@config(family="linear", default="linear", type="linear")
class LinearConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.nn_modules:Linear"
    default_init_role: ClassVar[str] = "linear"

    # The parent completes widths of partial projection overrides.
    in_features: OwnerFilled[int] = None
    out_features: OwnerFilled[int] = None
    bias: bool = False

    def __post_init__(self) -> None:
        for name in ("in_features", "out_features"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

    def require_features(self) -> tuple[int, int]:
        """Return completed widths; reject an unresolved standalone projection."""
        if self.in_features is None or self.out_features is None:
            raise ValueError(
                "LinearConfig requires in_features and out_features before use"
            )
        return self.in_features, self.out_features

    def build(self, **kwargs):
        self.require_features()
        return ModuleConfig.build(self, **kwargs)

    def fwd_flops_per_token(self, seq_len: int) -> int:
        in_features, out_features = self.require_features()
        return 2 * in_features * out_features


@config
class SiLUConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.nn_modules:SiLU"
