import dataclasses
from typing import ClassVar

from torchtitan.config.configurable import (
    config,
    Derived,
    DERIVED,
    fields_with_role,
    OWNER_FILLED,
    OwnerFilled,
)
from torchtitan.config.schema.models.common.decoder_blocks.layer_group import (
    LayerGroupConfig,
    placed_configs,
)
from torchtitan.config.schema.models.common.decoder_blocks.transformer_block import (
    ResidualTransformerBlockConfig,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.norms.types import NormConfig
from torchtitan.config.schema.models.common.validation import (
    validate_linear_shapes,
    validate_values,
)
from torchtitan.config.schema.protocols.blocks import BlockConfig
from torchtitan.config.schema.protocols.module import ModuleConfig
from torchtitan.config.schema.components.lora import unwrap_frozen


@config
class MTPBlockConfig(BlockConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.decoder_blocks.mtp:MTPBlock"

    block: ResidualTransformerBlockConfig
    # Filled by the decoder when unset: its ``dim``, ``Linear(2 * dim, dim)``
    # and copies of its norm.
    dim: OwnerFilled[int] = None
    # ``in_features`` must be ``2 * dim`` (concat of hidden + embedding).
    input_proj: Derived[LinearConfig] = None
    token_norm: Derived[NormConfig] = None
    hidden_norm: Derived[NormConfig] = None
    final_norm: Derived[NormConfig] = None

    def __post_init__(self) -> None:
        cls = type(self)
        completed = fields_with_role(cls, OWNER_FILLED) | fields_with_role(cls, DERIVED)
        if (
            self.dim is None
            or self.input_proj is None
            or any(getattr(self, n) is None for n in completed)
            or self.input_proj.in_features is None
            or self.input_proj.out_features is None
            or any(
                getattr(self, name).normalized_shape is None
                for name in ("token_norm", "hidden_norm", "final_norm")
            )
        ):
            return  # partial: the decoder completes it
        owner = type(self).__qualname__
        validate_linear_shapes(
            owner, (self.input_proj, "input_proj", 2 * self.dim, self.dim)
        )
        validate_values(
            owner,
            *(
                (f"{n}", getattr(self, n).normalized_shape, self.dim)
                for n in ("token_norm", "hidden_norm", "final_norm")
            ),
        )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        # Sum over every ModuleConfig field, so a new sublayer is counted.
        total = 0
        for f in dataclasses.fields(self):
            sublayer = unwrap_frozen(getattr(self, f.name))
            if isinstance(sublayer, ModuleConfig):
                total += sublayer.fwd_flops_per_token(seq_len)
        return total


@config
class MTPConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.decoder_blocks.mtp:MTP"

    layers: list[LayerGroupConfig | MTPBlockConfig]
    """MTP heads in order: named decoder ``blocks`` or explicit heads."""

    def placed_layers(self) -> tuple[MTPBlockConfig, ...]:
        return placed_configs(self.layers, MTPBlockConfig, path="mtp.layers")

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError(
                "mtp.layers is empty; omit `mtp` (or set it to null) for no MTP heads"
            )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        return sum(m.fwd_flops_per_token(seq_len) for m in self.placed_layers())
