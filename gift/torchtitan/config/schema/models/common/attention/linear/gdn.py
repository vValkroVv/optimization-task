from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config, Derived
from torchtitan.config.schema.models.common.attention.linear.const import (
    GatedNormActivation,
)
from torchtitan.config.schema.models.common.attention.linear.gated_delta import (
    GatedDeltaAttentionConfig,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.norms.fused_gated_rmsnorm import (
    FusedRMSNormGatedConfig,
)
from torchtitan.config.schema.models.common.validation import resolve_child


@config(type="gdn")
class GatedDeltaNetConfig(GatedDeltaAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.linear.gdn:GatedDeltaNet"
    )

    o_norm: FusedRMSNormGatedConfig = field(
        default_factory=lambda: FusedRMSNormGatedConfig(
            activation=GatedNormActivation.SILU
        )
    )

    # Sized from the dims when left None (see ``GatedDeltaAttentionConfig``).
    # ba_proj packs (beta, decay-input a); z_proj gates the o_norm.
    ba_proj: Derived[LinearConfig] = None
    z_proj: Derived[LinearConfig] = None

    def _gating_projections(self) -> tuple[LinearConfig, ...]:
        assert self.ba_proj is not None
        assert self.z_proj is not None
        return (self.ba_proj, self.z_proj)

    def __post_init__(self) -> None:
        # slots=True breaks super().
        GatedDeltaAttentionConfig.__post_init__(self)
        if self.dim is None:
            return  # partial, see the base
        owner = type(self).__qualname__
        self.ba_proj = resolve_child(
            self.ba_proj,
            LinearConfig,
            name=f"{owner}.ba_proj",
            dims=dict(in_features=self.dim, out_features=2 * self.num_v_heads),
        )
        self.z_proj = resolve_child(
            self.z_proj,
            LinearConfig,
            name=f"{owner}.z_proj",
            dims=dict(in_features=self.dim, out_features=self.value_dim),
        )
