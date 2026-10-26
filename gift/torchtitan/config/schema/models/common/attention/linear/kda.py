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


@config(type="kda")
class KimiDeltaAttentionConfig(GatedDeltaAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.linear.kda:KimiDeltaAttention"
    )

    o_norm: FusedRMSNormGatedConfig = field(
        default_factory=lambda: FusedRMSNormGatedConfig(
            activation=GatedNormActivation.SIGMOID
        )
    )

    # Sized from the dims when left None (see ``GatedDeltaAttentionConfig``).
    # b_proj -> beta; f_proj -> gate features g; g_proj -> output gate, both
    # low-rank through head_v_dim.
    b_proj: Derived[LinearConfig] = None
    f_proj_in: Derived[LinearConfig] = None
    f_proj_out: Derived[LinearConfig] = None
    g_proj_in: Derived[LinearConfig] = None
    g_proj_out: Derived[LinearConfig] = None

    @property
    def gate_dim(self) -> int:
        return self.num_v_heads * self.head_k_dim

    def _gating_projections(self) -> tuple[LinearConfig, ...]:
        assert self.b_proj is not None
        assert self.f_proj_in is not None
        assert self.f_proj_out is not None
        assert self.g_proj_in is not None
        assert self.g_proj_out is not None
        return (
            self.b_proj,
            self.f_proj_in,
            self.f_proj_out,
            self.g_proj_in,
            self.g_proj_out,
        )

    def __post_init__(self) -> None:
        # slots=True breaks super(); call the parent explicitly.
        GatedDeltaAttentionConfig.__post_init__(self)
        if self.dim is None:
            return  # partial, see the base
        owner = type(self).__qualname__
        self.b_proj = resolve_child(
            self.b_proj,
            LinearConfig,
            name=f"{owner}.b_proj",
            dims=dict(in_features=self.dim, out_features=self.num_v_heads),
        )
        self.f_proj_in = resolve_child(
            self.f_proj_in,
            LinearConfig,
            name=f"{owner}.f_proj_in",
            dims=dict(in_features=self.dim, out_features=self.head_v_dim),
        )
        self.f_proj_out = resolve_child(
            self.f_proj_out,
            LinearConfig,
            name=f"{owner}.f_proj_out",
            dims=dict(in_features=self.head_v_dim, out_features=self.gate_dim),
        )
        self.g_proj_in = resolve_child(
            self.g_proj_in,
            LinearConfig,
            name=f"{owner}.g_proj_in",
            dims=dict(in_features=self.dim, out_features=self.head_v_dim),
        )
        self.g_proj_out = resolve_child(
            self.g_proj_out,
            LinearConfig,
            name=f"{owner}.g_proj_out",
            dims=dict(in_features=self.head_v_dim, out_features=self.value_dim),
        )
