from typing import ClassVar

from torchtitan.config.configurable import config, Derived, OwnerFilled
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.models.common.attention.base import BaseAttentionConfig
from torchtitan.config.schema.models.common.attention.const import (
    AttentionMaskKind,
    MaskType,
)
from torchtitan.config.schema.models.common.attention.linear.const import (
    linear_delta_core_flops,
)
from torchtitan.config.schema.models.common.attention.qkv import LinearAttnQKVConfig
from torchtitan.config.schema.models.common.nn_modules import Conv1dConfig, LinearConfig
from torchtitan.config.schema.models.common.norms.fused_gated_rmsnorm import (
    FusedRMSNormGatedConfig,
)
from torchtitan.config.schema.models.common.validation import (
    resolve_child,
    validate_values,
)


@config
class GatedDeltaAttentionConfig(BaseAttentionConfig):
    default_init_role: ClassVar[str] = "gated_delta"
    supports_tensor_parallel: ClassVar[bool] = False
    supports_context_parallel: ClassVar[bool] = False
    # None: the owner fills it, then the projections are sized.
    dim: OwnerFilled[int] = None
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_size: int
    # Width filled from ``head_v_dim``; each family defaults the activation it
    # was published with.
    o_norm: FusedRMSNormGatedConfig
    allow_neg_eigval: bool = False
    # Scales the normed output in ``forward``.
    o_norm_output_multiplier: float = 1.0

    def accept_rope(self, rope) -> None:
        """Nothing: the linear families carry position in their recurrent state."""

    qkv: Derived[LinearAttnQKVConfig] = None
    o_proj: Derived[LinearConfig] = None

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def attention_mask_kind(self) -> AttentionMaskKind:
        if self.mask_type is MaskType.BLOCK_CAUSAL:
            return AttentionMaskKind.VARLEN
        return AttentionMaskKind.NONE

    def _gating_projections(self) -> tuple[LinearConfig, ...]:
        """The family's own decay/gating projections, set by its ``__post_init__``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not declare its gating projections."
        )

    def _projections(self) -> tuple[LinearAttnQKVConfig | LinearConfig, ...]:
        assert self.qkv is not None
        assert self.o_proj is not None
        return (self.qkv, *self._gating_projections(), self.o_proj)

    def fwd_flops_per_token(self, seq_len: int) -> int:
        projections = sum(
            proj.fwd_flops_per_token(seq_len) for proj in self._projections()
        )
        core = linear_delta_core_flops(
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
        )
        return projections + core

    def __post_init__(self) -> None:
        """Fill and check the shared fields; families call this first."""
        for name in (
            "num_k_heads",
            "num_v_heads",
            "head_k_dim",
            "head_v_dim",
            "conv_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigFieldError(name, "must be a positive integer")
        if self.mask_type is not None:
            self.mask_type = MaskType(self.mask_type)
        if self.dim is None:
            return

        owner = type(self).__qualname__
        conv_dim = 2 * self.key_dim + self.value_dim

        if self.qkv is None:
            self.qkv = LinearAttnQKVConfig(
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                qkv_proj=LinearConfig(
                    in_features=self.dim,
                    out_features=conv_dim,
                ),
                conv=Conv1dConfig(
                    in_channels=conv_dim,
                    out_channels=conv_dim,
                    kernel_size=self.conv_size,
                    groups=conv_dim,
                    padding=self.conv_size - 1,
                    bias=False,
                ),
            )
        self.qkv = self.qkv._copy(
            qkv_proj=resolve_child(
                self.qkv.qkv_proj,
                LinearConfig,
                name=f"{owner}.qkv.qkv_proj",
                dims=dict(in_features=self.dim, out_features=conv_dim),
            ),
            conv=resolve_child(
                self.qkv.conv,
                Conv1dConfig,
                name=f"{owner}.qkv.conv",
                dims=dict(
                    in_channels=conv_dim,
                    out_channels=conv_dim,
                    kernel_size=self.conv_size,
                    groups=conv_dim,
                    padding=self.conv_size - 1,
                ),
                require=dict(bias=False, stride=1),
            ),
        )
        self.o_proj = resolve_child(
            self.o_proj,
            LinearConfig,
            name=f"{owner}.o_proj",
            dims=dict(in_features=self.value_dim, out_features=self.dim),
        )
        self.o_norm = resolve_child(
            self.o_norm,
            FusedRMSNormGatedConfig,
            name=f"{owner}.o_norm",
            dims=dict(hidden_size=self.head_v_dim),
        )

        # A supplied qkv must take the same input and split heads the same way;
        # its own level checks the packed width and the conv channels.
        validate_values(
            owner,
            ("qkv.in_features", self.qkv.in_features, self.dim),
            ("qkv.num_k_heads", self.qkv.num_k_heads, self.num_k_heads),
            ("qkv.num_v_heads", self.qkv.num_v_heads, self.num_v_heads),
            ("qkv.head_k_dim", self.qkv.head_k_dim, self.head_k_dim),
            ("qkv.head_v_dim", self.qkv.head_v_dim, self.head_v_dim),
            ("qkv.conv.kernel_size", self.qkv.conv.kernel_size, self.conv_size),
        )
