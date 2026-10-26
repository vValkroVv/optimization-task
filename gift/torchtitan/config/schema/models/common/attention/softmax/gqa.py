from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config, Derived, OwnerFilled
from torchtitan.config.schema.components.lora import unwrap_frozen
from torchtitan.config.schema.models.common.attention.base import BaseAttentionConfig
from torchtitan.config.schema.models.common.attention.const import (
    AttentionMaskKind,
    MaskType,
)
from torchtitan.config.schema.models.common.attention.qkv import (
    BaseQKVLinearConfig,
    resolve_qkv,
)
from torchtitan.config.schema.models.common.attention.softmax.gate import (
    SigmoidGateConfig,
)
from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
    InnerAttentionConfig,
)
from torchtitan.config.schema.models.common.attention.softmax.kernels.sdpa import (
    ScaledDotProductAttentionConfig,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.norms.rmsnorm import RMSNormConfig
from torchtitan.config.schema.models.common.validation import (
    resolve_child,
    validate_values,
)


def causal_softmax_core_flops(
    *, n_heads: int, seq_len: int, qk_head_dim: int, v_head_dim: int
) -> int:
    """Forward FLOPs per token of the causal softmax-attention core.

    The two sequence-quadratic matmuls (``Q·Kᵀ`` and ``scores·V``), halved because
    a causal kernel skips the masked triangle. Softmax is elementwise, not counted.
    This is the dense causal upper bound — sliding-window / sparse cores need their
    own helper.
    """
    # (2·n_heads·s·qk + 2·n_heads·s·v) · ½  ==  n_heads·s·(qk + v)
    return n_heads * seq_len * (qk_head_dim + v_head_dim)


@config
class SoftmaxAttentionConfig(BaseAttentionConfig):
    context_parallel_kernels: ClassVar[tuple[type, ...] | None] = None
    """Inner kernels Context Parallel supports for this attention; None: every
    kernel that declares ``supports_context_parallel``."""

    n_heads: int
    inner_attention: InnerAttentionConfig
    gate: SigmoidGateConfig | None = None

    def __post_init__(self):
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {self.n_heads}")
        kernel = self._kernel_config()
        if (
            type(kernel).__dict__.get("_owner_spec") is None
            and type(kernel).__dict__.get("_owner") is None
        ):
            raise ValueError(
                f"inner_attention {type(kernel).__name__} is not "
                "bound to a kernel class and cannot be built; name a concrete "
                "kernel config (e.g. FlexAttentionConfig())."
            )
        supported = type(kernel).supported_mask_types
        if self.mask_type is None:
            if getattr(self, "dim", None) is not None:
                self.mask_type = type(kernel).default_mask_type()
            return
        self.mask_type = MaskType(self.mask_type)
        if self.mask_type not in supported:
            raise ValueError(
                f"mask_type {self.mask_type.value!r} is not supported by "
                f"{type(kernel).__name__}; supported: "
                f"{[m.value for m in supported]}."
            )

    def _kernel_config(self) -> InnerAttentionConfig:
        kernel = unwrap_frozen(self.inner_attention)
        if not isinstance(kernel, InnerAttentionConfig):
            raise ValueError(
                f"inner_attention must be an InnerAttentionConfig, "
                f"got {type(kernel).__name__}"
            )
        return kernel

    @property
    def attention_mask_kind(self) -> AttentionMaskKind:
        """Mask representation the configured inner kernel consumes."""
        return type(self._kernel_config()).mask_kind


@config(type="gqa")
class GQAttentionConfig(SoftmaxAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.softmax.gqa:GQAttention"
    )

    # None: the owner fills it, then the projections are sized.
    dim: OwnerFilled[int] = None
    inner_attention: InnerAttentionConfig = field(
        default_factory=ScaledDotProductAttentionConfig
    )
    # None: sized from the dims, with child-level initializers. A partial
    # ``qkv_linear`` (``{type: fused}``) is completed.
    qkv_linear: Derived[BaseQKVLinearConfig] = None
    wo: Derived[LinearConfig] = None
    qk_norm: RMSNormConfig | None = None
    n_kv_heads: int | None = None
    head_dim: int | None = None
    use_rope: bool = True

    def __post_init__(self) -> None:
        # slots=True breaks super(); call the parent explicitly.
        SoftmaxAttentionConfig.__post_init__(self)
        if self.n_kv_heads is not None and (
            self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads
        ):
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must be positive and divide "
                f"n_heads ({self.n_heads})"
            )
        if self.dim is None:
            return  # partial: the owner fills dim, then this runs again
        owner = type(self).__qualname__
        if self.head_dim is None and self.dim % self.n_heads:
            from torchtitan.config.errors import ConfigFieldError

            raise ConfigFieldError(
                "n_heads",
                f"n_heads ({self.n_heads}) must divide dim ({self.dim}) when head_dim is not specified",
            )
        per_head = self.head_dim or self.dim // self.n_heads
        n_kv = self.n_kv_heads or self.n_heads
        self.qkv_linear = resolve_qkv(
            self.qkv_linear,
            name=f"{owner}.qkv_linear",
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=n_kv,
            head_dim=per_head,
        )
        self.wo = resolve_child(
            self.wo,
            LinearConfig,
            name=f"{owner}.wo",
            dims=dict(in_features=self.n_heads * per_head, out_features=self.dim),
            extra=dict(init_role="linear_out"),
        )
        if self.gate is not None:
            # The gate multiplies the attention output before wo.
            self.gate = self.gate._copy(
                gate_proj=resolve_child(
                    self.gate.gate_proj,
                    LinearConfig,
                    name=f"{owner}.gate.gate_proj",
                    dims=dict(
                        in_features=self.dim, out_features=self.n_heads * per_head
                    ),
                )
            )
        if self.qk_norm is not None:
            self.qk_norm = resolve_child(
                self.qk_norm,
                RMSNormConfig,
                name=f"{owner}.qk_norm",
                dims=dict(normalized_shape=per_head),
            )
        # Each qkv strategy checks its own projections; check only its view.
        checks = [
            ("qkv_linear.in_features", self.qkv_linear.in_features, self.dim),
            ("qkv_linear.head_dim", self.qkv_linear.head_dim, per_head),
            ("qkv_linear.n_heads", self.qkv_linear.n_heads, self.n_heads),
            ("qkv_linear.n_kv_heads", self.qkv_linear.n_kv_heads, n_kv),
        ]
        if self.gate is not None:
            assert self.gate.gate_proj is not None
            checks.append(
                (
                    "gate.gate_proj.out_features",
                    self.gate.gate_proj.out_features,
                    self.wo.in_features,
                )
            )
        validate_values(owner, *checks)

    def _projections(self) -> tuple[BaseQKVLinearConfig, LinearConfig]:
        assert self.qkv_linear is not None and self.wo is not None
        return self.qkv_linear, self.wo

    def accept_rope(self, rope) -> None:
        """Check the rotated width against the decoder's RoPE.

        Nothing to fill: ``forward`` dispatches on the cache type.
        """
        if not self.use_rope:
            return
        assert self.dim is not None, "partial GQAttention.Config"
        per_head = self.head_dim or self.dim // self.n_heads
        if rope.dim != per_head:
            raise ValueError(
                f"{type(self).__qualname__} rotates {per_head} dims per head "
                f"but the decoder's RoPE was built for {rope.dim}."
            )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        assert self.dim is not None, "partial GQAttention.Config"
        head_dim = (
            self.head_dim if self.head_dim is not None else self.dim // self.n_heads
        )
        qkv_linear, wo = self._projections()
        projections = qkv_linear.fwd_flops_per_token(seq_len) + wo.fwd_flops_per_token(
            seq_len
        )
        if self.gate is not None:
            projections += self.gate.fwd_flops_per_token(seq_len)
        # GQA broadcasts each KV head across its query-head group, so the
        # score/value matmuls run over all n_heads query heads at head_dim
        # (key dim == value dim == head_dim). qk_norm is elementwise (0).
        core = causal_softmax_core_flops(
            n_heads=self.n_heads,
            seq_len=seq_len,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
        )
        return projections + core
