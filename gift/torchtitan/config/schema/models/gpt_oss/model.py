import dataclasses
from typing import ClassVar

from torchtitan.config.configurable import config, Derived, OwnerFilled
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.components.lora import unwrap_frozen
from torchtitan.config.schema.models.common.attention.const import MaskType
from torchtitan.config.schema.models.common.attention.qkv import (
    BaseQKVLinearConfig,
    resolve_qkv,
)
from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    SoftmaxAttentionConfig,
    causal_softmax_core_flops,
)
from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
    InnerAttentionConfig,
)
from torchtitan.config.schema.models.common.attention.softmax.kernels.flex import (
    FlexAttentionConfig,
)
from torchtitan.config.schema.models.common.decoder_blocks.decoder import DecoderConfig
from torchtitan.config.schema.models.common.decoder_blocks.transformer_block import (
    ResidualTransformerBlockConfig,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.validation import resolve_child


@config(type="gpt_oss_sink")
class AttentionConfig(SoftmaxAttentionConfig):
    """Sink attention with a per-layer window; the projections carry a bias.

    The shape is required, not defaulted, so a model ``dim`` that disagrees
    with the attention width cannot pass unnoticed."""

    _owner_spec: ClassVar[str] = "torchtitan.models.gpt_oss.model:Attention"
    default_init_role: ClassVar[str] = "sinks"

    n_heads: int
    n_kv_heads: int
    head_dim: int
    sliding_window_size: int
    # None: the owner fills it, then the projections are sized.
    dim: OwnerFilled[int] = None
    qkv_linear: Derived[BaseQKVLinearConfig] = None
    wo: Derived[LinearConfig] = None
    inner_attention: InnerAttentionConfig = dataclasses.field(
        default_factory=FlexAttentionConfig
    )

    def __post_init__(self) -> None:
        if type(self.sliding_window_size) is not int or self.sliding_window_size <= 0:
            raise ConfigFieldError("sliding_window_size", "must be a positive integer")
        if not isinstance(unwrap_frozen(self.inner_attention), FlexAttentionConfig):
            raise ConfigFieldError(
                "inner_attention", "GPT-OSS sink attention requires a flex kernel"
            )
        if self.mask_type is None:
            # Plain causal: ``get_attention_masks`` ands in the sliding window
            # as a separate mask mod.
            self.mask_type = MaskType.CAUSAL
        # slots=True breaks super(); call the parent explicitly.
        SoftmaxAttentionConfig.__post_init__(self)
        if self.dim is None:
            return  # partial: the owner fills dim, then this runs again
        owner = type(self).__qualname__
        self.qkv_linear = resolve_qkv(
            self.qkv_linear,
            name=f"{owner}.qkv_linear",
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            bias=True,
        )
        self.wo = resolve_child(
            self.wo,
            LinearConfig,
            name=f"{owner}.wo",
            dims=dict(in_features=self.n_heads * self.head_dim, out_features=self.dim),
            # gpt-oss checkpoints carry these biases.
            require=dict(bias=True),
            extra=dict(init_role="linear_out"),
        )

    def accept_rope(self, rope) -> None:
        """Require the cos/sin cache ``forward`` hard-codes: unlike GQA it
        does not dispatch, so a complex cache dies on a bare shape assert."""
        if rope.cache_is_complex:
            raise ValueError(
                f"{type(self).__qualname__} applies cos/sin RoPE, but "
                f"{type(rope).__qualname__} builds a complex cache."
            )
        if rope.dim != self.head_dim:
            raise ValueError(
                f"{type(self).__qualname__} rotates {self.head_dim} dims per "
                f"head but the decoder's RoPE was built for {rope.dim}."
            )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        assert self.qkv_linear is not None and self.wo is not None, (
            "partial Attention.Config"
        )
        projections = self.qkv_linear.fwd_flops_per_token(
            seq_len
        ) + self.wo.fwd_flops_per_token(seq_len)
        # Attention sinks add a per-head scalar to the logits (negligible
        # FLOPs). Sliding-window layers are counted as full causal here (an
        # upper bound); see the FLOP-accounting note in models/utils.py.
        core = causal_softmax_core_flops(
            n_heads=self.n_heads,
            seq_len=seq_len,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
        )
        return projections + core


@config(type="gpt_oss_sliding")
class GptOssTransformerBlockConfig(ResidualTransformerBlockConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.gpt_oss.model:GptOssTransformerBlock"
    )

    use_sliding_attention: bool = False


@config(type="gpt_oss")
class GptOssModelConfig(DecoderConfig):
    attention_types: ClassVar[tuple[type, ...]] = (AttentionConfig,)
    _owner_spec: ClassVar[str] = "torchtitan.models.gpt_oss.model:GptOssModel"
    family: ClassVar[str] = "gpt_oss"
    _runtime_spec: ClassVar[str] = "torchtitan.models.gpt_oss.versions:BASE"
    block_types: ClassVar[tuple[type, ...]] = (GptOssTransformerBlockConfig,)
