import dataclasses
from typing import ClassVar

from torchtitan.config.configurable import config, OwnerFilled
from torchtitan.config.schema.components.lora import FrozenConfig, unwrap_frozen
from torchtitan.config.schema.models.common.attention.base import BaseAttentionConfig
from torchtitan.config.schema.models.common.ffn import FeedForwardConfig
from torchtitan.config.schema.models.common.moe.moe import MoEConfig
from torchtitan.config.schema.models.common.norms.types import NormConfig
from torchtitan.config.schema.models.common.residual_scale import ResidualScaleConfig
from torchtitan.config.schema.models.common.validation import resolve_child
from torchtitan.config.schema.protocols.blocks import BlockConfig
from torchtitan.config.schema.protocols.module import ModuleConfig


@config(family="transformer_block", default="residual")
class TransformerBlockConfig(BlockConfig):
    supports_context_parallel: ClassVar[bool] = True
    """Whether the runtime's Context Parallel supports this block's masks."""
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.decoder_blocks.transformer_block:TransformerBlock"
    )

    attention: BaseAttentionConfig
    feed_forward: FeedForwardConfig | None = None
    moe: MoEConfig | None = None
    attention_norm: NormConfig
    ffn_norm: NormConfig
    # Set by the model builder. With ``dim`` set the block completes partial
    # sublayers; unset, they must already be complete.
    dim: OwnerFilled[int] = None
    layer_id: OwnerFilled[int] = None

    def __post_init__(self) -> None:
        self._complete_sublayers()

    def _complete_sublayers(self) -> None:
        if self.dim is None:
            return
        owner = type(self).__qualname__
        for name in ("attention", "feed_forward", "moe"):
            original = getattr(self, name)
            if original is None:
                continue
            child = unwrap_frozen(original)
            if not isinstance(child, ModuleConfig):
                raise ValueError(f"{owner}.{name} must be a module config")
            if isinstance(child, MoEConfig):
                experts = resolve_child(
                    child.experts,
                    type(child.experts),
                    name=f"{owner}.moe.experts",
                    dims=dict(dim=self.dim),
                )
                if experts is not child.experts:
                    child = child._copy(experts=experts)
            else:
                child = resolve_child(
                    child, type(child), name=f"{owner}.{name}", dims=dict(dim=self.dim)
                )
            if isinstance(original, FrozenConfig):
                if child is not unwrap_frozen(original):
                    original = original._copy(inner=child)
                child = original
            setattr(self, name, child)
        for name in self._norm_fields():
            norm = getattr(self, name)
            if norm is None:
                continue
            setattr(
                self,
                name,
                resolve_child(
                    norm,
                    type(norm),
                    name=f"{owner}.{name}",
                    dims=dict(normalized_shape=self.dim),
                ),
            )

    def _norm_fields(self) -> tuple[str, ...]:
        return ("attention_norm", "ffn_norm")

    def fwd_flops_per_token(self, seq_len: int) -> int:
        """Sum over every ModuleConfig field, so a new sublayer is counted."""
        total = 0
        for f in dataclasses.fields(self):
            sublayer = unwrap_frozen(getattr(self, f.name))
            if isinstance(sublayer, ModuleConfig):
                total += sublayer.fwd_flops_per_token(seq_len)
        return total


@config(type="residual")
class ResidualTransformerBlockConfig(TransformerBlockConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.decoder_blocks.transformer_block:ResidualTransformerBlock"
    )

    # None: pre-norm; set: sandwich norm. None rather than an identity module
    # keeps the pre-norm module tree unchanged.
    post_attn_norm: NormConfig | None = None
    post_ffn_norm: NormConfig | None = None
    # None -> no module, as for the post-norms.
    attn_residual_scale: ResidualScaleConfig | None = None
    ffn_residual_scale: ResidualScaleConfig | None = None

    def __post_init__(self) -> None:
        if (self.feed_forward is None) == (self.moe is None):
            raise ValueError(
                "a residual block requires exactly one of feed_forward or moe"
            )
        # slots=True breaks super(); call the parent explicitly.
        TransformerBlockConfig._complete_sublayers(self)

    def _norm_fields(self) -> tuple[str, ...]:
        return ("attention_norm", "ffn_norm", "post_attn_norm", "post_ffn_norm")
