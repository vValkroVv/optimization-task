from typing import TYPE_CHECKING, ClassVar

from torchtitan.config.configurable import config, Derived
from torchtitan.config.schema.models.common.attention.const import (
    AttentionMaskKind,
    MaskType,
)
from torchtitan.config.schema.protocols.module import ModuleConfig

if TYPE_CHECKING:
    from torchtitan.config.schema.models.common.embeddings.rope.rope import RoPEConfig


@config(family="attention")
class BaseAttentionConfig(ModuleConfig):
    supports_tensor_parallel: ClassVar[bool] = True
    """Whether the runtime shards this attention under Tensor Parallel."""
    supports_context_parallel: ClassVar[bool] = True
    """Whether the runtime's Context Parallel wraps this attention."""

    mask_type: Derived[MaskType] = None

    def fwd_flops_per_token(self, seq_len: int) -> int:
        """Forward matmul FLOPs per token (projections + attention core).

        Raises rather than returning 0 so an unaccounted variant cannot
        silently under-report MFU.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fwd_flops_per_token; "
            "implement it to enable analytical FLOP/MFU accounting."
        )

    def accept_rope(self, rope: "RoPEConfig") -> None:
        """Take what this family needs from the decoder's RoPE.

        Check what ``forward`` depends on, fill only what cannot be read at
        runtime. Non-rotating families override with an empty body.
        """
        raise NotImplementedError(
            f"{type(self).__qualname__} does not declare accept_rope; every "
            "attention family must say what it takes from the decoder's RoPE, "
            "or override with an empty body to declare that it does not rotate."
        )

    @property
    def attention_mask_kind(self) -> AttentionMaskKind:
        """How ``mask_type`` must be represented: a flex ``BlockMask``,
        varlen ``cu_seqlens``, or nothing at all."""
        raise NotImplementedError(
            f"{type(self).__name__} does not declare attention_mask_kind; "
            "every attention family must report one so the decoder knows "
            "which mask object to build."
        )
