from torchtitan.config.schema.protocols.types import StrEnum


class MaskType(StrEnum):
    """Attention mask types a decoder may request from ``get_attention_masks``."""

    CAUSAL = "causal"
    BLOCK_CAUSAL = "block_causal"


class AttentionMaskKind(StrEnum):
    """How a mask is represented (flex ``BlockMask`` or varlen ``cu_seqlens``).

    Orthogonal to :class:`MaskType`, which says what is masked.
    """

    NONE = "none"
    BLOCK_MASK = "block_mask"
    VARLEN = "varlen"
