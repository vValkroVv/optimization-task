"""The shape shared by the gated delta-rule families (GDN, KDA)."""

from typing import TypeAlias

from torchtitan.config.schema.models.common.attention.linear.gated_delta import (
    GatedDeltaAttentionConfig,
)
from torchtitan.models.common.attention.base import BaseAttention
from torchtitan.config.schema.models.common.attention.const import (
    MaskType,
)

__all__ = ["GatedDeltaAttention"]


class GatedDeltaAttention(BaseAttention):
    """Base of the gated delta-rule families: a packed Q/K/V with a short
    convolution, the family's own gating, a chunked scan, then a gated RMSNorm
    and an output projection.

    Each family keeps its own ``forward``: the scan and its gating *are* the family,
    and hiding them behind a hook would put the interesting part one indirection away.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GatedDeltaAttentionConfig

    def __init__(self, config: Config):
        super().__init__()
        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.value_dim = config.value_dim
        self.allow_neg_eigval = config.allow_neg_eigval
        self.varlen = config.mask_type is MaskType.BLOCK_CAUSAL

        assert config.qkv is not None and config.o_proj is not None
        self.qkv = config.qkv.build()
        self.o_proj = config.o_proj.build()

        # The output is scaled by ``o_norm_output_multiplier`` in forward.
        self.o_norm = config.o_norm.build()
        self.o_norm_output_multiplier = config.o_norm_output_multiplier
