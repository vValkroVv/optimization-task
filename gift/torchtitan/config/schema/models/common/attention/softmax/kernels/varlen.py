from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.const import (
    AttentionMaskKind,
    MaskType,
)
from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
    InnerAttentionConfig,
)


@config(type="varlen")
class VarlenAttentionConfig(InnerAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.softmax.kernels.varlen:VarlenAttention"
    )

    mask_kind: ClassVar[AttentionMaskKind] = AttentionMaskKind.VARLEN
    supported_mask_types: ClassVar[tuple[MaskType, ...]] = (MaskType.BLOCK_CAUSAL,)

    window_size: tuple[int, int] = (-1, 0)
    """ window_size=(left, right) controls the attention window relative to each
        query position. 'left' is how many tokens before the query to attend to,
        and 'right' is how many tokens after. A value of -1 means unlimited.

          - (-1, 0): Causal attention - each token attends to all previous tokens
                     and itself, but no future tokens. Equivalent to is_causal=True.
          - (-1, -1): Full bidirectional attention (no masking). Equivalent to
                      is_causal=False.
          - (W, 0): Sliding window causal - attend to at most W previous tokens.
    """
