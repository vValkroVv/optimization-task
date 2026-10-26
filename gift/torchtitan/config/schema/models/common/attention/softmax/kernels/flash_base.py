from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.const import (
    AttentionMaskKind,
    MaskType,
)
from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
    InnerAttentionConfig,
)


@config
class FlashAttentionConfig(InnerAttentionConfig):
    # (left, right) window; (-1, -1) = full, causal via the is_causal flag.
    # (W, 0) gives sliding-window causal attention.
    window_size: tuple[int, int] = (-1, -1)


@config
class FlashVarlenAttentionConfig(InnerAttentionConfig):
    mask_kind: ClassVar[AttentionMaskKind] = AttentionMaskKind.VARLEN
    supported_mask_types: ClassVar[tuple[MaskType, ...]] = (MaskType.BLOCK_CAUSAL,)

    window_size: tuple[int, int] = (-1, -1)
