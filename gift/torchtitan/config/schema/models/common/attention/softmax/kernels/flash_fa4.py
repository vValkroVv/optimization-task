from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.softmax.kernels.flash_base import (
    FlashAttentionConfig,
    FlashVarlenAttentionConfig,
)


@config(type="flash_fa4")
class FlashAttention4Config(FlashAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.softmax.kernels.flash_fa4:FlashAttention4"
    )


@config(type="flash_varlen_fa4")
class FlashVarlenAttention4Config(FlashVarlenAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.softmax.kernels.flash_fa4:FlashVarlenAttention4"
    )
