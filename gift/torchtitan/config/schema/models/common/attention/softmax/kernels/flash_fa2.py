from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.softmax.kernels.flash_base import (
    FlashAttentionConfig,
    FlashVarlenAttentionConfig,
)


@config(type="flash_fa2")
class FlashAttention2Config(FlashAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.softmax.kernels.flash_fa2:FlashAttention2"
    )


@config(type="flash_varlen_fa2")
class FlashVarlenAttention2Config(FlashVarlenAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.softmax.kernels.flash_fa2:FlashVarlenAttention2"
    )
