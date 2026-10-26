from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.softmax.kernels.flash_base import (
    FlashAttentionConfig,
    FlashVarlenAttentionConfig,
)


@config(type="flash_fa3")
class FlashAttention3Config(FlashAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.softmax.kernels.flash_fa3:FlashAttention3"
    )


@config(type="flash_varlen_fa3")
class FlashVarlenAttention3Config(FlashVarlenAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.softmax.kernels.flash_fa3:FlashVarlenAttention3"
    )
