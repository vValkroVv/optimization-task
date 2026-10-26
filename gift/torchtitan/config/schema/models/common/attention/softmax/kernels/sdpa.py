from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
    InnerAttentionConfig,
)


@config(type="sdpa")
class ScaledDotProductAttentionConfig(InnerAttentionConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.attention.softmax.kernels.sdpa:ScaledDotProductAttention"
    )
    supports_context_parallel: ClassVar[bool] = True

    # Masks via ``is_causal`` only, so block-causal is not supported.
    pass
