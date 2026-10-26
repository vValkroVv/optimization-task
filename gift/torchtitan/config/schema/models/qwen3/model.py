from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    GQAttentionConfig,
)
from torchtitan.config.schema.models.common.decoder_blocks.decoder import (
    TiedDecoderConfig,
)


@config(type="qwen3")
class Qwen3ModelConfig(TiedDecoderConfig):
    attention_types: ClassVar[tuple[type, ...]] = (GQAttentionConfig,)
    _owner_spec: ClassVar[str] = "torchtitan.models.qwen3.model:Qwen3Model"
    family: ClassVar[str] = "qwen3"
    _runtime_spec: ClassVar[str] = "torchtitan.models.qwen3.versions:BASE"
