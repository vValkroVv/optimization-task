from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    GQAttentionConfig,
)
from torchtitan.config.schema.models.common.decoder_blocks.decoder import (
    TiedDecoderConfig,
)


@config(type="llama3")
class Llama3ModelConfig(TiedDecoderConfig):
    attention_types: ClassVar[tuple[type, ...]] = (GQAttentionConfig,)
    _owner_spec: ClassVar[str] = "torchtitan.models.llama3.model:Llama3Model"
    family: ClassVar[str] = "llama3"
    _runtime_spec: ClassVar[str] = "torchtitan.models.llama3.versions:BASE"
