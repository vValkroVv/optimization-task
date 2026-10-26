from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig
from torchtitan.config.schema.models.common.decoder_blocks.decoder import DecoderConfig


@config(type="deepseek_v3")
class DeepSeekV3ModelConfig(DecoderConfig):
    attention_types: ClassVar[tuple[type, ...]] = (MLAConfig,)
    _owner_spec: ClassVar[str] = "torchtitan.models.deepseek_v3.model:DeepSeekV3Model"
    family: ClassVar[str] = "deepseek_v3"
    _runtime_spec: ClassVar[str] = "torchtitan.models.deepseek_v3.versions:BASE"
