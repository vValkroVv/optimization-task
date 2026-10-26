from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.components.tokenizer import BaseTokenizerConfig


@config(type="flux")
class FluxTokenizerContainerConfig(BaseTokenizerConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.flux.tokenizer:FluxTokenizerContainer"
    )

    t5_tokenizer_path: str = "google/t5-v1_1-small"
    """HuggingFace model name or local path for the T5 tokenizer."""
    clip_tokenizer_path: str = "openai/clip-vit-large-patch14"
    """HuggingFace model name or local path for the CLIP tokenizer."""
    max_t5_encoding_len: int = 256
    test_mode: bool = False
