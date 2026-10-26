from typing import ClassVar

from torchtitan.config.configurable import config, Configurable


@config(family="tokenizer", default="huggingface")
class BaseTokenizerConfig(Configurable.Config):
    pass


@config(type="huggingface")
class HuggingFaceTokenizerConfig(BaseTokenizerConfig):
    _owner_spec: ClassVar[str] = "torchtitan.components.tokenizer:HuggingFaceTokenizer"

    pass  # no fields — tokenizer_path passed at build time


@config(type="multimodal")
class MultiModalTokenizerConfig(HuggingFaceTokenizerConfig):
    _owner_spec: ClassVar[str] = "torchtitan.components.tokenizer:MultiModalTokenizer"

    image_token: str
    """Token string for image placeholders, e.g. ``"<|image_pad|>"``."""

    video_token: str
    """Token string for video placeholders, e.g. ``"<|video_pad|>"``."""

    vision_start_token: str
    vision_end_token: str
    pad_token: str
