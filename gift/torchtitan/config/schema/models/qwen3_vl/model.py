"""qwen3_vl: the qwen3 decoder plus a ViT encoder (``vision_encoder``).

The ``hf_multimodal`` dataloader must agree with the encoder on ``patch_size``,
``temporal_patch_size`` and ``spatial_merge_size``
(``cross_section.vision_dataloader_agrees`` checks it).
"""

from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.models.common.init_policy import InitPolicy
from torchtitan.config.schema.models.qwen3.model import Qwen3ModelConfig
from torchtitan.config.schema.models.qwen3_vl.vision_encoder import (
    Qwen3VLVisionEncoderConfig,
)


@config(type="qwen3_vl")
class Qwen3VLModelConfig(Qwen3ModelConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.qwen3_vl.model:Qwen3VLModel"
    family: ClassVar[str] = "qwen3_vl"
    _runtime_spec: ClassVar[str] = "torchtitan.models.qwen3_vl.versions:BASE"

    vision_encoder: Qwen3VLVisionEncoderConfig
    mrope_section: list[int]
    """Rotary dims given to the temporal, height and width axes, in that order."""

    def __post_init__(self) -> None:
        Qwen3ModelConfig.__post_init__(self)
        if self.vision_encoder.out_hidden_size != self.dim:
            raise ConfigFieldError(
                "vision_encoder.out_hidden_size",
                f"must match the text decoder dim ({self.dim}), got {self.vision_encoder.out_hidden_size}",
            )
        if len(self.mrope_section) != 3 or any(
            type(section) is not int or section < 0 for section in self.mrope_section
        ):
            raise ConfigFieldError(
                "mrope_section", "must contain three nonnegative integers"
            )

        rotary_half = self.rope.dim // 2
        for axis in (1, 2):
            section = self.mrope_section[axis]
            if section > 0 and axis + 3 * (section - 1) >= rotary_half:
                raise ConfigFieldError(
                    "mrope_section",
                    f"axis {axis} section {section} exceeds the interleaved rotary width ({rotary_half})",
                )

    def _complete(self, policy: InitPolicy) -> None:
        Qwen3ModelConfig._complete(self, policy)
        self.vision_encoder = self.vision_encoder.placed()
