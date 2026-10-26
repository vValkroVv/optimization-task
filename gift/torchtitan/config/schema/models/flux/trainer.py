from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.flux.configs import FluxEncoderConfig, Inference
from torchtitan.config.schema.models.flux.tokenizer import FluxTokenizerContainerConfig
from torchtitan.config.schema.trainer import TrainerConfig
from torchtitan.config.validation import Issue


@config
class FluxTrainerConfig(TrainerConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.trainer:FluxTrainer"

    tokenizer: FluxTokenizerContainerConfig = (  # pyrefly: ignore [bad-override]
        field(default_factory=FluxTokenizerContainerConfig)
    )
    encoder: FluxEncoderConfig = field(default_factory=FluxEncoderConfig)
    """T5 and CLIP text encoders and the autoencoder."""
    inference: Inference = field(default_factory=Inference)

    def validate(self, ctx):
        # slots=True breaks super().
        issues = list(TrainerConfig.validate(self, ctx))
        # `FluxTrainer.__init__` overwrites `training.seq_len` from the image
        # size and the T5 encoding length.
        if "training.seq_len" in ctx.configured_paths:
            issues.append(
                Issue(
                    path="training.seq_len",
                    message="is ignored under trainer type flux: the trainer "
                    "derives it from dataloader.img_size and "
                    "tokenizer.max_t5_encoding_len",
                    severity="warning",
                )
            )
        return issues
