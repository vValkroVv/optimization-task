from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.components.dataloader import BaseDataLoaderConfig
from torchtitan.config.schema.components.validate import ValidatorConfig
from torchtitan.config.schema.models.flux.configs import SamplingConfig
from torchtitan.config.schema.models.flux.flux_datasets import FluxDataLoaderConfig


@config(type="flux")
class FluxValidatorConfig(ValidatorConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.validate:FluxValidator"

    dataloader: BaseDataLoaderConfig = field(
        default_factory=lambda: FluxDataLoaderConfig(
            dataset="coco-validation",
            generate_timesteps=True,
        )
    )
    """DataLoader configuration for Flux validation"""

    all_timesteps: bool = False
    """Generate all 8 timesteps for each sample instead of round-robin"""

    save_img_count: int = -1
    """Number of images to save during validation (-1 for unlimited)"""

    save_img_folder: str = "validation_images"
    """Folder to save validation images"""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    """Sampling configuration for validation image generation"""
