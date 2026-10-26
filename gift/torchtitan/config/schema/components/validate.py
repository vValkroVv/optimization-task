from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.components.dataloader import BaseDataLoaderConfig
from torchtitan.config.schema.datasets.hf.text_datasets import (
    HuggingFaceTextDataLoaderConfig,
)


@config
class BaseValidatorConfig(Configurable.Config):
    freq: int = 10
    """Frequency of validation"""

    def __post_init__(self) -> None:
        if type(self.freq) is not int or self.freq <= 0:
            raise ConfigFieldError("freq", "must be a positive integer")


@config(family="validator", default="default", type="default")
class ValidatorConfig(BaseValidatorConfig):
    _owner_spec: ClassVar[str] = "torchtitan.components.validate:Validator"

    enable: bool = False
    """Enable validation to default run validation after each training loop"""

    steps: int = -1
    """
    Number of steps to take in the validation set, -1 means consuming
    all the data in the validation dataset.
    WARNING: When setting to -1 there could be hangs due to mismatch among ranks
    """

    dataloader: BaseDataLoaderConfig = field(
        default_factory=lambda: HuggingFaceTextDataLoaderConfig(
            dataset="c4_validation",
            infinite=False,
        )
    )
    """DataLoader configuration for validation"""

    def __post_init__(self):
        BaseValidatorConfig.__post_init__(self)
        if self.steps <= 0 and self.steps != -1:
            raise ValueError(
                f"validation steps must be positive or -1, got {self.steps}"
            )
