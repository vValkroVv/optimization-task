from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.components.dataloader import ParallelAwareDataloaderConfig


@config(type="flux")
class FluxDataLoaderConfig(ParallelAwareDataloaderConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.flux_datasets:FluxDataLoader"

    dataset: str = "cc12m-test"
    """Dataset to use"""

    infinite: bool = True
    """Whether to loop the dataset infinitely"""

    prompt_dropout_prob: float = 0.0
    """Probability of dropping out (replacing with empty string) each text encoding
    independently during training. This enables classifier-free guidance at inference
    time. If `n` text encoders are used, the unconditional model is trained in
    `p ^ n` of all steps. For example, if `n = 2` and `p = 0.447`, the unconditional
    model is trained in 20% of all steps.

    Should be 0.0 for validation (enforced automatically when generate_timesteps=True)."""

    img_size: int = 256
    """Image width to sample"""

    generate_timesteps: bool = False
    """Generate stratified timesteps in round-robin style (for validation)"""

    def __post_init__(self):
        if self.generate_timesteps and self.prompt_dropout_prob != 0.0:
            raise ValueError(
                f"prompt_dropout_prob must be 0.0 when generate_timesteps=True "
                f"(for validation), but got {self.prompt_dropout_prob}."
            )
