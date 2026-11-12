from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="campaign_timing_observer")
class CampaignTimingObserverConfig(CallbackConfig):
    """Opt-in timing and activation record for the CUDA Graph campaign."""

    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.cuda_graph_campaign.observer:CampaignTimingObserver"
    )

    start_step: int
    """First optimizer update included in the window, using trainer step numbers."""

    update_count: int
    """Number of complete optimizer updates in the window."""

    output_dir: str = "cuda_graph_timing"
    """Directory for one JSONL file per rank, relative to ``dump_folder``."""

    def __post_init__(self) -> None:
        for name in ("start_step", "update_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigFieldError(name, "must be a positive integer")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ConfigFieldError("output_dir", "must be a nonempty path")
