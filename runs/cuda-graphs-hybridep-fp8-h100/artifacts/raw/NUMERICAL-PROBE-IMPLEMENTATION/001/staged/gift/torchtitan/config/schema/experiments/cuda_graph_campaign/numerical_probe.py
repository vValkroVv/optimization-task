from typing import ClassVar, Literal

from torchtitan.config.configurable import config
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="campaign_numerical_probe")
class CampaignNumericalProbeConfig(CallbackConfig):
    """Opt in to a bounded native direct reference probe."""

    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.cuda_graph_campaign.numerical_probe:CampaignNumericalProbe"
    )
    requires: ClassVar[tuple[str, ...]] = ("rng_continuation",)

    mode: Literal["create_s0", "observe"]
    output_dir: str
    updates: Literal[1, 10] = 1

    def __post_init__(self) -> None:
        if self.mode not in {"create_s0", "observe"}:
            raise ConfigFieldError("mode", "must be create_s0 or observe")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ConfigFieldError("output_dir", "must be a nonempty path")
        if self.updates not in {1, 10}:
            raise ConfigFieldError("updates", "must be 1 or 10")
        if self.mode == "create_s0" and self.updates != 1:
            raise ConfigFieldError("updates", "create_s0 always uses one native update")
