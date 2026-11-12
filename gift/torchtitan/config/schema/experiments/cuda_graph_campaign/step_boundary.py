from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig
from torchtitan.config.validation import Issue


@config(type="campaign_cudagraph_step_boundary")
class CampaignCUDAGraphStepBoundaryConfig(CallbackConfig):
    """Begin one compiler graph generation per native optimizer update."""

    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.cuda_graph_campaign.step_boundary:"
        "CampaignCUDAGraphStepBoundary"
    )

    def validate_dependencies(self, previous: list[CallbackConfig]) -> list[Issue]:
        count = sum(
            isinstance(callback, CampaignCUDAGraphStepBoundaryConfig)
            for callback in previous
        )
        if count:
            return [
                Issue(
                    path="",
                    message=(
                        "campaign_cudagraph_step_boundary may be configured "
                        "only once per run."
                    ),
                )
            ]
        return []
