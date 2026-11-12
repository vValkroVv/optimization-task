from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="campaign_rng_continuation")
class CampaignRNGContinuationConfig(CallbackConfig):
    """Opt in to rank-local Python, NumPy, Torch CPU, and current CUDA RNG state."""

    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.cuda_graph_campaign.continuation:CampaignRNGContinuationObserver"
    )
