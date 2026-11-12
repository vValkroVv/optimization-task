from typing import TypeAlias

import torch

from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.config import Configurable
from torchtitan.config.schema.experiments.cuda_graph_campaign.step_boundary import (
    CampaignCUDAGraphStepBoundaryConfig,
)


class CampaignCUDAGraphStepBoundary(Configurable, Observer):
    """Starts one compiler graph generation at each complete optimizer update.

    This observer changes compiler iteration ownership only. It does not mutate
    training state; the selected candidate still requires target-path numerical
    and capture validation.
    """

    Config: TypeAlias = CampaignCUDAGraphStepBoundaryConfig

    def __init__(self, config: Config) -> None:
        self.config = config

    def fit_start(self, state: ObserverState) -> None:
        if state.parallel_dims.pp_enabled:
            raise ValueError(
                "campaign_cudagraph_step_boundary supports only pipeline "
                "parallel degree 1"
            )

    def batch_start(self, state: ObserverState) -> None:
        torch.compiler.cudagraph_mark_step_begin()
