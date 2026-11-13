from collections.abc import Iterable, Iterator
from typing import TypeAlias

import torch

from torchtitan.config.schema.experiments.cuda_graph_campaign.trainer import (
    CampaignTrainerConfig,
)
from torchtitan.experiments.cuda_graph_campaign.continuation import (
    CampaignRNGContinuationObserver,
)
from torchtitan.trainer import Trainer


class CampaignTrainer(Trainer):
    """Native trainer with one post-fetch, one-shot RNG continuation restore."""

    Config: TypeAlias = CampaignTrainerConfig
    config: CampaignTrainerConfig

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._bind_rng_continuation()

    def _bind_rng_continuation(self) -> None:
        observer = self.callbacks.find(CampaignRNGContinuationObserver)
        if observer is None:
            raise ValueError(
                "cuda_graph_campaign requires the campaign_rng_continuation callback"
            )
        self.rng_continuation = observer
        observer.bind_restore_owner(self)

    def batch_generator(
        self,
        data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        native_batches = super().batch_generator(data_iterable)
        try:
            first_batch = next(native_batches)
            self.rng_continuation.restore_after_first_batch()
            yield first_batch
            yield from native_batches
        finally:
            close = getattr(native_batches, "close", None)
            if close is not None:
                close()
