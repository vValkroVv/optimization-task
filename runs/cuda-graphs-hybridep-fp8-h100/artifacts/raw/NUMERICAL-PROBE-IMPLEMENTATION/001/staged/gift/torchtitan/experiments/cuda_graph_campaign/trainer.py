from collections.abc import Iterable, Iterator
from typing import Any
from typing import TypeAlias

import torch

from torchtitan.config.schema.experiments.cuda_graph_campaign.trainer import (
    CampaignTrainerConfig,
)
from torchtitan.experiments.cuda_graph_campaign.continuation import (
    CampaignRNGContinuationObserver,
)
from torchtitan.experiments.cuda_graph_campaign.numerical_probe import (
    CampaignNumericalProbe,
)
from torchtitan.trainer import Trainer


class CampaignTrainer(Trainer):
    """Native trainer with one post-fetch, one-shot RNG continuation restore."""

    Config: TypeAlias = CampaignTrainerConfig
    config: CampaignTrainerConfig

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._bind_rng_continuation()
        self._bind_numerical_probe()

    def _bind_rng_continuation(self) -> None:
        observer = self.callbacks.find(CampaignRNGContinuationObserver)
        if observer is None:
            raise ValueError(
                "cuda_graph_campaign requires the campaign_rng_continuation callback"
            )
        self.rng_continuation = observer
        observer.bind_restore_owner(self)

    def _bind_numerical_probe(self) -> None:
        observer = self.callbacks.find(CampaignNumericalProbe)
        self.numerical_probe = observer
        if observer is None:
            return
        try:
            observer.bind_trainer(self, self.rng_continuation)
        except Exception:
            observer.close()
            raise

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        prepared = super().post_dataloading_process(input_dict, labels)
        if self.numerical_probe is not None:
            self.numerical_probe.capture_prepared_batch(
                step=self.step,
                microbatch_index=self.numerical_probe.current_microbatch_index,
                prepared=prepared,
            )
        return prepared

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
