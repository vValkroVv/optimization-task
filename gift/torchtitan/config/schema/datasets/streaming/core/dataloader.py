from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.components.dataloader import BaseDataLoaderConfig
from torchtitan.config.schema.datasets.streaming.core.configs import (
    StreamingDatasetConfig,
)


@config
class BaseStreamingDataLoaderConfig(BaseDataLoaderConfig):
    replicate_batch_over: ClassVar[tuple[str, ...]] = ("pp",)
    use_data_world: ClassVar[bool] = True
    supports_reiteration: ClassVar[bool] = True
    drop_last: bool = True
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False
    prefetch_factor: int | None = None
    num_prefetch_batches: int = 0
    """Bounded CPU queue depth; zero keeps streaming batch fetching synchronous."""
    # pyrefly: ignore [bad-override]
    dataset: StreamingDatasetConfig = field(default_factory=StreamingDatasetConfig)
    infinite: bool = True

    def __post_init__(self) -> None:
        if self.num_prefetch_batches < 0:
            raise ValueError(
                "dataloader.num_prefetch_batches must be nonnegative, "
                f"got {self.num_prefetch_batches}"
            )

    def validate_for_validation_loop(self) -> None:
        """Only a full, ordered dataset re-iterates deterministically: without
        shuffling or sub-sampling, sample order is independent of the epoch."""
        if self.dataset.shuffle:
            raise ValueError(
                "streaming validation dataloader requires dataset.shuffle=False"
            )
        if self.dataset.epoch_size is not None:
            raise ValueError(
                "streaming validation dataloader requires dataset.epoch_size=None"
            )
        for idx, stream in enumerate(self.dataset.streams):
            for field_name in ("proportion", "repeat", "choose"):
                if getattr(stream, field_name) is not None:
                    raise ValueError(
                        "streaming validation dataloader requires "
                        f"streams[{idx}].{field_name}=None"
                    )
        self._validate_extra_for_validation_loop()

    def _validate_extra_for_validation_loop(self) -> None:
        """Subclass hook; zero-arg super() is unavailable in slots dataclasses"""
