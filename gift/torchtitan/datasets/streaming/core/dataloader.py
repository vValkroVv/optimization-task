from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeAlias

import torch
import torch.distributed as dist
from streaming.base import StreamingDataLoader, StreamingDataset, use_streaming_world

from torchtitan.config.schema.datasets.streaming.core.dataloader import (
    BaseStreamingDataLoaderConfig,
)
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.distributed.data_world import DataWorld
from torchtitan.tools.logging import logger


_CHECKPOINT_STATE_KEY = "dp_rank_0"


@contextmanager
def use_streaming_data_world(data_world: DataWorld) -> Iterator[None]:
    """
    Make streaming use Titan's data-loading world instead of process
    env vars, which describe the full global mesh (with PP)
    """
    if (
        data_world.process_group is None
        and data_world.world_size > 1
        and dist.is_available()
        and dist.is_initialized()
        and data_world.world_size != dist.get_world_size()
    ):
        # a data world of 1 never enters streaming collectives, so it needs no group
        raise ValueError(
            f"Data world size {data_world.world_size} != global world size "
            f"{dist.get_world_size()}: pass the data-axis process_group"
        )

    with use_streaming_world(
        rank=data_world.rank,
        world_size=data_world.world_size,
        local_rank=data_world.local_rank,
        local_world_size=data_world.local_world_size,
        init_dist=False,
        shared_memory_namespace=data_world.shared_memory_namespace,
        process_group=data_world.process_group,
    ):
        yield


class _TitanStreamingDataLoader(StreamingDataLoader):
    """
    StreamingDataLoader sizing Titan ``(inputs, labels)`` batches for
    resume cursor; the tuple contract is shared by all modalities
    """

    def _get_batch_size(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor]
    ) -> int:
        # labels, not batch[0]: len() of the inputs dict would count its keys
        return len(batch[1])


class BaseStreamingDataLoader(BaseDataLoader):
    """Base Titan adapter around upstream streaming.StreamingDataLoader"""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = BaseStreamingDataLoaderConfig

    dataset: StreamingDataset
    data_world: DataWorld

    def _init_streaming_loader(
        self,
        *,
        dataset: StreamingDataset,
        data_world: DataWorld,
        batch_size: int,
        collate_fn: Callable[..., Any],
        drop_last: bool,
        num_workers: int,
        persistent_workers: bool,
        pin_memory: bool,
        prefetch_factor: int | None,
        infinite: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if dataset.num_canonical_nodes is None and dataset.ncn_multiplier is None:
            raise ValueError(
                "Set dataloader.dataset.num_canonical_nodes (recommended) or ncn_multiplier"
            )

        self.dataset = dataset
        self.data_world = data_world
        self.infinite = infinite

        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "collate_fn": collate_fn,
            "drop_last": drop_last,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor

        self._loader = _TitanStreamingDataLoader(**loader_kwargs)

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        with use_streaming_data_world(self.data_world):
            # flag for resume check at the boundary of epochs + epochs change
            empty_epochs = 0
            while True:
                yielded = False
                for batch in self._loader:
                    yielded = True
                    yield batch

                if not self.infinite:
                    break
                if yielded:
                    empty_epochs = 0
                    continue
                empty_epochs += 1
                if empty_epochs >= 2:
                    raise RuntimeError(
                        "Streaming dataloader produced no batches in two consecutive epochs."
                        "Check epoch_size, drop_last and local_batch_size"
                    )

    def state_dict(self) -> dict[str, Any]:
        # upstream already derives the global cursor from the world and replication
        with use_streaming_data_world(self.data_world):
            streaming_state = self._loader.state_dict()
        assert streaming_state is not None, "dataset is always a StreamingDataset"
        return {_CHECKPOINT_STATE_KEY: streaming_state}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return

        streaming_state = state_dict.get(_CHECKPOINT_STATE_KEY)
        if streaming_state is None:
            logger.warning(
                f"DataLoader state is empty for data world rank {self.data_world.rank}, expected key {_CHECKPOINT_STATE_KEY}"
            )
            return

        with use_streaming_data_world(self.data_world):
            self._loader.load_state_dict(streaming_state)
