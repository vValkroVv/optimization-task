from collections.abc import Callable, Iterator
from copy import deepcopy
from contextlib import contextmanager
from typing import Any, TypeAlias
from weakref import ReferenceType, ref

import torch
import torch.distributed as dist
from streaming.base import StreamingDataLoader, StreamingDataset, use_streaming_world

from torchtitan.config.schema.datasets.streaming.core.dataloader import (
    BaseStreamingDataLoaderConfig,
)
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.distributed.data_world import DataWorld
from torchtitan.datasets.streaming.core.prefetch import PrefetchIterator
from torchtitan.tools.logging import logger


_CHECKPOINT_STATE_KEY = "dp_rank_0"
_PREFETCH_JOIN_TIMEOUT_SECONDS = 30.0


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

    def begin_prefetch_epoch(
        self,
    ) -> tuple[
        tuple[tuple[dict[str, torch.Tensor], torch.Tensor], dict[str, Any]] | None,
        Iterator[Any],
    ]:
        """Prime one finite epoch on the consumer before handing off its remainder."""
        self.num_samples_yielded = 0
        iterator = torch.utils.data.DataLoader.__iter__(self)
        try:
            batch = next(iterator)
        except StopIteration:
            return None, iterator
        except BaseException:
            self.shutdown_workers(iterator)
            raise

        self.num_samples_yielded += self._get_batch_size(batch)
        state = self.state_dict()
        assert state is not None, "dataset is always a StreamingDataset"
        return (batch, state), iterator

    def prefetch_epoch(
        self, iterator: Iterator[Any]
    ) -> Iterator[
        tuple[tuple[dict[str, torch.Tensor], torch.Tensor], dict[str, Any]]
    ]:
        """Read the rest of a finite epoch and snapshot after each source batch."""
        for batch in iterator:
            self.num_samples_yielded += self._get_batch_size(batch)
            state = self.state_dict()
            assert state is not None, "dataset is always a StreamingDataset"
            yield batch, state

    def shutdown_workers(self, iterator: Iterator[Any] | None = None) -> None:
        """Shut down current workers using Torch's pinned iterator cleanup API."""
        candidates = (iterator, self._iterator)
        seen: set[int] = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            shutdown = getattr(candidate, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            if candidate is self._iterator:
                self._iterator = None


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
        num_prefetch_batches: int,
        infinite: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if dataset.num_canonical_nodes is None and dataset.ncn_multiplier is None:
            raise ValueError(
                "Set dataloader.dataset.num_canonical_nodes (recommended) or ncn_multiplier"
            )
        if num_prefetch_batches < 0:
            raise ValueError(
                "dataloader.num_prefetch_batches must be nonnegative, "
                f"got {num_prefetch_batches}"
            )

        self.dataset = dataset
        self.data_world = data_world
        self.infinite = infinite
        self.num_prefetch_batches = num_prefetch_batches
        self._prefetch_iterator: PrefetchIterator[
            tuple[tuple[dict[str, torch.Tensor], torch.Tensor], dict[str, Any]]
        ] | None = None
        self._raw_iterator: Iterator[Any] | None = None
        self._active_outer_iterator: ReferenceType[
            Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
        ] | None = None
        self._iteration_generation = 0
        self._committed_state: dict[str, Any] | None = None

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
        if self.num_prefetch_batches == 0:
            return self._iter_default()

        self.close_iteration()
        generation = self._iteration_generation
        iterator = self._iter_prefetched(generation)
        self._active_outer_iterator = ref(iterator)
        return iterator

    def _iter_default(
        self,
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
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

    def _iter_prefetched(
        self, generation: int
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        try:
            with use_streaming_data_world(self.data_world):
                empty_epochs = 0
                while generation == self._iteration_generation:
                    first, raw_iterator = self._loader.begin_prefetch_epoch()
                    self._raw_iterator = raw_iterator
                    if first is None:
                        self._raw_iterator = None
                        if not self._loader.persistent_workers:
                            self._loader.shutdown_workers(raw_iterator)
                        if not self.infinite:
                            break
                        empty_epochs += 1
                        if empty_epochs >= 2:
                            raise RuntimeError(
                                "Streaming dataloader produced no batches in two "
                                "consecutive epochs. Check epoch_size, drop_last "
                                "and local_batch_size"
                            )
                        continue

                    loader = self._loader
                    prefetcher = PrefetchIterator(
                        lambda loader=loader, raw_iterator=raw_iterator: loader.prefetch_epoch(
                            raw_iterator
                        ),
                        capacity=self.num_prefetch_batches,
                        ready=[first],
                    )
                    self._prefetch_iterator = prefetcher
                    yielded = False
                    exhausted = False
                    try:
                        while generation == self._iteration_generation:
                            try:
                                (batch, state) = next(prefetcher)
                            except StopIteration:
                                exhausted = True
                                break
                            if generation != self._iteration_generation:
                                break
                            self._committed_state = state
                            yielded = True
                            yield batch
                    finally:
                        self._finish_prefetch(
                            prefetcher,
                            raw_iterator,
                            close=not exhausted,
                        )

                    if generation != self._iteration_generation:
                        return
                    if not self.infinite:
                        break
                    if yielded:
                        empty_epochs = 0
                    else:
                        empty_epochs += 1
                        if empty_epochs >= 2:
                            raise RuntimeError(
                                "Streaming dataloader produced no batches in two "
                                "consecutive epochs. Check epoch_size, drop_last "
                                "and local_batch_size"
                            )
        finally:
            if generation == self._iteration_generation:
                self._active_outer_iterator = None

    def _finish_prefetch(
        self,
        prefetcher: PrefetchIterator[Any],
        raw_iterator: Iterator[Any],
        *,
        close: bool,
    ) -> None:
        if close:
            prefetcher.close()
        prefetcher.join(_PREFETCH_JOIN_TIMEOUT_SECONDS)
        if not self._loader.persistent_workers:
            self._loader.shutdown_workers(raw_iterator)
        if self._prefetch_iterator is prefetcher:
            self._prefetch_iterator = None
        if self._raw_iterator is raw_iterator:
            self._raw_iterator = None

    def close_iteration(self) -> None:
        """Stop the current producer while retaining reusable workers."""
        if self.num_prefetch_batches == 0:
            return

        self._iteration_generation += 1
        prefetcher = self._prefetch_iterator
        if prefetcher is not None:
            prefetcher.close()
            prefetcher.join(_PREFETCH_JOIN_TIMEOUT_SECONDS)
            if self._prefetch_iterator is prefetcher:
                self._prefetch_iterator = None

        outer_iterator = self._active_outer_iterator
        self._active_outer_iterator = None
        iterator = outer_iterator() if outer_iterator is not None else None
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        self._raw_iterator = None

    def close(self) -> None:
        """Stop prefetch and perform final Torch DataLoader worker shutdown."""
        raw_iterator = self._raw_iterator
        self.close_iteration()
        self._loader.shutdown_workers(raw_iterator)

    def state_dict(self) -> dict[str, Any]:
        if self.num_prefetch_batches > 0:
            prefetcher = self._prefetch_iterator
            if prefetcher is not None:
                with prefetcher.paused():
                    state = self._committed_state
            elif self._committed_state is not None:
                state = self._committed_state
            else:
                with use_streaming_data_world(self.data_world):
                    state = self._loader.state_dict()
            assert state is not None, "dataset is always a StreamingDataset"
            return {_CHECKPOINT_STATE_KEY: deepcopy(state)}

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

        if self.num_prefetch_batches > 0:
            self.close_iteration()
            distributed_load = (
                self.data_world.world_size > 1
                and dist.is_available()
                and dist.is_initialized()
            )
            if distributed_load and self.data_world.process_group is None:
                raise ValueError(
                    "Prefetched streaming restore requires the explicit data-world "
                    "process_group when world_size is greater than one"
                )
            if distributed_load:
                dist.barrier(group=self.data_world.process_group)

            with use_streaming_data_world(self.data_world):
                self._loader.load_state_dict(streaming_state)
                saved_epoch = streaming_state.get("epoch")
                if saved_epoch is not None and self.data_world.local_rank == 0:
                    # Mosaic ignores an older resume JSON once next_epoch advances.
                    self.dataset.next_epoch = int(saved_epoch)

            if distributed_load:
                dist.barrier(group=self.data_world.process_group)
            self._committed_state = deepcopy(streaming_state)
            return

        with use_streaming_data_world(self.data_world):
            self._loader.load_state_dict(streaming_state)
