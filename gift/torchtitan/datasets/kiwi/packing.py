"""K-bounded best-fit packing of variable-length samples into token batches.

Samples arrive one at a time and are placed in the fullest open bank they fit
in (P-BF). When a new bank is needed and ``max_banks`` are already open, the
fullest bank is closed (C-BF). See Csirik & Johnson, Algorithmica 31(2), 2001.
"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any

import torch

from torchtitan.tools.logging import logger

from .batch import ModalityData, ModelBatchData
from .configs import DatasetConfig, PackingConfig
from .constants import (
    EMPTY_QUEUE_WARNING_TIMEOUT,
    ENCODER_ORDER,
    SLOW_GET,
    SLOW_GET_WARN_INTERVAL,
)
from .magi_ranges import MagiRanges, build_magi_ranges
from .preprocessing import DataClass, SampleClass, SampleRef


def iter_with_warnings_on_slow(it: Iterable[Any], *, name: str = "") -> Iterator[Any]:
    """Yield from ``it``; warn if ``next`` is slow after a warm-up period."""
    born = time.monotonic()
    iterator = iter(it)
    label = name or "<unnamed>"
    last_warn = born
    stalls = 0
    while True:
        t0 = time.monotonic()
        try:
            item = next(iterator)
        except StopIteration:
            return
        elapsed = time.monotonic() - t0
        if (
            elapsed > SLOW_GET
            and time.monotonic() - born >= EMPTY_QUEUE_WARNING_TIMEOUT
        ):
            stalls += 1
            now = time.monotonic()
            if now - last_warn >= SLOW_GET_WARN_INTERVAL:
                logger.warning(
                    f"{label} next took {elapsed:.3f}s "
                    f"({stalls} slow next(s) since last warning)"
                )
                last_warn = now
                stalls = 0
        yield item


class BatchClass:
    def __init__(
        self,
    ) -> None:
        self.type_of_content: list[str | None] = []
        self.samples_length: list[int] = []
        self.text: list[DataClass] = []
        self.semantic_visual: list[DataClass] = []
        self.semantic_audio: list[DataClass] = []
        self.pixel_visual: list[DataClass] = []
        self.pixel_audio: list[DataClass] = []
        self.gen_visual: list[DataClass] = []
        self.gen_audio: list[DataClass] = []

        self.length: int = 0

        self.item_length: list[list[int]] = [[] for _ in ENCODER_ORDER]
        self.mask_type: list[list[int]] = [[] for _ in ENCODER_ORDER]
        self.ref_offset = torch.zeros((len(ENCODER_ORDER),))
        self.ref_edges: list[torch.Tensor] = []

        self.attention_adjacency: torch.Tensor | None = None
        self.magi_ranges: MagiRanges | None = None

        self.n_spans = 0
        self.span_properties: list[set[str]] = []

    def _add_ref_edge(self, data: SampleClass) -> None:
        ref_edges = data.get_ref_edges() + self.ref_offset.unsqueeze(0).unsqueeze(-1)
        ref_offset = data.get_ref_offset()

        self.ref_edges.append(ref_edges)

        for idx in range(len(ENCODER_ORDER)):
            self.item_length[idx] += data.item_length[idx]
        for idx in range(len(ENCODER_ORDER)):
            self.mask_type[idx] += data.mask_type[idx]
        self.ref_offset += ref_offset

    def add(self, data: SampleClass) -> None:
        self.samples_length.append(len(data))
        self.type_of_content.append(data.type_of_content)

        for encoder_type in ENCODER_ORDER:
            for item in getattr(data, encoder_type):
                item.offset(self.length)
                item.offset_span_ids(self.n_spans)
                getattr(self, encoder_type).append(item)

        self._add_ref_edge(data)

        self.span_properties.extend(data.span_properties)
        self.n_spans += len(data.span_properties)
        self.length += len(data)

    def get_ref_edges(
        self,
    ) -> torch.Tensor:
        return torch.concat(self.ref_edges, dim=-1)

    def get_ref_offset(
        self,
    ) -> torch.Tensor:
        return self.ref_offset

    def get_ref_mask_types(
        self,
    ) -> list[int]:
        return [i for mask in self.mask_type for i in mask]

    def get_ref_item_length(
        self,
    ) -> list[int]:
        return [i for length in self.item_length for i in length]

    def get_adjacency_matrix(
        self,
    ) -> torch.Tensor:
        ref_edges = self.get_ref_edges()
        ref_edges = ref_edges + torch.concat(
            [torch.tensor([0]), self.ref_offset.cumsum(dim=0)[:-1]]
        ).unsqueeze(0).unsqueeze(-1)

        ref_edges = torch.nan_to_num(ref_edges, nan=0).max(dim=1).values

        mask_type = self.get_ref_mask_types()
        adj = torch.diag(torch.tensor(mask_type))
        adj[
            ref_edges.to(torch.int)[0].ravel(),
            ref_edges.to(torch.int)[1].ravel(),
        ] = 2
        return adj

    def __len__(
        self,
    ) -> int:
        return self.length

    def postprocessing(
        self,
    ) -> None:
        adj = self.get_adjacency_matrix()
        self.attention_adjacency = adj
        self.magi_ranges = build_magi_ranges(self.get_ref_item_length(), adj)

    def to_model_input(self) -> ModelBatchData:
        modalities = {}
        position_ids_global = []
        for modality in ENCODER_ORDER:
            data_list = getattr(self, modality)
            if data_list:
                position_ids_global.append(torch.cat([x.position for x in data_list]))
                position_ids_local = torch.cat(
                    [x.position - x.position.min(dim=0).values for x in data_list]
                )
                seqlens = torch.tensor([0, *(x.index.shape[0] for x in data_list)])

                modalities[modality] = ModalityData(
                    tokens=torch.cat([x.tokens for x in data_list]),
                    position_ids=position_ids_local,
                    cu_seqlens=torch.cumsum(seqlens, dim=0, dtype=torch.int32),
                    target_mask=torch.cat([x.target_mask for x in data_list]),
                    span_ids=torch.cat([x.span_ids for x in data_list]),
                )

        if not position_ids_global:
            raise ValueError("cannot build a model input from an empty batch")
        if self.magi_ranges is None:
            raise ValueError("magi_ranges not initialized")

        return ModelBatchData(
            position_ids=torch.cat(position_ids_global),
            magi_ranges=self.magi_ranges,
            span_properties=self.span_properties,
            **modalities,
        )


def queue_weights(
    datasets: list[DatasetConfig],
) -> dict[tuple[str, str], float]:
    """Joint P(dataset, stream) = P(dataset) * P(stream | dataset)."""
    weights: dict[tuple[str, str], float] = {}
    for dataset in datasets:
        stream_total = sum(stream.proportion for stream in dataset.streams)
        for stream in dataset.streams:
            weights[(dataset.name, stream.name)] = (
                dataset.proportion * stream.proportion / stream_total
            )
    return weights


class StreamMixer:
    """Iterable over stream iterators. Each step draws one by weight and takes next."""

    def __init__(
        self,
        streams: dict[tuple[str, str], Iterator[Any]],
        weights: dict[tuple[str, str], float],
        *,
        rng: random.Random,
    ) -> None:
        if set(streams) != set(weights):
            raise ValueError("streams and weights must use the same keys")
        self.streams = {
            key: iter_with_warnings_on_slow(stream, name=f"{key[0]}/{key[1]}")
            for key, stream in streams.items()
        }
        self.keys = list(self.streams)
        self.weights = [weights[key] for key in self.keys]
        self.rng = rng

    def __iter__(self) -> Iterator[tuple[Any, tuple[str, str]]]:
        while True:
            key = self.rng.choices(self.keys, weights=self.weights)[0]
            try:
                raw = next(self.streams[key])
            except StopIteration:
                return
            yield raw, key


class Bank:
    def __init__(
        self,
    ) -> None:
        self.length = 0
        self.batch = BatchClass()
        self.age = 0
        self.dataset_names: list[str] = []
        self.refs: list[SampleRef] = []

    def __len__(
        self,
    ) -> int:
        return self.length

    def inc_age(self) -> None:
        self.age += 1

    def add(self, sample: SampleClass, dataset_name: str, ref: SampleRef) -> None:
        self.batch.add(sample)
        self.length += sample.length
        self.dataset_names.append(dataset_name)
        self.refs.append(ref)

    def get(
        self,
    ) -> BatchClass:
        return self.batch


class BankRouter:
    """K-Bounded Best Fit (BBF_K) over open banks.

    Two rules, no tunable score:

      P-BF (pack)   place the sample in the *fullest* bank it fits in
      C-BF (close)  when a new bank is needed and max_banks are open,
                    close the *fullest* bank
    """

    def __init__(
        self,
        token_budget: int,
        config: PackingConfig,
    ) -> None:
        self.token_budget = token_budget
        self.cfg = config
        self.banks: list[Bank] = []
        self.out: deque[tuple[list[SampleRef], BatchClass]] = deque()
        self.pending: list[SampleRef] = []
        self.n_closed_full = 0
        self.n_closed_evict = 0
        self.n_skipped = 0
        self.batches_emitted = 0
        self.tokens_emitted = 0

    def remember(self, ref: SampleRef) -> None:
        self.pending.append(ref)

    def _best_fit(self, sample: SampleClass) -> int:
        """P-BF: index of the fullest bank the sample fits in, else -1."""
        best_i = -1
        best_length = -1
        for i, bank in enumerate(self.banks):
            if bank.length + sample.length > self.token_budget:
                continue
            if bank.length > best_length:
                best_i = i
                best_length = bank.length
        return best_i

    def _fullest(self) -> int:
        """C-BF: index of the fullest open bank."""
        best_i = 0
        best_length = -1
        for i, bank in enumerate(self.banks):
            if bank.length > best_length:
                best_i = i
                best_length = bank.length
        return best_i

    def _emit(self, index: int) -> None:
        bank = self.banks.pop(index)
        refs = [*self.pending, *bank.refs]
        self.pending = []
        self.out.append((refs, bank.get()))
        self.batches_emitted += 1
        self.tokens_emitted += bank.length

    def ready(self) -> bool:
        return bool(self.out)

    def add(self, sample: SampleClass, dataset_name: str, ref: SampleRef) -> bool:
        """Place ``sample``. False if dropped (longer than the budget)."""
        if sample.length > self.token_budget:
            logger.warning(
                "Dropping sample of dataset %r (length=%d): exceeds token_budget=%d",
                dataset_name,
                sample.length,
                self.token_budget,
            )
            self.remember(ref)
            self.n_skipped += 1
            return False

        i = self._best_fit(sample)
        if i < 0:
            if len(self.banks) >= self.cfg.max_banks:
                self.n_closed_evict += 1
                self._emit(self._fullest())
            self.banks.append(Bank())
            i = len(self.banks) - 1

        bank = self.banks[i]
        bank.add(sample, dataset_name, ref)
        if bank.length == self.token_budget:
            self.n_closed_full += 1
            self._emit(i)
        return True

    def pop(self) -> tuple[list[SampleRef], BatchClass]:
        if not self.out:
            raise RuntimeError("pop() called with nothing ready; check ready() first")
        return self.out.popleft()

    @property
    def waste(self) -> float:
        """Fraction of emitted batch capacity that is padding."""
        if not self.batches_emitted:
            return 0.0
        return 1.0 - self.tokens_emitted / (self.batches_emitted * self.token_budget)
