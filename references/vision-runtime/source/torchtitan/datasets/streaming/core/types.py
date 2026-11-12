"""Static types for shared streaming dataset interfaces."""

from __future__ import annotations

from typing import TypedDict

__all__ = ["StreamingDatasetKwargs", "StreamingOmniDatasetKwargs"]

from collections.abc import Sequence
from typing import Any

from streaming.base import Stream
from torchtitan.datasets.streaming.omni.processors.base import GiftOmniBaseProcessor


class StreamingDatasetKwargs(TypedDict, total=False):
    """Keyword arguments accepted by the bundled StreamingDataset fork."""

    streams: Sequence[Stream] | None
    remote: str | None
    local: str | None
    split: str | None
    download_retry: int
    download_timeout: float
    validate_hash: str | None
    keep_zip: bool
    epoch_size: int | str | None
    predownload: int | None
    cache_limit: int | str | None
    sampling_method: str
    sampling_granularity: int
    partition_algo: str
    num_canonical_nodes: int | None
    ncn_multiplier: int | None
    batch_size: int | None
    shuffle: bool
    shuffle_algo: str
    shuffle_seed: int
    shuffle_block_size: int | None
    batching_method: str
    allow_unsafe_types: bool
    replication: int | None
    stream_name: str
    stream_config: dict[str, Any] | None
    legacy: bool


class StreamingOmniDatasetKwargs(StreamingDatasetKwargs):
    """Keyword arguments required to construct the Omni dataset."""

    processors: list[GiftOmniBaseProcessor]
    files_folder_paths: tuple[str | None, ...]
    max_raw_seq_len: int | None
