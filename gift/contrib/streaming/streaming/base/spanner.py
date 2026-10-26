# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Mapping of global sample index to shard and relative sample index."""

import numpy as np
from numpy.typing import NDArray


class Spanner:
    """Given a list of shards, construct a mapping of global index to shard and relative index.

    Args:
        shard_sizes (NDArray[np.int64]): Number of samples in each shard.
        span_size (int): Size of the divisions of the sample space. Defaults to ``1 << 10``.
    """

    def __init__(self, shard_sizes: NDArray[np.int64], span_size: int = 1 << 10) -> None:
        self.shard_sizes = shard_sizes
        self.span_size = span_size
        self.num_samples = sum(shard_sizes)
        self.shard_bounds = np.concatenate([np.zeros(1, np.int64), shard_sizes.cumsum()])

        span_starts = np.arange(0, self.num_samples, span_size)
        span_ends = span_starts + span_size - 1
        span_lowest_shards = np.searchsorted(self.shard_bounds, span_starts, side='right') - 1
        span_highest_shards = np.searchsorted(self.shard_bounds, span_ends, side='right') - 1
        # The padding of the last span belongs to the last shard
        span_highest_shards = np.minimum(span_highest_shards, len(shard_sizes) - 1)

        self.spans = []
        for low, high in zip(span_lowest_shards, span_highest_shards):
            shards = np.arange(low, high + 1)
            self.spans.append(shards)

    def __getitem__(self, index: int) -> tuple[int, int]:
        """Map global sample index to shard and relative sample index.

        Args:
            index (int): Global sample index.

        Returns:
            Tuple[int, int]: Shard and relative sample index.
        """
        if not (0 <= index < self.num_samples):
            raise IndexError(f'Invalid sample index `{index}`: 0 <= {index} < {self.num_samples}')

        span = index // self.span_size
        for shard in self.spans[span]:
            shard_start = self.shard_bounds[shard]
            shard_stop = self.shard_bounds[shard + 1]
            if shard_start <= index < shard_stop:
                return shard, int(index - shard_start.item())  # pyright: ignore

        raise RuntimeError('Internal error: shards were indexed incorrectly')
