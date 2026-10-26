# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from streaming.base.sampling import get_sampling


@pytest.mark.parametrize('use_epoch', [True, False])
def test_choose_per_shard_adds_up(use_epoch: bool):
    for granularity in range(1, 100):
        for _ in range(10):
            samples_per_shard = 100 + np.random.choice(100, 10)
            samples = sum(samples_per_shard)
            choose = np.random.choice(samples)
            seed = np.random.choice(31337)
            epoch = np.random.choice(42)
            choose_per_shard = get_sampling(samples_per_shard, choose, granularity, seed, epoch,
                                            use_epoch)
            assert (0 <= choose_per_shard).all()
            assert (choose_per_shard <= samples_per_shard).all()
            assert sum(choose_per_shard) == choose


@pytest.mark.parametrize('use_epoch', [True, False])
def test_is_deterministic(use_epoch: bool):
    for granularity in range(1, 100):
        for _ in range(3):
            samples_per_shard = 100 + np.random.choice(100, 10)
            samples = sum(samples_per_shard)
            choose = np.random.choice(samples)
            seed = np.random.choice(31337)
            epoch = np.random.choice(42)
            last = None
            for _ in range(2):
                choose_per_shard = get_sampling(samples_per_shard, choose, granularity, seed,
                                                epoch, use_epoch)
                if last is not None:
                    assert (last == choose_per_shard).all()
                last = choose_per_shard


@pytest.mark.parametrize('samples_per_shard,choose,granularity,seed,epoch,use_epoch,expected', [
    ([7, 3, 9, 4], 10, 1, 4096, 0, True, [4, 2, 3, 1]),
    ([7, 3, 9, 4], 31, 2, 4096, 1, True, [9, 5, 13, 4]),
    ([5, 5, 5], 11, 3, 7, 0, False, [3, 3, 5]),
    ([1000, 1], 1500, 4, 4096, 2, True, [1498, 2]),
])
def test_matches_reference_values(samples_per_shard: list[int], choose: int, granularity: int,
                                  seed: int, epoch: int, use_epoch: bool, expected: list[int]):
    """Pins which samples each shard contributes, not just how many in total."""
    choose_per_shard = get_sampling(np.array(samples_per_shard, dtype=np.int64), choose,
                                    granularity, seed, epoch, use_epoch)
    assert choose_per_shard.dtype == np.int64
    assert choose_per_shard.tolist() == expected


@pytest.mark.parametrize('use_epoch', [True, False])
def test_balance(use_epoch: bool):
    samples_per_shard = 1_000 + np.random.choice(1_000, 10)
    samples = sum(samples_per_shard)
    choose = np.random.choice(samples)
    choose_per_shard = np.zeros(len(samples_per_shard))
    for granularity in range(1, 100):
        for _ in range(10):
            seed = np.random.choice(31337)
            epoch = np.random.choice(42)
            choose_per_shard += get_sampling(samples_per_shard, choose, granularity, seed, epoch,
                                             use_epoch)
    choose_per_shard /= 99 * 10
    rates = choose_per_shard / samples_per_shard
    imbalance = rates.std() / rates.mean()
    assert imbalance < 0.05
