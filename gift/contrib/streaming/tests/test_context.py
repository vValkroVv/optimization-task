"""Tests for the use_streaming_world context: validation and cleanup on error."""

import pytest

from streaming.base.context import get_streaming_world_context, use_streaming_world
from streaming.base.world import World


@pytest.mark.parametrize(('world_size', 'local_world_size'), [(2, 4), (1, 2), (4, 3), (6, 4)])
def test_invalid_local_world_size_rejected(world_size: int, local_world_size: int):
    with pytest.raises(ValueError, match='divisible'):
        with use_streaming_world(rank=0,
                                 world_size=world_size,
                                 local_rank=0,
                                 local_world_size=local_world_size,
                                 init_dist=False):
            pass
    assert get_streaming_world_context() is None


@pytest.mark.parametrize(('world_size', 'local_world_size'), [(1, 1), (2, 1), (4, 2), (4, 4)])
def test_valid_local_world_size_accepted(world_size: int, local_world_size: int):
    with use_streaming_world(rank=0,
                             world_size=world_size,
                             local_rank=0,
                             local_world_size=local_world_size,
                             init_dist=False):
        world = World.detect()
        assert world.num_nodes == world_size // local_world_size
        assert world.num_ranks == world_size
    assert get_streaming_world_context() is None


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        (dict(rank=0, world_size=0), 'world_size must be positive'),
        (dict(rank=2, world_size=2), 'rank must be in'),
        (dict(rank=0, world_size=2, local_world_size=0), 'local_world_size must be positive'),
        (dict(rank=0, world_size=2, local_rank=1, local_world_size=1), 'local_rank must be in'),
        (dict(rank=0, world_size=4, partition_num_nodes=0), 'partition_num_nodes'),
        (dict(rank=0, world_size=4, partition_num_nodes=3), 'partition_num_nodes'),
    ],
)
def test_coordinate_validation_rejects_bad_worlds(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        with use_streaming_world(init_dist=False, **kwargs):
            pass
    assert get_streaming_world_context() is None


@pytest.mark.parametrize('partition_num_nodes', [None, 1, 2, 4])
def test_partition_num_nodes_reaches_context(partition_num_nodes):
    with use_streaming_world(
        rank=1,
        world_size=4,
        init_dist=False,
        partition_num_nodes=partition_num_nodes
    ):
        context = get_streaming_world_context()
        assert context is not None
        assert context.partition_num_nodes == partition_num_nodes
    assert get_streaming_world_context() is None
