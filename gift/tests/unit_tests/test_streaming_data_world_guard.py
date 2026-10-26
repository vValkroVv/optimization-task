import pytest

pytest.importorskip("streaming")

import torch.distributed as dist
from streaming.base import distributed as streaming_dist

from torchtitan.datasets.streaming.core.dataloader import use_streaming_data_world
from torchtitan.distributed.data_world import DataWorld


@pytest.fixture
def single_rank_gloo(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_wide_world_allowed_when_dist_uninitialized():
    with use_streaming_data_world(DataWorld(rank=1, world_size=4)):
        assert streaming_dist.get_rank() == 1
        assert streaming_dist.get_world_size() == 4


def test_mismatched_world_without_group_raises(single_rank_gloo):
    with pytest.raises(ValueError, match="pass the data-axis process_group"):
        with use_streaming_data_world(DataWorld(rank=0, world_size=2)):
            pass


def test_matching_world_without_group_passes(single_rank_gloo):
    with use_streaming_data_world(DataWorld(rank=0, world_size=1)):
        assert streaming_dist.get_world_size() == 1
