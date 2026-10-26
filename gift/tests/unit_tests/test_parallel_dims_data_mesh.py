"""The flattened ``data`` mesh must be created lazily on first request:
its process group cannot be built under torchcomms, so ``build_mesh``
must not create it for runs that never ask for it"""

from unittest import mock

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.testing._internal.distributed.fake_pg  # noqa: F401  registers "fake"

from torchtitan.distributed import ParallelDims


@pytest.fixture
def single_rank_gloo(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


@mock.patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_build_mesh_does_not_create_data_mesh(single_rank_gloo):
    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
    )
    parallel_dims.build_mesh()

    assert "data" not in parallel_dims._single_axis_meshes


@mock.patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_disabled_data_mesh_is_none_and_never_built(single_rank_gloo):
    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
    )

    assert parallel_dims.get_optional_mesh("data") is None
    assert "data" not in parallel_dims._single_axis_meshes


def _lazy_data_mesh_worker(rank, world_size, store_path, pp):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size
    )
    try:
        with mock.patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
            parallel_dims = ParallelDims(
                dp_replicate=1,
                dp_shard=-1,
                cp=1,
                tp=2,
                pp=pp,
                ep=1,
                world_size=world_size,
            )
            parallel_dims.build_mesh()
            assert "data" not in parallel_dims._single_axis_meshes

            data_mesh = parallel_dims.get_optional_mesh("data")

            assert data_mesh is not None
            assert data_mesh.ndim == 1
            assert data_mesh.size() == world_size // pp
            assert parallel_dims.get_optional_mesh("data") is data_mesh
            assert "data" in parallel_dims.get_all_one_dimensional_meshes()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("pp", [1, 2])
def test_data_mesh_created_on_first_request(tmp_path, pp):
    world_size = 4
    mp.spawn(
        _lazy_data_mesh_worker,
        args=(world_size, str(tmp_path / "store"), pp),
        nprocs=world_size,
        join=True,
    )
