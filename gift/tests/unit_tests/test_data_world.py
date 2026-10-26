import os
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.distributed.data_world import (
    DataWorld,
    get_dataloader_shared_memory_namespace,
)


@pytest.fixture
def single_rank_gloo(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"rank": -1, "world_size": 2}, "rank must be in"),
        ({"rank": 2, "world_size": 2}, "rank must be in"),
        ({"rank": 0, "world_size": 0}, "world_size must be positive"),
        (
            {"rank": 0, "world_size": 1, "local_rank": 1, "local_world_size": 1},
            "local_rank must be in",
        ),
        (
            {"rank": 0, "world_size": 1, "local_rank": 0, "local_world_size": 0},
            "local_world_size must be positive",
        ),
        (
            {"rank": 0, "world_size": 4, "local_rank": 0, "local_world_size": 3},
            "must be divisible by",
        ),
        ({"rank": 0, "world_size": 4, "replication": 0}, "replication"),
        ({"rank": 0, "world_size": 4, "replication": 3}, "replication"),
    ],
)
def test_invalid_coordinates_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DataWorld(**kwargs)


def test_process_group_requires_initialized_dist(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    group = dist.group.WORLD
    dist.destroy_process_group()

    with pytest.raises(ValueError, match="requires initialized torch.distributed"):
        DataWorld(rank=0, world_size=1, process_group=group)


def test_process_group_world_size_mismatch_raises(single_rank_gloo):
    with pytest.raises(ValueError, match="process_group world size"):
        DataWorld(rank=0, world_size=2, process_group=single_rank_gloo)


def test_matching_process_group_accepted(single_rank_gloo):
    world = DataWorld(rank=0, world_size=1, process_group=single_rank_gloo)
    assert world.process_group is single_rank_gloo


def test_from_mesh_none_is_single_rank_world():
    world = DataWorld.from_mesh(None, role="train")

    assert (world.rank, world.world_size) == (0, 1)
    assert world.replication == 1
    assert world.process_group is None
    assert world.shared_memory_namespace is not None
    assert "train" in world.shared_memory_namespace


def test_from_mesh_none_rejects_replication():
    with pytest.raises(ValueError, match="requires a data mesh"):
        DataWorld.from_mesh(None, role="train", replication=2)


def test_from_mesh_rejects_multidimensional_mesh(single_rank_gloo):
    mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))
    with pytest.raises(ValueError, match="one-dimensional"):
        DataWorld.from_mesh(mesh, role="train")


def test_namespace_varies_by_role():
    train = get_dataloader_shared_memory_namespace("train")

    assert train != get_dataloader_shared_memory_namespace("validation")
    assert get_dataloader_shared_memory_namespace("train", "x").endswith("_x")


def _mesh_stub(first, world_size, rank=0):
    mesh = mock.Mock()
    mesh.ndim = 1
    mesh.mesh = torch.arange(first, first + world_size)
    mesh.size.return_value = world_size
    mesh.get_local_rank.return_value = rank
    mesh.get_group.return_value = None
    return mesh


def _node_env(monkeypatch, ranks_per_node):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(ranks_per_node))


def test_node_rule_whole_nodes(monkeypatch):
    _node_env(monkeypatch, 8)
    world = DataWorld.from_mesh(_mesh_stub(0, 16, rank=11), role="train")
    assert (world.local_rank, world.local_world_size) == (3, 8)


def test_node_rule_inside_one_node(monkeypatch):
    _node_env(monkeypatch, 8)
    first_group = DataWorld.from_mesh(_mesh_stub(0, 4, rank=1), role="train")
    second_group = DataWorld.from_mesh(_mesh_stub(4, 4, rank=1), role="train")
    assert (first_group.local_rank, first_group.local_world_size) == (1, 4)
    assert (second_group.local_rank, second_group.local_world_size) == (1, 4)
    assert first_group.shared_memory_namespace != second_group.shared_memory_namespace


@pytest.mark.parametrize(
    ("first", "world_size"),
    [
        (0, 12),
        (6, 6),
        (4, 8),
    ],
)
def test_node_rule_rejects_misaligned_worlds(monkeypatch, first, world_size):
    _node_env(monkeypatch, 8)
    with pytest.raises(ValueError, match="do not align with the node size"):
        DataWorld.from_mesh(_mesh_stub(first, world_size), role="train")


def test_from_mesh_rejects_strided_ranks(monkeypatch):
    _node_env(monkeypatch, 8)
    mesh = _mesh_stub(0, 4)
    mesh.mesh = torch.tensor([0, 2, 4, 6])
    with pytest.raises(ValueError, match="contiguous ascending"):
        DataWorld.from_mesh(mesh, role="train")


def test_from_parallel_dims_wires_data_mesh_and_replication(monkeypatch):
    _node_env(monkeypatch, 8)
    parallel_dims = mock.Mock(cp=2, tp=2)
    parallel_dims.get_optional_mesh.return_value = _mesh_stub(0, 8, rank=3)

    world = DataWorld.from_parallel_dims(parallel_dims, role="train")

    parallel_dims.get_optional_mesh.assert_called_once_with("data")
    assert (world.rank, world.world_size, world.replication) == (3, 8, 4)


def _from_mesh_worker(rank, world_size, store_path):
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))

        world = DataWorld.from_mesh(mesh["tp"], role="train", replication=2)
        assert (world.rank, world.world_size) == (rank % 2, 2)
        assert world.replication == 2
        assert (world.local_rank, world.local_world_size) == (rank % 2, 2)
        assert "train" in world.shared_memory_namespace

        with pytest.raises(ValueError, match="contiguous ascending"):
            DataWorld.from_mesh(mesh["dp"], role="train")

        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_from_mesh_on_2x2_mesh(tmp_path):
    world_size = 4
    mp.spawn(
        _from_mesh_worker,
        args=(world_size, str(tmp_path / "store")),
        nprocs=world_size,
        join=True,
    )
