"""CPU protocol parity for the task-local fixed-shape routing experiment."""

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from torch import distributed as dist, multiprocessing as mp

from row_owner_route_experiment import fixed_shape_route


WORLD, ROWS = 4, 16
VOCAB = WORLD * ROWS


def _row_values(ids):
    return torch.stack((ids, -ids - 1), dim=1).to(torch.bfloat16)


def _donor_gradient(ids, source):
    return (ids.to(torch.float32) * (source + 1) - 0.25).to(torch.bfloat16)


def _check_valid(owner, inputs):
    rank = owner.mesh_info.shard_mesh_rank
    unique_by_rank = [torch.unique(ids, sorted=True) for ids in inputs]
    expected_unique, expected_inverse = torch.unique(
        inputs[rank].flatten(), sorted=True, return_inverse=True
    )
    incoming = [
        unique[unique.div(ROWS, rounding_mode="floor") == rank]
        for unique in unique_by_rank
    ]
    expected_send = expected_unique.div(ROWS, rounding_mode="floor").bincount(
        minlength=WORLD
    )
    unique, inverse, requested, send, receive = fixed_shape_route(owner, inputs[rank])
    assert torch.equal(unique, expected_unique)
    assert torch.equal(inverse, expected_inverse)
    assert torch.equal(requested, torch.cat(incoming))
    assert send == tuple(expected_send.tolist())
    assert receive == tuple(ids.numel() for ids in incoming)

    returned = torch.empty((unique.numel(), 2), dtype=torch.bfloat16)
    dist.all_to_all_single(returned, _row_values(requested), list(send), list(receive))
    assert torch.equal(returned[inverse], _row_values(inputs[rank].flatten()))

    arrived = torch.empty(sum(receive), dtype=torch.bfloat16)
    dist.all_to_all_single(
        arrived, _donor_gradient(unique, rank), list(receive), list(send)
    )
    expected_arrivals = torch.cat(
        [_donor_gradient(ids, source) for source, ids in enumerate(incoming)]
    )
    assert torch.equal(arrived, expected_arrivals)


def _worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=store,
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=30),
    )
    stages = []

    def stage(name):
        stages.append(name)
        return nullcontext()

    owner = SimpleNamespace(
        mesh_info=SimpleNamespace(
            shard_mesh_size=WORLD,
            shard_mesh_rank=rank,
            shard_process_group=dist.group.WORLD,
        ),
        weight=torch.empty(0),
        num_embeddings=VOCAB,
        rows_per_owner=ROWS,
        _stage=stage,
    )
    empty = torch.empty(0, dtype=torch.int64)
    mixed = [
        torch.tensor([[63, 0, 40, 15], [16, 31, 32, 63]]).t(),
        torch.tensor([17, 15, 16, 32, 63, 32]),
        empty,
        torch.tensor([48, 1, 2, 16, 63, 48, 1]),
    ]
    try:
        for inputs in (
            mixed,
            [empty] * WORLD,
            [torch.zeros(8192, dtype=torch.int64)] * WORLD,
            [(torch.arange(8192) + source * 7) % VOCAB for source in range(WORLD)],
        ):
            _check_valid(owner, inputs)

        for invalid in (
            torch.tensor([torch.iinfo(torch.int64).min]),
            torch.tensor([torch.iinfo(torch.int64).max]),
            torch.tensor([-1]),
            torch.tensor([VOCAB]),
            torch.tensor([3], dtype=torch.int32),
            torch.empty(1, dtype=torch.int64, device="meta"),
        ):
            stages.clear()
            with pytest.raises(ValueError, match="on every rank"):
                fixed_shape_route(owner, invalid if rank == 1 else mixed[rank])
            assert stages == ["partition_ids", "counts"]
        _check_valid(owner, mixed)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_cpu_gloo_routing_and_bf16_payload_parity(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "gloo-store").as_uri(),), nprocs=WORLD)
