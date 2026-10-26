import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.components.dataloader import (
    BaseDataLoader,
    build_replicated_dataloader,
    ReplicatedBatchDataLoader,
)
from torchtitan.distributed import ParallelDims


class _StatefulBatchLoader(BaseDataLoader):
    def __init__(self, batches: list[Any]) -> None:
        self.batches = batches
        self.next_index = 0
        self.loaded_state: dict[str, Any] | None = None

    def __iter__(self) -> "_StatefulBatchLoader":
        self.next_index = 0
        return self

    def __next__(self) -> Any:
        if self.next_index == len(self.batches):
            raise StopIteration
        batch = self.batches[self.next_index]
        self.next_index += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {"next_index": self.next_index}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loaded_state = state_dict
        self.next_index = state_dict["next_index"]


def _batches(dp_rank: int) -> list[Any]:
    offset = 100 * dp_rank
    return [
        (
            {
                "input": torch.arange(6, dtype=torch.int64).reshape(2, 3) + offset,
                "positions": torch.arange(6, dtype=torch.int32).reshape(2, 3),
                "scale": torch.tensor(1.5, dtype=torch.float32),
                "mask": torch.tensor([True, False]),
                "embed": (torch.arange(8, dtype=torch.bfloat16) / 4).reshape(2, 4)
                + offset,
            },
            torch.arange(6, dtype=torch.int64).reshape(2, 3) + offset + 1,
        ),
        (
            {
                "input": torch.tensor([[offset + 9, offset + 10]]),
                "positions": torch.tensor([[0, 1]], dtype=torch.int32),
                "scale": torch.tensor(2.5, dtype=torch.float32),
                "mask": torch.tensor([False]),
                "embed": (
                    (torch.arange(4, dtype=torch.bfloat16) / 2).reshape(2, 2) + offset
                ).T,
            },
            torch.tensor([[offset + 10, offset + 11]]),
        ),
    ]


def _none_batches() -> list[Any]:
    return [
        (
            {"input": torch.tensor([[1, 2]]), "extra": None},
            torch.tensor([[2, 3]]),
        ),
        (
            {"input": torch.tensor([[4, 5]]), "extra": torch.tensor([7.0])},
            torch.tensor([[5, 6]]),
        ),
        (
            {"input": torch.tensor([[8, 9]]), "extra": None},
            torch.tensor([[9, 10]]),
        ),
    ]


def _init_process_group(rank: int, world_size: int, store_path: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )


def _spawn(worker: Any, world_size: int, *args: Any) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = str(Path(temp_dir) / "store")
        mp.spawn(
            worker,
            args=(world_size, store_path, *args),
            nprocs=world_size,
            join=True,
        )


def _make_wrapper(
    pp_mesh: Any, batches: list[Any], **kwargs: Any
) -> ReplicatedBatchDataLoader:
    is_owner = pp_mesh.get_local_rank() == 0
    return ReplicatedBatchDataLoader(
        _StatefulBatchLoader(batches) if is_owner else None,
        mesh=pp_mesh,
        device=torch.device("cpu"),
        **kwargs,
    )


def _assert_batch_matches(actual: Any, expected: Any, expected_keys: list[str]) -> None:
    assert list(actual[0]) == expected_keys
    for key in expected_keys:
        torch.testing.assert_close(actual[0][key], expected[0][key])
    torch.testing.assert_close(actual[1], expected[1])


def _check_two_epoch_round_trip(pp_mesh: Any, dp_rank: int, is_owner: bool) -> None:
    wrapper = _make_wrapper(pp_mesh, _batches(dp_rank))
    assert wrapper.is_owner == is_owner
    for _epoch in range(2):
        received = list(wrapper)
        assert len(received) == 2
        for actual, expected in zip(received, _batches(dp_rank), strict=True):
            _assert_batch_matches(actual, expected, list(expected[0]))
    assert wrapper.state_dict() == ({"next_index": 2} if is_owner else {})


def _check_owner_only_keys(pp_mesh: Any, dp_rank: int, is_owner: bool) -> None:
    wrapper = _make_wrapper(pp_mesh, _batches(dp_rank), owner_only_keys=("input",))
    for actual, expected in zip(list(wrapper), _batches(dp_rank), strict=True):
        expected_keys = [k for k in expected[0] if is_owner or k != "input"]
        _assert_batch_matches(actual, expected, expected_keys)


def _check_none_values_survive(pp_mesh: Any) -> None:
    wrapper = _make_wrapper(pp_mesh, _none_batches())
    for actual, expected in zip(list(wrapper), _none_batches(), strict=True):
        if expected[0]["extra"] is None:
            assert actual[0]["extra"] is None
        else:
            torch.testing.assert_close(actual[0]["extra"], expected[0]["extra"])


def _check_owner_error_propagates(pp_mesh: Any) -> None:
    wrapper = _make_wrapper(
        pp_mesh, [({"input": torch.ones(1), "bad": "not a tensor"}, torch.ones(1))]
    )
    with pytest.raises(
        RuntimeError, match="owner raised ValueError.*'bad' must be a Tensor"
    ):
        next(iter(wrapper))


def _check_changed_keys_propagate(pp_mesh: Any) -> None:
    wrapper = _make_wrapper(
        pp_mesh,
        [
            ({"input": torch.ones(1)}, torch.ones(1)),
            ({"other": torch.ones(1)}, torch.ones(1)),
        ],
    )
    iterator = iter(wrapper)
    next(iterator)
    with pytest.raises(RuntimeError, match="batch keys changed"):
        next(iterator)


def _transport_worker(
    rank: int, world_size: int, store_path: str, pp_size: int, dp_size: int
) -> None:
    _init_process_group(rank, world_size, store_path)
    try:
        world_mesh = init_device_mesh(
            "cpu", (pp_size, dp_size), mesh_dim_names=("pp", "batch")
        )
        pp_mesh = world_mesh["pp"]
        dp_rank = rank % dp_size
        is_owner = pp_mesh.get_local_rank() == 0

        _check_two_epoch_round_trip(pp_mesh, dp_rank, is_owner)
        _check_owner_only_keys(pp_mesh, dp_rank, is_owner)
        _check_none_values_survive(pp_mesh)
        _check_owner_error_propagates(pp_mesh)
        _check_changed_keys_propagate(pp_mesh)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("pp_size", "dp_size"), [(2, 2), (4, 1)])
def test_transport_distributed(pp_size: int, dp_size: int) -> None:
    _spawn(_transport_worker, pp_size * dp_size, pp_size, dp_size)


def _checkpoint_save_worker(
    rank: int, world_size: int, store_path: str, checkpoint_path: str
) -> None:
    _init_process_group(rank, world_size, store_path)
    try:
        pp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("pp",))
        wrapper = _make_wrapper(pp_mesh, _batches(0))
        first_batch = next(iter(wrapper))
        torch.testing.assert_close(first_batch[1], _batches(0)[0][1])
        dcp.save({"dataloader": wrapper}, checkpoint_id=checkpoint_path)
    finally:
        dist.destroy_process_group()


def test_checkpoint_resumes_into_pp1(tmp_path: Path) -> None:
    checkpoint_path = str(tmp_path / "checkpoint")
    _spawn(_checkpoint_save_worker, 2, checkpoint_path)

    store_path = str(tmp_path / "pp1_store")
    _init_process_group(0, 1, store_path)
    try:
        loader = _StatefulBatchLoader(_batches(0))
        dcp.load({"dataloader": loader}, checkpoint_id=checkpoint_path)
        assert loader.loaded_state == {"next_index": 1}
    finally:
        dist.destroy_process_group()


def _mock_pp_mesh(local_rank: int = 0) -> mock.Mock:
    pp_mesh = mock.Mock()
    pp_mesh.ndim = 1
    pp_mesh.mesh_dim_names = ("pp",)
    pp_mesh.get_group.return_value = object()
    pp_mesh.get_local_rank.return_value = local_rank
    pp_mesh.mesh = torch.tensor([0])
    pp_mesh.size.return_value = 2
    return pp_mesh


def _mocked_owner(batches: list[Any], **kwargs: Any) -> ReplicatedBatchDataLoader:
    return ReplicatedBatchDataLoader(
        _StatefulBatchLoader(batches),
        mesh=_mock_pp_mesh(),
        device=torch.device("cpu"),
        **kwargs,
    )


def test_two_collectives_per_batch() -> None:
    wrapper = _mocked_owner(_batches(0), owner_only_keys=("input",))
    with (
        mock.patch(
            "torchtitan.components.dataloader.dist.broadcast_object_list"
        ) as object_bcast,
        mock.patch("torchtitan.components.dataloader.dist.broadcast") as tensor_bcast,
    ):
        iterator = iter(wrapper)
        next(iterator)
        assert object_bcast.call_count == 1
        assert tensor_bcast.call_count == 2
        next(iterator)
        assert object_bcast.call_count == 1
        assert tensor_bcast.call_count == 4


def test_error_sent_before_tensor_collectives() -> None:
    wrapper = _mocked_owner(
        [({"input": torch.ones(1), "bad": object()}, torch.ones(1))]
    )
    with (
        mock.patch(
            "torchtitan.components.dataloader.dist.broadcast_object_list"
        ) as object_bcast,
        mock.patch("torchtitan.components.dataloader.dist.broadcast") as tensor_bcast,
    ):
        with pytest.raises(RuntimeError, match="owner raised ValueError"):
            next(iter(wrapper))
    object_bcast.assert_called_once()
    tensor_bcast.assert_not_called()


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (torch.zeros(2, dtype=torch.complex64), "Unsupported dtype"),
        (torch.zeros((1,) * 9), "exceeds transport limit"),
    ],
)
def test_wire_tensor_limits(value: torch.Tensor, match: str) -> None:
    wrapper = _mocked_owner([({"input": torch.ones(1), "extra": value}, torch.ones(1))])
    with mock.patch("torchtitan.components.dataloader.dist.broadcast_object_list"):
        with pytest.raises(RuntimeError, match=match):
            next(iter(wrapper))


def test_owner_delegates_raw_state() -> None:
    loader = _StatefulBatchLoader(_batches(0))
    wrapper = ReplicatedBatchDataLoader(
        loader, mesh=_mock_pp_mesh(), device=torch.device("cpu")
    )
    assert wrapper.state_dict() == {"next_index": 0}
    wrapper.load_state_dict({"next_index": 1})
    assert loader.loaded_state == {"next_index": 1}


def test_owner_rank_requires_real_loader() -> None:
    with pytest.raises(ValueError, match="rank 0 requires a real dataloader"):
        ReplicatedBatchDataLoader(
            None, mesh=_mock_pp_mesh(), device=torch.device("cpu")
        )


def test_receiver_rank_rejects_real_loader() -> None:
    with pytest.raises(ValueError, match="Only mesh axis rank 0"):
        ReplicatedBatchDataLoader(
            _StatefulBatchLoader(_batches(0)),
            mesh=_mock_pp_mesh(local_rank=1),
            device=torch.device("cpu"),
        )


def _parallel_dims(pp: int = 1) -> ParallelDims:
    return ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=pp, ep=1, world_size=pp
    )


def _as_loader_config(**fields: Any) -> BaseDataLoader.Config:
    return cast(BaseDataLoader.Config, SimpleNamespace(**fields))


def test_builder_passthrough_without_opt_in() -> None:
    loader = _StatefulBatchLoader(_batches(0))
    build = mock.Mock(return_value=loader)
    actual = build_replicated_dataloader(
        _as_loader_config(replicate_batch_over=()),
        build,
        parallel_dims=_parallel_dims(pp=2),
        device=torch.device("cpu"),
    )
    assert actual is loader
    build.assert_called_once_with()


def test_builder_passthrough_without_pp() -> None:
    loader = _StatefulBatchLoader(_batches(0))
    build = mock.Mock(return_value=loader)
    actual = build_replicated_dataloader(
        _as_loader_config(replicate_batch_over=("pp",)),
        build,
        parallel_dims=_parallel_dims(pp=1),
        device=torch.device("cpu"),
    )
    assert actual is loader
    build.assert_called_once_with()


@pytest.mark.parametrize(("local_rank", "expected_build_calls"), [(0, 1), (1, 0)])
def test_builder_splits_owner_and_receivers(
    local_rank: int, expected_build_calls: int
) -> None:
    parallel_dims = mock.Mock(pp=2, cp=1, tp=1)
    parallel_dims.get_mesh.return_value = _mock_pp_mesh(local_rank)
    build = mock.Mock(return_value=_StatefulBatchLoader(_batches(0)))

    wrapper = build_replicated_dataloader(
        _as_loader_config(replicate_batch_over=("pp",)),
        build,
        parallel_dims=parallel_dims,
        device=torch.device("cpu"),
    )

    parallel_dims.get_mesh.assert_called_once_with("pp")
    assert isinstance(wrapper, ReplicatedBatchDataLoader)
    assert wrapper.is_owner == (local_rank == 0)
    assert build.call_count == expected_build_calls


def test_builder_drops_degree_one_axes() -> None:
    parallel_dims = mock.Mock(pp=2, cp=1, tp=1)
    parallel_dims.get_mesh.return_value = _mock_pp_mesh(0)
    build = mock.Mock(return_value=_StatefulBatchLoader(_batches(0)))

    build_replicated_dataloader(
        _as_loader_config(replicate_batch_over=("tp", "pp", "cp")),
        build,
        parallel_dims=parallel_dims,
        device=torch.device("cpu"),
    )

    parallel_dims.get_mesh.assert_called_once_with("pp")
    build.assert_called_once_with()


def test_builder_flattens_replication_group() -> None:
    flat = _mock_pp_mesh(local_rank=1)
    multi = mock.Mock(ndim=3)
    multi._flatten.return_value = flat
    parallel_dims = mock.Mock(pp=2, cp=2, tp=2)
    parallel_dims.get_mesh.return_value = multi
    build = mock.Mock(return_value=_StatefulBatchLoader(_batches(0)))

    wrapper = build_replicated_dataloader(
        _as_loader_config(replicate_batch_over=("tp", "pp")),
        build,
        parallel_dims=parallel_dims,
        device=torch.device("cpu"),
    )

    parallel_dims.get_mesh.assert_called_once_with(["pp", "tp"])
    multi._flatten.assert_called_once_with("pp_tp")
    assert isinstance(wrapper, ReplicatedBatchDataLoader)
    assert wrapper.is_owner is False
    build.assert_not_called()


def test_builder_rejects_unsupported_axes() -> None:
    with pytest.raises(ValueError, match="only subsets of"):
        build_replicated_dataloader(
            _as_loader_config(replicate_batch_over=("batch", "ep")),
            mock.Mock(),
            parallel_dims=_parallel_dims(pp=2),
            device=torch.device("cpu"),
        )
