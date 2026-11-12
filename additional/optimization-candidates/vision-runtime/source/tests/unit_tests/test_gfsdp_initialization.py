"""Packed GFSDP initialization preserves weights without full-size temporaries."""

from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from torchtitan.distributed.gfsdp import fully_shard, MixedPrecisionPolicy


class _SeparatePayloadParameter(nn.Parameter):
    pass


@contextmanager
def _cpu_mesh(rank, world_size):
    # Construction has no collectives. Fake PG lets each rank's real constructor
    # run on CPU; reassembly below uses the actual shards, not fake collectives.
    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=world_size)
    try:
        yield init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size,empty_groups",
    [(4, False), (1, False), (4, True)],
    ids=["four_ranks", "one_rank", "empty_groups"],
)
@pytest.mark.parametrize("alignment", [1, 8])
@pytest.mark.parametrize(
    "dtype,trainable", [(torch.float32, True), (torch.bfloat16, False)]
)
@pytest.mark.parametrize("separate_payload", [False, True])
@torch.no_grad()
def test_packed_shards_preserve_values_offsets_and_aliases(
    alignment, dtype, trainable, separate_payload, world_size, empty_groups
):
    shapes = [(0,), (1,), (2, 3), (5, 7), (2,), (0,)]
    if empty_groups:
        shapes = [(0,)] * len(shapes)
    expected = [
        torch.arange(torch.Size(shape).numel(), dtype=dtype).view(shape) + index
        for index, shape in enumerate(shapes)
    ]
    pieces = [[] for _ in shapes]
    payload_groups = (
        ([0, 2, 4], [1, 3, 5]) if separate_payload else (list(range(len(shapes))),)
    )
    for rank in range(world_size):
        model = nn.Module()
        for index, value in enumerate(expected):
            parameter_cls = (
                _SeparatePayloadParameter
                if separate_payload and index % 2
                else nn.Parameter
            )
            model.register_parameter(
                f"weight_{index}",
                parameter_cls(value.clone(), requires_grad=trainable),
            )
        model.alias = model.weight_3
        policy = MixedPrecisionPolicy(
            all_gather_dtype_to_param_cls={torch.bfloat16: _SeparatePayloadParameter}
            if separate_payload
            else None
        )
        with _cpu_mesh(rank, world_size) as mesh:
            fully_shard(model, mesh=mesh, mp_policy=policy, shard_alignment=alignment)
            assert model.weight_3 is model.alias
            for indices in payload_groups:
                # Independent oracle: label each packed element explicitly, then
                # split the sequence, as opposed to recomputing interval overlap.
                owners = [
                    (index, offset)
                    for index in indices
                    for offset in range(expected[index].numel())
                ]
                while len(owners) % (world_size * alignment):
                    owners.append(None)
                shard_size = len(owners) // world_size
                local_owners = owners[rank * shard_size : (rank + 1) * shard_size]
                for index in indices:
                    param = getattr(model, f"weight_{index}")
                    local = param.to_local().detach().clone()
                    positions = [
                        entry[1]
                        for entry in local_owners
                        if entry is not None and entry[0] == index
                    ]
                    placement = param.placements[0]
                    assert placement.local_numel == len(positions)
                    assert placement.global_offset == (
                        positions[0] if positions else expected[index].numel()
                    )
                    assert placement.shard_numels == tuple(
                        sum(
                            entry is not None and entry[0] == index
                            for entry in owners[
                                rank_index * shard_size : (rank_index + 1) * shard_size
                            ]
                        )
                        for rank_index in range(world_size)
                    )
                    assert param.dtype == dtype and param.requires_grad == trainable
                    torch.testing.assert_close(
                        local, expected[index].flatten()[positions], rtol=0, atol=0
                    )
                    pieces[index].append(local)
    for value, shards in zip(expected, pieces, strict=True):
        torch.testing.assert_close(
            torch.cat(shards).view(value.shape), value, rtol=0, atol=0
        )


class _AllocationRecorder(TorchDispatchMode):
    def __init__(self, existing):
        super().__init__()
        self.storages = {tensor.untyped_storage()._cdata for tensor in existing}
        self.largest_new_storage = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for tensor in tree_flatten(result)[0]:
            if isinstance(tensor, torch.Tensor):
                # A DTensor wrapper describes the global shape without owning
                # that storage. Count its actual local tensor instead.
                tensor = getattr(tensor, "_local_tensor", tensor)
                storage = tensor.untyped_storage()
                if storage._cdata not in self.storages:
                    self.largest_new_storage = max(
                        self.largest_new_storage, storage.nbytes()
                    )
                    self.storages.add(storage._cdata)
        return result


@pytest.mark.parametrize("trainable", [False, True])
def test_constructor_allocates_at_most_local_shard_sized_storage(trainable):
    model = nn.Linear(1024, 1024, bias=False).requires_grad_(trainable)
    with _cpu_mesh(0, 4) as mesh:
        local_bytes = model.weight.numel() * model.weight.element_size() // 4
        recorder = _AllocationRecorder(list(model.parameters()))
        with recorder:
            fully_shard(model, mesh=mesh)
        assert recorder.largest_new_storage <= local_bytes
