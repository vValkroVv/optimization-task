"""Frozen storage contract shared by native FSDP2 and GFSDP2."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed.gfsdp._state import (
    _grad_buffer_ctx_idx_map,
    get_buffer_ctx2ctx_using_param_groups_map,
)
from torchtitan.distributed.gfsdp.fully_shard import (
    _partition_by_dtype_and_grad,
    _partition_params,
)
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.trainer import cast_frozen_parameters


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_storage_policy_preserves_trainable_parameters_aliases_and_buffers(
    device, dtype
):
    model = nn.Module()
    model.base = nn.Linear(4, 4, device=device, dtype=dtype)
    model.base.requires_grad_(False)
    model.adapter = nn.Linear(4, 4, device=device, dtype=dtype)
    model.alias = nn.Module()
    model.alias.register_parameter("weight", model.base.weight)
    model.register_buffer(
        "frequencies", torch.ones(4, device=device, dtype=torch.float32)
    )
    model.register_parameter(
        "indices",
        nn.Parameter(
            torch.ones(4, device=device, dtype=torch.int64), requires_grad=False
        ),
    )
    base, adapter, buffer = model.base.weight, model.adapter.weight, model.frequencies
    expected = base.detach().to(torch.bfloat16).clone()
    training = TrainingConfig()
    assert training.frozen_dtype == "bfloat16"
    cast_frozen_parameters(model, dtype=TORCH_DTYPE_MAP[training.frozen_dtype])
    cast_frozen_parameters(model, dtype=TORCH_DTYPE_MAP[training.frozen_dtype])
    assert model.base.weight is base is model.alias.weight
    assert model.adapter.weight is adapter
    assert base.dtype == torch.bfloat16
    assert model.base.bias.dtype == torch.bfloat16
    assert adapter.dtype == dtype and adapter.requires_grad
    assert not base.requires_grad
    assert model.frequencies is buffer and buffer.dtype == torch.float32
    assert model.indices.dtype == torch.int64
    assert base.device.type == device
    if device != "meta":
        torch.testing.assert_close(base, expected, rtol=0, atol=0)


def test_frozen_bf16_storage_still_propagates_input_gradients():
    layer = nn.Linear(4, 4).requires_grad_(False)
    cast_frozen_parameters(layer)
    x = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)
    layer(x).float().square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None for p in layer.parameters())


def test_full_finetuning_parameters_are_untouched():
    model = nn.Linear(4, 4)
    parameters = list(model.parameters())
    before = [p.detach().clone() for p in parameters]
    cast_frozen_parameters(model)
    for p, previous, expected in zip(
        model.parameters(), parameters, before, strict=True
    ):
        assert p is previous and p.dtype == torch.float32
        torch.testing.assert_close(p, expected, rtol=0, atol=0)


def test_partitions_follow_mesh_then_storage_and_grad_without_losing_parameters():
    params = [
        nn.Parameter(torch.zeros(4, dtype=dtype), requires_grad=trainable)
        for dtype, trainable in [
            (torch.bfloat16, False),
            (torch.float32, True),
            (torch.bfloat16, False),
            (torch.bfloat16, True),
            (torch.float32, True),
            (torch.bfloat16, False),
        ]
    ]
    expert_mesh = object()
    partitions = [
        (mesh, group)
        for mesh, mesh_params in _partition_params(
            params, [(expert_mesh, {params[4], params[5]})]
        )
        for group in _partition_by_dtype_and_grad(mesh_params)
    ]
    assert partitions == [
        (None, [params[0], params[2]]),
        (None, [params[1]]),
        (None, [params[3]]),
        (expert_mesh, [params[4]]),
        (expert_mesh, [params[5]]),
    ]
    assert len({id(p) for _, group in partitions for p in group}) == len(params)
    assert _partition_by_dtype_and_grad([]) == []


class _Group:
    def __init__(self, name, pg, dtype, trainable):
        self.name = name
        self.mesh_info = SimpleNamespace(shard_process_group=pg)
        self.orig_dtype = dtype
        self.requires_grad = trainable
        self.data_buffer_ctx = object()
        self.grad_buffer_ctx = object() if trainable else None


@pytest.mark.parametrize("trainable_dtype", [torch.float32, torch.bfloat16])
def test_mixed_unit_and_prefetched_next_unit_have_distinct_data_buffers(
    trainable_dtype,
):
    pg = object()
    units = [
        [
            _Group(f"{i}:train", pg, trainable_dtype, True),
            _Group(f"{i}:frozen", pg, torch.bfloat16, False),
        ]
        for i in range(4)
    ]
    groups = [group for unit in units for group in unit]
    owners, _ = get_buffer_ctx2ctx_using_param_groups_map(
        groups, "data_buffer_ctx", 2, None, None
    )
    by_group = {group: buffer for buffer, users in owners.items() for group in users}
    assert len(owners) == 4
    for current, following in zip(units, units[1:]):
        assert len({by_group[group] for group in current + following}) == 4
    assert by_group[units[0][0]] is by_group[units[2][0]]
    assert by_group[units[0][1]] is by_group[units[2][1]]


def test_gradient_buffers_preserve_owned_buffers_and_separate_trainable_dtypes():
    pg = object()
    groups = [
        _Group(str(i), pg, dtype, True)
        for i, dtype in enumerate(
            [torch.float32, torch.bfloat16, torch.float32, torch.bfloat16]
        )
    ]
    mapping = _grad_buffer_ctx_idx_map(groups, {groups[2]})
    assert mapping[groups[0]] != mapping[groups[1]]
    assert mapping[groups[2]] not in {mapping[groups[0]], mapping[groups[1]]}
    assert mapping[groups[1]] == mapping[groups[3]]
