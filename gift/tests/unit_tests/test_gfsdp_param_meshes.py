"""One GFSDP unit, more than one sharding mesh.

Routed experts shard over the expert-FSDP axis while the rest of their block
shards over the data-parallel one. Before ``param_meshes`` that needed a nested
``fully_shard`` on ``block.moe.experts`` -- a module inside the activation
checkpoint wrapper, so its hooks re-fired on recompute and desynchronized the
op sequence selective activation checkpointing replays.

``_partition_params`` is the part that decides which parameters land in which
group. It treats a mesh as an opaque key, so these tests stand sentinels in for
``DeviceMesh`` and need no distributed setup.
"""

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn

from torchtitan.distributed.gfsdp._state import _shard_axis_name
from torchtitan.distributed.gfsdp.fully_shard import (
    _partition_params,
    _UnshardHandleImpl,
)

DP_MESH = object()
EDP_MESH = object()


def _params(count: int) -> list[nn.Parameter]:
    return [nn.Parameter(torch.zeros(2)) for _ in range(count)]


def test_no_param_meshes_keeps_one_group_on_the_units_own_mesh():
    params = _params(3)

    assert _partition_params(params, None) == [(None, params)]
    assert _partition_params(params, []) == [(None, params)]


def test_a_unit_with_no_parameters_gets_no_group():
    assert _partition_params([], None) == []
    assert _partition_params([], [(EDP_MESH, set())]) == []


def test_the_units_own_group_comes_first_and_keeps_sharding_order():
    dense_a, expert, dense_b = _params(3)
    params = [dense_a, expert, dense_b]

    partitions = _partition_params(params, [(EDP_MESH, {expert})])

    assert partitions == [(None, [dense_a, dense_b]), (EDP_MESH, [expert])]


def test_every_parameter_claimed_leaves_no_empty_default_group():
    params = _params(2)

    partitions = _partition_params(params, [(EDP_MESH, set(params))])

    assert partitions == [(EDP_MESH, params)]


def test_a_mesh_matching_no_parameter_is_dropped():
    params = _params(2)

    partitions = _partition_params(params, [(EDP_MESH, set())])

    assert partitions == [(None, params)]


def test_several_meshes_keep_their_declared_order():
    first, second, dense = _params(3)
    params = [first, second, dense]

    partitions = _partition_params(params, [(EDP_MESH, {second}), (DP_MESH, {first})])

    assert partitions == [
        (None, [dense]),
        (EDP_MESH, [second]),
        (DP_MESH, [first]),
    ]


def test_an_unmanaged_parameter_is_refused():
    params = _params(2)
    (stranger,) = _params(1)

    with pytest.raises(ValueError, match="does not manage"):
        _partition_params(params, [(EDP_MESH, {stranger})])


def test_a_parameter_under_two_meshes_is_refused():
    params = _params(2)

    with pytest.raises(ValueError, match="more than one mesh"):
        _partition_params(params, [(EDP_MESH, {params[0]}), (DP_MESH, {params[0]})])


class _StubAllGatherResult:
    def __init__(self) -> None:
        self.waits = 0

    def wait(self) -> None:
        self.waits += 1


class _StubParamGroup:
    def __init__(self, result: _StubAllGatherResult | None) -> None:
        self._result = result
        self.unshard_waits = 0

    def wait_for_unshard(self) -> _StubAllGatherResult | None:
        self.unshard_waits += 1
        return self._result


def test_the_unshard_handle_waits_every_group_once():
    groups = [_StubParamGroup(_StubAllGatherResult()) for _ in range(2)]

    _UnshardHandleImpl(groups).wait()

    assert [group.unshard_waits for group in groups] == [1, 1]


def test_waiting_the_unshard_handle_twice_does_not_re_unshard():
    results = [_StubAllGatherResult() for _ in range(2)]
    groups = [_StubParamGroup(result) for result in results]

    handle = _UnshardHandleImpl(groups)
    handle.wait()
    handle.wait()

    assert [group.unshard_waits for group in groups] == [1, 1]
    assert [result.waits for result in results] == [1, 1]


def test_the_unshard_handle_drops_groups_with_nothing_pending():
    pending = _StubAllGatherResult()
    groups = [_StubParamGroup(None), _StubParamGroup(pending)]

    handle = _UnshardHandleImpl(groups)
    handle.wait()
    handle.wait()

    assert pending.waits == 1


def test_an_empty_unshard_handle_is_a_no_op():
    _UnshardHandleImpl([]).wait()


class _StubMesh:
    def __init__(self, mesh_dim_names):
        self.mesh_dim_names = mesh_dim_names


class _StubMeshInfo:
    def __init__(self, mesh_dim_names, shard_mesh_dim):
        self.mesh = _StubMesh(mesh_dim_names)
        self.shard_mesh_dim = shard_mesh_dim


def test_the_shard_axis_is_named_after_the_mesh():
    assert _shard_axis_name(_StubMeshInfo(("efsdp",), 0)) == "efsdp"
    assert _shard_axis_name(_StubMeshInfo(("dp_replicate", "fsdp"), 1)) == "fsdp"


def test_an_unnamed_mesh_falls_back_to_the_axis_index():
    assert _shard_axis_name(_StubMeshInfo(None, 1)) == "1"
