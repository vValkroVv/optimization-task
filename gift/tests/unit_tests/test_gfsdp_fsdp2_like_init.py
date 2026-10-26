"""gfsdp2's FSDP2-matching init lands FSDP2's own weights, bit for bit."""

import math

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import _random as random, DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.gfsdp._random import (
    _fsdp2_rng_offset_increment,
    fsdp2_equivalent_placements,
    match_fsdp2_init,
)
from torchtitan.distributed.gfsdp._tensor import (
    _ragged_flat_start,
    get_ragged_shard_axis,
    RaggedShard,
    RaggedShardDTensor,
)

_RAGGED = object()


def _empty_ragged_param(shape, mesh, placements, dtype):
    """An uninitialized parameter laid out as gfsdp lays it out.

    ``placements`` marks the ragged axis with ``_RAGGED``. The ragged split
    rounds up, so trailing ranks can own short or empty slices.
    """
    ragged_axis = placements.index(_RAGGED)
    replicated = tuple(Replicate() if p is _RAGGED else p for p in placements)
    partition_shape, _ = compute_local_shape_and_global_offset(shape, mesh, replicated)
    partition_numel = math.prod(partition_shape)
    num_shards = mesh.size(ragged_axis)
    chunk = (partition_numel + num_shards - 1) // num_shards
    sizes = tuple(
        max(0, min(chunk, partition_numel - i * chunk)) for i in range(num_shards)
    )
    index = mesh.get_coordinate()[ragged_axis]
    placement = RaggedShard(
        local_numel=sizes[index],
        global_offset=sum(sizes[:index]),
        shard_numels=sizes,
    )
    full_stride = torch.empty(shape).stride()
    spec = DTensorSpec(
        mesh,
        tuple(placement if p is _RAGGED else p for p in placements),
        TensorMeta(torch.Size(shape), full_stride, dtype),
    )
    local = torch.empty(sizes[index], device=mesh.device_type, dtype=dtype)
    return RaggedShardDTensor(local, spec, requires_grad=False)


def _fsdp2_slice(param, func, args, shard_dim):
    """This rank's ragged slice of the weights FSDP2 draws for ``param``.

    The draw goes through DTensor's own random-op path on the FSDP2 layout. The
    global tensor is reassembled from the shards the way DCP does it, since
    ``full_tensor`` reorders strided shards on uneven splits.
    """
    spec = param._spec
    shape = spec.tensor_meta.shape
    mesh = spec.mesh
    local = param._local_tensor
    placements = fsdp2_equivalent_placements(spec, shard_dim=shard_dim)
    local_shape, global_offset = compute_local_shape_and_global_offset(
        shape, mesh, placements
    )
    fsdp2_param = DTensor.from_local(
        local.new_empty(local_shape),
        mesh,
        placements,
        run_check=False,
        shape=shape,
        stride=spec.tensor_meta.stride,
    )
    func(fsdp2_param, *args)

    full = local.new_zeros(tuple(shape))
    if all(size > 0 for size in local_shape):
        region = tuple(
            slice(offset, offset + size)
            for offset, size in zip(global_offset, local_shape)
        )
        full[region] = fsdp2_param.to_local()
    for axis, placement in enumerate(placements):
        if placement.__class__ is not Replicate:
            torch.distributed.all_reduce(full, group=mesh.get_group(axis))
    start = _ragged_flat_start(param)
    return full.flatten()[start : start + local.numel()]


def _rng_offset(tracker) -> int:
    """The device generator's Philox offset, where the tracker keeps its state."""
    return int(random._PhiloxState(tracker._get_device_state()).offset.item())


def _set_rng_offset(tracker, offset: int) -> None:
    state = random._PhiloxState(tracker._get_device_state())
    state.offset = torch.tensor([offset], dtype=torch.int64)
    tracker._set_device_state(state.state)


class TestFSDP2LikeInit(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def _check_matches(self, mesh, placements, shapes, shard_dim=0):
        random.manual_seed(1234, mesh)
        tracker = random._rng_tracker
        draws = [
            (torch.ops.aten.normal_.default, [0.0, 0.02]),
            (torch.ops.aten.uniform_.default, [-0.1, 0.1]),
        ]
        for shape in shapes:
            for dtype in (torch.float32, torch.bfloat16):
                for func, args in draws:
                    param = nn.Parameter(
                        _empty_ragged_param(shape, mesh, placements, dtype),
                        requires_grad=False,
                    )
                    base = _rng_offset(tracker)
                    expected = _fsdp2_slice(param, func, args, shard_dim)
                    expected_offset = _rng_offset(tracker)

                    _set_rng_offset(tracker, base)
                    param._local_tensor.fill_(float("nan"))
                    # Through the ragged dispatch, which finds the shard dim by
                    # the parameter's identity, as init_weights reaches it.
                    shard_dims = {id(param): shard_dim} if shard_dim else {}
                    with match_fsdp2_init(fsdp2_shard_dims=shard_dims):
                        func(param, *args)
                    self.assertEqual(_rng_offset(tracker), expected_offset)
                    self.assertTrue(
                        torch.equal(param._local_tensor, expected),
                        f"{shape=} {dtype=} {func=} {param.placements=}",
                    )

    def _check_offset_formula(self, mesh, placements, shape):
        """Our copy of the tracker's formula agrees with the tracker itself."""
        random.manual_seed(1234, mesh)
        tracker = random._rng_tracker
        param = _empty_ragged_param(shape, mesh, placements, torch.float32)
        spec = param._spec
        fsdp2_spec = DTensorSpec(
            spec.mesh, fsdp2_equivalent_placements(spec), spec.tensor_meta
        )
        base = _rng_offset(tracker)
        with tracker._distribute_region(fsdp2_spec):
            actual = _rng_offset(tracker) - base
        self.assertEqual(
            actual,
            _fsdp2_rng_offset_increment(fsdp2_spec, mesh.get_coordinate()),
        )
        self.assertEqual(get_ragged_shard_axis(spec), placements.index(_RAGGED))

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_dp_only(self):
        mesh = init_device_mesh(self.device_type, (4,), mesh_dim_names=("dp",))
        placements = (_RAGGED,)
        # Even, uneven, fewer rows than ranks, one row.
        self._check_matches(mesh, placements, [(16, 3), (7, 5), (3, 2), (1, 9)])
        self._check_offset_formula(mesh, placements, (7, 5))

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_hsdp(self):
        mesh = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("replicate", "dp")
        )
        placements = (Replicate(), _RAGGED)
        self._check_matches(mesh, placements, [(8, 4), (5, 3), (1, 7)])
        self._check_offset_formula(mesh, placements, (5, 3))

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_expert_parallel(self):
        mesh = init_device_mesh(self.device_type, (2, 2), mesh_dim_names=("dp", "ep"))
        placements = (_RAGGED, Shard(0))
        # Experts split evenly and unevenly across EP, 3D expert weights.
        self._check_matches(mesh, placements, [(8, 6), (6, 5), (5, 3), (4, 3, 2)])
        self._check_offset_formula(mesh, placements, (6, 5))

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_fewer_experts_than_shards(self):
        """FSDP2 shards (E, D, F) experts on dim 1 and (E, 2, F, D) on dim 2."""
        mesh = init_device_mesh(self.device_type, (4,), mesh_dim_names=("dp",))
        placements = (_RAGGED,)
        # Even, uneven, and fewer rows than ranks on the sharded dim.
        self._check_matches(mesh, placements, [(2, 8, 3), (2, 5, 3)], shard_dim=1)
        self._check_matches(mesh, placements, [(2, 2, 4, 3), (3, 2, 3, 2)], shard_dim=2)

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_fewer_experts_than_shards_under_expert_parallel(self):
        mesh = init_device_mesh(self.device_type, (2, 2), mesh_dim_names=("dp", "ep"))
        placements = (_RAGGED, Shard(0))
        self._check_matches(mesh, placements, [(2, 6, 4), (2, 3, 5)], shard_dim=1)
