"""Random initialization of ``RaggedShardDTensor`` parameters."""

import contextlib
import logging
import math
from collections.abc import Iterator, Mapping
from typing import Any

import torch
import torch.distributed.tensor._random as random
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard, Placement, Shard

from ._tensor import _ragged_flat_start, get_ragged_shard_axis, RaggedShardDTensor

logger = logging.getLogger("gfsdp")

_match_fsdp2_init = False
# id(parameter) -> the dim FSDP2 shards it on, for those it does not shard on 0.
_fsdp2_shard_dims: Mapping[int, int] = {}
# The torch whose DTensor RNG internals _fill_random_matching_fsdp2 mirrors and
# was checked against bit for bit; the "+cu..." local suffix is ignored.
_FSDP2_MATCHING_INIT_TORCH_VERSION = "2.15.0.dev20260822"


@contextlib.contextmanager
def match_fsdp2_init(
    enabled: bool = True, *, fsdp2_shard_dims: Mapping[int, int] | None = None
) -> Iterator[None]:
    """Within the block, ragged parameters' random init reproduces FSDP2's.

    ``fsdp2_shard_dims`` maps ``id(parameter)`` to the dim FSDP2 shards it on,
    for parameters it does not shard on dim 0 (routed experts with fewer experts
    than shards); any other parameter is taken as ``Shard(0)``.
    """
    global _match_fsdp2_init, _fsdp2_shard_dims
    if enabled:
        logger.warning(
            "gfsdp2: initialization matches fsdp2 bit for bit, at roughly twice "
            "the RNG work of the default draw: a rank draws every FSDP2 shard its "
            "ragged range overlaps, usually its own and a neighbour's, and all of "
            "them for experts fsdp2 shards below dim 0."
        )
        torch_version = str(torch.__version__).split("+")[0]
        if torch_version != _FSDP2_MATCHING_INIT_TORCH_VERSION:
            logger.warning(
                f"gfsdp2: fsdp2-matching init mirrors the DTensor RNG internals of "
                f"torch {_FSDP2_MATCHING_INIT_TORCH_VERSION}, this is torch "
                f"{torch_version}; the init may silently stop matching fsdp2. Run "
                f"tests/unit_tests/test_gfsdp_fsdp2_like_init.py on GPUs to check."
            )
    previous = _match_fsdp2_init, _fsdp2_shard_dims
    _match_fsdp2_init, _fsdp2_shard_dims = enabled, dict(fsdp2_shard_dims or {})
    try:
        yield
    finally:
        _match_fsdp2_init, _fsdp2_shard_dims = previous


def fsdp2_equivalent_placements(
    spec: DTensorSpec, *, shard_dim: int = 0
) -> tuple[Placement, ...]:
    """The placements FSDP2 gives this parameter, in place of the ragged axis.

    A dim 0 already sharded on another axis makes the outer shard strided.
    """
    axis = get_ragged_shard_axis(spec)
    if shard_dim != 0:
        placements = list(spec.placements)
        placements[axis] = Shard(shard_dim)
        return tuple(placements)
    split_factor = 1
    for i, placement in enumerate(spec.placements):
        if i != axis and placement.__class__ is Shard:
            split_factor *= spec.mesh.size(i)

    placements = list(spec.placements)
    placements[axis] = (
        Shard(0) if split_factor == 1 else _StridedShard(0, split_factor=split_factor)
    )
    return tuple(placements)


def _fill_random(arg: RaggedShardDTensor, func: Any, args: Any, kwargs: Any) -> None:
    """Fill the local ragged slice with a random init op, cheaply by default.

    The default draw has the right distribution but lands different numbers at
    the same positions than FSDP2, so the two backends start from different
    models and a paired run compares two models rather than two backends.
    ``match_fsdp2_init`` buys the match back by modelling the RNG tracker's
    per-shard offsets, without communication.
    """
    if _match_fsdp2_init:
        _fill_random_matching_fsdp2(arg, func, args, kwargs)
    else:
        _fill_random_per_rank(arg, func, args, kwargs)


def _fill_random_per_rank(
    arg: RaggedShardDTensor, func: Any, args: Any, kwargs: Any
) -> None:
    """One draw per rank, straight into the local slice. The default.

    The FSDP2-equivalent spec is built for its offset rather than its values:
    the offset is distinct per rank, so ranks draw different numbers instead of
    all drawing the same ones, and the tracker advances it past the whole
    parameter, so no two parameters share a stream position.
    """
    spec = arg._spec
    reference = DTensorSpec(
        spec.mesh, fsdp2_equivalent_placements(spec), spec.tensor_meta
    )
    assert random._rng_tracker is not None, (
        "initialization reached a RaggedShardDTensor outside an RNG region; "
        "the tracker is what keeps the draw distinct per rank and reproducible"
    )
    with random._rng_tracker._distribute_region(reference):
        func(arg._local_tensor, *args, **kwargs)


def _fsdp2_rng_offset_increment(spec: DTensorSpec, coordinate: list[int]) -> int:
    """RNG offset, past the region's base, of the shard held at ``coordinate``.

    ``OffsetBasedRNGTracker._compute_rng_offsets``'s start increment, for any
    rank's coordinate rather than only the current one.
    """
    shard_idx_by_dim, num_shards_by_dim = random._calc_shard_info(coordinate, spec)
    shard_linear_idx = random._calc_shard_linear_idx(
        shard_idx_by_dim, num_shards_by_dim
    )
    first_shard_numel = math.prod(random._calc_first_shard_size(spec))
    # CUDA generator offsets must be multiples of 4.
    return (shard_linear_idx * first_shard_numel + 3) // 4 * 4


def _fill_random_matching_fsdp2(
    arg: RaggedShardDTensor, func: Any, args: Any, kwargs: Any
) -> None:
    """Fill the local ragged slice with the values FSDP2 would have put there.

    Draws every FSDP2 shard the ragged range overlaps, whole and at that shard's
    own RNG offset, and copies out the overlap: a partial draw lands different
    numbers, since a CUDA kernel's element-to-stream mapping depends on its numel.
    A shard on dim 0 overlaps the range as a flat run; a shard below dim 0 is a
    box, so the range is located in it element by element.

    Mirrors ``OffsetBasedRNGTracker._distribute_region`` (torch 2.15) around
    ``_fsdp2_rng_offset_increment``, and is tested bit for bit against FSDP2's
    own DTensor draw.
    """
    spec = arg._spec
    assert spec.tensor_meta is not None
    shape = spec.tensor_meta.shape
    mesh = spec.mesh
    local = arg._local_tensor
    ragged_axis = get_ragged_shard_axis(spec)
    shard_dim = _fsdp2_shard_dims.get(id(arg), 0)
    placements = fsdp2_equivalent_placements(spec, shard_dim=shard_dim)
    for axis, placement in enumerate(placements):
        if (
            axis != ragged_axis
            and isinstance(placement, Shard | _StridedShard)
            and placement.dim != 0
        ):
            raise NotImplementedError(
                f"FSDP2-matching init assumes a TP/EP shard on dim 0, got {placements}"
            )
    fsdp2_spec = DTensorSpec(mesh, placements, spec.tensor_meta)

    tracker = random._rng_tracker
    assert tracker is not None, (
        "initialization reached a RaggedShardDTensor outside an RNG region; "
        "the tracker is what keeps the draw distinct per rank and reproducible"
    )
    if not tracker.distribute_region_enabled:
        # _distribute_region draws from the ambient generator in this case.
        func(local, *args, **kwargs)
        return

    coordinate = mesh.get_coordinate()
    assert coordinate is not None
    row_numel = math.prod(shape[1:])
    start = _ragged_flat_start(arg)
    end = start + local.numel()
    local_flat = local.view(-1)
    if shard_dim != 0:
        global_index = torch.unravel_index(
            torch.arange(start, end, device=local.device), shape
        )

    # The tracker keeps no state of its own: the offset is the device
    # generator's, read and written through its packed Philox state.
    state = random._PhiloxState(tracker._get_device_state())
    base = state.offset.clone()
    with torch.random.fork_rng(
        devices=[tracker._device], device_type=tracker._device.type
    ):
        # Only the ragged coordinate varies: the TP/EP one pins the partition
        # and a Replicate one does not move the shard.
        for i in range(mesh.size(ragged_axis)):
            shard_coordinate = list(coordinate)
            shard_coordinate[ragged_axis] = i
            shard_shape, shard_offset = _compute_local_shape_and_global_offset(
                shape, mesh.shape, shard_coordinate, placements
            )
            if shard_dim == 0:
                shard_start = shard_offset[0] * row_numel
                lo = max(start, shard_start)
                hi = min(end, shard_start + shard_shape[0] * row_numel)
                if lo >= hi:
                    continue
            else:
                shard_index = [g - o for g, o in zip(global_index, shard_offset)]
                inside = torch.ones_like(local_flat, dtype=torch.bool)
                for index, size in zip(shard_index, shard_shape):
                    inside &= (index >= 0) & (index < size)
                if not inside.any():
                    continue
            state.offset = base + _fsdp2_rng_offset_increment(
                fsdp2_spec, shard_coordinate
            )
            tracker._device_handle.set_rng_state(state.state)
            buf = local.new_empty(shard_shape)
            func(buf, *args, **kwargs)
            if shard_dim == 0:
                local_flat[lo - start : hi - start] = buf.view(-1)[
                    lo - shard_start : hi - shard_start
                ]
            else:
                flat_index = sum(i * s for i, s in zip(shard_index, buf.stride()))
                local_flat[inside] = buf.view(-1)[flat_index[inside]]
    state.offset = base + (shape.numel() + 3) // 4 * 4
    tracker._set_device_state(state.state)
