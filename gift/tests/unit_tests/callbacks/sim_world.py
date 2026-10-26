"""A simulated multi-rank world for monitoring unit tests.

The monitoring pipeline reaches the distributed world through two seams only:

- ``torchtitan.callbacks.metrics.reducers.dist`` -- every collective a
  ``ParallelReducer`` issues ends in ``dist.all_reduce`` there;
- ``ParallelReducer.parallel_dims`` and ``ShardStats`` -- which mesh a group
  name resolves to, and how a sharded tensor exposes its local shard.

This module replaces exactly those seams and nothing else, so ``MonitorSource``,
``MetricPipeline``, ``_partial_stat_op``, ``ParallelReducer`` (group aliases,
pp validation, per-axis ``_reduce_mesh``, ``reduce_counting_copies``) and
``finalize_pending_reductions`` all run unmodified:

- ``SimWorld`` runs one thread per rank. A collective is a rendezvous of the
  group's threads that combines their buffers the way a ring all-reduce does
  (sequentially, in the buffer's dtype), so rounding behaves like NCCL's.
- ``SimTopology`` mirrors how ``ParallelDims.build_mesh`` lays ranks out.
- ``ShardedValue`` + ``SimShardStats`` stand in for a sharded parameter: the
  tensor carries the global shape, ``_local`` holds this rank's shard. They are
  registered as the reducer's ragged tensor type, so the GFSDP branches of the
  pipeline run; the FSDP2-only ``DTensor`` branch is covered by the real-PG
  tests.

Thread count is the simulated world size; 2048 threads are fine for the small
tensors these tests use.
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

from torchtitan.callbacks.metrics import reducers
from torchtitan.callbacks.metrics.reducers import (
    finalize_pending_reductions,
    ParallelReducer,
)
from torchtitan.callbacks.metrics.timers import gigatimer
from torchtitan.callbacks.metrics.sample import (
    Sample,
)
from torchtitan.config.schema.callbacks.metrics.sample import (
    AxisSpec,
)
from torchtitan.callbacks.metrics.stats import (
    _remaining_axes,
    apply_post_stats,
    apply_stat_op,
)
from torchtitan.callbacks.sources.base import MonitorSource
from torchtitan.distributed.gfsdp._tensor import RaggedShard

T = TypeVar("T")

# Same aliases as ``reducers._GROUP_ALIASES``; duplicated on purpose, so a
# change there shows up as a failing reference instead of moving both sides.
_GROUP_ALIASES = {
    "sp": "tp",
    "sequence": "tp",
    "dp": "batch",
    "data": "batch",
    "global": "world",
}


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SimLayout:
    """Parallel degrees of a simulated run (``ParallelDims`` without full_dtensor)."""

    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    tp: int = 1
    ep: int = 1
    pp: int = 1

    def __post_init__(self) -> None:
        if (self.fsdp * self.tp) % self.ep:
            raise ValueError(
                f"ep={self.ep} must divide dp_shard*cp*tp={self.fsdp * self.tp}"
            )

    @property
    def world_size(self) -> int:
        return self.pp * self.dp_replicate * self.dp_shard * self.cp * self.tp

    @property
    def batch(self) -> int:
        return self.dp_replicate * self.dp_shard

    @property
    def fsdp(self) -> int:
        return self.dp_shard * self.cp

    @property
    def efsdp(self) -> int:
        return self.fsdp * self.tp // self.ep

    def factorizations(self) -> dict[str, tuple[tuple[str, int], ...]]:
        """The global meshes ``build_mesh`` unflattens the world into, in its order."""
        return {
            "dataloading": (
                ("pp", self.pp),
                ("batch", self.batch),
                ("cp", self.cp),
                ("tp", self.tp),
            ),
            "dense": (
                ("pp", self.pp),
                ("dp_replicate", self.dp_replicate),
                ("fsdp", self.fsdp),
                ("tp", self.tp),
            ),
            "sparse": (
                ("pp", self.pp),
                ("dp_replicate", self.dp_replicate),
                ("efsdp", self.efsdp),
                ("ep", self.ep),
            ),
        }

    def describe(self) -> str:
        return (
            f"world={self.world_size} dp_replicate={self.dp_replicate} "
            f"dp_shard={self.dp_shard} cp={self.cp} tp={self.tp} ep={self.ep}"
        )


# Single-axis meshes of ``ParallelDims._single_axis_meshes`` (non full_dtensor):
# name -> (global mesh, the axes of it the mesh spans). ``loss`` is the
# flattened (batch, cp) mesh.
_SINGLE_AXIS = {
    "pp": ("dataloading", ("pp",)),
    "batch": ("dataloading", ("batch",)),
    "loss": ("dataloading", ("batch", "cp")),
    "dp_replicate": ("dense", ("dp_replicate",)),
    "cp": ("dataloading", ("cp",)),
    "tp": ("dataloading", ("tp",)),
    "ep": ("sparse", ("ep",)),
    "efsdp": ("sparse", ("efsdp",)),
    "fsdp": ("dense", ("fsdp",)),
}


@dataclass(frozen=True)
class SimGroup:
    """A process group of the simulated world."""

    ranks: tuple[int, ...]
    label: str

    def size(self) -> int:
        return len(self.ranks)


class SimMesh:
    """The subset of ``DeviceMesh`` the monitoring code reads, for one rank."""

    def __init__(self, rank: int, ranks: torch.Tensor, names: tuple[str, ...]):
        position = (ranks == rank).nonzero()
        if position.shape[0] != 1:
            raise AssertionError(f"rank {rank} is not in its own mesh {ranks}")
        self._rank = rank
        self._ranks = ranks
        self._coordinate = tuple(int(i) for i in position[0])
        self.mesh_dim_names = names

    @property
    def mesh(self) -> torch.Tensor:
        return self._ranks

    @property
    def ndim(self) -> int:
        return self._ranks.ndim

    def size(self, mesh_dim: int | None = None) -> int:
        if mesh_dim is None:
            return self._ranks.numel()
        return self._ranks.shape[mesh_dim]

    def get_coordinate(self) -> tuple[int, ...]:
        return self._coordinate

    def get_group(self, mesh_dim: int | str | None = None) -> SimGroup:
        if mesh_dim is None:
            if self.ndim != 1:
                raise ValueError("get_group() without a dim needs a 1D mesh")
            mesh_dim = 0
        if isinstance(mesh_dim, str):
            mesh_dim = self.mesh_dim_names.index(mesh_dim)
        index: list[Any] = list(self._coordinate)
        index[mesh_dim] = slice(None)
        ranks = tuple(int(r) for r in self._ranks[tuple(index)].flatten().tolist())
        return SimGroup(ranks=ranks, label=self.mesh_dim_names[mesh_dim])

    def __repr__(self) -> str:
        return f"SimMesh(names={self.mesh_dim_names}, ranks={self._ranks.tolist()})"


class SimTopology:
    """Which ranks share a mesh with which, exactly as ``ParallelDims`` lays them out.

    ``build_mesh`` unflattens the world mesh row-major, so a global mesh is
    ``arange(world).reshape(sizes)`` and a submesh is a slice of it.
    """

    def __init__(self, layout: SimLayout):
        self.layout = layout
        self._facts = layout.factorizations()
        self._grids = {
            fact: torch.arange(layout.world_size, dtype=torch.int64).reshape(
                [size for _, size in dims]
            )
            for fact, dims in self._facts.items()
        }
        self._mesh_cache: dict[tuple[int, tuple[str, ...]], SimMesh] = {}

    def grid(self, fact: str) -> torch.Tensor:
        """The whole global mesh ``fact``: ranks laid out over its axes."""
        return self._grids[fact]

    def axis_names(self, fact: str) -> tuple[str, ...]:
        return tuple(name for name, _ in self._facts[fact])

    def axis_exists(self, name: str) -> bool:
        """``ParallelDims._mesh_exist``: fsdp always, efsdp iff ep > 1, else size > 1."""
        if name == "fsdp":
            return True
        if name == "efsdp":
            return self.layout.ep > 1
        return self.axis_degree(name) > 1

    def axis_degree(self, name: str) -> int:
        fact, axes = _SINGLE_AXIS[name]
        sizes = dict(self._facts[fact])
        return math.prod(sizes[axis] for axis in axes)

    def _resolve(self, names: tuple[str, ...]) -> tuple[str, tuple[str, ...], bool]:
        """(global mesh, its axes ``names`` span, whether to flatten them)."""
        if len(names) == 1:
            fact, axes = _SINGLE_AXIS[names[0]]
            return fact, axes, True
        for fact in self._facts:
            order = self.axis_names(fact)
            if set(names) <= set(order):
                if list(names) != [name for name in order if name in names]:
                    raise ValueError(
                        f"mesh axes {names} are out of the {fact} mesh order {order}"
                    )
                return fact, names, False
        raise ValueError(f"Invalid mesh name combinations {list(names)}.")

    def _slice(self, fact: str, rank: int, axes: tuple[str, ...]) -> torch.Tensor:
        grid = self._grids[fact]
        order = self.axis_names(fact)
        coordinate = (grid == rank).nonzero()[0].tolist()
        index = tuple(
            slice(None) if name in axes else coordinate[i]
            for i, name in enumerate(order)
        )
        return grid[index]

    def mesh(self, rank: int, names: Sequence[str]) -> SimMesh:
        """Mesh of ``names`` for ``rank``, ignoring whether the axes are enabled."""
        names = tuple(names)
        key = (rank, names)
        cached = self._mesh_cache.get(key)
        if cached is not None:
            return cached
        fact, axes, flatten = self._resolve(names)
        ranks = self._slice(fact, rank, axes)
        mesh = SimMesh(rank, ranks.flatten() if flatten else ranks, names)
        self._mesh_cache[key] = mesh
        return mesh

    def partition(self, names: tuple[str, ...]) -> list[tuple[int, ...]]:
        """Every group the mesh of ``names`` cuts the world into."""
        fact, axes, _ = self._resolve(names)
        grid = self._grids[fact]
        order = self.axis_names(fact)
        inner = [order.index(axis) for axis in axes]
        outer = [i for i in range(len(order)) if i not in inner]
        rows = grid.permute(outer + inner).reshape(
            -1, math.prod(grid.shape[i] for i in inner)
        )
        return [tuple(sorted(row.tolist())) for row in rows]

    def reference_groups(
        self, group: str | tuple[str, ...] | None
    ) -> list[tuple[int, ...]]:
        """The partition of the world a reduce stage over ``group`` reduces within.

        Mirrors ``ParallelReducer.reduce_stage``: a disabled axis is a no-op
        (every rank alone), a tuple keeps its enabled axes only.
        """
        world = self.layout.world_size
        alone = [(rank,) for rank in range(world)]
        if group is None:
            return alone
        if isinstance(group, str):
            name = _GROUP_ALIASES.get(group, group)
            if name == "world":
                return [tuple(range(world))]
            if not self.axis_exists(name):
                return alone
            names: tuple[str, ...] = (name,)
        else:
            normalized = tuple(_GROUP_ALIASES.get(name, name) for name in group)
            if normalized == ("world",):
                return [tuple(range(world))]
            names = tuple(
                name
                for name in normalized
                if name in _SINGLE_AXIS and self.axis_exists(name)
            )
            if not names:
                return alone
        return self.partition(names)


class SimParallelDims:
    """The ``ParallelDims`` surface ``ParallelReducer`` reads, over a ``SimTopology``."""

    def __init__(self, layout: SimLayout, rank: Callable[[], int]):
        self.topology = SimTopology(layout)
        self._rank = rank

    @property
    def pp_enabled(self) -> bool:
        return self.topology.layout.pp > 1

    def get_optional_mesh(self, dims: str | Sequence[str]) -> SimMesh | None:
        names = [dims] if isinstance(dims, str) else list(dims)
        for name in names:
            if name not in _SINGLE_AXIS:
                raise ValueError(
                    f"Invalid mesh dim: '{name}'. "
                    f"Valid dimensions are: {list(_SINGLE_AXIS)}"
                )
        if any(not self.topology.axis_exists(name) for name in names):
            return None
        return self.topology.mesh(self._rank(), names)

    def get_activated_mesh(self, axes: Sequence[str]) -> SimMesh | None:
        enabled = [
            axis
            for axis in axes
            if axis in _SINGLE_AXIS and self.topology.axis_exists(axis)
        ]
        return self.get_optional_mesh(enabled) if enabled else None


# --------------------------------------------------------------------------
# Collectives
# --------------------------------------------------------------------------


class CollectiveMismatch(AssertionError):
    """Ranks of one group issued different collectives at the same position."""


class WorldAborted(RuntimeError):
    """Another rank failed; this one stopped waiting."""


@dataclass
class _Slot:
    meta: tuple
    values: dict[int, torch.Tensor] = field(default_factory=dict)
    result: torch.Tensor | None = None
    readers: int = 0
    done: threading.Event = field(default_factory=threading.Event)


class SimWorld:
    """``world_size`` ranks as threads, meeting at per-group rendezvous points."""

    def __init__(self, world_size: int, *, timeout: float = 30.0):
        self.world_size = world_size
        self.timeout = timeout
        self._lock = threading.Lock()
        self._slots: dict[tuple, _Slot] = {}
        self._sequence: dict[tuple[int, tuple[int, ...]], int] = {}
        self._local = threading.local()
        self._failure: BaseException | None = None
        # Per rank: (group label, group size, op, shape, dtype) of every collective.
        self.logs: list[list[tuple]] = [[] for _ in range(world_size)]

    @property
    def rank(self) -> int:
        rank = getattr(self._local, "rank", None)
        if rank is None:
            raise RuntimeError("SimWorld collective called outside SimWorld.run")
        return rank

    def all_reduce(self, tensor: torch.Tensor, op: Any, group: SimGroup | None) -> None:
        rank = self.rank
        ranks = group.ranks if group is not None else tuple(range(self.world_size))
        label = group.label if group is not None else "world"
        if rank not in ranks:
            raise AssertionError(
                f"rank {rank} reduces over {label} {ranks} without being in it"
            )
        meta = (str(op), tuple(tensor.shape), tensor.dtype)
        self.logs[rank].append((label, len(ranks), *meta))
        if len(ranks) == 1:
            return
        members = tuple(sorted(ranks))
        mine = tensor.detach().clone()  # outside the lock: 2048 ranks queue on it
        with self._lock:
            if self._failure is not None:
                raise WorldAborted("another rank failed") from self._failure
            seq = self._sequence.get((rank, members), 0)
            self._sequence[(rank, members)] = seq + 1
            key = (members, seq)
            slot = self._slots.setdefault(key, _Slot(meta=meta))
            if slot.meta != meta:
                error = CollectiveMismatch(
                    f"collective #{seq} over {label} {members[:8]}...: rank {rank} "
                    f"issued {meta}, an earlier rank issued {slot.meta}"
                )
                self._abort_locked(error)
                raise error
            slot.values[rank] = mine
            last = len(slot.values) == len(members)
        if last:
            # Every value is in and nobody writes the slot any more: combine
            # without holding the world lock.
            slot.result = _combine(slot.values, members, op)
            slot.done.set()
        if not slot.done.wait(self.timeout):
            with self._lock:
                arrived = sorted(slot.values)
                error = TimeoutError(
                    f"rank {rank}: collective #{seq} over {label} ({len(members)} ranks) "
                    f"timed out; {len(arrived)} arrived, missing "
                    f"{sorted(set(members) - set(arrived))[:16]}"
                )
                self._abort_locked(error)
            raise error
        if self._failure is not None and slot.result is None:
            raise WorldAborted("another rank failed") from self._failure
        assert slot.result is not None
        tensor.copy_(slot.result)
        with self._lock:
            slot.readers += 1
            if slot.readers == len(members):
                self._slots.pop(key, None)

    def _abort_locked(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
        for slot in self._slots.values():
            slot.done.set()

    def run(self, fn: Callable[[int], T]) -> list[T]:
        """Run ``fn(rank)`` on every rank concurrently; re-raise the first real failure."""
        results: list[Any] = [None] * self.world_size
        errors: dict[int, BaseException] = {}

        def target(rank: int) -> None:
            self._local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors[rank] = exc
                with self._lock:
                    self._abort_locked(exc)

        previous_threads = torch.get_num_threads()
        previous_stack = threading.stack_size()
        torch.set_num_threads(1)
        threading.stack_size(4 * 1024 * 1024)
        try:
            threads = [
                threading.Thread(target=target, args=(rank,), name=f"sim-rank-{rank}")
                for rank in range(self.world_size)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            threading.stack_size(previous_stack)
            torch.set_num_threads(previous_threads)
        if errors:
            real = {
                rank: exc
                for rank, exc in errors.items()
                if not isinstance(exc, WorldAborted)
            }
            rank, exc = min((real or errors).items())
            raise RuntimeError(
                f"simulated rank {rank} of {self.world_size} failed: {exc!r}"
            ) from exc
        return results


def _combine(
    values: dict[int, torch.Tensor], members: tuple[int, ...], op: Any
) -> torch.Tensor:
    """Ring-style reduction: one rank after another, in the buffer's dtype."""
    result = values[members[0]].clone()
    for rank in members[1:]:
        if op == dist.ReduceOp.SUM:
            result.add_(values[rank])
        elif op == dist.ReduceOp.MAX:
            torch.maximum(result, values[rank], out=result)
        elif op == dist.ReduceOp.MIN:
            torch.minimum(result, values[rank], out=result)
        else:
            raise NotImplementedError(f"SimWorld does not implement reduce op {op}")
    return result


def install(monkeypatch: Any, world: SimWorld) -> SimWorld:
    """Route the reducers' collectives into ``world`` for the rest of the test.

    Also pins the timer registry off: ``TimerRecord`` keeps its open span on
    the shared instance, so timed functions are not safe to enter from many
    threads at once (production runs them on one thread per process).
    """
    monkeypatch.setattr(reducers, "dist", SimDist(world))
    monkeypatch.setattr(gigatimer, "timers_threshold", 0)
    return world


class SimDist:
    """Drop-in for ``reducers.dist``: the calls ``_reduce_tensor`` and ``_dist_ready`` make."""

    ReduceOp = dist.ReduceOp

    def __init__(self, world: SimWorld):
        self._world = world

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if async_op:
            raise NotImplementedError("SimDist is synchronous")
        self._world.all_reduce(tensor, op, group)

    def get_world_size(self, group=None) -> int:
        return self._world.world_size if group is None else group.size()

    def get_rank(self, group=None) -> int:
        rank = self._world.rank
        return rank if group is None else group.ranks.index(rank)


# --------------------------------------------------------------------------
# Sharded parameters
# --------------------------------------------------------------------------


class ShardedValue(torch.Tensor):
    """A sharded parameter on one simulated rank: global shape, local shard inside.

    Torch ops on it return plain tensors (``__torch_function__`` disabled); the
    monitoring code only ever computes on ``SimShardStats.to_local`` /
    ``to_full_tensor``, plus the few tensor methods overridden below.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, full, local, placements, device_mesh):
        value = torch.Tensor._make_subclass(cls, full, False)
        value._full = full
        value._local = local
        value.placements = tuple(placements)
        value.device_mesh = device_mesh
        return value

    def _like(self, full: torch.Tensor, local: torch.Tensor) -> "ShardedValue":
        return ShardedValue(full, local, self.placements, self.device_mesh)

    def detach(self):  # type: ignore[override]
        return self

    def abs(self):  # type: ignore[override]
        return self._like(self._full.abs(), self._local.abs())

    def float(self):  # type: ignore[override]
        if self.dtype == torch.float32:
            return self
        return self._like(self._full.float(), self._local.float())


class SimShardStats:
    """``ShardStats`` for ``ShardedValue``: whole-tensor statistics only.

    The per-segment and per-row helpers belong to ``GFSDPShardStats`` and are
    tested on real ``RaggedShardDTensor`` in ``test_gfsdp_moe_monitor_mesh``.
    """

    tensor_types: tuple[type[torch.Tensor], ...] = (ShardedValue,)
    placement_types: tuple[type, ...] = (RaggedShard,)

    def to_local(self, value: torch.Tensor) -> torch.Tensor:
        assert isinstance(value, ShardedValue)
        return value._local

    def to_full_tensor(self, value: torch.Tensor) -> torch.Tensor:
        assert isinstance(value, ShardedValue)
        return value._full.clone()

    def local_squared_norms(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        raise NotImplementedError(
            "axis norms of ragged shards are tested on real RaggedShardDTensor"
        )


def split_sizes(
    numel: int, parts: int, policy: str, *, seed: int = 0
) -> tuple[int, ...]:
    """Shard sizes of a flat range of ``numel`` elements.

    - ``even``: ``ceil(numel / parts)`` per shard, empties at the tail (FSDP2's
      ``torch.chunk`` and GFSDP's default split);
    - ``uneven``: random cut points, so sizes differ and some shards are empty;
    - ``first``: everything on the first shard, every other shard empty.
    """
    if policy == "even":
        chunk = -(-numel // parts)
        return tuple(
            max(0, min(chunk, numel - index * chunk)) for index in range(parts)
        )
    if policy == "uneven":
        rng = random.Random(seed)
        cuts = sorted(rng.randint(0, numel) for _ in range(parts - 1))
        bounds = [0, *cuts, numel]
        return tuple(bounds[i + 1] - bounds[i] for i in range(parts))
    if policy == "first":
        return (numel,) + (0,) * (parts - 1)
    raise ValueError(f"unknown split policy {policy!r}")


@dataclass(frozen=True)
class ParamLayout:
    """How a parameter lies on its mesh: one kind per mesh axis.

    Kinds: ``replicate``; ``shard0`` (``torch.chunk`` of dim 0, FSDP2 / EP);
    ``ragged`` (a flat element range, GFSDP; at most one axis). ``shard0`` axes
    nest outermost-last, so ``("efsdp", "ep")`` with ``("ragged", "shard0")``
    is GFSDP's expert layout: EP partitions the experts, efsdp splits each
    partition into flat ranges.
    """

    axes: tuple[str, ...]
    kinds: tuple[str, ...]
    policy: str = "even"


def shard_for_rank(
    full: torch.Tensor,
    layout: ParamLayout,
    mesh: SimMesh,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, tuple[Placement, ...]]:
    """This rank's local shard of ``full`` and the placements describing it."""
    coordinate = mesh.get_coordinate()
    region = full
    placements: list[Placement | None] = [None] * len(layout.axes)
    for axis in reversed(range(len(layout.axes))):
        kind = layout.kinds[axis]
        if kind == "replicate":
            placements[axis] = Replicate()
        elif kind == "shard0":
            parts = mesh.size(axis)
            chunk = -(-region.shape[0] // parts)
            start = coordinate[axis] * chunk
            region = region[start : start + chunk]
            placements[axis] = Shard(0)
    ragged_axes = [i for i, kind in enumerate(layout.kinds) if kind == "ragged"]
    local = region
    if ragged_axes:
        (axis,) = ragged_axes
        flat = region.reshape(-1)
        sizes = split_sizes(flat.numel(), mesh.size(axis), layout.policy, seed=seed)
        offset = sum(sizes[: coordinate[axis]])
        local = flat[offset : offset + sizes[coordinate[axis]]]
        placements[axis] = RaggedShard(
            local_numel=local.numel(), global_offset=offset, shard_numels=sizes
        )
    assert all(p is not None for p in placements)
    return local.clone(), tuple(placements)  # type: ignore[arg-type]


def sharded_value(
    full: torch.Tensor,
    layout: ParamLayout,
    topology: SimTopology,
    rank: int,
    *,
    seed: int = 0,
) -> ShardedValue:
    mesh = topology.mesh(rank, layout.axes)
    local, placements = shard_for_rank(full, layout, mesh, seed=seed)
    return ShardedValue(full, local, placements, mesh)


# --------------------------------------------------------------------------
# Running sources and references
# --------------------------------------------------------------------------


def run_monitor(
    world: SimWorld,
    layout: SimLayout,
    build_samples: Callable[[int], Sequence[Sample]],
    *,
    shard_stats: SimShardStats | None = None,
    accumulate: bool = False,
    debug_full_tensor: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Every rank records its samples, then runs the batch_end chain of MonitoringCallback.

    Returns, per rank, ``{tag: value}`` after ``finalize_pending_reductions``
    and ``apply_post_stats`` -- what the emitter would publish.
    """
    parallel_dims = SimParallelDims(layout, lambda: world.rank)

    def body(rank: int) -> dict[str, torch.Tensor]:
        source = MonitorSource(MonitorSource.Config())
        source.active = True
        source.debug_full_tensor = debug_full_tensor
        source.reducer = ParallelReducer(parallel_dims, shard_stats=shard_stats)
        for sample in build_samples(rank):
            source._record(sample, accumulate=accumulate)
        if accumulate:
            source._flush_captures()
        records = finalize_pending_reductions(source.drain(), source.reducer)
        records = apply_post_stats(records)
        return {
            record.name: torch.as_tensor(record.value).detach().to(torch.float64)
            for record in records
        }

    return world.run(body)


def reference_stats(
    value: torch.Tensor, stats: Sequence[Any], axes: AxisSpec = AxisSpec()
) -> torch.Tensor:
    """A chain of stat ops on one full tensor in fp64: the single-process oracle."""
    result = value.detach().to(torch.float64)
    for op in stats:
        result = apply_stat_op(result, axes, op)
        if op.dim is not None:
            axes = _remaining_axes(axes, axes.resolve(op.dim))
    return result.to(torch.float64)
