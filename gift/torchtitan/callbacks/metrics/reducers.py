import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from torchtitan.callbacks.metrics.sample import MetricRecord
from torchtitan.config.schema.callbacks.metrics.sample import (
    ReduceOp,
    ReduceStageConfig,
)
from torchtitan.callbacks.metrics.timers import gigatimer
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.sharding_backend import ShardStats
from torchtitan.tools.logging import logger

STEP_SEP = "\x1f"

_GROUP_ALIASES = {
    # TorchTitan implements sequence-parallel activation sharding on the TP mesh.
    "sp": "tp",
    "sequence": "tp",
    # Data-loader batch mesh includes dp_replicate and fsdp/dp_shard.
    "dp": "batch",
    "data": "batch",
    "global": "world",
}

# "mean" maps onto SUM: it is a SUM collective followed by a division by the
# group size, which happens in _reduce_tensor.
_REDUCE_OPS = {
    ReduceOp.sum: dist.ReduceOp.SUM,
    ReduceOp.mean: dist.ReduceOp.SUM,
    ReduceOp.max: dist.ReduceOp.MAX,
    ReduceOp.min: dist.ReduceOp.MIN,
}


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _reduce_tensor(
    value: torch.Tensor,
    *,
    group: dist.ProcessGroup | None,
    op: ReduceOp,
) -> torch.Tensor:
    """Run one distributed reduction and return a detached result."""
    reduce_op = _REDUCE_OPS.get(op)
    if reduce_op is None:
        raise ValueError(f"Unsupported monitoring reduce op: {op}")

    result = value.detach().clone()
    dist.all_reduce(result, op=reduce_op, group=group)

    if op == ReduceOp.mean:
        result /= dist.get_world_size(group)

    return result


@dataclass(slots=True)
class ParallelReducer:
    parallel_dims: ParallelDims
    shard_stats: ShardStats | None = None
    ragged_tensor_types: tuple[type[torch.Tensor], ...] = field(init=False, repr=False)
    sharded_tensor_types: tuple[type[torch.Tensor], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ragged_tensor_types = (
            self.shard_stats.tensor_types if self.shard_stats is not None else ()
        )
        self.sharded_tensor_types = (DTensor, *self.ragged_tensor_types)

    def reduce_stages(
        self,
        value: torch.Tensor,
        stages: Iterable[ReduceStageConfig],
    ) -> torch.Tensor:
        """Apply a sequence of reduction stages in order."""
        result = value
        for stage in stages:
            result = self.reduce_stage(result, stage)
        return result

    @staticmethod
    def ranks_of(mesh: DeviceMesh) -> frozenset[int]:
        """The global ranks of ``mesh`` (this rank's slice of it, for a submesh)."""
        return frozenset(int(r) for r in mesh.mesh.flatten().tolist())

    def reduce_counting_copies(
        self,
        value: torch.Tensor,
        stages: Iterable[ReduceStageConfig],
        copies: frozenset[int],
    ) -> tuple[torch.Tensor, frozenset[int]]:
        """Reduce a value that the ranks in ``copies`` hold identically."""
        result = value
        for stage in stages:
            members = self._stage_ranks(stage)
            if members is None:
                continue
            if members <= copies:
                continue
            shared = members & copies
            if (
                stage.op == ReduceOp.sum
                and len(shared) > 1
                and dist.get_rank() != min(shared)
            ):
                result = torch.zeros_like(result)
            result = self.reduce_stage(result, stage)
            copies = copies | members
        return result, copies

    def _stage_ranks(self, stage: ReduceStageConfig) -> frozenset[int] | None:
        """Global ranks this rank reduces with in ``stage``; None for a no-op stage."""
        if stage.group is None or stage.op == ReduceOp.none or not _dist_ready():
            return None
        group = self._normalize_group(stage.group)
        self._validate_group(group, stage)
        if group == "world":
            return frozenset(range(dist.get_world_size()))
        mesh = self._get_mesh(group)
        if mesh is None or mesh.size() <= 1:
            return None
        return frozenset(int(r) for r in mesh.mesh.flatten().tolist())

    def reduce_stage(
        self,
        value: torch.Tensor,
        stage: ReduceStageConfig,
    ) -> torch.Tensor:
        """Apply one configured reduction stage.

        No-op stages return ``value`` unchanged; a new detached tensor is
        produced only when a collective actually runs.
        """
        if stage.group is None or stage.op == ReduceOp.none or not _dist_ready():
            return value

        group = self._normalize_group(stage.group)
        self._validate_group(group, stage)

        if group == "world":
            return _reduce_tensor(value, group=None, op=stage.op)

        mesh = self._get_mesh(group)
        if mesh is None or mesh.size() <= 1:
            return value

        return self._reduce_mesh(value, mesh, stage)

    def reduce_over_groups(
        self,
        value: torch.Tensor,
        groups: tuple[str, ...],
        op: ReduceOp,
    ) -> torch.Tensor:
        """Apply the same reduction operation over several groups."""
        stages = (ReduceStageConfig(group=group, op=op) for group in groups)
        return self.reduce_stages(value, stages)

    def _normalize_group(
        self,
        group: str | tuple[str, ...],
    ) -> str | tuple[str, ...]:
        if isinstance(group, str):
            return _GROUP_ALIASES.get(group, group)

        normalized = tuple(_GROUP_ALIASES.get(name, name) for name in group)

        if normalized == ("world",):
            return "world"

        if "world" in normalized:
            raise ValueError(
                "Monitoring reducer cannot mix 'world' with mesh dimensions"
            )

        return normalized

    def _validate_group(
        self,
        group_name: str | tuple[str, ...],
        stage: ReduceStageConfig,
    ) -> None:
        """Reject reductions that cross pipeline stages."""
        axes = (group_name,) if isinstance(group_name, str) else tuple(group_name)

        crosses_pp = "pp" in axes or (
            "world" in axes and getattr(self.parallel_dims, "pp_enabled", False)
        )

        if crosses_pp:
            raise ValueError(
                f"Monitoring reduce group {stage.group!r} spans the pipeline "
                "axis. Records are per model part, so the stages do not agree "
                "on how many values to reduce. Reduce over data axes (dp, cp) "
                "or ep instead; metrics reach the logging stage through "
                "gather_metrics_over_pp, not through a collective."
            )

    def _get_mesh(self, group: str | tuple[str, ...]) -> DeviceMesh | None:
        """Resolve a configured reduction group to a distributed mesh."""
        if not isinstance(group, str):
            # get_activated_mesh is required here: get_optional_mesh returns
            # None when any axis in a multi-axis group is disabled.
            return self.parallel_dims.get_activated_mesh(list(group))

        try:
            return self.parallel_dims.get_optional_mesh(group)
        except ValueError:
            if group == "fsdp":
                return self.parallel_dims.get_activated_mesh(["dp_shard", "cp"])
            raise

    def _reduce_mesh(
        self,
        value: torch.Tensor,
        mesh: DeviceMesh,
        stage: ReduceStageConfig,
    ) -> torch.Tensor:
        """Reduce over every axis represented by a mesh.

        A "mean" divides by each per-axis group size inside _reduce_tensor,
        so the combined divisor equals ``mesh.size()``.
        """
        axis_names = tuple(mesh.mesh_dim_names or ())

        if mesh.ndim > 1 and len(axis_names) != mesh.ndim:
            raise ValueError(
                f"Monitoring reduce group {stage.group!r} resolves to a "
                f"{mesh.ndim}-axis mesh whose axes are unnamed "
                f"({axis_names!r}), so they cannot be reduced one at a time. "
                "Use one axis per ReduceStageConfig, or a pre-flattened axis "
                "such as 'loss'."
            )

        groups = (
            [mesh.get_group()]
            if mesh.ndim == 1
            else [mesh.get_group(axis) for axis in axis_names]
        )

        result = value
        for group in groups:
            result = _reduce_tensor(result, group=group, op=stage.op)

        return result


FuseKey = tuple[
    tuple[ReduceStageConfig, ...], torch.device, torch.dtype, tuple[int, ...]
]


@gigatimer.timer("timers_callback.finalize_pending_reductions", 1)
def finalize_pending_reductions(
    records: list[MetricRecord],
    reducer: ParallelReducer,
) -> list[MetricRecord]:
    """Apply deferred reductions, fusing records by stage signature.

    Records sharing the same reduction-stage chain, device, dtype and SHAPE
    are stacked along a new leading dim into one tensor, so each chain
    requires only one collective per stage — this covers both the scalar
    records of the shorthand metrics and the equal-length vectors of
    ``[reduce..., stat...]`` metrics, whose per-layer inline collectives this
    fusing replaces. Element-wise reduction of the stacked tensor is identical
    to reducing each record separately, so values are unchanged. ``records``
    is updated in place and returned for convenience.

    Every rank of a reduce group must call this at the same point with the
    same record order, and with the same device/dtype per record: the fuse key
    partitions the records, so a rank that partitions them differently issues
    a different number of collectives and the group hangs. That holds for
    sources, whose records come from identical module/parameter iteration on
    every rank of a pp stage.
    """
    groups: dict[FuseKey, list[int]] = {}

    for index, record in enumerate(records):
        if not record.pending_stages:
            continue

        value = record.value
        if not isinstance(value, torch.Tensor):
            records[index] = _reduce_non_tensor_record(record, reducer)
            continue

        key = (record.pending_stages, value.device, value.dtype, tuple(value.shape))
        groups.setdefault(key, []).append(index)

    for indices in groups.values():
        _reduce_pending_group(records, indices, reducer)

    return records


def _reduce_pending_group(
    records: list[MetricRecord],
    indices: list[int],
    reducer: ParallelReducer,
) -> None:
    """Fuse pending reductions sharing a stage signature and a shape."""
    signature = records[indices[0]].pending_stages

    stacked = torch.stack([records[index].value.detach() for index in indices])
    stacked = reducer.reduce_stages(stacked, signature)

    for position, index in enumerate(indices):
        records[index] = replace(
            records[index], value=stacked[position], pending_stages=()
        )


def _reduce_non_tensor_record(
    record: MetricRecord,
    reducer: ParallelReducer,
) -> MetricRecord:
    """Reduce a pending record whose value is a bare Python number."""
    value = reducer.reduce_stages(
        torch.tensor(float(record.value)), record.pending_stages
    )
    return replace(record, value=value, pending_stages=())


@dataclass(slots=True)
class PPGatherCache:
    """One caller's memory of which rank sends which tags."""

    layout: tuple[int, ...] | None = None
    names: list[list[str]] | None = None
    warned_conflicts: set[str] | None = None


def gather_metrics_over_pp(
    metrics: dict[str, float],
    parallel_dims: ParallelDims,
    cache: PPGatherCache | None = None,
) -> dict[str, float]:
    """Deliver per-stage metrics to whichever PP rank owns the logger.

    With a ``cache`` the values travel as one tensor on steps whose tag sets
    are unchanged (see ``_gather_pp_metrics``); without one, every step takes
    the object path — correct, just slower.

    ``parallel_dims`` may be absent where a callback publishes outside a
    trainer (tests, tooling): there are no stages to gather across.
    """
    if parallel_dims is None:
        return metrics
    mesh = parallel_dims.get_optional_mesh("pp")

    if mesh is None or mesh.size() <= 1 or not _dist_ready():
        return metrics

    gathered = _gather_pp_metrics(metrics, mesh.get_group(), cache)
    merged, conflicting = _merge_pp_metrics(gathered)

    if conflicting:
        clean = {key.partition(STEP_SEP)[2] or key for key in conflicting}
        if cache.warned_conflicts is None:
            cache.warned_conflicts = set()
        fresh = sorted(clean - cache.warned_conflicts)
        if fresh:
            cache.warned_conflicts.update(fresh)
            logger.warning(
                "Monitoring: %d tag(s) differ between pipeline stages and were "
                "split per stage rather than overwritten: %s",
                len(fresh),
                ", ".join(fresh[:8]),
            )

    return merged


def _names_fingerprint(names: list[str]) -> tuple[int, int]:
    """(count, 64-bit digest) of an ordered tag list.

    blake2b rather than ``hash()``: the fingerprint is compared ACROSS ranks,
    and ``hash()`` is salted per process.
    """
    digest = hashlib.blake2b("\x00".join(names).encode(), digest_size=8).digest()
    return len(names), int.from_bytes(digest, "little", signed=True)


def _collective_device(group: dist.ProcessGroup) -> torch.device:
    if dist.get_backend(group) == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _gather_pp_metrics(
    metrics: dict[str, float],
    group: dist.ProcessGroup,
    cache: PPGatherCache | None,
) -> list[dict[str, float] | None]:
    """Gather metrics from all ranks participating in the PP group."""
    world = dist.get_world_size(group)

    if cache is None:
        # No memory to compare against: plain object path, no extra exchange.
        gathered: list[dict[str, float] | None] = [None] * world
        dist.all_gather_object(gathered, metrics, group=group)
        return gathered

    device = _collective_device(group)
    names = list(metrics)

    fingerprint = torch.tensor(
        _names_fingerprint(names), dtype=torch.int64, device=device
    )
    fingerprints = [torch.empty_like(fingerprint) for _ in range(world)]
    dist.all_gather(fingerprints, fingerprint, group=group)
    layout = tuple(torch.stack(fingerprints).cpu().reshape(-1).tolist())

    if layout == cache.layout and cache.names is not None:
        lengths = [len(rank_names) for rank_names in cache.names]
        width = max(lengths)
        values = torch.zeros(width, dtype=torch.float64, device=device)
        if names:
            values[: len(names)] = torch.tensor(
                [metrics[name] for name in names], dtype=torch.float64
            )
        parts = [torch.empty_like(values) for _ in range(world)]
        dist.all_gather(parts, values, group=group)
        return [
            dict(zip(cache.names[rank], parts[rank][: lengths[rank]].tolist()))
            for rank in range(world)
        ]

    gathered = [None] * world
    dist.all_gather_object(gathered, metrics, group=group)
    cache.layout = layout
    cache.names = [list(part or {}) for part in gathered]
    return gathered


def _merge_pp_metrics(
    gathered: list[dict[str, float] | None],
) -> tuple[dict[str, float], set[str]]:
    """Merge per-stage metrics and identify conflicting tags."""
    seen: dict[str, float] = {}
    conflicting: set[str] = set()

    for part in gathered:
        for tag, value in (part or {}).items():
            if tag not in seen:
                seen[tag] = value
            elif seen[tag] != value:
                conflicting.add(tag)

    merged: dict[str, float] = {}
    for pp_rank, part in enumerate(gathered):
        for tag, value in (part or {}).items():
            merged[f"{tag}/pp_rank{pp_rank}" if tag in conflicting else tag] = value

    return merged, conflicting
