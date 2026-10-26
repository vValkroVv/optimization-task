"""Torch-free sharding specs for config-based parallelization.

``torchtitan.protocols.sharding.resolve_placements`` turns these placement
specs into ``torch.distributed.tensor`` placements at parallelize time.
"""

from dataclasses import dataclass, field

from torchtitan.config.configurable import (
    component,
    component_family,
    to_plain,
)
from torchtitan.config.schema.protocols.types import MeshAxisName

__all__ = [
    "LocalMapConfig",
    "NamedPlacement",
    "Partial",
    "Placement",
    "Replicate",
    "Shard",
    "ShardingConfig",
]


@component_family("placement")
@dataclass(frozen=True, slots=True)
class Placement:
    """Placement spec for one mesh axis, picked in YAML by ``type``.

    No default type: ``{}`` would be ambiguous between ``Replicate()`` and
    ``Partial()``. Plain dataclasses, not ``@config``: ``kw_only`` would break
    the positional ``Shard(0)`` in model sharding tables. ``@component`` goes
    above ``@dataclass`` because ``slots=True`` returns a new class.
    """

    def to_dict(self) -> dict:
        """Serialize with the type name; ``Replicate()`` would dump as ``{}``."""
        return to_plain(self)


@component("shard")
@dataclass(frozen=True, slots=True)
class Shard(Placement):
    """Sharded along tensor dimension ``dim``."""

    dim: int


@component("replicate")
@dataclass(frozen=True, slots=True)
class Replicate(Placement):
    """Replicated across the mesh axis."""


@component("partial")
@dataclass(frozen=True, slots=True)
class Partial(Placement):
    """Pending reduction across the mesh axis."""

    reduce_op: str = "sum"


# Placement per mesh axis, keyed by MeshAxisName.
# Every sharding_config must declare a placement for every mesh axis it
# will be applied against; ``resolve_placements`` errors otherwise.
NamedPlacement = dict[MeshAxisName, Placement]


@dataclass(kw_only=True, slots=True)
class LocalMapConfig:
    """Spec for modules computing on local tensors.

    Wraps forward with ``local_map()``: DTensor -> local before forward,
    local -> DTensor after forward.

    Input placements come from ``ShardingConfig.in_dst_shardings``
    (already aligned by ``_redistribute_inputs``); output placements from
    ``ShardingConfig.out_src_shardings``. ``LocalMapConfig`` only carries
    ``in_grad_placements`` since there's no equivalent slot on
    ``ShardingConfig`` today.

    Attributes:
        in_grad_placements: Per-input-gradient NamedPlacements (positional,
            ordered by ``forward`` args).
    """

    in_grad_placements: tuple[NamedPlacement, ...]


@dataclass(kw_only=True, slots=True)
class ShardingConfig:
    """Declarative sharding for a Module's states and activations.

    All placements use ``NamedPlacement`` (``dict[MeshAxisName, Placement]``)
    keyed by mesh axis names.  At ``parallelize()`` time, NamedPlacements
    are resolved to ``tuple[Placement, ...]`` in mesh axis order.

    Completely dtype-agnostic at this moment — quantization (Float8/MXFP8) is
    orthogonal.

    Redistribution is expressed as a (source, destination) pair: src declares
    what the tensor's placement is entering the boundary, dst declares the
    desired placement after redistribution. For DTensor, the src is usually
    implicit in the tensor's ``placements``; declaring it explicitly keeps
    the contract uniform with future erased-type systems that require both
    sides of every redistribute.

    Attributes:
        state_shardings: Parameter/buffer placements for ``distribute_tensor``.
            Outer dict keys are param names.
            e.g. ``{"weight": {TP: Shard(0)}}`` for colwise.
        in_src_shardings: Source placements of inputs, keyed by ``forward()``
            arg name. Used to annotate plain tensors as DTensors via
            ``DTensor.from_local`` when inputs arrive plain (e.g. from
            dataloader or FSDP-only path). Also declares the src side of
            the input redistribute pair.
            e.g. ``{"x": {TP: Shard(1)}}``.
        in_dst_shardings: Desired input placements after redistribution,
            keyed by ``forward()`` arg name.
            e.g. ``{"x": {TP: Replicate()}}`` for all-gather.
            ``None`` means no input redistribution.
        out_src_shardings: Source placement of the forward's output as a
            DTensor. When ``local_map`` is set this also tells ``local_map``
            what to wrap the local output back to. Accepts a single
            ``NamedPlacement`` (single-output case) or a tuple (multi-
            output case, e.g. attention with ``return_lse=True``). ``None``
            means "infer from the output" (it's already a DTensor at the
            right placement, or there's no local_map to drive).
            e.g. ``{TP: Partial()}`` for the MoE wrapper.
        out_dst_shardings: Desired output placement after redistribution.
            e.g. ``{TP: Shard(1)}`` for reduce-scatter to sequence-parallel.
            ``None`` means no output redistribution.
        local_map: If set, wraps forward with ``local_map()``. Input and
            output placements come from ``in_dst_shardings`` and
            ``out_src_shardings``; ``LocalMapConfig`` only carries
            ``in_grad_placements``.
    """

    state_shardings: dict[str, NamedPlacement] = field(default_factory=dict)
    in_src_shardings: dict[str, NamedPlacement] | None = None
    in_dst_shardings: dict[str, NamedPlacement] | None = None
    out_src_shardings: NamedPlacement | tuple[NamedPlacement, ...] | None = None
    out_dst_shardings: NamedPlacement | None = None
    local_map: LocalMapConfig | None = None
