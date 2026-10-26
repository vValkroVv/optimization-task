# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding types for config-based parallelization.

``ShardingConfig`` is set on ``Module.Config`` by ``set_sharding_config()``
and read by ``Module.parallelize(parallel_dims)``.  All placements use
``NamedPlacement`` (dict keyed by ``MeshAxisName``) so they are
self-documenting and support multi-dimensional meshes.
"""

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Placement, Replicate, Shard

from torchtitan.config.schema.protocols import sharding as spec
from torchtitan.config.schema.protocols.sharding import (
    LocalMapConfig,
    NamedPlacement,
    ShardingConfig,
)
from torchtitan.config.schema.protocols.types import MeshAxisName


__all__ = [
    "LocalMapConfig",
    "NamedPlacement",
    "ShardingConfig",
    "resolve_placement",
    "resolve_placements",
]


def resolve_placement(p: spec.Placement) -> Placement:
    """The ``torch.distributed.tensor`` placement a spec stands for."""
    if isinstance(p, spec.Shard):
        return Shard(p.dim)
    if isinstance(p, spec.Replicate):
        return Replicate()
    if isinstance(p, spec.Partial):
        return Partial(p.reduce_op)
    raise TypeError(f"not a placement spec: {p!r}")


# Shard order: we implicitly assume the trivial outer -> inner order matching
# the mesh axis order. The only non-trivial case is FSDP + TP both sharding on
# tensor dim 0, but it doesn't need to be annotated today.
# TODO: integrate with global spmd types (e.g., ``TP: V`` + ``PartitionSpec``
# carrying explicit shard-order info) once that lands.


def resolve_placements(
    named: NamedPlacement,
    mesh: DeviceMesh,
) -> tuple[Placement, ...]:
    """Resolve NamedPlacement against a mesh in axis order.

    Every sharding_config must explicitly declare a placement for every mesh axis
    it will be applied against. Missing declarations raise ``ValueError``;
    extra declarations (axes not in the mesh) are ignored.

    ``Shard(d)`` on a size-1 mesh axis is normalized to ``Replicate()`` --
    the two are operationally identical on a 1-rank axis, but DTensor's op
    rules (placement-equality, view/reshape strict mode, ...) treat them
    as distinct and reject ``Shard`` in places where ``Replicate`` would
    work.
    """
    # TODO(fegin): remove the ``Shard(d)`` on a size-1 mesh to ``Replicate()``
    # conversion once FlexShard replaces ``fully_shard``.
    assert mesh.mesh_dim_names is not None, "DeviceMesh must have named axes"
    result = []
    for i, axis_name in enumerate(mesh.mesh_dim_names):
        key = MeshAxisName(axis_name)
        if key not in named:
            raise ValueError(
                f"ShardingConfig does not declare a placement for mesh axis "
                f"{axis_name!r}. Declared: "
                f"{sorted(k.value for k in named)}; "
                f"required: {list(mesh.mesh_dim_names)}."
            )
        p = resolve_placement(named[key])
        if isinstance(p, Shard) and mesh.size(i) == 1:
            p = Replicate()
        result.append(p)
    return tuple(result)
