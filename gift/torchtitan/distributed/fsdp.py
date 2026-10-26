# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    FSDPModule,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.tensor import Shard

from torchtitan.protocols.blocks import Segment
from torchtitan.protocols.module import Module
from torchtitan.distributed.fsdp_capabilities import fsdp_symmetric_memory_supported
from torchtitan.tools.logging import logger


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """
    Disable FSDP's automatic gradient division for all FSDP modules.

    Set gradient_divide_factor=1.0 to disable FSDP's automatic gradient division.
    We handle gradient scaling ourselves in the training loop with global token count.

    Note: This also works for ReplicateModule since it inherits from FSDPModule.

    Args:
        model: The model containing FSDP-wrapped or Replicate-wrapped modules
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


def enable_fsdp_symm_mem(model: nn.Module) -> None:
    """
    Enable symmetric-memory communication optimizations for all FSDP modules.
    """
    if not fsdp_symmetric_memory_supported():
        raise ValueError(
            "parallelism.enable_fsdp_symm_mem is only supported on NVIDIA "
            "GPUs with compute capability 9.0 or newer."
        )
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_force_sum_reduction_for_comms(True)
            module.set_symm_mem_for_comm()


def get_fsdp_reshard_after_forward_policy(
    reshard_after_forward_policy: str, pp_enabled: bool
) -> bool:
    """Resolve fsdp_reshard_after_forward policy string to a boolean.

    Args:
        reshard_after_forward_policy: One of "always", "never", or "default".
        pp_enabled: Whether pipeline parallelism is enabled.

    Returns:
        Boolean indicating whether to reshard after forward.
    """
    match reshard_after_forward_policy:
        case "always":
            return True
        case "never":
            return False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            return not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )


@dataclass
class _ShardUnit:
    """One ``fully_shard`` group, in forward order. Policy-side; models never see it."""

    modules: list[nn.Module]
    fqn: str
    is_block: bool
    reshard_after_forward: bool
    # The output run: where backward starts, which is not always last in forward
    # order (see the prefetch chain in ``apply_fsdp``).
    is_output: bool = False

    @property
    def forward_handle(self) -> nn.Module:
        """Module whose forward starts this group."""
        return self.modules[0]

    @property
    def backward_handle(self) -> nn.Module:
        """Module whose backward runs first."""
        return self.modules[-1]


def _output_run_bounds(segments: list[Segment]) -> tuple[int, int] | None:
    """Half-open segment range of the model's output run, or None.

    The output run is the last maximal run of consecutive non-block segments that
    has at least one block before it: ``norm`` + ``lm_head`` for a decoder,
    ``final_layer`` for flux. A leading stem has no block before it and so is not
    it, which is what keeps a stage of the shape ``[embedding, blocks...]`` from
    being mistaken for an output.

    This run is not resharded after forward (backward re-gathers it at once)
    and its segments stay separate groups: under a chunked loss the head and
    the norm are reduced by different backwards, and a shared group would
    break HSDP's deferred all-reduce.
    """
    seen_block = False
    best: tuple[int, int] | None = None
    index = 0
    while index < len(segments):
        if segments[index].is_block:
            seen_block = True
            index += 1
            continue
        stop = index
        while stop < len(segments) and not segments[stop].is_block:
            stop += 1
        if seen_block:
            best = (index, stop)
        index = stop
    return best


def _build_shard_units(
    segments: list[Segment],
    *,
    reshard_after_forward: bool,
    output_reshard_after_forward: bool,
    coalesce_groups: frozenset[str] = frozenset(),
) -> list[_ShardUnit]:
    """Turn a model's structural segments into FSDP groups. Pure policy.

    - one group per repeated block;
    - each output-run segment (see ``_output_run_bounds``) as its own group,
      not resharded after forward; the head segment is marked ``is_output``;
    - every other non-block segment as its own group;
    - non-block groups sharing a parameter merge, which is how tied embeddings
      land in one group with no flag declared anywhere. Blocks are excluded:
      cross-layer sharing must not silently collapse the per-block grouping;
    - a group in ``coalesce_groups`` becomes one FSDP group instead of one per
      block, for a tower whose per-block compute cannot hide an all-gather.
    """
    block_groups = {segment.group for segment in segments if segment.is_block}
    if unknown := sorted(coalesce_groups - block_groups):
        # Not an error: under PP another stage owns those blocks. A stale name is
        # the other cause, and it falls back to one group per block in silence.
        logger.warning(
            f"coalesce_block_groups {unknown} name no block group here "
            f"(have {sorted(block_groups)}); those blocks get one group each."
        )

    output = _output_run_bounds(segments)
    units: list[_ShardUnit] = []
    coalesced: dict[str, _ShardUnit] = {}

    for index, segment in enumerate(segments):
        if output is not None and output[0] <= index < output[1]:
            # one group per segment, not per run: see _output_run_bounds
            units.append(
                _ShardUnit(
                    modules=[segment.module],
                    fqn=segment.fqn,
                    is_block=False,
                    reshard_after_forward=output_reshard_after_forward,
                    is_output=index == output[1] - 1,
                )
            )
            continue

        if not segment.is_block:
            units.append(
                _ShardUnit(
                    modules=[segment.module],
                    fqn=segment.fqn,
                    is_block=False,
                    reshard_after_forward=reshard_after_forward,
                )
            )
            continue

        if segment.group in coalesce_groups:
            unit = coalesced.get(segment.group)
            if unit is None:
                unit = _ShardUnit(
                    modules=[],
                    fqn=segment.group,
                    is_block=True,
                    reshard_after_forward=reshard_after_forward,
                )
                coalesced[segment.group] = unit
                units.append(unit)
            unit.modules.append(segment.module)
            continue

        units.append(
            _ShardUnit(
                modules=[segment.module],
                fqn=segment.fqn,
                is_block=True,
                reshard_after_forward=reshard_after_forward,
            )
        )

    return _merge_units_sharing_params(units)


def _merge_units_sharing_params(units: list[_ShardUnit]) -> list[_ShardUnit]:
    """Merge non-block groups that share a parameter, keeping the earliest slot.

    Shared storage cannot be sharded twice. Detected by parameter identity, not by
    a model-declared flag, so the fact and its consequence cannot drift apart.

    Union-find rather than one pass of "point at the first owner": with three
    groups chained through two different shared parameters, a single pass can send
    a group to a target that is itself merged away, and its modules then belong to
    no surviving group — silently unsharded.
    """
    parent = list(range(len(units)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        first, second = sorted((find(left), find(right)))
        parent[second] = first

    owner: dict[int, int] = {}
    for index, unit in enumerate(units):
        if unit.is_block:
            continue
        for module in unit.modules:
            for param in module.parameters():
                previous = owner.setdefault(id(param), index)
                if previous != index:
                    union(previous, index)

    if all(parent[i] == i for i in range(len(units))):
        return units

    merged: list[_ShardUnit] = []
    hosts: dict[int, _ShardUnit] = {}
    for index, unit in enumerate(units):
        root = find(index)
        if root == index:
            hosts[root] = unit
            merged.append(unit)
            continue
        host = hosts[root]
        host.modules.extend(unit.modules)
        host.fqn = f"{host.fqn}+{unit.fqn}"
        # A merge that swallows the output run makes the host the output group:
        # backward still starts there, and it must keep the no-reshard policy.
        host.is_output = host.is_output or unit.is_output
        host.reshard_after_forward = (
            host.reshard_after_forward and unit.reshard_after_forward
        )
    return merged


def _grouped_param_placement(
    param: nn.Parameter, axis_len: int, *, num_shards: int
) -> Shard:
    if num_shards <= axis_len:
        return Shard(0)
    return Shard(2) if param.ndim == 4 else Shard(1)


def _stamp_group_kinds(host: nn.Module, kind: str) -> None:
    """Label every FSDPParamGroup created on ``host`` with its semantic kind
    ("dense", "moe" or "mixed"), so selective grad-reduce policies can pick
    groups without guessing from meshes. The kind comes from
    ``grouped_param_axes()`` (what _fully_shard_unit routed where), the only
    place expert params are actually distinguished."""
    # pyrefly: ignore[missing-attribute]  # fully_shard.state exists at runtime
    for group in fully_shard.state(host)._fsdp_param_groups:
        group.grad_reduce_kind = kind


def _fully_shard_unit(
    unit: _ShardUnit,
    *,
    fsdp_config: dict[str, Any],
    grouped_axes: dict[int, int],
    ep_degree: int,
    edp_mesh: DeviceMesh | None,
    dp_mesh: DeviceMesh,
) -> None:
    """Wrap one unit into a single FSDP group, honouring group-axis parameters.

    A parameter in ``grouped_axes`` has a group axis on dim 0 (routed experts):
    sharding it below the axis length would pad, and under EP those parameters
    live on ``edp_mesh``. Both decided here, from what ``grouped_param_axes``
    reported.
    """
    own = {
        id(p): (p, grouped_axes[id(p)])
        for m in unit.modules
        for p in m.parameters()
        if id(p) in grouped_axes
    }
    if not own:
        fully_shard(
            unit.modules,
            **fsdp_config,
            reshard_after_forward=unit.reshard_after_forward,
        )
        _stamp_group_kinds(unit.modules[0], "dense")
        return

    if ep_degree > 1:
        assert edp_mesh is not None
        assert (
            edp_mesh.mesh_dim_names is not None and "efsdp" in edp_mesh.mesh_dim_names
        ), (
            f"edp_mesh must have an 'efsdp' axis when ep_degree > 1; got {edp_mesh.mesh_dim_names}"
        )
        efsdp_ep_size = edp_mesh["efsdp"].size() * ep_degree
    else:
        efsdp_ep_size = fsdp_config["mesh"].size()

    placements = {
        pid: _grouped_param_placement(param, axis_len, num_shards=efsdp_ep_size)
        for pid, (param, axis_len) in own.items()
    }

    if ep_degree == 1:
        if all(isinstance(p, Shard) and p.dim == 0 for p in placements.values()):
            # Nothing to override: Shard(0) is already the default.
            fully_shard(
                unit.modules,
                **fsdp_config,
                reshard_after_forward=unit.reshard_after_forward,
            )
            # one mesh -> one group holding both expert and dense params
            _stamp_group_kinds(unit.modules[0], "mixed")
            return

        def _grouped_shard_placement_fn(
            param: nn.Parameter,
            _placements: dict[int, Shard] = placements,
        ) -> Shard | None:
            return _placements.get(id(param))

        fully_shard(
            unit.modules,
            **fsdp_config,
            reshard_after_forward=unit.reshard_after_forward,
            shard_placement_fn=_grouped_shard_placement_fn,
        )
        _stamp_group_kinds(unit.modules[0], "mixed")
        return

    # EP > 1: group-axis params go to edp_mesh, everything else to dp_mesh.
    from torch.distributed.fsdp._fully_shard._fsdp_common import (
        FSDPMeshInfo,
        HSDPMeshInfo,
        ShardPlacementResult,
    )

    assert edp_mesh is not None

    def _get_fsdp_mesh_info(mesh: DeviceMesh) -> FSDPMeshInfo:
        if mesh.ndim == 1:
            return FSDPMeshInfo(mesh=mesh, shard_mesh_dim=0)
        if mesh.ndim == 2:
            return HSDPMeshInfo(mesh=mesh, replicate_mesh_dim=0, shard_mesh_dim=1)
        raise ValueError(f"Expected 1D or 2D FSDP mesh, got {mesh.ndim}D mesh.")

    edp_mesh_info = _get_fsdp_mesh_info(edp_mesh)
    dp_mesh_info = _get_fsdp_mesh_info(dp_mesh)

    def _shard_placement_fn(
        param: nn.Parameter,
        _placements: dict[int, Shard] = placements,
        _edp_mesh_info: FSDPMeshInfo = edp_mesh_info,
        _dp_mesh_info: FSDPMeshInfo = dp_mesh_info,
    ) -> "ShardPlacementResult":
        grouped = _placements.get(id(param))
        if grouped is not None:
            return ShardPlacementResult(placement=grouped, mesh_info=_edp_mesh_info)
        return ShardPlacementResult(placement=Shard(0), mesh_info=_dp_mesh_info)

    fully_shard(
        unit.modules,
        **fsdp_config,
        reshard_after_forward=unit.reshard_after_forward,
        shard_placement_fn=_shard_placement_fn,
    )
    # The fork partitions params into one group per process group; reading
    # mesh_info back tells which group got the grouped (expert) params.
    # pyrefly: ignore[missing-attribute]  # fully_shard.state exists at runtime
    for group in fully_shard.state(unit.modules[0])._fsdp_param_groups:
        group.grad_reduce_kind = "moe" if group.mesh_info is edp_mesh_info else "dense"


def apply_fsdp(
    model: Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    enable_symm_mem: bool = False,
    dp_mesh_dims: DataParallelMeshDims | None = None,
    coalesce_block_groups: Sequence[str] = (),
):
    """
    Apply data parallelism (via FSDP2) to the model.

    The model is read only through ``named_segments()`` and
    ``grouped_param_axes()``; everything derived from them — what shares an
    all-gather, what reshards, which mesh a group-axis parameter lands on, how
    prefetch is chained — is policy and lives here. Hence no model name, FQN or
    attribute appears below, and any model with blocks is covered as-is.

    Args:
        model (Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.
        dp_mesh_dims: Under full_dtensor, ``fully_shard`` flattens ``dp_shard``
            and ``cp`` into one FSDP shard axis, so it must know which axes of the
            SPMD mesh are data-parallel. Passed explicitly rather than inferred
            from axis names: that naming contract is not strong enough, and an
            explicit declaration avoids silent miscategorization when new axes
            appear.
        coalesce_block_groups: Block-group FQNs (``Segment.group``, e.g.
            ``"vision_encoder.layers"``) whose blocks share one FSDP group
            instead of one each. For a tower whose per-block compute is too small
            to hide a per-block all-gather.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    segments = list(model.named_segments())
    known_groups = {s.group for s in segments if s.is_block}
    if unknown := set(coalesce_block_groups) - known_groups:
        raise ValueError(
            f"coalesce_block_groups names {sorted(unknown)}, which is not a block "
            f"group of {type(model).__name__}; it has {sorted(known_groups)}. A "
            "silently ignored name would just leave the blocks ungrouped."
        )

    units = _build_shard_units(
        segments,
        reshard_after_forward=get_fsdp_reshard_after_forward_policy(
            reshard_after_forward_policy, pp_enabled
        ),
        output_reshard_after_forward=reshard_after_forward_policy == "always",
        coalesce_groups=frozenset(coalesce_block_groups),
    )
    grouped_axes = {id(param): axis for param, axis in model.grouped_param_axes()}

    for unit in units:
        _fully_shard_unit(
            unit,
            fsdp_config=fsdp_config,
            grouped_axes=grouped_axes,
            ep_degree=ep_degree,
            edp_mesh=edp_mesh,
            dp_mesh=dp_mesh,
        )

    fully_shard(model, **fsdp_config)
    _stamp_group_kinds(model, "dense")  # root group: leftover non-block params

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)

    # NOTE: set up explicit prefetching when EP is enabled, as D2H syncs
    # in EP could interfere with implicit prefetching in FSDP
    if ep_degree == 1:
        return

    # Units are in forward order, so the forward chain is pairwise.
    for current, following in zip(units, units[1:]):
        current.forward_handle.set_modules_to_forward_prefetch(following.modules)

    # Backward starts at the output group, which is not always last in forward
    # order: with a chunked loss the output head is applied by the loss, after the
    # trailing blocks (MTP heads). Putting it first keeps the rest of the chain
    # intact — otherwise the link out of the last trailing block lands on a module
    # whose backward never runs, and the trunk goes unprefetched.
    output = next((u for u in units if u.is_output), None)
    backward_order = [u for u in reversed(units) if u is not output]
    if output is not None:
        backward_order.insert(0, output)
    for current, preceding in zip(backward_order, backward_order[1:]):
        current.backward_handle.set_modules_to_backward_prefetch(preceding.modules)
