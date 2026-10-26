import traceback
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, NoReturn

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement, Replicate


@dataclass
class DataParallelMeshInfo:
    mesh: DeviceMesh
    shard_mesh_dim: int | None = None
    replicate_mesh_dim: int | None = None

    def __post_init__(self) -> None:
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise AssertionError(
                "At least one of shard_mesh_dim and replicate_mesh_dim must not be None"
            )

    def dp_placements(self, shard_placement: Placement) -> tuple[Placement, ...]:
        raise NotImplementedError


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size = self.mesh.size(self.shard_mesh_dim)
        self.shard_process_group = self.mesh.get_group(self.shard_mesh_dim)
        self.shard_mesh_rank = self.shard_process_group.rank()

    def dp_placements(self, shard_placement: Placement) -> tuple[Placement, ...]:
        return (shard_placement,)


@dataclass
class HSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self) -> None:
        super().__post_init__()

        # Shard-axis
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size = self.mesh.size(self.shard_mesh_dim)
        self.shard_process_group = self.mesh.get_group(self.shard_mesh_dim)
        self.shard_mesh_rank = self.shard_process_group.rank()

        # Replicate-axis
        if self.mesh.ndim != 2 or self.replicate_mesh_dim is None:
            raise AssertionError(
                "HSDP expects non-None replicate_mesh_dim and mesh.ndim == 2 for dp submesh"
            )
        if not (self.replicate_mesh_dim == 0 and self.shard_mesh_dim == 1):
            raise AssertionError(
                "HSDP expects replicate_mesh_dim=0 and shard_mesh_dim=1 for dp submesh"
            )
        self.replicate_mesh_size = self.mesh.size(self.replicate_mesh_dim)
        self.replicate_process_group = self.mesh.get_group(self.replicate_mesh_dim)
        self.replicate_mesh_rank = self.replicate_process_group.rank()

    def dp_placements(self, shard_placement: Placement) -> tuple[Placement, ...]:
        """
        Expects ("dp_replicate", "fsdp") placements for the 2D mesh
        """
        return (Replicate(), shard_placement)


class TrainingState(Enum):
    # Transition to forward starting pre-forward until post-forward
    FORWARD = auto()
    # Transition to pre-backward when unsharding in backward
    PRE_BACKWARD = auto()
    # Transition to post-backward when resharding and reducing gradients
    POST_BACKWARD = auto()
    # Idle before/after forward or before pre-backward/after post-backward
    IDLE = auto()


def _raise_assert_with_print(*args: Any, **kwargs: Any) -> NoReturn:
    print(f"[Rank {dist.get_rank()}] ", end="")
    print(*args, **kwargs)
    traceback.print_stack()
    raise AssertionError(*args, **kwargs)


def _is_composable_with_fsdp(module: nn.Module) -> bool:
    registry = _get_registry(module)
    if registry is None:
        return True
    # Registry keys by function name
    return "replicate" not in registry


def _cast_fp_tensor(dtype: torch.dtype, x: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(x, torch.Tensor)
        or not torch.is_floating_point(x)
        or x.dtype == dtype
    ):
        return x
    return x.to(dtype)


def is_bw() -> bool:
    return torch._C._current_graph_task_id() != -1
