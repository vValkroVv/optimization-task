import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims


def get_dataloader_shared_memory_namespace(
    role: str,
    suffix: str | None = None,
) -> str:
    """Return a stable per-job namespace for dataloader shared memory"""
    run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if run_id is None:
        master_addr = os.environ.get("MASTER_ADDR", "standalone")
        master_port = os.environ.get("MASTER_PORT", "0")
        run_id = f"{master_addr}:{master_port}"
    namespace = f"torchtitan_{run_id}_{role}"
    return namespace if suffix is None else f"{namespace}_{suffix}"


@dataclass(frozen=True, kw_only=True, slots=True)
class DataWorld:
    """
    Coordinates of this process in the data-loading world

    Data world spans the ``batch``, ``cp``, and ``tp`` axes of one pp
    coordinate; ``replication = cp * tp`` hands identical samples to replicas of one batch coordinate
    """

    rank: int
    world_size: int
    local_rank: int = 0
    local_world_size: int = 1
    replication: int = 1
    shared_memory_namespace: str | None = None
    process_group: dist.ProcessGroup | None = None

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")
        if self.local_world_size <= 0:
            raise ValueError(
                f"local_world_size must be positive, got {self.local_world_size}"
            )
        if not 0 <= self.local_rank < self.local_world_size:
            raise ValueError(
                f"local_rank must be in [0, {self.local_world_size}), got {self.local_rank}"
            )
        if self.world_size % self.local_world_size:
            raise ValueError(
                f"world_size={self.world_size} must be divisible by "
                f"local_world_size={self.local_world_size}"
            )
        if self.replication < 1 or self.world_size % self.replication:
            raise ValueError(
                f"replication={self.replication} must be positive and divide "
                f"world_size={self.world_size}"
            )
        if self.process_group is not None:
            if not dist.is_available() or not dist.is_initialized():
                raise ValueError("process_group requires initialized torch.distributed")
            group_world_size = dist.get_world_size(self.process_group)
            group_rank = dist.get_rank(self.process_group)
            if group_world_size != self.world_size:
                raise ValueError(
                    f"process_group world size {group_world_size} does not match "
                    f"world_size={self.world_size}"
                )
            if group_rank != self.rank:
                raise ValueError(
                    f"process_group rank {group_rank} does not match rank={self.rank}"
                )

    @classmethod
    def from_parallel_dims(
        cls, parallel_dims: "ParallelDims", *, role: str
    ) -> "DataWorld":
        """Data world of this run: ``data`` mesh with cp*tp replication"""
        return cls.from_mesh(
            parallel_dims.get_optional_mesh("data"),
            role=role,
            replication=parallel_dims.cp * parallel_dims.tp,
        )

    @classmethod
    def from_mesh(
        cls,
        data_mesh: DeviceMesh | None,
        *,
        role: str,
        replication: int = 1,
    ) -> "DataWorld":
        """Coordinates from flattened data mesh; ``None`` means a single-rank
        world. ``replication`` counts consecutive ranks seeing identical samples"""
        if data_mesh is None:
            if replication != 1:
                raise ValueError(
                    f"replication={replication} requires a data mesh, got None"
                )
            return cls(
                rank=0,
                world_size=1,
                shared_memory_namespace=get_dataloader_shared_memory_namespace(role),
            )
        if data_mesh.ndim != 1:
            raise ValueError(
                f"data mesh must be one-dimensional, got shape {data_mesh.shape}"
            )
        mesh_ranks = [int(r) for r in data_mesh.mesh.detach().cpu().flatten().tolist()]
        first = mesh_ranks[0]
        if mesh_ranks != list(range(first, first + len(mesh_ranks))):
            raise ValueError(
                f"data mesh ranks must be contiguous ascending global ranks "
                f"(pp must be the outermost mesh axis), got {mesh_ranks}"
            )

        world_size = data_mesh.size()
        rank = data_mesh.get_local_rank()
        ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if ranks_per_node <= 0:
            raise ValueError(f"LOCAL_WORLD_SIZE must be positive, got {ranks_per_node}")
        # shm works within one machine and streaming elects one leader per
        # local_world_size block: the world must not straddle a node boundary
        if world_size % ranks_per_node == 0 and first % ranks_per_node == 0:
            local_world_size = ranks_per_node
        elif (
            world_size < ranks_per_node
            and first % ranks_per_node + world_size <= ranks_per_node
        ):
            local_world_size = world_size
        else:
            raise ValueError(
                f"data world ranks {first}..{first + world_size - 1} do not align "
                f"with the node size {ranks_per_node}: pp groups must span whole "
                f"nodes or fit inside one"
            )
        # first rank keeps namespaces of same-role worlds on one node distinct
        return cls(
            rank=rank,
            world_size=world_size,
            local_rank=rank % local_world_size,
            local_world_size=local_world_size,
            replication=replication,
            shared_memory_namespace=get_dataloader_shared_memory_namespace(
                role, f"r{first}"
            ),
            process_group=data_mesh.get_group(),
        )
