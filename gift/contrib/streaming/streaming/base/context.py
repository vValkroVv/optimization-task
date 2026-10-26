"""Runtime context overrides for StreamingDataset integration layers."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@dataclass(frozen=True)
class StreamingWorldContext:
    """Distributed world values to use instead of process environment values."""

    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    init_dist: bool
    process_group: Optional['ProcessGroup']
    partition_num_nodes: Optional[int] = None


_STREAMING_WORLD_CONTEXT = ContextVar(
    'streaming_world_context',
    default=None,
)  # type: ContextVar[Optional[StreamingWorldContext]]

_SHARED_MEMORY_NAMESPACE = ContextVar(
    'streaming_shared_memory_namespace',
    default=None,
)  # type: ContextVar[Optional[str]]


def get_streaming_world_context() -> Optional[StreamingWorldContext]:
    """Get the active world override, if any."""
    return _STREAMING_WORLD_CONTEXT.get()


def get_shared_memory_namespace() -> Optional[str]:
    """Get the active shared-memory namespace, if any."""
    return _SHARED_MEMORY_NAMESPACE.get()


@contextmanager
def use_streaming_world(
    rank: int,
    world_size: int,
    local_rank: int = 0,
    local_world_size: int = 1,
    init_dist: bool = True,
    shared_memory_namespace: Optional[str] = None,
    process_group: Optional['ProcessGroup'] = None,
    partition_num_nodes: Optional[int] = None,
) -> Iterator[None]:
    """Temporarily override the world seen by StreamingDataset.

    This is intended for frameworks whose process group is larger than the data-parallel
    group that should partition the dataset. The shared-memory namespace only affects
    StreamingDataset's local shared-memory identity; it does not change dataset paths.
    ``partition_num_nodes`` overrides only the logical node grid that partitioning,
    shuffling and the resume cursor use; physical coordination still follows ``local_rank``/``local_world_size``
    """
    if world_size <= 0:
        raise ValueError(f'world_size must be positive, got {world_size}')
    if not 0 <= rank < world_size:
        raise ValueError(f'rank must be in [0, {world_size}), got {rank}')
    if local_world_size <= 0:
        raise ValueError(f'local_world_size must be positive, got {local_world_size}')
    if not 0 <= local_rank < local_world_size:
        raise ValueError(f'local_rank must be in [0, {local_world_size}), got {local_rank}')
    if world_size % local_world_size:
        raise ValueError(f'{world_size=} must be divisible by {local_world_size=}')
    if partition_num_nodes is not None:
        if partition_num_nodes <= 0:
            raise ValueError(f'partition_num_nodes must be positive, got {partition_num_nodes}')
        if world_size % partition_num_nodes:
            raise ValueError(f'{world_size=} must be divisible by {partition_num_nodes=}')

    world_token = _STREAMING_WORLD_CONTEXT.set(
        StreamingWorldContext(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            init_dist=init_dist,
            process_group=process_group,
            partition_num_nodes=partition_num_nodes,
        ))
    namespace_token = _SHARED_MEMORY_NAMESPACE.set(shared_memory_namespace)
    try:
        yield
    finally:
        _SHARED_MEMORY_NAMESPACE.reset(namespace_token)
        _STREAMING_WORLD_CONTEXT.reset(world_token)
