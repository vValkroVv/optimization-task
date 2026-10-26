"""What the trainer and its callbacks may ask of a sharding backend.

The trainer builds one of these from ``parallelism.fsdp_backend`` and hands it
to whoever needs it, so nothing outside ``torchtitan/distributed`` branches on
the name; the implementations live in ``torchtitan/distributed/backends``.
"""

from collections.abc import Callable, Collection
from contextlib import AbstractContextManager
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.config.schema.protocols.model import BaseModelConfig

from torchtitan.protocols.checkpoint_adapter import CheckpointAdapter

__all__ = ["EMA_MODEL", "EMAStore", "ShardStats", "ShardingBackend"]


EMA_MODEL = "ema_model"


class ShardStats(Protocol):
    """How the metrics pipeline computes over flat ragged shards.

    A ragged shard is an element range of the flattened parameter. Most
    statistics use local accumulators; others need the full logical tensor.
    """

    @property
    def tensor_types(self) -> tuple[type[torch.Tensor], ...]:
        """The ragged tensor types the metrics pipeline should recognise."""
        ...

    @property
    def placement_types(self) -> tuple[type, ...]:
        """Placement types marking the mesh axes partial stats reduce over."""
        ...

    def sum_flat_segments(
        self,
        value: torch.Tensor,
        num_segments: int,
        *,
        transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        sum_dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Per-segment sums as a DTensor with lazy Partials."""
        ...

    def max_flat_segments(self, value: torch.Tensor, num_segments: int) -> torch.Tensor:
        """Per-segment max, same contract as ``sum_flat_segments``."""
        ...

    def min_flat_segments(self, value: torch.Tensor, num_segments: int) -> torch.Tensor:
        """Per-segment min, same contract as ``sum_flat_segments``."""
        ...

    def abs_max_flat_segments(
        self, value: torch.Tensor, num_segments: int
    ) -> torch.Tensor:
        """Per-segment maximum of absolute values."""
        ...

    def moment_flat_segments(
        self,
        value: torch.Tensor,
        num_segments: int,
        mean: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """Sum centered powers per segment using global segment indices."""
        ...

    def to_local(self, value: torch.Tensor) -> torch.Tensor:
        """This rank's flat shard as a plain tensor."""
        ...

    def local_squared_norms(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        """Local fp32 squared contributions to norms along one logical axis."""
        ...

    def to_full_tensor(self, value: torch.Tensor) -> torch.Tensor:
        """Gather a separate full logical tensor without changing shard storage."""
        ...


class EMAStore(Protocol):
    """Where the averaged weights live, and how they move.

    The ``EMA`` callback decides *when* to average; a store decides *where*.
    """

    holds_average: bool
    """Whether the shadows are an average rather than the weights they were
    seeded with"""

    use_ema: bool
    """Whether the run publishes the average; ``EMA.Config.use_ema``."""

    @property
    def checkpoint_carries_ema(self) -> bool:
        """Whether an ordinary checkpoint already holds the averaged weights.

        False when the shadows only reach a checkpoint because this store puts
        them there, which is what makes a separate EMA-only export meaningful.
        """
        ...

    def seed(self, only: Collection[str] | None = None) -> None:
        """Copy the live weights into the shadows, or into just ``only``."""
        ...

    def update(self, alpha: float) -> None:
        """``shadow = shadow + alpha * (live - shadow)``, in place."""
        ...

    def swap(self) -> None:
        """Exchange live weights and shadows. Not idempotent; must be paired."""
        ...

    def after_load(self) -> Collection[str]:
        """Settle the shadows after a checkpoint load.

        Returns the parameters whose shadows the checkpoint did not carry and
        which were therefore seeded from the live weights.
        """
        ...

    def stateful_objects(self) -> dict[str, Stateful]:
        """What this store contributes to the checkpoint, keyed as it is saved."""
        ...


class ShardingBackend(Protocol):
    """One sharding implementation, as the trainer and its callbacks see it."""

    @property
    def name(self) -> str:
        """The value of ``parallelism.fsdp_backend`` that selects this one."""
        ...

    @property
    def exports_hf_checkpoint(self) -> bool:
        """Whether ``checkpoint.save_in_hf`` can write an HF export for it.

        False when the model state dict is not in a shape the converter reads,
        which also decides how the autometrics job is handed a checkpoint.
        """
        ...

    def validate_model(self, model_config: BaseModelConfig) -> None:
        """Refuse the models this backend is not implemented for.

        A backend is applied by the model's own ``parallelize_fn``, and only
        some of those implement more than one. A model whose parallelize
        function ignores ``parallelism.fsdp_backend`` would shard with whatever
        it hardcodes and report the backend the user asked for, so the refusal
        belongs in the shared capability contract. The factory checks that
        contract before construction; this method also guards direct use.
        """
        ...

    def validate_parallelism(self, parallelism: Any) -> None:
        """Refuse the parallelism combinations this backend cannot run.

        Called by ``build_sharding_backend``, which is the only way a run gets
        a backend, so the guards cannot be skipped by a trainer that forgot
        them. Not ``ParallelismConfig.__post_init__``: presets set
        ``fsdp_backend`` after construction.
        """
        ...

    def weight_init_context(self, model: nn.Module) -> AbstractContextManager[None]:
        """The context the trainer runs ``model``'s ``init_weights`` in."""
        ...

    def checkpoint_adapter(
        self, model_parts: list[nn.Module]
    ) -> CheckpointAdapter | None:
        """The adapter DCP goes through, or None for the upstream path."""
        ...

    def prepare_for_checkpoint_load(self, model_parts: list[nn.Module]) -> None:
        """Bring the model to a state a checkpoint can be read into."""
        ...

    def ema_store(self, model_parts: list[nn.Module]) -> EMAStore:
        """Allocate the averaged weights for this model."""
        ...

    def filter_monitor_sources(self, sources: dict[str, Any]) -> dict[str, Any]:
        """Drop the monitoring this backend cannot compute, warning per source."""
        ...

    def shard_stats(self) -> ShardStats | None:
        """How the metrics pipeline computes over this backend's shards.

        None when the standard tensor/DTensor paths already cover them.
        """
        ...
