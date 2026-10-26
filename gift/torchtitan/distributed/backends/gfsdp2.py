"""The in-repo GFSDP2 backend, and what it answers differently from fsdp2."""

import math
from collections.abc import Callable, Collection
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import Any, cast

import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor._dtensor_spec import DTensorSpec

from torchtitan.config.schema.distributed.sharding_backend import (
    model_sharding_backend_issues,
    sharding_backend_issues,
)
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.config.validation import Issue
from torchtitan.distributed.fsdp import _grouped_param_placement
from torchtitan.distributed.gfsdp import (
    GFSDPCheckpointAdapter,
    GFSDPModule,
    match_fsdp2_init,
    RaggedShard,
    RaggedShardDTensor,
    local_squared_norms,
    reduce_flat_segments,
)
from torchtitan.distributed.gfsdp._dcp import _lazy_init_gfsdp_roots
from torchtitan.protocols.checkpoint_adapter import CheckpointAdapter
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding_backend import ShardStats

__all__ = ["BackendBufferStore", "GFSDP2Backend", "GFSDPShardStats"]


def _raise_first(issues: list[Issue]) -> None:
    if issues:
        raise ValueError(str(issues[0]))


class GFSDPShardStats:
    """Implements ``ShardStats`` over ``torchtitan.distributed.gfsdp``."""

    tensor_types: tuple[type[torch.Tensor], ...] = (RaggedShardDTensor,)
    placement_types: tuple[type, ...] = (RaggedShard,)

    def local_squared_norms(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        assert isinstance(value, RaggedShardDTensor)
        return local_squared_norms(value, dim)

    def sum_flat_segments(
        self,
        value: torch.Tensor,
        num_segments: int,
        *,
        transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        sum_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return reduce_flat_segments(
            value,
            num_segments,
            reduce_op="sum",
            identity=0.0,
            transform=transform,
            sum_dtype=sum_dtype,
        )

    def max_flat_segments(self, value: torch.Tensor, num_segments: int) -> torch.Tensor:
        return reduce_flat_segments(
            value, num_segments, reduce_op="max", identity=float("-inf")
        )

    def min_flat_segments(self, value: torch.Tensor, num_segments: int) -> torch.Tensor:
        return reduce_flat_segments(
            value, num_segments, reduce_op="min", identity=float("inf")
        )

    def abs_max_flat_segments(
        self, value: torch.Tensor, num_segments: int
    ) -> torch.Tensor:
        return self.max_flat_segments(value.abs(), num_segments)

    @staticmethod
    def _centered_power_sum(
        rows: torch.Tensor,
        segments: torch.Tensor,
        mean: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        centered = rows - mean[segments].to(rows.dtype).unsqueeze(1)
        return (
            torch.linalg.vector_norm(centered, ord=order, dim=1, dtype=torch.float32)
            .pow(order)
            .unsqueeze(1)
        )

    def moment_flat_segments(
        self, value: torch.Tensor, num_segments: int, mean: torch.Tensor, order: int
    ) -> torch.Tensor:
        return self.sum_flat_segments(
            value,
            num_segments,
            transform=lambda rows, segments: self._centered_power_sum(
                rows, segments, mean, order
            ),
        )

    def to_local(self, value: torch.Tensor) -> torch.Tensor:
        assert isinstance(value, RaggedShardDTensor)
        return value._local_tensor

    def to_full_tensor(self, value: torch.Tensor) -> torch.Tensor:
        assert isinstance(value, RaggedShardDTensor)
        return value.full_tensor()


class BackendBufferStore:
    """EMA shadows as buffers inside the backend's own parameter groups.

    They reach the checkpoint as ``ema_model.*`` through
    ``GFSDPCheckpointAdapter``, so this store contributes no checkpoint state
    of its own; ``GFSDPLoadPlanner`` handles a checkpoint that lacks them.
    """

    def __init__(self, model_parts: list[nn.Module]) -> None:
        not_sharded = [
            type(part).__name__
            for part in model_parts
            if not isinstance(part, GFSDPModule)
        ]
        if not_sharded:
            raise ValueError(
                "EMA on fsdp_backend='gfsdp2' keeps its shadows in the "
                "backend's parameter groups, but these model parts are not "
                f"GFSDP modules: {not_sharded}."
            )
        self.model_parts = model_parts
        self.holds_average = False
        self.use_ema = True
        for part in model_parts:
            part.setup_ema_buffer()
        self.seed()

    @property
    def checkpoint_carries_ema(self) -> bool:
        return True

    def seed(self, only: Collection[str] | None = None) -> None:
        # Whole-group buffers: the backend has no per-parameter handle to seed.
        del only
        for part in self.model_parts:
            part.update_ema_buffer_after_load(force_update=True)

    def update(self, alpha: float) -> None:
        for part in self.model_parts:
            part.update_ema_params(alpha)

    def swap(self) -> None:
        for part in self.model_parts:
            part.swap_ema_and_weights()

    def after_load(self) -> Collection[str]:
        return ()

    def stateful_objects(self) -> dict[str, Stateful]:
        return {}


def _fsdp2_grouped_num_shards(spec: DTensorSpec) -> int:
    """Shards fsdp2 splits a routed-expert weight into; see ``_fully_shard_unit``."""
    names = spec.mesh.mesh_dim_names
    assert names is not None

    def size(name: str) -> int:
        return spec.mesh.size(names.index(name)) if name in names else 1

    if "ep" in names:
        return size("efsdp") * size("ep")
    return size("dp_replicate") * size("fsdp")


def _fsdp2_shard_dims(model: nn.Module) -> dict[int, int]:
    """``id(parameter)`` -> the dim fsdp2 shards it on, where that is not dim 0.

    fsdp2 shards routed experts below dim 0 when there are fewer of them than
    shards; ``_fully_shard_unit`` picks the dim with ``_grouped_param_placement``.
    """
    shard_dims = {}
    for param, axis_len in cast(Module, model).grouped_param_axes():
        assert isinstance(param, RaggedShardDTensor), type(param)
        placement = _grouped_param_placement(
            param, axis_len, num_shards=_fsdp2_grouped_num_shards(param._spec)
        )
        if placement.dim != 0:
            shard_dims[id(param)] = placement.dim
    return shard_dims


class GFSDP2Backend:
    """Implements ``ShardingBackend`` over ``torchtitan.distributed.gfsdp``."""

    name = "gfsdp2"
    exports_hf_checkpoint = False

    def __init__(self, *, match_fsdp2_init: bool = False) -> None:
        self._match_fsdp2_init = match_fsdp2_init

    def weight_init_context(self, model: nn.Module) -> AbstractContextManager[None]:
        if not self._match_fsdp2_init:
            return match_fsdp2_init(False)
        return match_fsdp2_init(fsdp2_shard_dims=_fsdp2_shard_dims(model))

    # Same rules as pre-submit, re-checked here so a trainer cannot skip them.
    def validate_model(self, model_config: BaseModelConfig) -> None:
        _raise_first(model_sharding_backend_issues(self.name, model_config))

    def validate_parallelism(self, parallelism: Any) -> None:
        _raise_first(
            sharding_backend_issues(replace(parallelism, fsdp_backend=self.name))
        )

    def checkpoint_adapter(
        self, model_parts: list[nn.Module]
    ) -> CheckpointAdapter | None:
        return GFSDPCheckpointAdapter(model_parts)

    def prepare_for_checkpoint_load(self, model_parts: list[nn.Module]) -> None:
        _lazy_init_gfsdp_roots(model_parts)

    def ema_store(self, model_parts: list[nn.Module]) -> BackendBufferStore:
        return BackendBufferStore(model_parts)

    def filter_monitor_sources(self, sources: dict[str, Any]) -> dict[str, Any]:
        """Reject unsupported expert stats."""
        from torchtitan.callbacks.sources import ModellingSource

        kept: dict[str, Any] = {}
        for name, source in sources.items():
            if isinstance(source, ModellingSource.Config) and source.moe.expert_weights:
                for metric in source.moe.weight_stats:
                    steps = metric.effective_steps()
                    op = steps[-1].stat if steps else None
                    if op is None or op.op not in (
                        "norm",
                        "sum",
                        "mean",
                        "rms",
                        "abs_max",
                        "min",
                        "max",
                        "count",
                        "nonzero_count",
                        "nan_count",
                        "inf_count",
                        "zero_ratio",
                        "finite_ratio",
                        "positive_ratio",
                        "std",
                        "var",
                        "cv",
                        "kurtosis",
                        "max_min",
                        "max_min_ratio",
                        "max_vio",
                        "actabs",
                        "actrel",
                        "entropy",
                        "raw_entropy_norm",
                        "quantile",
                        "min_median_ratio",
                        "gini",
                    ):
                        raise ValueError(
                            f"gfsdp2: unsupported final stat in moe.weight_stats "
                            f"metric '{metric.name}': {op.op if op else None}"
                        )
                    if op.op == "norm":
                        p = 2.0 if op.p is None else float(op.p)
                        if math.isnan(p):
                            raise ValueError(
                                f"gfsdp2: moe.weight_stats metric '{metric.name}' "
                                f"requires a non-NaN norm p, got {p}"
                            )
            kept[name] = source
        return kept

    def shard_stats(self) -> ShardStats:
        return GFSDPShardStats()
