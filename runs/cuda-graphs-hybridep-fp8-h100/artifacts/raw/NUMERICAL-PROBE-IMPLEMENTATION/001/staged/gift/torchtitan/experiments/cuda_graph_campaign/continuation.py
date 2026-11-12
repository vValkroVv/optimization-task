"""Rank-local RNG checkpoint state for the exact CUDA graph campaign target."""

import pickle
import random
import weakref
from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.config import Configurable
from torchtitan.config.schema.experiments.cuda_graph_campaign.continuation import (
    CampaignRNGContinuationConfig,
)


_SCHEMA_VERSION = 1
_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "global_rank",
        "world_size",
        "device_type",
        "device_index",
        "python_numpy_state",
        "torch_cpu_state",
        "torch_cuda_state",
    }
)


class RNGContinuationState(Stateful):
    """Immutable rank-owned RNG snapshots with a deferred, one-shot restore."""

    def __init__(self, *, global_rank: int, world_size: int, device: torch.device):
        if type(global_rank) is not int or global_rank < 0:
            raise ValueError("global_rank must be a nonnegative integer")
        if type(world_size) is not int or world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if global_rank >= world_size:
            raise ValueError("global_rank must be less than world_size")
        device = torch.device(device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("RNG continuation requires an indexed CUDA device")

        self.global_rank = global_rank
        self.world_size = world_size
        self.device = device
        self._pending_restore: dict[str, Any] | None = None
        self._restore_attempted = False

    def _validate_active_owner(self) -> None:
        if not dist.is_initialized():
            raise ValueError("RNG continuation requires an initialized process group")
        if dist.get_rank() != self.global_rank:
            raise ValueError(
                f"RNG continuation belongs to global rank {self.global_rank}, "
                f"current rank is {dist.get_rank()}"
            )
        if dist.get_world_size() != self.world_size:
            raise ValueError(
                f"RNG continuation belongs to world size {self.world_size}, "
                f"current world size is {dist.get_world_size()}"
            )
        current_device = torch.cuda.current_device()
        if current_device != self.device.index:
            raise ValueError(
                f"RNG continuation belongs to cuda:{self.device.index}, "
                f"current device is cuda:{current_device}"
            )

    def state_dict(self) -> dict[str, Any]:
        """Return fresh copies so checkpoint serialization cannot mutate state."""
        self._validate_active_owner()
        python_numpy_state = pickle.dumps(
            (random.getstate(), np.random.get_state()),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return {
            "schema_version": _SCHEMA_VERSION,
            "global_rank": self.global_rank,
            "world_size": self.world_size,
            "device_type": self.device.type,
            "device_index": self.device.index,
            "python_numpy_state": python_numpy_state,
            "torch_cpu_state": torch.get_rng_state().clone(),
            "torch_cuda_state": torch.cuda.get_rng_state(self.device).clone(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Validate and stage checkpoint data without changing process RNGs."""
        self._validate_active_owner()
        staged = self._validate_state_dict(state_dict)
        if self._restore_attempted:
            raise RuntimeError("RNG continuation restore has already been attempted")
        self._pending_restore = staged

    def pending_restore_snapshot(self) -> dict[str, Any] | None:
        """Copy the validated pending payload without consuming or restoring it."""
        if self._pending_restore is None:
            return None
        return {
            **self._pending_restore,
            "torch_cpu_state": self._pending_restore["torch_cpu_state"].clone(),
            "torch_cuda_state": self._pending_restore["torch_cuda_state"].clone(),
        }

    def _validate_state_dict(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(state_dict, Mapping) or set(state_dict) != _STATE_FIELDS:
            raise ValueError("RNG continuation state has an unsupported field set")

        self._require_int(state_dict, "schema_version", _SCHEMA_VERSION)
        self._require_int(state_dict, "global_rank", self.global_rank)
        self._require_int(state_dict, "world_size", self.world_size)
        if state_dict["device_type"] != self.device.type:
            raise ValueError(
                f"RNG continuation device type {state_dict['device_type']!r} "
                f"does not match {self.device.type!r}"
            )
        self._require_int(state_dict, "device_index", self.device.index)

        python_numpy_state = state_dict["python_numpy_state"]
        if not isinstance(python_numpy_state, bytes):
            raise ValueError("python_numpy_state must be pickled bytes")
        try:
            python_state, numpy_state = pickle.loads(python_numpy_state)
            random.Random().setstate(python_state)
            np.random.RandomState().set_state(numpy_state)
        except Exception as error:
            raise ValueError("python_numpy_state is invalid") from error

        cpu_state = state_dict["torch_cpu_state"]
        current_cpu_state = torch.get_rng_state()
        self._validate_rng_tensor(cpu_state, "torch_cpu_state", current_cpu_state.shape)

        cuda_state = state_dict["torch_cuda_state"]
        current_cuda_state = torch.cuda.get_rng_state(self.device)
        self._validate_rng_tensor(
            cuda_state, "torch_cuda_state", current_cuda_state.shape
        )

        return {
            **state_dict,
            "torch_cpu_state": cpu_state.clone(),
            "torch_cuda_state": cuda_state.clone(),
        }

    @staticmethod
    def _require_int(state_dict: Mapping[str, Any], field: str, expected: int) -> None:
        value = state_dict[field]
        if type(value) is not int or value != expected:
            raise ValueError(
                f"RNG continuation {field} {value!r} does not match {expected}"
            )

    @staticmethod
    def _validate_rng_tensor(
        value: Any, field: str, expected_shape: torch.Size
    ) -> None:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.device.type != "cpu"
            or value.ndim != 1
            or value.shape != expected_shape
        ):
            raise ValueError(
                f"{field} must be a CPU uint8 vector of shape {tuple(expected_shape)}"
            )

    def restore_after_first_batch(self) -> bool:
        """Apply staged RNG state once after lazy iterator creation and fetch."""
        if self._pending_restore is None:
            return False
        if self._restore_attempted:
            raise RuntimeError("RNG continuation restore has already been attempted")
        self._validate_active_owner()
        self._restore_attempted = True

        state = self._pending_restore
        python_state, numpy_state = pickle.loads(state["python_numpy_state"])
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(state["torch_cpu_state"])
        torch.cuda.set_rng_state(state["torch_cuda_state"], device=self.device)

        self._pending_restore = None
        return True


class CampaignRNGContinuationObserver(Configurable, Observer):
    """Registers one checkpoint state for this process's RNG generators."""

    Config: TypeAlias = CampaignRNGContinuationConfig

    def __init__(self, config: Config) -> None:
        if not dist.is_initialized():
            raise ValueError("campaign_rng_continuation requires distributed training")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", torch.cuda.current_device())
        self.rng_state = RNGContinuationState(
            global_rank=global_rank,
            world_size=world_size,
            device=device,
        )
        self.state_key = f"continuation_rng_rank_{global_rank}"
        self._restore_owner: weakref.ReferenceType[Any] | None = None

    def bind_restore_owner(self, owner: Any) -> None:
        from torchtitan.experiments.cuda_graph_campaign.trainer import CampaignTrainer

        if not isinstance(owner, CampaignTrainer):
            raise TypeError(
                "campaign_rng_continuation can only bind to CampaignTrainer"
            )
        if getattr(owner, "rng_continuation", None) is not self:
            raise ValueError("CampaignTrainer does not own this continuation observer")
        bound_owner = self._restore_owner() if self._restore_owner is not None else None
        if self._restore_owner is not None and bound_owner is not owner:
            raise RuntimeError("campaign_rng_continuation is already bound")
        self._restore_owner = weakref.ref(owner)

    def _require_restore_owner(self) -> None:
        owner = self._restore_owner() if self._restore_owner is not None else None
        if owner is None or getattr(owner, "rng_continuation", None) is not self:
            raise RuntimeError(
                "campaign_rng_continuation requires CampaignTrainer binding "
                "before fit_start or RNG restore"
            )

    def fit_start(self, state: ObserverState) -> None:
        self._require_restore_owner()

    def stateful_objects(self) -> dict[str, Stateful]:
        return {self.state_key: self.rng_state}

    def restore_after_first_batch(self) -> bool:
        self._require_restore_owner()
        return self.rng_state.restore_after_first_batch()

    def pending_restore_snapshot(self) -> dict[str, Any] | None:
        """Expose a copy for the opt-in probe without advancing RNG state."""
        return self.rng_state.pending_restore_snapshot()
