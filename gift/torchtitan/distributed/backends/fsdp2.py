"""torch's own ``fully_shard``, and the EMA shadows kept beside it."""

import contextlib
from collections.abc import Collection
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.config.schema.distributed.sharding_backend import (
    model_sharding_backend_issues,
)
from torchtitan.config.schema.protocols.model import BaseModelConfig

from torchtitan.protocols.checkpoint_adapter import CheckpointAdapter
from torchtitan.protocols.sharding_backend import EMA_MODEL
from torchtitan.tools.logging import logger

__all__ = ["FSDP2Backend", "ShadowParamStore"]


_WRAPPER_SEGMENTS = frozenset(
    {"_checkpoint_wrapped_module", "_orig_mod", "_fsdp_wrapped_module"}
)


def _dcp_fqn(name: str) -> str:
    """The name ``get_model_state_dict`` would give this parameter or buffer."""
    return ".".join(
        segment for segment in name.split(".") if segment not in _WRAPPER_SEGMENTS
    )


class ShadowParamStore(Stateful):
    """EMA shadows as sharded DTensor clones, checkpointed under ``ema_model``.

    Read through ``get_model_state_dict`` so the clones stay sharded, and
    filtered to parameters: a buffer like the MoE router bias is an integral
    term of a controller, meaningless against a router it was not derived
    from.
    """

    def __init__(self, model_parts: list[nn.Module]) -> None:
        self.model_parts = model_parts
        self._weights: list[tuple[str, torch.Tensor]] | None = None
        self.shadow_params: dict[str, torch.Tensor] = {}
        self._restored = False
        self._missing_shadows: list[str] = []
        self.holds_average = False
        self.use_ema = True
        self.seed()

    @property
    def checkpoint_carries_ema(self) -> bool:
        return False

    def weights(self):
        """Floating-point model **parameters**, read through DCP."""
        for model_part in self.model_parts:
            buffers = {_dcp_fqn(name) for name, _ in model_part.named_buffers()}
            for name, tensor in get_model_state_dict(model_part).items():
                if name in buffers:
                    continue
                if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
                    yield name, tensor

    def _live_weights(self) -> list[tuple[str, torch.Tensor]]:
        if self._weights is None:
            self._weights = list(self.weights())
        return self._weights

    @torch.no_grad()
    def seed(self, only: Collection[str] | None = None) -> None:
        for name, tensor in self._live_weights():
            if only is None or name in only:
                self.shadow_params[name] = tensor.detach().clone()

    @torch.no_grad()
    def update(self, alpha: float) -> None:
        for name, tensor in self._live_weights():
            self.shadow_params[name].lerp_(tensor.detach(), alpha)

    @torch.no_grad()
    def swap(self) -> None:
        for name, tensor in self._live_weights():
            shadow = self.shadow_params[name]
            held = tensor.detach().clone()
            tensor.copy_(shadow)
            shadow.copy_(held)

    @torch.no_grad()
    def after_load(self) -> Collection[str]:
        # The tensors are stable from here on (the optimizer writes into them
        # in place), so the walk happens once, not on every update.
        self._weights = list(self.weights())
        if not self._restored:
            self.seed()
            return ()
        missing, self._missing_shadows = self._missing_shadows, []
        self.seed(only=set(missing))
        return missing

    def stateful_objects(self) -> dict[str, Stateful]:
        return {EMA_MODEL: self}

    def model_state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow_params

    def state_dict(self) -> dict[str, Any]:
        return self.shadow_params

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        missing = sorted(set(self.shadow_params) - set(state_dict))
        if missing:
            logger.warning(
                "The checkpoint carries no EMA weights for %d parameter(s), so"
                " theirs are seeded from the loaded LIVE weights and their"
                " average starts over from this step: %s",
                len(missing),
                missing,
            )
        self.shadow_params = {**self.shadow_params, **state_dict}
        self._missing_shadows = missing
        self._restored = True


class FSDP2Backend:
    """Implements ``ShardingBackend`` over torch's ``fully_shard``."""

    name = "fsdp2"
    exports_hf_checkpoint = True

    def validate_model(self, model_config: BaseModelConfig) -> None:
        issues = model_sharding_backend_issues(self.name, model_config)
        if issues:
            raise ValueError(str(issues[0]))

    def validate_parallelism(self, parallelism: Any) -> None:
        if parallelism.gfsdp_match_fsdp2_init:
            logger.warning(
                "parallelism.gfsdp_match_fsdp2_init is set, and fsdp_backend is "
                "'fsdp2', whose init is the one it matches: ignoring it."
            )

    def weight_init_context(self, model: nn.Module) -> AbstractContextManager[None]:
        return contextlib.nullcontext()

    def checkpoint_adapter(
        self, model_parts: list[nn.Module]
    ) -> CheckpointAdapter | None:
        return None

    def prepare_for_checkpoint_load(self, model_parts: list[nn.Module]) -> None:
        return None

    def ema_store(self, model_parts: list[nn.Module]) -> ShadowParamStore:
        return ShadowParamStore(model_parts)

    def filter_monitor_sources(self, sources: dict[str, Any]) -> dict[str, Any]:
        return sources

    def shard_stats(self) -> None:
        return None
