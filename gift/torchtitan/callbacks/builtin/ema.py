from typing import TypeAlias

import torch
from torch import nn
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.config.schema.callbacks.builtin.ema import EMAConfig
from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.config import Configurable
from torchtitan.protocols.sharding_backend import EMAStore, ShardingBackend


class EMA(Configurable, Observer):
    """Maintains an exponential moving average of the model weights.

    After every ``update_period``-th optimizer step from ``start_step`` on,
    each floating-point parameter updates its shadow copy as
    ``shadow = decay * shadow + (1 - decay) * param`` (llm-f ``ema_weights``
    with ``alpha = 1 - decay``). At ``start_step`` itself the shadows are first
    re-seeded from the live weights, so the average does not carry the
    initialization.

    This callback decides *when*; the backend's ``EMAStore`` decides where the
    shadows live and how they reach a checkpoint. Either way a checkpoint
    carries the averaged weights as ``ema_model.<fqn>``.

    Example:
        ``callbacks["ema"] = EMA.Config(decay=0.99)`` in the preset
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = EMAConfig

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        sharding_backend: ShardingBackend,
    ) -> None:
        self.config = config
        self.sharding_backend = sharding_backend
        self.store: EMAStore = sharding_backend.ema_store(model_parts)
        self.store.use_ema = config.use_ema

    def fit_start(self, state: ObserverState) -> None:
        if (
            self.store.checkpoint_carries_ema
            and state.config.checkpoint.last_save_ema_model_also
        ):
            raise ValueError(
                "checkpoint.last_save_ema_model_also is not supported on "
                f"fsdp_backend={self.sharding_backend.name!r}: every checkpoint "
                "already carries the EMA weights as ema_model.* tensors, which "
                "the converters read with use_ema."
            )

    @torch.no_grad()
    def after_load(self, state: ObserverState) -> None:
        """Seed the shadows from the weights the run will actually start from."""
        self.store.holds_average = state.step >= self.config.start_step
        self.store.after_load()

    @torch.no_grad()
    def after_optimizer_step(self, state: ObserverState) -> None:
        config, step = self.config, state.step
        self.store.holds_average = step >= config.start_step
        if step < config.start_step:
            return
        if step == config.start_step and step > 0:
            self.store.seed()
        if step % config.update_period != 0:
            return
        self.store.update(1.0 - config.decay)

    @torch.no_grad()
    def swap_weights(self) -> None:
        """Exchange the live weights with the shadows, in place.

        Not idempotent, and it must be paired. An unpaired call leaves the
        model running on averaged weights.
        """
        self.store.swap()

    def stateful_objects(self) -> dict[str, Stateful]:
        return self.store.stateful_objects()
