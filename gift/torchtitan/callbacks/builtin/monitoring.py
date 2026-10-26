from typing import TypeAlias

import torch

from torchtitan.config.schema.callbacks.builtin.monitoring import (
    MonitoringCallbackConfig,
)
from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.callbacks.metrics.reducers import (
    finalize_pending_reductions,
    ParallelReducer,
)
from torchtitan.callbacks.metrics.sample import MetricRecord
from torchtitan.callbacks.metrics.stats import apply_post_stats
from torchtitan.callbacks.metrics.stores import MetricStore
from torchtitan.callbacks.metrics.timers import callback_timer
from torchtitan.callbacks.sources import MonitorSource
from torchtitan.config import Configurable
from torchtitan.distributed.compile import EagerWindow
from torchtitan.protocols.sharding_backend import ShardingBackend
from torchtitan.tools.logging import logger


class MonitoringCallback(Configurable, Observer):
    """Runs the monitoring sources over the training loop."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = MonitoringCallbackConfig

    def __init__(self, config: Config, *, sharding_backend: ShardingBackend) -> None:
        self._shard_stats = sharding_backend.shard_stats()
        self.config = config._copy(
            sources=sharding_backend.filter_monitor_sources(config.sources)
        )
        self.sources: list[MonitorSource] = self.build_sources()
        self.reducer: ParallelReducer | None = None
        self.store: MetricStore | None = None
        self._eager_window = EagerWindow((), owner="monitoring")

    def stateful_objects(self) -> dict[str, object]:
        states: dict[str, object] = {}
        for source in self.sources:
            states.update(source.stateful_objects())
        return states

    def fit_start(self, state: ObserverState) -> None:
        hooked = any(
            source.uses_module_hooks for source in self.config.sources.values()
        )
        self._eager_window = EagerWindow(
            state.model_parts if hooked else (), owner="monitoring"
        )
        if len(self._eager_window) and self.config.log_interval == 1:
            logger.warning(
                "MonitoringCallback: log_interval=1 with %d compiled blocks — "
                "the modelling hooks need every logged step eager, so every "
                "step runs uncompiled. Raise log_interval to get compile back.",
                len(self._eager_window),
            )
        elif len(self._eager_window):
            logger.warning(
                "MonitoringCallback: %d compiled blocks run eagerly on every "
                "%d-th step (1/%d of training) so the modelling hooks see "
                "them; expect throughput and memory to differ on those steps",
                len(self._eager_window),
                self.config.log_interval,
                self.config.log_interval,
            )
        if self.reducer is None:
            self.reducer = ParallelReducer(
                state.parallel_dims, shard_stats=self._shard_stats
            )
            self.store = MetricStore(self.config.store)
        for source in self.sources:
            source.reducer = self.reducer
            source.debug_full_tensor = self.config.debug_full_tensor
            source.log_interval = self.config.log_interval
            source.fit_start(state)

    def batch_start(self, state: ObserverState) -> None:
        if not self.sources:
            return
        active = state.step % self.config.log_interval == 0
        if active:
            self._eager_window.open()
        for source in self.sources:
            source.active = active
            source.batch_start(state)

    def microbatch_start(self, state: ObserverState) -> None:
        for source in self.sources:
            source.microbatch_start(state)

    def after_load(self, state: ObserverState) -> None:
        for source in self.sources:
            source.after_load(state)

    def after_checkpoint_save(self, state: ObserverState) -> None:
        for source in self.sources:
            source.after_checkpoint_save(state)

    def fit_end(self, state: ObserverState) -> None:
        for source in self.sources:
            source.fit_end(state)

    def validation_start(self, state: ObserverState) -> None:
        for source in self.sources:
            source.validation_start(state)

    def validation_end(self, state: ObserverState) -> None:
        for source in self.sources:
            source.validation_end(state)

    def before_dataloader(self, state: ObserverState) -> None:
        for source in self.sources:
            source.before_dataloader(state)

    def after_dataloader(self, state: ObserverState) -> None:
        for source in self.sources:
            source.after_dataloader(state)

    def after_forward(self, state: ObserverState) -> None:
        for source in self.sources:
            source.after_forward(state)

    def before_optimizer_step(self, state: ObserverState) -> None:
        if (
            self.config.empty_cuda_cache_before_optimizer
            and self.sources
            and state.step % self.config.log_interval == 0
            and torch.device(state.device).type == "cuda"
        ):
            with torch.cuda.device(state.device):
                torch.cuda.empty_cache()
        for source in self.sources:
            source.before_optimizer_step(state)

    def after_optimizer_step(self, state: ObserverState) -> None:
        for source in self.sources:
            source.after_optimizer_step(state)

    @callback_timer(timer_level=1)
    def batch_end(self, state: ObserverState) -> None:
        self._eager_window.close()
        if self.store is None:
            return
        if state.step % self.config.log_interval != 0:
            return
        records: list[MetricRecord] = []
        for source in self.sources:
            source.batch_end(state)
            records.extend(source.drain())

        # Invariant: the reducer is created together with the store in
        # fit_start, and this method returns early while store is None.
        assert self.reducer is not None, "batch_end before fit_start"
        records = finalize_pending_reductions(records, self.reducer)
        # [reduce..., stat...] metrics carry their stats past the fused
        # reductions; this is where they finish. Rank-local, no collectives.
        records = apply_post_stats(records)

        for record in records:
            self.store.write(record)
        state._records.extend(records)

    def find_source(self, source_cls: type) -> MonitorSource | None:
        for source in self.sources:
            if isinstance(source, source_cls):
                return source
        return None

    def close(self) -> None:
        self._eager_window.close()
        for source in self.sources:
            source.close()
        self.sources.clear()

    def build_sources(self) -> list[MonitorSource]:
        """The sources this run lists, in the order it lists them."""
        return [config.build() for config in self.config.sources.values()]

    def described_children(self):
        return {
            key: type(source).__name__
            for key, source in zip(self.config.sources, self.sources)
        }
