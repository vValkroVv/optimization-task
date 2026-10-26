from typing import TypeAlias

from torchtitan.config.schema.callbacks.builtin.timers_monitor import (
    TimersMonitorConfig,
)
from torchtitan.callbacks.base import Aggregator, ObserverState
from torchtitan.callbacks.metrics.timers import gigatimer
from torchtitan.config import Configurable


class TimersMonitor(Configurable, Aggregator):
    """Publishes gigatimer aggregations.

    An :class:`Aggregator`, and that is what puts it after every other callback
    of the event: ``build_all_timers_metrics`` clears each timer as it reads
    it, so anything instrumented that ran later would be counted in the next
    interval. Being one by type, it no longer depends on sitting last in the
    run's ``callbacks`` dict.

    Instrumented functions record one duration per call into the module-global
    ``gigatimer`` registry the moment they run. This callback only sets the
    registry's ``timers_threshold`` at fit start and flushes ready-to-log
    metrics every ``interval`` steps — an interval with no calls simply produces
    no tags for that timer, so the cadence never has to line up with when the
    instrumented code actually fires.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = TimersMonitorConfig

    def __init__(self, config: Config) -> None:
        self.config = config

    def fit_start(self, state: ObserverState) -> None:
        del state
        gigatimer.reset_all()
        gigatimer.set_timers_threshold(self.config.timers_threshold)

    def batch_end(self, state: ObserverState) -> None:
        if state.step % self.config.interval != 0:
            return
        metrics = gigatimer.build_all_timers_metrics()
        if metrics:
            state.record_all(metrics)
