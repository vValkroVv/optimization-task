from __future__ import annotations
import functools
import time
from dataclasses import dataclass, field

import torch


def validate_timer_name(name: str) -> None:
    """Timer names are ``<namespace>.<leaf>``; the namespace groups the tags."""
    parts = name.split(".")
    if len(parts) < 2 or not all(parts):
        raise ValueError(
            f"Timer name '{name}' must be '<namespace>.<leaf>' with no empty segments."
        )


@dataclass
class TimerRecord:
    """Wall-clock durations of one instrumented function, one entry per call.

    ``level`` selects what a flush reports: 1 gives ``calls`` and ``avg_ms``, 2
    adds ``min_ms`` and ``max_ms``. The registry threshold caps it, so lowering
    the threshold silences a timer without touching its decorator.
    """

    name: str
    level: int = 1
    durations_ms: list[float] = field(default_factory=list)
    _start: float = 0.0
    _span: object = None

    @property
    def effective_level(self) -> int:
        return min(self.level, gigatimer.timers_threshold)

    def __enter__(self) -> TimerRecord:
        self._span = torch.profiler.record_function(self.name)
        self._span.__enter__()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.durations_ms.append((time.perf_counter() - self._start) * 1000.0)
        self._span.__exit__(exc_type, exc_value, traceback)
        self._span = None
        return False  # never suppress

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.effective_level < 1:
                return func(*args, **kwargs)
            with self:
                return func(*args, **kwargs)

        return wrapper

    def reset(self) -> None:
        self.durations_ms = []
        self._start = 0.0


class Timers:
    """Process-global registry of instrumented functions.

    Timers register themselves at import time through :func:`callback_timer`.
    ``TimersMonitor`` sets the threshold at fit start and flushes on its own
    cadence; a flush reports only the calls made since the previous one, so the
    cadence never has to line up with when the instrumented code runs.
    """

    _instance: Timers | None = None

    def __new__(cls) -> Timers:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.timers = {}
            # Fail closed: without TimersMonitor nothing raises this, so an
            # importer that never flushes cannot accumulate durations forever.
            cls._instance.timers_threshold = 0
            cls._instance._cumulative_ms = {}
        return cls._instance

    def set_timers_threshold(self, timers_threshold: int) -> None:
        """Cap every timer's level. 0 disables all of them."""
        if timers_threshold not in (0, 1, 2):
            raise ValueError(
                f"`timers_threshold` must be one of {{0, 1, 2}}, got {timers_threshold}."
            )
        self.timers_threshold = timers_threshold

    def timer(self, name: str, level: int) -> TimerRecord:
        """Register the timer called ``name``, or re-level the existing one."""
        validate_timer_name(name)
        if level not in (0, 1, 2):
            raise ValueError(f"`timer_level` must be one of {{0, 1, 2}}, got {level}.")
        timer = self.timers.get(name)
        if timer is None:
            timer = TimerRecord(name=name, level=level)
            self.timers[name] = timer
        else:
            timer.level = level
        return timer

    def reset_all(self) -> None:
        for timer in self.timers.values():
            timer.reset()
        self._cumulative_ms = {}

    def build_all_timers_metrics(self) -> dict[str, float]:
        """Flush every timer into ready-to-log tags and clear their histories.

        Emits ``{namespace}/{leaf}/{calls,avg_ms,min_ms,max_ms}`` per timer plus
        ``{namespace}/sum_ms_{interval,cumulative}`` per namespace. A timer with
        no calls this interval contributes no tags.
        """
        if self.timers_threshold < 1:
            return {}

        metrics: dict[str, float] = {}
        interval_ms: dict[str, float] = {}
        for name, timer in sorted(self.timers.items()):
            level = timer.effective_level
            durations = timer.durations_ms
            timer.reset()
            if level < 1 or not durations:
                continue
            namespace, leaf = name.split(".", 1)
            total_ms = sum(durations)
            metrics[f"{namespace}/{leaf}/calls"] = float(len(durations))
            metrics[f"{namespace}/{leaf}/avg_ms"] = total_ms / len(durations)
            if level >= 2:
                metrics[f"{namespace}/{leaf}/min_ms"] = min(durations)
                metrics[f"{namespace}/{leaf}/max_ms"] = max(durations)
            interval_ms[namespace] = interval_ms.get(namespace, 0.0) + total_ms

        for namespace, total_ms in sorted(interval_ms.items()):
            cumulative = self._cumulative_ms.get(namespace, 0.0) + total_ms
            self._cumulative_ms[namespace] = cumulative
            metrics[f"{namespace}/sum_ms_interval"] = total_ms
            metrics[f"{namespace}/sum_ms_cumulative"] = cumulative
        return metrics


gigatimer = Timers()


def callback_timer(timer_level: int = 1):
    """Time a method, naming the timer after its class and its own name.

    Produces ``timers_callback.{Class}.{method}``, so the published tags are
    ``timers_callback/{Class}.{method}/...``. ``timer_level`` is what a flush
    reports: 1 calls and avg, 2 adds min and max; ``timers_threshold`` caps it.
    """

    def decorator(fn):
        qualname_parts = fn.__qualname__.split(".")
        cls_name = qualname_parts[-2] if len(qualname_parts) >= 2 else "global"
        return gigatimer.timer(
            f"timers_callback.{cls_name}.{fn.__name__}", timer_level
        )(fn)

    return decorator
