"""Tiny stage profiler shared across ``weights/`` commands.

Usage::

    timer = StageTimer()
    with timer.stage("io_load_many"):
        tensors = reader.load_many(names)
    with timer.stage("basic", sync_cuda=True):
        ...
    print(timer.summary())

Each ``stage(name)`` accumulates wall-time. CUDA-side work needs
``sync_cuda=True`` to charge time to the right bucket (otherwise the
launching CPU sees the kernel as instant).

Outputs at the end::

    === timing summary  (wall=143.2s) ===
                       io_load_many   total= 76.20s ( 53.2%)  count=    2  avg=38100.0ms
                              basic   total= 41.30s ( 28.8%)  count= 1024  avg=   40.3ms
                           spectral   total= 15.70s ( 11.0%)  count=  640  avg=   24.5ms
                             rowcol   total=  6.40s (  4.5%)  count= 1024  avg=    6.3ms
                          to_device   total=  3.50s (  2.4%)  count= 1024  avg=    3.4ms
                        plan_chunks   total=  0.10s (  0.1%)  count=    1  avg=  100.0ms

— so you can tell at a glance whether you're I/O- or compute-bound and
which kernel dominates.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class StageTimer:
    """Aggregate per-stage wall time across many invocations."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._start = time.time()

    @contextmanager
    def stage(self, name: str, *, sync_cuda: bool = False) -> Iterator[None]:
        """Time a named stage.

        Set ``sync_cuda=True`` for GPU work to avoid undercounting due to
        async kernel launches.
        """
        if sync_cuda:
            self._sync()
        t0 = time.time()
        try:
            yield
        finally:
            if sync_cuda:
                self._sync()
            dt = time.time() - t0
            self._totals[name] = self._totals.get(name, 0.0) + dt
            self._counts[name] = self._counts.get(name, 0) + 1

    @staticmethod
    def _sync() -> None:
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()

    def elapsed(self) -> float:
        """Total wall time since the timer was constructed."""
        return time.time() - self._start

    def totals(self) -> dict[str, dict[str, float]]:
        """Return raw counters as ``{stage: {total, count, avg_ms}}``."""
        return {
            name: {
                "total_sec": total,
                "count": self._counts[name],
                "avg_ms": (total / self._counts[name] * 1000.0)
                if self._counts[name]
                else 0.0,
            }
            for name, total in self._totals.items()
        }

    def summary(self) -> str:
        elapsed = self.elapsed()
        if not self._totals:
            return (
                f"\n=== timing summary  (wall={elapsed:.1f}s; no stages recorded) ==="
            )
        items = sorted(self._totals.items(), key=lambda kv: -kv[1])
        lines = [f"\n=== timing summary  (wall={elapsed:.1f}s) ==="]
        for name, total in items:
            count = self._counts[name]
            avg = (total / count) if count else 0.0
            pct = 100.0 * total / elapsed if elapsed > 0 else 0.0
            lines.append(
                f"  {name:>34}  total={total:7.2f}s ({pct:5.1f}%)  "
                f"count={count:>5}  avg={avg * 1000:8.1f}ms"
            )
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary(), flush=True)
