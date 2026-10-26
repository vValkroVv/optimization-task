from __future__ import annotations

from collections.abc import Callable

import torch

TIMING_REPEATS = 50
TIMING_WARMUP = 5
OBSERVE_TIMING_REPEATS = 200
OBSERVE_TIMING_WARMUP = 20
MICROSECONDS_PER_MILLISECOND = 1000


def time_cuda(function: Callable[[], object], repeats: int, warmup: int = TIMING_WARMUP) -> float:
    """Milliseconds per call of `function`, synchronizing only before and after the whole batch of calls."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats
