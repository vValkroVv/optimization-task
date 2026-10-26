from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE

_Result = TypeVar("_Result")


@functools.cache
def _shared_pool() -> _POOL_HANDLE:  # pragma: no cover
    return torch.cuda.graph_pool_handle()


class GraphedComputation(Generic[_Result]):  # pragma: no cover
    """`compute(*inputs)` captured once as a CUDA graph for fixed input shapes, strides and dtypes,
    then replayed: one launch instead of one per kernel, and the same kernels, so the same values.
    `compute` must not synchronize with the host. Every graph's intermediates share one memory pool,
    so outputs are valid only until the next call of any instance."""

    def __init__(self, compute: Callable[..., _Result], *like: torch.Tensor) -> None:
        device = like[0].device
        self._inputs = tuple(tensor.clone() for tensor in like)
        # Lazy initialization inside compute (handles, compiled kernels) happens outside the capture.
        warmup = torch.cuda.Stream(device=device)
        warmup.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup):
            compute(*self._inputs)
        torch.cuda.current_stream(device).wait_stream(warmup)
        self._graph = torch.cuda.CUDAGraph()
        # Other threads (communication, the recorder's marker) keep using the device meanwhile.
        with torch.cuda.graph(self._graph, pool=_shared_pool(), capture_error_mode="thread_local"):
            self._outputs = compute(*self._inputs)

    def __call__(self, *values: torch.Tensor) -> _Result:
        for static, value in zip(self._inputs, values, strict=True):
            static.copy_(value)
        self._graph.replay()
        return self._outputs
