from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch

from stand.activation_kernel_checks.cuda_timing import MICROSECONDS_PER_MILLISECOND
from stand.activation_kernel_checks.fake_recorder import WORDS_WITH_QUANTILES
from stand.activation_kernel_checks.kernel_calls import observe_with_quantiles
from stand.activation_kernel_checks.populations import ACTIVATION_SCALE

# Calls captured in one CUDA graph and the graph replays timed. The shapes are tokens by width: three widths of
# activations, then the rows a mixture of experts routes.
GRAPH_CALLS = 20
GRAPH_REPLAYS = 10
TIMED_SHAPES = ((49152, 896), (49152, 2304), (49152, 7168), (393216, 256))
BATCHED_TENSORS = 8
EXPERT_SEGMENTS = 32
BYTES_PER_MEGABYTE = 1e6
GIGABYTES_PER_SECOND_PER_MEGABYTE_PER_MICROSECOND = 1e3


@dataclass
class _TimedCalls:
    """The tensors one shape's timed calls take, and the calls of a module of the extension on them."""

    activations: torch.Tensor
    accumulator: torch.Tensor
    words: torch.Tensor

    @cached_property
    def batch(self) -> list[torch.Tensor]:
        return [self.activations] * BATCHED_TENSORS

    @cached_property
    def segment_counts(self) -> list[int]:
        rows = self.activations.shape[0]
        return [rows // EXPERT_SEGMENTS] * (EXPERT_SEGMENTS - 1) + [
            rows - rows // EXPERT_SEGMENTS * (EXPERT_SEGMENTS - 1)
        ]

    def statistics(self, module) -> Callable[[], None]:
        return lambda: observe_with_quantiles(module, self.activations, self.words)

    def row_norms(self, module) -> Callable[[], object]:
        return lambda: module.row_norm_accumulate(self.activations, self.accumulator, 0, False)

    def batched_row_norms(self, module) -> Callable[[], object]:
        return lambda: module.row_norm_accumulate_batch(
            self.batch, self.accumulator, [1] * BATCHED_TENSORS, list(range(BATCHED_TENSORS)))

    def expert_sums(self, module) -> Callable[[], object]:
        return lambda: module.segment_sumsq_into(
            self.activations, self.segment_counts, self.words.view(torch.int32), 0, False)


def graph_microseconds(call: Callable[[], object]) -> float:
    """GPU microseconds per call of `call`: `GRAPH_CALLS` calls captured in one CUDA graph and replayed, so
    that neither the CPU's launches nor its gaps are in the time."""
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_CALLS):
            call()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(GRAPH_REPLAYS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * MICROSECONDS_PER_MILLISECOND / (GRAPH_REPLAYS * GRAPH_CALLS)


def kernel_totals(events: list[dict]) -> dict[str, tuple[int, float]]:
    """The launches and the GPU microseconds of each kernel in the events of a profiler's trace."""
    totals: dict[str, tuple[int, float]] = {}
    for event in events:
        if event.get("cat") == "kernel":
            launches, microseconds = totals.get(event["name"], (0, 0.0))
            totals[event["name"]] = (launches + 1, microseconds + event["dur"])
    return totals


def print_kernels(call: Callable[[], object]) -> None:
    """The kernels `GRAPH_CALLS` calls of `call` launch, as launches and GPU microseconds per call."""
    call()
    torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
            for _ in range(GRAPH_CALLS):
                call()
            torch.cuda.synchronize()
        trace = Path(directory) / "trace.json"
        profile.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    for name, (launches, microseconds) in sorted(kernel_totals(events).items(), key=lambda item: -item[1][1]):
        print(f"    {name[:90]:90s} {launches / GRAPH_CALLS:5.1f} launches {microseconds / GRAPH_CALLS:9.1f} us")


def time_kernels(extension, baseline, breakdown: bool) -> None:
    """GPU time per call of the statistics with exact quantiles, the row norms of one tensor and of a batch,
    and the per-expert sums of squares, for each shape of `TIMED_SHAPES`: the baseline's, the current one's
    and their ratio, the current one's rate against the bytes it reads once, and the time of a plain
    reduction of the same tensor, which is what reading the bytes takes on the device. With `breakdown`, the
    kernels of each call and their times follow it."""
    device = torch.device("cuda")
    accumulator = torch.zeros(BATCHED_TENSORS, 2, dtype=torch.float64, device=device)
    words = torch.zeros(WORDS_WITH_QUANTILES, device=device)
    print(f"\n{'call':34s} {'shape':>15s} {'MB read':>8s} {'baseline us':>12s} {'current us':>11s} {'ratio':>6s} "
          f"{'GB/s':>6s} {'plain read us':>14s}")
    for rows, width in TIMED_SHAPES:
        for dtype in (torch.bfloat16, torch.float16):
            if dtype == torch.float16 and (rows, width) != TIMED_SHAPES[0]:
                continue
            # Nothing keeps the shape's tensors once its calls are timed, so they are freed before the next shape's
            # are allocated.
            activations = (torch.randn(rows, width, device=device) * ACTIVATION_SCALE).to(dtype)
            _time_shape(_TimedCalls(activations, accumulator, words), extension, baseline, breakdown)
            del activations


def _time_shape(calls: _TimedCalls, extension, baseline, breakdown: bool) -> None:
    activations = calls.activations
    rows, width = activations.shape
    floor_microseconds = graph_microseconds(lambda: activations.amax())
    for label, call, tensors in (("statistics + quantiles", calls.statistics, 1), ("row norms", calls.row_norms, 1),
                                 (f"row norms of {BATCHED_TENSORS}", calls.batched_row_norms, BATCHED_TENSORS),
                                 ("expert sums of squares", calls.expert_sums, 1)):
        megabytes = tensors * activations.numel() * activations.element_size() / BYTES_PER_MEGABYTE
        baseline_microseconds = graph_microseconds(call(baseline)) if baseline is not None else math.nan
        current_microseconds = graph_microseconds(call(extension))
        dtype_name = str(activations.dtype).removeprefix("torch.")
        print(f"{f'{label} {dtype_name}':34s} {f'{rows}x{width}':>15s} "
              f"{megabytes:8.1f} {baseline_microseconds:12.1f} "
              f"{current_microseconds:11.1f} {baseline_microseconds / current_microseconds:6.2f} "
              f"{megabytes * GIGABYTES_PER_SECOND_PER_MEGABYTE_PER_MICROSECOND / current_microseconds:6.0f} "
              f"{floor_microseconds * tensors:14.1f}")
        if breakdown:
            for name, module in (("baseline", baseline), ("current", extension)):
                if module is not None:
                    print(f"  {name}:")
                    print_kernels(call(module))
