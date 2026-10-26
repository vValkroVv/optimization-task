from __future__ import annotations

from collections.abc import Callable

import torch

from metric_sampler.capture.stats.activations import ActivationStats
from stand.activation_kernel_checks.cuda_timing import (
    MICROSECONDS_PER_MILLISECOND,
    OBSERVE_TIMING_REPEATS,
    OBSERVE_TIMING_WARMUP,
    TIMING_REPEATS,
    time_cuda,
)
from stand.activation_kernel_checks.fake_recorder import WORDS_WITH_QUANTILES, FakeRecorder, make_recorder
from stand.activation_kernel_checks.kernel_calls import (
    launch_statistics_kernel,
    observe_statistics,
    observe_with_quantiles,
)
from stand.activation_kernel_checks.populations import ACTIVATION_SCALE
from stand.activation_kernel_checks.statistics_oracle import TRITON_TILE_ELEMENTS, compare, run_reference

TRITON_NUM_WARPS = (2, 4, 8)
ZEROED_FRACTION = 0.5
PERCENT = 100
QUANTILE_PATH_CASES = (
    (49152, 896, torch.bfloat16, False), (49152, 896, torch.bfloat16, True),
    (65536, 7168, torch.bfloat16, False), (65536, 7168, torch.bfloat16, True),
    (49152, 896, torch.float16, False),
)
BREAKDOWN_SHAPE = (4096, 896)
TUNING_SHAPES = ((49152, 896), (49152, 1792), (65536, 7168))
OVERHEAD_SHAPES = ((4096, 896), (4096, 7168))


def time_extension_paths(extension) -> list[str]:
    for rows, width, dtype, zeros in QUANTILE_PATH_CASES:
        time_quantile_paths(rows, width, dtype, extension, zeros)
    time_observe_breakdown(*BREAKDOWN_SHAPE)
    tuning_problems = []
    for rows, width in TUNING_SHAPES:
        tuning_problems.extend(time_kernel_vs_sum(rows, width, extension))
    for rows, width in OVERHEAD_SHAPES:
        time_observe_overhead(rows, width, extension)
    return tuning_problems


def time_kernel_vs_sum(rows: int, width: int, extension=None) -> list[str]:
    activations = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda") * ACTIVATION_SCALE
    sum_milliseconds = time_cuda(lambda: activations.sum(), TIMING_REPEATS)
    print(f"[{rows}, {width}] bf16: x.sum() {sum_milliseconds:.4f} ms/call; kernel timings include buffer reset")
    all_problems = []
    for detailed in (False, True):
        reference = run_reference(activations, detailed)
        label = "detailed" if detailed else "non-detailed"
        if extension is not None:
            all_problems.extend(_time_extension(activations, extension, reference, label, sum_milliseconds))
        all_problems.extend(_time_triton_configurations(activations, reference, label, sum_milliseconds))
    return all_problems


def _time_extension(
    activations: torch.Tensor, extension, reference: FakeRecorder, label: str, sum_milliseconds: float
) -> list[str]:
    extension_recorder = make_recorder(activations.device)

    def run_extension() -> None:
        # Reset between calls: repeated large populations would otherwise overflow int32 counters.
        extension_recorder.float_words.zero_()
        observe_statistics(extension, activations, extension_recorder.float_words, False)

    run_extension()
    torch.cuda.synchronize()
    problems = compare(f"{label} extension", reference, extension_recorder)
    extension_milliseconds = time_cuda(run_extension, TIMING_REPEATS)
    status = "ok" if not problems else "; ".join(problems)
    print(f"  {label:12s} extension        : {extension_milliseconds:.4f} ms/call "
          f"({extension_milliseconds / sum_milliseconds:.1f}x sum) {status}")
    return problems


def _time_triton_configurations(
    activations: torch.Tensor, reference: FakeRecorder, label: str, sum_milliseconds: float
) -> list[str]:
    all_problems = []
    for tile_elements in TRITON_TILE_ELEMENTS:
        for num_warps in TRITON_NUM_WARPS:
            recorder = make_recorder(activations.device)
            run_kernel = _kernel_launcher(activations, recorder, tile_elements, num_warps)
            run_kernel()
            torch.cuda.synchronize()
            problems = compare(f"{label} tile={tile_elements} warps={num_warps}", reference, recorder)
            all_problems.extend(problems)
            kernel_milliseconds = time_cuda(run_kernel, TIMING_REPEATS)
            status = "ok" if not problems else "; ".join(problems)
            print(
                f"  {label:12s} tile={tile_elements:5d} warps={num_warps}: {kernel_milliseconds:.4f} ms/call "
                f"({kernel_milliseconds / sum_milliseconds:.1f}x sum) {status}"
            )
    return all_problems


def _kernel_launcher(
    activations: torch.Tensor, recorder: FakeRecorder, tile_elements: int, num_warps: int
) -> Callable[[], None]:
    def run_kernel() -> None:
        recorder.float_words.zero_()
        launch_statistics_kernel(
            activations, recorder.float_words, tile_elements=tile_elements, num_warps=num_warps,
        )

    return run_kernel


def time_observe_overhead(rows: int, width: int, extension=None) -> None:
    # The first call of `observe` caches the point's slots; `time_cuda` synchronizes only around the whole batch, so
    # the time is the cost of issuing a call. Setting `stats._extension` picks the path without a build or load.
    activations = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda") * ACTIVATION_SCALE
    for detailed in (False, True):
        label = "detailed" if detailed else "non-detailed"
        for path_name, stats_extension in (("triton", False), ("extension", extension)):
            if path_name == "extension" and extension is None:
                continue
            recorder = make_recorder(activations.device)
            stats = ActivationStats(recorder)
            stats.begin_step(detailed)
            stats._extension = stats_extension
            observe_milliseconds = time_cuda(
                _observe_call(recorder, stats, detailed, activations), OBSERVE_TIMING_REPEATS, OBSERVE_TIMING_WARMUP
            )
            print(
                f"[{rows}, {width}] {label} {path_name} ActivationStats.observe including buffer/step reset: "
                f"{observe_milliseconds * MICROSECONDS_PER_MILLISECOND:.2f} us/call"
            )


def _observe_call(
    recorder: FakeRecorder, stats: ActivationStats, detailed: bool, activations: torch.Tensor
) -> Callable[[], None]:
    def run_observe() -> None:
        recorder.float_words.zero_()
        stats.begin_step(detailed)
        stats.observe("point", 0, activations)

    return run_observe


def time_quantile_paths(rows: int, width: int, dtype: torch.dtype, extension, zeros: bool = False) -> None:
    """Times a statistics call with its exact quantiles through the histogram of all 16-bit keys and
    through the refining path (the first digit counted, then refined), and the statistics
    alone for scale. `zeros` turns half of the elements into exact zeros, which put the most atomics on
    one counter."""
    activations = (torch.randn(rows, width, device="cuda") * ACTIVATION_SCALE).to(dtype)
    if zeros:
        activations = torch.where(
            torch.rand(rows, width, device="cuda") < ZEROED_FRACTION, activations, torch.zeros_like(activations)
        )
    words = torch.zeros(WORDS_WITH_QUANTILES, device="cuda")

    def statistics() -> None:
        words.zero_()
        observe_statistics(extension, activations, words, False)

    def with_quantiles(key_histogram: bool):
        def run() -> None:
            words.zero_()
            observe_with_quantiles(extension, activations, words, key_histogram)
        return run

    statistics_milliseconds = time_cuda(statistics, TIMING_REPEATS)
    refined_milliseconds = time_cuda(with_quantiles(False), TIMING_REPEATS)
    keys_milliseconds = time_cuda(with_quantiles(True), TIMING_REPEATS)
    print(f"[{rows}, {width}] {dtype}{' half zeros' if zeros else ''}: "
          f"statistics alone {statistics_milliseconds:.3f} ms; with quantiles "
          f"through the refining path {refined_milliseconds:.3f} ms, "
          f"through the keys' histogram {keys_milliseconds:.3f} ms "
          f"({refined_milliseconds - keys_milliseconds:.3f} ms, "
          f"{PERCENT * (refined_milliseconds - keys_milliseconds) / refined_milliseconds:.0f}% less)")


def time_observe_breakdown(rows: int, width: int) -> None:
    # Decomposes the Python-side cost of `observe` into its parts: the reshape, the grad and graph-task guards, the
    # cached-point lookup, mark_recorded, and the Triton kernel launch itself, each timed like `time_observe_overhead`.
    activations = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda") * ACTIVATION_SCALE
    recorder = make_recorder(activations.device)
    stats = ActivationStats(recorder)
    stats._extension = False
    stats.begin_step(False)
    stats.observe("point", 0, activations)  # warms up the point cache
    resolved = stats._points[("point", 0, 0)]
    matrix = activations.detach().reshape(-1, activations.shape[-1])

    parts = {
        "observe (full, Triton path)": lambda: stats.observe("point", 0, activations),
        "reshape only": lambda: activations.detach().reshape(-1, activations.shape[-1]),
        "grad-enabled + graph-task-id guards": lambda: (torch.is_grad_enabled(), torch._C._current_graph_task_id()),
        "point cache lookup": lambda: stats._points.get(("point", 0, 0)),
        "mark_recorded": lambda: recorder.mark_recorded(resolved.slot_indices),
        "kernel launch (Triton)": lambda: launch_statistics_kernel(matrix, recorder.float_words),
    }
    print(f"[{rows}, {width}] observe() breakdown:")
    for label, function in parts.items():
        elapsed_milliseconds = time_cuda(function, OBSERVE_TIMING_REPEATS, OBSERVE_TIMING_WARMUP)
        print(f"  {label:38s}: {elapsed_milliseconds * MICROSECONDS_PER_MILLISECOND:.2f} us/call")
