from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import RankRecorder
from metric_sampler.capture.ring import ShmRing
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import ACTIVATION_COUNTS_WIDTH, ValueType
from metric_sampler.node.aggregator import NodeAggregator, RankRing

_RANKS_PER_NODE = 8
_RANK_TIMEOUT_SECONDS = 5.0
_RING_NAME_SUFFIX_LENGTH = 20
_ACTIVATION_MOMENTS_WIDTH = 4
_COUNT_LIMIT = 1000
_MICROSECONDS_PER_SECOND = 1e6
_MILLISECONDS_PER_SECOND = 1e3
_BYTES_PER_KIB = 1024
_PERCENT = 100
# Step durations (ms) the aggregator's cost per step is related to in the table.
_FAST_STEP_MS = 1000
_SLOW_STEP_MS = 4000

_COUNTS = metric_by_name("router/tokens_per_expert")
_DISPATCHER_METRICS = tuple(metric_by_name(name) for name in (
    "dispatcher/assignments_before_capacity", "dispatcher/valid_assignments_before_capacity",
    "dispatcher/assignments_after_capacity", "dispatcher/valid_assignments_after_capacity",
))
_SCORES = metric_by_name("router/scores_mean")
_BIAS = metric_by_name("router/expert_bias_before_update")
_GRAD_DENSE = metric_by_name("grad/pre_reduce_sq_dense")
_GRAD_EXPERT = metric_by_name("grad/pre_reduce_sq_expert")

_PER_MICROBATCH_SCALARS = [
    metric_by_name(name) for name in ("loss/total", "loss/ce", "loss/mtp", "loss/z_loss",
                                       "time/forward_gpu_s", "time/backward_gpu_s")
]
_PER_STEP_SCALARS = [
    metric_by_name(name) for name in (
        "time/step_s", "time/optimizer_gpu_s", "lr", "sampler/dropped_steps",
        "sampler/step_end_s", "grad/clip_total_norm", "memory/allocated_gb",
        "memory/peak_allocated_gb", "memory/reserved_gb", "memory/alloc_retries",
        "time/step_gpu_s", "sampler/microbatches",
    )
]
_ACTIVATION_POINTS = ("residual", "attn_out", "mlp_out")


@dataclass(frozen=True)
class BenchConfig:
    name: str
    layers: int
    experts: int
    microbatches: int
    activation_points: tuple[str, ...] = ()


_CONFIGS: dict[str, BenchConfig] = {
    "L8": BenchConfig("L8", layers=8, experts=256, microbatches=2),
    "L16": BenchConfig("L16", layers=16, experts=256, microbatches=2),
    "L61": BenchConfig("L61", layers=61, experts=256, microbatches=4),
    "L61+act": BenchConfig("L61+act", layers=61, experts=256, microbatches=4,
                            activation_points=_ACTIVATION_POINTS),
}


def _build_layout(config: BenchConfig) -> Layout:
    requests: list[tuple] = []
    for layer in range(config.layers):
        requests.append((_SCORES, layer, -1, config.experts))
        requests.append((_BIAS, layer, -1, config.experts))
        requests.append((_GRAD_DENSE, layer, -1, 1))
        requests.append((_GRAD_EXPERT, layer, -1, 1))
        for microbatch in range(config.microbatches):
            requests.append((_COUNTS, layer, microbatch, config.experts))
            for metric in _DISPATCHER_METRICS:
                requests.append((metric, layer, microbatch, config.experts))
            for point in config.activation_points:
                moments, counts, amax = (
                    metric_by_name(f"act/{point}/moments"),
                    metric_by_name(f"act/{point}/counts"),
                    metric_by_name(f"act/{point}/amax"),
                )
                requests.append((moments, layer, microbatch, _ACTIVATION_MOMENTS_WIDTH))
                requests.append((counts, layer, microbatch, ACTIVATION_COUNTS_WIDTH))
                requests.append((amax, layer, microbatch, 1))
    for microbatch in range(config.microbatches):
        for metric in _PER_MICROBATCH_SCALARS:
            requests.append((metric, -1, microbatch, 1))
    for metric in _PER_STEP_SCALARS:
        requests.append((metric, -1, -1, 1))
    return Layout(requests)


def _vector_sources(layout: Layout) -> dict[tuple[ValueType, int], torch.Tensor]:
    # One reusable tensor per (value_type, width), so allocation never dominates the timings
    # this benchmark is trying to isolate.
    sources: dict[tuple[ValueType, int], torch.Tensor] = {}
    for slot in layout.slots:
        if slot.width == 1:
            continue
        key = (slot.metric.value_type, slot.width)
        if key not in sources:
            sources[key] = (
                torch.rand(slot.width) if slot.metric.value_type is ValueType.FLOAT32
                else torch.randint(0, _COUNT_LIMIT, (slot.width,), dtype=torch.int32)
            )
    return sources


def _populate(recorder: RankRecorder, layout: Layout, sources: dict[tuple[ValueType, int], torch.Tensor]) -> None:
    for slot in layout.slots:
        if slot.metric.detailed:
            continue
        if slot.width == 1:
            recorder.record_host(slot.metric.name, slot.layer, slot.microbatch, 1.0)
        else:
            source = sources[(slot.metric.value_type, slot.width)]
            recorder.record(slot.metric.name, slot.layer, slot.microbatch, source)


@dataclass
class BenchResult:
    config: BenchConfig
    steps: int
    window_steps: int
    step_end_us: list[float] = field(default_factory=list)
    poll_ms: list[float] = field(default_factory=list)
    write_ms: list[float] = field(default_factory=list)
    window_bytes: list[int] = field(default_factory=list)


def run_config(config: BenchConfig, steps: int, window_steps: int, slot_count: int, output_dir: Path) -> BenchResult:
    layout = _build_layout(config)
    sources = _vector_sources(layout)
    rings = [
        ShmRing.create(f"ms_bench_{uuid.uuid4().hex[:_RING_NAME_SUFFIX_LENGTH]}", layout, slot_count)
        for _ in range(_RANKS_PER_NODE)
    ]
    recorders = [RankRecorder(layout, ring, torch.device("cpu")) for ring in rings]
    rank_rings = [RankRing(rank, ring) for rank, ring in enumerate(rings)]
    result = BenchResult(config, steps, window_steps)

    try:
        aggregator = NodeAggregator(
            rank_rings, output_dir, node="bench", attempt="bench", window_steps=window_steps,
            rank_timeout_s=_RANK_TIMEOUT_SECONDS, nodes_in_job=1, primary_node=True,
        )
        writer = aggregator._writer
        for step in range(1, steps + 1):
            for recorder in recorders:
                recorder.begin_step(False)
                _populate(recorder, layout, sources)
                start = time.perf_counter()
                recorder.step_end(step)
                result.step_end_us.append((time.perf_counter() - start) * _MICROSECONDS_PER_SECOND)

            start = time.perf_counter()
            aggregator._collector.collect_ready_slots()
            aggregator._settle_pending_steps()
            result.poll_ms.append((time.perf_counter() - start) * _MILLISECONDS_PER_SECOND)

            while writer.buffered_step_count >= window_steps:
                start = time.perf_counter()
                written = writer.write_window(window_steps)
                result.write_ms.append((time.perf_counter() - start) * _MILLISECONDS_PER_SECOND)
                result.window_bytes.append(sum(path.stat().st_size for path in written))

        if writer.buffered_step_count:
            start = time.perf_counter()
            written = writer.write_window(writer.buffered_step_count)
            result.write_ms.append((time.perf_counter() - start) * _MILLISECONDS_PER_SECOND)
            result.window_bytes.append(sum(path.stat().st_size for path in written))
    finally:
        for recorder in recorders:
            recorder.close()
        for ring in rings:
            ring.unlink()

    return result


def _profile_config(
    config: BenchConfig, steps: int, window_steps: int, slot_count: int, output_dir: Path, top_count: int,
) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    run_config(config, steps, window_steps, slot_count, output_dir)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(top_count)
    return stream.getvalue()


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _print_table(results: list[BenchResult]) -> None:
    header = (
        f"{'config':<8} {'layers':>6} {'experts':>7} {'mb':>3} {'act':>4} "
        f"{'step_end_us':>12} {'poll_ms':>8} {'write_ms':>9} {'window_KiB':>11} "
        f"{'cpu%@1s':>8} {'cpu%@4s':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        config = result.config
        step_end = _mean(result.step_end_us)
        poll = _mean(result.poll_ms)
        write = _mean(result.write_ms)
        window_kib = _mean(result.window_bytes) / _BYTES_PER_KIB
        amortized_write_per_step_ms = write / result.window_steps
        total_per_step_ms = poll + amortized_write_per_step_ms
        cpu_1s = _PERCENT * total_per_step_ms / _FAST_STEP_MS
        cpu_4s = _PERCENT * total_per_step_ms / _SLOW_STEP_MS
        print(
            f"{config.name:<8} {config.layers:>6} {config.experts:>7} {config.microbatches:>3} "
            f"{('yes' if config.activation_points else 'no'):>4} "
            f"{step_end:>12.1f} {poll:>8.2f} {write:>9.2f} {window_kib:>11.0f} "
            f"{cpu_1s:>7.2f}% {cpu_4s:>7.2f}%"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Node aggregator scale benchmark.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--window-steps", type=int, default=4)
    parser.add_argument("--slot-count", type=int, default=4)
    parser.add_argument("--configs", nargs="+", default=list(_CONFIGS), choices=list(_CONFIGS))
    parser.add_argument("--profile", action="store_true", help="cProfile the last listed config too")
    parser.add_argument("--profile-top", type=int, default=15)
    arguments = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="ms_bench_") as temporary_name:
        temporary_directory = Path(temporary_name)
        results = []
        for name in arguments.configs:
            print(f"running {name} ...", file=sys.stderr)
            config = _CONFIGS[name]
            results.append(run_config(
                config, arguments.steps, arguments.window_steps, arguments.slot_count, temporary_directory / name
            ))
        _print_table(results)

        if arguments.profile:
            profiled = arguments.configs[-1]
            print(f"\ncProfile for {profiled}:")
            report = _profile_config(
                _CONFIGS[profiled], arguments.steps, arguments.window_steps, arguments.slot_count,
                temporary_directory / f"{profiled}-profile", arguments.profile_top,
            )
            print(report)


if __name__ == "__main__":
    main()
