#!/usr/bin/env python3
"""Measure caching the gathered EP4 FP32 scales for each weight generation."""

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
import torch.distributed as dist

from fp8_gather_coalescing import (
    LOCAL_DATA,
    LOCAL_SCALES,
    METRICS,
    TORCH_REVISION,
    WORLD_SIZE,
    Planes,
    make_input,
    positive_int,
    verify_planes,
    write_json,
)

NCCL_REVISION = (2, 30, 7)
ROUTES = ("separate", "cached")
ORDER = (*ROUTES, *reversed(ROUTES))
PHASES = ("refresh", "hit")
NUM_FIXTURES = 2
TILE_ELEMENTS = 128 * 128


def change_input(planes: Planes, previous: int, tag: int) -> None:
    data, scales = planes
    # Exact binary increments preserve the distinct block and rank values.
    scales.add_((tag - previous) * 2**-16)
    data[:1].fill_(tag)
    data[-1:].fill_(255 - tag)


@dataclass
class Fixture:
    inputs: Planes
    cached_scales: torch.Tensor
    generation: int = 0
    gathered_generation: int = -1
    tag: int = 0

    def advance(self, tag: int) -> None:
        change_input(self.inputs, self.tag, tag)
        self.tag = tag
        self.generation += 1


def enqueue(
    route: str,
    shared: Planes,
    fixture: Fixture,
    group: dist.ProcessGroup,
    *,
    async_op: bool,
) -> tuple[Planes, tuple[dist.Work, ...], int]:
    cached = route == "cached"
    output = shared if not cached else (shared[0], fixture.cached_scales)
    send_scales = not cached or fixture.gathered_generation != fixture.generation
    count = 2 if send_scales else 1
    work = tuple(
        handle
        for receive, source in zip(output[:count], fixture.inputs[:count], strict=True)
        if (
            handle := dist.all_gather_into_tensor(
                receive, source, group=group, async_op=async_op
            )
        )
        is not None
    )
    if cached and send_scales:
        # The slot's release event gates reuse, including a discarded async gather.
        fixture.gathered_generation = fixture.generation
    return output, work, count


def expected_calls(route: str, phase: str) -> int:
    return 1 if route == "cached" and phase == "hit" else 2


def verify(
    fixtures: list[Fixture],
    shared: Planes,
    rank: int,
    device: torch.device,
    group: dist.ProcessGroup,
    submit: torch.cuda.Stream,
) -> None:
    consumer = torch.cuda.Stream(device=device)
    release = submit.record_event()
    with torch.cuda.stream(submit):
        remote_inputs = [
            make_input(index * WORLD_SIZE + (1 - rank), device)
            for index in range(NUM_FIXTURES)
        ]
        for fixture in fixtures:
            fixture.cached_scales.fill_(float("nan"))
    tag = 0
    for route in ROUTES:
        for async_refresh, discard_first in (
            (False, False),
            (True, False),
            (True, True),
        ):
            tag += 1
            for phase in PHASES:
                for index, (fixture, remote) in enumerate(
                    zip(fixtures, remote_inputs, strict=True)
                ):
                    # A discarded prefetch hands off to a synchronous refresh of the other fixture.
                    discard = discard_first and phase == "refresh" and index == 0
                    async_op = (
                        async_refresh
                        and phase == "refresh"
                        and (not discard_first or discard)
                    )
                    with torch.cuda.stream(submit):
                        submit.wait_event(release)
                        if phase == "refresh":
                            change_input(remote, fixture.tag, tag)
                            fixture.advance(tag)
                        output, work, count = enqueue(
                            route, shared, fixture, group, async_op=async_op
                        )
                        submitted = submit.record_event()
                    if count != expected_calls(route, phase) or len(
                        work
                    ) != count * int(async_op):
                        raise RuntimeError(
                            f"Unexpected collective count for {route} {phase}"
                        )
                    with torch.cuda.stream(consumer):
                        consumer.wait_event(submitted)
                        for handle in work:
                            handle.wait()
                        if not discard:
                            expected = (
                                (fixture.inputs, remote)
                                if rank == 0
                                else (remote, fixture.inputs)
                            )
                            verify_planes(output, expected)
                        release.record(consumer)
    submit.wait_event(release)
    submit.synchronize()


def measure(
    route: str,
    phase: str,
    fixture: Fixture,
    shared: Planes,
    group: dist.ProcessGroup,
    start: torch.cuda.Event,
    end: torch.cuda.Event,
) -> tuple[float, float]:
    start.record()
    before = perf_counter_ns()
    _, _, count = enqueue(route, shared, fixture, group, async_op=False)
    host_us = (perf_counter_ns() - before) / 1_000
    end.record()
    end.synchronize()
    if count != expected_calls(route, phase):
        raise RuntimeError(f"Unexpected collective count for {route} {phase}")
    return host_us, start.elapsed_time(end) * 1_000


def summarize(ranks: list[dict]) -> dict:
    blocks = []
    for first, second in zip(*(rank["blocks"] for rank in ranks), strict=True):
        if first["route"] != second["route"]:
            raise RuntimeError("Ranks measured different routes")
        row = {"route": first["route"]}
        for phase in PHASES:
            pairs = list(zip(first[phase], second[phase], strict=True))
            row[phase] = {
                metric: median(max(left[i], right[i]) for left, right in pairs)
                for i, metric in enumerate(METRICS)
            }
        blocks.append(row)
    return {
        "block_medians_of_samplewise_rank_maxima": blocks,
        "medians_of_block_medians": {
            route: {
                phase: {
                    metric: median(
                        row[phase][metric] for row in blocks if row["route"] == route
                    )
                    for metric in METRICS
                }
                for phase in PHASES
            }
            for route in ROUTES
        },
    }


def run_rank(args: argparse.Namespace, rank: int, device: torch.device) -> None:
    group = dist.group.WORLD
    stream = torch.cuda.Stream(device=device, priority=-1)
    with torch.cuda.stream(stream):
        inputs = [
            make_input(index * WORLD_SIZE + rank, device)
            for index in range(NUM_FIXTURES)
        ]
        shared = (
            torch.empty(WORLD_SIZE * LOCAL_DATA, dtype=torch.uint8, device=device),
            torch.empty(WORLD_SIZE * LOCAL_SCALES, dtype=torch.float32, device=device),
        )
        allocated_before = torch.cuda.memory_allocated(device)
        fixtures = [Fixture(source, torch.empty_like(shared[1])) for source in inputs]
        cache_allocated_bytes = torch.cuda.memory_allocated(device) - allocated_before
        verify(fixtures, shared, rank, device, group, stream)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        blocks = []
        for route in ORDER * args.repeats:
            dist.barrier()
            block = {"route": route, "refresh": [], "hit": []}
            for sweep in range(args.warmup + args.samples):
                # Repeat the same inputs in every arm. This setup is outside both timers.
                tag = (sweep + 7) % 256
                for fixture in fixtures:
                    fixture.advance(tag)
                for phase in PHASES:
                    for fixture in fixtures:
                        sample = measure(
                            route, phase, fixture, shared, group, start, end
                        )
                        if sweep >= args.warmup:
                            block[phase].append(sample)
            blocks.append(block)

    full_scale_bytes = shared[1].numel() * shared[1].element_size()
    write_json(
        args.output_dir / f"rank-{rank}.json",
        {
            "rank": rank,
            "host": platform.node(),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "torch_revision": torch.version.git_version,
            "cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "caller_stream_priority": stream.priority,
            "timing_async_op": False,
            "uint8_bytes_per_rank_per_fixture": LOCAL_DATA,
            "fp32_scales_per_rank_per_fixture": LOCAL_SCALES,
            "fixtures_per_sweep": NUM_FIXTURES,
            "collectives_per_unshard": {
                route: {phase: expected_calls(route, phase) for phase in PHASES}
                for route in ROUTES
            },
            "checks": ["sync_consumed", "async_consumed", "async_discarded_then_hit"],
            "benchmark_extra_live_scale_bytes_per_rank": NUM_FIXTURES
            * full_scale_bytes,
            "benchmark_cache_plane_allocator_delta_bytes_per_rank": cache_allocated_bytes,
            "ep4_separate_input_scale_delta_bytes_per_rank": (7 - 2) * full_scale_bytes,
            "warmup_sweeps_per_block": args.warmup,
            "measured_sweeps_per_block": args.samples,
            "sample_fields": METRICS,
            "blocks": blocks,
        },
    )
    dist.barrier()
    if rank == 0:
        ranks = [
            json.loads((args.output_dir / f"rank-{peer}.json").read_text())
            for peer in range(WORLD_SIZE)
        ]
        write_json(
            args.output_dir / "summary.json", {"command": sys.argv, **summarize(ranks)}
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument(
        "--samples", type=positive_int, default=12, help="sweeps per block"
    )
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--pg-timeout", type=positive_int, default=45)
    args = parser.parse_args()

    if LOCAL_DATA != LOCAL_SCALES * TILE_ELEMENTS:
        raise RuntimeError("Expected one FP32 scale per 128×128 FP8 tile")
    if torch.version.git_version != TORCH_REVISION:
        raise RuntimeError(
            f"Expected Torch {TORCH_REVISION}, found {torch.version.git_version}"
        )
    if any(
        os.environ.get(key) != str(WORLD_SIZE)
        for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")
    ):
        raise RuntimeError("Use torchrun with two ranks on one node")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Expose exactly two H100s to the two local ranks")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if "H100" not in torch.cuda.get_device_name(
        device
    ) or torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError(f"Rank {rank} requires an H100")
    if torch.cuda.nccl.version() != NCCL_REVISION:
        raise RuntimeError(
            f"Expected NCCL {NCCL_REVISION}, found {torch.cuda.nccl.version()}"
        )
    if not args.output_dir.is_dir() or any(
        (args.output_dir / file).exists()
        for file in ("rank-0.json", "rank-1.json", "summary.json")
    ):
        raise RuntimeError("Use an existing fresh output directory")
    dist.init_process_group(
        "nccl", device_id=device, timeout=timedelta(seconds=args.pg_timeout)
    )
    try:
        run_rank(args, rank, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
