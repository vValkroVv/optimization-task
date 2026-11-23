#!/usr/bin/env python3
"""Compare the two typed EP4 weight gathers with pinned c10d coalescing."""

import argparse
import json
import os
import platform
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _coalescing_manager

TORCH_REVISION = "8b353065ce260336c5fdfa74d754e8bd9b91b062"
WORLD_SIZE = 2
LOCAL_DATA = 301_989_888
LOCAL_SCALES = 18_432
SEED = 2048
ABBA = ("separate", "coalesced", "coalesced", "separate")
METRICS = ("host_submit_us", "device_span_us")
Planes = tuple[torch.Tensor, torch.Tensor]


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def make_input(rank: int, device: torch.device) -> Planes:
    generator = torch.Generator(device=device).manual_seed(SEED + rank)
    data = torch.randint(
        0, 256, (LOCAL_DATA,), dtype=torch.uint8, device=device, generator=generator
    )
    # All values are distinct across shards and represented exactly in FP32.
    first = rank * LOCAL_SCALES + 1
    scales = torch.arange(
        first, first + LOCAL_SCALES, dtype=torch.float32, device=device
    ).mul_(2**-16)
    return data, scales


def gather_separately(output: Planes, source: Planes, group: dist.ProcessGroup) -> None:
    for receive, send in zip(output, source, strict=True):
        dist.all_gather_into_tensor(receive, send, group=group, async_op=False)


def gather_coalesced(output: Planes, source: Planes, group: dist.ProcessGroup) -> None:
    # device=None selects the pinned all_gather_single_coalesced path.
    with _coalescing_manager(group=group, async_ops=False):
        gather_separately(output, source, group)


Gather = Callable[[Planes, Planes, dist.ProcessGroup], None]
ROUTES: dict[str, Gather] = {
    "separate": gather_separately,
    "coalesced": gather_coalesced,
}


def verify_planes(output: Planes, expected_shards: tuple[Planes, ...]) -> None:
    for rank, (data, scales) in enumerate(expected_shards):
        data_start = rank * LOCAL_DATA
        scale_start = rank * LOCAL_SCALES
        if not torch.equal(output[0].narrow(0, data_start, LOCAL_DATA), data):
            raise RuntimeError(f"FP8 payload differs in source rank {rank}'s slice")
        # Compare scale representations, including the sign bit.
        actual_bits = output[1].narrow(0, scale_start, LOCAL_SCALES).view(torch.int32)
        if not torch.equal(actual_bits, scales.view(torch.int32)):
            raise RuntimeError(f"FP32 scales differ in source rank {rank}'s slice")


def verify_route(
    gather: Gather,
    output: Planes,
    source: Planes,
    group: dist.ProcessGroup,
    expected_shards: tuple[Planes, ...],
) -> None:
    # Start each correctness check from independent sentinel output.
    output[0].fill_(0)
    output[1].fill_(float("nan"))
    gather(output, source, group)
    verify_planes(output, expected_shards)


def measure_pair(
    gather: Gather,
    output: Planes,
    source: Planes,
    group: dist.ProcessGroup,
    stream: torch.cuda.Stream,
    start: torch.cuda.Event,
    end: torch.cuda.Event,
) -> tuple[float, float]:
    start.record(stream)
    before_ns = perf_counter_ns()
    gather(output, source, group)
    submit_us = (perf_counter_ns() - before_ns) / 1_000
    end.record(stream)
    end.synchronize()
    return submit_us, start.elapsed_time(end) * 1_000


def write_json(path: Path, payload: dict) -> None:
    with path.open("x") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")


def summarize(ranks: list[dict]) -> dict:
    """Compare the slower rank span for matching samples in each ABBA block."""
    by_block = []
    for left, right in zip(*(rank["blocks"] for rank in ranks), strict=True):
        if left["route"] != right["route"]:
            raise RuntimeError("Ranks measured different arms")
        row = {"route": left["route"]}
        for index, metric in enumerate(METRICS):
            row[metric] = median(
                max(first[index], second[index])
                for first, second in zip(left["samples"], right["samples"], strict=True)
            )
        by_block.append(row)
    return {
        "block_medians_of_samplewise_rank_maxima": by_block,
        "medians_of_block_medians": {
            route: {
                metric: median(row[metric] for row in by_block if row["route"] == route)
                for metric in METRICS
            }
            for route in ROUTES
        },
    }


def run_rank(args: argparse.Namespace, rank: int, device: torch.device) -> None:
    group = dist.group.WORLD
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        expected_shards = tuple(make_input(i, device) for i in range(WORLD_SIZE))
        source = expected_shards[rank]
        # Both arms borrow the same independently allocated typed receive planes.
        output = (
            torch.empty(WORLD_SIZE * LOCAL_DATA, dtype=torch.uint8, device=device),
            torch.empty(WORLD_SIZE * LOCAL_SCALES, dtype=torch.float32, device=device),
        )
        for name, gather in ROUTES.items():
            try:
                verify_route(gather, output, source, group, expected_shards)
            except RuntimeError as error:
                raise RuntimeError(f"{name} correctness check failed") from error
        del expected_shards
        stream.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        blocks = []
        for route in ABBA * args.repeats:
            dist.barrier()
            gather = ROUTES[route]
            for _ in range(args.warmup):
                measure_pair(gather, output, source, group, stream, start, end)
            samples = [
                measure_pair(gather, output, source, group, stream, start, end)
                for _ in range(args.samples)
            ]
            blocks.append({"route": route, "samples": samples})

    receipt = {
        "rank": rank,
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_revision": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "nccl_version": torch.cuda.nccl.version(),
        "world_size": WORLD_SIZE,
        "process_group_backend": dist.get_backend(group),
        "async_op": False,
        "local_bytes_uint8": LOCAL_DATA,
        "local_elements_float32": LOCAL_SCALES,
        "data_seed": SEED + rank,
        "verify_bitwise_both_routes": True,
        "warmups_per_block": args.warmup,
        "samples_per_block": args.samples,
        "sample_fields": METRICS,
        "blocks": blocks,
    }
    write_json(args.output_dir / f"rank-{rank}.json", receipt)
    dist.barrier()
    if rank == 0:
        ranks = [
            json.loads((args.output_dir / f"rank-{peer}.json").read_text())
            for peer in range(WORLD_SIZE)
        ]
        write_json(
            args.output_dir / "summary.json",
            {"command": sys.argv, **summarize(ranks)},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--samples", type=positive_int, default=20)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--pg-timeout", type=positive_int, default=45)
    args = parser.parse_args()

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
