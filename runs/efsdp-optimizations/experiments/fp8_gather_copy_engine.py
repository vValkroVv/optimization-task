#!/usr/bin/env python3
"""Compare EP4 FP8 weight gathers with default and zero-CTA NCCL."""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
import torch.distributed as dist

from fp8_gather_coalescing import (
    LOCAL_DATA,
    LOCAL_SCALES,
    TORCH_REVISION,
    WORLD_SIZE,
    gather_separately,
    make_input,
    positive_int,
    verify_route,
    write_json,
)

NCCL_REVISION = (2, 30, 7)
ROUTES = ("ordinary_default", "symmetric_default", "symmetric_zero")
ORDER = (*ROUTES, *reversed(ROUTES))
PROXY_SHAPE = (2048, 4096, 3072)  # Routed rows, model dimension, w13 width.
PLANE_BYTES = (LOCAL_DATA, LOCAL_SCALES * torch.float32.itemsize)
CE_MARKER = re.compile(r"\bAllGather \[Copy Engine\]: (\d+) Bytes -> cudaMemcpy")
TIMING_FIELDS = (
    "host_gather_submit_us",
    "gather_span_us",
    "gemm_span_us",
    "joint_span_us",
    "two_stream_span_overlap_us",
)


def checked_markers(markers: dict) -> None:
    expected = {ROUTES[0]: [], ROUTES[1]: [], ROUTES[2]: sorted(PLANE_BYTES)}
    if markers != expected:
        raise RuntimeError(
            f"Unexpected NCCL copy-engine selection. Expected {expected}, got {markers}"
        )


def gather_markers(log: Path, offset: int) -> list[int]:
    with log.open("rb") as stream:
        stream.seek(offset)
        lines = stream.read().decode(errors="replace")
    return sorted(int(size) for size in CE_MARKER.findall(lines))


def runtime_identity(device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    tuning_environment = sorted(
        (key, value)
        for key, value in os.environ.items()
        if key.startswith(("NCCL_", "TORCH_NCCL_", "TORCH_SYMM_MEM_"))
        and not key.startswith("NCCL_DEBUG")
    )
    return {
        "host": platform.node(),
        "gpu": props.name,
        "gpu_uuid": str(props.uuid),
        "device_pair": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_revision": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "nccl_version": list(torch.cuda.nccl.version()),
        "nccl_environment_sha256": hashlib.sha256(
            json.dumps(tuning_environment).encode()
        ).hexdigest(),
        "plane_bytes": list(PLANE_BYTES),
    }


def memory_snapshot(device: torch.device) -> dict:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device_used_bytes": total - free,
        "torch_allocated_bytes": torch.cuda.memory_allocated(device),
        "torch_reserved_bytes": torch.cuda.memory_reserved(device),
    }


def prepare_routes(rank, device, stream, zero_group, probe_log):
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem.set_backend("NCCL")
    default_group = dist.group.WORLD
    allocations = {}
    handles = []
    markers = {}
    memory = {"before_planes": memory_snapshot(device)}
    with torch.cuda.stream(stream):
        expected = tuple(make_input(i, device) for i in range(WORLD_SIZE))
        for route, group in zip(
            ROUTES, (default_group, default_group, zero_group), strict=True
        ):
            empty = torch.empty if route == ROUTES[0] else symm_mem.empty
            source = (
                empty((LOCAL_DATA,), dtype=torch.uint8, device=device),
                empty((LOCAL_SCALES,), dtype=torch.float32, device=device),
            )
            output = (
                empty((WORLD_SIZE * LOCAL_DATA,), dtype=torch.uint8, device=device),
                empty((WORLD_SIZE * LOCAL_SCALES,), dtype=torch.float32, device=device),
            )
            if route != ROUTES[0]:
                for tensor in (*source, *output):
                    handle = symm_mem.rendezvous(tensor, group=group.group_name)
                    if handle is None:
                        raise RuntimeError(f"Symmetric rendezvous failed for {route}")
                    handles.append(handle)
            for tensor, fixture in zip(source, expected[rank], strict=True):
                tensor.copy_(fixture)
            stream.synchronize()
            if probe_log is not None:
                offset = probe_log.stat().st_size if probe_log.exists() else 0
            verify_route(gather_separately, output, source, group, expected)
            stream.synchronize()
            if probe_log is not None:
                markers[route] = gather_markers(probe_log, offset)
            allocations[route] = (output, source, group)
            memory[f"after_{route}_validation"] = memory_snapshot(device)
    del expected, fixture
    memory["all_routes_ready"] = memory_snapshot(device)
    return allocations, handles, markers, memory


class Sampler:
    def __init__(self, device, comm_stream):
        self.comm_stream = comm_stream
        self.compute_stream = torch.cuda.Stream(device=device)
        self.gate, self.comm_start, self.comm_end, self.gemm_start, self.gemm_end = (
            torch.cuda.Event(enable_timing=True) for _ in range(5)
        )
        rows, inner, width = PROXY_SHAPE
        with torch.cuda.stream(self.compute_stream):
            generator = torch.Generator(device=device).manual_seed(2048)
            self.inputs = torch.randn(
                (rows, inner), device=device, dtype=torch.bfloat16, generator=generator
            )
            self.weight = torch.randn(
                (width, inner), device=device, dtype=torch.bfloat16, generator=generator
            ).T
            self.result = torch.empty(
                (rows, width), device=device, dtype=torch.bfloat16
            )
        self.compute_stream.synchronize()

    def measure(self, route, *, gather: bool, gemm: bool) -> tuple:
        self.gate.record()
        submit_us = gather_us = gemm_us = joint_us = overlap_us = None
        if gather:
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(self.gate)
                self.comm_start.record()
                before_ns = perf_counter_ns()
                gather_separately(*route)
                submit_us = (perf_counter_ns() - before_ns) / 1_000
                self.comm_end.record()
        if gemm:
            with torch.cuda.stream(self.compute_stream):
                self.compute_stream.wait_event(self.gate)
                self.gemm_start.record()
                torch.mm(self.inputs, self.weight, out=self.result)
                self.gemm_end.record()
            self.gemm_end.synchronize()
            gemm_us = self.gemm_start.elapsed_time(self.gemm_end) * 1_000
        if gather:
            self.comm_end.synchronize()
            gather_us = self.comm_start.elapsed_time(self.comm_end) * 1_000
        if gather and gemm:
            gather_start = self.gate.elapsed_time(self.comm_start)
            gather_end = self.gate.elapsed_time(self.comm_end)
            compute_start = self.gate.elapsed_time(self.gemm_start)
            compute_end = self.gate.elapsed_time(self.gemm_end)
            joint_us = max(gather_end, compute_end) * 1_000
            overlap_us = (
                max(
                    0.0, min(gather_end, compute_end) - max(gather_start, compute_start)
                )
                * 1_000
            )
        return submit_us, gather_us, gemm_us, joint_us, overlap_us


def benchmark(args, sampler, allocations) -> list[dict]:
    blocks = []
    modes = {
        "gather_only": (True, False),
        "gemm_only": (False, True),
        "concurrent": (True, True),
    }
    for route in ORDER * args.repeats:
        block = {"route": route, "modes": {}}
        for mode, (gather, gemm) in modes.items():
            dist.barrier()
            for _ in range(args.warmup):
                sampler.measure(allocations[route], gather=gather, gemm=gemm)
            block["modes"][mode] = [
                sampler.measure(allocations[route], gather=gather, gemm=gemm)
                for _ in range(args.samples)
            ]
        blocks.append(block)
    return blocks


def summarize(ranks: list[dict]) -> list[dict]:
    block_statistics = []
    for first, second in zip(*(rank["blocks"] for rank in ranks), strict=True):
        if first["route"] != second["route"]:
            raise RuntimeError("Ranks measured different arms")
        block = {"route": first["route"], "modes": {}}
        for mode, samples in first["modes"].items():
            stats = {}
            peers = second["modes"][mode]
            for index, field in enumerate(TIMING_FIELDS):
                if samples[0][index] is None:
                    continue
                reduction = min if field == "two_stream_span_overlap_us" else max
                stats[field] = median(
                    reduction(left[index], right[index])
                    for left, right in zip(samples, peers, strict=True)
                )
            block["modes"][mode] = stats
        block_statistics.append(block)
    return block_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "bench"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--activation", type=Path)
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--samples", type=positive_int, default=12)
    parser.add_argument("--repeats", type=positive_int, default=2)
    args = parser.parse_args()
    if (args.mode == "bench") != (args.activation is not None):
        parser.error(
            "bench requires --activation pointing to a completed probe directory"
        )
    output = args.output_dir.resolve(strict=True)
    if (
        not output.is_dir()
        or any(output.glob("rank-*.json"))
        or (output / "summary.json").exists()
    ):
        parser.error("use a fresh output directory")
    if torch.version.git_version != TORCH_REVISION:
        raise RuntimeError(f"Torch revision {TORCH_REVISION} is required")
    if any(
        os.environ.get(key) != str(WORLD_SIZE)
        for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")
    ):
        raise RuntimeError("Use torchrun with two ranks on one node")
    rank = int(os.environ["RANK"])
    if rank != int(os.environ["LOCAL_RANK"]) or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Expose exactly two H100s to the two local ranks")
    if os.environ.get("NCCL_CTA_POLICY"):
        raise RuntimeError("Unset NCCL_CTA_POLICY so A and B use the default")
    probe_log = None
    if args.mode == "probe":
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "TUNING"
        os.environ["NCCL_DEBUG_FILE"] = str(output / "nccl-%p.log")
        if rank == 0:
            probe_log = output / f"nccl-{os.getpid()}.log"
    elif os.environ.get("NCCL_DEBUG", "").upper() in {"INFO", "TRACE"}:
        raise RuntimeError("Disable verbose NCCL logging for measured blocks")

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    identity = runtime_identity(device)
    if "H100" not in identity["gpu"] or not torch.cuda.can_device_access_peer(
        rank, 1 - rank
    ):
        raise RuntimeError("Use one NVLink-connected, peer-accessible H100 pair")
    if tuple(identity["nccl_version"]) != NCCL_REVISION:
        raise RuntimeError(f"NCCL {NCCL_REVISION} is required for the selection check")
    if args.activation is not None:
        probe = json.loads((args.activation / "summary.json").read_text())
        checked_markers(probe["markers"])
        if probe["rank_identities"][rank] != identity:
            raise RuntimeError("Probe and benchmark runtime identities differ")

    timeout = timedelta(seconds=45)
    dist.init_process_group("nccl", device_id=device, timeout=timeout)
    try:
        options = dist.ProcessGroupNCCL.Options()
        options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        zero_group = dist.new_group(
            ranks=list(range(WORLD_SIZE)),
            timeout=timeout,
            pg_options=options,
            group_desc="fp8_gather_zero_cta",
        )
        stream = torch.cuda.Stream(device=device)
        allocations, handles, markers, memory = prepare_routes(
            rank, device, stream, zero_group, probe_log
        )
        receipt = {
            "rank": rank,
            "identity": identity,
            "bitwise_three_routes": True,
            "logical_plane_bytes_per_route": sum(PLANE_BYTES) * (1 + WORLD_SIZE),
            "simultaneously_live_routes": len(allocations),
            "memory": memory,
        }
        if probe_log is not None:
            checked_markers(markers)
        if args.mode == "bench":
            sampler = Sampler(device, stream)
            memory["proxy_ready"] = memory_snapshot(device)
            torch.cuda.reset_peak_memory_stats(device)
            receipt["blocks"] = benchmark(args, sampler, allocations)
            memory["timing_complete"] = memory_snapshot(device)
            memory["timing_complete"].update(
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        write_json(output / f"rank-{rank}.json", receipt)
        dist.barrier()
        if rank == 0:
            ranks = [
                json.loads((output / f"rank-{peer}.json").read_text())
                for peer in range(WORLD_SIZE)
            ]
            summary = {
                "command": sys.argv,
                "rank_identities": [row["identity"] for row in ranks],
            }
            if args.mode == "probe":
                summary["markers"] = markers
            else:
                summary.update(
                    activation=str(args.activation.resolve()),
                    num_warmups_per_block=args.warmup,
                    num_samples_per_block=args.samples,
                    gemm_shape_mkn=PROXY_SHAPE,
                    sample_fields=TIMING_FIELDS,
                    block_medians=summarize(ranks),
                )
            write_json(output / "summary.json", summary)
        del handles, allocations
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
