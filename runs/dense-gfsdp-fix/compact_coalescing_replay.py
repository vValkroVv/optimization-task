"""Verify and measure the source-local native embedding gradient on the pinned GPU.

Only the ATen backward and, for a full table, the unique-row gather are timed.
Routing, NCCL payload transport, owner accumulation and native training are
outside this measurement. A seeded BF16 derivative is shared by all variants.
"""

from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from statistics import median
import argparse
import json
import runpy
import time

import torch
from torch import distributed as dist

from row_owner_route_numeric import SOURCE_ROOT, WORLD, check_runtime


VOCAB, WIDTH, TOKENS, SEED = 128256, 4096, 8192, 2048
UNIQUE_COUNTS = (1, 12, 54, 55, 131, 132)
VARIANTS = ("full", "guarded", "padded")


def choose_rows(variant, n, u, width, sms):
    if variant == "full":
        return VOCAB, True
    if n <= 3072:
        return u, False
    stride_warped = (width + 31) // 32 * 32
    block = min(stride_warped, 1024)
    # Invert ceil(min(N, rows) * stride_warped / block) >= 4 * SMs.
    minimum = ((4 * sms - 1) * block) // stride_warped + 1
    if u >= minimum:
        return u, False
    if variant == "padded" and min(n, VOCAB) >= minimum:
        return minimum, False
    return VOCAB, True


def coalesce(grad, ids, unique, inverse, send, rows, full):
    indices = ids if full else inverse
    result = torch.ops.aten.embedding_dense_backward.default(
        grad, indices, rows, -1, False
    )
    if full:
        return torch.index_select(result, 0, unique, out=send)
    return result.narrow(0, 0, unique.numel())


def synthetic(n, u, seed):
    # Spread labels over all eight owners before repeating the first label.
    labels = torch.arange(u, dtype=torch.int64)
    labels = (labels % WORLD) * (VOCAB // WORLD) + labels // WORLD
    counts = torch.ones(u, dtype=torch.int64)
    if u >= 7:
        counts[1:7] = torch.tensor([9, 10, 11, 19, 20, 21])
    if u:
        counts[0] += n - int(counts.sum())
    ids = torch.repeat_interleave(labels, counts)
    return ids[torch.randperm(n, generator=torch.Generator().manual_seed(seed))]


def low_fixture(census, data, rank):
    import numpy as np
    from streaming.base.local import LocalDataset

    copies = [json.loads((census / f"rank{i}.json").read_text()) for i in range(WORLD)]
    if any(
        report["rank"] != i or report["depth"] != 0 or report["status"] != "PASS"
        for i, report in enumerate(copies)
    ):
        raise ValueError("expected the eight passed depth0 loader rank receipts")
    length = len(copies[0]["records"])
    if not length or any(len(report["records"]) != length for report in copies):
        raise ValueError("loader receipts differ in microbatch count")
    minimum, mb, source = min(
        (item["unique_token_ids"], index, report["rank"])
        for report in copies
        for index, item in enumerate(report["records"])
    )
    if minimum >= 132:
        raise ValueError(
            "the supplied production fixture does not exercise the low-U guard"
        )
    record = copies[rank]["records"][mb]
    dataset = LocalDataset(local=str(data))
    rows = []
    for index, digest in zip(record["row_indices"], record["row_hashes"], strict=True):
        raw = dataset[index]["tokens"]
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f"MDS row {index} differs from its loader receipt")
        rows.append(torch.from_numpy(np.frombuffer(raw, dtype="<u4").astype(np.int64)))
    ids = torch.stack(rows)
    if ids.shape != (2, 4096) or ids.unique().numel() != record["unique_token_ids"]:
        raise ValueError(
            "real low-U fixture differs from preserved shape or unique count"
        )
    return ids, {
        "microbatch_zero_based": mb,
        "lowest_u": minimum,
        "lowest_u_rank": source,
        "census_sha256": sha256((census / f"rank{rank}.json").read_bytes()).hexdigest(),
        "row_indices": record["row_indices"],
        "row_hashes": record["row_hashes"],
    }


def prepare(cpu_ids, width, seed, cancellation):
    cpu_ids = cpu_ids.reshape(-1)
    ids = cpu_ids.cuda()
    unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    grad = torch.randn(
        (ids.numel(), width),
        generator=torch.Generator(device="cuda").manual_seed(seed),
        device="cuda",
        dtype=torch.bfloat16,
    )
    cancellation_row = None
    if cancellation:
        positions = torch.where(cpu_ids == 0)[0].cuda()
        grad[positions, 0] = 0
        # For the ordered large-N path these land in four consecutive
        # ten-token partials: float32 addition of 2**24 + 1 loses the 1.
        grad[positions[:40:10], 0] = grad.new_tensor([2**24, 1, -(2**24), 1])
        cancellation_row = 0  # zero is the first sorted row label
    send = grad.new_empty((unique.numel(), width))
    return (grad, ids, unique, inverse, send), cancellation_row


def verify(inputs, cancellation_row, configs):
    grad, _, unique, _, _ = inputs
    reference = coalesce(*inputs, *configs["full"]).clone()
    cancellation_value = (
        None if cancellation_row is None else reference[cancellation_row, 0].item()
    )
    matches = {}
    for variant in VARIANTS:
        result = coalesce(*inputs, *configs[variant])
        if result.dtype != torch.bfloat16 or result.shape != (
            unique.numel(),
            grad.shape[-1],
        ):
            raise RuntimeError("coalesced payload changed dtype or shape")
        words_changed = torch.count_nonzero(
            result.view(torch.int16) != reference.view(torch.int16)
        ).item()
        matches[variant] = {
            "identical_bytes": words_changed == 0,
            "different_bf16_words": words_changed,
        }
    return matches, cancellation_value


def measure(inputs, config, warmup, samples):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    for _ in range(warmup):
        coalesce(*inputs, *config)
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    records = []
    for _ in range(samples):
        torch.cuda.reset_peak_memory_stats()
        starting = torch.cuda.memory_allocated()
        start.record()
        begin = time.perf_counter()
        output = coalesce(*inputs, *config)
        submitted = time.perf_counter()
        end.record()
        end.synchronize()
        finished = time.perf_counter()
        records.append(
            {
                "synchronized_wall_ms": (finished - begin) * 1000,
                "cuda_event_span_ms": start.elapsed_time(end),
                "host_call_ms": (submitted - begin) * 1000,
                "starting_allocated_bytes": starting,
                "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated()
                - starting,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        del output
    return records


def profile_low_u(inputs, configs, output, rank):
    traces = {}
    for variant in VARIANTS:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            coalesce(*inputs, *configs[variant])
            torch.cuda.synchronize()
        path = output / f"synthetic-u12-{variant}-rank{rank}.trace.json"
        profiler.export_chrome_trace(str(path))
        kernels = [
            row["name"]
            for row in json.loads(path.read_text())["traceEvents"]
            if row.get("cat") == "kernel"
        ]
        if not any("sum_and_scatter" in name for name in kernels) or any(
            "atomic_accumulate" in name for name in kernels
        ):
            raise RuntimeError(
                f"{path.name} lacked the expected ordered ATen dispatcher evidence"
            )
        traces[variant] = {"file": path.name, "kernels": kernels}
    return traces


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(args, rank):
    reference = json.loads(
        Path(__file__).with_name("embedding-replay-result.json").read_text()
    )
    loader = runpy.run_path(
        str(SOURCE_ROOT / "runs/production-data-abcd/component_fixtures.py")
    )["load_fixture"]
    captured, receipt = loader(args.high_fixture, rank)
    if receipt["input_ids_int64_sha256"] != reference["input_ids_sha256_by_rank"][rank]:
        raise ValueError("real high-U input differs from the accepted capture")
    pending = [
        (
            "real-high-u",
            captured.pop("input_ids"),
            WIDTH,
            False,
            {"capture_tensors_sha256": receipt["tensors_sha256"]},
        )
    ]
    del captured
    if args.low_census:
        ids, provenance = low_fixture(args.low_census, args.data_root, rank)
        pending.append(("real-low-u", ids, WIDTH, False, provenance))
    for unique in UNIQUE_COUNTS:
        pending.append(
            (
                f"synthetic-u{unique}",
                synthetic(TOKENS, unique, SEED + rank),
                WIDTH,
                True,
                {},
            )
        )
    for n, u in ((0, 0), (3072, 12), (3073, 12), (8192, 12)):
        pending.append((f"edge-n{n}-d37", synthetic(n, u, SEED + rank), 37, False, {}))

    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if sms != 132 or torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "expected 132-SM H100 and the original ATen determinism default"
        )
    cases = {}
    for name, ids, width, cancellation, provenance in pending:
        inputs, row = prepare(ids, width, SEED + rank, cancellation)
        u = inputs[2].numel()
        configs = {
            variant: choose_rows(variant, ids.numel(), u, width, sms)
            for variant in VARIANTS
        }
        checks, cancellation_value = verify(inputs, row, configs)
        cases[name] = {
            "tokens": ids.numel(),
            "unique": u,
            "width": width,
            "input_ids_int64_sha256": sha256(
                ids.contiguous().numpy().tobytes()
            ).hexdigest(),
            "seed": SEED + rank,
            "provenance": provenance,
            "actual_output_rows": {
                variant: config[0] for variant, config in configs.items()
            },
            "used_full_table": {
                variant: config[1] for variant, config in configs.items()
            },
            "checks": checks,
            "expected_cancellation_bf16": 1 if cancellation else None,
            "observed_cancellation_bf16": cancellation_value,
        }
        del inputs
    parity = [None] * WORLD
    dist.all_gather_object(parity, cases)
    if rank == 0:
        write(args.output / "parity.json", {"ranks": parity})
    if any(
        not check["identical_bytes"]
        for report in parity
        for case in report.values()
        for check in case["checks"].values()
    ):
        raise RuntimeError(
            "BF16 source-local gradient changed before timing; see parity.json"
        )
    if any(
        case["expected_cancellation_bf16"] != case["observed_cancellation_bf16"]
        for report in parity
        for case in report.values()
    ):
        raise RuntimeError(
            "unexpected ordered ATen cancellation result; see parity.json"
        )

    for index, (name, ids, width, cancellation, _) in enumerate(pending):
        if width != WIDTH:
            continue  # verify edge cases without publishing tiny-call timings
        inputs, _ = prepare(ids, width, SEED + rank, cancellation)
        u = inputs[2].numel()
        configs = {
            variant: choose_rows(variant, ids.numel(), u, width, sms)
            for variant in VARIANTS
        }
        order = VARIANTS[index % 3 :] + VARIANTS[: index % 3]
        cases[name]["order"] = order
        cases[name]["samples"] = {
            variant: measure(inputs, configs[variant], args.warmup, args.samples)
            for variant in order
        }
        del inputs
    if args.profile:
        inputs, _ = prepare(
            synthetic(TOKENS, 12, SEED + rank), WIDTH, SEED + rank, True
        )
        configs = {
            variant: choose_rows(variant, TOKENS, 12, WIDTH, sms)
            for variant in VARIANTS
        }
        cases["synthetic-u12"]["traces"] = profile_low_u(
            inputs, configs, args.output, rank
        )
    write(
        args.output / f"rank{rank}.json",
        {
            "rank": rank,
            "cases": cases,
            "gpu": torch.cuda.get_device_name(),
            "torch_git": torch.version.git_version,
            "cuda": torch.version.cuda,
        },
    )
    all_ranks = [None] * WORLD
    dist.all_gather_object(all_ranks, cases)
    if rank == 0:
        summary = {}
        for name, case in cases.items():
            if "samples" not in case:
                continue
            summary[name] = {
                "unique_by_rank": [row[name]["unique"] for row in all_ranks],
                "variants": {},
            }
            for variant in VARIANTS:
                records = [row[name]["samples"][variant] for row in all_ranks]
                metrics = {}
                for metric in records[0][0]:
                    series = [
                        max(sample[metric] for sample in group)
                        for group in zip(*records, strict=True)
                    ]
                    metrics[metric] = {
                        "rank_max_samples": series,
                        "median": median(series),
                        "max": max(series),
                    }
                summary[name]["variants"][variant] = metrics
        write(
            args.output / "summary.json",
            {
                "source_op_only": True,
                "bf16_parity_all_ranks": True,
                "real_low_u": "measured"
                if args.low_census
                else "fixture paths not supplied",
                "profiled_synthetic_u12": args.profile,
                "timing": summary,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--high-fixture", type=Path, required=True)
    parser.add_argument("--low-census", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if (args.low_census is None) != (args.data_root is None):
        parser.error("real low-U replay requires both --low-census and --data-root")
    if not (1 <= args.warmup <= 10 and 2 <= args.samples <= 10):
        parser.error("require 1–10 warmups and 2–10 samples")
    rank = check_runtime()
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        run(args, rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
