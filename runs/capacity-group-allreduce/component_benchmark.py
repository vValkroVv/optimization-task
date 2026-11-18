#!/usr/bin/env python3
"""Bounded eager functional-NCCL counter diagnostic; launch with eight ranks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    ordered = sorted(values)
    return {
        "n": len(values),
        "min_us": ordered[0],
        "p10_us": ordered[round(0.10 * (len(values) - 1))],
        "median_us": statistics.median(values),
        "p90_us": ordered[round(0.90 * (len(values) - 1))],
        "p99_us": ordered[round(0.99 * (len(values) - 1))],
        "max_us": ordered[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--pairs", type=int, default=200)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = dist.new_group(ranks=list(range(8)), backend="nccl")
    args.output.mkdir(parents=True, exist_ok=True)

    # Integer counters isolate aggregation from tokenizer/data/model behavior.
    # The largest exact result stays far below int64 overflow and exceeds int32.
    counters = torch.arange(128, device="cuda", dtype=torch.int64) + rank * 19
    group_counts = counters.view(8, 16).sum(1)

    def reduced(value):
        return funcol.all_reduce(value, "sum", group).wait()

    def operation(scope, variant, value=counters, grouped=group_counts):
        if scope == "collective_only":
            return reduced(value if variant == "A" else grouped)
        if variant == "A":
            return reduced(value).view(8, 16).sum(1)
        return reduced(value.view(8, 16).sum(1))

    parity = []
    for name, values in (
        ("bounded_counts", counters),
        ("above_int32", counters + 2**31),
        ("empty_source_rank0", counters if rank else torch.zeros_like(counters)),
    ):
        a = operation("fragment", "A", values)
        b = operation("fragment", "B", values)
        torch.cuda.synchronize()
        assert torch.equal(a, b), name
        parity.append({"case": name, "group_loads": a.cpu().tolist(), "exact": True})

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    records = {}
    for scope in ("collective_only", "fragment"):
        for iteration in range(args.warmup):
            for variant in ("AB" if iteration % 2 == 0 else "BA"):
                operation(scope, variant)
        torch.cuda.synchronize()
        records[scope] = {variant: [] for variant in "AB"}
        for pair in range(args.pairs):
            for variant in ("AB" if pair % 2 == 0 else "BA"):
                # Equal arrivals and empty queues; both are outside the timer.
                dist.barrier(group=group)
                torch.cuda.synchronize()
                before = time.perf_counter_ns()
                start.record()
                result = operation(scope, variant)
                end.record()
                submitted = time.perf_counter_ns()
                end.synchronize()
                completed = time.perf_counter_ns()
                records[scope][variant].append({
                    "pair": pair,
                    "event_inclusive_us": start.elapsed_time(end) * 1000,
                    "host_submission_us": (submitted - before) / 1000,
                    "host_completion_wait_us": (completed - submitted) / 1000,
                    "host_total_us": (completed - before) / 1000,
                })
                # Keep the completed result live through synchronization.
                del result

    rank_path = args.output / f"rank{rank}.json"
    rank_path.write_text(json.dumps({"rank": rank, "records": records, "parity": parity}, indent=2) + "\n")
    dist.barrier(group=group)
    if rank == 0:
        root = Path(__file__).resolve().parents[2]
        all_ranks = [json.loads((args.output / f"rank{r}.json").read_text()) for r in range(8)]
        summary = {}
        for scope in records:
            summary[scope] = {}
            for metric in records[scope]["A"][0]:
                if metric == "pair":
                    continue
                per_variant = {}
                maxima = {}
                for variant in "AB":
                    samples = [[sample[metric] for sample in r["records"][scope][variant]] for r in all_ranks]
                    maxima[variant] = [max(values) for values in zip(*samples)]
                    per_variant[variant] = {
                        "per_rank": {str(i): distribution(values) for i, values in enumerate(samples)},
                        "max_rank_per_pair": distribution(maxima[variant]),
                    }
                per_variant["paired_max_rank_B_minus_A"] = distribution([
                    b - a for a, b in zip(maxima["A"], maxima["B"])
                ])
                summary[scope][metric] = per_variant
        payload = {
            "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(), "python": os.sys.executable, "torch": torch.__version__,
            "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(), "ranks": list(range(8)),
            "group": "NCCL ProcessGroup ranks 0..7, matching native EP8 rank membership",
            "dtype": "int64", "expert_count": 128, "ep_size": 8,
            "payload_bytes": {"A": 1024, "B": 64},
            "warmup_pairs_per_scope": args.warmup, "measured_pairs_per_scope": args.pairs,
            "pair_order": "AB on even pairs, BA on odd pairs",
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "source_sha256": {str(Path(__file__).relative_to(root)): sha256(__file__),
                "installed_functional_collectives": sha256(funcol.__file__)},
            "raw_samples": [{"path": str(args.output / f"rank{r}.json"), "sha256": sha256(args.output / f"rank{r}.json")} for r in range(8)],
            "parity": parity,
            "summary": summary,
            "scope": {
                "execution": "eager functional all_reduce then AsyncCollectiveTensor.wait; no model/compile/SAC timing",
                "event_inclusive_us": "CUDA events enclose functional collective and explicit wait; fragment also includes the relocated int64 sum. Includes host submission gaps, NCCL communication and device peer waits; not isolated kernel compute time.",
                "host_submission_us": "perf_counter through event recording and operation enqueue; Python/dispatch/allocation and possible wrapper overhead included",
                "host_completion_wait_us": "host time in final CUDA end-event synchronize; overlaps device work, not additive to event span",
                "host_total_us": "host submission plus completion wait, actual completion included",
                "excluded": "process setup, inputs, initial exact parity, 50 warmup pairs, pre-sample NCCL barrier and device synchronization",
                "unavailable": "separate device compute/kernel and communication duration attribution, production fusion/compile behavior and full-training throughput",
                "data": "synthetic counter-only diagnostic independent of token values; not a production-data/model benchmark",
                "memory": "tiny eager allocations included in submission; no production memory claim",
            },
        }
        destination = Path(__file__).with_name("component-results.json")
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        rows = []
        for scope in records:
            stats = summary[scope]["event_inclusive_us"]
            rows.append(f"| {scope} | {stats['A']['max_rank_per_pair']['median_us']:.3f} | {stats['B']['max_rank_per_pair']['median_us']:.3f} | {stats['paired_max_rank_B_minus_A']['median_us']:+.3f} |")
        print("\n".join(rows), flush=True)
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
