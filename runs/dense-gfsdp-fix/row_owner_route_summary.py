"""Validate and summarize the exact, ABAB, and trace row-owner evidence."""

from hashlib import sha256
from pathlib import Path
from statistics import median
import argparse
import json
import math
import re


ARMS = ("A1", "B1", "A2", "B2", "diag-A", "diag-B")
RANKS = list(range(8))
SEED = 2048
PINNED = "2.15.0.dev20260822+cu130"
REVISION = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"
METRICS = (
    "synchronized_wall_ms",
    "cuda_elapsed_span_ms",
    "host_call_ms",
    "host_end_sync_ms",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)
ACTIVATION = "Experimental fixed-shape row-owner route active: sha256="
CASES = {"initial", "accumulated", "reset"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def check_fixture(fixture, expected):
    receipts = [read(fixture / f"rank{rank}/receipt.json") for rank in RANKS]
    require([row["rank"] for row in receipts] == RANKS, "fixture rank mismatch")
    require(
        [row["input_ids_int64_sha256"] for row in receipts] == expected,
        "fixture differs from embedding-replay-001",
    )


def completed(root, arm):
    require((root / f"{arm}.exit-code").read_text().strip() == "0", f"{arm} failed")
    stdout = (root / f"{arm}.log").read_text(errors="replace")
    require((root / f"{arm}.command").stat().st_size > 0, f"{arm} lacks argv")
    return stdout


def check_exact(root, expected):
    unit_log = completed(root, "owner-tests")
    tested = re.findall(
        r"row_owner_checks rank=(\d+) backend=nccl mesh_device=cuda", unit_log
    )
    require(
        sorted(map(int, tested)) == RANKS, "missing reused NCCL owner-test receipts"
    )
    completed(root, "exact")
    report = read(root / "exact.json")
    rows = report["ranks"]
    require(
        report["exact"] and [row["rank"] for row in rows] == RANKS,
        "incomplete exact replay",
    )
    require(
        report["seed"] == SEED and report["torch"] == PINNED,
        "exact replay runtime drift",
    )
    require(
        [row["input_ids_int64_sha256"] for row in rows] == expected,
        "exact replay IDs differ",
    )
    for row in rows:
        control, candidate = row["control"], row["candidate"]
        require(
            row["exact"] and control == candidate, f"rank {row['rank']} exact mismatch"
        )
        require(set(control["passes"]) == CASES, f"rank {row['rank']} missing F+B pass")
        for result in control["passes"].values():
            messages = result["messages"]
            require(
                result["gradient_fp32_bytes"] == 262_668_288,
                "unexpected gradient bytes",
            )
            for direction, splits in (("sent", "send"), ("received", "receive")):
                counts = messages[f"{splits}_split_rows"]
                require(len(counts) == 8 and min(counts) >= 0, "invalid NCCL splits")
                require(
                    messages[f"{direction}_bytes"] == 8192 * sum(counts),
                    "unexpected BF16 payload",
                )
                require(
                    len(messages[f"{direction}_bf16_sha256"]) == 64,
                    "missing BF16 message digest",
                )
    return [row["fixture_tensors_sha256"] for row in rows]


def read_arm(root, arm, expected, fixture_hashes):
    fixed = arm.startswith("B") or arm == "diag-B"
    profile = arm.startswith("diag-")
    scope = "stages" if profile else "integrated"
    count = 1 if profile else 10
    log = completed(root, arm)
    activation = re.findall(rf"{re.escape(ACTIVATION)}([0-9a-f]{{64}})", log)
    require(len(activation) == (8 if fixed else 0), f"{arm} has wrong activation count")
    if fixed:
        source = Path(__file__).with_name("row_owner_route_experiment.py")
        require(
            set(activation) == {sha256(source.read_bytes()).hexdigest()},
            f"{arm} route hash drift",
        )
    rows = read(root / arm / f"owner-{scope}-all-ranks.json")["ranks"]
    require([row["rank"] for row in rows] == RANKS, f"{arm} has incomplete ranks")
    for row in rows:
        rank = row["rank"]
        require(
            row["execution"] and row["activation"] == "RowOwnerEmbedding",
            f"{arm} inactive owner",
        )
        require(
            row["scope"] == scope and row["cases"] == ["fixture"], f"{arm} scope drift"
        )
        require(
            row["mesh_ranks"] == RANKS and row["seed"] == SEED,
            f"{arm} topology or seed drift",
        )
        require(
            row["source"]["revision"] == REVISION and row["runtime"]["torch"] == PINNED,
            f"{arm} runtime drift",
        )
        require(
            row["warmup"] == 10 and row["samples"] == count, f"{arm} sample count drift"
        )
        require(
            ("--profile" in row["command"]) == profile, f"{arm} profiler flag drift"
        )
        require(
            ("--fixed-shape-route" in row["command"]) == fixed,
            f"{arm} route flag drift",
        )
        receipt = row["input_fixture"]
        (result,) = row["results"]
        require(
            receipt["rank"] == rank
            and receipt["tensors_sha256"] == fixture_hashes[rank],
            f"{arm} capture drift",
        )
        require(
            result["owner_route_census"]["input_ids_int64_sha256"] == expected[rank],
            f"{arm} rank {rank} input mismatch",
        )
        require(
            row["gfsdp_gradient_buffer_inventory"][
                "owner_scratch_aliases_private_lm_head"
            ],
            f"{arm} scratch changed",
        )
        require(
            [sample["sample"] for sample in result["measurements"]]
            == list(range(count)),
            f"{arm} missing samples",
        )
    return rows


def integrated(rows):
    samples = [row["results"][0]["measurements"] for row in rows]
    metrics = {}
    for metric in METRICS:
        values = [sample["complete"][metric] for rank in samples for sample in rank]
        require(
            all(math.isfinite(value) and value >= 0 for value in values),
            f"invalid {metric} sample",
        )
        series = [
            max(sample["complete"][metric] for sample in same_step)
            for same_step in zip(*samples, strict=True)
        ]
        metrics[metric] = {
            "rank_max_samples": series,
            "median": median(series),
            "max": max(series),
        }
    return metrics


def trace_scopes(path):
    events = [event for event in read(path)["traceEvents"] if event.get("ph") == "X"]
    runtime = [
        event for event in events if event.get("cat") in {"cuda_runtime", "cuda_driver"}
    ]
    device = [event for event in events if event.get("cat") in {"kernel", "gpu_memcpy"}]
    scopes = {}
    for name in (
        "partition_ids",
        "counts",
        "request_exchange",
        "value_exchange",
        "gradient_exchange",
    ):
        stages = [
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and event["name"] == f"components::{name}"
        ]
        require(len(stages) == 1, f"{path.name} missing or repeated {name} annotation")
        stage = stages[0]
        calls = [
            event
            for event in runtime
            if event["pid"] == stage["pid"]
            and event["tid"] == stage["tid"]
            and stage["ts"] <= event["ts"] < stage["ts"] + stage["dur"]
        ]
        links = {
            str(event["args"]["correlation"])
            for event in calls
            if "correlation" in event.get("args", {})
        }
        endpoints = [
            event
            for event in device
            if str(event.get("args", {}).get("correlation")) in links
        ]
        sync = [event for event in calls if "Synchronize" in event["name"]]
        dtoh = [
            event
            for event in endpoints
            if event["cat"] == "gpu_memcpy" and "DtoH" in event["name"]
        ]
        nccl = [
            event["name"]
            for event in endpoints
            if event["cat"] == "kernel" and "nccl" in event["name"].lower()
        ]
        scopes[name] = {
            "host_sync_api_count": len(sync),
            "host_sync_api_ms": sum(event["dur"] for event in sync) / 1000,
            "dtoh_count": len(dtoh),
            "dtoh_device_ms": sum(event["dur"] for event in dtoh) / 1000,
            "nccl_kernels": nccl,
        }
    require(
        scopes["counts"]["dtoh_count"] == 1, f"{path.name} missing correlated count D2H"
    )
    for name, family in (
        ("counts", "AllGather"),
        ("request_exchange", "SendRecv"),
        ("value_exchange", "SendRecv"),
        ("gradient_exchange", "SendRecv"),
    ):
        calls = scopes[name]["nccl_kernels"]
        require(
            len(calls) == 1 and family.lower() in calls[0].lower(),
            f"{path.name} missing {name} NCCL endpoint",
        )
    return scopes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("original_replay", type=Path)
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args()
    expected = read(args.original_replay)["input_ids_sha256_by_rank"]
    if args.fixture_only:
        check_fixture(args.path, expected)
        return
    exact_hashes = check_exact(args.path, expected)
    perf = {}
    traces = {}
    reference = None
    for arm in ARMS:
        rows = read_arm(args.path, arm, expected, exact_hashes)
        identity = [
            (
                row["source"],
                row["runtime"],
                row["results"][0]["owner_route_census"],
                row["gfsdp_gradient_buffer_inventory"]["unique_storage_bytes"],
            )
            for row in rows
        ]
        if reference is None:
            reference = identity
        require(
            identity == reference,
            f"{arm} source, hardware, route census or pool mismatch",
        )
        if arm.startswith("diag-"):
            traces[arm] = [
                trace_scopes(args.path / arm / f"owner-fixture-rank{rank}.trace.json")
                for rank in RANKS
            ]
        else:
            perf[arm] = integrated(rows)
    removed = all(
        a["partition_ids"]["host_sync_api_count"] >= 1
        and a["partition_ids"]["dtoh_count"] == 1
        and b["partition_ids"]["host_sync_api_count"] == 0
        and b["partition_ids"]["dtoh_count"] == 0
        for a, b in zip(traces["diag-A"], traces["diag-B"], strict=True)
    )
    wall = {
        arm: values["synchronized_wall_ms"]["median"] for arm, values in perf.items()
    }
    summary = {
        "exact_all_eight_ranks": True,
        "extra_partition_sync_removed_on_all_ranks": removed,
        "timing_scope": "unprofiled synchronized embedding forward and backward, median and max of per-sample rank maxima",
        "timing": perf,
        "synchronized_wall_change_pct": {
            f"{right}/{left}": 100 * (wall[right] / wall[left] - 1)
            for left, right in (("A1", "B1"), ("A2", "B2"), ("A1", "A2"))
        },
        "trace_scope": "one profiled F+B per arm and rank, host annotations linked to NCCL and D2H by CUDA correlation ID",
        "diagnostics": traces,
        "memory_scope": "component peak allocator bytes and unchanged deduplicated gradient pool, with private head allocation resident",
        "gradient_pool_bytes_by_rank": [item[-1] for item in reference],
    }
    output = args.path / "summary.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        f"Exact output, FP32 gradients, and BF16 gradient messages on all eight ranks. Partition fence removed: {removed}."
    )
    for arm, values in perf.items():
        print(
            f"{arm} measured {wall[arm]:.3f} ms synchronized F+B, peak allocated {values['peak_allocated_bytes']['max'] / 2**30:.3f} GiB"
        )
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
