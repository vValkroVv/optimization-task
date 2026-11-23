"""Collect all fresh-process timing, numerical, and GIFT memory comparisons."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys


def read(path):
    return json.loads(path.read_text())


def summary(samples):
    wall = [receipt["median_ms"] for receipt in samples]
    stages = defaultdict(list)
    for receipt in samples:
        run_stage = defaultdict(float)
        for name, elapsed in receipt["diagnostic_stage_events_ms"]:
            run_stage[name] += elapsed
        for name, elapsed in run_stage.items():
            stages[name].append(elapsed)
        if run_stage:
            stages["sum"].append(sum(run_stage.values()))
    return {
        "repeats": len(samples),
        "wall_ms": median(wall),
        "process_medians_ms": wall,
        "process_spread_ms": max(wall) - min(wall),
        "peak_total_gib": max(receipt["peak_total_gpu_gib"] for receipt in samples),
        "peak_allocated_gib": max(sample["allocated_peak_bytes"] for r in samples for sample in r["memory"]) / 2**30,
        "peak_reserved_gib": max(sample["reserved_peak_bytes"] for r in samples for sample in r["memory"]) / 2**30,
        "max_external_gib": max(sample["external_gib"] for r in samples for sample in r["memory"]),
        "host_submit_ms": median(sample for receipt in samples for sample in receipt["host_submit_ms"]),
        "host_wait_ms": median(sample for receipt in samples for sample in receipt["host_wait_ms"]),
        "diagnostic_stage_event_ms": {name: median(values) for name, values in stages.items()},
        "numerics": [receipt["fp64"] for receipt in samples if "fp64" in receipt],
    }


def gather(directory):
    bundles = defaultdict(list)
    for path in sorted(directory.glob("[0-9]*.json")):
        if path.name.endswith((".digest.json", ".fp64.json")):
            continue
        receipt = read(path)
        receipt["digest"] = read(path.with_suffix(".digest.json"))
        fp64 = path.with_suffix(".fp64.json")
        if fp64.is_file():
            receipt["fp64"] = read(fp64)
        receipt["path"] = path.name
        bundles[receipt["arm"]].append(receipt)

    pairs = {}
    controls = bundles["control"]
    assert controls and len({r["device"]["visible"] for group in bundles.values() for r in group}) == 1
    native_digest = controls[0]["digest"]
    control_numerics = [receipt["fp64"] for receipt in controls if "fp64" in receipt]
    for arm, samples in bundles.items():
        reduced = pairs[arm] = summary(samples)
        reduced["bitwise_equal"] = {
            name: all(receipt["digest"][name] == native_digest[name] for receipt in samples)
            for name in native_digest
        }
        if control_numerics and reduced["numerics"]:
            reduced["no_worse_fp64"] = all(
                all(numeric[key] <= control_numerics[0][key] for key in control_numerics[0])
                for numeric in reduced["numerics"]
            )
        for receipt in samples:
            assert receipt["native_inplace_ce_chunks"] == receipt["chunks"]
            assert receipt["ce_route_calls"] == {"folded_inplace": receipt["chunks"]}
            assert receipt["expandable_segments"]
            assert receipt["compile"]["counters"]["unique_graphs"] == 1
            assert not any(receipt["allocator_retries_and_ooms"].values())
            assert abs(sum(receipt["host_submit_ms"]) + sum(receipt["host_wait_ms"])
                       - sum(receipt["elapsed_ms"])) < 1e-7
    for reduced in pairs.values():
        reduced["wall_pct_vs_control"] = (reduced["wall_ms"] / pairs["control"]["wall_ms"] - 1) * 100
        reduced["total_gib_vs_control"] = reduced["peak_total_gib"] - pairs["control"]["peak_total_gib"]
    return pairs


if __name__ == "__main__":
    directories = map(Path, sys.argv[1:])
    print(json.dumps({path.name: gather(path) for path in directories}, indent=2))
