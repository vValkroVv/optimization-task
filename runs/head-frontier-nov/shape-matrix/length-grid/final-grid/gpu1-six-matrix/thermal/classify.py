"""Join physical-GPU NVML telemetry to rank-local matrix call windows."""
import argparse
import csv
import gzip
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median

NS = 10**9
ACTIVE = tuple(range(6))
THERMAL = 0x20 | 0x40
BRAKE = 0x80
SW_POWER = 0x4
HW_SLOWDOWN = 0x8
IDLE = 0x1


@dataclass(frozen=True, slots=True)
class Sample:
    series: int
    utc_ns: int
    counter_utc_ns: int
    temp_c: int
    memory_c: int | None
    sm_mhz: int
    util_pct: int
    mem_used_mib: float
    mask: int
    thermal_counter_ns: int
    sw_power_counter_ns: int | None


def load(path):
    return json.loads(path.read_text())


def concat_windows(windows):
    merged = []
    for lo, hi in sorted(windows):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def complement(windows, lo, hi):
    gaps = []
    for start, stop in concat_windows(windows):
        if lo < start:
            gaps.append((lo, start - 1))
        lo = max(lo, stop + 1)
    if lo <= hi:
        gaps.append((lo, hi))
    return gaps


def record_stats(selected):
    masks = [s.mask for s in selected]
    temps = [s.temp_c for s in selected]
    hbm = [s.memory_c for s in selected if s.memory_c is not None]
    clocks = [s.sm_mhz for s in selected]
    return {
        "samples": len(selected),
        "clock_reason_mask_counts": {m: masks.count(m) for m in sorted(set(masks))},
        "sw_thermal_hits": sum(bool(m & 0x20) for m in masks),
        "hw_thermal_hits": sum(bool(m & 0x40) for m in masks),
        "thermal_hits": sum(bool(m & THERMAL) for m in masks),
        "power_brake_hits": sum(bool(m & BRAKE) for m in masks),
        "hw_slowdown_hits": sum(bool(m & HW_SLOWDOWN) for m in masks),
        "sw_power_hits": sum(bool(m & SW_POWER) for m in masks),
        "idle_hits": sum(bool(m & IDLE) for m in masks),
        "core_c_minmax": [min(temps), max(temps)] if temps else None,
        "hbm_c_minmax": [min(hbm), max(hbm)] if hbm else None,
        "sm_mhz_minmedianmax": [min(clocks), median(clocks), max(clocks)] if clocks else None,
        "max_util_pct": max((s.util_pct for s in selected), default=None),
        "max_mem_used_mib": max((s.mem_used_mib for s in selected), default=None),
    }


class Card:
    def __init__(self, samples):
        self.samples = samples
        self.times = [s.utc_ns for s in samples]
        self.read_times = [s.counter_utc_ns for s in samples]
        if len(samples) < 2 or any(
            a.series + 1 != b.series or a.utc_ns >= b.utc_ns
            or a.counter_utc_ns >= b.counter_utc_ns
            for a, b in pairwise(samples)
        ):
            raise ValueError("Missing or unordered samples")

    def assess(self, raw_windows, gap_limit_ns):
        windows = concat_windows(raw_windows)
        starts = [lo for lo, _ in windows]
        selected = []
        in_scope = set()
        longest_gap = 0
        missing_bracket = 0
        for lo, hi in windows:
            selected.extend(self.samples[bisect_left(self.times, lo):bisect_right(self.times, hi)])
            left = bisect_right(self.read_times, lo) - 1
            right = bisect_left(self.read_times, hi)
            if left < 0 or right >= len(self.samples):
                missing_bracket += 1
            in_scope.update(range(max(left, 0), min(right, len(self.samples) - 1)))
        definite = possible = reset = 0
        thermal_increments = []
        for i in sorted(in_scope):
            a, b = self.samples[i:i + 2]
            gap = b.counter_utc_ns - a.counter_utc_ns
            longest_gap = max(longest_gap, gap)
            diff = b.thermal_counter_ns - a.thermal_counter_ns
            reset += diff < 0
            if diff <= 0:
                continue
            j = bisect_right(starts, a.counter_utc_ns) - 1
            contained = j >= 0 and b.counter_utc_ns <= windows[j][1]
            if contained:
                definite += diff
            else:
                possible += diff
            thermal_increments.append([a.counter_utc_ns, b.counter_utc_ns, diff, contained])
        complete = bool(windows) and not missing_bracket and not reset and longest_gap <= gap_limit_ns
        return {
            **record_stats(selected),
            "windows": len(raw_windows),
            "windows_with_mask_sample": sum(
                bisect_left(self.times, lo) < bisect_right(self.times, hi) for lo, hi in raw_windows
            ),
            "windows_missing_counter_bracket": missing_bracket,
            "max_bracketing_gap_s": longest_gap / NS if in_scope else None,
            "covered": complete,
            "thermal_counter_reset_intervals": reset,
            "thermal_counter_delta_fully_inside_ns": definite,
            "thermal_counter_delta_boundary_ns": possible,
            "thermal_counter_increment_intervals": thermal_increments,
        }


def open_csv(path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open(newline="")


def read_cards(path):
    records = [[] for _ in range(8)]
    with open_csv(path) as stream:
        for row in csv.DictReader(stream):
            if any(row.get(k) is None for k in (
                "utc_ns", "counter_utc_ns", "gpu", "mask", "thermal_counter_ns"
            )):
                # A live reader can see the final physical line in mid-write.
                break
            records[int(row["gpu"])].append(Sample(
                int(row["series"]), int(row["utc_ns"]), int(row["counter_utc_ns"]),
                int(row["temp_c"]), int(row["memory_c"]) if row["memory_c"] else None,
                int(row["sm_mhz"]), int(row["util_pct"]), float(row["mem_used_mib"]),
                int(row["mask"]), int(row["thermal_counter_ns"]),
                int(row["sw_power_counter_ns"]) if row["sw_power_counter_ns"] else None,
            ))
    return [Card(samples) for samples in records]


def positive(region):
    return region["thermal_hits"] or region["thermal_counter_delta_fully_inside_ns"]


def classify(gpus):
    active = gpus[:6]
    if any(positive(g["measured"]) for g in active):
        return "timed_thermal_observed"
    if not all(g["measured"]["covered"] for g in active):
        return "insufficient_coverage"
    if any(g["measured"]["thermal_counter_delta_boundary_ns"] for g in active):
        return "timed_thermal_possible"
    if any(positive(g["arm"]) for g in active):
        return "outside_timed_calls"
    return "no_thermal_observed"


def check_windows(windows, start, stop):
    if not windows or not all(start <= lo < hi <= stop for lo, hi in windows):
        raise ValueError("Call outside the arm or empty call window")
    if any(b[0] < a[1] for a, b in pairwise(windows)):
        raise ValueError("Overlapping call windows on one rank")


def assess_arm(folder, relative, status, cards, meta, gap_limit_ns):
    cell, pair, variant, attempt = relative.parts
    report = {
        "path": str(relative), "cell": cell, "pair": pair, "variant": variant,
        "attempt": attempt, "status": status["status"], "reason": status.get("reason"),
        "started_utc_ns": status["started_utc_ns"], "ended_utc_ns": status["ended_utc_ns"],
    }
    start, stop = report["started_utc_ns"], report["ended_utc_ns"]
    if status["status"] != "success":
        report["verdict"] = "attempt_failed"
        report["gpus"] = [
            {"gpu": gpu, "role": "participant" if gpu in ACTIVE else "unused",
             "arm": card.assess([(start, stop)], gap_limit_ns)}
            for gpu, card in enumerate(cards)
        ]
        return report
    invocation = load(folder / "invocation.json")
    rank_map = invocation["physical_gpu_index_by_logical_rank"]
    if rank_map != list(ACTIVE) or invocation["world_size"] != len(ACTIVE):
        raise ValueError(f"Unexpected rank-to-physical-GPU map: {rank_map}")
    measured = []
    warmup = []
    for rank, gpu in enumerate(rank_map):
        physical = load(folder / f"physical-device-{rank}.json")
        uuid = meta["gpu_uuids"][gpu]
        if (physical["logical_rank"] != rank or physical["visible_uuid"] != uuid
                or physical["runtime_uuid"].removeprefix("GPU-") != uuid.removeprefix("GPU-")):
            raise ValueError(f"Rank {rank} physical identity differs from sampler GPU {gpu}")
        times = load(folder / f"timing-{rank}.json")
        contract = load(folder / f"contract-{rank}.json")
        if len(times) != invocation["samples"] or any(
            t["profiled"] or not t["start_utc_ns"] <= t["submitted_utc_ns"] <= t["stop_utc_ns"]
            for t in times
        ):
            raise ValueError(f"Incomplete profiled or malformed samples on rank {rank}")
        rank_measured = [(w["start_utc_ns"], w["stop_utc_ns"]) for w in times]
        rank_warmup = [(lo, hi) for lo, hi in contract["warmup_host_utc_ns"]]
        if (len(rank_warmup) != invocation["warmup"]
                or rank_warmup[-1][1] > rank_measured[0][0]):
            raise ValueError(f"Incomplete or overlapping warmup on rank {rank}")
        check_windows(rank_measured, start, stop)
        check_windows(rank_warmup, start, stop)
        measured.append(rank_measured)
        warmup.append(rank_warmup)
    measured_union = concat_windows(w for rank in measured for w in rank)
    warmup_union = concat_windows(w for rank in warmup for w in rank)
    gpus = []
    for gpu, card in enumerate(cards):
        timed = measured[gpu] if gpu in ACTIVE else measured_union
        warm = warmup[gpu] if gpu in ACTIVE else warmup_union
        gpus.append({
            "gpu": gpu, "role": "participant" if gpu in ACTIVE else "unused",
            "arm": card.assess([(start, stop)], gap_limit_ns),
            "measured": card.assess(timed, gap_limit_ns),
            "warmup": card.assess(warm, gap_limit_ns),
            "outside_warmup_and_measured": card.assess(
                complement([*timed, *warm], start, stop), gap_limit_ns
            ),
        })
    call_ms = [max((end - begin) / 10**6 for begin, end in windows) for windows in zip(*measured, strict=True)]
    report.update(
        measured_call_median_max_rank_ms=median(call_ms),
        measured_calls_max_rank_ms=call_ms,
        gpus=gpus,
        verdict=classify(gpus),
        active_gpu_thermal_mask_indices_measured=[
            g["gpu"] for g in gpus[:6] if g["measured"]["thermal_hits"]
        ],
        active_gpu_indices_thermal_evidence_measured=[
            g["gpu"] for g in gpus[:6] if positive(g["measured"])
        ],
        active_gpu_indices_power_brake_measured=[
            g["gpu"] for g in gpus[:6] if g["measured"]["power_brake_hits"]
        ],
    )
    return report


def summarize_cell(arms):
    expected = {(pair, variant) for pair in ("pair-01", "pair-02") for variant in ("baseline", "cuda13")}
    accepted = {}
    for arm in arms:
        if arm["status"] == "success" and "gpus" in arm:
            accepted[(arm["pair"], arm["variant"])] = arm
    timing_by_variant = {}
    for variant in ("baseline", "cuda13"):
        variant_arms = [a for (_, v), a in sorted(accepted.items()) if v == variant]
        medians = [a["measured_call_median_max_rank_ms"] for a in variant_arms]
        timing_by_variant[variant] = {
            "median_ms_by_repeat": medians,
            "spread_pct_relative_to_fastest": (100 * (max(medians) / min(medians) - 1)
                                                if len(medians) == 2 else None),
            "thermal_verdict_by_repeat": [a["verdict"] for a in variant_arms],
        }
    if accepted.keys() != expected:
        verdict = "incomplete"
    else:
        verdict = next(v for v in (
            "timed_thermal_observed", "insufficient_coverage", "timed_thermal_possible",
            "outside_timed_calls", "no_thermal_observed",
        ) if v in {a["verdict"] for a in accepted.values()})
    return {
        "verdict": verdict,
        "attempts": len(arms),
        "successful_arms": len(accepted),
        "arm_verdicts": {a["path"]: a["verdict"] for a in arms},
        "timing_by_variant": timing_by_variant,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("meta", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gap-limit-s", type=float, default=1.0)
    args = parser.parse_args()
    cards = read_cards(args.samples)
    meta = load(args.meta)
    arm_reports = []
    for path in sorted(args.results.glob("*/*/*/attempt-*/status.json")):
        status = load(path)
        folder = path.parent
        relative = folder.relative_to(args.results)
        try:
            arm = assess_arm(folder, relative, status, cards, meta, args.gap_limit_s * NS)
        except (OSError, KeyError, ValueError) as error:
            cell, pair, variant, attempt = relative.parts
            arm = {"path": str(relative), "cell": cell, "pair": pair, "variant": variant,
                   "attempt": attempt, "status": status["status"],
                   "started_utc_ns": status["started_utc_ns"], "ended_utc_ns": status["ended_utc_ns"],
                   "verdict": "invalid_or_incomplete_receipts", "error": str(error)}
        arm_reports.append(arm)
    by_cell = defaultdict(list)
    for arm in arm_reports:
        by_cell[arm["cell"]].append(arm)
    campaign = []
    first = max(card.read_times[0] for card in cards)
    last = min(card.read_times[-1] for card in cards)
    between = complement([
        (max(first, a["started_utc_ns"]), min(last, a["ended_utc_ns"]))
        for a in arm_reports if first <= a["ended_utc_ns"] and a["started_utc_ns"] <= last
    ], first, last)
    for gpu, card in enumerate(cards):
        start, end = card.samples[0], card.samples[-1]
        campaign.append({
            "gpu": gpu, "role": "participant" if gpu in ACTIVE else "unused",
            "first_utc_ns": start.utc_ns, "last_utc_ns": end.utc_ns,
            "samples": record_stats(card.samples),
            "thermal_counter_first_ns": start.thermal_counter_ns,
            "thermal_counter_last_ns": end.thermal_counter_ns,
            "sw_power_counter_first_ns": start.sw_power_counter_ns,
            "sw_power_counter_last_ns": end.sw_power_counter_ns,
            "outside_recorded_arms": card.assess(between, args.gap_limit_s * NS),
        })
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "source": {"nvml_samples": args.samples.name, "nvml_metadata": args.meta.name,
                   "matrix_receipts": str(args.results), "sampler": meta,
                   "coverage_gap_limit_s": args.gap_limit_s},
        "sampling": campaign, "arms": arm_reports,
        "cells": {cell: summarize_cell(arms) for cell, arms in sorted(by_cell.items())},
    }, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
