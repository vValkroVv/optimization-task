"""Measure the GDN gated-norm forward through FP8 activation packing in Kineto."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
from statistics import median

from profile_attribution import DEVICE_CATEGORIES, HOST_CATEGORIES, summarize


ROOTS = {
    "baseline": "torchtitan::gated_rmsnorm",
    "combined": "torchtitan::gdn_output_norm_quantize",
}
PHASES = {"CompiledFunction": "forward", "CompiledFunctionBackward": "sac_replay"}
ACTIVATION_PACK = "cudacherry::quantize_vec128"
WEIGHT_PACK = "cudacherry::quantize_blk128x128"
LEGACY_VJP = "torchtitan::fla_vjp"


def thread(event):
    return event["pid"], event["tid"]


def correlation(event):
    return event.get("args", {}).get("correlation")


def intervals(events):
    ordered = sorted(events, key=lambda event: event["ts"])
    starts = [event["ts"] for event in ordered]

    def containing(event):
        index = bisect_right(starts, event["ts"]) - 1
        if index >= 0 and event["ts"] < starts[index] + ordered[index]["dur"]:
            return ordered[index]
        return None

    return containing


def analyze(task):
    path, variant, rank = task
    open_trace = gzip.open if path.suffix == ".gz" else open
    with open_trace(path, "rt") as stream:
        events = [event for event in json.load(stream)["traceEvents"] if event.get("ph") == "X"]

    marker = f"task.profile_update.11.microbatch.rank{rank}"
    microbatches = sorted((event for event in events if event.get("cat") == "user_annotation"
                           and event["name"] == marker),
                          key=lambda event: event["ts"])
    if len(microbatches) != 32:
        raise ValueError(f"{path}: expected 32 microbatches of update 11")
    microbatch_lookup = intervals(microbatches)
    roots = [event for event in events if event.get("cat") == "cpu_op" and event["name"] == ROOTS[variant]]
    root_threads = {thread(event) for event in roots}

    annotations = defaultdict(list)
    hosts = defaultdict(list)
    devices = []
    names = set(PHASES) | {ACTIVATION_PACK, WEIGHT_PACK, LEGACY_VJP}
    for event in events:
        category = event.get("cat")
        if category in HOST_CATEGORIES and correlation(event) is not None:
            hosts[correlation(event)].append(event)
        elif category in DEVICE_CATEGORIES:
            devices.append(event)
        elif category == "cpu_op" and event["name"] in names and thread(event) in root_threads:
            annotations[thread(event), event["name"]].append(event)
    lookups = {key: intervals(value) for key, value in annotations.items()}

    launches_by_thread = defaultdict(list)
    paired_launches = set()
    for event in devices:
        links = [host for host in hosts.get(correlation(event), ()) if host["ts"] <= event["ts"] + 1]
        if any(thread(host) in root_threads for host in links):
            if len(links) != 1:
                raise ValueError(f"{path}: ambiguous CUDA launch for {event['name']}")
            launches_by_thread[thread(links[0])].append((links[0], event))
            paired_launches.add(id(links[0]))
    launch_indexes = {}
    for launch_thread, launches in launches_by_thread.items():
        launches.sort(key=lambda pair: pair[0]["ts"])
        launch_indexes[launch_thread] = [host["ts"] for host, _ in launches]

    def owner(name, event):
        lookup = lookups.get((thread(event), name))
        return lookup(event) if lookup else None

    def describe(events_to_describe):
        return [event["name"] for event in events_to_describe]

    phase_counts = Counter()
    included_by_stage = defaultdict(list)
    previous_kernels = Counter()
    included_scopes = {id(root) for root in roots}
    for root in roots:
        phases = [(phase, owner(name, root)) for name, phase in PHASES.items()]
        phases = [(phase, parent) for phase, parent in phases if parent is not None]
        microbatch = microbatch_lookup(root)
        if len(phases) != 1 or microbatch is None or owner(LEGACY_VJP, root) is not None:
            raise ValueError(f"{path}: norm is outside an unambiguous forward or SAC replay")
        phase, parent = phases[0]
        phase_counts[microbatch["ts"], phase] += 1
        launches = launches_by_thread[thread(root)]
        starts = launch_indexes[thread(root)]
        start = bisect_left(starts, root["ts"])
        end = bisect_left(starts, root["ts"] + root["dur"])
        norm = [event for _, event in launches[start:end]]
        expected_norm = ("layer_norm_gated_fwd_kernel" if variant == "baseline"
                         else "_gdn_output_norm_quantize_128_kernel")
        if len(norm) != 1 or expected_norm not in norm[0]["name"]:
            raise ValueError(f"{path}: unexpected norm device work {describe(norm)}")
        if start > 0:
            previous_kernels[launches[start - 1][1]["name"]] += 1

        if variant == "baseline":
            (scale_host, scale), (pack_host, pack) = launches[end:end + 2]
            weight_host, _ = launches[end + 2]
            pack_scope = owner(ACTIVATION_PACK, pack_host)
            if ("triton_poi_fused_mul_quantize_vec128_view_" not in scale["name"]
                    or "quantize_vec128_kernel<8, 2, true, false, false>" not in pack["name"]
                    or pack_scope is None or owner(LEGACY_VJP, scale_host) is not None
                    or weight_host["ts"] >= parent["ts"] + parent["dur"]):
                raise ValueError(f"{path}: unexpected activation pack {describe([scale, pack])}")
            included_scopes.add(id(pack_scope))
            included_by_stage[phase, "native_gated_norm"].extend(norm)
            included_by_stage[phase, "multiplier_and_view"].append(scale)
            included_by_stage[phase, "fp8_activation_pack"].append(pack)
        else:
            weight_host, _ = launches[end]
            included_by_stage[phase, "fused_norm_and_fp8_activation_pack"].extend(norm)
        if owner(WEIGHT_PACK, weight_host) is None:
            raise ValueError(f"{path}: expected the excluded weight pack just after activation pack, "
                             f"found {launches[end + (2 if variant == 'baseline' else 0)][1]['name']}")

    for microbatch in microbatches:
        for phase in PHASES.values():
            if phase_counts[microbatch["ts"], phase] != 6:
                raise ValueError(f"{path}: expected six GDN calls per microbatch for {phase}")

    root_lookup = {key: intervals(root for root in roots if thread(root) == key)
                   for key in root_threads}
    for host_group in hosts.values():
        for host in host_group:
            if (thread(host) not in root_lookup or id(host) in paired_launches
                    or not any(token in host["name"].lower() for token in ("launchkernel", "memcpy", "memset"))):
                continue
            scope = root_lookup[thread(host)](host) or owner(ACTIVATION_PACK, host)
            if scope is not None and id(scope) in included_scopes:
                raise ValueError(f"{path}: an owned launch has no recorded device work")

    stages = defaultdict(dict)
    selected = []
    phase_devices = defaultdict(list)
    for (phase, stage), stage_events in included_by_stage.items():
        stages[phase][stage] = summarize(stage_events)
        selected.extend(stage_events)
        phase_devices[phase].extend(stage_events)
    return {
        "trace": str(path),
        "rank": rank,
        "variant": variant,
        "calls_per_phase": 192,
        "device": summarize(selected),
        "phases": {phase: summarize(phase_devices[phase]) for phase in PHASES.values()},
        "stages": stages,
        "kernel_counts": dict(Counter(event["name"] for event in selected)),
        "previous_kernel_counts": dict(previous_kernels),
    }


def compact(result):
    return {key: result[key] for key in ("rank", "variant", "device", "phases")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="parent of the four profile run directories")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ep", nargs="+", type=int, choices=(8, 4), default=[8, 4])
    parser.add_argument("--ranks", nargs="+", type=int, choices=range(8), default=list(range(8)))
    args = parser.parse_args()

    output = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for ep in args.ep:
            tasks = [
                (args.raw / f"profile-10x1-mb32-001-ep{ep}-{variant}"
                 / f"native/profiling/traces/iteration_11/rank{rank}_trace.json", variant, rank)
                for rank in args.ranks for variant in ROOTS
            ]
            results = []
            for result in pool.map(analyze, tasks):
                results.append(result)
                print(json.dumps({"ep": ep, **compact(result)}), flush=True)
            baseline = [row["device"]["duration_sum_us"] / 1000 for row in results[::2]]
            combined = [row["device"]["duration_sum_us"] / 1000 for row in results[1::2]]
            deltas = [new - old for old, new in zip(baseline, combined, strict=True)]
            output[f"ep{ep}"] = {
                "summary": {
                    "baseline_median_kernel_sum_ms": median(baseline),
                    "combined_median_kernel_sum_ms": median(combined),
                    "median_change_of_rank_pairs_ms": median(deltas),
                    "range_change_of_rank_pairs_ms": [min(deltas), max(deltas)],
                    "median_change_of_rank_pairs_pct": median(delta / old * 100
                                                               for delta, old in zip(deltas, baseline, strict=True)),
                },
                "traces": results,
            }
            print(json.dumps({f"ep{ep}": output[f"ep{ep}"]["summary"]}), flush=True)

    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
