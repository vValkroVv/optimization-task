"""Match GDN backward scopes in the four existing 10+1 Kineto captures."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import re
from statistics import median

from profile_attribution import DEVICE_CATEGORIES, HOST_CATEGORIES, summarize


LEGACY_ROOT = "torchtitan::fla_vjp"
LEGACY_NORM_MULTIPLIER = "triton_poi_fused_mul_quantize_vec128_view_3"
SPECIALIZED_ROOTS = {
    "torchtitan::gdn_fla_backward": "scan",
    "torchtitan::gdn_short_conv_backward": "short_conv",
    "torchtitan::gdn_output_norm_backward": "output_norm",
}
MARKERS = {
    "baseline": {
        "scan": "chunk_bwd_kernel",
        "short_conv": "causal_conv1d_bwd_kernel",
        "output_norm": "layer_norm_gated_bwd_kernel",
    },
    "combined": {
        "scan": "chunk_bwd_kernel",
        "short_conv": "_gdn_short_conv_backward_kernel",
        "output_norm": "_gdn_output_norm_backward",
    },
}
COMPONENTS = tuple(SPECIALIZED_ROOTS.values())
MICROBATCHES = 32
GDN_LAYERS = 6


def time_lookup(events):
    ordered = sorted(events, key=lambda event: event["ts"])
    starts = [event["ts"] for event in ordered]
    for first, second in zip(ordered, ordered[1:]):
        if first["ts"] + first["dur"] > second["ts"]:
            raise ValueError(f"Overlapping scopes {first['name']} and {second['name']}")

    def find(ts):
        index = bisect_right(starts, ts) - 1
        if index >= 0 and ts < starts[index] + ordered[index]["dur"]:
            return index
        return None

    return ordered, find


def work_summary(events):
    result = summarize(events)
    return {
        "count": result["count"],
        "duration_sum_ms": result["duration_sum_us"] / 1000,
        "busy_union_ms": result["busy_union_us"] / 1000,
    }


def analyze_trace(task):
    path, variant, rank = task
    open_trace = gzip.open if path.suffix == ".gz" else open
    with open_trace(path, "rt") as stream:
        events = [
            event
            for event in json.load(stream)["traceEvents"]
            if event.get("ph") == "X"
        ]

    microbatch_pattern = re.compile(
        rf"task\.profile_update\.(\d+)\.microbatch\.rank{rank}"
    )
    microbatches, find_microbatch = time_lookup(
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and microbatch_pattern.fullmatch(event["name"])
    )
    if (
        len(microbatches) != MICROBATCHES
        or len({event["name"] for event in microbatches}) != 1
    ):
        raise ValueError(
            f"{path}: expected {MICROBATCHES} microbatches of a single update"
        )
    update = int(microbatch_pattern.fullmatch(microbatches[0]["name"])[1])
    if update != 11:
        raise ValueError(f"{path}: expected update 11, found {update}")

    roots_by_thread = defaultdict(list)
    for event in events:
        if event.get("cat") == "cpu_op" and (
            event["name"] == LEGACY_ROOT or event["name"] in SPECIALIZED_ROOTS
        ):
            roots_by_thread[event["pid"], event["tid"]].append(event)
    thread_lookups = {
        thread: time_lookup(roots) for thread, roots in roots_by_thread.items()
    }

    hosts_by_correlation = defaultdict(list)
    owned_launches = {}
    for event in events:
        if event.get("cat") not in HOST_CATEGORIES:
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is None:
            continue
        hosts_by_correlation[correlation].append(event)
        thread = event["pid"], event["tid"]
        if thread in thread_lookups:
            _, find_root = thread_lookups[thread]
            index = find_root(event["ts"])
            if index is not None:
                owned_launches[id(event)] = thread, index

    root_devices = defaultdict(list)
    paired_hosts = set()
    kernel_links_by_thread = defaultdict(list)
    for event in events:
        if event.get("cat") not in DEVICE_CATEGORIES:
            continue
        correlation = event.get("args", {}).get("correlation")
        hosts = [
            host
            for host in hosts_by_correlation.get(correlation, ())
            if host["ts"] <= event["ts"] + 1
        ]
        owners = [
            owned_launches[id(host)] for host in hosts if id(host) in owned_launches
        ]
        if len(hosts) != 1 and owners:
            raise ValueError(
                f"{path}: ambiguous host for GDN device event {event['name']}"
            )
        if variant == "baseline" and len(hosts) == 1 and event["cat"] == "kernel":
            host = hosts[0]
            if (host["pid"], host["tid"]) in thread_lookups:
                kernel_links_by_thread[host["pid"], host["tid"]].append((host, event))
        if not owners:
            continue
        root_devices[owners[0]].append(event)
        paired_hosts.add(id(hosts[0]))

    owned_unpaired = sorted(
        {
            host["name"]
            for hosts in hosts_by_correlation.values()
            for host in hosts
            if id(host) in owned_launches
            and id(host) not in paired_hosts
            and any(
                token in host["name"].lower()
                for token in ("launchkernel", "memcpy", "memset")
            )
        }
    )
    if owned_unpaired:
        raise ValueError(
            f"{path}: GDN CUDA launches without recorded device work: {owned_unpaired}"
        )

    counts_by_microbatch = defaultdict(Counter)
    component_roots = defaultdict(list)
    component_devices = defaultdict(list)
    baseline_norm_multipliers = []
    seen_multipliers = set()
    for thread, (roots, _) in thread_lookups.items():
        thread_kernels = sorted(
            kernel_links_by_thread[thread], key=lambda link: link[0]["ts"]
        )
        kernel_launch_times = [host["ts"] for host, _ in thread_kernels]
        for index, root in enumerate(roots):
            owned = root_devices[thread, index]
            kernel_names = [
                event["name"] for event in owned if event["cat"] == "kernel"
            ]
            marker_matches = [
                component
                for component, token in MARKERS[variant].items()
                if any(token in name for name in kernel_names)
            ]
            if len(marker_matches) != 1:
                raise ValueError(
                    f"{path}: expected one backward marker in {root['name']}; "
                    f"found {marker_matches} in {kernel_names}"
                )
            component = marker_matches[0]
            expected = (
                LEGACY_ROOT if variant == "baseline" else f"specialized {component}"
            )
            if (variant == "baseline" and root["name"] != LEGACY_ROOT) or (
                variant == "combined"
                and SPECIALIZED_ROOTS.get(root["name"]) != component
            ):
                raise ValueError(f"{path}: expected {expected}, found {root['name']}")
            microbatch = find_microbatch(root["ts"])
            if (
                microbatch is None
                or root["ts"] + root["dur"]
                > microbatches[microbatch]["ts"] + microbatches[microbatch]["dur"]
            ):
                raise ValueError(
                    f"{path}: {root['name']} outside any complete microbatch"
                )
            counts_by_microbatch[microbatch][component] += 1
            component_roots[component].append(root)
            component_devices[component].extend(owned)
            if variant == "baseline" and component == "output_norm":
                # The retained Inductor wrapper passes this kernel's in-place
                # BF16 output to the norm VJP. The forward path reuses the name.
                prior = bisect_left(kernel_launch_times, root["ts"]) - 1
                if prior < 0:
                    raise ValueError(
                        f"{path}: no kernel launched before output norm VJP"
                    )
                host, multiplier = thread_kernels[prior]
                if (
                    multiplier["name"] != LEGACY_NORM_MULTIPLIER
                    or id(host) in owned_launches
                    or id(multiplier) in seen_multipliers
                    or find_microbatch(host["ts"]) != microbatch
                ):
                    raise ValueError(
                        f"{path}: unmatched output norm multiplier: {multiplier['name']}"
                    )
                seen_multipliers.add(id(multiplier))
                baseline_norm_multipliers.append(multiplier)
                component_devices[component].append(multiplier)

    expected_counts = Counter(dict.fromkeys(COMPONENTS, GDN_LAYERS))
    for index in range(MICROBATCHES):
        if counts_by_microbatch[index] != expected_counts:
            raise ValueError(
                f"{path}: microbatch {index + 1}: {counts_by_microbatch[index]}"
            )

    return {
        "path": str(path),
        "rank": rank,
        "variant": variant,
        "update": update,
        "microbatches": len(microbatches),
        "baseline_norm_multiplier": work_summary(baseline_norm_multipliers),
        "components": {
            component: {
                "calls": len(component_roots[component]),
                "host_scope_sum_ms": sum(
                    root["dur"] for root in component_roots[component]
                )
                / 1000,
                "device_work": {
                    category: work_summary(
                        [
                            event
                            for event in component_devices[component]
                            if event["cat"] == category
                        ]
                    )
                    for category in sorted(DEVICE_CATEGORIES)
                },
                "kernel_counts": dict(
                    Counter(
                        event["name"]
                        for event in component_devices[component]
                        if event["cat"] == "kernel"
                    )
                ),
            }
            for component in COMPONENTS
        },
    }


def summarize_pair(baseline, combined):
    output = {}
    for component in COMPONENTS:
        old_by_rank = [rank["components"][component] for rank in baseline]
        new_by_rank = [rank["components"][component] for rank in combined]
        old_ms = [
            row["device_work"]["kernel"]["duration_sum_ms"] for row in old_by_rank
        ]
        new_ms = [
            row["device_work"]["kernel"]["duration_sum_ms"] for row in new_by_rank
        ]
        paired_ms = [new - old for old, new in zip(old_ms, new_ms, strict=True)]
        output[component] = {
            "calls_per_rank_per_variant": old_by_rank[0]["calls"],
            "baseline_median_kernel_count": median(
                row["device_work"]["kernel"]["count"] for row in old_by_rank
            ),
            "combined_median_kernel_count": median(
                row["device_work"]["kernel"]["count"] for row in new_by_rank
            ),
            "baseline_median_kernel_sum_ms": median(old_ms),
            "combined_median_kernel_sum_ms": median(new_ms),
            "median_change_of_rank_pairs_ms": median(paired_ms),
            "range_change_of_rank_pairs_ms": [min(paired_ms), max(paired_ms)],
            "median_change_of_rank_pairs_pct": median(
                (new - old) / old * 100 for old, new in zip(old_ms, new_ms, strict=True)
            ),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw", type=Path, help="parent of the four profile run directories"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ranks", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()
    if len(set(args.ranks)) != len(args.ranks) or any(
        not 0 <= rank < 8 for rank in args.ranks
    ):
        parser.error("ranks must be unique values in 0 through 7")

    tasks = []
    for rank in sorted(args.ranks):
        for ep in (8, 4):
            for variant in ("baseline", "combined"):
                run = args.raw / f"profile-10x1-mb32-001-ep{ep}-{variant}"
                path = (
                    run / f"native/profiling/traces/iteration_11/rank{rank}_trace.json"
                )
                tasks.append((path, variant, rank))

    results_by_ep = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(analyze_trace, tasks)):
            ep = ("ep8", "ep4")[(index % 4) // 2]
            results_by_ep[ep].append(result)
            if index < 4 and result["variant"] == "combined":
                baseline, combined = results_by_ep[ep]
                print(
                    json.dumps(
                        {
                            "group": ep,
                            "rank": result["rank"],
                            "summary": summarize_pair([baseline], [combined]),
                        }
                    ),
                    flush=True,
                )

    output = {
        ep: {"summary": summarize_pair(results[::2], results[1::2]), "traces": results}
        for ep, results in results_by_ep.items()
    }
    print(
        json.dumps({ep: result["summary"] for ep, result in output.items()}), flush=True
    )

    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
