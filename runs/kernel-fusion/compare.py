"""Summarize the measured optimizer window and rank-0 update-21 kernel chains."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

from run import HERE, ROOT, SWITCHES, load


def read_json(path):
    return json.loads(path.read_bytes())


def timing(summary):
    window = summary["window"]
    if (
        summary["status"] != "complete"
        or window["warmup_updates"] != 10
        or window["updates"] != 10
    ):
        raise ValueError("Expected a complete 10+10 run")
    cadence = summary["host_boundary_cadence"]["by_rank"]
    by_step = list(
        zip(
            *(
                [row["interval_s"] for row in cadence[str(rank)]["intervals"]]
                for rank in range(8)
            ),
            strict=True,
        )
    )
    maximum_rank_seconds = [max(row) for row in by_step]
    return {
        "global_physical_tokens_per_second": summary["tokens"][
            "physical_input_tokens_per_second_global"
        ],
        "total_physical_tokens": summary["tokens"]["physical_input_tokens_global"],
        "synchronized_window_seconds": window["primary_elapsed_s"],
        "host_cadence_max_rank_seconds": maximum_rank_seconds,
        "last_five_median_percent_vs_first_five": 100
        * (
            statistics.median(maximum_rank_seconds[5:])
            / statistics.median(maximum_rank_seconds[:5])
            - 1
        ),
        "max_rank_peak_gib": summary["memory"]["maximum_peak_gib_across_ranks"],
        "allocator_counter_deltas_by_rank": summary["memory"][
            "allocator_counter_deltas_by_rank"
        ],
        "compiler_observer": summary["compilation_observer"]["status"],
    }


def short_conv_chains(raw, analyzer):
    device, apis, owners = [], [], []
    anchors = {
        "torchtitan::fla_vjp": "_qkv_bwd_pack_kernel",
        "torchtitan::gdn_short_conv_backward": "_gdn_short_conv_backward_kernel",
    }
    for index, row in enumerate(raw):
        if row.get("ph") != "X":
            continue
        category = row.get("cat")
        if category in analyzer.GPU:
            collection = device
        elif category in analyzer.API:
            collection = apis
        elif category == "cpu_op" and row["name"] in anchors:
            collection = owners
        else:
            continue
        collection.append(
            analyzer.Event(
                index,
                row["name"],
                category,
                analyzer.ns(row["ts"]),
                analyzer.ns(row["dur"]),
                row["pid"],
                row["tid"],
                row.get("args", {}),
            )
        )

    submits, _ = analyzer.launches(raw, device, apis, single_process_device=True)
    ancestors = analyzer.ancestors(apis, owners)
    activities = defaultdict(list)
    for event, host in submits.items():
        for scope in ancestors[host]:
            activities[scope].append(event)
    groups = defaultdict(list)
    spans, device_ns, host_ns = [], 0, []
    for scope, events in activities.items():
        if not any(anchors[scope.name] in event.name for event in events):
            continue
        spans.append(max(event.end for event in events) - min(event.start for event in events))
        host_ns.append(scope.dur)
        device_ns += sum(event.dur for event in events)
        for event in events:
            name = event.name
            if "reduce_kernel" in name:
                name = "FP32 weight partial reduction"
            elif "vectorized_elementwise_kernel" in name:
                name = "Weight gradient cast"
            groups[name].append(raw[event.index])
    if not spans:
        raise ValueError("No backward short-convolution scopes have attributed GPU work")
    return summarize_groups(groups), {
        "count": len(spans),
        "device_sum_ms": analyzer.ms(device_ns),
        "device_elapsed_sum_ms": analyzer.ms(sum(spans)),
        "device_elapsed_median_ms": statistics.median(spans) / 1_000_000,
        "host_scope_median_ms": statistics.median(host_ns) / 1_000_000,
    }


def kernel_groups(variant, events):
    """Identify candidates and the device work they replace by symbol and layout."""
    streams = defaultdict(list)
    residual_op_ids = set()
    for event in events:
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            streams[event["tid"]].append(event)
        if (
            variant == "cute-residual"
            and event.get("cat") == "cpu_op"
            and event.get("name") == "torchtitan::cute_gated_norm_residual_fwd"
        ):
            external_id = event["args"].get("External id")
            if external_id is not None:
                residual_op_ids.add(external_id)
    groups = defaultdict(list)
    for stream in streams.values():
        stream.sort(key=lambda row: row["ts"])
        for i, row in enumerate(stream):
            name = row["name"]
            prev = stream[i - 1]["name"] if i else ""
            successor = stream[i + 1]["name"] if i + 1 < len(stream) else ""
            arguments = row["args"]
            grid = arguments.get("grid", [])
            block = arguments.get("block", [])

            def add(label):
                groups[label].append(row)

            if variant == "dense-pad":
                if "constant_pad_nd_quantize_blk128x128" in name and grid[:1] == [
                    44032
                ]:
                    add("BF16 weight pad")
                if (
                    "quantize_blk128x128_kernel" in name
                    and 86 in grid[:2]
                    and 32 in grid[:2]
                ):
                    add("CudaCherry weight quantization")
            elif variant in {"gdn-output-fp8", "mla-gate-fp8"}:
                prefix = "gdn" if variant == "gdn-output-fp8" else "mla"
                middle = (
                    "mul_quantize_vec128_view"
                    if prefix == "gdn"
                    else "mul_quantize_vec128_sigmoid"
                )
                candidate = (
                    "gdn_output_norm_quantize_kernel"
                    if prefix == "gdn"
                    else "mla_gate_quantize"
                )
                if candidate in name:
                    add("fused output quantization")
                if (
                    middle in name
                    and "quantize_vec128_kernel" in successor
                    and (prefix == "mla" or "layer_norm_gated_fwd_kernel" in prev)
                ):
                    if prefix == "gdn":
                        groups["gated RMSNorm"].append(stream[i - 1])
                    add("pointwise at BF16 boundary")
                    groups["CudaCherry activation quantization"].append(stream[i + 1])
            elif variant == "moe-dual-fp8":
                if "_swiglu_quant_bwd_kernel" in name:
                    add(f"SwiGLU backward with {block[0]} threads per CTA")
                if "requantize_grouped_kernel" in name:
                    add(
                        "w13 gradient requantization"
                        if grid[1:2] == [24]
                        else "other grouped requantization"
                    )
            elif variant == "cute-residual":
                if (
                    "GnFwdOut" in name
                    and arguments.get("External id") in residual_op_ids
                ):
                    add("CuTe fused norm and residual")
                elif "GnFwdOut" in name and "644096" in name:
                    add("CuTe norm output (width 4096)")
                    if (
                        i + 1 < len(stream)
                        and "fused" in successor
                        and (
                            re.search(
                                r"_fused_(?:_to_copy_)?add(?:_\d+$|_cute_gated_norm_fwd_view)",
                                successor,
                            )
                            and stream[i + 1]["args"].get("grid", [])[:1] == [32768]
                        )
                    ):
                        add("CuTe norm immediately before a residual add")
                        groups["width-4096 residual add after norm output"].append(
                            stream[i + 1]
                        )
            elif variant == "mla-rope":
                if "_mla_rope_pack" in name or "_mla_rope_unpack" in name:
                    add(name.partition("<")[0])
                if "view_as_complex" in name:
                    add("Inductor complex-input preparation")
                if "c10::complex<float>" in name and "MulFunctor" in name:
                    add("ATen complex multiplication")
                if (
                    "scaled_dot_product_cudnn_attention__to_copy__unsafe_view_cat_expand_mul_split"
                    in name
                ):
                    add("Inductor final cuDNN input packing")

    return summarize_groups(groups)


def summarize_groups(groups):
    return {
        name: {
            "count": len(group),
            "sum_ms": sum(event["dur"] for event in group) / 1_000,
            "examples": [
                {
                    "name": event["name"],
                    "external_id": event["args"].get("External id"),
                    "stream": event["tid"],
                    "duration_us": event["dur"],
                }
                for event in group[:3]
            ],
        }
        for name, group in sorted(groups.items())
    }


def profile(path, variant):
    data = path.read_bytes()
    analyzer = load(ROOT / "runs/compile-baseline/analyze.py", "_fusion_trace_analyzer")
    whole = analyzer.analyze(path, 21, 0, 3)
    events = read_json(path)["traceEvents"]
    result = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "rank0_three_microbatches": whole["integrated"],
    }
    if variant == "gdn-short-conv":
        groups, scopes = short_conv_chains(events, analyzer)
        result["targeted_kernel_groups"] = groups
        result["targeted_device_scopes"] = scopes
    else:
        result["targeted_kernel_groups"] = kernel_groups(variant, events)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=SWITCHES)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = timing(read_json(args.baseline_summary))
    candidate = timing(read_json(args.candidate_summary))
    key = "global_physical_tokens_per_second"
    if baseline["total_physical_tokens"] != candidate["total_physical_tokens"]:
        raise ValueError("The runs timed different physical token counts")
    output = {
        "variant": args.variant,
        "throughput_delta_percent": 100 * (candidate[key] / baseline[key] - 1),
        "unprofiled_baseline": baseline,
        "unprofiled_candidate": candidate,
        "profiled_baseline": profile(args.baseline_trace, args.variant),
        "profiled_candidate": profile(args.candidate_trace, args.variant),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
