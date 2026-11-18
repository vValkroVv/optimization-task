#!/usr/bin/env python3
"""Reproduce a conditional removal estimate from accepted saved MCP exports."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def exclusive_ms(selected, others):
    others = merge(others)
    total = 0
    cursor = 0
    for start, end in merge(selected):
        while cursor < len(others) and others[cursor][1] <= start:
            cursor += 1
        index, position = cursor, start
        while index < len(others) and others[index][0] < end:
            left, right = others[index]
            total += max(0, min(end, left) - position)
            position = max(position, right)
            if position >= end:
                break
            index += 1
        total += max(0, end - position)
    return total / 1e6


def query_rows(data, key):
    pages = data["queries"][key]
    if not isinstance(pages, list):
        pages = [pages]
    result = []
    for page in pages:
        assert not page.get("isError", False)
        decoded = json.loads(page["content"][0]["text"])
        assert not decoded.get("truncated", False)
        result.extend(dict(zip(decoded["columns"], row)) for row in decoded["rows"])
    return result


def analyze(path, rank):
    source_bytes = path.read_bytes()
    data = json.loads(source_bytes)
    scopes = query_rows(data, "scopes")
    scope_by_id = {row["id"]: row for row in scopes}
    device = query_rows(data, "device")
    assert len({row["id"] for row in device}) == len(device)
    assert all(row["category"] in {"kernel", "gpu_memcpy", "gpu_memset"} for row in device)
    microbatches = sorted(
        [row for row in scopes if row["name"] == f"task.profile_update.21.microbatch.rank{rank}"
         and row["category"] == "user_annotation"], key=lambda row: row["ts"])
    assert len(microbatches) == 3
    pre_backward = [row for row in scopes if row["category"] == "user_annotation"
                    and re.match(r"GFSDP::pre_backward \(", row["name"])]
    blocks = collections.defaultdict(list)
    for row in device:
        match = re.search(r"(\d+):CompiledFunctionBackward(?: >|$)", row["ancestors"] or "")
        if match:
            blocks[int(match[1])].append(row)
    selected = {"core_moe": {}, "mla": {}}
    for block_id, events in blocks.items():
        timestamp = scope_by_id[block_id]["ts"]
        preceding = [row for row in pre_backward if row["ts"] <= timestamp]
        if not preceding:
            continue
        owner = max(preceding, key=lambda row: row["ts"])["name"].split("(")[1].split(")")[0]
        match = re.match(r"layers\.(\d+)", owner)
        layer = int(match[1]) if match else None
        gradient_ep_start = min(
            [row["launch_ts"] for row in events
             if "hybridep::dispatch_bwd" in (row["ancestors"] or "")] + [10**30])
        for row in events:
            ancestors, name = row["ancestors"] or "", row["name"]
            category = None
            if layer in (1, 2, 3, 4, 6, 7):
                for operation in ("dispatch", "combine"):
                    if re.search(r"hybridep::" + operation + r"(?: >|$)", ancestors):
                        category = operation
                        break
                if "deep_gemm::sm90_fp8_gemm_1d2d_impl" in name and row["launch_ts"] < gradient_ep_start:
                    if "0u, 4096u, 1536u" in name:
                        category = "down"
                    elif layer == 1 and "0u, 3072u, 4096u" in name:
                        category = "up"
            if category:
                selected["core_moe"][row["id"]] = (row, f"L{layer}_{category}")
            if (layer in (3, 7) or owner.startswith("mtp.layers.0")) and "aten::_cudnn_attention_forward" in ancestors:
                selected["mla"][row["id"]] = (row, "MTP" if owner.startswith("mtp") else f"L{layer}")
    selected["combined"] = {**selected["core_moe"], **selected["mla"]}
    result = {"rank": rank, "source": str(path), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "device_event_count": len(device), "groups": {}}
    for group, chosen in selected.items():
        others = [(row["ts"], row["ts"] + row["dur"]) for row in device if row["id"] not in chosen]
        batches = []
        for number, scope in enumerate(microbatches):
            events = [(row, category) for row, category in chosen.values()
                      if scope["ts"] <= row["launch_ts"] < scope["ts"] + scope["dur"]]
            intervals = [(row["ts"], row["ts"] + row["dur"]) for row, _ in events]
            assert len(events) == {"core_moe": 115, "mla": 6, "combined": 121}[group]
            batches.append({"microbatch": number, "host_scope_id": scope["id"],
                            "selected_event_ids": sorted(row["id"] for row, _ in events),
                            "selection_counts": dict(collections.Counter(category for _, category in events)),
                            "device_categories": dict(collections.Counter(row["category"] for row, _ in events)),
                            "sum_ms": sum(row["dur"] for row, _ in events) / 1e6,
                            "union_ms": sum(end - start for start, end in merge(intervals)) / 1e6,
                            "exclusive_ms": exclusive_ms(intervals, others)})
        result["groups"][group] = {"microbatches": batches,
            "total_union_ms": sum(row["union_ms"] for row in batches),
            "total_exclusive_ms": sum(row["exclusive_ms"] for row in batches)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [analyze(args.input_dir / f"se-004-rank{rank}.mcp.json", rank) for rank in (0, 1)]
    baseline, tokens, accumulation = 12.8479317002, 2097152, 32
    estimates = []
    for result in results:
        for label, group, metric in (("core_moe_exclusive", "core_moe", "exclusive"),
                                     ("combined_exclusive", "combined", "exclusive"),
                                     ("combined_full_union", "combined", "union")):
            removal = result["groups"][group][f"total_{metric}_ms"] / 3 * accumulation / 1000
            duration = baseline - removal
            estimates.append({"rank": result["rank"], "scenario": label,
                "conditional_update_removal_s": removal, "conditional_update_s": duration,
                "conditional_tps": tokens / duration, "conditional_tps_gain_percent": (baseline / duration - 1) * 100})
    output = {"method": "Fixed-timeline selected-device interval removal; all original streams; no dependency resimulation",
        "scope": "Accepted SE update21 ranks0/1, three of32 microbatches; optimizer outside capture",
        "selection_provenance": {
            "ownership": "Device launch ancestor CompiledFunctionBackward; preceding GFSDP::pre_backward on host identifies layer; launch_ts assigns microbatch including asynchronous tails",
            "core_moe": "L1,2,3,4,6,7 hybridep::dispatch/combine ancestors (not *_bwd), plus DeepGEMM forward down shape4096x1536 before first dispatch_bwd; additional L1 up shape3072x4096",
            "mla": "L3,L7,MTP aten::_cudnn_attention_forward launch ancestors, including associated memsets; enclosing aten::_scaled_dot_product_cudnn_attention present",
            "exclusive": "Selected interval union minus union of ALL unselected device events (kernel, gpu_memcpy, gpu_memset), including NCCL; GPU annotations excluded"},
        "baseline_update_s": baseline, "tokens_per_update": tokens, "microbatches_per_update": accumulation,
        "baseline_tps": tokens / baseline,
        "formulas": {"removal_s": "sum_of_three_captured_ms / 3 * 32 / 1000",
                     "conditional_update_s": "baseline_update_s - removal_s",
                     "conditional_tps": "tokens_per_update / conditional_update_s"},
        "limitations": ["Conditional extrapolation assumes all32 microbatches match captured averages",
                        "Complete CUDA event and host dependencies are unavailable; exclusive device occupancy is not exact critical path",
                        "Removing work can change peer waiting, scheduling and contention; mandatory backward remains",
                        "No native speedup, convergence or memory acceptance claim"],
        "ranks": results, "conditional_tps_table": estimates}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "conditional_tps_table": estimates}, indent=2))


if __name__ == "__main__":
    main()
