"""Find the active allocator state at the replay's first measured peak."""

import json
import pickle
from pathlib import Path
import sys

import torch

import frontier_stack
import replay


def allocated_at_start(snapshot):
    blocks = {}
    for segment in snapshot["segments"]:
        for block in segment["blocks"]:
            if block["state"] != "active_allocated":
                continue
            blocks[block["address"]] = (block["size"], block["frames"])
    return blocks


def as_mib(num_bytes):
    return round(num_bytes / 2**20, 3)


def summarize(start, end, start_stats, end_stats, roles):
    blocks = allocated_at_start(start)
    allocated = sum(size for size, _ in blocks.values())
    assert allocated == start_stats["allocated_bytes.all.current"]
    peak_bytes, peak_blocks, peak_index = allocated, blocks.copy(), -1
    history = end["device_traces"][0]
    for index, event in enumerate(history):
        action = event["action"]
        if action == "alloc":
            addr = event["addr"]
            size = ((event["size"] + 511) // 512) * 512
            blocks[addr] = (size, event.get("frames", []))
            allocated += size
        elif action == "free_requested":
            size, _ = blocks.pop(event["addr"])
            allocated -= size
        if allocated > peak_bytes:
            peak_bytes, peak_blocks, peak_index = allocated, blocks.copy(), index
    assert peak_bytes == end_stats["allocated_bytes.all.peak"], (peak_bytes, end_stats["allocated_bytes.all.peak"])
    assert allocated == end_stats["allocated_bytes.all.current"]
    largest = sorted(peak_blocks.items(), key=lambda item: item[1][0], reverse=True)[:20]
    frames = history[peak_index].get("frames", []) if peak_index >= 0 else []

    def active_frames(stack):
        return [
            {"file": frame["filename"], "function": frame["name"], "line": frame["line"]}
            for frame in stack
            if "lm-head-mtp" in frame["filename"] or "torchtitan" in frame["filename"] or "torch/_inductor" in frame["filename"]
        ][:8]

    return {
        "start_allocated_mib": as_mib(start_stats["allocated_bytes.all.current"]),
        "allocated_peak_mib": as_mib(peak_bytes),
        "reserved_peak_mib": as_mib(end_stats["reserved_bytes.all.peak"]),
        "reserve_at_end_mib": as_mib(end_stats["reserved_bytes.all.current"]),
        "event_count": len(history),
        "allocator_settings": start["allocator_settings"],
        "peak_frames": active_frames(frames),
        "largest_peak_blocks": [
            {"mib": as_mib(size), "frames": active_frames(stack), "roles": roles.get(addr, [])}
            for addr, (size, stack) in largest
        ],
    }


def main():
    path = Path(sys.argv[3])
    reset = torch.cuda.reset_peak_memory_stats
    synchronize = torch.cuda.synchronize
    start_snapshot = None
    evidence = None
    label_by_address = {}
    head = None
    native_make_head = replay.make_head

    def capture_head(weight):
        nonlocal head
        head = native_make_head(weight)
        return head

    def tensor_roles():
        (group,) = head._get_fsdp_state()._fsdp_param_groups
        for role, tensor in {
            "FP32 local parameter": group._padded_sharded_param_data[None],
            "FP32 local accumulated gradient": group._padded_sharded_param_grad[None],
            "FP32 unsharded reduce-scatter input": group._reduce_scatter_input[None],
            "BF16 all-gather input": group._all_gather_input[None],
            "BF16 all-gather output": group._all_gather_output[None],
        }.items():
            label_by_address.setdefault(tensor.untyped_storage().data_ptr(), []).append(role)

    def start(*args, **kwargs):
        nonlocal start_snapshot, start_stats
        reset(*args, **kwargs)
        if start_snapshot is None:
            torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=8192, clear_history=True)
            tensor_roles()
            start_snapshot = torch.cuda.memory._snapshot()
            start_stats = torch.cuda.memory_stats()

    def finish(*args, **kwargs):
        nonlocal evidence
        synchronize(*args, **kwargs)
        if start_snapshot is not None and evidence is None:
            end_snapshot = torch.cuda.memory._snapshot()
            end_stats = torch.cuda.memory_stats()
            torch.cuda.memory._record_memory_history(enabled=None)
            evidence = {"start": start_snapshot, "end": end_snapshot}
            path.with_suffix(".pickle").write_bytes(pickle.dumps(evidence))
            analysis = summarize(start_snapshot, end_snapshot, start_stats, end_stats, label_by_address)
            path.with_suffix(".memory.json").write_text(json.dumps(analysis, indent=2) + "\n")

    start_stats = None
    torch.cuda.reset_peak_memory_stats = start
    torch.cuda.synchronize = finish
    replay.make_head = capture_head
    frontier_stack.main()
    assert evidence is not None


if __name__ == "__main__":
    main()
