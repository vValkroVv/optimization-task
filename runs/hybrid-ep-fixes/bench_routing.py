"""Replay the standalone post-capacity HybridEP metadata builder on one GPU."""

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from functools import partial
from pathlib import Path

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from torchtitan.distributed.deepep.routing import indices_to_map as reference
from torchtitan.experiments.four_fusion import routing_metadata as current


EXPERT_COUNTS = (64, 128, 256)
NATIVE_SHAPE = (8192, 128, 8)
SIZE_SWEEP = (NATIVE_SHAPE, (16384, 128, 8), (8192, 64, 8), (8192, 256, 8))
TOP_K_SWEEP = ((8192, 128, 4), NATIVE_SHAPE, (8192, 128, 16))


@gluon.jit
def grouped_rows(
    ids, scores, flags, probs, tokens: gl.constexpr, rows: gl.constexpr,
    warps: gl.constexpr, experts: gl.constexpr, top_k: gl.constexpr,
):
    # Give each warp a token row and tile the expert columns over its lanes.
    layout: gl.constexpr = gl.BlockedLayout(
        [1, experts // 32], [1, 32], [warps, 1], [1, 0]
    )
    block = gl.program_id(0)
    token = block.to(gl.int64) * rows + gl.arange(0, rows, layout=gl.SliceLayout(1, layout))
    expert = gl.arange(0, experts, layout=gl.SliceLayout(0, layout))
    active = token < tokens
    route = gl.full((rows, experts), False, gl.int1, layout)
    weight = gl.full((rows, experts), 0.0, gl.float32, layout)
    for slot in gl.static_range(top_k):
        input_offset = token * top_k + slot
        raw_id = gl.load(ids + input_offset, active, -1)
        # Keep all negative IDs inactive, including values outside int32.
        expert_id = gl.where(raw_id < 0, -1, raw_id).to(gl.int32)
        score = gl.load(scores + input_offset, active, 0).to(gl.float32)
        hit = expert[None, :] == expert_id[:, None]
        route = route | hit
        weight = gl.where(hit, score[:, None], weight)
    offset = token[:, None] * experts + expert[None, :]
    gl.store(flags + offset, route, active[:, None])
    gl.store(probs + offset, weight, active[:, None])
    if block == 0:
        gl.store(flags + tokens * experts, False)
        gl.store(probs + tokens * experts, 0.0)


def output_buffers(tokens, experts, device):
    entries = tokens * experts
    return (
        torch.empty(entries + 1, dtype=torch.bool, device=device),
        torch.empty(entries + 1, dtype=torch.float32, device=device),
    )


def grouped(ids, scores, experts, *, rows, warps):
    assert experts in EXPERT_COUNTS
    assert ids.is_contiguous() and scores.is_contiguous()
    tokens, top_k = ids.shape
    flags, probs = output_buffers(tokens, experts, ids.device)
    grouped_rows[(max(1, triton.cdiv(tokens, rows)),)](
        ids, scores, flags, probs, tokens, rows, warps, experts, top_k, num_warps=warps
    )
    return current._metadata_views(flags, probs, tokens, experts)


@torch.library.custom_op(
    "hybrid_ep_bench::grouped_metadata", mutates_args=(), device_types="cuda"
)
def grouped_op(
    ids: torch.Tensor, scores: torch.Tensor, experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return grouped(ids, scores, experts, rows=8, warps=4)


def wrapped_grouped(ids, scores, experts):
    if (
        not current._can_use_cuda_kernel(ids, scores, experts)
        or experts not in EXPERT_COUNTS
        or not ids.is_contiguous()
        or not scores.is_contiguous()
    ):
        return current.indices_to_map(ids, scores, experts)
    return grouped_op(ids, scores, experts)


def one_warp(ids, scores, experts):
    tokens, top_k = ids.shape
    flags, probs = output_buffers(tokens, experts, ids.device)
    current._routing_metadata_kernel[(max(tokens, 1),)](
        ids, scores, flags, probs, tokens, top_k, experts,
        *ids.stride(), *scores.stride(), triton.next_power_of_2(experts), num_warps=1,
    )
    return current._metadata_views(flags, probs, tokens, experts)


def load_native_inputs(directory):
    """Verify the native group gate, then recover the metadata operands."""
    records = []
    operands = []
    global_counts = torch.zeros(128, dtype=torch.int64)
    for rank in range(8):
        path = directory / f"rank{rank}.pt"
        fixture = torch.load(path, map_location="cpu", weights_only=True)
        assert fixture["rank"] == rank and fixture["num_experts"] == 128
        assert fixture["source_quota"] == 6250
        assert fixture["capacity_sha256"] == (
            "2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120"
        )
        ids, scores = fixture["expert_ids"], fixture["scores"]
        assert ids.dtype == torch.int64 and scores.dtype == torch.float32
        assert ids.shape == scores.shape == (8192, 8)
        assert ids.is_contiguous() and scores.is_contiguous()
        assert ((ids >= 0) & (ids < 128)).all() and (scores != 0).all()
        counts = torch.bincount(ids.flatten(), minlength=128)
        global_counts += counts
        operands.append((ids, scores))
        records.append({
            "rank": rank,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "largest_expert_count": int(counts.max()),
        })
    # Production group gating requires a global destination group above quota × E.
    group_counts = global_counts.view(8, 16).sum(dim=1).tolist()
    group_budget = 6250 * 128
    assert max(group_counts) <= group_budget
    skew_rank = max(range(1, 8), key=lambda rank: records[rank]["largest_expert_count"])
    return {
        "directory": str(directory),
        "rank_receipts": records,
        "global_destination_counts": group_counts,
        "group_budget": group_budget,
        "dropped_routes": 0,
    }, [(rank, *operands[rank]) for rank in (0, skew_rank)]


def resolve_duplicate_ids(ids, experts, first_slot):
    for slot in range(first_slot, ids.shape[1]):
        previous_slots = ids[:, :slot]
        candidate = ids[:, slot]
        # At most `slot` earlier IDs are occupied in any row.
        for _ in range(slot):
            occupied = (candidate[:, None] == previous_slots).any(dim=1)
            candidate = (candidate + occupied).remainder(experts)
        ids[:, slot] = candidate


def fixture_cases(fixtures, shapes):
    for tokens, experts, top_k in shapes:
        for index, (rank, ids, scores) in enumerate(fixtures):
            other_rank, other_ids, other_scores = fixtures[1 - index]
            ranks = [rank]
            adaptation = "captured native routes"
            if tokens == 16384:
                ids = torch.cat((ids, other_ids))
                scores = torch.cat((scores, other_scores))
                ranks.append(other_rank)
                adaptation = "concatenate two native rank samples"
            if experts == 64:
                ids = ids.remainder(experts)
                resolve_duplicate_ids(ids, experts, first_slot=1)
                adaptation = "fold IDs modulo 64 and advance duplicates to the next free ID"
            if experts == 256:
                bank_offset = 128 * (torch.arange(ids.shape[1]) % 2)
                ids = ids + bank_offset
                adaptation = "split alternating top-k slots across two 128-expert banks"
            if top_k == 4:
                scores, slots = scores.topk(top_k, dim=1)
                ids = ids.gather(1, slots)
                adaptation = "retain the four highest-scoring captured slots"
            if top_k == 16:
                native_k = ids.shape[1]
                ids = torch.cat((ids, other_ids), dim=1)
                scores = torch.cat((scores, other_scores), dim=1)
                resolve_duplicate_ids(ids, experts, first_slot=native_k)
                ranks.append(other_rank)
                adaptation = "append the other rank's eight routes and resolve ID collisions"
            assert ids.shape == scores.shape == (tokens, top_k)
            assert ((ids >= 0) & (ids < experts)).all()
            sorted_ids = ids.sort(dim=1).values
            assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()
            yield ranks, experts, adaptation, ids, scores


def assert_parity(expected, actual):
    for index, (a, b) in enumerate(zip(expected, actual, strict=True)):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.stride() == b.stride() and b.storage_offset() == 0
        assert b.untyped_storage().nbytes() >= (b.numel() + 1) * b.element_size()
        a_flat = a.as_strided((a.numel() + 1,), (1,))
        b_flat = b.as_strided((b.numel() + 1,), (1,))
        if index == 1:
            a_flat = a_flat.view(torch.int32)
            b_flat = b_flat.view(torch.int32)
        assert torch.equal(a_flat, b_flat), f"bit mismatch in output {index}"


def check_sentinels(variants, experts, top_k):
    ids = torch.arange(top_k, device="cuda").repeat(2, 1)
    ids[0, :2] = ids.new_tensor([-1, -(1 << 32)])
    ids[:, -1] = experts - 1
    scores = torch.ones_like(ids, dtype=torch.float32)
    scores[0, :4] = scores.new_tensor([float("nan"), 1.0, -0.0, float("nan")])
    scores[1, 0] = 0.0
    expected = reference(ids, scores, experts)
    for name in ("current", "grouped_8x128_wrapped"):
        assert_parity(expected, variants[name](ids, scores, experts))


def capture(fn, batch, *, distinct_outputs=False):
    graph = torch.cuda.CUDAGraph()
    held_outputs = []
    with torch.cuda.graph(graph):
        for _ in range(batch):
            if distinct_outputs:
                held_outputs.append(fn())
            else:
                fn()
    if distinct_outputs:
        assert len({pair[0].data_ptr() for pair in held_outputs}) == batch
        assert len({pair[1].data_ptr() for pair in held_outputs}) == batch
    return graph, held_outputs


def graph_us(graph, batch):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(12):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (batch * 12)


def host_us(fn, batch):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(batch):
        fn()
    submitted = time.perf_counter_ns()
    torch.cuda.synchronize()
    end = time.perf_counter_ns()
    return {
        "enqueue_us": (submitted - start) / (batch * 1000),
        "completion_wait_us": (end - submitted) / (batch * 1000),
        "integrated_us": (end - start) / (batch * 1000),
    }


def profile(fn, batch):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiler:
        for _ in range(batch):
            fn()
        torch.cuda.synchronize()
    events = [e for e in profiler.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not events:
        raise RuntimeError("CUDA profiler returned no device activity")
    device = {name: [] for name in ("kernels", "copies", "memsets")}
    for event in events:
        lower = event.name.lower()
        category = "copies" if "memcpy" in lower else "memsets" if "memset" in lower else "kernels"
        device[category].append(event)
    return {
        category: {
            "sum_us_per_call": sum(e.time_range.elapsed_us() for e in selected) / batch,
            "names_and_counts": dict(Counter(e.name for e in selected)),
        }
        for category, selected in device.items()
    }


def timed_case(variants, ids, scores, experts, batch):
    expected = reference(ids, scores, experts)
    bound = {name: partial(fn, ids, scores, experts) for name, fn in variants.items()}
    for name, fn in bound.items():
        print(f"Warmup and parity: {name}", flush=True)
        assert_parity(expected, fn())
        for _ in range(12):
            fn()
    torch.cuda.synchronize()
    graphs = {name: capture(fn, batch) for name, fn in bound.items()}
    for graph, _ in graphs.values():
        graph_us(graph, batch)
    result = {name: {"graph_us": [], "host": []} for name in bound}
    names = list(bound)
    for repeat in range(8):
        ordered = names if repeat % 2 == 0 else names[::-1]
        for name in ordered:
            result[name]["graph_us"].append(graph_us(graphs[name][0], batch))
            result[name]["host"].append(host_us(bound[name], batch))
    del graphs
    for name, fn in bound.items():
        values = result[name]
        values["medians"] = {"graph_us": statistics.median(values["graph_us"])}
        for key in values["host"][0]:
            values["medians"][key] = statistics.median(sample[key] for sample in values["host"])
        try:
            values["profile"] = profile(fn, 8)
        except RuntimeError as exc:
            values["profile_error"] = str(exc)
        print(name, values["medians"], flush=True)
    return result


def time_distinct_outputs(variants, ids, scores, experts, batch):
    graphs = {
        name: capture(partial(fn, ids, scores, experts), batch, distinct_outputs=True)
        for name, fn in variants.items()
    }
    result = {name: [] for name in graphs}
    for graph, _ in graphs.values():
        graph_us(graph, batch)
    names = list(graphs)
    for repeat in range(8):
        for name in names if repeat % 2 == 0 else names[::-1]:
            result[name].append(graph_us(graphs[name][0], batch))
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument("--size-sweep", action="store_true", help="Sweep token and expert counts")
    sweep.add_argument("--top-k-sweep", action="store_true", help="Sweep top-4, top-8 and top-16")
    args = parser.parse_args()
    shapes = TOP_K_SWEEP if args.top_k_sweep else SIZE_SWEEP if args.size_sweep else (NATIVE_SHAPE,)
    compiled_reference = torch.compile(reference, fullgraph=True, dynamic=False)
    variants = {
        "baseline_eager": reference,
        "baseline_eager_repeat": reference,
        "baseline_compiled": compiled_reference,
        "current": current.indices_to_map,
        "grouped_8x128_wrapped": wrapped_grouped,
    }
    if not (args.size_sweep or args.top_k_sweep):
        variants.update({
            "current_one_warp_direct": one_warp,
            "grouped_4x128_direct": partial(grouped, rows=4, warps=4),
            "grouped_8x128_direct": partial(grouped, rows=8, warps=4),
        })
    provenance, fixtures = load_native_inputs(args.fixture_dir)
    sentinel_shapes = sorted({(experts, top_k) for _, experts, top_k in shapes})
    for experts, top_k in sentinel_shapes:
        check_sentinels(variants, experts, top_k)
    result = {
        "input_provenance": provenance,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "triton": triton.__version__,
        "cuda_build": torch.version.cuda,
        "sentinel_parity": "negative int64 IDs, valid zero, signed zero and NaN",
        "sentinel_experts_top_k": sentinel_shapes,
        "sources_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(current.__file__), Path(reference.__code__.co_filename))
        },
        "batch": 64,
        "distinct_outputs_batch": 24,
        "cases": [],
    }
    for ranks, experts, adaptation, cpu_ids, cpu_scores in fixture_cases(fixtures, shapes):
        ids, scores = cpu_ids.cuda(), cpu_scores.cuda()
        case = {
            "fixture_rank": ranks[0], "fixture_ranks": ranks, "input_adaptation": adaptation,
            "shape": list(ids.shape), "experts": experts,
            "input_types": [str(ids.dtype), str(scores.dtype)],
            "input_strides": [list(ids.stride()), list(scores.stride())],
        }
        print("Case:", case, flush=True)
        case["variants"] = timed_case(variants, ids, scores, experts, result["batch"])
        distinct = {
            name: fn for name, fn in variants.items()
            if name in ("baseline_eager", "baseline_compiled", "current", "grouped_8x128_wrapped")
        }
        case["distinct_outputs_graph_us"] = time_distinct_outputs(
            distinct, ids, scores, experts, result["distinct_outputs_batch"]
        )
        print("Distinct output buffers:", {
            name: statistics.median(samples)
            for name, samples in case["distinct_outputs_graph_us"].items()
        }, flush=True)
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
