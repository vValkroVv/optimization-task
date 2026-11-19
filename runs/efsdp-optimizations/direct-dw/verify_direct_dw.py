"""Check native Hopper dW epilogue parity, storage, activation, and timing."""

import argparse
import json
from math import prod
from pathlib import Path
from statistics import median

import torch

from direct_dw import load_direct_dw, verify_baseline

SHAPES = ((16, 3072, 4096), (16, 4096, 1536))
FULL = (4096,) * 16
MIXED = (0, 4096, 3840, 0, 0, 3968, 4096, 128, 4096, 4096, 4096, 0, 4096, 4096, 4096, 0)
EMPTY = (0,) * 16
BASELINE_KERNEL = "sm90_fp8_gemm_1d1d_impl"
CANDIDATE_KERNEL = "sm90_k_grouped_fp8_gemm_rounded_f32_impl"
GUARD_SIZE = 256
GUARD_VALUE = -23.5


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


SPECIAL_FP32_BITS = (
    0x00000000, 0x80000000, 0x00000001, 0x80000001,  # signed zeros and smallest FP32 subnormals
    0x00008000, 0x80008000, 0x00008001, 0x80008001,  # halfway to a BF16 subnormal and above
    0x00010000, 0x80010000, 0x007F0000, 0x807F0000,  # extreme BF16 subnormals
    0x007F8000, 0x807F8000, 0x00800000, 0x80800000,  # min-normal boundary
    0x3F807FFF, 0xBF807FFF, 0x3F808000, 0xBF808000,  # round-to-nearest even, below and at tie
    0x3F808001, 0xBF808001, 0x3F818000, 0xBF818000,  # above tie and odd tie
    0x7F7F0000, 0xFF7F0000, 0x7F800000, 0xFF800000,  # largest finite BF16 and infinities
    0x7F800001, 0xFF800001, 0x7FC12345, 0xFFC12345,  # signaling and quiet NaNs
)


def make_operand(ks, width, device, generator):
    total_k = sum(ks)
    data = torch.empty(total_k * width, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty((total_k // 128, width), dtype=torch.float32, device=device)
    offset = 0
    for k in ks:
        if k:
            values = torch.randn((width, k), device=device, generator=generator)
            data[offset * width : (offset + k) * width].copy_(values.flatten())
            offset += k
    scales.uniform_(0.015625, 0.0625, generator=generator)
    return data, scales.mT


def make_slot(device):
    first_size, second_size = [prod(shape) for shape in SHAPES]
    slot = torch.empty(3 * GUARD_SIZE + first_size + second_size, device=device)
    first = slot.narrow(0, GUARD_SIZE, first_size).view(SHAPES[0])
    second = slot.narrow(0, 2 * GUARD_SIZE + first_size, second_size).view(SHAPES[1])
    canaries = (slot[:GUARD_SIZE], slot[GUARD_SIZE + first_size : 2 * GUARD_SIZE + first_size], slot[-GUARD_SIZE:])
    return slot, (first, second), canaries


def probe_values(module, device):
    ordered_bits = SPECIAL_FP32_BITS + SPECIAL_FP32_BITS[1:] + SPECIAL_FP32_BITS[:1]
    signed_bits = [bits if bits < 2**31 else bits - 2**32 for bits in ordered_bits]
    inputs = torch.tensor(signed_bits, device=device, dtype=torch.int32).view(torch.float32)
    reference = torch.zeros(64, device=device, dtype=torch.bfloat16)
    actual = torch.empty_like(inputs)
    module.probe_special_values(inputs, reference, actual)
    expected_bits = reference.to(torch.float32).view(torch.int32).tolist()
    actual_bits = actual.view(torch.int32).tolist()
    failures = [
        (f"{source:08x}", f"{expected & 0xffffffff:08x}", f"{observed & 0xffffffff:08x}")
        for source, expected, observed in zip(ordered_bits, expected_bits, actual_bits, strict=True)
        if observed != expected
    ]
    if failures:
        raise AssertionError(f"Special-value bit mismatches (FP32 input, legacy, direct): {failures}")
    return {
        "input_bits": [f"{bits:08x}" for bits in ordered_bits],
        "output_bits": [f"{bits & 0xffffffff:08x}" for bits in actual_bits],
    }


def peak_allocation(operation, device):
    torch.cuda.synchronize(device)
    initial_allocated = torch.cuda.memory_allocated(device)
    initial_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    operation()
    torch.cuda.synchronize(device)
    return {
        "incremental_allocated_bytes": torch.cuda.max_memory_allocated(device) - initial_allocated,
        "initial_reserved_bytes": initial_reserved,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def trace_operation(operation, device, path):
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        operation()
        torch.cuda.synchronize(device)
    profile.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [
        {"name": event["name"], "duration_us": event["dur"]}
        for event in events
        if event.get("cat") == "kernel" and "dur" in event
    ]
    return {
        "kernels": kernels,
        "foreach_copy_cpu": sum(event.get("name") == "aten::_foreach_copy_" for event in events),
        "device_memsets": [
            {"name": event["name"], "duration_us": event["dur"]}
            for event in events
            if event.get("cat") == "gpu_memset" and "dur" in event
        ],
    }


def graph_timing(legacy, direct, device, samples):
    # Replay excludes the original Python and dispatcher work. Scratch allocations
    # are captured in a graph-private pool and the inputs remain resident.
    plan = {"legacy": legacy, "legacy_control": legacy, "direct": direct}
    graphs = {}
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        for name, operation in plan.items():
            operation()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                operation()
            graphs[name] = graph
    torch.cuda.synchronize(device)
    elapsed = {name: [] for name in graphs}
    for trial in range(samples):
        names = tuple(plan) if trial % 2 == 0 else tuple(reversed(plan))
        for name in names:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record()
                graphs[name].replay()
                stop.record()
            stop.synchronize()
            elapsed[name].append(start.elapsed_time(stop))
    return {
        "scope": "single-stream CUDA event span per warm CUDA Graph replay, milliseconds",
        "raw_ms": elapsed,
        "median_ms": {name: median(times) for name, times in elapsed.items()},
    }


def compare_case(name, ks, baseline, candidate, device, evidence, samples):
    generator = torch.Generator(device=device).manual_seed(1944)
    # Both dW producers use the same expert counts but different operand widths.
    operands = [
        (make_operand(ks, m, device, generator), make_operand(ks, n, device, generator))
        for _, m, n in SHAPES
    ]
    legacy_slot, legacy_views, _ = make_slot(device)
    candidate_slot, candidate_views, canaries = make_slot(device)
    legacy_slot.fill_(GUARD_VALUE)
    candidate_slot.fill_(GUARD_VALUE)
    gpu_counts = torch.tensor(ks, device=device, dtype=torch.int32)
    k_sizes = list(ks)

    def legacy():
        temporaries = [torch.zeros(shape, device=device, dtype=torch.bfloat16) for shape in SHAPES]
        if any(ks):
            for (a, b), tmp in zip(operands, temporaries, strict=True):
                baseline.k_grouped_fp8_gemm_nt_contiguous(a, b, tmp, k_sizes, gpu_counts, tmp)
        torch._foreach_copy_(legacy_views, temporaries)

    def direct():
        for (a, b), output in zip(operands, candidate_views, strict=True):
            candidate.k_grouped_fp8_gemm_nt_contiguous_rounded_f32(a, b, output, k_sizes, gpu_counts)

    legacy()
    direct()
    for shape, actual, expected in zip(SHAPES, candidate_views, legacy_views, strict=True):
        if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
            raise AssertionError(f"{name} differs bitwise on {shape}")
        for group, k in enumerate(ks):
            if k == 0 and torch.count_nonzero(actual[group].view(torch.int32)).item():
                raise AssertionError(f"{name} changed bits in the +0 contract for expert {group}")
    for canary in canaries:
        if torch.any(canary != GUARD_VALUE).item():
            raise AssertionError(f"{name} overwrote a slot canary")

    old_memory = peak_allocation(legacy, device)
    new_memory = peak_allocation(direct, device)
    full_bf16_bytes = 2 * sum(map(prod, SHAPES))
    if old_memory["incremental_allocated_bytes"] < full_bf16_bytes:
        raise AssertionError(f"{name} baseline failed to account for both BF16 temporaries")
    if new_memory["incremental_allocated_bytes"] >= min(2 * prod(shape) for shape in SHAPES):
        raise AssertionError(f"{name} allocated a full-gradient-sized temporary")

    old_trace = trace_operation(legacy, device, evidence / f"{name}-legacy.trace.json")
    new_trace = trace_operation(direct, device, evidence / f"{name}-direct.trace.json")
    expected_gemms = len(SHAPES) if any(ks) else 0
    for receipt, marker in ((old_trace, BASELINE_KERNEL), (new_trace, CANDIDATE_KERNEL)):
        count = sum(marker in kernel["name"] for kernel in receipt["kernels"])
        if count != expected_gemms:
            raise AssertionError(f"{name} trace expected {expected_gemms} {marker} launches, found {count}")
    if old_trace["foreach_copy_cpu"] != 1 or new_trace["foreach_copy_cpu"]:
        raise AssertionError(f"{name} expected one legacy packing call and no candidate packing call")

    elapsed = graph_timing(legacy, direct, device, samples) if samples and any(ks) else None
    return {
        "k_sizes": ks,
        "shape_offsets_elements": {str(tuple(view.shape)): view.storage_offset() for view in candidate_views},
        "bitwise_match": True,
        "legacy_memory": old_memory,
        "direct_memory": new_memory,
        "legacy_trace": old_trace,
        "direct_trace": new_trace,
        "graph_timing": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--num-sms", type=positive_int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=0, help="Optional warm graph replays per variant and case")
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    baseline = verify_baseline()
    baseline.set_num_sms(args.num_sms)
    candidate = load_direct_dw(args.build_root, num_sms=args.num_sms)

    result = {
        "device": torch.cuda.get_device_name(device),
        "num_sms": args.num_sms,
        "torch_version": torch.__version__,
        "deep_gemm_version": baseline.__version__,
        "build": json.loads((args.build_root / "manifest.json").read_text()),
        "special_values": probe_values(candidate, device),
    }
    for name, sizes in (("full", FULL), ("mixed", MIXED), ("empty", EMPTY)):
        result[name] = compare_case(name, sizes, baseline, candidate, device, args.evidence_dir, args.samples)
        (args.evidence_dir / "component.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: bitwise parity, empty experts, canaries, allocation and activation passed", flush=True)


if __name__ == "__main__":
    main()
