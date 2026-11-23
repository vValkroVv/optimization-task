"""Measure only the resident CTA order of the pinned routed SwiGLU producer."""

import argparse
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
import linecache
from pathlib import Path
import platform
from random import Random
from statistics import mean, median, quantiles
from time import perf_counter_ns


FIXTURE_SHA256 = "4b3c072c701a76d65ce9360e34acae6ee4271c6a50afd39703bc774636929e0f"
FIXTURE_SHAPE = (63_616, 3_072)
PRODUCTION_KERNEL = "_swiglu_quant_fwd_kernel"
CANDIDATE_KERNEL = "_routed_swiglu_fwd_feature_first"


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def swapped_source(source):
    row = "pid_token = tl.program_id(axis=0)"
    col = "pid_hid = tl.program_id(axis=1)"
    definition = f"def {PRODUCTION_KERNEL}("
    if any(source.count(line) != 1 for line in (row, col, definition)) or (
        source.count("tl.program_id(") != 2
    ):
        raise ValueError("The production kernel no longer has the expected CTA mapping")
    return (
        source.replace(definition, f"def {CANDIDATE_KERNEL}(")
        .replace(row, "pid_token = tl.program_id(axis=1)")
        .replace(col, "pid_hid = tl.program_id(axis=0)")
    )


def make_candidate(kernel, jit):
    # Triton obtains the callable's source through inspect. Give the derived
    # kernel its own source identity and leave the production JIT object alone.
    source = swapped_source(kernel.src)
    filename = str(Path(__file__).with_suffix(".generated.py"))
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = kernel.fn.__globals__ | {"__name__": __name__}
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    return jit(namespace[CANDIDATE_KERNEL])


def make_launch(swiglu, kernel, grid, x, probs, output, scale, limit):
    args = swiglu.tensor_to_kernel_args
    width = output.shape[1]
    kwargs = swiglu.params_to_kernel_kwargs(
        swiglu.STATIC_TRITON_PARAMS_FWD, width, num_warps=4, maxnreg=160
    )
    operands = (
        *args(x, 2),
        *args(probs, 1),
        *args(output, 2),
        *args(scale, 2),
        x.shape[0] // swiglu._SWIGLU_TOKEN_BUCKET_SIZE,
        width,
    )

    def launch():
        kernel[grid](
            *operands,
            ROW_BLOCK=swiglu.FWD_ROW_BLOCK,
            COL_BLOCK=swiglu.COL_BLOCK,
            APPLY_CLAMP=limit > 0,
            swiglu_limit=limit,
            clip_counts_ptr=None,
            ACTIVATION_MONITOR_ON=False,
            **kwargs,
        )

    return launch


def check_parity(launches, swiglu, x, probs, output, scale, limit, torch):
    _, reference = swiglu.fused_swiglu_quantized_fn(None, x, probs, limit)
    expected = (reference._data, reference._scale)
    receipt = {}
    for arm, launch in launches.items():
        # Seed each destination slot with the complement of its expected bits.
        for dst, src, bit_dtype in zip(
            (output, scale), expected, (torch.uint8, torch.int32), strict=True
        ):
            dst.view(bit_dtype).copy_(src.view(bit_dtype)).bitwise_not_()
        launch()
        receipt[arm] = {
            name: {
                "shape_stride_dtype_equal": (
                    dst.shape == src.shape
                    and dst.stride() == src.stride()
                    and dst.dtype == src.dtype
                ),
                "mismatched_elements": int(
                    torch.count_nonzero(dst.view(bits) != src.view(bits)).item()
                ),
            }
            for name, dst, src, bits in zip(
                ("fp8", "scales"),
                (output, scale),
                expected,
                (torch.uint8, torch.int32),
                strict=True,
            )
        }
    return receipt


def capture_graphs(launches, warmup, calls, torch):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for launch in launches.values():
            for _ in range(warmup):
                launch()
    stream.synchronize()
    graphs = {}
    for arm, launch in launches.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls):
                launch()
        graphs[arm] = graph
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    return graphs


def measure_pairs(graphs, pairs, calls, torch, device_memory_usage):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    stop.record()
    stop.synchronize()

    def measure(arm):
        before = torch.cuda.memory_stats()
        torch.cuda.reset_peak_memory_stats()
        begin = perf_counter_ns()
        start.record()
        graphs[arm].replay()
        stop.record()
        submitted = perf_counter_ns()
        stop.synchronize()
        drained = perf_counter_ns()
        after = torch.cuda.memory_stats()
        if any(after[key] != before[key] for key in ("num_alloc_retries", "num_ooms")):
            raise RuntimeError("Allocator retry or OOM during timed replay")
        _, device_peak = device_memory_usage(after)
        return {
            "event_us_per_call": start.elapsed_time(stop) * 1_000 / calls,
            "synchronized_us_per_call": (drained - begin) / 1_000 / calls,
            "host_submit_us_per_graph": (submitted - begin) / 1_000,
            "host_wait_us_per_graph": (drained - submitted) / 1_000,
            "peak_allocated_bytes": after["allocated_bytes.all.peak"],
            "peak_reserved_bytes": after["reserved_bytes.all.peak"],
            "gift_device_peak_gib": device_peak,
        }

    results = []
    for index in range(pairs):
        order = ("A", "B", "A_control", "A_same")
        if index % 2:
            order = order[::-1]
        results.append(
            {
                "order": order,
                "arms": {
                    arm: measure("A" if arm.startswith("A") else "B") for arm in order
                },
            }
        )
    return results


def summarize(samples, reference, trial, field):
    control = [sample["arms"][reference][field] for sample in samples]
    candidate = [sample["arms"][trial][field] for sample in samples]
    deltas = [100 * (b / a - 1) for a, b in zip(control, candidate, strict=True)]
    rng = Random(1729)
    bootstrapped = [mean(rng.choices(deltas, k=len(deltas))) for _ in range(5_000)]
    percentiles = quantiles(bootstrapped, n=40, method="inclusive")
    return {
        "control_median": median(control),
        "candidate_median": median(candidate),
        "paired_mean_change_pct": mean(deltas),
        "paired_bootstrap_95_pct_interval": [percentiles[0], percentiles[-1]],
    }


def profile(launch, kernel, grid, path, torch):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        launch()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_bytes())["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    transfers = [
        event for event in events if event.get("cat") in ("gpu_memcpy", "gpu_memset")
    ]
    active = (
        len(kernels) == 1
        and kernel in kernels[0]["name"]
        and tuple(kernels[0]["args"].get("grid", ())) == (*grid, 1)
        and not transfers
    )
    return {
        "activation_verified": active,
        "kernels": [
            {"name": e["name"], "duration_us": e["dur"], "launch": e["args"]}
            for e in kernels
        ],
        "transfer_count": len(transfers),
        "path": str(path),
        "sha256": digest(path),
    }


def run(args, trace_paths):
    import torch
    import triton
    from torchtitan.ops.float8.triton_kernels import fused_swiglu_quantization as swiglu
    from torchtitan.tools.utils import device_memory_usage

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one Hopper GPU")
    torch.cuda.set_device(0)
    fixture_sha256 = digest(args.fixture)
    if fixture_sha256 != FIXTURE_SHA256:
        raise ValueError("Fixture differs from the captured EP8 rank 0 operands")
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True, mmap=True)
    x, probs = fixture["x"], fixture["p"].reshape(-1)
    if (
        x.shape != FIXTURE_SHAPE
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or probs.shape != (x.shape[0],)
        or not probs.is_floating_point()
    ):
        raise ValueError(
            "Expected packed BF16 gate/up and captured routing probabilities"
        )

    x, probs = x.cuda(), probs.contiguous().cuda()
    limit = float(fixture["limit"])
    rows, gate_up_width = x.shape
    width = gate_up_width // 2
    row_tiles, col_tiles = rows // swiglu.FWD_ROW_BLOCK, width // swiglu.COL_BLOCK
    assert (rows, width) == (
        row_tiles * swiglu.FWD_ROW_BLOCK,
        col_tiles * swiglu.COL_BLOCK,
    )
    grids = {"A": (row_tiles, col_tiles), "B": (col_tiles, row_tiles)}
    baseline = swiglu._swiglu_quant_fwd_kernel
    candidate = make_candidate(baseline, triton.jit)
    output = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((rows, col_tiles), dtype=torch.float32, device=x.device)
    launches = {
        arm: make_launch(swiglu, kernel, grids[arm], x, probs, output, scales, limit)
        for arm, kernel in (("A", baseline), ("B", candidate))
    }
    with torch.no_grad():
        parity = check_parity(launches, swiglu, x, probs, output, scales, limit, torch)
    exact = all(
        plane["shape_stride_dtype_equal"] and plane["mismatched_elements"] == 0
        for arm in parity.values()
        for plane in arm.values()
    )
    receipt = {
        "scope": "single resident routed SwiGLU forward producer with monitoring off",
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "triton": version("triton"),
        "cuda": torch.version.cuda,
        "runner_sha256": digest(Path(__file__)),
        "production_file_sha256": digest(Path(swiglu.__file__)),
        "production_kernel_sha256": sha256(baseline.src.encode()).hexdigest(),
        "candidate_kernel_sha256": sha256(candidate.src.encode()).hexdigest(),
        "input": {
            "kind": "captured EP8 rank 0",
            "path": str(args.fixture),
            "sha256": fixture_sha256,
        },
        "gate_up_shape": list(x.shape),
        "probability_dtype": str(probs.dtype),
        "swiglu_limit": limit,
        "grids": grids,
        "tile": [swiglu.FWD_ROW_BLOCK, swiglu.COL_BLOCK],
        "logical_min_bytes_per_call": sum(
            tensor.numel() * tensor.element_size()
            for tensor in (x, probs, output, scales)
        ),
        "parity_vs_production": parity,
        "bitwise": exact,
    }
    if exact:
        torch.cuda.empty_cache()
        graphs = capture_graphs(launches, args.warmup, args.replays, torch)
        measurements = measure_pairs(
            graphs, args.pairs, args.replays, torch, device_memory_usage
        )
        receipt["timing"] = {
            "scope": "one submission per resident graph, same preallocated operands and outputs",
            "warmup_launches_per_arm": args.warmup,
            "launches_per_graph": args.replays,
            "pairs": measurements,
            "peak_by_arm": {
                arm: {
                    field: max(pair["arms"][arm][field] for pair in measurements)
                    for field in (
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                        "gift_device_peak_gib",
                    )
                }
                for arm in ("A", "B")
            },
            "statistics": {
                label: {
                    field: summarize(measurements, ref, trial, field)
                    for field in ("event_us_per_call", "synchronized_us_per_call")
                }
                for label, ref, trial in (
                    ("AB", "A", "B"),
                    ("AA", "A_control", "A_same"),
                )
            },
        }
        receipt["profiles"] = {
            arm: profile(launches[arm], name, grids[arm], trace_paths[arm], torch)
            for arm, name in (("A", PRODUCTION_KERNEL), ("B", CANDIDATE_KERNEL))
        }
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    if not exact:
        raise RuntimeError(f"SwiGLU bitwise parity failed. See {args.output}")
    if not all(item["activation_verified"] for item in receipt["profiles"].values()):
        raise RuntimeError(f"Unexpected producer launch or transfer. See {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--replays", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=16)
    args = parser.parse_args()
    if args.warmup < 1 or args.replays < 1 or args.pairs < 4 or args.pairs % 2:
        parser.error(
            "Require positive warmup and replays and an even pair count of at least four"
        )
    trace_paths = {
        arm: args.output.with_suffix(f".{arm}.trace.json") for arm in ("A", "B")
    }
    if any(path.exists() for path in (args.output, *trace_paths.values())):
        parser.error("Specify a fresh output path")
    run(args, trace_paths)


if __name__ == "__main__":
    main()
