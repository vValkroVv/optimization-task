"""Compare compiled row quantization and FTZ transpose with one Hopper kernel."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns

import torch
import triton
import triton.language as tl


NATIVE_ROWS = 8192
SEED = 1729
VENDOR_REFERENCE = "d3e0febbf5bf4a08c7f2d9499e96eaddced5e510"
TILE = 128
PLANES = {
    "row_fp8": torch.uint8,
    "row_scale": torch.int32,
    "column_fp8": torch.uint8,
    "column_scale": torch.int32,
}


@triton.jit
def dense_dy_quant_transpose_kernel(
    X,
    QRow,
    SRow,
    QCol,
    SCol,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)
    offsets = tl.arange(0, 128)
    columns = block_n.to(tl.int64) * 128 + offsets
    column_scale_ptr = SCol + block_m.to(tl.int64) * COLS + columns
    if ROWS % 512 == 0 or block_m < ROWS // 128:
        rows = block_m.to(tl.int64) * 128 + offsets
        row_offsets = rows[:, None] * COLS + columns[None, :]
        x = tl.load(X + row_offsets).to(tl.float32)

        # fmaxf ignores NaNs. The floor gives the same scale for an all-NaN row.
        magnitude = x.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        maximum = tl.max(tl.where(magnitude > 0x7F800000, 0, magnitude), axis=1)
        raw_scale = tl.inline_asm_elementwise(
            "mul.rn.ftz.f32 $0, $1, 0f3B124925;",  # FP32 reciprocal of 448
            "=f,f",
            [maximum.to(tl.float32, bitcast=True)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        raw_bits = tl.maximum(raw_scale, 1.0e-30).to(tl.int32, bitcast=True)
        scale_bits = (raw_bits & 0x7F800000) + tl.where(
            (raw_bits & 0x007FFFFF) != 0, 0x00800000, 0
        )
        scale = scale_bits.to(tl.float32, bitcast=True)
        reciprocal = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            [scale],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        # Match CudaCherry's FTZ multiply and paired SATFINITE conversion,
        # including Inf*0 and signed zero.
        row_bytes = tl.inline_asm_elementwise(
            """
            {
                .reg .f32 scaled<4>;
                .reg .b16 lo, hi;
                mul.rn.ftz.f32 scaled0, $1, $5;
                mul.rn.ftz.f32 scaled1, $2, $6;
                mul.rn.ftz.f32 scaled2, $3, $7;
                mul.rn.ftz.f32 scaled3, $4, $8;
                cvt.rn.satfinite.e4m3x2.f32 lo, scaled1, scaled0;
                cvt.rn.satfinite.e4m3x2.f32 hi, scaled3, scaled2;
                mov.b32 $0, {lo, hi};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            [x, reciprocal[:, None]],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )
        tl.store(QRow + row_offsets, row_bytes)
        tl.store(SRow + block_n.to(tl.int64) * ROWS + rows, scale)

        target_bits = tl.max(scale_bits, axis=0)
        shift = (target_bits - scale_bits) >> 23
        values = row_bytes.to(tl.int32)
        exponent = (values >> 3) & 15
        shifted = tl.where(exponent > shift[:, None], values - (shift[:, None] << 3), 0)
        column_bytes = tl.where(
            shift[:, None] == 0, tl.where((values & 127) != 0, values, 0), shifted
        ).to(tl.uint8)
        transposed_offsets = columns[:, None] * ROWS + rows[None, :]
        tl.store(QCol + transposed_offsets, tl.trans(column_bytes))
        tl.store(column_scale_ptr, target_bits.to(tl.float32, bitcast=True))
    else:
        tl.store(column_scale_ptr, 0.0)


def candidate(x):
    rows, cols = x.shape
    padded_row_blocks = triton.cdiv(rows // TILE, 4) * 4
    row_data = torch.empty((rows, cols), dtype=torch.uint8, device=x.device)
    row_scale = torch.empty((cols // TILE, rows), dtype=torch.float32, device=x.device)
    col_data = torch.empty((cols, rows), dtype=torch.uint8, device=x.device)
    col_scale = torch.empty(
        (padded_row_blocks, cols), dtype=torch.float32, device=x.device
    )
    torch.library.wrap_triton(dense_dy_quant_transpose_kernel)[
        (padded_row_blocks, cols // TILE)
    ](x, row_data, row_scale, col_data, col_scale, rows, cols, num_warps=8)
    return (
        row_data.view(torch.float8_e4m3fn),
        row_scale,
        col_data.view(torch.float8_e4m3fn),
        col_scale,
    )


def digest(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def compile_pair(quantize, transpose):
    def control(x):
        row = quantize(x, pack=False)
        col = transpose(row, pack=False)
        return row[0], row[1], col[0], col[1]

    return {
        name: torch.compile(
            operation,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        for name, operation in (("A", control), ("B", candidate))
    }


def load_native(path, key, shape):
    if path is None:
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        x = torch.randn(shape, generator=generator) * (2.0**-10)
        return x.to(device="cuda", dtype=torch.bfloat16), {
            "kind": "synthetic",
            "seed": SEED,
            "distribution": "normal, std 2**-10",
        }
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    x = payload if isinstance(payload, torch.Tensor) else payload[key]
    if (
        x.dtype != torch.bfloat16
        or tuple(x.shape) not in (shape, (2, shape[0] // 2, shape[1]))
        or not x.is_contiguous()
    ):
        raise ValueError("Expected native contiguous BF16 dense dY")
    return x.view(shape).to("cuda"), {
        "kind": "supplied dense dY",
        "path": str(path),
        "key": key,
        "fixture_sha256": digest(path),
        "source_shape": list(x.shape),
    }


def edge_cases():
    # One occurrence of every BF16 encoding, including padded column scales.
    all_bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    yield "all_bf16_encodings", all_bits.view(torch.bfloat16).view(256, 256).to("cuda")

    row = torch.arange(128, dtype=torch.float32)
    cols = torch.arange(256, dtype=torch.float32)
    scale = torch.exp2(row.remainder(65) - 32)[:, None]
    x = ((cols.remainder(31) - 15) * scale).to(torch.bfloat16)
    x[:, :128].clamp_(-448, 448)
    x[:, 0] = 448
    x[:, 1] = -0.0
    x[:3, 128] = torch.tensor([446, 448, 450], dtype=torch.bfloat16)
    x[0, 2:6] = torch.tensor(
        [2.0**-10, 2.0**-9, -(2.0**-9), -(2.0**-10)], dtype=torch.bfloat16
    )
    x[5, 128:] = float("nan")
    x[6, 128] = float("inf")
    x[7, 128] = -float("inf")
    x[8, 128:] = -0.0
    x[9, 128:] = 0.0
    yield "mixed_scale_ftz_and_special_values", x.to("cuda")


def compare(pair, x):
    left, right = (pair[arm](x) for arm in ("A", "B"))
    result = {}
    for (label, bit_dtype), a, b in zip(PLANES.items(), left, right, strict=True):
        metadata = (a.dtype, a.shape, a.stride()) == (b.dtype, b.shape, b.stride())
        result[label] = {
            "metadata_equal": metadata,
            "mismatched_elements": (
                int(torch.count_nonzero(a.view(bit_dtype) != b.view(bit_dtype)))
                if metadata
                else None
            ),
        }
    return result


def measure(operation, x, count):
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    stop.record()
    stop.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_stats()
    samples = []
    for _ in range(count):
        begin = perf_counter_ns()
        start.record()
        result = operation(x)
        stop.record()
        submitted = perf_counter_ns()
        stop.synchronize()
        drained = perf_counter_ns()
        samples.append(
            {
                "cuda_event_ms": start.elapsed_time(stop),
                "host_submission_ms": (submitted - begin) / 1e6,
                "host_drain_ms": (drained - submitted) / 1e6,
                "wall_ms": (drained - begin) / 1e6,
            }
        )
        del result
    after = torch.cuda.memory_stats()
    if any(before[key] != after[key] for key in ("num_alloc_retries", "num_ooms")):
        raise RuntimeError("Unexpected allocator retry or OOM during measurement")
    return {
        "samples": samples,
        "medians": {key: median(row[key] for row in samples) for key in samples[0]},
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def trace(operation, x, arm, path):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        result = operation(x)
        torch.cuda.synchronize()
    del result
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_bytes())["traceEvents"]
    kernels = sorted(
        (event for event in events if event.get("cat") == "kernel"),
        key=lambda event: event["ts"],
    )
    expected = (
        ("quantize_vec128_kernel", "scaling_aware_fp8_transpose_kernel")
        if arm == "A"
        else ("dense_dy_quant_transpose_kernel",)
    )
    active = len(kernels) == len(expected) and all(
        marker in event["name"] for marker, event in zip(expected, kernels, strict=True)
    )
    copies = [e for e in events if e.get("cat") in {"gpu_memcpy", "gpu_memset"}]
    busy = 0.0
    start = tail = kernels[0]["ts"] if kernels else 0.0
    for event in kernels:
        end = event["ts"] + event["dur"]
        busy += max(0.0, end - max(tail, event["ts"]))
        tail = max(tail, end)
    span = tail - start
    return {
        "activation_verified": active and not copies,
        "kernels": [{"name": e["name"], "duration_us": e["dur"]} for e in kernels],
        "kernel_sum_us": sum(e["dur"] for e in kernels),
        "kernel_union_us": busy,
        "kernel_interval_span_us": span,
        "kernel_interval_gaps_us": max(0.0, span - busy),
        "copy_memset_count": len(copies),
        "copy_memset_sum_us": sum(e["dur"] for e in copies),
        "path": str(path),
        "sha256": digest(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--fixture-key", default="tangent")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(4096, 8192), default=4096)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    trace_paths = {
        arm: args.output.with_suffix(f".{arm}.trace.json") for arm in ("A", "B")
    }
    if any(path.exists() for path in (args.output, *trace_paths.values())):
        parser.error("Specify a fresh output path")
    if args.warmup < 1 or args.samples < 1 or args.repeats < 2:
        parser.error("Require positive warmup and samples and at least two repeats")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one Hopper GPU")
    torch.cuda.set_device(0)

    from torchtitan.ops.float8._cudacherry import (
        quantize_vec128,
        scaling_aware_fp8_transpose,
    )
    from cudacherry._loader import load_family
    from torchtitan.tools.utils import device_memory_usage

    vendor = load_family("quantize")
    shape = (NATIVE_ROWS, args.width)
    x, provenance = load_native(args.fixture, args.fixture_key, shape)
    pair = compile_pair(quantize_vec128, scaling_aware_fp8_transpose)
    parity = {"native_shape": compare(pair, x)}
    for label, edge in edge_cases():
        parity[label] = compare(pair, edge)
    exact = all(
        plane["metadata_equal"] and plane["mismatched_elements"] == 0
        for case in parity.values()
        for plane in case.values()
    )
    receipt = {
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "cudacherry_reference_revision": VENDOR_REFERENCE,
        "cudacherry_extension": str(vendor.__file__),
        "cudacherry_extension_sha256": digest(vendor.__file__),
        "input": provenance,
        "shape": list(shape),
        "tile": [TILE, TILE],
        "num_warps": 8,
        "compile": {"backend": "inductor", "fullgraph": True, "cudagraphs": False},
        "parity": parity,
        "bitwise": exact,
    }
    if exact:
        torch._dynamo.config.error_on_recompile = True
        for operation in pair.values():
            for _ in range(args.warmup):
                operation(x)
            torch.cuda.synchronize()
        rounds = []
        for repeat in range(args.repeats):
            order = ["A", "B"] if repeat % 2 == 0 else ["B", "A"]
            arms = {}
            for arm in order:
                torch.cuda.empty_cache()
                for _ in range(4):
                    pair[arm](x)
                block = measure(pair[arm], x, args.samples)
                _, block["device_peak_gib"] = device_memory_usage(
                    torch.cuda.memory_stats()
                )
                arms[arm] = block
            delta = {
                key: arms["B"]["medians"][key] - value
                for key, value in arms["A"]["medians"].items()
            }
            rounds.append(
                {"order": order, "arms": arms, "candidate_minus_control_ms": delta}
            )
            print(
                f"Repeat {repeat + 1}, candidate minus control ms {delta}", flush=True
            )
        receipt["warmup_per_arm"] = args.warmup
        receipt["paired_measurements"] = rounds
        receipt["profiles"] = {
            arm: trace(pair[arm], x, arm, trace_paths[arm]) for arm in pair
        }
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    if not exact:
        raise RuntimeError(f"Dense dY bitwise parity failed. See {args.output}")
    if not all(
        profile["activation_verified"] for profile in receipt["profiles"].values()
    ):
        raise RuntimeError(
            f"Unexpected kernel launches or transfers. See {args.output}"
        )


if __name__ == "__main__":
    main()
