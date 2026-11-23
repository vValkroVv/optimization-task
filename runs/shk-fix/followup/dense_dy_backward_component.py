"""Replay two native dense FP8 backward cuts through dgrad and wgrad."""

import argparse
import json
from math import prod
from pathlib import Path
import platform

import torch
import triton

import dense_dy_quant_transpose as quant_probe


LANES = {"w2": (4096, 1536), "qkv": (8192, 4096)}
TOKEN_SHAPE = (2, quant_probe.NATIVE_ROWS // 2)
PLANES = {
    "row_fp8": (torch.float8_e4m3fn, torch.uint8),
    "row_scale": (torch.float32, torch.int32),
    "column_fp8": (torch.float8_e4m3fn, torch.uint8),
    "column_scale": (torch.float32, torch.int32),
    "dX": (torch.bfloat16, torch.int16),
    "dW": (torch.bfloat16, torch.int16),
}
COMPILE = {"backend": "inductor", "fullgraph": True, "dynamic": False}


def load_inputs(fixture, dy_width, x_width):
    shapes = {
        "dy": (*TOKEN_SHAPE, dy_width),
        "x": (*TOKEN_SHAPE, x_width),
        "weight": (dy_width, x_width),
    }
    if fixture is None:
        seed = quant_probe.SEED
        generator = torch.Generator(device="cpu").manual_seed(seed)
        std = {"dy": 2.0**-10, "x": 1.0, "weight": x_width**-0.5}
        tensors = {
            name: (torch.randn(shape, generator=generator) * std[name]).to(
                torch.bfloat16
            )
            for name, shape in shapes.items()
        }
        provenance = {"kind": "synthetic", "seed": seed, "normal_std": std}
    else:
        tensors = torch.load(fixture, map_location="cpu", weights_only=True, mmap=True)
        provenance = {
            "kind": "supplied BF16 operands",
            "path": str(fixture),
            "sha256": quant_probe.digest(fixture),
            "source_shapes": {name: list(tensors[name].shape) for name in shapes},
            "derivation": tensors.get("provenance"),
        }
    inputs = {}
    for name, shape in shapes.items():
        tensor = tensors[name]
        supported = {shape}
        if len(shape) == 3:
            supported.add((quant_probe.NATIVE_ROWS, shape[-1]))
        if (
            tensor.dtype != torch.bfloat16
            or tuple(tensor.shape) not in supported
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Expected contiguous BF16 {name} of shape {shape}")
        inputs[name] = tensor.view(shape).to("cuda")
    return inputs, provenance


def compile_arm(native, layouts, lane, *, fused, expose_planes):
    quantize = native.quantize_vec128
    quantize_weight = native.quantize_blk128x128
    transpose = native.scaling_aware_fp8_transpose
    gemm = native.fp8_gemm

    transpose_x = lane == "w2"

    def backward(inputs):
        dy, weight, x = inputs
        weight_col = quantize_weight(weight, transpose_output=True, pack=False)
        if fused:
            row_data, row_scale, col_data, col_scale = quant_probe.candidate(
                dy.view(-1, dy.shape[-1])
            )
            dy_row = (row_data.view(dy.shape), row_scale, layouts.ROWWISE, 1)
            dy_col = (
                col_data.view(dy.shape[-1], *dy.shape[:-1]),
                col_scale,
                layouts.COLUMNWISE,
                1,
            )
        else:
            dy_row = quantize(dy, pack=False)

        if transpose_x:
            x_col = transpose(x, pack=False)
            if not fused:
                dy_col = transpose(dy_row, pack=False)
        else:
            # QKV reuses the columnwise activation already available at the cut.
            x_col = x
        dx = gemm(
            dy_row, weight_col, out_dtype=torch.bfloat16, use_split_accumulator=True
        )
        if not transpose_x and not fused:
            dy_col = transpose(dy_row, pack=False)
        dw = gemm(x_col, dy_col, out_dtype=torch.bfloat16, use_split_accumulator=True)
        if expose_planes:
            return (*dy_row[:2], *dy_col[:2], dx, dw)
        return dx, dw

    return torch.compile(backward, **COMPILE, options={"triton.cudagraphs": False})


def parity(left, right, expected_shapes):
    checks = {}
    for name, a, b in zip(expected_shapes, left, right, strict=True):
        dtype, bits = PLANES[name]
        metadata_equal = (
            a.dtype == b.dtype == dtype
            and tuple(a.shape) == tuple(b.shape) == expected_shapes[name]
            and a.stride() == b.stride()
        )
        checks[name] = {
            "shape": list(a.shape),
            "metadata_equal": metadata_equal,
            "mismatched_elements": (
                int(torch.count_nonzero(a.view(bits) != b.view(bits)))
                if metadata_equal
                else None
            ),
        }
    return checks


def inspect_schedule(events, arm, lane):
    dy_width, x_width = LANES[lane]
    events = sorted(
        (event for event in events if event.get("ph") == "X"),
        key=lambda event: event["ts"],
    )
    weight_op = "cudacherry::quantize_blk128x128"
    row_op = "cudacherry::quantize_vec128"
    transpose_op = "cudacherry::scaling_aware_fp8_transpose"
    gemm_op = "torchtitan::fp8_dense_gemm"
    operands = []
    for event in events:
        name = event["name"]
        dims = event.get("args", {}).get("Input Dims", [])
        if name in {weight_op, row_op, transpose_op}:
            operands.append((name, dims[:1]))
        elif name == gemm_op:
            operands.append((name, dims[:1] + dims[4:5]))

    expected_operands = [(weight_op, [[dy_width, x_width]])]
    if arm == "A":
        expected_operands.append((row_op, [[quant_probe.NATIVE_ROWS, dy_width]]))
    transpose_dy = (transpose_op, [[*TOKEN_SHAPE, dy_width]])
    if lane == "w2":
        expected_operands.append((transpose_op, [[*TOKEN_SHAPE, x_width]]))
        if arm == "A":
            expected_operands.append(transpose_dy)
    expected_operands.append((gemm_op, [[*TOKEN_SHAPE, dy_width], [x_width, dy_width]]))
    if lane == "qkv" and arm == "A":
        expected_operands.append(transpose_dy)
    expected_operands.append(
        (gemm_op, [[x_width, *TOKEN_SHAPE], [dy_width, *TOKEN_SHAPE]])
    )

    kernels = [event for event in events if event.get("cat") == "kernel"]
    markers = {
        "weight": "quantize_blk128x128_kernel_t_computed_scale",
        "row": "quantize_vec128_kernel",
        "fused": "dense_dy_quant_transpose_kernel",
        "transpose": "scaling_aware_fp8_transpose_kernel",
    }
    kernel_order = []
    other_count = 0
    for event in kernels:
        role = next(
            (role for role, marker in markers.items() if marker in event["name"]),
            "other",
        )
        if role == "other":
            other_count += 1
            # cuBLASLt may launch auxiliary kernels in a GEMM interval.
            if kernel_order and kernel_order[-1] == role:
                continue
        kernel_order.append(role)
    expected_kernel_order = ["weight", "row" if arm == "A" else "fused"]
    if lane == "w2":
        expected_kernel_order.extend(["transpose"] * (2 if arm == "A" else 1))
    expected_kernel_order.append("other")
    if lane == "qkv" and arm == "A":
        expected_kernel_order.extend(("transpose", "other"))
    copies = [e for e in events if e.get("cat") in {"gpu_memcpy", "gpu_memset"}]
    return {
        "activation_verified": operands == expected_operands
        and kernel_order == expected_kernel_order
        and other_count >= 2,
        "native_ops_and_data_shapes": operands,
        "expected_native_ops_and_data_shapes": expected_operands,
        "kernel_order": kernel_order,
        "expected_kernel_order": expected_kernel_order,
        "kernels": [{"name": e["name"], "duration_us": e["dur"]} for e in kernels],
        "transfers_or_memsets": [
            {"name": e["name"], "duration_us": e["dur"]} for e in copies
        ],
    }


def profile(operation, inputs, arm, lane, path):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profiler:
        result = operation(inputs)
        torch.cuda.synchronize()
    del result
    profiler.export_chrome_trace(str(path))
    report = inspect_schedule(json.loads(path.read_bytes())["traceEvents"], arm, lane)
    report.update(path=str(path), sha256=quant_probe.digest(path))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=LANES, required=True)
    parser.add_argument(
        "--fixture", type=Path, help="BF16 dy, x and weight tensor dict"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    traces = {arm: args.output.with_suffix(f".{arm}.trace.json") for arm in ("A", "B")}
    if any(path.exists() for path in (args.output, *traces.values())):
        parser.error("Specify a fresh output path")
    if args.warmup < 1 or args.samples < 1 or args.repeats < 2:
        parser.error("Require positive warmup and samples and at least two repeats")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one Hopper GPU")
    torch.cuda.set_device(0)

    from torchtitan.ops.float8 import _cudacherry as native
    from torchtitan.ops.float8.fp8_tensor import InputLayoutType
    from torchtitan.tools.utils import device_memory_usage
    from cudacherry._loader import load_family

    vendors = {}
    for family in ("quantize", "gemm"):
        binary = load_family(family).__file__
        vendors[family] = {"path": binary, "sha256": quant_probe.digest(binary)}
    dy_width, x_width = LANES[args.lane]
    bf16, source = load_inputs(args.fixture, dy_width, x_width)
    shapes = {name: list(tensor.shape) for name, tensor in bf16.items()}
    x = native.quantize_vec128(bf16.pop("x"), pack=False)
    if args.lane == "qkv":
        x = native.scaling_aware_fp8_transpose(x, pack=False)
    inputs = (bf16["dy"], bf16["weight"], x)
    outputs = {
        "row_fp8": (*TOKEN_SHAPE, dy_width),
        "row_scale": (dy_width // quant_probe.TILE, quant_probe.NATIVE_ROWS),
        "column_fp8": (dy_width, *TOKEN_SHAPE),
        "column_scale": (quant_probe.NATIVE_ROWS // quant_probe.TILE, dy_width),
        "dX": (*TOKEN_SHAPE, x_width),
        "dW": (dy_width, x_width),
    }
    timed, debug = (
        {
            arm: compile_arm(
                native,
                InputLayoutType,
                args.lane,
                fused=arm == "B",
                expose_planes=expose_planes,
            )
            for arm in ("A", "B")
        }
        for expose_planes in (False, True)
    )
    gradients = {name: outputs[name] for name in ("dX", "dW")}
    debug_parity = parity(debug["A"](inputs), debug["B"](inputs), outputs)
    timed_parity = parity(timed["A"](inputs), timed["B"](inputs), gradients)
    exact = all(
        result["metadata_equal"] and result["mismatched_elements"] == 0
        for result in (*debug_parity.values(), *timed_parity.values())
    )
    receipt = {
        "lane": args.lane,
        "scope": "Local weight quantization through both GEMMs. The standalone "
        "caller holds BF16 dY and weight throughout. Independent training work, "
        "collectives and overlap are outside the cut.",
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gift_bridge": str(Path(native.__file__).resolve()),
        "candidate_sha256": quant_probe.digest(quant_probe.__file__),
        "cudacherry_reference_revision": quant_probe.VENDOR_REFERENCE,
        "cudacherry_extensions": vendors,
        "input": source,
        "bf16_input_shapes": shapes,
        "preexisting_x": {
            "layout": "rowwise" if args.lane == "w2" else "columnwise",
            "shapes": [list(plane.shape) for plane in x[:2]],
        },
        "compile": {**COMPILE, "cudagraphs": False},
        "gemm": {"out_dtype": "bfloat16", "use_split_accumulator": True},
        "debug_parity": debug_parity,
        "timed_parity": timed_parity,
        "bitwise": exact,
        "candidate_column_plane_bytes": sum(
            prod(outputs[name]) * PLANES[name][0].itemsize
            for name in ("column_fp8", "column_scale")
        ),
        "warmup_per_arm": args.warmup,
        "paired_measurements": [],
    }
    active = False
    if exact:
        torch._dynamo.config.error_on_recompile = True
        for operation in timed.values():
            for _ in range(args.warmup):
                operation(inputs)
            torch.cuda.synchronize()
        receipt["profiles"] = {
            arm: profile(operation, inputs, arm, args.lane, traces[arm])
            for arm, operation in timed.items()
        }
        active = all(
            report["activation_verified"] for report in receipt["profiles"].values()
        )
    if active:
        for repeat in range(args.repeats):
            order = ["A", "B"] if repeat % 2 == 0 else ["B", "A"]
            arms = {}
            for arm in order:
                torch.cuda.empty_cache()
                for _ in range(4):
                    timed[arm](inputs)
                block = quant_probe.measure(timed[arm], inputs, args.samples)
                ext_gib, total_gib = device_memory_usage(torch.cuda.memory_stats())
                block.update(non_torch_gib=ext_gib, device_peak_gib=total_gib)
                arms[arm] = block
            delta = {
                key: arms["B"]["medians"][key] - value
                for key, value in arms["A"]["medians"].items()
            }
            receipt["paired_measurements"].append(
                {"order": order, "arms": arms, "candidate_minus_control_ms": delta}
            )
            print(
                f"Repeat {repeat + 1}, candidate minus control ms {delta}", flush=True
            )
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    if not exact or not active:
        raise RuntimeError(
            f"Dense backward parity or schedule verification failed. See {args.output}"
        )


if __name__ == "__main__":
    main()
