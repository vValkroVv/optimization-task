#!/usr/bin/env python3
"""Compare padded and logical w2 wgrad widths on the captured dense fixture."""

import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter_ns

import torch

from torchtitan.ops.float8._cudacherry import (
    fp8_gemm,
    quantize_blk128x128,
    quantize_vec128,
    scaling_aware_fp8_transpose,
    swiglu_quantize,
)
from torchtitan.ops.float8.fp8_utils import unpad_tensor_if_needed


HIDDEN = 10944
PADDED_HIDDEN = 11008
DIM = 4096
SHAPE = (2, 4096, DIM)


def load_inputs(path):
    fixture = torch.load(path, map_location="cpu", weights_only=True)
    if fixture["schema_version"] != 1 or fixture["case"] != "dense-fp8":
        raise ValueError("expected the captured dense-fp8 fixture")
    x = fixture["input"].to("cuda")
    dy = fixture["tangent"].to("cuda")
    gate, down, up = (
        fixture["weights"][f"{name}.weight"] for name in ("w1", "w2", "w3")
    )
    expected = (
        (x, SHAPE), (dy, SHAPE), (gate, (HIDDEN, DIM)),
        (down, (DIM, HIDDEN)), (up, (HIDDEN, DIM)),
    )
    if any(
        tensor.shape != shape
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
        for tensor, shape in expected
    ):
        raise ValueError("dense fixture tensor shape, dtype, or layout changed")

    # Rebuild the padded fused SwiGLU boundary using the accepted weight-tail path.
    x_fp8 = quantize_vec128(x, pack=True)
    gate_fp8 = quantize_blk128x128(gate.to("cuda"), pack=False)
    up_fp8 = quantize_blk128x128(up.to("cuda"), pack=False)
    gate_output = fp8_gemm(x_fp8, gate_fp8, out_dtype=x.dtype)
    up_output = fp8_gemm(x_fp8, up_fp8, out_dtype=x.dtype)
    activation = swiglu_quantize(
        gate_output,
        up_output,
        dtype=torch.float8_e4m3fn,
        fp8_input_store=True,
        power_two_max_round=True,
        orig_shape=(*SHAPE[:-1], HIDDEN),
    )
    if activation.shape != (*SHAPE[:-1], PADDED_HIDDEN):
        raise ValueError(f"unexpected padded SwiGLU output shape {activation.shape}")
    return activation, quantize_vec128(dy, pack=False)


def logical_columns(operand):
    data, scale, layout, block = operand
    # Columnwise FP8 stores features on the first axis. Its prefix is already
    # contiguous. cuBLASLt infers the vector-scale row stride from this extent.
    return data[:HIDDEN], scale[:, :HIDDEN].contiguous(), layout, block


def baseline(activation, grad):
    activation_col = scaling_aware_fp8_transpose(activation, pack=False)
    grad_col = scaling_aware_fp8_transpose(grad, pack=False)
    result = fp8_gemm(activation_col, grad_col, out_dtype=torch.bfloat16)
    return unpad_tensor_if_needed(result, (DIM, HIDDEN))


def candidate(activation, grad):
    activation_col = scaling_aware_fp8_transpose(activation, pack=False)
    grad_col = scaling_aware_fp8_transpose(grad, pack=False)
    return fp8_gemm(logical_columns(activation_col), grad_col, out_dtype=torch.bfloat16)


def equal_bits(reference, actual, bit_dtype):
    return (
        reference.shape == actual.shape
        and reference.dtype == actual.dtype
        and torch.equal(reference.view(bit_dtype), actual.view(bit_dtype))
    )


def operand_parity(activation):
    data, scale, layout, block = scaling_aware_fp8_transpose(activation, pack=False)
    compact_data, compact_scale, _, _ = logical_columns((data, scale, layout, block))
    if data.shape != (PADDED_HIDDEN, *SHAPE[:-1]) or scale.shape != (64, PADDED_HIDDEN):
        raise ValueError(f"unexpected columnwise buffers {data.shape}, {scale.shape}")
    return {
        "data_is_contiguous_prefix": (
            compact_data.is_contiguous() and compact_data.data_ptr() == data.data_ptr()
        ),
        "fp8_data_bits": equal_bits(data[:HIDDEN], compact_data, torch.uint8),
        "fp32_scale_bits": equal_bits(scale[:, :HIDDEN], compact_scale, torch.int32),
        "original_scale_stride": list(scale.stride()),
        "compact_scale_stride": list(compact_scale.stride()),
    }


def measure(fn, operands):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = perf_counter_ns()
    result = fn(*operands)
    submitted = perf_counter_ns()
    end.record()
    end.synchronize()
    drained = perf_counter_ns()
    del result
    return {
        "event_ms": start.elapsed_time(end),
        "submission_ms": (submitted - wall_start) / 1e6,
        "wall_ms": (drained - wall_start) / 1e6,
    }


def profile(fn, operands):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as profiler:
        result = fn(*operands)
        torch.cuda.synchronize()
    del result
    return [
        {"name": event.name, "duration_us": event.time_range.elapsed_us()}
        for event in profiler.events()
        if event.device_type.name == "CUDA"
    ]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    if args.samples < 2 or args.output.exists():
        parser.error("provide a new output path and at least two samples")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("expose exactly one Hopper GPU")

    operands = load_inputs(args.fixture)
    report = {
        "fixture": str(args.fixture),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "transposed_operand": operand_parity(operands[0]),
    }
    compiled = {
        name: torch.compile(fn, fullgraph=True, options={"triton.cudagraphs": False})
        for name, fn in (("baseline", baseline), ("candidate", candidate))
    }
    reference = compiled["baseline"](*operands)
    try:
        actual = compiled["candidate"](*operands)
        torch.cuda.synchronize()
    except (RuntimeError, torch._dynamo.exc.Unsupported) as exc:
        report["candidate_failure"] = str(exc)
    else:
        report["wgrad_bf16_bits"] = equal_bits(reference, actual, torch.int16)
        report["wgrad_contiguous"] = actual.is_contiguous()
        if not report["wgrad_bf16_bits"]:
            report["wgrad_mismatched_values"] = (
                reference.view(torch.int16) != actual.view(torch.int16)
            ).count_nonzero().item()
        del actual
    del reference

    if report.get("wgrad_bf16_bits") and report["wgrad_contiguous"]:
        for fn in compiled.values():
            for _ in range(4):
                fn(*operands)
        torch.cuda.synchronize()
        samples = {"baseline": [], "candidate": []}
        for i in range(args.samples):
            order = ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
            for name in order:
                samples[name].append(measure(compiled[name], operands))
        report["compiled_call_samples"] = samples
        report["event_median_ms"] = {
            name: statistics.median(row["event_ms"] for row in rows)
            for name, rows in samples.items()
        }
        traces = {name: profile(fn, operands) for name, fn in compiled.items()}
        report["profiled_gpu_work"] = traces
        gemms = {
            name: [row["name"] for row in rows if "nvjet_sm90" in row["name"]]
            for name, rows in traces.items()
        }
        report["same_gemm_kernel"] = (
            len(gemms["baseline"]) == 1 and gemms["baseline"] == gemms["candidate"]
        )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
