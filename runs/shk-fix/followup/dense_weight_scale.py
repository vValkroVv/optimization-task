#!/usr/bin/env python3
"""Measure compiled CudaCherry weight quantization with reused block scales."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns

import torch


CHECKOUT = Path(__file__).resolve().parents[3]
CUDA_CHERRY_PIN = "585bd50a641857ccae8b5ec691bb15791c9529e7"
VENDOR_SOURCES = {
    "quantization/blockwise.py": (
        "7fbe0417ed3a775be0e11171e09de4b733a0b0f5acffd5b42fcc7617f4247094"
    ),
    "csrc/quantization/blockwise/block_quantization_kernel.cu": (
        "b596cbb44f50c92ea1cfd0ef01ad221fdd367ec53496ae4080ea22d95c0b8c8f"
    ),
}
WEIGHT_SHAPES = {"w1.weight": (10944, 4096), "w2.weight": (4096, 10944)}
KERNEL_PREFIX = "quantize_blk128x128_kernel_"
TRANSPOSE_KERNEL = {"A": "t_computed_scale", "B": "t_given_scale"}


def sha256(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def load_cases(fixture_path, include_aligned):
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    if fixture["schema_version"] != 1 or fixture["case"] != "dense-fp8":
        raise ValueError("Expected the captured dense-fp8 fixture")
    source = fixture["input"]
    cases = {name: fixture["weights"][name] for name in WEIGHT_SHAPES}
    for tensor, shape in [
        (source, (2, 4096, 4096)),
        *((cases[name], shape) for name, shape in WEIGHT_SHAPES.items()),
    ]:
        if (
            tensor.shape != shape
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"Expected contiguous BF16 fixture tensor of shape {shape}"
            )
    if include_aligned:
        for width in (1536, 4096):
            cases[f"w1.weight-derived-{width}"] = cases["w1.weight"][:width]
    return cases, source[:1, :128].contiguous()


def compile_chains(quantize):
    def computed(weight):
        normal = quantize(weight, pack=False)
        transposed = quantize(weight, transpose_output=True, pack=False)
        return normal, transposed

    def given(weight):
        normal = quantize(weight, pack=False)
        transposed = quantize(
            weight,
            transpose_output=True,
            precomputed_scale=normal[1],
            use_precomputed_scale=True,
            pack=False,
        )
        return normal, transposed

    return {
        arm: torch.compile(
            fn,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        for arm, fn in (("A", computed), ("B", given))
    }


def bits_equal(a, b, bit_dtype):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.stride() == b.stride()
        and torch.equal(a.contiguous().view(bit_dtype), b.contiguous().view(bit_dtype))
    )


def verify_outputs(
    chains, weight, original_weight, rows, fp8_gemm, quantize_vec128, pad
):
    control = chains["A"](weight)
    candidate = chains["B"](weight)
    parity = {}
    for orientation, a, b, width in zip(
        ("normal", "transposed"), control, candidate, reversed(weight.shape)
    ):
        data, scale, layout, block = a
        if (
            data.dtype != torch.float8_e4m3fn
            or scale.dtype != torch.float32
            or (orientation == "normal" and not scale.is_contiguous())
        ):
            raise RuntimeError(f"Unexpected {orientation} quantizer output format")
        # Repeat captured rows only to supply the desired K dimension to the
        # parity GEMM. These consumers are outside the timing scope.
        repeats = (width + rows.shape[-1] - 1) // rows.shape[-1]
        activation = rows.repeat(1, 1, repeats)[..., :width].contiguous()
        operand = quantize_vec128(pad(activation), pack=False)
        out_a = fp8_gemm(
            operand, a, out_dtype=torch.bfloat16, use_split_accumulator=True
        )
        out_b = fp8_gemm(
            operand, b, out_dtype=torch.bfloat16, use_split_accumulator=True
        )
        parity[orientation] = {
            "fp8_data_bits": bits_equal(data, b[0], torch.uint8),
            "fp32_scale_bits": bits_equal(scale, b[1], torch.int32),
            "layout_and_block_match": (layout, block) == b[2:],
            "bf16_gemm_bits": (
                out_a.dtype == torch.bfloat16 and bits_equal(out_a, out_b, torch.int16)
            ),
            "data_shape": list(data.shape),
            "data_stride": list(data.stride()),
            "scale_shape": list(scale.shape),
            "scale_stride": list(scale.stride()),
        }
    unchanged = bits_equal(weight.cpu(), original_weight, torch.int16)
    if not unchanged or any(
        not result[key]
        for result in parity.values()
        for key in (
            "fp8_data_bits",
            "fp32_scale_bits",
            "layout_and_block_match",
            "bf16_gemm_bits",
        )
    ):
        raise RuntimeError(
            f"Bitwise mismatch for {tuple(weight.shape)}: {parity}, input={unchanged}"
        )
    return {"bf16_weight_unchanged": unchanged, **parity}


def measure(chain, weight, samples):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_allocated = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()
    retries_before = torch.cuda.memory_stats()["num_alloc_retries"]
    times = {
        "cuda_event_ms": [],
        "host_submission_ms": [],
        "host_drain_ms": [],
        "wall_ms": [],
    }
    for _ in range(samples):
        start.record()
        begin = perf_counter_ns()
        outputs = chain(weight)
        submitted = perf_counter_ns()
        end.record()
        end.synchronize()
        drained = perf_counter_ns()
        times["cuda_event_ms"].append(start.elapsed_time(end))
        times["host_submission_ms"].append((submitted - begin) / 1e6)
        times["host_drain_ms"].append((drained - submitted) / 1e6)
        times["wall_ms"].append((drained - begin) / 1e6)
        del outputs
    return {
        "samples": times,
        "median": {name: median(values) for name, values in times.items()},
        "memory_bytes": {
            "start_allocated": start_allocated,
            "start_reserved": start_reserved,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        },
        "allocator_retries": torch.cuda.memory_stats()["num_alloc_retries"]
        - retries_before,
    }


def profile(chain, weight, arm):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        outputs = chain(weight)
        torch.cuda.synchronize()
    del outputs
    device_events = [
        {"name": event.name, "duration_us": event.time_range.elapsed_us()}
        for event in sorted(profiler.events(), key=lambda event: event.time_range.start)
        if event.device_type.name == "CUDA"
    ]
    actual = [event for event in device_events if KERNEL_PREFIX in event["name"]]
    expected = ("nt_computed_scale", TRANSPOSE_KERNEL[arm])
    if len(actual) != len(expected) or not all(
        KERNEL_PREFIX + suffix in event["name"]
        for event, suffix in zip(actual, expected)
    ):
        raise RuntimeError(f"Unexpected arm {arm} quantizer kernels: {actual}")
    return {
        "device_events": device_events,
        "quantizer_kernel_sum_us": sum(event["duration_us"] for event in actual),
    }


def run_case(chains, weight, args):
    measurements = []
    paired_deltas = []
    with torch._dynamo.config.patch(error_on_recompile=True):
        for chain in chains.values():
            for _ in range(args.warmup):
                chain(weight)
            torch.cuda.synchronize()
        for repeat in range(args.repeats):
            order = ("A", "B") if repeat % 2 == 0 else ("B", "A")
            pair = {}
            for arm in order:
                result = measure(chains[arm], weight, args.samples)
                pair[arm] = result["median"]
                measurements.append({"arm": arm, "repeat": repeat + 1, **result})
            paired_deltas.append(
                {key: pair["B"][key] - pair["A"][key] for key in pair["A"]}
            )
        activation = {arm: profile(chain, weight, arm) for arm, chain in chains.items()}
    return {
        "measurements": measurements,
        "paired_candidate_minus_control_ms": paired_deltas,
        "median_paired_candidate_minus_control_ms": {
            key: median(delta[key] for delta in paired_deltas)
            for key in paired_deltas[0]
        },
        "profile": activation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-aligned", action="store_true")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 1 or args.repeats < 2:
        parser.error("Require positive warmup and samples and at least two repeats")
    if (
        torch.cuda.get_device_capability() != (9, 0)
        or "H100" not in torch.cuda.get_device_name()
    ):
        raise RuntimeError("This component benchmark requires an H100")

    from torchtitan.ops.float8 import _cudacherry
    from torchtitan.ops.float8.fp8_utils import pad_tensor_if_needed

    if not Path(_cudacherry.__file__).resolve().is_relative_to(CHECKOUT / "gift"):
        raise RuntimeError(f"Unexpected GIFT import: {_cudacherry.__file__}")
    vendor_path = Path(_cudacherry.cudacherry.__file__).resolve().parent
    vendor_hashes = {name: sha256(vendor_path / name) for name in VENDOR_SOURCES}
    if vendor_hashes != VENDOR_SOURCES:
        raise RuntimeError(
            f"Unexpected CudaCherry quantizer source hashes: {vendor_hashes}"
        )

    cases, rows = load_cases(args.fixture, args.include_aligned)
    results = {}
    with torch.no_grad():
        rows = rows.to("cuda")
        # Register the vendor operators outside Dynamo's full-graph capture.
        _cudacherry.quantize_blk128x128(rows[0, :, :128].contiguous(), pack=False)
        torch.cuda.synchronize()
        chains = compile_chains(_cudacherry.quantize_blk128x128)
        for name, original_weight in cases.items():
            weight = original_weight.to("cuda")
            parity = verify_outputs(
                chains,
                weight,
                original_weight,
                rows,
                _cudacherry.fp8_gemm,
                _cudacherry.quantize_vec128,
                pad_tensor_if_needed,
            )
            results[name] = {
                "shape": list(weight.shape),
                "input_kind": "captured"
                if name in WEIGHT_SHAPES
                else "derived prefix of w1.weight",
                "parity": parity,
                **run_case(chains, weight, args),
            }
            del weight
            print(
                name,
                results[name]["median_paired_candidate_minus_control_ms"],
                flush=True,
            )

    receipt = {
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda,
        "vendor_pin": CUDA_CHERRY_PIN,
        "vendor_package": str(vendor_path),
        "vendor_sources_sha256": vendor_hashes,
        "runner_sha256": sha256(Path(__file__)),
        "fixture": str(args.fixture),
        "fixture_sha256": sha256(args.fixture),
        "warmup_calls_per_arm_shape": args.warmup,
        "samples_per_arm_block": args.samples,
        "paired_blocks_per_shape": args.repeats,
        "scope": (
            "A single fullgraph Inductor call per sample returns both quantized operands. "
            "CUDA events include launch gaps. Wall includes submission and end-event drain. "
            "Allocator peaks are from warmed arms sharing one process pool. "
            "The component has no autograd saves or collectives."
        ),
        "cases": results,
    }
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
