#!/usr/bin/env python3
"""Replay compiled routed wgrad with CudaCherry's original or strided scale loads."""

import argparse
import json
import os
from pathlib import Path
from statistics import median

from routed_dw import probe


CHECKOUT = Path(__file__).resolve().parents[3]
SOURCES = {
    "quantization/row2col_dq.py": "3839e99e3f3de2af14fbc0562d5d155f4d93fcc3b116a2a6077c488ef9cf72ae",
    "csrc/quantization/row2col_dq/row2col_kernel.cu": (
        "07be6d51740d03d6bdb8750e924576dff7e47742c2f0c7d5d362ffb07b707194"
    ),
    "csrc/quantization/row2col_dq/bindings.cpp": (
        "f1bf13ad271f0239526b09949ea555ccbebd6a0b362ac987c829ed9844ade816"
    ),
}
ACTIVATION = "requantize_grouped_kernel"
ROLE_PAIRS = {"w2": ("w2_dy", "w2_x"), "w13": ("w13_dy", "w13_x")}


def compile_paths(weight_shape, target, native, row2col, wgrad, torch):
    def strided(data, indices, counts, scale):
        rows, width = data.shape
        packed = torch.empty(
            rows * width, device=data.device, dtype=torch.float8_e4m3fn
        )
        packed_scale = torch.empty(
            (rows // 128, width), device=data.device, dtype=torch.float32
        )
        torch.ops.cudacherry.row2col_dq(
            native.view_as_uint8(data).contiguous(),
            scale,
            counts,
            indices.contiguous().to(torch.int32),
            packed,
            packed_scale,
        )
        return packed, packed_scale

    def build(converter):
        def convert(inputs):
            operands, indices, counts = inputs
            data, scale = operands[target]
            return converter(data, indices, counts, scale)

        def chain(inputs):
            operands, indices, counts, counts_cpu = inputs
            result = []
            for index, (data, scale) in enumerate(operands):
                fn = converter if index == target else row2col
                packed, packed_scale = fn(data, indices, counts, scale)
                result.append((packed.view(torch.float8_e4m3fn), packed_scale.mT))
            return wgrad(*result, weight_shape, counts.device, None, counts_cpu, counts)

        return {
            scope: torch.compile(
                fn,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            for scope, fn in (("converter", convert), ("through_wgrad", chain))
        }

    return {arm: build(fn) for arm, fn in (("A", row2col), ("B", strided))}


def verify_outputs(paths, convert_inputs, chain_inputs, rows, width, torch):
    def completed(arm, scope, inputs):
        result = paths[arm][scope](inputs)
        torch.cuda.synchronize()
        return result

    a, b = (completed(arm, "converter", convert_inputs) for arm in ("A", "B"))
    parity = {
        "fp8_bytes": torch.equal(
            a[0].view(torch.uint8)[: rows * width],
            b[0].view(torch.uint8)[: rows * width],
        ),
        "fp32_scale_bits": torch.equal(
            a[1][: rows // 128].view(torch.int32), b[1][: rows // 128].view(torch.int32)
        ),
        "layout": [(t.shape, t.stride(), t.dtype) for t in a]
        == [(t.shape, t.stride(), t.dtype) for t in b],
    }
    del a, b
    a, b = (completed(arm, "through_wgrad", chain_inputs) for arm in ("A", "B"))
    parity["bf16_wgrad_bits"] = a.dtype == b.dtype == torch.bfloat16 and torch.equal(
        a.view(torch.int16), b.view(torch.int16)
    )
    if not all(parity.values()):
        raise AssertionError(
            f"Row2col comparison changed consumed values or layout: {parity}"
        )
    return parity


def measure_scope(paths, inputs, scope, args, name, device, measure, harness, torch):
    for arm in ("A", "B"):
        for _ in range(args.warmup):
            paths[arm][scope](inputs)
        torch.cuda.synchronize(device)
    blocks = []
    for repeat in range(args.repeats):
        order = ("A", "B") if repeat % 2 == 0 else ("B", "A")
        pair = {arm: measure(paths[arm][scope], inputs, args.samples) for arm in order}
        timings_a, timings_b = (pair[arm]["median"] for arm in ("A", "B"))
        blocks.append(
            {
                "order": order,
                "arms": pair,
                "candidate_minus_control_ms": {
                    key: timings_b[key] - value for key, value in timings_a.items()
                },
            }
        )

    profiles = {}
    for arm in ("A", "B"):
        path = args.output / f"{name}-{scope}-{arm}.trace.json"
        trace = harness.trace_operation(lambda: paths[arm][scope](inputs), device, path)
        kernels = trace["kernels"]
        converter_count = sum(ACTIVATION in event["name"] for event in kernels)
        wgrad_count = sum(probe.BASELINE_KERNEL in event["name"] for event in kernels)
        if scope == "converter":
            # Only scale materialization and the converter are in this graph.
            expected = 2 if arm == "A" else 1
            if (
                converter_count != 1
                or len(kernels) != expected
                or trace["device_memsets"]
            ):
                raise AssertionError(
                    f"Expected {expected} device kernels for {arm}: {trace}"
                )
        elif (converter_count, wgrad_count) != (2, 1):
            raise AssertionError(f"Unexpected converter/wgrad activation: {trace}")
        profiles[arm] = {"trace": str(path), **trace}
    return {
        "blocks": blocks,
        "median_candidate_minus_control_ms": {
            key: median(row["candidate_minus_control_ms"][key] for row in blocks)
            for key in blocks[0]["candidate_minus_control_ms"]
        },
        "profiles": profiles,
    }


def run_projection(name, projection, fixture, indices, counts, runtime, args):
    torch, native, row2col, wgrad, device, measure, harness = runtime
    target = 0 if projection == "w2" else 1
    operands = tuple(fixture.roles[role] for role in ROLE_PAIRS[projection])
    data, scale = operands[target]
    rows, width = data.shape
    if indices.device != device or any(
        tensor.device != device for operands_pair in operands for tensor in operands_pair
    ):
        raise ValueError(f"Row2col operands and indices must all be on {device}")
    if (
        width != 4096
        or not data.is_contiguous()
        or not scale.is_contiguous()
        or scale.shape != (rows, 32)
        or indices.shape != (rows,)
        or indices.dtype != torch.int32
        or not indices.is_contiguous()
    ):
        raise ValueError(
            "Expected native row-major FP8 input and FP32 scales for the 4096-wide site"
        )
    experts = len(counts)
    weight_shape = torch.Size(probe.shapes(experts)[target])
    counts_cpu = torch.tensor(counts, dtype=torch.int32, pin_memory=True)
    counts_gpu = counts_cpu.to(device, non_blocking=True)
    convert_inputs = operands, indices, counts_gpu
    chain_inputs = *convert_inputs, counts_cpu
    paths = compile_paths(weight_shape, target, native, row2col, wgrad, torch)
    os.environ["DG_CACHE_FORCE_SKIP"] = "1"
    parity = verify_outputs(
        paths, convert_inputs, chain_inputs, sum(counts), width, torch
    )
    os.environ["DG_CACHE_FORCE_SKIP"] = "0"
    with torch._dynamo.config.patch(error_on_recompile=True):
        results = {
            scope: measure_scope(
                paths, inputs, scope, args, name, device, measure, harness, torch
            )
            for scope, inputs in (
                ("converter", convert_inputs),
                ("through_wgrad", chain_inputs),
            )
        }
    return {
        "target_role": ROLE_PAIRS[projection][target],
        "weight_shape": list(weight_shape),
        "input_shape": list(data.shape),
        "input_scale_stride": list(scale.stride()),
        "consumed_rows": sum(counts),
        "k_sizes": counts,
        "scale_copy_logical_bytes": 2 * scale.numel() * scale.element_size(),
        "parity": parity,
        **results,
    }


def run(args):
    os.environ["CUDACHERRY_DISABLE_JIT"] = "1"
    os.environ.pop("DG_JIT_SHARED_CACHE_DIR", None)
    import torch

    from dense_weight_scale import measure
    from torchtitan.ops.float8 import _cudacherry, deep_gemm_handle

    import cudacherry.quantization.row2col_dq as native

    if os.environ.get("GIFT_ROUTED_DW_BF16_OVERWRITE", "0") not in ("", "0"):
        raise RuntimeError(
            "Run with the installed legacy routed BF16 wgrad in both arms"
        )
    if not Path(deep_gemm_handle.__file__).resolve().is_relative_to(CHECKOUT / "gift"):
        raise RuntimeError(f"Unexpected GIFT import: {deep_gemm_handle.__file__}")
    package = Path(native.__file__).resolve().parents[1]
    source_hashes = {name: probe.sha256(package / name) for name in SOURCES}
    if source_hashes != SOURCES:
        raise RuntimeError(
            f"Unverified installed CudaCherry row2col source: {source_hashes}"
        )
    if probe.sha256(args.fixture) != probe.FIXTURE_SHA256:
        raise RuntimeError("Expected the preserved native EP8 rank-zero fixture")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if (
        (properties.major, properties.minor) != (9, 0)
        or "H100" not in properties.name
        or properties.multi_processor_count < probe.NUM_SMS
    ):
        raise RuntimeError(
            f"Expected full H100 with {probe.NUM_SMS} usable SMs, found {properties}"
        )
    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    ):
        raise RuntimeError(
            "Deterministic empty-buffer fills would change the launch census"
        )

    legacy = deep_gemm_handle._load_deep_gemm()
    legacy_hash = probe.sha256(legacy._C.__file__)
    if legacy_hash != probe.BASELINE_SHA256:
        raise RuntimeError(f"Unexpected installed DeepGEMM binary {legacy_hash}")
    legacy.set_num_sms(probe.NUM_SMS)
    extension = Path(
        native._ext().__file__
    ).resolve()  # Register the existing op before Dynamo.
    extension_hash = probe.sha256(extension)
    harness = probe.reference_module(
        args.reference_root / "runs/efsdp-optimizations/direct-dw", "verify_direct_dw"
    )
    replay = probe.reference_module(
        args.reference_root / "runs/backward-components", "row2col_replay"
    )
    args.output.mkdir(parents=True)
    fixture = replay.load_fixture(args.fixture, device)
    runtime = (
        torch,
        native,
        _cudacherry.requantize_rowwise_to_columnwise,
        deep_gemm_handle.routed_bf16_wgrad,
        device,
        measure,
        harness,
    )
    receipt = {
        "torch": torch.__version__,
        "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda,
        "gpu": properties.name,
        "fixture_sha256": probe.FIXTURE_SHA256,
        "fixture_metadata": fixture.metadata,
        "installed_cudacherry_sources_sha256": source_hashes,
        "installed_row2col_extension": str(extension),
        "installed_row2col_extension_sha256": extension_hash,
        "installed_deepgemm_sha256": legacy_hash,
        "deepgemm_sm_budget": probe.NUM_SMS,
        "script_sha256": probe.sha256(__file__),
        "warmup_per_arm_scope": args.warmup,
        "samples_per_arm_block": args.samples,
        "ab_ba_paired_blocks": args.repeats,
        "scope": (
            "Resident captured inputs, one synchronized fullgraph Inductor call per sample. "
            "CUDA event spans can include launch gaps. Profiles are separate. The chain "
            "converts both operands and calls the legacy mutating BF16 wgrad, including "
            "its output allocation. Only the HybridEP operand omits the scale copy."
        ),
        "cases": {},
    }
    counts = fixture.padded_sizes.tolist()
    derived = probe.derived_counts(counts)
    with torch.no_grad():
        for label, indices, layout in (
            ("native-ep8-rank0", fixture.m_indices.to(device), counts),
            (
                "ep4-shaped-derived-from-ep8",
                replay.expected_indices(derived, fixture.m_indices.numel()).to(device),
                derived,
            ),
        ):
            for projection in ROLE_PAIRS:
                name = f"{label}-{projection}"
                torch._dynamo.reset()
                result = run_projection(
                    name, projection, fixture, indices, layout, runtime, args
                )
                receipt["cases"][name] = result
                print(
                    name,
                    result["through_wgrad"]["median_candidate_minus_control_ms"],
                    flush=True,
                )
                (args.output / "component.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("reference-root", "fixture", "output"):
        parser.add_argument(f"--{option}", type=probe.absolute_path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if min(args.warmup, args.samples) < 1 or args.repeats < 2:
        parser.error("Require positive warmup and samples and at least two repeats")
    run(args)


if __name__ == "__main__":
    main()
