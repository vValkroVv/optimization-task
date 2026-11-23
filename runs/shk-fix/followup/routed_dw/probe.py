"""Compare pinned Hopper routed BF16 dW accumulation and exact overwrite."""

import argparse
from hashlib import file_digest
import importlib
import json
import os
from pathlib import Path
import shutil
from statistics import median
import sys
from time import perf_counter_ns


SOURCE = Path(__file__).resolve().parent
MODULE_NAME = "shk_routed_dw_bf16"
BASELINE_SHA256 = "99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26"
FIXTURE_SHA256 = "15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727"
NUM_SMS = 116
BASELINE_KERNEL = "sm90_fp8_gemm_1d1d_impl"
OVERWRITE_KERNEL = "sm90_k_grouped_fp8_gemm_bf16_overwrite_impl"
# The existing build driver names its staged inputs after the FP32 experiment.
BUILD_INPUTS = {
    "deepgemm-bf16-overwrite.patch": "deepgemm-rounded-f32.patch",
    "bf16_overwrite.cpp": "direct_dw.cpp",
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def reference_module(directory, name):
    sys.path.insert(0, str(directory))
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != directory / f"{name}.py":
        raise RuntimeError(f"Imported {name} from an unexpected location")
    return module


def reference_builder(reference_root, build_root):
    builder = reference_module(
        reference_root / "runs/efsdp-optimizations/direct-dw", "direct_dw"
    )
    builder.SOURCE = build_root / "builder-inputs"
    builder.MODULE_NAME = MODULE_NAME
    builder.BASELINE_HEADER_DIGESTS["impls/smxx_layout.cuh"] = (
        "8ded250c492c8cc2968a0bdfe68931eb5671571373e9788d9254b9d8e9f14be2"
    )
    return builder


def check_installed_binary(builder):
    baseline = builder.verify_baseline()
    if sha256(baseline._C.__file__) != BASELINE_SHA256:
        raise RuntimeError(
            "Installed DeepGEMM extension differs from the pinned binary"
        )
    return baseline


def build(args):
    builder = reference_builder(args.reference_root, args.build_root)
    if not args.stage_only:
        check_installed_binary(builder)
    builder.SOURCE.mkdir(parents=True)
    for current, staged in BUILD_INPUTS.items():
        shutil.copyfile(SOURCE / current, builder.SOURCE / staged)
    source = args.build_root / "source"
    builder.stage(args.donor, source)
    if not args.stage_only:
        builder.build(source, args.build_root)


def prepare_native_operands(fixture, k_sizes, indices, device, converter, torch):
    """Run the installed converter once and trim exactly the routed K prefix."""
    rows = sum(k_sizes)
    sizes = torch.tensor(k_sizes, dtype=torch.int32, device=device)
    device_indices = indices.to(device)
    operands = {}
    for role, (data, scale) in fixture.roles.items():
        packed, scales = converter(data, device_indices, sizes, scale)
        width = data.shape[1]
        packed = packed.view(torch.float8_e4m3fn)[: rows * width]
        scales = scales.mT[:, : rows // 128]
        if packed.shape != (rows * width,) or scales.shape != (width, rows // 128):
            raise RuntimeError(f"Invalid converted operand for {role}")
        operands[role] = (packed, scales)
    return tuple(
        (operands[f"{name}_dy"], operands[f"{name}_x"]) for name in ("w2", "w13")
    )


def derived_counts(k_sizes):
    # Split each captured physical segment into two 128-aligned experts while
    # preserving the operand rows and aggregate K.
    counts = []
    for size in k_sizes:
        first = size // 256 * 128
        counts.extend((first, size - first))
    return counts


def shapes(experts):
    return ((experts, 4096, 1536), (experts, 3072, 4096))


def probe_special_values(candidate, harness, device, torch):
    bits = harness.SPECIAL_FP32_BITS
    ordered = bits + bits[1:] + bits[:1]
    signed = [value if value < 2**31 else value - 2**32 for value in ordered]
    inputs = torch.tensor(signed, device=device, dtype=torch.int32).view(torch.float32)
    reference = torch.zeros(inputs.shape, device=device, dtype=torch.bfloat16)
    actual = torch.empty_like(reference)
    candidate.probe_special_values(inputs, reference, actual)
    if not torch.equal(reference.view(torch.int16), actual.view(torch.int16)):
        raise AssertionError("BF16 TMA overwrite differs on special-value inputs")
    return {
        "inputs": [f"{value:08x}" for value in ordered],
        "output_bf16": [
            f"{value & 0xFFFF:04x}" for value in actual.view(torch.int16).tolist()
        ],
    }


def launch_geometry(path, marker):
    events = json.loads(path.read_text())["traceEvents"]
    return [
        {key: event["args"][key] for key in ("grid", "block")}
        for event in events
        if event.get("cat") == "kernel" and marker in event["name"]
    ]


def eager_timing(legacy, overwrite, samples, torch):
    operations = (
        ("legacy", legacy),
        ("overwrite", overwrite),
        ("legacy_control", legacy),
    )
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    observations = {name: [] for name, _ in operations}
    for trial in range(-4, samples):
        order = operations if trial % 2 == 0 else reversed(operations)
        for name, operation in order:
            start.record()
            before = perf_counter_ns()
            operation()
            stop.record()
            stop.synchronize()
            after = perf_counter_ns()
            if trial >= 0:
                observations[name].append(
                    {
                        "cuda_ms": start.elapsed_time(stop),
                        "host_to_completion_ms": (after - before) / 1e6,
                    }
                )
    return {
        "samples": observations,
        "medians_ms": {
            name: {key: median(row[key] for row in rows) for key in rows[0]}
            for name, rows in observations.items()
        },
    }


def check_case(
    name,
    operands,
    k_sizes,
    baseline,
    candidate,
    harness,
    device,
    output,
    samples,
    torch,
):
    weight_shapes = shapes(len(k_sizes))
    k_tensor = torch.tensor(k_sizes, dtype=torch.int32, device=device)
    has_tokens = any(k_sizes)

    def legacy():
        outputs = []
        for (a, b), shape in zip(operands, weight_shapes, strict=True):
            grad = torch.zeros(shape, device=device, dtype=torch.bfloat16)
            if has_tokens:
                baseline.k_grouped_fp8_gemm_nt_contiguous(
                    a, b, grad, k_sizes, k_tensor, grad
                )
            outputs.append(grad)
        return outputs

    def overwrite():
        outputs = []
        for (a, b), shape in zip(operands, weight_shapes, strict=True):
            grad = torch.empty(shape, device=device, dtype=torch.bfloat16)
            candidate.k_grouped_fp8_gemm_nt_contiguous_bf16_overwrite(
                a, b, grad, k_sizes, k_tensor
            )
            outputs.append(grad)
        return outputs

    os.environ["DG_CACHE_FORCE_SKIP"] = "1"
    expected = legacy()
    for (a, b), shape, ref in zip(operands, weight_shapes, expected, strict=True):
        guard = harness.GUARD_SIZE
        storage = torch.full(
            (ref.numel() + 2 * guard,),
            harness.GUARD_VALUE,
            dtype=torch.bfloat16,
            device=device,
        )
        actual = storage.narrow(0, guard, ref.numel()).view(shape)
        candidate.k_grouped_fp8_gemm_nt_contiguous_bf16_overwrite(
            a, b, actual, k_sizes, k_tensor
        )
        actual_bits = actual.view(torch.int16)
        if not torch.equal(actual_bits, ref.view(torch.int16)):
            raise AssertionError(f"{name} differs bitwise on {shape}")
        for expert, size in enumerate(k_sizes):
            if size == 0 and torch.count_nonzero(actual_bits[expert]).item():
                raise AssertionError(
                    f"{name} expert {expert} is not BF16 positive zero"
                )
        if not (
            torch.all(storage[:guard] == harness.GUARD_VALUE).item()
            and torch.all(storage[-guard:] == harness.GUARD_VALUE).item()
        ):
            raise AssertionError(f"{name} overwrote a destination canary")
    del expected, ref, storage, actual, actual_bits
    overwrite()
    torch.cuda.synchronize(device)

    # Every shape has already executed both entry points. Additional JIT misses
    # during memory, trace or timing checks invalidate the run.
    os.environ["DG_CACHE_FORCE_SKIP"] = "0"
    traces = {}
    launches = {}
    memory = {}
    expected_launches = len(weight_shapes) if has_tokens else 0
    for arm, operation, marker in (
        ("legacy", legacy, BASELINE_KERNEL),
        ("overwrite", overwrite, OVERWRITE_KERNEL),
    ):
        torch.cuda.empty_cache()
        memory[arm] = harness.peak_allocation(operation, device)
        trace_path = output / f"{name}-{arm}.json"
        traces[arm] = harness.trace_operation(operation, device, trace_path)
        launches[arm] = launch_geometry(trace_path, marker)
        if len(launches[arm]) != expected_launches:
            raise AssertionError(
                f"Unexpected {arm} grouped GEMM launch count for {name}"
            )
    if launches["legacy"] != launches["overwrite"]:
        raise AssertionError(f"Grouped GEMM launch geometry changed for {name}")
    return {
        "k_sizes": k_sizes,
        "shapes": weight_shapes,
        "bitwise_bf16_and_canaries": True,
        "full_output_bytes_per_pair": 2 * sum(m * n * e for e, m, n in weight_shapes),
        "output_zero_bytes": {
            "legacy": 2 * sum(m * n * e for e, m, n in weight_shapes),
            "overwrite": 2
            * sum(size == 0 for size in k_sizes)
            * sum(m * n for _, m, n in weight_shapes),
        },
        "memory": memory,
        "traces": traces,
        "gemm_launch_geometry": launches["legacy"],
        "eager": eager_timing(legacy, overwrite, samples, torch) if samples else None,
        "cuda_graph_replay": harness.graph_timing(legacy, overwrite, device, samples)
        if samples
        else None,
    }


def verify(args):
    os.environ.pop("DG_JIT_SHARED_CACHE_DIR", None)
    os.environ["DG_JIT_CACHE_DIR"] = str(args.evidence_dir.parent / "legacy-jit")
    import torch

    if sha256(args.fixture) != FIXTURE_SHA256:
        raise RuntimeError("Expected the preserved native EP8 rank-0 fixture")
    builder = reference_builder(args.reference_root, args.build_root)
    for current, staged in BUILD_INPUTS.items():
        if sha256(SOURCE / current) != sha256(builder.SOURCE / staged):
            raise RuntimeError(f"Build input changed since staging: {current}")
    harness = reference_module(
        args.reference_root / "runs/efsdp-optimizations/direct-dw", "verify_direct_dw"
    )
    replay = reference_module(
        args.reference_root / "runs/backward-components", "row2col_replay"
    )
    from torchtitan.ops.float8._cudacherry import requantize_rowwise_to_columnwise

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (9, 0) or "H100" not in properties.name:
        raise RuntimeError(f"Expected H100, found {properties}")
    if properties.multi_processor_count < NUM_SMS:
        raise RuntimeError(f"Expected at least {NUM_SMS} SMs, found {properties}")
    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    ):
        raise RuntimeError(
            "Deterministic empty-buffer filling changes the overwrite measurement"
        )

    args.evidence_dir.mkdir(parents=True)
    os.environ["DG_CACHE_FORCE_SKIP"] = "1"
    baseline = check_installed_binary(builder)
    baseline.set_num_sms(NUM_SMS)
    candidate = builder.load_direct_dw(args.build_root, num_sms=NUM_SMS)
    result = {
        "device": properties.name,
        "sm_budget": NUM_SMS,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "installed_deepgemm_extension_sha256": BASELINE_SHA256,
        "candidate": json.loads((args.build_root / "manifest.json").read_text()),
        "fixture": {"path": str(args.fixture), "sha256": FIXTURE_SHA256},
        "scope": "Paired BF16 zero/empty allocation and native Hopper dW GEMM, including scale alignment and any empty-slice writes",
        "samples": args.samples,
        "special_values": probe_special_values(candidate, harness, device, torch),
        "cases": {},
    }

    def run_case(name, operands, counts, samples):
        result["cases"][name] = check_case(
            name,
            operands,
            counts,
            baseline,
            candidate,
            harness,
            device,
            args.evidence_dir,
            samples,
            torch,
        )
        (args.evidence_dir / "component.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(
            f"{name}: BF16 bits, positive zeros, canaries and activation passed",
            flush=True,
        )

    fixture = replay.load_fixture(args.fixture, device)
    counts = fixture.padded_sizes.tolist()
    result["fixture"]["metadata"] = fixture.metadata
    native_operands = prepare_native_operands(
        fixture,
        counts,
        fixture.m_indices,
        device,
        requantize_rowwise_to_columnwise,
        torch,
    )
    run_case("native-ep8-captured-rank0", native_operands, counts, args.samples)
    del native_operands

    counts = derived_counts(counts)
    indices = replay.expected_indices(counts, fixture.m_indices.numel())
    os.environ["DG_CACHE_FORCE_SKIP"] = "1"
    derived_operands = prepare_native_operands(
        fixture, counts, indices, device, requantize_rowwise_to_columnwise, torch
    )
    run_case(
        "ep4-shaped-derived-by-splitting-captured-ep8-groups",
        derived_operands,
        counts,
        args.samples,
    )
    del fixture, derived_operands

    for experts in (16, 32):
        for pattern, counts in (
            ("mixed", ([0, 128, 0, 0, 256, 512, 128, 0] * (experts // 8))),
            ("all-empty", [0] * experts),
        ):
            generator = torch.Generator(device=device).manual_seed(1944)
            operands = tuple(
                (
                    harness.make_operand(counts, m, device, generator),
                    harness.make_operand(counts, n, device, generator),
                )
                for _, m, n in shapes(experts)
            )
            run_case(
                f"synthetic-{experts}-experts-{pattern}", operands, counts, args.samples
            )


def absolute_path(value):
    return Path(value).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder, verifier = commands.add_parser("build"), commands.add_parser("verify")
    for command in (builder, verifier):
        command.add_argument("--reference-root", type=absolute_path, required=True)
        command.add_argument("--build-root", type=absolute_path, required=True)
    builder.add_argument("--donor", type=absolute_path, required=True)
    builder.add_argument("--stage-only", action="store_true")
    verifier.add_argument("--fixture", type=absolute_path, required=True)
    verifier.add_argument("--evidence-dir", type=absolute_path, required=True)
    verifier.add_argument("--device", type=int, default=0)
    verifier.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    if getattr(args, "samples", 0) < 0:
        parser.error("--samples must be nonnegative")
    if args.command == "build":
        build(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
