#!/usr/bin/env python3
"""Compare the compiled GIFT routed BF16 weight-gradient allocation and op."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

import probe


CHECKOUT = Path(__file__).resolve().parents[4]
SWITCH = "GIFT_ROUTED_DW_BF16_OVERWRITE"
BUILD = "GIFT_ROUTED_DW_BF16_OVERWRITE_BUILD"
ARTIFACT_SHA256 = "dbd1fa5b36e5694c3ec7768910b5e1cedb3e63b3660744fb452347940ba0fe90"
OP_NAMES = {
    "A": "torchtitan::k_grouped_fp8_gemm_contiguous",
    "B": "torchtitan::k_grouped_fp8_gemm_bf16_overwrite",
}
KERNEL_NAMES = {"A": probe.BASELINE_KERNEL, "B": probe.OVERWRITE_KERNEL}


def over_capacity(operands, rows, torch):
    """Poison two extra K tiles beyond the groups' true packed prefix."""
    padded = []
    for weight_operands in operands:
        pair = []
        for data, scale in weight_operands:
            width, blocks = scale.shape
            if data.numel() != rows * width or blocks != rows // 128:
                raise ValueError("Unexpected packed operand")
            full_data = torch.full(
                ((rows + 256) * width,), 0x38, dtype=torch.uint8, device=data.device
            ).view(torch.float8_e4m3fn)
            full_scale = torch.full(
                (blocks + 2, width), 4.0, dtype=torch.float32, device=scale.device
            ).mT
            full_data[: data.numel()].copy_(data)
            full_scale[:, :blocks].copy_(scale)
            pair.append((full_data, full_scale))
        padded.append(tuple(pair))
    return tuple(padded)


def compile_pair(experts, routed_bf16_wgrad, torch):
    weight_shapes = tuple(torch.Size(shape) for shape in probe.shapes(experts))

    def run(operands, ks_cpu, ks_gpu):
        return tuple(
            routed_bf16_wgrad(a, b, shape, a[0].device, None, ks_cpu, ks_gpu)
            for (a, b), shape in zip(operands, weight_shapes)
        )

    return torch.compile(
        run,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False},
    )


def bind_counts(pair, operands, counts, device, torch):
    ks_cpu = torch.tensor(counts, dtype=torch.int32, pin_memory=True)
    ks_gpu = ks_cpu.to(device, non_blocking=True)
    return lambda: pair(operands, ks_cpu, ks_gpu)


def output_signatures(operation, counts, torch):
    outputs = operation()
    signatures = []
    for out, shape in zip(outputs, probe.shapes(len(counts)), strict=True):
        if (
            tuple(out.shape) != shape
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
        ):
            raise AssertionError("Compiled op returned an unexpected output")
        raw = out.detach().cpu().view(torch.int16)
        for expert, size in enumerate(counts):
            if size == 0 and torch.count_nonzero(raw[expert]).item():
                raise AssertionError(
                    f"Expert {expert} did not return BF16 positive zeros"
                )
        signatures.append(sha256(memoryview(raw.numpy())).hexdigest())
    return signatures


def measure(operation, device, samples, torch):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    initial_allocated = torch.cuda.memory_allocated(device)
    initial_reserved = torch.cuda.memory_reserved(device)
    retries_before = torch.cuda.memory_stats(device)["num_alloc_retries"]
    observations = []
    for _ in range(samples):
        start.record()
        begin = perf_counter_ns()
        outputs = operation()
        submitted = perf_counter_ns()
        end.record()
        end.synchronize()
        drained = perf_counter_ns()
        observations.append(
            {
                "cuda_event_ms": start.elapsed_time(end),
                "host_submission_ms": (submitted - begin) / 1e6,
                "host_drain_ms": (drained - submitted) / 1e6,
                "wall_ms": (drained - begin) / 1e6,
            }
        )
        del outputs
    return {
        "samples": observations,
        "medians_ms": {
            key: median(row[key] for row in observations) for key in observations[0]
        },
        "memory_bytes": {
            "initial_allocated": initial_allocated,
            "initial_reserved": initial_reserved,
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
        },
        "allocator_retries": torch.cuda.memory_stats(device)["num_alloc_retries"]
        - retries_before,
    }


def profile(operation, arm, nonempty, device, path, harness):
    trace = harness.trace_operation(operation, device, path)
    events = json.loads(path.read_text())["traceEvents"]
    ops = [event["name"] for event in events if event.get("name") in OP_NAMES.values()]
    launches = probe.launch_geometry(path, KERNEL_NAMES[arm])
    other_arm = "B" if arm == "A" else "A"
    if (
        ops != [OP_NAMES[arm]] * 2
        or len(launches) != (2 if nonempty else 0)
        or any(KERNEL_NAMES[other_arm] in item["name"] for item in trace["kernels"])
    ):
        raise AssertionError(f"Unexpected arm {arm} activation: {ops}, {launches}")
    return {"trace": str(path), "ops": ops, "launches": launches, **trace}


def check_case(name, operation, operands, counts, args, device, harness, torch, timed):
    os.environ["DG_CACHE_FORCE_SKIP"] = "1"
    signatures = output_signatures(operation, counts, torch)
    os.environ["DG_CACHE_FORCE_SKIP"] = "0"
    with torch._dynamo.config.patch(error_on_recompile=True):
        if timed:
            for _ in range(args.warmup):
                operation()
            measurement = measure(operation, device, args.samples, torch)
        else:
            measurement = None
        activation = profile(
            operation,
            args.worker_arm,
            any(counts),
            device,
            args.worker_dir / f"{name}.trace.json",
            harness,
        )
    return {
        "k_sizes": counts,
        "capacity_rows": operands[0][0][1].shape[1] * 128,
        "output_sha256_bf16": signatures,
        "measurement": measurement,
        "activation": activation,
    }


def worker(args):
    os.environ.pop("DG_JIT_SHARED_CACHE_DIR", None)
    import torch

    from torchtitan.ops.float8 import deep_gemm_handle
    from torchtitan.ops.float8._cudacherry import requantize_rowwise_to_columnwise

    if not Path(deep_gemm_handle.__file__).resolve().is_relative_to(CHECKOUT / "gift"):
        raise RuntimeError(f"Unexpected GIFT import: {deep_gemm_handle.__file__}")
    if os.environ[SWITCH] != str(int(args.worker_arm == "B")):
        raise RuntimeError("Overwrite switch does not match the worker arm")
    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    ):
        raise RuntimeError("Deterministic empty-buffer filling changes the measurement")
    harness = probe.reference_module(
        args.reference_root / "runs/efsdp-optimizations/direct-dw", "verify_direct_dw"
    )
    replay = probe.reference_module(
        args.reference_root / "runs/backward-components", "row2col_replay"
    )
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
    baseline = probe.check_installed_binary(
        probe.reference_builder(args.reference_root, args.build_root)
    )
    baseline.set_num_sms(probe.NUM_SMS)
    fixture = replay.load_fixture(args.fixture, device)
    cases = {}

    def run_case(name, pair, operands, counts, *, timed=False):
        operation = bind_counts(pair, operands, counts, device, torch)
        cases[name] = check_case(
            name, operation, operands, counts, args, device, harness, torch, timed
        )
        print(f"{args.worker_arm} {name} passed", flush=True)

    for derived in (False, True):
        counts = fixture.padded_sizes.tolist()
        if derived:
            counts = probe.derived_counts(counts)
            indices = replay.expected_indices(counts, fixture.m_indices.numel())
            name = "ep4-shaped-derived"
        else:
            indices = fixture.m_indices
            name = "native-ep8-rank0"
        operands = probe.prepare_native_operands(
            fixture, counts, indices, device, requantize_rowwise_to_columnwise, torch
        )
        pair = compile_pair(len(counts), deep_gemm_handle.routed_bf16_wgrad, torch)
        run_case(name, pair, operands, counts, timed=True)
        if not derived:
            padded = over_capacity(operands, sum(counts), torch)
            run_case("native-ep8-over-capacity", pair, padded, counts)
            if (
                cases["native-ep8-over-capacity"]["output_sha256_bf16"]
                != cases[name]["output_sha256_bf16"]
            ):
                raise AssertionError("Over-capacity tail changed the native output")
            del padded
        del operands
    del fixture

    mixed = [0, 128, 0, 0, 256, 512, 128, 0] * 2
    generator = torch.Generator(device=device).manual_seed(1944)
    operands = tuple(
        (
            harness.make_operand(mixed, m, device, generator),
            harness.make_operand(mixed, n, device, generator),
        )
        for _, m, n in probe.shapes(len(mixed))
    )
    operands = over_capacity(operands, sum(mixed), torch)
    pair = compile_pair(len(mixed), deep_gemm_handle.routed_bf16_wgrad, torch)
    run_case("mixed-with-capacity", pair, operands, mixed)
    # Changing only count values must reuse the mixed compiled graph.
    empty_counts = [0] * len(mixed)
    with torch._dynamo.config.patch(error_on_recompile=True):
        run_case("all-empty-with-capacity", pair, operands, empty_counts)
    empty = tuple(
        tuple((data[:0], scale[:, :0]) for data, scale in weight_operands)
        for weight_operands in operands
    )
    run_case("all-empty-zero-capacity", pair, empty, empty_counts)

    receipt = {
        "arm": args.worker_arm,
        "gpu": properties.name,
        "torch": torch.__version__,
        "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda,
        "deepgemm_sm_budget": probe.NUM_SMS,
        "installed_binary_sha256": probe.BASELINE_SHA256,
        "integration_source_sha256": probe.sha256(deep_gemm_handle.__file__),
        "cases": cases,
    }
    (args.worker_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def compare_pair(left, right):
    for key in (
        "gpu",
        "torch_revision",
        "cuda",
        "deepgemm_sm_budget",
        "installed_binary_sha256",
        "integration_source_sha256",
    ):
        if left[key] != right[key]:
            raise AssertionError(f"Compared arms differ on {key}")
    if left["cases"].keys() != right["cases"].keys():
        raise AssertionError("Compared arms cover different cases")
    deltas = {}
    for name, control in left["cases"].items():
        candidate = right["cases"][name]
        for key in ("k_sizes", "capacity_rows", "output_sha256_bf16"):
            if control[key] != candidate[key]:
                raise AssertionError(f"{name} differs on {key}")
        if control["activation"]["launches"] != candidate["activation"]["launches"]:
            raise AssertionError(f"{name} changed GEMM launch geometry")
        if control["measurement"] is not None:
            a, b = (case["measurement"]["medians_ms"] for case in (control, candidate))
            deltas[name] = {key: b[key] - a[key] for key in a}
    return deltas


def campaign(args):
    if probe.sha256(args.fixture) != probe.FIXTURE_SHA256:
        raise RuntimeError("Expected the preserved native EP8 rank-zero fixture")
    manifest = json.loads((args.build_root / "manifest.json").read_text())
    if (
        manifest["artifact_sha256"] != ARTIFACT_SHA256
        or probe.sha256(args.build_root / manifest["artifact"]) != ARTIFACT_SHA256
    ):
        raise RuntimeError("Unexpected private BF16 overwrite build")
    args.output.mkdir(parents=True)
    measurements = []
    for repeat in range(args.pairs):
        order = ("A", "B") if repeat % 2 == 0 else ("B", "A")
        pair = {}
        for arm in order:
            directory = args.output / f"{repeat + 1}-{arm}"
            directory.mkdir()
            env = {
                **os.environ,
                SWITCH: str(int(arm == "B")),
                BUILD: str(args.build_root),
            }
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    __file__,
                    *sys.argv[1:],
                    "--worker-arm",
                    arm,
                    "--worker-dir",
                    str(directory),
                ],
                env=env,
                check=True,
            )
            pair[arm] = json.loads((directory / "receipt.json").read_text())
        measurements.append(
            {
                "order": order,
                "candidate_minus_control_ms": compare_pair(pair["A"], pair["B"]),
                "arms": pair,
            }
        )
    for repeat in measurements[1:]:
        compare_pair(measurements[0]["arms"]["A"], repeat["arms"]["B"])
    deltas = [pair["candidate_minus_control_ms"] for pair in measurements]
    receipt = {
        "fixture_sha256": probe.FIXTURE_SHA256,
        "candidate_artifact_sha256": ARTIFACT_SHA256,
        "runner_sha256": probe.sha256(__file__),
        "warmup": args.warmup,
        "samples_per_arm": args.samples,
        "scope": (
            "Fullgraph Inductor allocation and two mutating GIFT ops per measured call. "
            "Packed inputs are resident and CUDA Graphs are disabled. CUDA events may include "
            "launch gaps. Profiles are collected separately. This backward-only component "
            "does not run the autograd save or communication paths."
        ),
        "all_case_output_bytes_match": True,
        "median_candidate_minus_control_ms": {
            name: {key: median(delta[name][key] for delta in deltas) for key in timings}
            for name, timings in deltas[0].items()
        },
        "pairs": measurements,
    }
    (args.output / "compiled.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("reference-root", "build-root", "fixture", "output"):
        parser.add_argument(f"--{option}", type=probe.absolute_path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--worker-arm", choices=("A", "B"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-dir", type=probe.absolute_path, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if min(args.warmup, args.samples) < 1 or args.pairs < 2:
        parser.error("Require positive warmup and samples and at least two AB/BA pairs")
    if args.worker_arm is None:
        campaign(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
