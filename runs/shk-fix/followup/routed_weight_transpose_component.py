#!/usr/bin/env python3
"""Compare the native and generated compiled routed FP8 weight transposes."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import platform
from statistics import median


CHECKOUT = Path(__file__).resolve().parents[3]
# GroupedExperts flattens the first two projections and quantizes 128x128 blocks.
WEIGHT_DIMS = {"w13_E2FD": (3072, 4096), "w2_EDF": (4096, 1536)}
TOPOLOGIES = {"ep8": 16, "ep4": 32}
NATIVE_KERNEL = "_transpose_last_two_kernel"


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def compile_transposes(native_transpose, torch):
    def native(inputs):
        weight, scale = inputs
        return native_transpose(weight), native_transpose(scale)

    def generated(inputs):
        weight, scale = inputs
        data = weight.view(torch.uint8).mT.contiguous().view(torch.float8_e4m3fn)
        return data, scale.mT.contiguous()

    return {
        arm: torch.compile(
            fn,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        for arm, fn in (("A", native), ("B", generated))
    }


def make_inputs(shape, seed, device, torch):
    experts, rows, columns = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    data = torch.randint(
        256, shape, dtype=torch.uint8, generator=generator, device=device
    ).view(torch.float8_e4m3fn)
    scale = (
        torch.randint(
            -16,
            9,
            (experts, rows // 128, columns // 128),
            dtype=torch.int32,
            generator=generator,
            device=device,
        )
        .float()
        .exp2()
    )
    return data, scale


def verify_outputs(chains, inputs, torch):
    a, b = (chains[arm](inputs) for arm in ("A", "B"))
    result = {}
    for name, reference, native, generated, bits in zip(
        ("fp8_data", "fp32_scales"),
        inputs,
        a,
        b,
        (torch.uint8, torch.int32),
        strict=True,
    ):
        wanted = reference.view(bits).mT
        matches = (
            native.shape == generated.shape == wanted.shape
            and native.dtype == generated.dtype == reference.dtype
            and native.is_contiguous()
            and generated.is_contiguous()
            and torch.equal(native.view(bits), wanted)
            and torch.equal(native.view(bits), generated.view(bits))
        )
        if not matches:
            raise AssertionError(f"{name} transpose changed bits or output layout")
        result[name] = {
            "input_shape": list(reference.shape),
            "input_stride": list(reference.stride()),
            "output_shape": list(native.shape),
            "output_stride": list(native.stride()),
            "bitwise_and_contiguous": matches,
            "logical_read_write_bytes": 2
            * reference.numel()
            * reference.element_size(),
        }
    return result


def profile(chain, inputs, arm, device, torch):
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as trace:
        outputs = chain(inputs)
        torch.cuda.synchronize(device)
    del outputs
    kernels = [
        {"name": event.name, "duration_us": event.time_range.elapsed_us()}
        for event in sorted(trace.events(), key=lambda event: event.time_range.start)
        if event.device_type.name == "CUDA"
    ]
    native_calls = sum(NATIVE_KERNEL in event["name"] for event in kernels)
    if not kernels or native_calls != (2 if arm == "A" else 0):
        raise AssertionError(f"Unexpected arm {arm} transpose kernels: {kernels}")
    return {
        "cuda_events": kernels,
        "cuda_event_sum_us": sum(event["duration_us"] for event in kernels),
    }


def measure_case(chains, inputs, args, device, measure, torch):
    blocks = []
    with torch._dynamo.config.patch(error_on_recompile=True):
        for chain in chains.values():
            for _ in range(args.warmup):
                chain(inputs)
            torch.cuda.synchronize(device)
        for repeat in range(args.repeats):
            order = ("A", "B") if repeat % 2 == 0 else ("B", "A")
            arms = {arm: measure(chains[arm], inputs, args.samples) for arm in order}
            a, b = (arms[arm]["median"] for arm in ("A", "B"))
            blocks.append(
                {
                    "order": order,
                    "arms": arms,
                    "candidate_minus_control_ms": {
                        name: b[name] - value for name, value in a.items()
                    },
                }
            )
        activation = {
            arm: profile(chain, inputs, arm, device, torch)
            for arm, chain in chains.items()
        }
    return {
        "blocks": blocks,
        "median_candidate_minus_control_ms": {
            name: median(row["candidate_minus_control_ms"][name] for row in blocks)
            for name in blocks[0]["candidate_minus_control_ms"]
        },
        "profiles": activation,
    }


def run(args):
    import torch
    import triton

    from dense_weight_scale import measure
    from torchtitan.ops.float8.triton_kernels import block_transpose

    source = Path(block_transpose.__file__).resolve()
    if not source.is_relative_to(CHECKOUT / "gift"):
        raise RuntimeError(f"Unexpected GIFT import: {source}")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (9, 0) or "H100" not in properties.name:
        raise RuntimeError(f"Expected H100, found {properties}")
    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    ):
        raise RuntimeError(
            "Deterministic empty-buffer fills would change the measurement"
        )

    cases = {}
    with torch.no_grad():
        for topology, experts in TOPOLOGIES.items():
            for weight, (rows, columns) in WEIGHT_DIMS.items():
                name = f"{topology}-{weight}"
                torch._dynamo.reset()
                inputs = make_inputs((experts, rows, columns), args.seed, device, torch)
                chains = compile_transposes(
                    block_transpose.transpose_last_two_dims_fn, torch
                )
                parity = verify_outputs(chains, inputs, torch)
                results = measure_case(chains, inputs, args, device, measure, torch)
                del inputs
                cases[name] = {"parity": parity, **results}
                print(name, results["median_candidate_minus_control_ms"], flush=True)

    receipt = {
        "hostname": platform.node(),
        "gpu": properties.name,
        "torch": torch.__version__,
        "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "native_source_sha256": digest(source),
        "runner_sha256": digest(Path(__file__)),
        "seed": args.seed,
        "warmup_per_arm_case": args.warmup,
        "samples_per_arm_block": args.samples,
        "ab_ba_paired_blocks": args.repeats,
        "scope": (
            "Seeded synthetic E4M3 bytes and FP32 scales at captured/configured native shapes. "
            "One fullgraph Inductor call per sample returns both contiguous planes. "
            "CUDA events include possible launch gaps and wall covers submission and drain. "
            "Allocator peaks are warmed same-process measurements. "
            "CUDA Graphs, GEMMs, autograd and collectives are outside this component."
        ),
        "arms": {
            "A": "GIFT Triton transpose",
            "B": "Inductor-generated transpose and copy",
        },
        "cases": cases,
    }
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if min(args.warmup, args.samples) < 1 or args.repeats < 2:
        parser.error("Require positive warmup and samples and at least two repeats")
    run(args)


if __name__ == "__main__":
    main()
