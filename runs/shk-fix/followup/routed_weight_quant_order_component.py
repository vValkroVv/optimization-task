"""Compare CTA order in the native Hopper routed expert weight quantizer."""

import argparse
from hashlib import sha256
import json
import linecache
from pathlib import Path
from statistics import median

from routed_swiglu_fwd_order_component import (
    capture_graphs,
    digest,
    measure_pairs,
    profile,
    summarize,
)
from routed_weight_transpose_component import TOPOLOGIES, WEIGHT_DIMS


CHECKOUT = Path(__file__).resolve().parents[3]
NATIVE_KERNEL = "_block_quantization_kernel"
CANDIDATE_KERNEL = "_routed_weight_quant_feature_first"
BLOCK = 128
NUM_WARPS = 8
TRACE_SHA256 = "cd288911e3b562e7d7835e1ea2d51a0349b7fbff0b4a055bb1b5dae5c01c5183"
EP8_GRIDS = {"w13_E2FD": (384, 32), "w2_EDF": (512, 12)}
EVENT_FIELDS = ("event_us_per_call", "synchronized_us_per_call")
PEAK_FIELDS = ("peak_allocated_bytes", "peak_reserved_bytes", "gift_device_peak_gib")


def swapped_source(source):
    row = "pid_token = tl.program_id(axis=0)"
    column = "pid_hid = tl.program_id(axis=1)"
    definition = f"def {NATIVE_KERNEL}("
    if source.count("tl.program_id(") != 2 or any(
        source.count(anchor) != 1 for anchor in (row, column, definition)
    ):
        raise ValueError("The native weight quantizer CTA mapping changed")
    return (
        source.replace(definition, f"def {CANDIDATE_KERNEL}(")
        .replace(row, "pid_token = tl.program_id(axis=1)")
        .replace(column, "pid_hid = tl.program_id(axis=0)")
    )


def make_candidate(kernel, jit):
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


def make_launches(block_quantization, kernels, weight, torch):
    experts, rows, columns = weight.shape
    flat = weight.view(-1, columns)
    row_tiles, col_tiles = flat.shape[0] // BLOCK, columns // BLOCK
    output = torch.empty_like(flat, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (row_tiles, col_tiles), device=weight.device, dtype=torch.float32
    )
    grids = {"A": (row_tiles, col_tiles), "B": (col_tiles, row_tiles)}
    params = block_quantization.params_to_kernel_kwargs(
        block_quantization.STATIC_TRITON_PARAMS,
        columns,
        num_warps=NUM_WARPS,
        maxnreg=255,
    )
    tensor_args = block_quantization.tensor_to_kernel_args
    arguments = (
        *tensor_args(flat, 2),
        *tensor_args(output, 2),
        *tensor_args(scales, 2),
    )

    def make_launch(arm):
        def launch():
            return kernels[arm][grids[arm]](
                *arguments, BLOCK_SIZE=BLOCK, POWER_TWO_MAX_ROUND=True, **params
            )

        return launch

    return (
        {arm: make_launch(arm) for arm in grids},
        grids,
        (output.view_as(weight), scales.view(experts, rows // BLOCK, col_tiles)),
        params,
    )


def verify(block_quantization, weight, launches, planes, torch):
    native = block_quantization.block_quantization_128x128_fn(weight)
    parity = {}
    binaries = {}
    for arm, launch in launches.items():
        compiled = launch()
        ptx = compiled.asm["ptx"]
        binaries[arm] = {
            "name": compiled.name,
            "ptx_sha256": sha256(ptx.encode()).hexdigest(),
            "cubin_sha256": sha256(compiled.asm["cubin"]).hexdigest(),
            "registers_per_thread": compiled.n_regs,
            "shared_bytes": compiled.metadata.shared,
            "num_warps": compiled.metadata.num_warps,
        }
        parity[arm] = {
            field: (
                (actual.shape, actual.stride(), actual.dtype)
                == (reference.shape, reference.stride(), reference.dtype)
                and torch.equal(actual.view(bits), reference.view(bits))
            )
            for field, actual, reference, bits in zip(
                ("fp8_bytes", "fp32_scale_bits"),
                planes,
                native,
                (torch.uint8, torch.int32),
                strict=True,
            )
        }
    return parity, binaries


def edge_case(block_quantization, kernels, columns, torch):
    weight = torch.zeros((1, BLOCK, columns), dtype=torch.bfloat16, device="cuda")
    weight.view(torch.int16)[0, 0, 1] = -32768
    weight[:, :, BLOCK : 2 * BLOCK] = torch.finfo(torch.bfloat16).tiny
    for tile, amax in enumerate(
        (446.0, 448.0, 450.0, torch.finfo(torch.bfloat16).max), 2
    ):
        weight[0, 0, tile * BLOCK] = amax
    launches, _, planes, _ = make_launches(block_quantization, kernels, weight, torch)
    parity, _ = verify(block_quantization, weight, launches, planes, torch)
    scales = planes[1][0, 0, :6].tolist()
    return {
        "parity": parity,
        "signed_zero_input_bits": weight.view(torch.int16)[0, 0, :2].tolist(),
        "signed_zero_output_bytes": planes[0].view(torch.uint8)[0, 0, :2].tolist(),
        "selected_amax": [
            0.0,
            torch.finfo(torch.bfloat16).tiny,
            446.0,
            448.0,
            450.0,
            torch.finfo(torch.bfloat16).max,
        ],
        "selected_scales": scales,
        "input_has_distinct_zero_signs": weight.view(torch.int16)[0, 0, :2].tolist()
        == [0, -32768],
        "edge_scales_match": scales == [2.0**-99, 2.0**-99, 1.0, 1.0, 2.0, 2.0**120],
    }


def timed_replays(launches, args, torch, device_memory_usage):
    graphs = capture_graphs(launches, args.warmup, args.replays, torch)
    pairs = measure_pairs(graphs, args.pairs, args.replays, torch, device_memory_usage)
    return {
        "pairs": pairs,
        "summary": {
            label: {
                field: summarize(pairs, ref, trial, field) for field in EVENT_FIELDS
            }
            for label, ref, trial in (("AB", "A", "B"), ("AA", "A_control", "A_same"))
        },
        "peaks": {
            arm: {
                field: max(row["arms"][arm][field] for row in pairs)
                for field in PEAK_FIELDS
            }
            for arm in ("A", "B")
        },
        "median_host_per_graph_us": {
            arm: {
                field: median(row["arms"][arm][field] for row in pairs)
                for field in ("host_submit_us_per_graph", "host_wait_us_per_graph")
            }
            for arm in ("A", "B")
        },
    }


def all_bits(parity):
    return all(match for arm in parity.values() for match in arm.values())


def run_case(
    name, topology, projection, block_quantization, kernels, args, torch, memory
):
    experts = TOPOLOGIES[topology]
    shape = (experts, *WEIGHT_DIMS[projection])
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    weight = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    weight.normal_(mean=0.0, std=0.02, generator=generator)
    launches, grids, planes, params = make_launches(
        block_quantization, kernels, weight, torch
    )
    if topology == "ep8" and grids["A"] != EP8_GRIDS[projection]:
        raise RuntimeError(
            f"Unexpected historical launch grid for {name}: {grids['A']}"
        )
    parity, binaries = verify(block_quantization, weight, launches, planes, torch)
    result = {
        "shape": shape,
        "input_stride": weight.stride(),
        "output_shape_stride_dtype": [
            (tuple(plane.shape), plane.stride(), str(plane.dtype)) for plane in planes
        ],
        "shape_origin": "observed EP8 launch"
        if topology == "ep8"
        else "derived EP4 expert count",
        "logical_min_bytes_per_launch": sum(
            tensor.numel() * tensor.element_size() for tensor in (weight, *planes)
        ),
        "grids": grids,
        "launch_params": params,
        "parity_vs_stock": parity,
        "binaries": binaries,
        "registers_delta": binaries["B"]["registers_per_thread"]
        - binaries["A"]["registers_per_thread"],
        "shared_bytes_delta": binaries["B"]["shared_bytes"]
        - binaries["A"]["shared_bytes"],
    }
    if not all_bits(parity):
        return result, False
    profiles = {
        arm: profile(
            launches[arm],
            kernel,
            grids[arm],
            args.output.with_suffix(f".{name}.{arm}.trace.json"),
            torch,
        )
        for arm, kernel in (("A", NATIVE_KERNEL), ("B", CANDIDATE_KERNEL))
    }
    active = all(
        item["activation_verified"]
        and tuple(item["kernels"][0]["launch"].get("block", ()))
        == (NUM_WARPS * 32, 1, 1)
        and binaries[arm]["num_warps"] == NUM_WARPS
        for arm, item in profiles.items()
    )
    result.update(profiles=profiles, activation_verified=active)
    if active:
        torch.cuda.empty_cache()
        result["timing"] = timed_replays(launches, args, torch, memory)
        print(name, result["timing"]["summary"]["AB"]["event_us_per_call"], flush=True)
    return result, active


def run(args, case_specs):
    import torch
    import triton
    from torchtitan.ops.float8.triton_kernels import block_quantization
    from torchtitan.tools.utils import device_memory_usage

    source = Path(block_quantization.__file__).resolve()
    if (
        not source.is_relative_to(CHECKOUT / "gift")
        or block_quantization._BLOCK_SIZE != BLOCK
    ):
        raise RuntimeError(f"Unexpected GIFT weight quantizer source or tile {source}")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("The probe needs one visible Hopper GPU")
    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    ):
        raise RuntimeError(
            "Disable deterministic empty-buffer fills for this standalone probe"
        )
    native = block_quantization._block_quantization_kernel
    candidate = make_candidate(native, triton.jit)
    kernels = {"A": native, "B": candidate}
    receipt = {
        "scope": "Resident weight quantizer CUDA Graph replay on fixed BF16 input and shared preallocated outputs",
        "peak_scope": "Includes the resident weight, outputs and both graphs, with no training or overlap",
        "input": {
            "kind": "seeded synthetic BF16",
            "distribution": "normal, mean=0, std=0.02",
            "seed": args.seed,
        },
        "historical_launch_trace_sha256": TRACE_SHA256,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "runner_sha256": digest(Path(__file__)),
        "native_source_sha256": digest(source),
        "native_kernel_sha256": sha256(native.src.encode()).hexdigest(),
        "candidate_kernel_sha256": sha256(candidate.src.encode()).hexdigest(),
        "block": (BLOCK, BLOCK),
        "warmup_launches_per_arm_case": args.warmup,
        "launches_per_graph": args.replays,
        "paired_replays_per_case": args.pairs,
        "cases": {},
    }
    with torch.no_grad():
        edges = {
            columns: edge_case(block_quantization, kernels, columns, torch)
            for _, columns in WEIGHT_DIMS.values()
        }
        receipt["signed_zero_and_edge_scale_checks"] = edges
        passed = all(
            edge["input_has_distinct_zero_signs"]
            and edge["edge_scales_match"]
            and all_bits(edge["parity"])
            for edge in edges.values()
        )
        for name, topology, projection in case_specs:
            if not passed:
                break
            receipt["cases"][name], passed = run_case(
                name,
                topology,
                projection,
                block_quantization,
                kernels,
                args,
                torch,
                device_memory_usage,
            )
    receipt["passed"] = passed
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    if not passed:
        raise RuntimeError(
            f"Routed weight quantizer parity or launch verification failed. See {args.output}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topology", choices=("ep8", "ep4", "both"), default="both")
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--replays", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=16)
    args = parser.parse_args()
    if min(args.warmup, args.replays) < 1 or args.pairs < 4 or args.pairs % 2:
        parser.error(
            "Require positive warmup and replays and an even pair count of at least four"
        )
    cases = [
        (f"{topology}-{projection.split('_')[0]}", topology, projection)
        for topology in TOPOLOGIES
        for projection in WEIGHT_DIMS
        if args.topology in (topology, "both")
    ]
    if args.output.exists() or any(
        args.output.with_suffix(f".{name}.{arm}.trace.json").exists()
        for name, _, _ in cases
        for arm in ("A", "B")
    ):
        parser.error("Specify a fresh output path")
    run(args, cases)


if __name__ == "__main__":
    main()
