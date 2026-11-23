"""Compare CTA order in compiled routed SwiGLU backward on captured Hopper data."""

import argparse
from collections import Counter
from hashlib import sha256
import inspect
import json
import linecache
import os
from pathlib import Path
from textwrap import dedent

from routed_swiglu_fwd_order_component import (
    FIXTURE_SHA256,
    FIXTURE_SHAPE,
    digest,
    summarize,
)


PRODUCTION_KERNEL = "_swiglu_quant_bwd_kernel"
CANDIDATE_KERNEL = "_routed_swiglu_bwd_feature_first"
DISPATCHER = "_swiglu_quantized_backward"
TILE = (32, 128)
NUM_WARPS = 4
DEEPGEMM_SMS = 116
INPUTS = ("dy", "x", "p", "weight", "weight_scale", "m_indices")
OUTPUTS = ("grad_data", "grad_scale", "dprobs", "dgrad")


def swapped_source(source):
    row = "pid_token = tl.program_id(axis=0)"
    col = "pid_hid = tl.program_id(axis=1)"
    definition = f"def {PRODUCTION_KERNEL}("
    if any(source.count(line) != 1 for line in (row, col, definition)) or (
        source.count("tl.program_id(") != 2
    ):
        raise ValueError("The native backward kernel CTA mapping changed")
    return (
        source.replace(definition, f"def {CANDIDATE_KERNEL}(")
        .replace(row, "pid_token = tl.program_id(axis=1)")
        .replace(col, "pid_hid = tl.program_id(axis=0)")
    )


def source_function(source, namespace, name):
    # Triton and Dynamo inspect the source of functions defined by the probe.
    path = str(Path(__file__).with_suffix(f".{name}.py"))
    linecache.cache[path] = (len(source), None, source.splitlines(keepends=True), path)
    exec(compile(source, path, "exec", dont_inherit=True), namespace)
    return namespace[name]


def make_candidate(swiglu, jit):
    kernel = swiglu._swiglu_quant_bwd_kernel
    swapped = swapped_source(kernel.src)
    candidate = jit(
        source_function(
            swapped, kernel.fn.__globals__ | {"__name__": __name__}, CANDIDATE_KERNEL
        )
    )
    source = dedent(inspect.getsource(swiglu._swiglu_quantized_backward))
    call = f"{PRODUCTION_KERNEL}[grid]("
    if source.count(call) != 1:
        raise ValueError("The native backward dispatcher no longer has one CTA launch")
    source = source.replace(call, f"{CANDIDATE_KERNEL}[(grid[1], grid[0])](")
    dispatch = source_function(
        source, vars(swiglu) | {CANDIDATE_KERNEL: candidate}, DISPATCHER
    )
    return dispatch, swapped, source


def compile_arm(dispatch, limit, gemm, torch):
    def operation(inputs):
        dy, x, probs, weight, weight_scale, m_indices = inputs
        data, scale, dprobs = dispatch(dy, x, probs, limit)
        dgrad = gemm(data, scale, weight, weight_scale, m_indices)
        return data, scale, dprobs, dgrad

    return torch.compile(
        operation,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False},
    )


def bitwise(left, right, torch, rows=None):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    for start in range(0, left.size(0), 1024):
        tile = slice(start, start + 1024)
        x = left[tile].cpu().contiguous().view(torch.uint8)
        y = right[tile].cpu().contiguous().view(torch.uint8)
        if rows is not None:
            x, y = x[rows[tile]], y[rows[tile]]
        if not torch.equal(x, y):
            return False
    return True


def check_parity(arms, inputs, fixture, torch):
    # DeepGEMM leaves padding rows unwritten. Its active rows and every byte
    # of the SwiGLU results participate in the exact comparison.
    valid = fixture["m_indices"] >= 0
    control, candidate = (arm(inputs) for arm in arms.values())
    return {
        "valid_dgrad_rows": valid.sum().item(),
        "bitwise_vs_capture": {
            name: bitwise(
                value, fixture[name], torch, valid if name == "dgrad" else None
            )
            for name, value in zip(OUTPUTS, control, strict=True)
            if name != "dprobs"
        },
        "bitwise_between_arms": {
            name: ref.stride() == value.stride()
            and bitwise(ref, value, torch, valid if name == "dgrad" else None)
            for name, ref, value in zip(OUTPUTS, control, candidate, strict=True)
        },
        "row_major_scale": control[1].is_contiguous(),
    }


def profile(operation, inputs, path, name, grid, torch):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        outputs = operation(inputs)
        torch.cuda.synchronize()
    del outputs
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_bytes())["traceEvents"]
    device = [
        e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    device.sort(key=lambda event: event["ts"])
    kernels = [e for e in device if e["cat"] == "kernel"]
    targets = [e for e in kernels if name in e["name"]]
    extra = Counter(e["name"] for e in kernels if name not in e["name"])
    return {
        "activation_verified": len(targets) == 1
        and tuple(targets[0]["args"].get("grid", ())) == (*grid, 1)
        and tuple(targets[0]["args"].get("block", ())) == (32 * NUM_WARPS, 1, 1)
        and all(
            wrong not in e["name"]
            for wrong in (PRODUCTION_KERNEL, CANDIDATE_KERNEL)
            if wrong != name
            for e in kernels
        ),
        "non_target_kernel_counts": dict(extra),
        "transfer_counts": dict(
            Counter(e["cat"] for e in device if e["cat"] != "kernel")
        ),
        "events": [
            {"name": e["name"], "duration_us": e["dur"], "launch": e.get("args", {})}
            for e in device
        ],
        "path": str(path),
        "sha256": digest(path),
    }


def time_pairs(arms, inputs, args, torch, device_memory_usage):
    from dense_dy_quant_transpose import measure

    for operation in arms.values():
        for _ in range(args.warmup):
            operation(inputs)
        torch.cuda.synchronize()
    pairs = []
    for index in range(args.pairs):
        order = ("A", "B") if index % 2 == 0 else ("B", "A")
        blocks = {}
        for arm in order:
            torch.cuda.empty_cache()
            for _ in range(4):
                arms[arm](inputs)
            block = measure(arms[arm], inputs, args.samples)
            _, block["gift_peak_gib"] = device_memory_usage(torch.cuda.memory_stats())
            blocks[arm] = block
        pairs.append({"order": order, "arms": blocks})
        a, b = (blocks[arm]["medians"]["cuda_event_ms"] for arm in ("A", "B"))
        print(f"Pair {index + 1} CUDA event A {a:.6f} ms, B {b:.6f} ms", flush=True)
    paired_medians = [
        {"arms": {arm: block["medians"] for arm, block in pair["arms"].items()}}
        for pair in pairs
    ]
    return {
        "pairs": pairs,
        "timing": {
            field: summarize(paired_medians, "A", "B", field)
            for field in paired_medians[0]["arms"]["A"]
        },
    }


def run(args, trace_paths):
    import deep_gemm
    import torch
    import triton
    from torchtitan.ops.float8 import deep_gemm_handle
    from torchtitan.ops.float8.triton_kernels import fused_swiglu_quantization as swiglu
    from torchtitan.tools.utils import device_memory_usage

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("The probe requires one visible Hopper GPU")
    if os.environ.get(swiglu._BWD_MN_MAJOR_SCALE_ENV, "") not in ("", "0"):
        raise RuntimeError("The probe requires native row-major backward scales")
    fixture_hash = digest(args.fixture)
    if fixture_hash != FIXTURE_SHA256:
        raise ValueError(f"Unexpected native SwiGLU capture SHA-256 {fixture_hash}")
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True, mmap=True)
    if tuple(fixture["x"].shape) != FIXTURE_SHAPE:
        raise ValueError(f"Unexpected native SwiGLU capture shape {fixture['x'].shape}")
    rows, width = FIXTURE_SHAPE
    if (swiglu.BWD_ROW_BLOCK, swiglu.COL_BLOCK) != TILE:
        raise ValueError("The native SwiGLU backward tile no longer matches the probe")
    row_tiles, col_tiles = rows // TILE[0], width // 2 // TILE[1]
    grids = {"A": (row_tiles, col_tiles), "B": (col_tiles, row_tiles)}
    deep_gemm.set_num_sms(DEEPGEMM_SMS)
    inputs = tuple(fixture[key].cuda() for key in INPUTS)
    derived, kernel_source, dispatch_source = make_candidate(swiglu, triton.jit)
    native = swiglu._swiglu_quantized_backward
    arms = {
        arm: compile_arm(
            dispatch,
            fixture["limit"],
            deep_gemm_handle._m_grouped_fp8_gemm_nt_contiguous,
            torch,
        )
        for arm, dispatch in (("A", native), ("B", derived))
    }
    parity = check_parity(arms, inputs, fixture, torch)
    exact = (
        parity["valid_dgrad_rows"] > 0
        and parity["row_major_scale"]
        and all(parity["bitwise_vs_capture"].values())
        and all(parity["bitwise_between_arms"].values())
    )
    receipt = {
        "input": str(args.fixture),
        "input_sha256": fixture_hash,
        "input_shape_stride_dtype": {
            key: [list(value.shape), list(value.stride()), str(value.dtype)]
            for key, value in zip(INPUTS, inputs, strict=True)
        },
        "clamp": fixture["limit"],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "deepgemm_sms": DEEPGEMM_SMS,
        "probe_sha256": digest(Path(__file__)),
        "native_swiglu_source_sha256": digest(Path(swiglu.__file__)),
        "candidate_kernel_sha256": sha256(kernel_source.encode()).hexdigest(),
        "candidate_dispatch_sha256": sha256(dispatch_source.encode()).hexdigest(),
        "compile": {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "cudagraphs": False,
        },
        "scope": "routed SwiGLU backward and stock DeepGEMM dgrad",
        "warmup": args.warmup,
        "samples_per_pair_arm": args.samples,
        "pair_count": args.pairs,
        "tile": TILE,
        "num_warps": NUM_WARPS,
        "grids": grids,
        "parity": parity,
        "bitwise": exact,
    }
    del fixture
    active = False
    if exact:
        torch._dynamo.config.error_on_recompile = True
        profiles = {
            arm: profile(arms[arm], inputs, trace_paths[arm], name, grids[arm], torch)
            for arm, name in (("A", PRODUCTION_KERNEL), ("B", CANDIDATE_KERNEL))
        }
        matched = all(
            profiles["A"][key] == profiles["B"][key]
            for key in ("non_target_kernel_counts", "transfer_counts")
        )
        active = matched and all(
            profile["activation_verified"] for profile in profiles.values()
        )
        receipt.update(profiles=profiles, non_target_launches_matched=matched)
        if active:
            receipt.update(time_pairs(arms, inputs, args, torch, device_memory_usage))
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    if not exact:
        raise RuntimeError(
            f"Routed SwiGLU backward bitwise parity failed. See {args.output}"
        )
    if not active:
        raise RuntimeError(
            f"Unexpected routed SwiGLU backward launch. See {args.output}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=16)
    args = parser.parse_args()
    if min(args.warmup, args.samples) < 1 or args.pairs < 4 or args.pairs % 2:
        parser.error(
            "Require positive warmup and samples, and at least four even pairs"
        )
    trace_paths = {
        arm: args.output.with_suffix(f".{arm}.trace.json") for arm in ("A", "B")
    }
    if any(path.exists() for path in (args.output, *trace_paths.values())):
        parser.error("Specify a fresh output path")
    run(args, trace_paths)


if __name__ == "__main__":
    main()
