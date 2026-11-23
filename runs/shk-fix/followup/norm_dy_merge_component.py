"""Screen the in-place dy merge against the separate Triton pass on one H100."""

import argparse
from importlib import import_module, util
from importlib.metadata import version
import inspect
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

import torch
import triton
import triton.language as tl
import yaml

from norm_skip import NATIVE_SHAPE, digest, mismatched_bytes, restore


GRAD_NAMES = ("dx", "dw", "dgate_up", "dgate_down")


def load_experimental_backward(path):
    package = import_module("torchtitan.models.common.norms.cute_gated_norm")
    name = f"{package.__name__}.backward"
    if name in sys.modules:
        raise RuntimeError("Load the experimental backward before the installed backend")
    spec = util.spec_from_file_location(name, path.resolve(strict=True))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load experimental Python source {path}")
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    package.backward = module
    return module.gated_norm_bwd


@triton.jit
def _merge_in_place(dst, branch_b, branch_c, count: tl.constexpr, block: tl.constexpr):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    valid = idx < count
    a = tl.load(dst + idx, valid, 0).to(tl.float32)
    b = tl.load(branch_b + idx, valid, 0).to(tl.float32)
    c = tl.load(branch_c + idx, valid, 0).to(tl.float32)
    tl.store(dst + idx, (a + b) + c, valid)


def prepare_backward(backward, forward, x, parameters, eps, gating_weight, skip, skip_first):
    _, rho, p, *weights = forward(x, *parameters, eps, gating_weight)
    grad_dtypes = tuple(parameter.dtype for parameter in parameters)

    def run(dst, branch_b, branch_c, *, fused):
        dy_branches = (branch_b, branch_c) if fused else None
        if not fused:
            size = dst.numel()
            _merge_in_place[(triton.cdiv(size, 1024),)](
                dst, branch_b, branch_c, size, 1024, num_warps=4, enable_fp_fusion=False
            )
        return backward(
            dst, x, rho.reshape(-1), p.reshape(-1, 16), *weights,
            gating_weight, grad_dtypes, skip=skip, skip_first=skip_first,
            dy_branches=dy_branches,
        )

    return run


def check_parity(run, original, branch_b, branch_c, dst):
    dst.copy_(original)
    control = (run(dst, branch_b, branch_c, fused=False), dst.clone())
    dst.copy_(original)
    candidate = (run(dst, branch_b, branch_c, fused=True), dst)
    mismatches = mismatched_bytes(
        ("merged_dy", *GRAD_NAMES), (control[1], *control[0]), (candidate[1], *candidate[0])
    )
    if any(mismatches.values()):
        raise RuntimeError(f"Bytewise parity failed {mismatches}")
    return mismatches


def check_tail(backward, forward, x, parameters, eps, gating_weight, branches):
    hidden = x.shape[-1]
    x_tail = x.reshape(-1, hidden)[:65].clone()
    tail = tuple(branch.reshape(-1, hidden)[:65].clone() for branch in branches)
    skip_tail = torch.roll(tail[0], shifts=3, dims=-1)
    dst = torch.empty_like(tail[0])
    # The first sum loses one with an intermediate BF16 round. The second
    # loses one if the FP32 adds are reassociated.
    for branch, first, last in zip(
        tail, (256, 1, -256), (2**25, -(2**25), 1), strict=True
    ):
        branch[0, 0] = first
        branch[-1, -1] = last

    results = {}
    for skip_first in (None, False, True):
        run = prepare_backward(
            backward, forward, x_tail, parameters, eps, gating_weight,
            None if skip_first is None else skip_tail, skip_first,
        )
        results[str(skip_first)] = check_parity(run, *tail, dst)
        if (dst[0, 0].item(), dst[-1, -1].item()) != (1.0, 1.0):
            raise RuntimeError("The two precision sentinels must both produce BF16 one")
    try:
        run(dst, dst, tail[2], fused=True)
    except ValueError as error:
        if "private first branch" not in str(error):
            raise
    else:
        raise RuntimeError("The fused op accepted an aliased merge destination")
    return results


def measure(run, reset, samples):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_stats()
    series = []
    for _ in range(samples):
        reset()
        torch.cuda.current_stream().synchronize()
        start.record()
        enqueue_start = perf_counter_ns()
        gradients = run()
        end.record()
        enqueued = perf_counter_ns()
        end.synchronize()
        finished = perf_counter_ns()
        series.append({
            "device_event_ms": start.elapsed_time(end),
            "host_submit_ms": (enqueued - enqueue_start) / 1e6,
            "wall_drain_ms": (finished - enqueued) / 1e6,
            "synchronized_call_ms": (finished - enqueue_start) / 1e6,
        })
        del gradients
    final = torch.cuda.memory_stats()
    if any(final[key] != initial[key] for key in ("num_alloc_retries", "num_ooms")):
        raise RuntimeError("Allocation retry or OOM during the measured window")
    return {
        "samples": series,
        "medians": {key: median(record[key] for record in series) for key in series[0]},
        "allocated_at_start_bytes": initial["allocated_bytes.all.current"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def profile(run, reset, output, *, fused):
    reset()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA)
    ) as profiler:
        gradients = run()
        torch.cuda.synchronize()
    del gradients
    profiler.export_chrome_trace(str(output))
    events = json.loads(output.read_bytes())["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    merge = [event for event in kernels if "_merge_in_place" in event["name"]]
    if len(merge) != int(not fused):
        raise RuntimeError(f"Expected {int(not fused)} separate merge, observed {len(merge)}")
    stages = {}
    for name in ("GnBwdDs", "GnBwdFin", "GnBwdDx", "GnBwdWFin"):
        (kernel,) = (event for event in kernels if name in event["name"])
        attributes = kernel["args"]
        stages[name] = {
            "name": kernel["name"],
            "duration_us": kernel["dur"],
            "stream": attributes["stream"],
            "grid": attributes["grid"],
            "block": attributes["block"],
            "shared_memory_bytes": attributes["shared memory"],
            "registers_per_thread": attributes["registers per thread"],
        }
    return {
        "merge_launches": len(merge),
        "merge_duration_us": sum(kernel["dur"] for kernel in merge),
        "device_kernel_sum_us": sum(kernel["dur"] for kernel in kernels),
        "stages": stages,
        "trace": str(output),
        "trace_sha256": digest(output),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental-backward", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-only", action="store_true", help="Run native and tail parity")
    parser.add_argument("--with-skip", action="store_true", help="Time a separate skip operand")
    parser.add_argument("--skip-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    traces = {
        arm: args.output.with_suffix(f".{arm}.trace.json")
        for arm in ("separate", "fused")
    }
    if any(path.exists() for path in (args.output, *traces.values())):
        parser.error("Specify a fresh output path")
    if args.warmup < 1 or args.samples < 1 or args.repeats < 2:
        parser.error("Require positive warmup and samples and at least two repeats")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose exactly one Hopper GPU")
    torch.cuda.set_device(0)

    gated_norm_bwd = load_experimental_backward(args.experimental_backward)
    from torchtitan.models.common.norms.cute_gated_norm.op import _forward

    if "dy_branches" not in inspect.signature(gated_norm_bwd).parameters:
        raise RuntimeError("Apply the dy merge experiment patch in the isolated checkout")
    deterministic = yaml.safe_load(args.resolved_config.read_bytes())["debug"]["deterministic"]
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        torch.utils.deterministic.fill_uninitialized_memory = False

    fixture = torch.load(args.fixture, map_location="cpu", mmap=True, weights_only=True)
    if fixture["torch_revision"] != torch.version.git_version:
        raise RuntimeError("Capture and replay use different PyTorch revisions")
    record = fixture["records"]["plain"]
    tensors, layouts = record["tensors"], record["layouts"]
    x, *parameters = (
        restore(tensors[name], layouts[name]) for name in ("x", "weight", "up", "down")
    )
    if "dy" in tensors:
        dy = restore(tensors["dy"], layouts["dy"])
        dy_source = "plain"
    else:
        tangent = fixture["first_tangent"]
        dy = restore(tangent["dy"], tangent["layout"])
        dy_source = "first_tangent"
    if any(tensor.shape != NATIVE_SHAPE or tensor.dtype != torch.bfloat16 for tensor in (x, dy)):
        raise RuntimeError("Expected BF16 native B2/S4096/H4096 activations and tangent")
    dy = dy.contiguous()
    branch_b = torch.roll(dy, shifts=1, dims=-1)
    branch_c = torch.roll(dy, shifts=-3, dims=-2)
    skip = torch.roll(dy, shifts=3, dims=-1) if args.with_skip else None
    skip_first = args.skip_first if args.with_skip else None
    dst = torch.empty_like(dy)
    eps, gating_weight = record["eps"], record["gating_weight"]

    run = prepare_backward(
        gated_norm_bwd, _forward, x, parameters, eps, gating_weight, skip, skip_first
    )
    native = check_parity(run, dy, branch_b, branch_c, dst)
    print("Native parity passed", flush=True)
    tail = check_tail(
        gated_norm_bwd, _forward, x, parameters, eps, gating_weight,
        (dy, branch_b, branch_c),
    )
    print("65-row tail parity passed", flush=True)
    measurements = []
    activation = {}
    if not args.parity_only:
        def reset():
            dst.copy_(dy)

        def separate():
            return run(dst, branch_b, branch_c, fused=False)

        def fused():
            return run(dst, branch_b, branch_c, fused=True)

        arms = {"separate": separate, "fused": fused}
        for arm in arms.values():
            for _ in range(args.warmup):
                reset()
                arm()
                torch.cuda.synchronize()
        for repeat in range(args.repeats):
            order = tuple(arms) if repeat % 2 == 0 else tuple(reversed(arms))
            for arm in order:
                result = measure(arms[arm], reset, args.samples)
                measurements.append({"arm": arm, "repeat": repeat, **result})
                print(f"{arm} repeat {repeat + 1} {result['medians']}", flush=True)
        activation = {
            arm: profile(run_arm, reset, traces[arm], fused=arm == "fused")
            for arm, run_arm in arms.items()
        }
        base = activation["separate"]["stages"]["GnBwdDs"]
        candidate = activation["fused"]["stages"]["GnBwdDs"]
        for key in ("grid", "block", "shared_memory_bytes"):
            if base[key] != candidate[key]:
                raise RuntimeError(f"Unexpected B1 {key} change")
    backward_path = Path(inspect.getfile(gated_norm_bwd))
    backend = inspect.getmodule(gated_norm_bwd)
    receipt = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "backward_path": str(backward_path),
        "backward_sha256": digest(backward_path),
        "fixture_sha256": digest(args.fixture),
        "resolved_config_sha256": digest(args.resolved_config),
        "deterministic": deterministic,
        "shape": list(NATIVE_SHAPE),
        "dy_source": dy_source,
        "other_branches": "native tangent rolled by +1 column and -3 sequence positions",
        "with_skip": args.with_skip,
        "skip": "native tangent rolled by +3 columns" if args.with_skip else None,
        "skip_first": skip_first,
        "b2_stages": backend.B2_SKIP_STAGES if args.with_skip else backend.B2_STAGES,
        "parity_only": args.parity_only,
        "warmup_calls_per_arm": 0 if args.parity_only else args.warmup,
        "timing_scope": (
            "Direct backend from standalone premerge or fused B1 through Wfin. "
            "Device events can include submission gaps. The timed calls start after "
            "the first-branch restore completes. Profiling uses separate invocations."
        ),
        "memory_scope": (
            "Both component arms hold the two read-only branches for the full call. "
            "An opaque production fusion can retain up to 128 MiB through B2 and Wfin "
            "if Inductor currently releases both immediately after its standalone merge."
        ),
        "parity": {"native": native, "tail_65_rows": tail},
        "activation": activation,
        "measurements": measurements,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
