"""Measure one isolated compiled Hopper component in a fresh process."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter_ns

import torch
import yaml


CASES = ("norm-residual", "dense-fp8", "sac-plain", "sac-residual")


def differentiate(y, leaves, tangent):
    return torch.autograd.grad(y, tuple(leaves.values()), tangent)


def tensor_signature(tensor):
    tensor = tensor.detach().cpu()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(
            tensor.contiguous().flatten().view(torch.uint8).numpy()
        ).hexdigest(),
    }


def phase_events(events, phase):
    scopes = [
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name") == f"isolated::{phase}"
    ]
    if len(scopes) != 1:
        raise RuntimeError(f"expected one {phase} range, got {len(scopes)}")
    scope = scopes[0]
    start = scope["ts"]
    end = start + scope["dur"]
    cpu_ops = Counter(
        event["name"]
        for event in events
        if event.get("cat") == "cpu_op"
        and event["name"].startswith("torchtitan::")
        and start <= event["ts"] <= end
    )
    work = sorted(
        (
            event
            for event in events
            if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
            and start <= event["ts"] <= end
        ),
        key=lambda event: event["ts"],
    )
    if not work:
        raise RuntimeError(f"no CUDA work in {phase} range")
    kernels = Counter(event["name"] for event in work if event["cat"] == "kernel")
    tail = work[0]["ts"]
    busy = 0.0
    for event in work:
        event_end = event["ts"] + event["dur"]
        busy += max(0.0, event_end - max(event["ts"], tail))
        tail = max(event_end, tail)
    span = tail - work[0]["ts"]
    return {
        "kernel_counts": dict(kernels),
        "custom_op_counts": dict(cpu_ops),
        "gpu_work_events": len(work),
        "kernel_sum_us": sum(
            event["dur"] for event in work if event["cat"] == "kernel"
        ),
        "copy_memset_event_count": len(work) - sum(kernels.values()),
        "copy_memset_sum_us": sum(
            event["dur"] for event in work if event["cat"] != "kernel"
        ),
        "gpu_span_us": span,
        "gpu_busy_union_us": busy,
        "gpu_gaps_us": max(0.0, span - busy),
        "streams": sorted({event["args"]["stream"] for event in work}),
    }


def profile(adapter, args, forward, leaves, tangent):
    trace_path = args.output.parent / f"{args.output.stem}.trace.json"
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as profiler:
        with torch.profiler.record_function("isolated::forward"):
            y = forward()
            torch.cuda.synchronize()
        with torch.profiler.record_function("isolated::backward"):
            gradients = differentiate(y, leaves, tangent)
            torch.cuda.synchronize()
    del y, gradients
    profiler.export_chrome_trace(str(trace_path))
    trace_bytes = trace_path.read_bytes()
    events = json.loads(trace_bytes)["traceEvents"]
    forward_trace = phase_events(events, "forward")
    backward_trace = phase_events(events, "backward")
    observed = forward_trace["gpu_work_events"] + backward_trace["gpu_work_events"]
    total = sum(
        event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"} for event in events
    )
    if observed != total:
        raise RuntimeError(
            f"phase ranges cover {observed} of {total} GPU events in {trace_path}"
        )
    return {
        "path": str(trace_path),
        "sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "forward": forward_trace,
        "backward": backward_trace,
        "kernel_activation": adapter.verify_kernels(
            args.case,
            args.arm,
            forward_trace["kernel_counts"],
            backward_trace["kernel_counts"],
        ),
        "custom_op_activation": adapter.verify_ops(
            args.case,
            args.arm,
            forward_trace["custom_op_counts"],
            backward_trace["custom_op_counts"],
        )
        if hasattr(adapter, "verify_ops")
        else None,
    }


def measure(forward, leaves, tangent, count):
    from torchtitan.tools.utils import device_capacity_report, reset_peak_stats

    start, middle, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    for event in (start, middle, end):
        event.record()
    torch.cuda.synchronize()

    samples = []
    for _ in range(count):
        reset_peak_stats()
        started = perf_counter_ns()
        start.record()
        y = forward()
        middle.record()
        gradients = differentiate(y, leaves, tangent)
        end.record()
        submitted = perf_counter_ns()
        torch.cuda.synchronize()
        drained = perf_counter_ns()
        _, total_peak = device_capacity_report()
        samples.append(
            {
                "forward_ms": start.elapsed_time(middle),
                "backward_ms": middle.elapsed_time(end),
                "wall_ms": (drained - started) / 1e6,
                "host_submission_ms": (submitted - started) / 1e6,
                "wall_drain_ms": (drained - submitted) / 1e6,
                "total_gpu_gib": total_peak,
            }
        )
        del y, gradients
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.warmup < 1 or args.samples < 1:
        parser.error("provide a new output path and positive warmup and sample counts")

    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("expose exactly one Hopper GPU per process")
    torch.cuda.set_device(0)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("unexpected CUDA Graph capture")
    config_bytes = args.resolved_config.read_bytes()
    resolved = yaml.safe_load(config_bytes)
    deterministic = resolved["debug"]["deterministic"]
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        torch.utils.deterministic.fill_uninitialized_memory = False
    inductor_config.triton.cudagraphs = False

    os.environ["GIFT_CUTE_NORM_FUSE_RESIDUAL"] = str(
        int(
            args.case == "sac-residual"
            or (args.case == "norm-residual" and args.arm == "B")
        )
    )
    os.environ["GIFT_CUTE_NORM_SAC_STATS_ONLY"] = str(
        int(args.case.startswith("sac-") and args.arm == "B")
    )
    os.environ["GIFT_FP8_QUANTIZE_WEIGHT_TAIL"] = str(
        int(args.case == "dense-fp8" and args.arm == "B")
    )
    if args.case == "dense-fp8":
        import dense as adapter
    else:
        import norm as adapter

    props = torch.cuda.get_device_properties(0)
    print(f"{args.case} {args.arm} {props.name}, starting compiled setup", flush=True)
    forward, tangent, leaves = adapter.build(args.case, args.arm, args.fixture)
    y = forward()
    gradients = differentiate(y, leaves, tangent)
    torch.cuda.synchronize()
    del y, gradients
    dynamo_config.error_on_recompile = True
    for _ in range(args.warmup):
        y = forward()
        gradients = differentiate(y, leaves, tangent)
        torch.cuda.synchronize()
        del y, gradients
    print(f"{args.case} {args.arm} completed {args.warmup} warmup calls", flush=True)
    pre = torch.cuda.memory_stats()
    samples = measure(forward, leaves, tangent, args.samples)
    post = torch.cuda.memory_stats()
    failure_deltas = {
        name: post[name] - pre[name] for name in ("num_alloc_retries", "num_ooms")
    }
    if any(failure_deltas.values()):
        raise RuntimeError(
            f"allocator failure or retry in measured calls: {failure_deltas}"
        )

    y = forward()
    gradients = differentiate(y, leaves, tangent)
    output_signatures = {
        "y": tensor_signature(y),
        **{
            name: tensor_signature(grad)
            for name, grad in zip(leaves, gradients, strict=True)
        },
    }
    del y, gradients
    receipt = {
        "case": args.case,
        "arm": args.arm,
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu": props.name,
        "torch_version": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "fixture": str(args.fixture),
        "resolved_config": str(args.resolved_config),
        "resolved_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "deterministic_algorithms": deterministic,
        "compile": {"backend": "inductor", "fullgraph": True, "cudagraphs": False},
        "switches": {
            key: os.environ[key]
            for key in (
                "GIFT_CUTE_NORM_FUSE_RESIDUAL",
                "GIFT_CUTE_NORM_SAC_STATS_ONLY",
                "GIFT_FP8_QUANTIZE_WEIGHT_TAIL",
            )
        },
        "warmup_calls": args.warmup,
        "samples": samples,
        "allocator_failure_deltas": failure_deltas,
        "output_signatures": output_signatures,
        "trace": profile(adapter, args, forward, leaves, tangent)
        if args.profile
        else None,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{args.case} {args.arm} wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
