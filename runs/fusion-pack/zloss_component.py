"""Replay the native final-norm input through the isolated hidden z-loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import triton

from torchtitan.models.common import regularization as reg


ROOT = Path(__file__).resolve().parents[2]
SWITCH = "GIFT_FUSED_HIDDEN_Z_LOSS"
COEF = 1e-7


def load_hidden(fixture: Path, rank: int):
    sys.path.insert(0, str(ROOT / "runs/production-data-abcd"))
    from component_fixtures import load_fixture

    tensors, receipt = load_fixture(fixture, rank)
    hidden = tensors["mtp"]["hidden"]
    if hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
        raise ValueError(
            f"Unexpected native final-norm input {hidden.dtype}, {hidden.stride()}"
        )
    return hidden.cuda().requires_grad_(), receipt


def loss(hidden):
    return reg.hidden_z_loss(hidden, coef=COEF)


def gradient(value, hidden, upstream=None):
    return torch.autograd.grad(value, hidden, grad_outputs=upstream)[0]


def same_bits(left, right, *, allow_nan_payload=False):
    integer_type = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[left.dtype]
    same = left.view(integer_type) == right.view(integer_type)
    if allow_nan_payload:
        same |= left.isnan() & right.isnan()
    return bool(same.all())


def compare(hidden, upstream, *, allow_nan_payload=False):
    os.environ[SWITCH] = "0"
    ref = loss(hidden)
    ref_grad = gradient(ref, hidden, upstream)
    os.environ[SWITCH] = "1"
    out = loss(hidden)
    grad = gradient(out, hidden, upstream)
    results = {
        "loss_bits": same_bits(ref, out, allow_nan_payload=allow_nan_payload),
        "gradient_bits": same_bits(ref_grad, grad, allow_nan_payload=allow_nan_payload),
        "loss_isfinite": bool(ref.isfinite()),
        "gradient_isfinite": bool(ref_grad.isfinite().all()),
    }
    if not (results["loss_bits"] and results["gradient_bits"]):
        raise AssertionError(f"Hidden z-loss differs: {results}")
    return results


def hook(hidden, upstream):
    module = torch.nn.Identity()
    reg.attach_hidden_z_loss(reg.HiddenZLoss(coef=COEF, on="input"), module)
    state = reg.wire_aux_loss(module, device=hidden.device)
    state.scale.fill_(0.25)
    output = module(hidden)
    grad = gradient(output, hidden, upstream)
    torch.testing.assert_close(output, hidden, atol=0, rtol=0)
    module.eval()
    assert module(hidden) is hidden
    return grad, state.totals


def check(hidden):
    from torchtitan.ops.hidden_z_loss import magnitude_logsumexp

    unit = torch.ones((), device=hidden.device)
    zero = torch.zeros_like(unit)
    cases = {
        "native": compare(hidden, unit),
        "native_zero_upstream": compare(hidden, zero),
    }
    with torch.no_grad():
        ref_lse = hidden.float().abs().logsumexp(dim=-1)
        cases["native_row_logsumexp_bits"] = same_bits(
            ref_lse, magnitude_logsumexp(hidden)
        )
    assert cases["native_row_logsumexp_bits"]
    # An input near exp underflow exercises BF16 subnormal gradient stores.
    small = torch.zeros((1, 4096), device=hidden.device, dtype=hidden.dtype)
    small[0, 0] = 90.0
    small[0, 1] = 0.5
    small.requires_grad_()
    reference = small.float().abs().logsumexp(dim=-1).sum()
    ref_grad = gradient(reference, small)
    grad = gradient(magnitude_logsumexp(small).sum(), small)
    cases["subnormal_gradient_bits"] = same_bits(ref_grad, grad)
    assert 0 < ref_grad[0, 1].abs() < torch.finfo(torch.bfloat16).tiny
    assert cases["subnormal_gradient_bits"]
    edges = torch.zeros((3, 4096), dtype=hidden.dtype, device=hidden.device)
    edges[0, :4] = torch.tensor([0.0, -0.0, 3.0e38, -3.0e38], device=hidden.device)
    edges[1, :3] = torch.tensor([0.0, -0.0, 1.5], device=hidden.device)
    cases["zero_and_large"] = compare(
        edges.requires_grad_(), unit, allow_nan_payload=True
    )
    special = edges.detach().clone()
    special[0, :3] = torch.tensor(
        [0.0, float("inf"), -float("inf")], device=hidden.device
    )
    special[1, 0] = float("nan")
    cases["nonfinite"] = compare(special.requires_grad_(), unit, allow_nan_payload=True)
    cases["nonfinite_zero_upstream"] = compare(special, zero, allow_nan_payload=True)

    hook_input = hidden[:, :2, :].contiguous().detach().requires_grad_()
    upstream = torch.zeros_like(hook_input)
    os.environ[SWITCH] = "0"
    ref_grad, ref_total = hook(hook_input, upstream)
    os.environ[SWITCH] = "1"
    grad, total = hook(hook_input, upstream)
    cases["input_hook_total_and_gradient_bits"] = same_bits(
        ref_grad, grad
    ) and same_bits(ref_total, total)
    assert cases["input_hook_total_and_gradient_bits"]
    if not (cases["native"]["loss_isfinite"] and cases["native"]["gradient_isfinite"]):
        raise AssertionError("Native hidden z-loss fixture must be finite")
    return cases


def full_step(hidden):
    return gradient(loss(hidden), hidden)


def summarize(samples):
    return {"median": statistics.median(samples), "samples": samples}


def pairs(events, phase):
    """Summarize the serial device stream by correlated CPU launch range."""
    (region,) = (
        e for e in events if e.get("cat") == "user_annotation" and e["name"] == phase
    )
    host = [
        e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        # PyTorch can submit backward kernels from an autograd worker thread.
        and e["pid"] == region["pid"]
        and region["ts"] <= e["ts"] < region["ts"] + region["dur"]
    ]
    ids = {e.get("args", {}).get("correlation") for e in host} - {None}
    device = sorted(
        (
            e
            for e in events
            if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
            and e.get("args", {}).get("correlation") in ids
        ),
        key=lambda e: e["ts"],
    )
    streams = {(e["pid"], e["args"]["stream"]) for e in device}
    if len(streams) != 1:
        raise ValueError(
            f"Expected one correlated device stream for {phase}, saw {streams}"
        )
    kernels = [e for e in device if e["cat"] == "kernel"]
    copies = [e for e in device if e["cat"] != "kernel"]
    return {
        "host_scope_us": region["dur"],
        "host_cuda_wait_us": sum(e["dur"] for e in host if "Synchronize" in e["name"]),
        "device_kernel_us": sum(e["dur"] for e in kernels),
        "device_copy_or_memset_us": sum(e["dur"] for e in copies),
        "device_span_us": device[-1]["ts"] + device[-1]["dur"] - device[0]["ts"],
        "device_between_events_us": sum(
            max(0.0, right["ts"] - left["ts"] - left["dur"])
            for left, right in zip(device, device[1:])
        ),
        "kernel_launches": [
            {
                "name": e["name"],
                "duration_us": e["dur"],
                "grid": e["args"]["grid"],
                "block": e["args"]["block"],
            }
            for e in kernels
        ],
    }


def benchmark(hidden, trace_path, *, fused):
    for _ in range(10):
        full_step(hidden)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        with torch.profiler.record_function("hidden_z_loss_forward"):
            value = loss(hidden)
        with torch.profiler.record_function("hidden_z_loss_backward"):
            grad = gradient(value, hidden)
        torch.cuda.synchronize()
    peaks = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    del value, grad
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())["traceEvents"]
    trace = {
        phase: pairs(events, f"hidden_z_loss_{phase}")
        for phase in ("forward", "backward")
    }
    for phase, name in (
        ("forward", "_shifted_exp_kernel"),
        ("backward", "_magnitude_grad_kernel"),
    ):
        launches = trace[phase]["kernel_launches"]
        if any(name in e["name"] for e in launches) != fused:
            raise AssertionError(f"Unexpected CUDA kernel dispatch in {phase}")

    eager_ms = []
    for _ in range(10):
        torch.cuda.synchronize()
        start_wall = time.perf_counter()
        for _ in range(10):
            full_step(hidden)
        torch.cuda.synchronize()
        eager_ms.append((time.perf_counter() - start_wall) * 100)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_grad = full_step(hidden)
    for _ in range(10):
        graph.replay()
    graph_ms = []
    for _ in range(10):
        start_event, end_event = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start_event.record()
        for _ in range(20):
            graph.replay()
        end_event.record()
        end_event.synchronize()
        graph_ms.append(start_event.elapsed_time(end_event) / 20)
    del graph_grad
    return {
        "profile_memory_bytes": {
            "baseline_live": start,
            "forward_backward_peak": peaks,
        },
        "trace_us": trace,
        "eager_wall_ms_per_call": summarize(eager_ms),
        "cuda_graph_event_ms_per_replay": summarize(graph_ms),
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "baseline", "fused"))
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hidden, receipt = load_hidden(args.fixture, args.rank)
    info = {
        "mode": args.mode,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "fixture": str(args.fixture),
        "fixture_rank": args.rank,
        "fixture_source_head": receipt["source_head"],
        "tensors_sha256": receipt["tensors_sha256"],
        "input_ids_int64_sha256": receipt["input_ids_int64_sha256"],
        "fixture_component": "mtp.hidden, input to the final norm in Decoder._forward_mtp",
        "shape": hidden.shape,
        "stride": hidden.stride(),
        "dtype": str(hidden.dtype),
        "coefficient": COEF,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_revision": torch.version.git_version,
        "triton": triton.__version__,
    }
    if args.mode == "check":
        info["checks"] = check(hidden)
    else:
        os.environ[SWITCH] = "1" if args.mode == "fused" else "0"
        info.update(
            benchmark(
                hidden,
                args.output.with_suffix(".trace.json"),
                fused=args.mode == "fused",
            )
        )
    args.output.write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
