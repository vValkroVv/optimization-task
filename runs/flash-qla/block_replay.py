"""Matched isolated GDN replay; run each variant in a fresh process.

Numerical metrics are evidence, not universal acceptance thresholds. CUDA Graph
timings are repeated warm-cache diagnostics, not production throughput.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch

from capture import NAMES, sha256, snapshot, tensor_meta


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metrics(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {reference.shape}")
    a, b = actual.detach().double(), reference.detach().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite": False, "max_abs": None, "rms": None, "relative_rms": None}
    error = a - b
    rms, norm = error.square().mean().sqrt().item(), b.square().mean().sqrt().item()
    return {"finite": True, "max_abs": error.abs().max().item(), "rms": rms,
            "reference_rms": norm, "relative_rms": rms / norm if norm else (0.0 if rms == 0 else None),
            "reference_zero": norm == 0}


def compare(actual, reference):
    return {name: metrics(actual[name], reference[name]) for name in ("output", *NAMES)}


def recurrent_double(inputs):
    """FP64 adaptation of FLA naive_recurrent_gated_delta_rule (MIT).

    Origin: fla/ops/gated_delta_rule/naive.py, FLA upstream revision
    1d2281ff1b7988d8070f00dba5ffa9747ae3a60e. Preserve recurrence;
    replace hardcoded fp32 with fp64 and expand grouped q/k value heads.
    Prefix only, zero initial state; no claim about long-context oracle parity.
    """
    q, k, v, raw_g, beta, A_log, dt_bias = inputs
    q, k, v, raw_g, beta, A_log, dt_bias = [t.double() for t in inputs]
    if v.shape[2] % q.shape[2]:
        raise ValueError("Nonintegral grouped-query head ratio")
    group = v.shape[2] // q.shape[2]
    q, k = q.repeat_interleave(group, 2), k.repeat_interleave(group, 2)
    decay = -A_log.exp() * torch.nn.functional.softplus(raw_g + dt_bias)
    state = v.new_zeros(v.shape[0], v.shape[2], k.shape[-1], v.shape[-1])
    output = []
    for i in range(q.shape[1]):
        state = state * decay[:, i].exp()[..., None, None]
        residual = v[:, i] - (state * k[:, i, :, :, None]).sum(-2)
        state = state + k[:, i, :, :, None] * (residual * beta[:, i, :, None])[..., None, :]
        output.append(torch.einsum("bhk,bhkv->bhv", q[:, i] * (q.shape[-1] ** -0.5), state))
    return torch.stack(output, 1)


def statistics_ms(samples):
    return {"median_ms": statistics.median(samples), "min_ms": min(samples),
            "max_ms": max(samples), "spread_ms": max(samples) - min(samples), "samples_ms": samples}


def measure(fn, args, prepare=None):
    """prepare is excluded from backward-only timing, including graph capture."""
    capture_stream = None
    if args.timer == "graph":
        # Backward follows its forward's stream. Warmup and the final untimed
        # preparatory F must use the same nondefault stream as graph capture.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(args.warmup):
                if prepare:
                    prepare()
                fn()
        capture_stream.synchronize()
    else:
        for _ in range(args.warmup):
            if prepare:
                prepare()
            fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    result = {"warmup": args.warmup, "repeats": args.repeats,
              "cache": "same-input warm cache; diagnostic only", "graph_replays_per_sample": args.replays,
              "starting_allocated_bytes": torch.cuda.memory_allocated(),
              "starting_reserved_bytes": torch.cuda.memory_reserved()}
    if args.timer == "graph":
        with torch.cuda.stream(capture_stream):
            if prepare:
                prepare()
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            fn()
        torch.cuda.current_stream().wait_stream(capture_stream)
        graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / args.replays)
        result["timer_scope"] = "CUDA Graph device replay elapsed; excludes Python/autograd submission and preparatory F"
        result["device_replay"] = statistics_ms(samples)
    else:
        events, walls = [], []
        for _ in range(args.repeats):
            if prepare:
                prepare()
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            begin.record()
            fn()
            end.record()
            end.synchronize()
            walls.append((time.perf_counter() - wall) * 1000)
            events.append(begin.elapsed_time(end))
        result["timer_scope"] = "eager call latency; includes host submission and potential device gaps, not pure kernel cost"
        result["cuda_event_latency"] = statistics_ms(events)
        result["synchronized_wall_latency"] = statistics_ms(walls)
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["memory_scope"] = (
        "Inclusive process peak above live fixture/input/output state; reset before "
        "backward preparatory forward and graph capture, so those allocations are included. "
        "Not an independent phase allocation or production memory saving.")
    result["reserved_memory_scope"] = (
        "Allocator cache is retained across numerical checks and prior timing scopes; "
        "reserved peak includes carried-over high-water reservations.")
    result["host_work_attribution"] = "unavailable; no subtraction estimate"
    result["wait_copy_communication_attribution"] = "unavailable; requires trace"
    return result


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def to_device(source):
    span = 0 if source.numel() == 0 else 1 + sum((n - 1) * s for n, s in zip(source.shape, source.stride()))
    result = torch.zeros(span, dtype=source.dtype, device="cuda").as_strided(source.shape, source.stride())
    return result.copy_(source)


def main(args, report):
    # TileLang reads these settings on import; never use the shared home cache.
    cache_root = args.output.resolve() / "cache" / "tilelang"
    temporary_root = cache_root / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_CACHE_DIR"] = str(cache_root)
    os.environ["TILELANG_TMP_DIR"] = str(temporary_root)
    report["tilelang_cache"] = {"cache": str(cache_root), "tmp": str(temporary_root)}
    from torchtitan.experiments.flash_qla.backend import install_backend, run_gdn
    from torchtitan.models.common.attention.linear.fla_custom_ops import gdn_chunk_op

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA replay requires authorized GPU execution")
    if args.fixture.with_suffix(".json").exists():
        manifest = json.loads(args.fixture.with_suffix(".json").read_text())
        if sha256(args.fixture) != manifest["fixture_sha256"]:
            raise ValueError("Fixture checksum differs from capture manifest")
        report["capture_manifest"] = manifest
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    report["activation_identity"] = install_backend(args.variant)
    report["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(), "python": sys.version,
                         "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                         "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                         "command": sys.argv, "pid": os.getpid()}
    root = Path(__file__).resolve().parents[2]
    # Contrib trees are flattened task files, not independent Git checkouts.
    report["sources"] = {"task_revision": git_revision(root),
                         "fla_upstream_revision": "1d2281ff1b7988d8070f00dba5ffa9747ae3a60e",
                         "flash_qla_upstream_revision": "a14dbf3474514e1674526431e960325f72081e9f",
                         "backend_sha256": sha256(root / "gift/torchtitan/experiments/flash_qla/backend.py"),
                         "harness_sha256": sha256(Path(__file__))}
    report["fixture_sha256"] = sha256(args.fixture)
    report["inputs"] = {n: tensor_meta(fixture[n]) for n in NAMES}
    inputs = []
    for n in NAMES:
        t = to_device(fixture[n])
        inputs.append(t.requires_grad_(True))
    cu = fixture.get("cu_seqlens")
    cu = None if cu is None else cu.cuda()
    grad = to_device(fixture["grad_out"])
    scale = fixture.get("scale")
    if scale is not None and scale != inputs[0].shape[-1] ** -0.5:
        raise ValueError("Actual GIFT opaque op has no scale override; nondefault fixture cannot match")

    def native():
        return run_gdn(args.variant, *inputs, cu_seqlens=cu, scale=scale)[0]

    def integrated(*leaves):
        return gdn_chunk_op(*leaves, cu)

    def numerical(fn, upstream):
        out = fn(*inputs)
        gradients = torch.autograd.grad(out, inputs, upstream)
        return {"output": snapshot(out), **{n: snapshot(t) for n, t in zip(NAMES, gradients)}}

    report["numerics"] = {}
    eager = numerical(integrated, grad)
    torch.save(eager, args.output / "eager.pt")
    if any(not torch.isfinite(t).all() for t in eager.values()):
        raise RuntimeError("Nonfinite eager output or gradients")
    report["numerics"]["captured_output"] = metrics(eager["output"], fixture["output"])
    zero = numerical(integrated, torch.zeros_like(grad))
    report["numerics"]["zero_upstream"] = {n: metrics(zero[n], torch.zeros_like(zero[n])) for n in NAMES}
    if any(not row["finite"] or row["max_abs"] != 0 for row in report["numerics"]["zero_upstream"].values()):
        raise RuntimeError("Zero-upstream gradient check failed")
    compiled = None
    if not args.skip_compiled:
        compiled = torch.compile(integrated, fullgraph=True)
        result = numerical(compiled, grad)
        torch.save(result, args.output / "compiled.pt")
        if any(not torch.isfinite(t).all() for t in result.values()):
            raise RuntimeError("Nonfinite compiled output or gradients")
        report["numerics"]["compiled_vs_eager"] = compare(result, eager)
    if args.compare:
        for path in args.compare:
            reference_report = json.loads((path / "report.json").read_text())
            if reference_report["fixture_sha256"] != report["fixture_sha256"]:
                raise ValueError("Cannot compare different captured fixtures")
            report["numerics"][str(path)] = compare(eager, torch.load(path / "eager.pt", weights_only=True))
    if args.prefix:
        if cu is not None:
            raise ValueError("FP64 prefix diagnostic currently supports unpacked capture only")
        prefix = [t[:, :args.prefix].detach().requires_grad_(True) if t.ndim > 1 else t.detach().requires_grad_(True) for t in inputs]
        double_inputs = [t.double().detach().requires_grad_(True) for t in prefix]
        reference = recurrent_double(double_inputs)
        reference_grads = torch.autograd.grad(reference, double_inputs, grad[:, :args.prefix].double())
        actual = run_gdn(args.variant, *prefix, cu_seqlens=None, scale=scale)[0]
        actual_grads = torch.autograd.grad(actual, prefix, grad[:, :args.prefix].contiguous())
        report["numerics"]["fp64_prefix"] = {"tokens": args.prefix, "scope": "short zero-state prefix; not long-context oracle",
            "metrics": {"output": metrics(actual, reference), **{n: metrics(a, b) for n, a, b in zip(NAMES, actual_grads, reference_grads)}}}
    torch.cuda.synchronize()
    report["timings"] = {}

    def native_forward():
        with torch.no_grad():
            native()

    backward_output = None

    def prepare_backward():
        nonlocal backward_output
        backward_output = native()

    def native_backward():
        torch.autograd.grad(backward_output, inputs, grad, retain_graph=True)

    def integrated_fb():
        out = integrated(*inputs)
        torch.autograd.grad(out, inputs, grad)

    def compiled_fb():
        out = compiled(*inputs)
        torch.autograd.grad(out, inputs, grad)

    def synthetic_sac():
        native_forward()
        integrated_fb()

    scopes = {"native_F": (native_forward, None), "native_B_preparatory_F_excluded": (native_backward, prepare_backward),
              "opaque_F_VJP_2F_B": (integrated_fb, None), "synthetic_SAC_3F_B": (synthetic_sac, None)}
    if compiled is not None:
        scopes["compiled_opaque_F_VJP_2F_B"] = (compiled_fb, None)
    for name, (fn, prepare) in scopes.items():
        if args.scope and name not in args.scope:
            continue
        print(f"[BLOCK_REPLAY] measuring {name}", flush=True)
        try:
            report["timings"][name] = measure(fn, args, prepare)
        except RuntimeError as error:
            text = str(error).lower()
            unsupported = any(marker in text for marker in (
                "stream is capturing", "not supported during capture", "not permitted during capture",
                "cudaerrorstreamcaptureunsupported", "operation failed due to a previous error during capture"))
            if args.timer != "graph" or not unsupported:
                raise
            graph_error = traceback.format_exc()
            report["timings"][name] = {"graph_status": "unsupported", "graph_error": graph_error}
            write_json(args.output / "report.json", report)
            eager_args = copy.copy(args)
            eager_args.timer = "eager"
            report["timings"][name].update(measure(fn, eager_args, prepare))
        write_json(args.output / "report.json", report)
    report["synthetic_SAC_limit"] = "3F+B operation-count diagnostic only; excludes model-level SAC schedule, sharding, communication, FP8 projections and optimizer"
    report["status"] = "measured; numerical acceptance requires interpretation of retained metrics"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--variant", choices=("fla", "external_gate", "fused_gate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, action="append", default=[])
    parser.add_argument("--timer", choices=("graph", "eager"), default="graph")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--prefix", type=int, default=64)
    parser.add_argument("--skip-compiled", action="store_true")
    parser.add_argument("--scope", action="append", help="Select one or more timing scope names")
    options = parser.parse_args()
    if min(options.warmup, options.repeats, options.replays) < 1 or options.prefix < 0:
        parser.error("warmup/repeats/replays must be positive; prefix must be nonnegative")
    options.output.mkdir(parents=True, exist_ok=True)
    if (options.output / "report.json").exists():
        parser.error("Output already has report.json; use a fresh directory")
    report = {"schema": 1, "variant": options.variant, "fixture": str(options.fixture), "status": "running"}
    try:
        main(options, report)
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(options.output / "report.json", report)
