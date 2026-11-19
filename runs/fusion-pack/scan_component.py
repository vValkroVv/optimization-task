"""Replay the FlashQLA opaque VJP on a captured native GDN boundary."""

import argparse
from collections import Counter
from functools import partial
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flash-qla"))
from block_replay import to_device
from capture import NAMES, sha256, tensor_meta

SWITCH = "GIFT_FLASH_QLA_DIRECT_BACKWARD"
OUTPUT_SCAN = "tilelang_fused_chunk_gdr_fwd_kernel"


def compare_bits(actual, expected):
    checks = {}
    for name, value, reference in zip(("output", *NAMES), actual, expected, strict=True):
        left = value.detach().contiguous().view(torch.uint8)
        right = reference.detach().contiguous().view(torch.uint8)
        checks[name] = value.dtype == reference.dtype and torch.equal(left, right)
    if not all(checks.values()):
        raise ValueError(f"Different output or gradient bits: {checks}")
    return checks


def elapsed_calls(fn, samples):
    events, submit, sync, wall = [], [], [], []
    before, after = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    for _ in range(samples):
        before.record()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        after.record()
        after.synchronize()
        t2 = time.perf_counter()
        events.append(before.elapsed_time(after))
        submit.append((t1 - t0) * 1000)
        sync.append((t2 - t1) * 1000)
        wall.append((t2 - t0) * 1000)
    return {
        "cuda_event_elapsed_ms": events,
        "host_call_ms": submit,
        "event_record_and_wait_ms": sync,
        "synchronized_call_ms": wall,
    }


def device_graph(make_call, leaves, samples):
    # Autograd runs backward on the stream that ran forward for its node.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Share immutable fixture storage with fresh leaf identities so their
        # AccumulateGrad nodes cannot inherit the default stream from eager timing.
        graph_leaves = tuple(t.detach().requires_grad_(True) for t in leaves)
        fn = make_call(graph_leaves)
        for _ in range(5):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    graph.replay()
    torch.cuda.synchronize()

    before, after = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    result = []
    for _ in range(samples):
        before.record()
        for _ in range(20):
            graph.replay()
        after.record()
        after.synchronize()
        result.append(before.elapsed_time(after) / 20)
    return result


def memory(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = [torch.cuda.memory_allocated(), torch.cuda.memory_reserved()]
    fn()
    torch.cuda.synchronize()
    return {
        "start_allocated_bytes": start[0],
        "start_reserved_bytes": start[1],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def trace(fn, path):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as profiler:
        with torch.profiler.record_function("scan_component"):
            fn()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    launches = sorted(
        (event for event in events if event.get("cat") in {"kernel", "gpu_memset", "gpu_memcpy"}),
        key=lambda event: event["ts"],
    )
    streams = {event["args"]["stream"] for event in launches}
    if len(streams) != 1:
        raise ValueError(f"Expected a single stream in {path.name}, found {streams}")
    launches_by_kind = Counter(event["cat"] for event in launches)
    kernels = [event for event in launches if event["cat"] == "kernel"]
    span = launches[-1]["ts"] + launches[-1]["dur"] - launches[0]["ts"]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "launches_by_kind": dict(launches_by_kind),
        "device_span_us": span,
        "device_gap_us": sum(
            max(0.0, right["ts"] - left["ts"] - left["dur"])
            for left, right in zip(launches, launches[1:])
        ),
        "kernel_time_us": sum(event["dur"] for event in kernels),
        "copy_and_memset_time_us": sum(event["dur"] for event in launches if event["cat"] != "kernel"),
        "kernels": [
            [event["name"], event["args"]["grid"], event["args"]["block"], event["dur"]]
            for event in kernels
        ],
    }


def kernel_signatures(result):
    return [kernel[:3] for kernel in result["kernels"]]


def run(args):
    args.output.mkdir(parents=True)
    caches = {
        "TILELANG_CACHE_DIR": "tilelang", "TILELANG_TMP_DIR": "tilelang-tmp",
        "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor",
    }
    for key, name in caches.items():
        directory = args.output.resolve() / name
        directory.mkdir()
        os.environ[key] = str(directory)

    from torchtitan.distributed.activation_checkpoint import _sac_context_fn
    from torchtitan.experiments.flash_qla.backend import install_backend
    from torchtitan.models.common.attention.linear.fla_custom_ops import gdn_chunk_op

    if torch.cuda.device_count() != 1:
        raise ValueError("Expose only the assigned CUDA device before launching")
    manifest = json.loads(args.fixture.with_suffix(".json").read_text())
    digest = sha256(args.fixture)
    if digest != manifest["fixture_sha256"]:
        raise ValueError("The saved fixture hash does not match its native capture manifest")
    fixture = torch.load(args.fixture, weights_only=True, map_location="cpu")
    leaves = tuple(to_device(fixture[name]).requires_grad_(True) for name in NAMES)
    incoming = to_device(fixture["grad_out"])
    cu = fixture.get("cu_seqlens")
    cu = to_device(cu) if cu is not None else None

    def eager(*xs):
        return gdn_chunk_op(*xs, cu)

    def sac(*xs):
        return checkpoint(eager, *xs, use_reentrant=False, context_fn=partial(_sac_context_fn, (), ()))

    compiled_sac = torch.compile(sac, fullgraph=True)

    def results(model=eager, inputs=leaves):
        out = model(*inputs)
        return (out, *torch.autograd.grad(out, inputs, incoming))

    def backward_call(inputs):
        out = eager(*inputs)

        def run_backward():
            return torch.autograd.grad(out, inputs, incoming, retain_graph=True)

        return run_backward

    def compiled_sac_call(inputs):
        return partial(results, compiled_sac, inputs)

    def activate(mode):
        os.environ[SWITCH] = str(int(mode == "direct"))
        identity = install_backend("external_gate")
        if identity["direct_backward"] != (mode == "direct"):
            raise ValueError(f"Wrong backend activation: {identity}")
        return identity

    checks = {}
    identity = {"baseline": activate("baseline")}
    baseline_eager = results()
    if manifest.get("vjp") == "external_gate baseline":
        captured_output = to_device(fixture["output"])
        if not torch.equal(baseline_eager[0].view(torch.uint8), captured_output.contiguous().view(torch.uint8)):
            raise ValueError("Baseline replay differs from the captured native external-gate output")
        checks["baseline_matches_captured_output_bits"] = True
        del captured_output
    baseline_sac = results(compiled_sac)
    checks["baseline_compiled_sac_vs_eager"] = compare_bits(baseline_sac, baseline_eager)
    identity["direct"] = activate("direct")
    checks["direct_eager_vs_baseline_eager"] = compare_bits(results(), baseline_eager)
    checks["direct_compiled_sac_vs_baseline_compiled_sac"] = compare_bits(results(compiled_sac), baseline_sac)
    for value in baseline_eager:
        if not torch.isfinite(value).all().item():
            raise ValueError("Native reference contains nonfinite values")
    del baseline_eager, baseline_sac

    measurements = {}
    traces = {}
    for scope, make_call in [("backward", backward_call), ("compiled_sac", compiled_sac_call)]:
        blocks = []
        for position, mode in enumerate(("baseline", "direct", "direct", "baseline")):
            activate(mode)
            fn = make_call(leaves)
            for _ in range(5):
                fn()
            row = {"mode": mode, **elapsed_calls(fn, args.samples), **memory(fn)}
            if position in (0, 1):
                traces[f"{scope}_{mode}"] = trace(fn, args.output / f"{scope}_{mode}_trace.json")
            try:
                row["cuda_graph_replay_ms"] = device_graph(make_call, leaves, args.samples)
            except RuntimeError as exc:
                errors = (
                    "stream is capturing", "not permitted during capture",
                    "not supported during capture", "stale reference to the default stream",
                )
                if not any(error in str(exc).lower() for error in errors):
                    raise
                row["cuda_graph_capture_error"] = str(exc)
            blocks.append(row)
            del fn
        before = kernel_signatures(traces[f"{scope}_baseline"])
        after = kernel_signatures(traces[f"{scope}_direct"])
        output_indices = [i for i, kernel in enumerate(before) if OUTPUT_SCAN in kernel[0]]
        if not any(before[:i] + before[i + 1:] == after for i in output_indices):
            raise ValueError(f"Expected exactly one removed output scan in {scope}: {before} -> {after}")
        measurements[scope] = blocks

    import tilelang
    import triton
    from fla.ops.gated_delta_rule import gate

    return {
        "fixture": str(args.fixture.resolve()),
        "fixture_sha256": digest,
        "manifest": manifest,
        "identity": identity,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "tilelang": tilelang.__version__,
        "fla_gate_source_sha256": sha256(Path(gate.__file__)),
        "harness_sha256": sha256(Path(__file__)),
        "device": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": sys.argv,
        "inputs": {name: tensor_meta(fixture[name]) for name in (*NAMES, "grad_out")},
        "cu_seqlens": None if cu is None else tensor_meta(cu),
        "checks": checks,
        "measurements": measurements,
        "traces": traces,
        "median_ms": {
            scope: [
                {key: statistics.median(value) for key, value in block.items() if key.endswith("_ms")}
                for block in blocks
            ]
            for scope, blocks in measurements.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    arguments = parser.parse_args()
    result = run(arguments)
    (arguments.output / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
