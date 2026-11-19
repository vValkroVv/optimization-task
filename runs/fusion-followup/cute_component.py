"""Replay CuTe gated norm forward plus compiled SAC backward on native inputs."""

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from cute_liveness import digest, generated_calls


def timing_helpers():
    path = Path(__file__).resolve().parents[1] / "fusion-pack" / "scan_component.py"
    spec = importlib.util.spec_from_file_location("_fusion_pack_component_timing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bits_equal(actual, expected):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return False
    return torch.equal(
        actual.detach().contiguous().flatten().view(torch.uint8),
        expected.detach().contiguous().flatten().view(torch.uint8),
    )


def to_cuda(tensor):
    return torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device="cuda"
    ).copy_(tensor)


def compile_block(eps, gating_weight, enabled, *, live=False):
    from torchtitan.config import ActivationCheckpointConfig, CompileConfig
    from torchtitan.distributed.activation_checkpoint import _apply_op_sac
    from torchtitan.distributed.compile import apply_compile
    from torchtitan.models.common.norms.cute_gated_norm import gated_norm

    class Norm(nn.Module):
        def forward(self, x, weight, up, down, residual=None):
            y = gated_norm(
                x,
                weight,
                up,
                down,
                eps=eps,
                gating_weight=gating_weight,
                residual=residual,
            )
            return y.square() if live else y

    block = _apply_op_sac(
        Norm(), ActivationCheckpointConfig(mode="selective", early_stop=False)
    )
    model = SimpleNamespace(
        named_segments=lambda: [
            SimpleNamespace(is_block=True, fqn="layers.0", module=block, group="layers")
        ]
    )
    with patch.dict(os.environ, {"GIFT_CUTE_NORM_SAC_STATS_ONLY": str(int(enabled))}):
        apply_compile(model, CompileConfig(enable=True))
    return block


def profile(fn, path):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        with torch.profiler.record_function(
            "CuTe norm compiled SAC forward and backward"
        ):
            fn()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    data = path.read_bytes()
    events = json.loads(data)["traceEvents"]
    work = sorted(
        (
            event
            for event in events
            if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        ),
        key=lambda event: event["ts"],
    )
    streams = {event["args"]["stream"] for event in work}
    if len(streams) != 1:
        raise RuntimeError(f"Expected one CUDA stream, saw {streams}")
    ops = Counter(
        event["name"]
        for event in events
        if event.get("cat") == "cpu_op"
        and event["name"].startswith("torchtitan::cute_gated_norm_")
    )
    return {
        "trace": str(path),
        "sha256": digest(data),
        "op_counts": dict(ops),
        "elapsed_device_us": work[-1]["ts"] + work[-1]["dur"] - work[0]["ts"],
        "device_gap_us": sum(
            max(0.0, right["ts"] - left["ts"] - left["dur"])
            for left, right in zip(work, work[1:])
        ),
        "kernels": [
            dict(
                name=event["name"],
                grid=event["args"]["grid"],
                block=event["args"]["block"],
                duration_us=event["dur"],
                shared_bytes=event["args"].get("shared memory"),
                registers_per_thread=event["args"].get("registers per thread"),
            )
            for event in work
            if event["cat"] == "kernel"
        ],
        "copies_and_memset_us": sum(
            event["dur"] for event in work if event["cat"] != "kernel"
        ),
    }


def replay_case(role, record, fallback, output, timing, samples):
    from torchtitan.models.common.norms.cute_gated_norm.op import (
        _forward,
        _forward_residual,
        _forward_stats,
    )

    cpu = record["tensors"]
    residual = role == "residual"
    param_names = ["x", "weight", "up", "down"] + (["residual"] if residual else [])
    leaves = tuple(to_cuda(cpu[name]).requires_grad_(True) for name in param_names)
    tangent = to_cuda(cpu.get("dy", fallback))
    eps, gating_weight = record["eps"], record["gating_weight"]
    baseline = compile_block(eps, gating_weight, enabled=False)
    candidate = compile_block(eps, gating_weight, enabled=True)

    def make_call(block, args):
        def forward_backward():
            y = block(*args)
            grads = torch.autograd.grad(y, args, tangent)
            return y, *grads

        return forward_backward

    baseline_ref = make_call(baseline, leaves)()
    candidate_ref = make_call(candidate, leaves)()
    checks = {
        name: bits_equal(actual, expected)
        for name, actual, expected in zip(
            ["y", *param_names], candidate_ref, baseline_ref, strict=True
        )
    }
    checks["native_y"] = bits_equal(baseline_ref[0], to_cuda(cpu["y"]))
    del baseline_ref, candidate_ref
    with torch.no_grad():
        for rows in (8192, 65, 0):
            # Native tensors supply both the full shape and the partial/empty shards.
            x, weight, up, down = leaves[:4]
            full_op = _forward_residual if residual else _forward
            full_args = ()
            if rows != 8192:
                x = x.flatten(0, 1)[:rows]
            if residual:
                res = leaves[4] if rows == 8192 else leaves[4].flatten(0, 1)[:rows]
                full_args = (res,)
            ordinary = full_op(x, weight, up, down, eps, gating_weight, *full_args)[1:]
            stats = _forward_stats(x, weight, up, down, eps)
            checks[f"saved_outputs_{rows}_rows"] = {
                name: bits_equal(actual, expected)
                for name, actual, expected in zip(
                    ["rho", "P", "w_f", "gd_bf", "gd_t", "gu_t"],
                    stats,
                    ordinary,
                    strict=True,
                )
            }
    if not all(
        value if isinstance(value, bool) else all(value.values())
        for value in checks.values()
    ):
        raise RuntimeError(f"{role} byte parity failed: {checks}")

    traces = {}
    timings = []
    for label, block in (
        ("baseline", baseline),
        ("stats", candidate),
        ("stats", candidate),
        ("baseline", baseline),
    ):
        fn = make_call(block, leaves)
        for _ in range(5):
            fn()
        row = {
            "mode": label,
            **timing.elapsed_calls(fn, samples),
            "memory_full_scope": timing.memory(fn),
        }
        y = block(*leaves)
        row["memory_backward_scope"] = timing.memory(
            lambda y=y: torch.autograd.grad(y, leaves, tangent)
        )
        del y
        if label not in traces:
            traces[label] = profile(fn, output / f"{role}-{label}.json")
        row["cuda_graph_replay_ms"] = timing.device_graph(
            lambda graph_leaves: make_call(block, graph_leaves), leaves, samples
        )
        timings.append(row)

    ordinary_name = (
        "torchtitan::cute_gated_norm_residual_fwd"
        if residual
        else "torchtitan::cute_gated_norm_fwd"
    )
    backward_name = "torchtitan::cute_gated_norm_bwd"
    stats_name = "torchtitan::cute_gated_norm_fwd_stats"
    expected_ops = {
        "baseline": {ordinary_name: 2, backward_name: 1},
        "stats": {ordinary_name: 1, stats_name: 1, backward_name: 1},
    }
    for mode, expected in expected_ops.items():
        if traces[mode]["op_counts"] != expected:
            raise RuntimeError(
                f"{role} {mode} did not execute the expected SAC operations: {traces[mode]['op_counts']}"
            )

    live_result = None
    if not residual:
        # Squaring y requires its activation during VJP. Check the generated graph
        # and compare complete compiled SAC outputs and gradients with the flag off.
        live_base = compile_block(eps, gating_weight, enabled=False, live=True)
        live_candidate = compile_block(eps, gating_weight, enabled=True, live=True)
        live_fn = make_call(live_candidate, leaves)
        live_expected = make_call(live_base, leaves)()
        live_actual = live_fn()
        if not all(
            bits_equal(actual, expected)
            for actual, expected in zip(live_actual, live_expected, strict=True)
        ):
            raise RuntimeError("Live activation control differs from its baseline")
        del live_actual, live_expected
        live_result = profile(live_fn, output / "plain-live.json")
        if live_result["op_counts"] != expected_ops["baseline"]:
            raise RuntimeError(
                f"Expected the live replay to retain y: {live_result['op_counts']}"
            )

    return {
        "checks": checks,
        "inputs": record["layouts"],
        "tangent_matches_norm_input": "dy" in cpu,
        "gating_weight": gating_weight,
        "eps": eps,
        "timings": timings,
        "traces": traces,
        "live_y_control": live_result,
    }


def run(args):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise ValueError("Expose exactly one Hopper GPU")
    args.output.mkdir(parents=True)
    for env, name in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("CUTE_DSL_CACHE_DIR", "cute"),
        ("TVM_FFI_CACHE_DIR", "tvm_ffi"),
    ):
        path = args.output.resolve() / name
        path.mkdir()
        os.environ[env] = str(path)
    torch.use_deterministic_algorithms(True)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True, mmap=True)
    if fixture["torch_revision"] != torch.version.git_version:
        raise RuntimeError("Capture and replay use different PyTorch revisions")
    timing = timing_helpers()
    result = {
        "fixture": str(args.fixture),
        "fixture_sha256": digest(args.fixture.read_bytes()),
        "torch": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(),
        "samples": args.samples,
        "cases": {},
    }
    for role, record in fixture["records"].items():
        print(f"Testing {role} on native B2/S4096/H4096", flush=True)
        result["cases"][role] = replay_case(
            role,
            record,
            fixture["first_tangent"]["dy"],
            args.output,
            timing,
            args.samples,
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    wrappers = []
    for path in sorted(args.output.glob("inductor/[a-z0-9][a-z0-9]/*.py")):
        data = path.read_bytes()
        if b"cute_gated_norm_" not in data:
            continue
        names, sites = generated_calls(path, data)
        if names:
            wrappers.append(
                {
                    "file": str(path.relative_to(args.output)),
                    "sha256": digest(data),
                    "names": names,
                    "sites": sites,
                }
            )
    result["generated_graph_wrappers"] = wrappers
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=25)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
