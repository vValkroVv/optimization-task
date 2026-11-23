"""Replay compiled CuTe pre-norm backward with matched native inputs and SAC."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import yaml


SWITCH = "GIFT_CUTE_NORM_BWD_SKIP"
NATIVE_SHAPE = (2, 4096, 4096)
BACKWARD_OP = "torchtitan::cute_gated_norm_bwd"


def restore(tensor, layout):
    if (tuple(tensor.shape), tensor.stride(), str(tensor.dtype)) != (
        tuple(layout["shape"]),
        tuple(layout["stride"]),
        layout["dtype"],
    ):
        raise ValueError(f"Capture tensor does not match layout {layout}")
    span = 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    prefix = layout["address_mod_256"] // tensor.element_size()
    storage = torch.empty(span + prefix, device="cuda", dtype=tensor.dtype)
    result = storage.as_strided(tensor.shape, tensor.stride(), storage_offset=prefix)
    if result.data_ptr() % 256 != layout["address_mod_256"]:
        raise RuntimeError(f"Could not restore capture alignment {layout}")
    return result.copy_(tensor)


def compile_block(eps, gating_weight, fuse):
    from torchtitan.config import ActivationCheckpointConfig, CompileConfig
    from torchtitan.distributed.activation_checkpoint import _apply_op_sac
    from torchtitan.distributed.compile import apply_compile
    from torchtitan.models.common.norms.cute_gated_norm import gated_norm

    class PreNorm(nn.Module):
        def forward(self, x, weight, up, down, residual):
            y = gated_norm(x, weight, up, down, eps=eps, gating_weight=gating_weight)
            # Independent outputs let both captured dy and a separate skip
            # tangent enter the same AOT backward without rounding either.
            return x + residual, y

    block = _apply_op_sac(
        PreNorm(),
        ActivationCheckpointConfig(
            mode="selective",
            selective_save=[],
            early_stop=False,
        ),
    )
    segment = SimpleNamespace(
        is_block=True, fqn="layers.0", module=block, group="layers"
    )
    model = SimpleNamespace(named_segments=lambda: [segment])
    with patch.dict(
        os.environ, {SWITCH: str(int(fuse)), "GIFT_CUTE_NORM_SAC_STATS_ONLY": "1"}
    ):
        apply_compile(
            model, CompileConfig(enable=True, cudagraph_exclude_blocks=[segment.fqn])
        )
    return block


def measure(forward, backward, samples):
    start, middle, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    for event in (start, middle, end):
        event.record()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_stats()
    records = []
    for _ in range(samples):
        started = perf_counter_ns()
        start.record()
        values = forward()
        middle.record()
        gradients = backward(values)
        end.record()
        submitted = perf_counter_ns()
        end.synchronize()
        drained = perf_counter_ns()
        records.append(
            {
                "forward_event_ms": start.elapsed_time(middle),
                "backward_event_ms": middle.elapsed_time(end),
                "integrated_wall_ms": (drained - started) / 1e6,
                "host_submit_ms": (submitted - started) / 1e6,
                "wall_drain_ms": (drained - submitted) / 1e6,
            }
        )
        del values, gradients
    after = torch.cuda.memory_stats()
    if any(after[name] != before[name] for name in ("num_alloc_retries", "num_ooms")):
        raise RuntimeError("Allocation retry or OOM during measurement")
    return {
        "samples": records,
        "medians": {
            key: median(sample[key] for sample in records) for key in records[0]
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def profile_backward(forward, backward, path, fuse):
    values = forward()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        gradients = backward(values)
        torch.cuda.synchronize()
    del values, gradients
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_bytes())["traceEvents"]
    ops = Counter(
        e["name"]
        for e in events
        if e.get("cat") == "cpu_op" and e["name"].startswith(BACKWARD_OP)
    )
    kernels = Counter(e["name"] for e in events if e.get("cat") == "kernel")
    if ops != {BACKWARD_OP + ("_skip" if fuse else ""): 1}:
        raise RuntimeError(f"Unexpected backward route {dict(ops)}")
    pointwise = sum(count for name, count in kernels.items() if "triton_poi" in name)
    if (
        sum(count for name, count in kernels.items() if "GnBwdDx_" in name) != 1
        or pointwise != int(not fuse)
        or any(e.get("cat") == "cpu_op" and e["name"] == "aten::add" for e in events)
    ):
        raise RuntimeError(f"Unexpected backward kernels or fallback {dict(kernels)}")
    work = sorted(
        (e for e in events if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}),
        key=lambda e: e["ts"],
    )
    tail = work[0]["ts"]
    busy = 0.0
    for event in work:
        end = event["ts"] + event["dur"]
        busy += max(0.0, end - max(tail, event["ts"]))
        tail = max(tail, end)
    span = tail - work[0]["ts"]
    return {
        "custom_ops": dict(ops),
        "kernels": dict(kernels),
        "kernel_sum_us": sum(e["dur"] for e in work if e["cat"] == "kernel"),
        "copy_memset_sum_us": sum(e["dur"] for e in work if e["cat"] != "kernel"),
        "gpu_span_us": span,
        "gpu_busy_union_us": busy,
        "gpu_gaps_us": max(0.0, span - busy),
        "streams": sorted({e["args"]["stream"] for e in work if "stream" in e["args"]}),
        "trace": str(path),
        "trace_sha256": digest(path),
    }


def mismatched_bytes(names, reference, candidate):
    counts = {}
    for name, left, right in zip(names, reference, candidate, strict=True):
        if (left.dtype, left.shape, left.stride()) != (
            right.dtype,
            right.shape,
            right.stride(),
        ):
            raise RuntimeError(f"Changed output metadata for {name}")
        differences = left.contiguous().view(torch.uint8) != right.contiguous().view(
            torch.uint8
        )
        counts[name] = int(torch.count_nonzero(differences))
    return counts


@torch.no_grad()
def verify_edge_layouts(leaves, dy, skip, eps, gating_weight):
    from torchtitan.models.common.norms.cute_gated_norm.op import (
        _backward,
        _backward_skip,
        _forward,
    )

    x, *parameters = leaves
    dtypes = tuple(p.dtype for p in parameters)
    names = ("x", "weight", "up", "down")
    checks = {}
    for rows in (0, 65):
        x_slice, dy_slice, skip_slice = (value[0, :rows] for value in (x, dy, skip))
        _, *saved = _forward(x_slice, *parameters, eps, gating_weight)
        args = (dy_slice, x_slice, *saved, gating_weight, *dtypes)
        reference_dx, *parameter_grads = _backward(*args)
        for skip_first in (True, False):
            values = (
                (skip_slice, reference_dx) if skip_first else (reference_dx, skip_slice)
            )
            reference = (torch.add(*values), *parameter_grads)
            candidate = _backward_skip(*args, skip_slice, skip_first)
            checks[f"{rows}_rows_skip_first_{skip_first}"] = mismatched_bytes(
                names,
                reference,
                candidate,
            )
        if rows:
            storage = torch.empty(
                skip_slice.numel() + 1, dtype=skip_slice.dtype, device=skip_slice.device
            )
            unaligned = storage[1:].view_as(skip_slice).copy_(skip_slice)
            assert unaligned.data_ptr() % 16 != 0
            reference = (reference_dx + skip_slice, *parameter_grads)
            candidate = _backward_skip(*args, unaligned, False)
            checks["65_rows_unaligned_skip"] = mismatched_bytes(
                names, reference, candidate
            )
    return checks


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    traces = {arm: args.output.with_suffix(f".{arm}.trace.json") for arm in ("A", "B")}
    if any(path.exists() for path in (args.output, *traces.values())):
        parser.error("Specify a fresh output path")
    if args.warmup < 1 or args.samples < 1 or args.repeats < 2:
        parser.error("Require positive warmup and samples and at least two repeats")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one Hopper GPU")
    torch.cuda.set_device(0)
    deterministic = yaml.safe_load(args.resolved_config.read_bytes())["debug"][
        "deterministic"
    ]
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        torch.utils.deterministic.fill_uninitialized_memory = False

    fixture = torch.load(args.fixture, map_location="cpu", mmap=True, weights_only=True)
    if fixture["torch_revision"] != torch.version.git_version:
        raise RuntimeError("CuTe capture and replay use different PyTorch revisions")
    record = fixture["records"]["plain"]
    tensors, layouts = record["tensors"], record["layouts"]
    names = ("x", "weight", "up", "down")
    leaves = tuple(
        restore(tensors[name], layouts[name]).requires_grad_() for name in names
    )
    if "dy" in tensors:
        dy = restore(tensors["dy"], layouts["dy"])
        dy_source = "plain"
    else:
        tangent = fixture["first_tangent"]
        dy = restore(tangent["dy"], tangent["layout"])
        dy_source = "first_tangent"
    residual_record = fixture["records"]["residual"]
    residual = restore(
        residual_record["tensors"]["residual"], residual_record["layouts"]["residual"]
    )
    if any(
        tensor.shape != NATIVE_SHAPE or tensor.dtype != torch.bfloat16
        for tensor in (leaves[0], dy, residual)
    ):
        raise ValueError("Expected native B2/S4096/H4096 BF16 capture tensors")
    # A separate model-shaped skip contribution, derived from the native
    # incoming gradient because the capture does not contain the skip edge.
    skip = torch.roll(dy, shifts=1, dims=-1).contiguous()
    blocks = {
        arm: compile_block(record["eps"], record["gating_weight"], fuse=arm == "B")
        for arm in ("A", "B")
    }

    def backward(values):
        return torch.autograd.grad(values, leaves, (skip, dy))

    def evaluate(arm):
        out = blocks[arm](*leaves, residual)
        return (*out, *backward(out))

    reference, candidate = evaluate("A"), evaluate("B")
    output_names = ("residual_output", "norm_output", *names)
    mismatches = mismatched_bytes(output_names, reference, candidate)
    del reference, candidate

    torch._dynamo.config.error_on_recompile = True
    for arm in blocks:
        for _ in range(args.warmup):
            evaluate(arm)
            torch.cuda.synchronize()
    repeated_measurements = []
    for repeat in range(args.repeats):
        order = ("A", "B") if repeat % 2 == 0 else ("B", "A")
        for arm in order:
            result = measure(
                lambda: blocks[arm](*leaves, residual), backward, args.samples
            )
            repeated_measurements.append({"arm": arm, "repeat": repeat, **result})
            print(f"{arm} repeat {repeat + 1} {result['medians']}", flush=True)
    activation = {
        arm: profile_backward(
            lambda: blocks[arm](*leaves, residual),
            backward,
            traces[arm],
            fuse=arm == "B",
        )
        for arm in blocks
    }
    edge_checks = verify_edge_layouts(
        leaves,
        dy,
        skip,
        record["eps"],
        record["gating_weight"],
    )
    with args.output.open("x") as handle:
        json.dump(
            {
                "hostname": platform.node(),
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "torch_revision": torch.version.git_version,
                "cuda": torch.version.cuda,
                "fixture_sha256": digest(args.fixture),
                "resolved_config_sha256": digest(args.resolved_config),
                "capture_site": {
                    name: fixture[name] for name in ("rank", "update", "microbatch")
                },
                "shape": list(NATIVE_SHAPE),
                "dy_source": dy_source,
                "skip_source": f"model-shaped one-column roll of native {dy_source} tangent",
                "deterministic": deterministic,
                "compile": {
                    "backend": "inductor",
                    "fullgraph": True,
                    "cudagraphs": False,
                },
                "sac": {"selective_save": [], "early_stop": False, "stats_only": True},
                "warmup": args.warmup,
                "mismatched_bytes": mismatches,
                "edge_layout_mismatched_bytes": edge_checks,
                "activation": activation,
                "measurement": repeated_measurements,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    if any(mismatches.values()) or any(
        any(case.values()) for case in edge_checks.values()
    ):
        raise RuntimeError(f"Parity failed {mismatches} {edge_checks}")


if __name__ == "__main__":
    main()
