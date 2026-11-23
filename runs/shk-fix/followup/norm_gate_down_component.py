"""Compare B2 gate-down staging on captured inputs through compiled norm backward."""

import argparse
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
import platform
from unittest.mock import patch

import torch
import yaml

from norm_skip import (
    NATIVE_SHAPE,
    compile_block,
    digest,
    measure,
    mismatched_bytes,
    profile_backward,
    restore,
)


def b2_evidence(trace):
    events = json.loads(trace.read_bytes())["traceEvents"]
    (kernel,) = (
        event
        for event in events
        if event.get("cat") == "kernel" and "GnBwdDx_" in event["name"]
    )
    return {
        "name": kernel["name"],
        "stream": kernel["args"]["stream"],
        "grid": kernel["args"]["grid"],
        "block": kernel["args"]["block"],
        "shared_memory_bytes": kernel["args"]["shared memory"],
        "registers_per_thread": kernel["args"]["registers per thread"],
        "profiled_us": kernel["dur"],
    }


@torch.no_grad()
def edge_parity(select, leaves, dy, skip, record, fuse):
    from torchtitan.models.common.norms.cute_gated_norm.op import (
        _backward,
        _backward_skip,
        _forward,
    )

    x, weight, up, down = leaves
    results = {}
    for hidden in (512, NATIVE_SHAPE[-1]):
        x_slice, dy_slice, skip_slice = (
            value[0, :65, :hidden].contiguous() for value in (x, dy, skip)
        )
        params = (
            weight[:hidden].contiguous(),
            up[:, :hidden].contiguous(),
            down[:hidden].contiguous(),
        )
        _, *saved = _forward(x_slice, *params, record["eps"], record["gating_weight"])
        args = (
            dy_slice,
            x_slice,
            *saved,
            record["gating_weight"],
            *(tensor.dtype for tensor in params),
        )
        for skip_first in (True, False) if fuse else (None,):
            pair = []
            for arm in ("A", "B"):
                with select(arm):
                    if skip_first is None:
                        pair.append(_backward(*args))
                    else:
                        pair.append(_backward_skip(*args, skip_slice, skip_first))
            results[f"65_rows_{hidden}_hidden_skip_first_{skip_first}"] = (
                mismatched_bytes(("x", "weight", "up", "down"), *pair)
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip", action=argparse.BooleanOptionalAction, default=True)
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

    from torchtitan.models.common.norms.cute_gated_norm import backward as bwd

    original_backward = bwd.gated_norm_bwd

    def select(arm):
        return patch.object(
            bwd, "gated_norm_bwd", partial(original_backward, keep_gate_down=arm == "B")
        )

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
    leaves = tuple(restore(tensors[n], layouts[n]).requires_grad_() for n in names)
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
    # The capture has no separate skip tangent. Use the same rolled dy for both arms.
    skip = torch.roll(dy, shifts=1, dims=-1).contiguous()
    block = compile_block(record["eps"], record["gating_weight"], fuse=args.skip)

    def forward():
        return block(*leaves, residual)

    def backward(outputs):
        return torch.autograd.grad(outputs, leaves, (skip, dy))

    parity_outputs = []
    for arm in ("A", "B"):
        with select(arm):
            result = forward()
            parity_outputs.append((*result, *backward(result)))
    mismatches = mismatched_bytes(("residual", "norm", *names), *parity_outputs)
    del result, parity_outputs
    edges = edge_parity(select, leaves, dy, skip, record, args.skip)
    if any(mismatches.values()) or any(any(case.values()) for case in edges.values()):
        raise RuntimeError(f"Bitwise parity failed {mismatches} {edges}")

    torch._dynamo.config.error_on_recompile = True
    for arm in ("A", "B"):
        with select(arm):
            for _ in range(args.warmup):
                backward(forward())
                torch.cuda.synchronize()
    measurements = []
    for repeat in range(args.repeats):
        order = ("A", "B") if repeat % 2 == 0 else ("B", "A")
        for arm in order:
            with select(arm):
                result = measure(forward, backward, args.samples)
            measurements.append({"arm": arm, "repeat": repeat, **result})
            print(f"{arm} repeat {repeat + 1} {result['medians']}", flush=True)

    activation = {}
    for arm, trace in traces.items():
        with select(arm):
            profile = profile_backward(forward, backward, trace, fuse=args.skip)
        activation[arm] = {**profile, "b2": b2_evidence(trace)}

    hidden = NATIVE_SHAPE[-1]
    budget = bwd.resolve_num_sms(leaves[0].device)
    chunks, groups = bwd.pick_bwd_split(hidden, budget, bwd.B2_MAX_HC)
    stages = bwd.B2_SKIP_STAGES if args.skip else bwd.B2_STAGES
    subtiles = hidden // chunks // bwd.BLK_H
    expected_shared_delta = (subtiles - stages) * bwd.BLK_H * bwd.RANK * 2
    baseline_b2, candidate_b2 = (activation[arm]["b2"] for arm in ("A", "B"))
    if (
        any(activation[arm]["b2"]["grid"] != [chunks * groups, 1, 1] for arm in traces)
        or baseline_b2["block"] != candidate_b2["block"]
        or candidate_b2["shared_memory_bytes"] - baseline_b2["shared_memory_bytes"]
        != expected_shared_delta
    ):
        raise RuntimeError(f"Unexpected B2 specialization {activation}")

    with args.output.open("x") as handle:
        json.dump(
            {
                "hostname": platform.node(),
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "torch_revision": torch.version.git_version,
                "cuda": torch.version.cuda,
                "cutlass_dsl": version("nvidia-cutlass-dsl"),
                "kernel_source_sha256": digest(Path(bwd.__file__)),
                "fixture_sha256": digest(args.fixture),
                "resolved_config_sha256": digest(args.resolved_config),
                "capture_site": {
                    name: fixture[name] for name in ("rank", "update", "microbatch")
                },
                "shape": NATIVE_SHAPE,
                "dy_source": dy_source,
                "skip_source": f"one-column roll of {dy_source} dy",
                "fused_skip_both_arms": args.skip,
                "deterministic": deterministic,
                "sm_budget": budget,
                "b2_stages": stages,
                "warmup": args.warmup,
                "mismatched_bytes": mismatches,
                "edge_mismatched_bytes": edges,
                "profiled_activation": activation,
                "unprofiled_measurements": measurements,
            },
            handle,
            indent=2,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
