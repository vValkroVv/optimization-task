"""Compare residual fusion with live and dead CuTe norm outputs during SAC replay."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
from types import SimpleNamespace

import torch
from torch import nn
import yaml

import measure
import norm


WARMUP = 100
SAMPLES = 256


def build(fixture_path, scenario, arm):
    from torchtitan.config import ActivationCheckpointConfig, CompileConfig
    from torchtitan.distributed.activation_checkpoint import _apply_op_sac
    from torchtitan.distributed.compile import apply_compile
    from torchtitan.models.common.norms.cute_gated_norm import gated_norm

    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True, mmap=True)
    if fixture["torch_revision"] != torch.version.git_version:
        raise RuntimeError("CuTe capture and replay use different PyTorch revisions")
    record = fixture["records"]["residual"]
    tensors, layouts = record["tensors"], record["layouts"]
    if "dy" in tensors:
        dy, dy_layout = tensors["dy"], layouts["dy"]
        tangent_source = "residual"
    else:
        dy = fixture["first_tangent"]["dy"]
        dy_layout = fixture["first_tangent"]["layout"]
        tangent_source = "first_tangent"
    if tensors["x"].shape != norm._NATIVE_SHAPE or dy.shape != norm._NATIVE_SHAPE:
        raise ValueError("Expected the captured B2/S4096/H4096 norm and tangent")
    if any(
        tensor.dtype != torch.bfloat16
        for tensor in (tensors["x"], tensors["residual"], dy)
    ):
        raise ValueError("Expected BF16 norm input, residual and tangent")

    names = ("x", "weight", "up", "down", "residual")
    leaves = {
        name: norm._to_cuda(tensors[name], layouts[name]).requires_grad_()
        for name in names
    }
    tangent = norm._to_cuda(dy, dy_layout)
    input_layouts = {name: layouts[name] for name in names} | {"dy": dy_layout}
    fuse_residual = arm == "B"
    live = scenario == "live"
    eps, gating_weight = record["eps"], record["gating_weight"]
    # Native live sites feed another plain CuTe gated norm. It consumes the
    # BF16 result across an op boundary and needs the same tensor in its VJP.
    consumer = fixture["records"]["plain"]
    consumer_eps, consumer_gating_weight = consumer["eps"], consumer["gating_weight"]
    if live:
        for name in ("weight", "up", "down"):
            key = f"consumer_{name}"
            input_layouts[key] = consumer["layouts"][name]
            leaves[key] = norm._to_cuda(
                consumer["tensors"][name], input_layouts[key]
            ).requires_grad_()

    class Norm(nn.Module):
        def forward(self, x, weight, up, down, residual, *consumer_weights):
            y = gated_norm(
                x,
                weight,
                up,
                down,
                eps=eps,
                gating_weight=gating_weight,
                residual=residual if fuse_residual else None,
            )
            if not fuse_residual:
                y = residual + y
            if live:
                return gated_norm(
                    y,
                    *consumer_weights,
                    eps=consumer_eps,
                    gating_weight=consumer_gating_weight,
                )
            return y

    block = _apply_op_sac(
        Norm(),
        ActivationCheckpointConfig(
            mode="selective", selective_save=[], early_stop=False
        ),
    )
    segment = SimpleNamespace(
        is_block=True, fqn="layers.0", module=block, group="layers"
    )
    model = SimpleNamespace(named_segments=lambda: [segment])
    apply_compile(
        model, CompileConfig(enable=True, cudagraph_exclude_blocks=[segment.fqn])
    )
    inputs = tuple(leaves.values())

    def forward():
        return block(*inputs)

    return (
        forward,
        leaves,
        tangent,
        {
            "role": "residual",
            "capture_site": {
                name: fixture[name] for name in ("rank", "update", "microbatch")
            },
            "tangent_source": tangent_source,
            "input_layouts": input_layouts,
            "eps": eps,
            "gating_weight": gating_weight,
            "consumer": {
                "role": "plain",
                "eps": consumer_eps,
                "gating_weight": consumer_gating_weight,
            }
            if live
            else None,
        },
    )


def verify_ops(scenario, arm, forward, backward):
    ordinary = norm._RESIDUAL_OP if arm == "B" else norm._NORM_OP
    expected_forward = Counter((ordinary,))
    expected_backward = Counter((norm._STATS_OP, norm._BACKWARD_OP))
    if scenario == "live":
        expected_forward[norm._NORM_OP] += 1
        expected_backward.update((ordinary, norm._BACKWARD_OP))

    def norm_ops(counts):
        return {
            name: count
            for name, count in counts.items()
            if name.startswith("torchtitan::cute_gated_norm_")
        }

    return norm._check_phases(
        scenario,
        arm,
        norm_ops(forward),
        norm_ops(backward),
        expected_forward,
        expected_backward,
    )


def verify_kernels(scenario, arm, forward, backward):
    def classify(counts):
        cute, pointwise, other = Counter(), {}, {}
        for name, count in counts.items():
            kernel = next(
                (kernel for kernel in norm._KERNELS if f"{kernel}_" in name), None
            )
            if kernel is not None:
                cute[kernel] += count
            elif "triton_poi" in name:
                pointwise[name] = count
            else:
                other[name] = count
        return cute, {"pointwise": pointwise, "other": other}

    forward_cute, forward_extra = classify(forward)
    backward_cute, backward_extra = classify(backward)
    expected_forward = Counter((*norm._PREP_KERNELS, norm._FULL_KERNEL))
    expected_backward = Counter(
        (*norm._PREP_KERNELS, norm._STATS_KERNEL, *norm._BACKWARD_KERNELS)
    )
    if scenario == "live":
        expected_forward.update((*norm._PREP_KERNELS, norm._FULL_KERNEL))
        expected_backward.update(
            (*norm._PREP_KERNELS, norm._FULL_KERNEL, *norm._BACKWARD_KERNELS)
        )
    routes = norm._check_phases(
        scenario,
        arm,
        forward_cute,
        backward_cute,
        expected_forward,
        expected_backward,
    )
    return {**routes, "forward_extra": forward_extra, "backward_extra": backward_extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("dead", "live"), required=True)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    trace_path = args.output.parent / f"{args.output.stem}.trace.json"
    if args.output.exists() or (args.profile and trace_path.exists()):
        parser.error(
            "provide a fresh JSON output path and a fresh trace path when profiling"
        )

    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("expose exactly one Hopper GPU per process")
    torch.cuda.set_device(0)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("unexpected CUDA Graph capture")
    config_bytes = args.resolved_config.read_bytes()
    deterministic = yaml.safe_load(config_bytes)["debug"]["deterministic"]
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        torch.utils.deterministic.fill_uninitialized_memory = False
    inductor_config.triton.cudagraphs = False
    switches = {
        "GIFT_CUTE_NORM_FUSE_RESIDUAL": str(int(args.arm == "B")),
        "GIFT_CUTE_NORM_SAC_STATS_ONLY": "1",
        "GIFT_FP8_QUANTIZE_WEIGHT_TAIL": "0",
    }
    os.environ.update(switches)

    props = torch.cuda.get_device_properties(0)
    label = f"{args.scenario}/{args.arm}"
    print(f"{label} {props.name}, starting compiled setup", flush=True)
    forward, leaves, tangent, capture = build(args.fixture, args.scenario, args.arm)

    def warmup_call():
        y = forward()
        gradients = measure.differentiate(y, leaves, tangent)
        torch.cuda.synchronize()
        del y, gradients

    warmup_call()
    dynamo_config.error_on_recompile = True
    for _ in range(WARMUP):
        warmup_call()
    print(f"{label} completed {WARMUP} post-compile warmup calls", flush=True)
    before = torch.cuda.memory_stats()
    samples = measure.measure(forward, leaves, tangent, SAMPLES)
    after = torch.cuda.memory_stats()
    failures = {
        key: after[key] - before[key] for key in ("num_alloc_retries", "num_ooms")
    }
    if any(failures.values()):
        raise RuntimeError(f"allocator failure or retry in measured calls: {failures}")

    y = forward()
    gradients = measure.differentiate(y, leaves, tangent)
    signatures = {"y": measure.tensor_signature(y)} | {
        name: measure.tensor_signature(gradient)
        for name, gradient in zip(leaves, gradients, strict=True)
    }
    del y, gradients
    with args.fixture.open("rb") as handle:
        fixture_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    trace = None
    if args.profile:
        adapter = SimpleNamespace(verify_kernels=verify_kernels, verify_ops=verify_ops)
        profile_args = SimpleNamespace(
            case=args.scenario, arm=args.arm, output=args.output
        )
        trace = measure.profile(adapter, profile_args, forward, leaves, tangent)
    receipt = {
        "scenario": args.scenario,
        "arm": args.arm,
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": props.name,
        "torch_version": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "fixture": str(args.fixture),
        "fixture_sha256": fixture_hash,
        "capture": capture,
        "resolved_config": str(args.resolved_config),
        "resolved_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "deterministic_algorithms": deterministic,
        "sac": {"selective_save": [], "early_stop": False},
        "compile": {"backend": "inductor", "fullgraph": True, "cudagraphs": False},
        "switches": switches,
        "warmup_calls": WARMUP,
        "samples": samples,
        "allocator_failure_deltas": failures,
        "output_signatures": signatures,
        "trace": trace,
    }
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(f"{label} wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
