"""Scale the compiled GDN output-norm backward comparison beyond the 10+10 crop."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
from statistics import median
import subprocess

import torch

import gdn_output_norm_backward_bench as bench
from gdn_numerics import compare
from gdn_output_fp8_component import (
    load_inputs,
    paired_call_timing,
    verify_10plus10_config,
)


SHAPES = {
    "b1-s4096-h32-d128": (1, 4096, 32, 128),
    "b2-s8192-h32-d128": (2, 8192, 32, 128),
    "b2-s4096-h64-d128": (2, 4096, 64, 128),
    "b2-s4096-h32-d96": (2, 4096, 32, 96),
}
VARIANTS = {"baseline": False, "candidate": True}


def load_crop(path, config, seed, helpers):
    fixture, receipt = helpers.fixture_on_device(path)
    inputs, projection, cotangent = load_inputs(fixture, seed)
    geometry = verify_10plus10_config(
        config, fixture, inputs, projection, bench.NORM_ARGS, helpers.digest
    )
    return (
        inputs,
        projection,
        cotangent,
        {"path": str(path.resolve()), **receipt},
        geometry,
    )


@torch.no_grad()
def adapt_crop(crop, shape, seed):
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    (x, gate, weight), projection, cotangent, receipt, geometry = crop
    batch, sequence, heads, width = shape
    source = {"rank0": receipt}
    if batch < x.shape[0]:
        x, gate, cotangent = (value[:batch].clone() for value in (x, gate, cotangent))
        derivation = "Keep local batch row 0 from the rank-0 capture."
    elif sequence > x.shape[1]:
        x, gate, cotangent = (
            torch.cat((value, value.flip(0)), dim=1) for value in (x, gate, cotangent)
        )
        derivation = "Join the two distinct rank-0 batch sequences as [0|1, 1|0] along the token axis."
    elif heads > x.shape[2]:
        x, gate = (value.repeat(1, 1, 2, 1) for value in (x, gate))
        derivation = (
            "Duplicate the replayed rank-0 head bank along the value-head axis."
        )
    else:
        x, gate = (value[..., :width].contiguous() for value in (x, gate))
        weight = weight[:width].clone()
        derivation = "Keep the first 96 channels per replayed rank-0 value head."

    in_features = heads * width
    if in_features != projection.in_features:
        config = FP8Linear.Config(
            in_features=in_features, out_features=geometry["model_dim"], bias=False
        )
        projection = FP8Linear(config).to(device=x.device, dtype=x.dtype)
        rng = torch.Generator(device=x.device).manual_seed(seed + 2)
        projection.weight.normal_(std=in_features**-0.5, generator=rng)
        weight_origin = f"Rebuilt FP8 o_proj with in_features={in_features}; generator seed {seed + 2}."
    else:
        weight_origin = (
            "Reused the verified crop's seeded FP8 o_proj with matching dimensions."
        )
    if x.shape != shape or gate.shape != shape or weight.shape != (width,):
        raise RuntimeError("Unexpected adapted norm geometry")
    if cotangent.shape != (batch, sequence, projection.out_features):
        raise RuntimeError("Output cotangent does not match the adapted FP8 projection")
    return (x, gate, weight), projection, cotangent, source, derivation, weight_origin


def projection_dgrad(inputs, projection, cotangent, helpers):
    from torchtitan.models.common.attention.linear.gdn_output_fp8 import (
        gdn_output_norm_fp8,
    )

    leaves = tuple(value.detach().requires_grad_() for value in inputs)
    activations = {
        name: gdn_output_norm_fp8(*leaves, *bench.NORM_ARGS, direct_backward=direct)
        for name, direct in VARIANTS.items()
    }
    saved = {}
    for name, activation in activations.items():
        saved_tensors = activation.grad_fn.saved_tensors
        if len(saved_tensors) != 3 or any(
            value.data_ptr() != leaf.data_ptr()
            for value, leaf in zip(saved_tensors, leaves, strict=True)
        ):
            raise RuntimeError(
                f"{name} did not save only the three adapted norm inputs"
            )
        saved[name] = [bench.tensor_contract(value) for value in saved_tensors]
    payload = helpers.comparisons(
        ("fp8_bytes", "fp32_scales"),
        *((activation._data, activation._scale) for activation in activations.values()),
    )
    helpers.assert_bits(payload)
    baseline = activations["baseline"]
    (dgrad,) = torch.autograd.grad(projection(baseline), baseline, cotangent)
    if dgrad.shape != baseline.shape or dgrad.dtype != inputs[0].dtype:
        raise RuntimeError("Unexpected FP8 projection dgrad geometry or dtype")
    return dgrad, payload, saved


def pair_deltas(timing):
    result = {}
    for key, baseline in timing["baseline"].items():
        result[key] = {}
        for variant in ("candidate", "control"):
            values = [
                100.0 * (value / ref - 1.0)
                for ref, value in zip(
                    baseline["samples"], timing[variant][key]["samples"], strict=True
                )
            ]
            result[key][f"{variant}_vs_baseline"] = {
                "samples": values,
                "median": median(values),
            }
    return result


def measure(calls, output, helpers, args, *, width, projection=False):
    output.mkdir()
    traces = {
        name: bench.trace_call(call, output / f"{name}.json", helpers)
        for name, call in calls.items()
    }
    errors = bench.require_activation(traces)
    for name, trace in traces.items():
        marker = bench.FLA_BACKWARD if name == "baseline" else bench.DIRECT_KERNEL
        launches = [row for row in trace["kernels"] if marker in row[0]]
        if len(launches) == 1:
            trace["fp32_weight_partial_bytes"] = launches[0][1][0] * width * 4
        if not projection:
            continue
        forward_markers = (
            (bench.FP8_FORWARD,)
            if width == 128
            else ("_gdn_output_rstd_kernel", "_gdn_output_norm_quantize_blocks_kernel")
        )
        trace["fp8_forward_launches"] = {
            marker: sum(marker in row[0] for row in trace["kernels"])
            for marker in forward_markers
        }
        if any(count != 2 for count in trace["fp8_forward_launches"].values()):
            errors.append(
                f"{name}: expected each FP8 producer kernel in forward and SAC replay"
            )
    result = {"traces": traces, "activation_errors": errors}
    if errors or args.check_only:
        return result
    timing = paired_call_timing(calls, args.samples)
    memory = {name: helpers.memory(call) for name, call in calls.items()}
    result.update(
        timing_us=timing,
        paired_time_change_pct=pair_deltas(timing),
        memory=memory,
        peak_memory_change_pct={
            metric: 100.0
            * (memory["candidate"][metric] / memory["baseline"][metric] - 1.0)
            for metric in ("peak_allocated_bytes", "peak_reserved_bytes")
        },
    )
    return result


def run(args):
    import triton
    import yaml
    from fla.modules import fused_norm_gate
    from torchtitan.config import ActivationCheckpointConfig
    from torchtitan.distributed.activation_checkpoint import resolve_ac_mode
    from torchtitan.models.common.attention.linear import (
        fla_custom_ops,
        gdn_output_fp8,
        gdn_output_norm_backward,
    )
    from torchtitan.ops.float8 import cuda_gemm

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one assigned H100 GPU")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("Set FLA_DISABLE_BACKEND_DISPATCH=1 for native FLA")
    output = args.output.resolve()
    output.mkdir(parents=True)
    spec = spec_from_file_location("gdn_component_helpers", args.helpers)
    helpers = module_from_spec(spec)
    spec.loader.exec_module(helpers)
    torch.manual_seed(args.seed)
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

    shape = SHAPES[args.shape]
    crop = load_crop(args.fixture, args.resolved_config, args.seed, helpers)
    geometry = crop[-1]
    if crop[3]["rank"] != 0:
        raise ValueError("Expected the original rank-0 production capture")
    inputs, projection, cotangent, source, derivation, weight_origin = adapt_crop(
        crop, shape, args.seed
    )
    del crop
    resolved = yaml.safe_load(args.resolved_config.read_text())
    ac = resolved["activation_checkpoint"]
    ac_config = ActivationCheckpointConfig(**ac)
    ac_mode = resolve_ac_mode("layers.0", ac_config)
    if ac_mode != "selective":
        raise ValueError(f"Expected selective AC for crop layer 0, got {ac_mode}")
    dgrad, payload, saved = projection_dgrad(inputs, projection, cotangent, helpers)

    batch, sequence, heads, width = shape
    gpu = torch.cuda.get_device_properties(0)
    result = {
        "shape_name": args.shape,
        "input_shape": shape,
        "local_tokens": batch * sequence,
        "normalized_rows": batch * sequence * heads,
        "fixture_sources": source,
        "original_crop_geometry": geometry,
        "fixture_adaptation": derivation,
        "input_generation": "Native QKV/conv and hidden, replayed FLA scan, seeded FP8 gate weight, captured dV as output cotangent; isolated VJP uses native FP8 o_proj dgrad.",
        "projection_weight_origin": weight_origin,
        "inputs": [bench.tensor_contract(value) for value in inputs],
        "projection_weight": bench.tensor_contract(projection.weight),
        "projection_dgrad": bench.tensor_contract(dgrad),
        "projection_cotangent": bench.tensor_contract(cotangent),
        "norm_saved_tensors": saved,
        "forward_payload": payload,
        "norm_args": bench.NORM_ARGS,
        "activation_checkpoint": {"layer0_mode": ac_mode, **ac},
        "compile": {"fullgraph": True, "options": bench.COMPILE_OPTIONS},
        "seed": args.seed,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": gpu.name,
        "gpu_memory_bytes": gpu.total_memory,
        "gpu_multiprocessors": gpu.multi_processor_count,
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "source_sha256": {
            str(path): helpers.digest(path)
            for path in map(
                Path,
                (
                    args.helpers,
                    __file__,
                    bench.__file__,
                    fla_custom_ops.__file__,
                    fused_norm_gate.__file__,
                    gdn_output_fp8.__file__,
                    gdn_output_norm_backward.__file__,
                    cuda_gemm.__file__,
                ),
            )
        },
        "measurement": {
            "samples_per_variant": 0 if args.check_only else args.samples,
            "warmup_calls_per_variant": 10,
            "order_even": ["baseline", "candidate", "control"],
            "order_odd": ["control", "candidate", "baseline"],
            "control": "Second execution of the baseline at the same inputs.",
            "event_span_us": "Single-stream CUDA event elapsed span, including possible host submission gaps.",
            "host_call_us": "Python/autograd/allocator and launch submission until the call returns.",
            "host_wait_us": "End-event record plus host blocking wait.",
            "wall_us": "Call, end-event record, and blocking wait together.",
            "profile": "Separate warmed CPU/CUDA trace with kernel, idle gap, and copy/memset times on one stream. No distributed communication.",
            "memory": "Allocated/reserved peaks after an allocator cache flush and stats reset; includes output buffers until synchronization.",
        },
        "errors": [],
    }
    report = output / "result.json"

    def save():
        report.write_text(json.dumps(result, indent=2) + "\n")

    save()
    norm_calls = {
        name: bench.make_norm_backward(
            inputs, dgrad, bench.NORM_ARGS, direct=direct, compiled=True
        )
        for name, direct in VARIANTS.items()
    }
    result["isolated_backward"] = {
        "parity": bench.check_gradients(norm_calls, inputs, dgrad, bench.NORM_ARGS)
    }
    save()
    norm_measurement = measure(
        norm_calls, output / "isolated_backward", helpers, args, width=width
    )
    result["isolated_backward"].update(norm_measurement)
    result["errors"].extend(norm_measurement["activation_errors"])
    save()

    projection_calls = {
        name: bench.make_projection_call(
            inputs, projection, cotangent, ac_config, ac_mode, direct
        )
        for name, direct in VARIANTS.items()
    }
    names = ("output", *bench.GRAD_NAMES, "dprojection_weight")
    baseline = projection_calls["baseline"]()
    repeat = compare(names, baseline, projection_calls["baseline"]())
    parity = compare(names, baseline, projection_calls["candidate"]())
    del baseline
    if parity["output"]["different_bytes"]:
        result["errors"].append("Candidate changed the projection forward output.")
    result["projection_chain"] = {
        "baseline_repeat": repeat,
        "candidate_vs_baseline": parity,
    }
    save()
    projection_measurement = measure(
        projection_calls,
        output / "projection_chain",
        helpers,
        args,
        width=width,
        projection=True,
    )
    result["projection_chain"].update(projection_measurement)
    result["errors"].extend(projection_measurement["activation_errors"])
    save()
    print(report, flush=True)
    if result["errors"]:
        raise RuntimeError("; ".join(result["errors"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", required=True, choices=SHAPES)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run parity and activation checks only",
    )
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least two")
    run(args)
