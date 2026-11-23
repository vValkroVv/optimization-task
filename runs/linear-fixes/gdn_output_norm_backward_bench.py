"""Compare GDN FP8 norm backward against the original FLA VJP on one H100."""

import argparse
from collections import Counter
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
from statistics import median
import subprocess

import torch

from gdn_numerics import compare
from gdn_output_fp8_component import (
    load_inputs,
    paired_call_timing,
    verify_10plus10_config,
)


NORM_ARGS = (1e-6, "sigmoid", True, 2.0)
COMPILE_OPTIONS = {"triton.cudagraphs": False}
GRAD_NAMES = ("dx", "dgate", "dnorm_weight")
VJP_OP = "torchtitan::fla_vjp"
DIRECT_OP = "torchtitan::gdn_output_norm_backward"
DIRECT_KERNEL = "_gdn_output_norm_backward_kernel"
WEIGHT_KERNEL = "_gdn_output_norm_weight_grad_kernel"
FLA_FORWARD = "layer_norm_gated_fwd_kernel"
FLA_BACKWARD = "layer_norm_gated_bwd_kernel"
FP8_FORWARD = "_gdn_output_norm_quantize_128_kernel"


def make_norm_backward(inputs, dgrad, norm_args, *, direct, compiled):
    from torchtitan.models.common.attention.linear.fla_custom_ops import fla_vjp
    from torchtitan.models.common.attention.linear.gdn_output_norm_backward import (
        gdn_output_norm_backward,
    )

    eps, activation, zero_centered, multiplier = norm_args

    def backward(x, gate, weight, dgrad):
        if direct:
            return gdn_output_norm_backward(x, gate, weight, dgrad, *norm_args)
        _, seq_len, heads, width = x.shape
        scaled = (dgrad[:, :seq_len, : heads * width] * multiplier).reshape(x.shape)
        return tuple(
            fla_vjp(
                "gated_rmsnorm",
                [x, gate, weight],
                [scaled],
                [],
                [eps, zero_centered],
                activation,
            )
        )

    if compiled:
        backward = torch.compile(backward, fullgraph=True, options=COMPILE_OPTIONS)
    return partial(backward, *inputs, dgrad)


def make_projection_call(inputs, projection, cotangent, ac_config, ac_mode, direct):
    from torchtitan.distributed.activation_checkpoint import _apply_ac_to_block
    from torchtitan.models.common.attention.linear.gdn_output_fp8 import (
        gdn_output_norm_fp8,
    )

    class ProjectionTail(torch.nn.Module):
        def __init__(self, projection):
            super().__init__()
            self.o_proj = projection
            self.direct_backward = direct

        def forward(self, x, gate, weight):
            activation = gdn_output_norm_fp8(
                x, gate, weight, *NORM_ARGS, direct_backward=self.direct_backward
            )
            return self.o_proj(activation)

    leaves = tuple(value.detach().requires_grad_() for value in inputs)
    tail = _apply_ac_to_block(
        ProjectionTail(projection),
        ac_config,
        mode=ac_mode,
        base_fqn="layers.0.attention",
    )
    forward = torch.compile(tail, fullgraph=True, options=COMPILE_OPTIONS)

    def call():
        output = forward(*leaves)
        grads = torch.autograd.grad(output, (*leaves, projection.weight), cotangent)
        return output.detach(), *grads

    return call


def prepare_projection_dgrad(inputs, projection, cotangent, helpers):
    from torchtitan.models.common.attention.linear.gdn_output_fp8 import (
        gdn_output_norm_fp8,
    )

    leaves = tuple(value.detach().requires_grad_() for value in inputs)
    baseline = gdn_output_norm_fp8(*leaves, *NORM_ARGS, direct_backward=False)
    candidate = gdn_output_norm_fp8(*leaves, *NORM_ARGS, direct_backward=True)
    saved = {}
    for name, activation in (("baseline", baseline), ("candidate", candidate)):
        values = activation.grad_fn.saved_tensors
        if len(values) != len(leaves) or any(
            actual.data_ptr() != expected.data_ptr()
            for actual, expected in zip(values, leaves, strict=True)
        ):
            raise RuntimeError(f"{name} norm saves tensors beyond the crop inputs")
        saved[name] = [tensor_contract(value) for value in values]
    payload = helpers.comparisons(
        ("fp8_bytes", "fp32_scales"),
        (baseline._data, baseline._scale),
        (candidate._data, candidate._scale),
    )
    helpers.assert_bits(payload)

    (dgrad,) = torch.autograd.grad(projection(baseline), baseline, cotangent)
    if dgrad.shape != (2, 4096, 4096) or dgrad.dtype != torch.bfloat16:
        raise RuntimeError("Expected the crop's actual BF16 FP8 projection dgrad")
    return dgrad, payload, saved


def tensor_contract(value):
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "bytes": value.numel() * value.element_size(),
    }


def oracle(inputs, dgrad, norm_args, *, round_scaled=True):
    """Use higher precision arithmetic between the existing dtype boundaries."""
    x, gate, weight = inputs
    eps, activation, zero_centered, multiplier = norm_args
    _, seq_len, heads, width = x.shape
    scaled = dgrad[:, :seq_len, : heads * width].float() * multiplier
    if round_scaled:
        scaled = scaled.to(dgrad.dtype)
    # The legacy VJP's autograd engine converts the incoming cotangent to
    # the FLA output dtype after the multiplier's own dtype conversion.
    scaled = scaled.to(x.dtype).reshape(-1, width)
    weight_eff = weight.float() + 1.0 if zero_centered else weight.float()
    weight_eff = weight_eff.double()
    dx = torch.empty_like(x).reshape(-1, width)
    dgate = torch.empty_like(gate).reshape(-1, width)
    dw = torch.zeros_like(weight, dtype=torch.float64)
    # Bound temporary storage while retaining a higher precision weight sum.
    for first in range(0, scaled.shape[0], 8192):
        rows = slice(first, first + 8192)
        features = x.reshape_as(dx)[rows].double()
        gates = gate.reshape_as(dgate)[rows].double()
        grad = scaled[rows].double()
        rstd = (features.square().mean(-1, keepdim=True) + eps).rsqrt()
        normalized = features * rstd
        sigmoid = gates.sigmoid()
        derivative = sigmoid * (1.0 - sigmoid)
        if activation in ("silu", "swish"):
            derivative = sigmoid + gates * derivative
            sigmoid = gates * sigmoid
        grad_gate = grad * (normalized * weight_eff) * derivative
        grad_norm = grad * sigmoid
        dw += (grad_norm * normalized).sum(0)
        weighted = grad_norm * weight_eff
        dx[rows] = (
            weighted - normalized * (weighted * normalized).mean(-1, True)
        ) * rstd
        dgate[rows] = grad_gate.to(x.dtype).to(gate.dtype)
    return dx.reshape_as(x), dgate.reshape_as(gate), dw.to(weight.dtype)


def check_gradients(calls, inputs, dgrad, norm_args, *, rounding_probe=False):
    baseline, repeat, candidate = (
        calls["baseline"](),
        calls["baseline"](),
        calls["candidate"](),
    )
    report = {
        "baseline_repeat": compare(GRAD_NAMES, baseline, repeat),
        "candidate_vs_baseline": compare(GRAD_NAMES, baseline, candidate),
    }
    del repeat
    reference = oracle(inputs, dgrad, norm_args)
    report["baseline_vs_oracle"] = compare(GRAD_NAMES, reference, baseline)
    report["candidate_vs_oracle"] = compare(GRAD_NAMES, reference, candidate)
    if rounding_probe:
        unrounded = oracle(inputs, dgrad, norm_args, round_scaled=False)
        probe = compare(GRAD_NAMES, reference, unrounded)
        if not any(row["different_elements"] for row in probe.values()):
            raise RuntimeError(
                "The rounding probe did not distinguish the FP16-to-BF16 casts"
            )
        report["oracle_with_omitted_gradient_rounding"] = probe
        report["candidate_vs_omitted_gradient_rounding"] = compare(
            GRAD_NAMES, unrounded, candidate
        )
        for name in GRAD_NAMES:
            rounded = report["candidate_vs_oracle"][name]["relative_l2"]
            omitted = report["candidate_vs_omitted_gradient_rounding"][name][
                "relative_l2"
            ]
            if rounded >= omitted:
                raise RuntimeError(
                    f"Candidate {name} does not exhibit the FP16-to-BF16 casts"
                )
    return report


def edge_inputs(seed):
    rng = torch.Generator(device="cuda").manual_seed(seed + 17)
    specs = (
        (
            "padded_v128",
            (2, 67, 3, 128),
            torch.bfloat16,
            torch.bfloat16,
            "sigmoid",
            True,
            2.0,
        ),
        (
            "mixed_v96",
            (2, 7, 3, 96),
            torch.float32,
            torch.float16,
            "swish",
            False,
            1.7,
        ),
        (
            "strided_odd_v127",
            (2, 19, 3, 127),
            torch.bfloat16,
            torch.float32,
            "sigmoid",
            True,
            0.8,
        ),
        (
            "wide_v513",
            (1, 17, 2, 513),
            torch.bfloat16,
            torch.float16,
            "silu",
            True,
            2.0,
        ),
    )
    for name, shape, param_dtype, grad_dtype, activation, centered, multiplier in specs:
        batch, seq, heads, width = shape
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=rng)
        gate = torch.randn(shape, device="cuda", dtype=param_dtype, generator=rng)
        weight = (
            torch.randn((width,), device="cuda", dtype=param_dtype, generator=rng)
            * 0.25
        )
        features = ((heads * width + 127) // 128) * 128
        storage = torch.randn(
            (batch, 128, 2 * features), device="cuda", dtype=grad_dtype, generator=rng
        )
        dgrad = (
            storage[..., ::2]
            if name == "strided_odd_v127"
            else storage[..., :features].contiguous()
        )
        yield name, (x, gate, weight), dgrad, (1e-6, activation, centered, multiplier)


def trace_call(call, path, helpers):
    trace = helpers.trace(call, path)
    events = json.loads(path.read_text())["traceEvents"]
    trace["custom_op_counts"] = Counter(
        event["name"]
        for event in events
        if event.get("cat") == "cpu_op" and event["name"] in (VJP_OP, DIRECT_OP)
    )
    return trace


def require_activation(traces, *, projection=False):
    errors = []
    for name, trace in traces.items():
        candidate = name == "candidate"
        kernels = [row[0] for row in trace["kernels"]]
        expected = (
            (DIRECT_KERNEL, WEIGHT_KERNEL) if candidate else (FLA_FORWARD, FLA_BACKWARD)
        )
        absent = (
            (FLA_FORWARD, FLA_BACKWARD, "fla_vjp_mul")
            if candidate
            else (DIRECT_KERNEL, WEIGHT_KERNEL)
        )
        op, other_op = (DIRECT_OP, VJP_OP) if candidate else (VJP_OP, DIRECT_OP)
        counts = trace["custom_op_counts"]
        if not counts[op] or counts[other_op]:
            errors.append(f"{name}: unexpected custom op counts {counts}")
        for marker in expected:
            if sum(marker in kernel for kernel in kernels) != 1:
                errors.append(f"{name}: expected one {marker} launch")
        for marker in absent:
            if any(marker in kernel for kernel in kernels):
                errors.append(f"{name}: unexpected {marker} launch")
        if not candidate:
            before_recompute = [
                kernels[index - 1]
                for index, kernel in enumerate(kernels)
                if index and FLA_FORWARD in kernel
            ]
            trace["legacy_multiplier_kernel"] = next(
                (kernel for kernel in before_recompute if "mul" in kernel), None
            )
            if trace["legacy_multiplier_kernel"] is None:
                errors.append(f"{name}: no separate multiplier before FLA replay")
        if projection and sum(FP8_FORWARD in kernel for kernel in kernels) != 2:
            errors.append(
                f"{name}: expected forward and SAC replay of the FP8 producer"
            )
    return errors


def measure(calls, output, helpers, args, *, projection=False):
    traces = {
        name: trace_call(call, output / f"{name}.json", helpers)
        for name, call in calls.items()
    }
    errors = require_activation(traces, projection=projection)
    result = {"traces": traces, "activation_errors": errors}
    for name, trace in traces.items():
        marker = FLA_BACKWARD if name == "baseline" else DIRECT_KERNEL
        launch = next((row for row in trace["kernels"] if marker in row[0]), None)
        if launch is not None:
            trace["fp32_weight_partial_bytes"] = launch[1][0] * 128 * 4
    if not args.check_only and not errors:
        result["timing_us"] = paired_call_timing(calls, args.samples)
        result["paired_time_change_pct"] = {}
        for key in result["timing_us"]["baseline"]:
            baseline, candidate, control = (
                result["timing_us"][variant][key]["samples"]
                for variant in ("baseline", "candidate", "control")
            )
            deltas = {
                label: [
                    100.0 * (value / ref - 1.0)
                    for ref, value in zip(baseline, series, strict=True)
                ]
                for label, series in (
                    ("candidate_vs_baseline", candidate),
                    ("control_vs_baseline", control),
                )
            }
            result["paired_time_change_pct"][key] = {
                label: {"samples": values, "median": median(values)}
                for label, values in deltas.items()
            }
        memory = {name: helpers.memory(call) for name, call in calls.items()}
        result["memory"] = memory
        result["peak_memory_change_pct"] = {
            name: 100.0 * (memory["candidate"][name] / memory["baseline"][name] - 1.0)
            for name in ("peak_allocated_bytes", "peak_reserved_bytes")
        }
    return result


def run(args):
    import triton
    import yaml
    from fla.modules import fused_norm_gate
    from torchtitan.config import ActivationCheckpointConfig
    from torchtitan.distributed.activation_checkpoint import resolve_ac_mode
    from torchtitan.models.common.attention.linear import fla_custom_ops, gdn_output_fp8
    from torchtitan.models.common.attention.linear import gdn_output_norm_backward
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

    fixture, receipt = helpers.fixture_on_device(args.fixture)
    inputs, projection, cotangent = load_inputs(fixture, args.seed)
    geometry = verify_10plus10_config(
        args.resolved_config, fixture, inputs, projection, NORM_ARGS, helpers.digest
    )
    del fixture
    resolved = yaml.safe_load(args.resolved_config.read_text())
    ac = resolved["activation_checkpoint"]
    ac_config = ActivationCheckpointConfig(**ac)
    ac_mode = resolve_ac_mode("layers.0", ac_config)
    if ac_mode != "selective":
        raise ValueError(
            f"Expected selective AC for the dense crop layer, got {ac_mode}"
        )
    dgrad, payload, saved = prepare_projection_dgrad(
        inputs, projection, cotangent, helpers
    )

    result = {
        "fixture": str(args.fixture.resolve()),
        "fixture_receipt": receipt,
        "fixture_adaptation": "Native QKV/conv and hidden, replayed FLA scan, seeded gate/output FP8 weights, captured dV as output cotangent; native FP8 projection dgrad feeds isolated VJP",
        "crop_geometry": geometry,
        "crop_inputs": [tensor_contract(value) for value in inputs],
        "crop_projection_dgrad": tensor_contract(dgrad),
        "crop_projection_cotangent": tensor_contract(cotangent),
        "crop_local_tokens": inputs[0].shape[0] * inputs[0].shape[1],
        "crop_normalized_rows": inputs[0].numel() // inputs[0].shape[-1],
        "activation_checkpoint": {"layer0_mode": ac_mode, **ac},
        "forward_payload": payload,
        "norm_saved_tensors": saved,
        "oracle_precision": "FP64 norm arithmetic after the FP32 multiplier, rounding to dgrad dtype then FLA output dtype; dgate rounds to x dtype before gate dtype",
        "seed": args.seed,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": torch.cuda.get_device_name(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "gpu_multiprocessors": torch.cuda.get_device_properties(
            0
        ).multi_processor_count,
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
                    fla_custom_ops.__file__,
                    fused_norm_gate.__file__,
                    gdn_output_fp8.__file__,
                    gdn_output_norm_backward.__file__,
                    cuda_gemm.__file__,
                ),
            )
        },
        "compile": {"fullgraph": True, "options": COMPILE_OPTIONS},
        "measurement": {
            "samples_per_variant": 0 if args.check_only else args.samples,
            "order_even": ["baseline", "candidate", "control"],
            "order_odd": ["control", "candidate", "baseline"],
            "control": "Second execution of baseline at the same inputs",
            "warmup_calls_per_variant": 10,
            "event_span_us": "Single-stream CUDA elapsed span, may include host submission gaps",
            "host_call_us": "Python/autograd/dispatch/allocator and CUDA submission until return",
            "host_wait_us": "End-event record and blocking host wait",
            "wall_us": "Synchronized component call including submission and wait",
            "profile": "Separate warmed CPU/CUDA trace; kernel, copies/memset and idle gaps in us on one stream. No distributed communication",
            "memory": "Allocated/reserved peaks after cache flush and stats reset, including returned tensors until synchronization",
        },
        "errors": [],
    }
    receipt_path = output / "result.json"

    def save():
        receipt_path.write_text(json.dumps(result, indent=2) + "\n")

    variants = {"baseline": False, "candidate": True}
    for name, values, incoming, norm_args in edge_inputs(args.seed):
        calls = {
            key: make_norm_backward(
                values, incoming, norm_args, direct=direct, compiled=False
            )
            for key, direct in variants.items()
        }
        report = check_gradients(
            calls, values, incoming, norm_args, rounding_probe=name == "mixed_v96"
        )
        trace = trace_call(calls["candidate"], output / f"edge-{name}.json", helpers)
        result.setdefault("edges", {})[name] = {
            "inputs": [tensor_contract(value) for value in values],
            "incoming": tensor_contract(incoming),
            "norm_args": norm_args,
            "parity": report,
            "trace": trace,
        }
        result["errors"].extend(require_activation({"candidate": trace}))
        save()

    norm_calls = {
        name: make_norm_backward(inputs, dgrad, NORM_ARGS, direct=direct, compiled=True)
        for name, direct in variants.items()
    }
    result["isolated_backward"] = {
        "scope": "Compiled fullgraph norm VJP including legacy multiplier, replay, gradients and weight reduction",
        "parity": check_gradients(norm_calls, inputs, dgrad, NORM_ARGS),
    }
    save()
    norm_output = output / "isolated_backward"
    norm_output.mkdir()
    norm_measurement = measure(norm_calls, norm_output, helpers, args)
    result["isolated_backward"].update(norm_measurement)
    result["errors"].extend(norm_measurement["activation_errors"])
    save()

    projection_calls = {
        name: make_projection_call(
            inputs, projection, cotangent, ac_config, ac_mode, direct
        )
        for name, direct in variants.items()
    }
    names = ("output", *GRAD_NAMES, "dprojection_weight")
    baseline = projection_calls["baseline"]()
    repeat = projection_calls["baseline"]()
    baseline_repeat = compare(names, baseline, repeat)
    del repeat
    projection_parity = compare(names, baseline, projection_calls["candidate"]())
    del baseline
    if projection_parity["output"]["different_bytes"]:
        result["errors"].append(
            "Candidate changed the compiled projection forward output"
        )
    result["projection_chain"] = {
        "scope": "Inductor fullgraph forward and backward with crop layer-0 SAC, rowwise FP8 projection forward/dgrad/wgrad and norm gradients",
        "baseline_repeat": baseline_repeat,
        "candidate_vs_baseline": projection_parity,
    }
    save()
    projection_output = output / "projection_chain"
    projection_output.mkdir()
    projection_measurement = measure(
        projection_calls, projection_output, helpers, args, projection=True
    )
    result["projection_chain"].update(projection_measurement)
    result["errors"].extend(projection_measurement["activation_errors"])
    save()
    print(receipt_path, flush=True)
    if result["errors"]:
        raise RuntimeError("; ".join(result["errors"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run parity and activation without timing or memory checks",
    )
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least two")
    run(args)
