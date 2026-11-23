"""Measure GDN gated RMSNorm through the native rowwise FP8 projection."""

import argparse
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter

import torch
from torch.utils.checkpoint import checkpoint

from gdn_direct_component import load_scan_inputs

SPECIALIZED_REVISION = "fe5950b56de5b2cec7ff4cd063f7338c98200c45"
SPECIALIZED_SHA256 = "4e8568f22239f7bab4b9535e6f3b16d20c6cfb4ed98f6b61fe6fb47db758a785"
ORIGINAL_A_CONFIG_SHA256 = (
    "5e389eebaf3f09d0c9d40232afee8f0ef52ce40530ce2a3dc3510f009ba2d6aa"
)


def load_specialized_source(path, output, digest):
    source = path.read_bytes()
    if digest(path) != SPECIALIZED_SHA256:
        raise ValueError("Historical specialized GDN FP8 source hash mismatch")
    op_name = b"torchtitan::gdn_output_norm_quantize"
    if source.count(op_name) != 1:
        raise ValueError("Expected one historical custom op registration")

    # Both implementations run in one process. Isolate only the dispatcher name.
    overlay = output / "gdn_output_fp8_specialized.py"
    overlay.write_bytes(source.replace(op_name, op_name + b"_specialized"))
    spec = spec_from_file_location("gdn_output_fp8_specialized", overlay)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, {
        "revision": SPECIALIZED_REVISION,
        "original_source": str(path.resolve()),
        "original_sha256": SPECIALIZED_SHA256,
        "overlay_source": str(overlay),
        "overlay_sha256": digest(overlay),
        "overlay_change": "dispatcher registration name only",
    }


def verify_10plus10_config(path, fixture, inputs, projection, norm_args, digest):
    import yaml

    if digest(path) != ORIGINAL_A_CONFIG_SHA256:
        raise ValueError("Expected the resolved Original A 10+10 configuration")
    resolved = yaml.safe_load(path.read_text())
    model = resolved["model"]["config"]
    gdn = model["blocks"]["gdn_dense"]["attention"]
    training = resolved["training"]
    batch_seq = (training["local_batch_size"], training["seq_len"])
    norm = gdn["o_norm"]
    expected_norm = (
        norm["eps"],
        norm["activation"],
        norm["zero_centered"],
        gdn["o_norm_output_multiplier"],
    )
    value_shape = (*batch_seq, gdn["num_v_heads"], gdn["head_v_dim"])
    key_shape = (*batch_seq, gdn["num_k_heads"], gdn["head_k_dim"])
    if (
        inputs[0].shape != value_shape
        or fixture["grad_q"].shape != key_shape
        or fixture["conv_weight"].shape[-1] != gdn["conv_size"]
        or projection.weight.shape != (model["dim"], value_shape[-2] * value_shape[-1])
        or norm_args != expected_norm
    ):
        raise ValueError("Replay geometry or norm settings differ from the 10+10 run")
    return {
        "run": "original-a-vs-cute-001-a2",
        "resolved_config": str(path.resolve()),
        "resolved_config_sha256": ORIGINAL_A_CONFIG_SHA256,
        "value_shape": list(value_shape),
        "key_shape": list(key_shape),
        "model_dim": model["dim"],
        "conv_size": gdn["conv_size"],
        "norm_args": expected_norm,
    }


def load_inputs(fixture, seed):
    from torchtitan.models.common.attention.linear.fla_custom_ops import gdn_chunk_op
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    @torch.no_grad()
    def make_projection(in_features, out_features, generator):
        config = FP8Linear.Config(
            in_features=in_features, out_features=out_features, bias=False
        )
        projection = FP8Linear(config).to(device="cuda", dtype=torch.bfloat16)
        projection.weight.normal_(std=in_features**-0.5, generator=generator)
        return projection

    with torch.no_grad():
        scan_inputs, metadata = load_scan_inputs(fixture, seed)
        x = gdn_chunk_op(*scan_inputs, metadata, False)
        hidden = fixture["x"]
        num_heads, head_dim = x.shape[-2:]
        rng = torch.Generator(device=x.device).manual_seed(seed + 1)
        gate_proj = make_projection(hidden.shape[-1], num_heads * head_dim, rng)
        gate = gate_proj(hidden).reshape(x.shape)
        weight = torch.zeros(head_dim, dtype=x.dtype, device=x.device)
        projection = make_projection(num_heads * head_dim, hidden.shape[-1], rng)
        cotangent = fixture["grad_v"].flatten(2).contiguous()
        if cotangent.shape != (*x.shape[:2], projection.out_features):
            raise ValueError(
                "The captured cotangent does not fit the crop output projection"
            )
    return (x, gate, weight), projection, cotangent


def paired_call_timing(calls, samples):
    """Measure synchronized calls on one stream, including autograd submission."""
    calls = {**calls, "control": calls["baseline"]}
    order = tuple(calls)
    for name in order:
        for _ in range(10):
            calls[name]()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    readings = {
        name: {
            key: []
            for key in ("event_span_us", "host_call_us", "host_wait_us", "wall_us")
        }
        for name in order
    }
    for iteration in range(samples):
        for name in order if iteration % 2 == 0 else reversed(order):
            start.record()
            t0 = perf_counter()
            calls[name]()
            t1 = perf_counter()
            end.record()
            end.synchronize()
            t2 = perf_counter()
            reading = readings[name]
            reading["event_span_us"].append(start.elapsed_time(end) * 1000)
            reading["host_call_us"].append((t1 - t0) * 1e6)
            reading["host_wait_us"].append((t2 - t1) * 1e6)
            reading["wall_us"].append((t2 - t0) * 1e6)
    return {
        name: {
            key: {"samples": values, "median": median(values)}
            for key, values in row.items()
        }
        for name, row in readings.items()
    }


def run(args):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one assigned H100 GPU")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("Set FLA_DISABLE_BACKEND_DISPATCH=1 for native FLA")

    output = args.output.resolve()
    output.mkdir(parents=True)
    spec = spec_from_file_location("gdn_component_helpers", args.helpers)
    helpers = module_from_spec(spec)
    spec.loader.exec_module(helpers)

    import triton
    from torchtitan.distributed.activation_checkpoint import _sac_context_fn
    from torchtitan.models.common.attention.linear import fla_custom_ops, gdn_output_fp8
    from torchtitan.ops.float8 import cuda_gemm
    from torchtitan.ops.float8._cudacherry import quantize_vec128
    from torchtitan.ops.float8.fp8_utils import pad_tensor_if_needed

    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    fixture, receipt = helpers.fixture_on_device(args.fixture)
    inputs, projection, cotangent = load_inputs(fixture, args.seed)
    norm_args = (1e-6, "sigmoid", True, 2.0)
    compile_options = {"triton.cudagraphs": False}
    specialized = specialized_receipt = training_receipt = None
    if args.specialized_source is not None:
        specialized, specialized_receipt = load_specialized_source(
            args.specialized_source, output, helpers.digest
        )
        training_receipt = verify_10plus10_config(
            args.resolved_config, fixture, inputs, projection, norm_args, helpers.digest
        )
    del fixture

    def norm(x, gate, weight):
        if specialized is not None:
            return specialized.GDNOutputNormFP8.apply(x, gate, weight, *norm_args[:-1])
        return (
            fla_custom_ops.gated_rmsnorm_op(x, gate, weight, *norm_args[:-1])
            * norm_args[-1]
        ).flatten(2)

    @torch.compile(fullgraph=True, options=compile_options)
    def baseline_payload(x, gate, weight):
        if specialized is not None:
            return specialized.gdn_output_norm_quantize(
                x, gate, weight, *norm_args[:-1]
            )
        data, scale, _, _ = quantize_vec128(
            pad_tensor_if_needed(norm(x, gate, weight)), pack=False
        )
        return data, scale

    @torch.compile(fullgraph=True, options=compile_options)
    def fused_payload(x, gate, weight):
        return gdn_output_fp8.gdn_output_norm_quantize(x, gate, weight, *norm_args)

    def make_projection_call(fused):
        leaves = tuple(value.detach().requires_grad_() for value in inputs)

        def tail(x, gate, weight):
            activation = (
                gdn_output_fp8.gdn_output_norm_fp8(x, gate, weight, *norm_args)
                if fused
                else norm(x, gate, weight)
            )
            return projection(activation)

        @torch.compile(fullgraph=True, options=compile_options)
        def forward(*values):
            return checkpoint(
                tail,
                *values,
                use_reentrant=False,
                context_fn=partial(_sac_context_fn, (), ()),
                early_stop=False,
            )

        def call():
            out = forward(*leaves)
            grads = torch.autograd.grad(out, (*leaves, projection.weight), cotangent)
            return out.detach(), *grads

        return call

    payload_calls = {
        "baseline": partial(baseline_payload, *inputs),
        "candidate": partial(fused_payload, *inputs),
    }
    projection_calls = {
        "baseline": make_projection_call(False),
        "candidate": make_projection_call(True),
    }
    payload_check = helpers.comparisons(
        ("fp8_bytes", "scales"), *(call() for call in payload_calls.values())
    )
    helpers.assert_bits(payload_check)
    projection_check = helpers.comparisons(
        ("output", "dx", "dgate", "dnorm_weight", "dprojection_weight"),
        *(call() for call in projection_calls.values()),
    )
    helpers.assert_bits(projection_check)

    result = {
        "baseline": "historical specialized fusion"
        if specialized is not None
        else "unfused native norm and FP8 packing",
        "specialized_source": specialized_receipt,
        "10plus10_geometry": training_receipt,
        "fixture": str(args.fixture.resolve()),
        "fixture_receipt": receipt,
        "adaptation": "FLA scan replay from captured native QKV and real-data hidden, seeded gate/output weights, captured dV as projection cotangent",
        "seed": args.seed,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": torch.cuda.get_device_name(),
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "input_shape": list(inputs[0].shape),
        "input_dtype": str(inputs[0].dtype),
        "projection_weight_shape": list(projection.weight.shape),
        "norm_args": norm_args,
        "source_sha256": {
            str(path): helpers.digest(path)
            for path in map(
                Path,
                (
                    args.helpers,
                    __file__,
                    fla_custom_ops.__file__,
                    gdn_output_fp8.__file__,
                    cuda_gemm.__file__,
                ),
            )
        },
        "forward_payload": {
            "parity": payload_check,
            "scope": "compiled norm, multiplier and FP8 packing, no projection GEMM",
            "graph_replays_per_sample": 20,
            "timing_us": helpers.pair_timing(payload_calls, args.samples),
            "memory": {
                name: helpers.memory(call) for name, call in payload_calls.items()
            },
            "traces": {
                name: helpers.trace(call, output / f"payload-{name}.json")
                for name, call in payload_calls.items()
            },
        },
        "projection_forward_backward": {
            "parity": projection_check,
            "scope": "Inductor fullgraph with selective AC, native FP8 GEMM forward, dgrad and wgrad, FLA norm gradients",
            "timing_us": paired_call_timing(projection_calls, args.samples),
            "memory": {
                name: helpers.memory(call) for name, call in projection_calls.items()
            },
            "traces": {
                name: helpers.trace(call, output / f"projection-{name}.json")
                for name, call in projection_calls.items()
            },
        },
    }
    summary = output / "result.json"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    for section in ("forward_payload", "projection_forward_backward"):
        traces = result[section]["traces"]
        marker = "_gdn_output_norm_quantize_128_kernel"
        if not any(marker in row[0] for row in traces["candidate"]["kernels"]):
            raise RuntimeError(f"Missing candidate activation in {section}: {summary}")
        if any(marker in row[0] for row in traces["baseline"]["kernels"]):
            raise RuntimeError(
                f"Unexpected baseline activation in {section}: {summary}"
            )
        if specialized is not None and not any(
            "_gdn_output_norm_quantize_kernel" in row[0]
            for row in traces["baseline"]["kernels"]
        ):
            raise RuntimeError(
                f"Missing specialized activation in {section}: {summary}"
            )
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--specialized-source", type=Path)
    parser.add_argument("--resolved-config", type=Path)
    args = parser.parse_args()
    if args.specialized_source and not args.resolved_config:
        parser.error("--specialized-source requires --resolved-config")
    run(args)
