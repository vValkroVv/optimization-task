"""Check the cast boundaries of the native and fused GDN FP8 tails on SM90."""

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import torch
from gdn_output_fp8_component import load_inputs, verify_10plus10_config
from torchtitan.models.common.attention.linear import fla_custom_ops, gdn_output_fp8
from torchtitan.ops.float8._cudacherry import quantize_vec128
from torchtitan.ops.float8.cuda_gemm import FP8Linear
from torchtitan.ops.float8.fp8_utils import pad_tensor_if_needed


def compare_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    return {
        "dtype": str(actual.dtype),
        "elements": actual.numel(),
        "bitwise_mismatches": int(
            (
                actual.contiguous().view(torch.uint8)
                != expected.contiguous().view(torch.uint8)
            )
            .reshape(actual.numel(), actual.element_size())
            .any(dim=1)
            .sum()
        ),
    }


def exact(actual, expected):
    result = compare_bits(actual, expected)
    assert not result["bitwise_mismatches"], result
    return result


def check_tail(inputs, projection, cotangent, norm_args, check_omitted_round=False):
    from fla.modules.fused_norm_gate import layer_norm_gated_fwd

    x, gate, weight = inputs
    eps, activation, zero_centered, multiplier = norm_args
    with torch.no_grad():
        norm = fla_custom_ops.gated_rmsnorm_op(x, gate, weight, *norm_args[:-1])
        fp32_norm, *_ = layer_norm_gated_fwd(
            x.reshape(-1, x.shape[-1]),
            gate.reshape(-1, x.shape[-1]),
            weight,
            None,
            activation=activation,
            eps=eps,
            out_dtype=torch.float32,
            is_rms_norm=True,
            zero_centered=zero_centered,
        )
        fp32_norm = fp32_norm.reshape_as(x)
        scaled = norm * multiplier
        fp32_product = norm.float() * multiplier
        padded = pad_tensor_if_needed(scaled.flatten(2))
        native = quantize_vec128(padded, pack=True)
        candidate = gdn_output_fp8.gdn_output_norm_fp8(*inputs, *norm_args)
        max_abs = padded.float().reshape(padded.shape[0] * padded.shape[1], -1, 128)
        scale = (max_abs.abs().amax(-1) / 448.0).clamp_min(1e-30)
        power_two_scale = (scale.view(torch.int32) + 0x7FFFFF) & 0x7F800000
        power_two_scale = power_two_scale.view(torch.float32).transpose(0, 1)
        result = {
            "shape": list(x.shape),
            "gate_dtype": str(gate.dtype),
            "multiplier": multiplier,
            "native_norm_store_matches_bf16_round": exact(norm, fp32_norm.bfloat16()),
            "native_multiplier_matches_bf16_round": exact(
                scaled, fp32_product.bfloat16()
            ),
            "candidate_fp8_bytes": exact(candidate._data, native._data),
            "candidate_native_fp32_scale": exact(candidate._scale, native._scale),
            "candidate_recomputed_fp32_scale": exact(candidate._scale, power_two_scale),
        }
        if check_omitted_round:
            omitted_round = ((fp32_norm * multiplier).bfloat16()).flatten(2)
            omitted_payload = quantize_vec128(
                pad_tensor_if_needed(omitted_round), pack=True
            )
            differences = compare_bits(omitted_payload._data, native._data)
            assert differences["bitwise_mismatches"] > 0
            result["fp8_bytes_if_norm_bf16_round_omitted"] = differences

    vjp = fla_custom_ops.fla_vjp

    def run_backward(fused):
        norm_cotangents = []
        projection_cotangents = []

        def inspect_vjp(kernel, leaves, grad_outputs, *args):
            assert kernel == "gated_rmsnorm"
            norm_cotangents.append(grad_outputs[0])
            return vjp(kernel, leaves, grad_outputs, *args)

        leaves = tuple(t.detach().requires_grad_(True) for t in inputs)
        if fused:
            activation_out = gdn_output_fp8.gdn_output_norm_fp8(*leaves, *norm_args)
        else:
            norm = fla_custom_ops.gated_rmsnorm_op(*leaves, *norm_args[:-1])
            activation_out = (norm * multiplier).flatten(2)
            activation_out.register_hook(projection_cotangents.append)

        with (
            patch.object(fla_custom_ops, "fla_vjp", inspect_vjp),
            patch.object(gdn_output_fp8, "fla_vjp", inspect_vjp),
        ):
            out = projection(activation_out)
            grads = torch.autograd.grad(out, (*leaves, projection.weight), cotangent)
        (norm_cotangent,) = norm_cotangents
        return (out.detach(), *grads), norm_cotangent, projection_cotangents

    native, native_dnorm, (dprojection_in,) = run_backward(False)
    candidate, fused_dnorm, _ = run_backward(True)
    unrounded_dnorm = dprojection_in.reshape_as(x).float() * multiplier
    result["candidate_input_to_fla_vjp"] = exact(fused_dnorm, native_dnorm)
    result["fla_vjp_input_matches_bf16_round"] = exact(
        fused_dnorm, unrounded_dnorm.bfloat16()
    )
    if check_omitted_round:
        differences = compare_bits(fused_dnorm.float(), unrounded_dnorm)
        assert differences["bitwise_mismatches"] > 0
        result["fla_vjp_input_if_bf16_round_omitted"] = differences
    result["projection"] = {
        name: exact(actual, ref)
        for name, actual, ref in zip(
            ("out", "dx", "dgate", "dnorm_weight", "dprojection_weight"),
            candidate,
            native,
            strict=True,
        )
    }
    return result


def main():
    from fla.modules import fused_norm_gate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (
        9,
        0,
    )
    spec = spec_from_file_location("gdn_component_helpers", args.helpers)
    helpers = module_from_spec(spec)
    spec.loader.exec_module(helpers)
    fixture, receipt = helpers.fixture_on_device(args.fixture)
    production = load_inputs(fixture, seed=2048)
    norm_args = (1e-6, "sigmoid", True, 2.0)
    geometry = verify_10plus10_config(
        args.resolved_config,
        fixture,
        production[0],
        production[1],
        norm_args,
        helpers.digest,
    )
    del fixture

    rng = torch.Generator(device="cuda").manual_seed(310)
    shape = (2, 67, 3, 96)
    x = torch.randn(shape, device="cuda", generator=rng).bfloat16()
    gate = torch.randn(shape, device="cuda", generator=rng) / 7
    weight = torch.randn(shape[-1], device="cuda", generator=rng).bfloat16()
    x[0, 0].zero_()
    projection = FP8Linear.Config(in_features=288, out_features=143, bias=False).build()
    projection = projection.to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        projection.weight.normal_(std=288**-0.5, generator=rng)
    cotangent = torch.randn((*shape[:2], 143), device="cuda", generator=rng).bfloat16()
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "fixture": receipt,
        "10plus10_geometry": geometry,
        "source_sha256": {
            path: helpers.digest(Path(path))
            for path in (gdn_output_fp8.__file__, fused_norm_gate.__file__, __file__)
        },
        "native_crop": check_tail(*production, norm_args),
        "mixed_gate_padded": check_tail(
            (x, gate, weight),
            projection,
            cotangent,
            (1e-6, "sigmoid", True, 1.375),
            check_omitted_round=True,
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
