"""Check the independent GDN candidates on Hopper using the training operators."""

import argparse
import json
from pathlib import Path
from statistics import median

import torch
from torch import nn

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import _apply_ac_to_block
from torchtitan.models.common.attention.linear.fla_custom_ops import (
    fla_vjp,
    gated_rmsnorm_op,
    qkv_conv1d_unpack_op,
)


def bf16_rand(*shape, std=1.0):
    return (torch.randn(shape, device="cuda") * std).to(torch.bfloat16)


def prepare(module, compiled_sac):
    if not compiled_sac:
        return module
    checkpointed = _apply_ac_to_block(
        module,
        ActivationCheckpointConfig(mode="selective", selective_save=[]),
        mode="selective",
    )
    return torch.compile(
        checkpointed, fullgraph=True, options={"triton.cudagraphs": False}
    )


def compare_grad(actual, expected):
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    norm = expected_f32.norm()
    # BF16 differences should be limited to final rounding changes. The small
    # RMS-relative floor covers near-zero values in different FP32 sum trees.
    rms = norm.item() / expected.numel() ** 0.5
    stats = {
        "relative_l2": (difference.norm() / norm).item(),
        "max_abs": difference.abs().max().item(),
        "unequal": torch.count_nonzero(actual != expected).item(),
        "elements": expected.numel(),
    }
    torch.testing.assert_close(
        actual, expected, rtol=2**-6, atol=rms * 2**-8, msg=str(stats)
    )
    assert stats["relative_l2"] < 2**-9, stats
    return stats


def graph_replay_us(operation):
    """Measure a warm-cache CUDA Graph with preallocated capture outputs."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = operation()
    for _ in range(4):
        graph.replay()

    samples_us = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0 / 20)
    del outputs
    return {"median_us": median(samples_us), "samples_us": samples_us}


def graph_comparison(scope, baseline, candidate):
    report = {
        "scope": scope,
        "method": "CUDA Graph, 5 event spans of 20 replays after warmup",
    }
    for name, operation in (("baseline", baseline), ("candidate", candidate)):
        try:
            report[name] = graph_replay_us(operation)
        except RuntimeError as exc:
            report[name] = {"unavailable": str(exc)}
    return report


class ShortConv(nn.Module):
    def __init__(self, num_heads, num_v_heads, fused):
        super().__init__()
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.fused = fused

    def forward(self, x, weight, lengths):
        return qkv_conv1d_unpack_op(
            x,
            weight,
            self.num_heads,
            self.num_v_heads,
            128,
            128,
            lengths,
            True,
            1e-6,
            self.fused,
        )


def short_conv_case(batch, seq_len, heads, value_heads, boundaries, compiled_sac):
    from torchtitan.models.common.attention.linear.gdn_short_conv import (
        gdn_short_conv_backward,
    )

    width = (heads * 2 + value_heads) * 128
    x = bf16_rand(batch, seq_len, width, std=0.25).requires_grad_()
    weight = bf16_rand(width, 1, 4, std=0.05).requires_grad_()
    lengths = (
        torch.tensor(boundaries, device="cuda", dtype=torch.int32)
        if boundaries
        else None
    )
    cotangents = (
        bf16_rand(batch, seq_len, heads, 128),
        bf16_rand(batch, seq_len, heads, 128),
        bf16_rand(batch, seq_len, value_heads, 128),
    )
    results = []
    for fused in (False, True):
        module = prepare(ShortConv(heads, value_heads, fused), compiled_sac)
        outputs = module(x, weight, lengths)
        grads = torch.autograd.grad(outputs, (x, weight), cotangents)
        results.append((tuple(t.detach() for t in outputs), grads))

    baseline, candidate = results
    for expected, actual in zip(baseline[0], candidate[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    report = {
        "batch": batch,
        "sequence_length": seq_len,
        "packed_boundaries": boundaries,
        "compiled_sac": compiled_sac,
        "forward_bitwise": True,
        **{
            name: compare_grad(actual, expected)
            for name, actual, expected in zip(
                ("dx", "dw"), candidate[1], baseline[1], strict=True
            )
        },
    }
    if (batch, seq_len, heads, value_heads) == (2, 4096, 16, 32):
        replay_inputs = [x.detach(), weight.detach()]
        report["timing"] = graph_comparison(
            "QKV short-conv backward, including baseline VJP replay and both dW reductions",
            lambda: fla_vjp(
                "qkv_conv1d",
                replay_inputs,
                list(cotangents),
                [None],
                [heads, value_heads, 128, 128, True, 1e-6],
                "",
            ),
            lambda: gdn_short_conv_backward(
                *replay_inputs, *cotangents, None, heads, value_heads, True, 1e-6
            ),
        )
    return report


def unfused_output_norm(x, gate, norm_weight):
    y = gated_rmsnorm_op(x, gate, norm_weight, 1e-6, "sigmoid", True) * 2.0
    return y.flatten(2)


class OutputProjection(nn.Module):
    def __init__(self, projection, fused):
        super().__init__()
        self.projection = projection
        self.fused = fused

    def forward(self, x, gate, norm_weight):
        from torchtitan.models.common.attention.linear.gdn_output_fp8 import (
            GDNOutputNormFP8,
        )

        y = (
            GDNOutputNormFP8.apply(x, gate, norm_weight, 1e-6, "sigmoid", True)
            if self.fused
            else unfused_output_norm(x, gate, norm_weight)
        )
        return self.projection(y)


def norm_case(batch, seq_len, heads, compiled_sac):
    from torchtitan.models.common.attention.linear.gdn_output_fp8 import (
        gdn_output_norm_quantize,
    )
    from torchtitan.ops.float8._cudacherry import quantize_vec128
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    # Each case represents a fixed training shape. Do not let Dynamo infer a
    # dynamic-shape workload from recompiles of the smaller fixture.
    torch._dynamo.reset()
    out_dim = heads * 128
    x = bf16_rand(batch, seq_len, heads, 128, std=0.25)
    gate = bf16_rand(batch, seq_len, heads, 128, std=2.0)
    norm_weight = bf16_rand(128, std=0.15)
    x[0, 0] = 0  # Exercise the all-zero per-head scale floor.
    projection = FP8Linear(
        FP8Linear.Config(in_features=out_dim, out_features=out_dim, bias=False)
    )
    projection = projection.to(device="cuda", dtype=torch.bfloat16)

    reference = quantize_vec128(unfused_output_norm(x, gate, norm_weight), pack=True)
    data, scales = gdn_output_norm_quantize(x, gate, norm_weight, 1e-6, "sigmoid", True)
    torch.testing.assert_close(scales, reference._scale, rtol=0, atol=0)
    torch.testing.assert_close(
        data.view(torch.uint8), reference._data.view(torch.uint8), rtol=0, atol=0
    )

    leaves = tuple(v.requires_grad_() for v in (x, gate, norm_weight))
    cotangent = bf16_rand(batch, seq_len, out_dim, std=0.25)
    results = []
    for fused in (False, True):
        module = prepare(OutputProjection(projection, fused), compiled_sac)
        out = module(*leaves)
        grads = torch.autograd.grad(out, (*leaves, projection.weight), cotangent)
        results.append((out.detach(), grads))

    baseline, candidate = results
    torch.testing.assert_close(candidate[0], baseline[0], atol=0, rtol=0)
    stats = {
        name: compare_grad(actual, expected)
        for name, actual, expected in zip(
            ("dx", "dg", "dnorm_weight", "dprojection_weight"),
            candidate[1],
            baseline[1],
            strict=True,
        )
    }
    report = {
        "batch": batch,
        "sequence_length": seq_len,
        "compiled_sac": compiled_sac,
        "data_and_scales_bitwise": True,
        "projected_output_bitwise": True,
        **stats,
    }
    if (batch, seq_len, heads) == (2, 4096, 32):
        replay_inputs = tuple(v.detach() for v in leaves)

        def baseline_payload(x, gate, weight):
            data, scales, _, _ = quantize_vec128(
                unfused_output_norm(x, gate, weight), pack=False
            )
            return data, scales

        compiled_payload = torch.compile(
            baseline_payload, fullgraph=True, options={"triton.cudagraphs": False}
        )
        report["timing"] = graph_comparison(
            "GDN gated norm, multiplier and rowwise FP8 quantization before the output GEMM",
            lambda: compiled_payload(*replay_inputs),
            lambda: gdn_output_norm_quantize(*replay_inputs, 1e-6, "sigmoid", True),
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("short-conv", "output-norm"))
    parser.add_argument(
        "--quick", action="store_true", help="Skip the production-shaped case."
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This component check requires Hopper CUDA.")
    torch.manual_seed(910)
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    cases = []
    if args.component == "short-conv":
        cases.append(short_conv_case(1, 161, 2, 4, [0, 1, 3, 68, 161], False))
        cases.append(short_conv_case(2, 129, 2, 4, None, True))
        if not args.quick:
            cases.append(short_conv_case(2, 4096, 16, 32, None, True))
    else:
        cases.append(norm_case(2, 64, 4, False))
        cases.append(norm_case(2, 64, 4, True))
        if not args.quick:
            cases.append(norm_case(2, 4096, 32, True))
    record = {
        "component": args.component,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cases": cases,
    }
    evidence = json.dumps(record, indent=2) + "\n"
    if args.output:
        args.output.write_text(evidence)
    print(evidence, end="", flush=True)


if __name__ == "__main__":
    main()
