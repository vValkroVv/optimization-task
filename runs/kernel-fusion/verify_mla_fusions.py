"""Validate one MLA fusion at the selected crop geometry on a single H100."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import statistics

import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

from torchtitan.models.common.attention.softmax.gate import SigmoidGate
from torchtitan.models.common.attention.softmax.kernels.sdpa import (
    ScaledDotProductAttention,
)
from torchtitan.models.common.attention.softmax.mla import MLA
from torchtitan.models.common.attention.softmax.mla_rope_pack import MLARoPEPack
from torchtitan.models.common.embeddings.rope.rope import (
    ComplexRoPE,
    apply_rotary_emb_complex,
)
from torchtitan.ops.float8.cuda_gemm import FP8Linear


def make(*shape):
    return torch.randn(shape, dtype=torch.bfloat16, device="cuda")


def compare(actual, reference):
    diff = actual.float() - reference.float()
    return {
        "elements": diff.numel(),
        "mismatches": int(torch.count_nonzero(diff)),
        "max_abs": float(diff.abs().max()),
        "rmse": float(diff.square().mean().sqrt()),
    }


def sac_context():
    return create_selective_checkpoint_contexts(
        [torch.ops.aten._scaled_dot_product_cudnn_attention.default]
    )


def selective(fn, *args):
    return checkpoint(
        fn,
        *args,
        use_reentrant=False,
        context_fn=sac_context,
    )


def observe(fn, inputs, incoming):
    return measured(fn, inputs, incoming)[0]


def measured(fn, inputs, incoming):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*inputs)
    grads = torch.autograd.grad(out, inputs, incoming)
    end.record()
    end.synchronize()
    return (
        tuple(x.detach() for x in (out, *grads)),
        1000.0 * start.elapsed_time(end),
    )


def paired_check(baseline, candidate, inputs, incoming, names, kernel_names):
    baseline = torch.compile(baseline, fullgraph=True, dynamic=False)
    candidate = torch.compile(candidate, fullgraph=True, dynamic=False)
    reference = observe(baseline, inputs, incoming)
    proposed = observe(candidate, inputs, incoming)
    control = observe(baseline, inputs, incoming)
    differences = dict(
        zip(names, (compare(a, b) for a, b in zip(proposed, reference)), strict=True)
    )
    noise = dict(
        zip(names, (compare(a, b) for a, b in zip(control, reference)), strict=True)
    )
    del reference, proposed, control
    timings = {"baseline": [], "candidate": []}
    peaks = {}
    for repeat in range(6):
        order = (("baseline", baseline), ("candidate", candidate))
        if repeat % 2:
            order = reversed(order)
        for name, fn in order:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            outputs, elapsed = measured(fn, inputs, incoming)
            timings[name].append(elapsed)
            additional_bytes = torch.cuda.max_memory_allocated() - before
            peaks[name] = max(peaks.get(name, 0), additional_bytes)
            del outputs
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        observe(candidate, inputs, incoming)
    kernels = sorted(
        {
            event.name
            for event in profiler.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        }
    )
    for name in kernel_names:
        assert any(name in kernel for kernel in kernels), (name, kernels)
    return {
        "baseline_aa": noise,
        "candidate_vs_baseline": differences,
        "forward_backward_cuda_event_us": timings,
        "median_forward_backward_cuda_event_us": {
            key: statistics.median(samples) for key, samples in timings.items()
        },
        "max_additional_peak_allocated_bytes": peaks,
        "candidate_cuda_kernels": kernels,
    }


def verify_rope(metrics):
    shape = (2, 4096, 32, 192)
    q = make(*shape).requires_grad_()
    kv = make(*shape[:3], 256)
    k = kv[..., :128].requires_grad_()
    v = kv[..., 128:].requires_grad_()
    k_rope = make(*shape[:2], 576)[..., 512:].requires_grad_()
    with torch.device("cuda"):
        cache = ComplexRoPE(
            ComplexRoPE.Config(dim=64, max_seq_len=4096, theta=100000.0)
        ).cache
    positions = torch.arange(4096, dtype=torch.int32, device="cuda").repeat(2, 1)
    inner = ScaledDotProductAttention.Config().build()
    incoming = make(*v.shape)

    def run(q, k, k_rope, v, *, fused):
        if fused:
            packed_q, packed_k = MLARoPEPack.apply(q, k, k_rope, cache, positions)
        else:
            (q_rope,) = apply_rotary_emb_complex(
                q[..., 128:], freqs_cis=cache, positions=positions
            )
            (key_rope,) = apply_rotary_emb_complex(
                k_rope.unsqueeze(2), freqs_cis=cache, positions=positions
            )
            packed_q = torch.cat((q[..., :128], q_rope), dim=-1)
            packed_k = torch.cat((k, key_rope.expand(-1, -1, 32, -1)), dim=-1)
        assert packed_q.is_contiguous() and packed_k.is_contiguous()
        return inner(packed_q, packed_k, v, scale=192**-0.5)

    metrics["selected_shape"] = paired_check(
        partial(selective, partial(run, fused=False)),
        partial(selective, partial(run, fused=True)),
        (q, k, k_rope, v),
        incoming,
        ("attention", "query", "key", "shared_key_rotary", "value"),
        ("_mla_rope_pack", "_mla_rope_unpack", "cudnn"),
    )
    del q, k, k_rope, v, incoming, kv

    def build_mla(enabled, module_dtype):
        os.environ["GIFT_MLA_FUSED_ROPE"] = "1" if enabled else "0"
        with torch.device("cuda"), torch.no_grad():
            return (
                MLA.Config(
                    n_heads=32,
                    dim=4096,
                    use_mla_scaling_factor=True,
                )
                .build()
                .to(dtype=module_dtype)
            )

    baseline_mla = build_mla(False, torch.bfloat16)
    candidate_mla = build_mla(True, torch.bfloat16)
    candidate_mla.load_state_dict(baseline_mla.state_dict(), strict=True)
    assert candidate_mla._fused_rope
    x = make(2, 4096, 4096).requires_grad_()
    incoming = make(*x.shape)
    metrics["mla_dispatch"] = paired_check(
        lambda x: baseline_mla(x, cache, None, positions),
        lambda x: candidate_mla(x, cache, None, positions),
        (x,),
        incoming,
        ("mla", "input"),
        ("_mla_rope_pack", "_mla_rope_unpack", "cudnn"),
    )


def verify_gate(metrics):
    from torchtitan.ops.float8._cudacherry import quantize_vec128

    attention = make(2, 32, 4096, 128).transpose(1, 2).requires_grad_()
    logits = make(2, 4096, 4096).requires_grad_()
    gate = SigmoidGate.Config(gate_proj=MLA.Config(n_heads=32, dim=4096).wo).build()
    gate.gate_proj = torch.nn.Identity()
    with torch.device("cuda"):
        projection = FP8Linear.Config(in_features=4096, out_features=4096).build()
    projection.to(torch.bfloat16)
    with torch.no_grad():
        projection.weight.normal_(std=0.01)

    def unfused_gate(attention, logits):
        return gate(logits, attention.contiguous().flatten(2))

    def fused_gate(attention, logits):
        return gate(logits, attention, quantize_mla_output=True)

    baseline = torch.compile(unfused_gate, fullgraph=True)
    candidate = torch.compile(fused_gate, fullgraph=True)
    with torch.no_grad():
        reference = quantize_vec128(baseline(attention, logits), pack=True)
        actual = candidate(attention, logits)
        metrics["operand_data"] = compare(
            actual._data.view(torch.uint8), reference._data.view(torch.uint8)
        )
        metrics["operand_scales"] = compare(actual._scale, reference._scale)

    def run(attention, logits, weight, *, fused):
        gate_out = gate(
            logits,
            attention if fused else attention.contiguous().flatten(2),
            quantize_mla_output=fused,
        )
        return projection._apply_gemm(gate_out, weight, None, False, True)

    incoming = make(*logits.shape)
    metrics["gate_to_projection"] = paired_check(
        partial(selective, partial(run, fused=False)),
        partial(selective, partial(run, fused=True)),
        (attention, logits, projection.weight),
        incoming,
        ("projection", "attention", "gate_logits", "projection_weight"),
        ("_mla_gate_quantize",),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("rope", "gate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(2048)
    torch._inductor.config.triton.cudagraphs = False
    assert torch.cuda.get_device_capability() == (9, 0)
    results = {
        "variant": args.variant,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    verification = verify_rope if args.variant == "rope" else verify_gate
    verification(results)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if args.variant == "rope":
        numerics = results["selected_shape"]["candidate_vs_baseline"]
        for name in ("attention", "key", "shared_key_rotary", "value"):
            assert numerics[name]["max_abs"] == 0, (name, numerics[name])
        assert numerics["query"]["rmse"] < 1e-6, numerics["query"]
    else:
        assert results["operand_data"]["mismatches"] == 0, results["operand_data"]
        assert results["operand_scales"]["mismatches"] == 0, results["operand_scales"]
        numerics = results["gate_to_projection"]["candidate_vs_baseline"]
        assert numerics["projection"]["max_abs"] == 0, numerics["projection"]
        for name in ("attention", "gate_logits", "projection_weight"):
            assert numerics[name]["rmse"] < 1e-5, (name, numerics[name])


if __name__ == "__main__":
    main()
