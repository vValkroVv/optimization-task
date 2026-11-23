"""Measure the native GDN scan with replayed or direct backward on one GPU."""

import argparse
from collections import Counter
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess

import torch
from torch.utils.checkpoint import checkpoint


def load_scan_inputs(fixture, seed):
    from fla.ops.common.fused_conv_qkv import fused_qkv_conv1d_unpack

    x = fixture["x"]
    heads, value_heads = fixture["num_heads"], fixture["num_v_heads"]
    metadata = fixture["cu_seqlens"]
    q, k, v = fused_qkv_conv1d_unpack(
        fixture["qkv"],
        fixture["conv_weight"],
        num_heads=heads,
        num_v_heads=value_heads,
        head_dim=fixture["grad_q"].shape[-1],
        head_v_dim=fixture["grad_v"].shape[-1],
        cu_seqlens=metadata,
        normalize_qk=fixture["normalize_qk"],
        eps=fixture["eps"],
    )
    rng = torch.Generator(device=x.device).manual_seed(seed)
    ba_weight = torch.randn(
        (2 * value_heads, x.shape[-1]), generator=rng, device=x.device
    )
    ba_weight.mul_(x.shape[-1] ** -0.5)
    ba = torch.nn.functional.linear(x, ba_weight.to(x.dtype))
    ba = ba.reshape(*x.shape[:2], heads, 2 * value_heads // heads)
    b, a = ba.chunk(2, dim=-1)
    beta = b.reshape(*x.shape[:2], value_heads).sigmoid()
    gate = a.reshape_as(beta).contiguous()
    decay = torch.ones(value_heads, dtype=torch.float32, device=x.device)
    bias = torch.zeros_like(decay)
    return (q, k, v, gate, beta, decay, bias), metadata


def make_call(inputs, metadata, incoming, direct):
    from torchtitan.distributed.activation_checkpoint import _sac_context_fn
    from torchtitan.models.common.attention.linear.fla_custom_ops import gdn_chunk_op

    leaves = tuple(value.detach().requires_grad_() for value in inputs)

    def scan(*values):
        return gdn_chunk_op(*values, metadata, direct)

    @torch.compile(fullgraph=True)
    def forward(*values):
        return checkpoint(
            scan,
            *values,
            use_reentrant=False,
            context_fn=partial(_sac_context_fn, ("experts_up",), ()),
            early_stop=False,
        )

    def call():
        output = forward(*leaves)
        return (output, *torch.autograd.grad(output, leaves, incoming))

    return call


def main(args):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one assigned H100 GPU")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("Set FLA_DISABLE_BACKEND_DISPATCH=1 to compare native FLA")
    out = args.output.resolve()
    out.mkdir(parents=True)
    spec = spec_from_file_location("gdn_component_helpers", args.helpers)
    helpers = module_from_spec(spec)
    spec.loader.exec_module(helpers)

    from fla.ops.gated_delta_rule import chunk as fla_chunk, chunk_fwd
    from torchtitan.models.common.attention.linear import (
        fla_custom_ops,
        gdn_fla_backward,
    )
    import triton

    fixture, receipt = helpers.fixture_on_device(args.fixture)
    inputs, metadata = load_scan_inputs(fixture, args.seed)
    # The fixture captures dV from training. Reuse its values as a matched dO
    # of the same shape because the scan's actual incoming gradient was not saved.
    incoming = fixture["grad_v"]
    calls = {
        "baseline": make_call(inputs, metadata, incoming, False),
        "candidate": make_call(inputs, metadata, incoming, True),
    }
    names = ("output", "dq", "dk", "dv", "dgate", "dbeta", "dA_log", "ddt_bias")
    baseline, candidate = calls["baseline"](), calls["candidate"]()
    for name, reference, actual in zip(names, baseline, candidate, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0, msg=name)
    del baseline, candidate

    timing = helpers.pair_timing(calls, args.samples)
    memory = {name: helpers.memory(call) for name, call in calls.items()}
    traces = {
        name: helpers.trace(call, out / f"{name}-trace.json")
        for name, call in calls.items()
    }
    markers = {
        name: Counter(
            marker
            for kernel, *_ in row["kernels"]
            for marker in ("chunk_fwd_kernel_o", "recompute_w_u_fwd_kernel")
            if marker in kernel
        )
        for name, row in traces.items()
    }
    result = {
        "scope": "native GDN scan forward plus backward, Inductor fullgraph with SAC",
        "fixture": str(args.fixture.resolve()),
        "fixture_receipt": receipt,
        "fixture_adaptation": "captured qkv/conv weights, initialized ba and decay, captured dV replayed as dO",
        "seed": args.seed,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "shapes": [list(value.shape) for value in inputs],
        "dtypes": [str(value.dtype) for value in inputs],
        "packed": metadata is not None,
        "parity": dict.fromkeys(names, "exact"),
        "cuda_graph_replays_per_sample": 20,
        "timing_us": timing,
        "memory": memory,
        "traces": traces,
        "kernel_counts": markers,
        "source_sha256": {
            str(path): helpers.digest(path)
            for path in map(
                Path,
                (
                    args.helpers,
                    __file__,
                    fla_chunk.__file__,
                    chunk_fwd.__file__,
                    fla_custom_ops.__file__,
                    gdn_fla_backward.__file__,
                ),
            )
        },
    }
    summary = out / "result.json"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    for marker in ("chunk_fwd_kernel_o", "recompute_w_u_fwd_kernel"):
        if not markers["candidate"][marker] or (
            markers["baseline"][marker] - markers["candidate"][marker] != 1
        ):
            raise RuntimeError(f"Expected one fewer {marker} launch in {summary}")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2048)
    main(parser.parse_args())
