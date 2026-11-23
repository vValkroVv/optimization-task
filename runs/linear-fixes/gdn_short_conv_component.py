"""Compare short-conv backwards under compiled SAC."""

import argparse
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint


SPECIALIZED_REVISION = "fe5950b56de5b2cec7ff4cd063f7338c98200c45"
SPECIALIZED_SHA256 = "02e92ec483524f05c87a207ddd7b3dccc85f62bdb4cbd071579d82a2c3a21edd"
ORIGINAL_A_CONFIG_SHA256 = (
    "5e389eebaf3f09d0c9d40232afee8f0ef52ce40530ce2a3dc3510f009ba2d6aa"
)


def specialized_wrapper(path, current_wrapper, setup):
    # Both backwards must be available for interleaved CUDA Graph replay.
    # Register the historical operator under a distinct name without editing
    # the source file or changing the JIT kernel.
    custom_op = torch.library.custom_op

    def register_historical(name, **kwargs):
        if name != "torchtitan::gdn_short_conv_backward":
            raise ValueError(f"Unexpected operator in historical source: {name}")
        return custom_op("linear_fixes::specialized_short_conv_backward", **kwargs)

    spec = spec_from_file_location("specialized_short_conv", path)
    historical = module_from_spec(spec)
    sys.modules[spec.name] = historical
    with patch.object(torch.library, "custom_op", register_historical):
        spec.loader.exec_module(historical)

    # The exact same forward body and fake metadata sit at this boundary.
    # Both ops use PREFER_RECOMPUTE in the experts_up SAC policy.
    specialized = custom_op(
        "linear_fixes::specialized_qkv_conv1d_unpack",
        current_wrapper._init_fn,
        mutates_args=(),
    )
    specialized.register_fake(current_wrapper._abstract_fn)

    def backward(ctx, grad_q, grad_k, grad_v):
        qkv, weight = ctx.saved_tensors
        nh, nvh, _hd, _hvd, cu_seqlens, normalize, eps, _fuse = ctx.fwd_args
        dqkv, dw = historical.gdn_short_conv_backward(
            qkv, weight, grad_q, grad_k, grad_v, cu_seqlens, nh, nvh, normalize, eps
        )
        return dqkv, dw, None, None, None, None, None, None, None, None

    specialized.register_autograd(backward, setup_context=setup)
    return specialized


def make_call(fixture, fuse_backward, conv_op=None):
    from torchtitan.distributed.activation_checkpoint import _sac_context_fn
    from torchtitan.models.common.attention.linear.fla_custom_ops import (
        qkv_conv1d_unpack_op,
    )

    if conv_op is None:
        conv_op = qkv_conv1d_unpack_op
    qkv = fixture["qkv"].detach().requires_grad_()
    weight = fixture["conv_weight"].detach().requires_grad_()
    incoming = tuple(fixture[key] for key in ("grad_q", "grad_k", "grad_v"))
    metadata = fixture["cu_seqlens"]

    def short_conv(x, w):
        return conv_op(
            x,
            w,
            fixture["num_heads"],
            fixture["num_v_heads"],
            incoming[0].shape[-1],
            incoming[2].shape[-1],
            metadata,
            fixture["normalize_qk"],
            fixture["eps"],
            fuse_backward,
        )

    @torch.compile(fullgraph=True)
    def forward(x, w):
        return checkpoint(
            short_conv,
            x,
            w,
            use_reentrant=False,
            context_fn=partial(_sac_context_fn, ("experts_up",), ()),
            early_stop=False,
        )

    def call():
        outputs = forward(qkv, weight)
        return (*outputs, *torch.autograd.grad(outputs, (qkv, weight), incoming))

    return call


def compare(baseline, candidate):
    result = {}
    for name, reference, actual in zip(
        ("q", "k", "v", "dqkv", "dweight"), baseline, candidate, strict=True
    ):
        is_gradient = name.startswith("d")
        torch.testing.assert_close(
            actual,
            reference,
            rtol=torch.finfo(actual.dtype).eps if is_gradient else 0,
            atol=3e-5 if is_gradient else 0,
        )
        result[name] = {
            "dtype": str(actual.dtype),
            "shape": list(actual.shape),
            "equal_elements": int(torch.count_nonzero(actual == reference)),
            "total_elements": actual.numel(),
            "max_abs_error": float((actual.float() - reference.float()).abs().max()),
        }
    return result


def operator_events(trace):
    return {
        event.get("name")
        for event in json.loads(Path(trace["path"]).read_text())["traceEvents"]
        if event.get("cat") == "cpu_op"
    }


def main(args):
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one assigned H100 GPU")
    output = args.output.resolve()
    output.mkdir(parents=True)
    helpers_path = args.helpers.resolve()
    spec = spec_from_file_location("gdn_component_helpers", helpers_path)
    helpers = module_from_spec(spec)
    spec.loader.exec_module(helpers)

    from fla.modules.conv.triton import kernels as fla_conv_kernels
    from fla.ops.common import fused_conv_qkv as fla_conv
    from torchtitan.models.common.attention.linear import (
        fla_custom_ops,
        gdn_short_conv,
    )
    import triton

    fixture, receipt = helpers.fixture_on_device(args.fixture)
    specialized = None
    historical_provenance = None
    if args.specialized_source is not None:
        if args.original_a_config is None:
            raise ValueError(
                "The specialized comparison needs the resolved 10+10 Original A config"
            )
        historical_source = args.specialized_source.resolve()
        original_a_config = args.original_a_config.resolve()
        if helpers.digest(historical_source) != SPECIALIZED_SHA256:
            raise ValueError(
                "The historical kernel hash does not match its donor revision"
            )
        if helpers.digest(original_a_config) != ORIGINAL_A_CONFIG_SHA256:
            raise ValueError(
                "The resolved config hash does not match the 10+10 receipt"
            )
        if (
            fixture["qkv"].shape != (2, 4096, 8192)
            or fixture["qkv"].dtype != torch.bfloat16
            or fixture["conv_weight"].shape != (8192, 1, 4)
            or fixture["grad_q"].shape != (2, 4096, 16, 128)
            or fixture["grad_k"].shape != (2, 4096, 16, 128)
            or fixture["grad_v"].shape != (2, 4096, 32, 128)
            or fixture["num_heads"] != 16
            or fixture["num_v_heads"] != 32
            or fixture["cu_seqlens"] is not None
        ):
            raise ValueError(
                "The specialized kernel needs the original dense BF16 crop"
            )
        specialized = specialized_wrapper(
            historical_source,
            fla_custom_ops.qkv_conv1d_unpack_op,
            fla_custom_ops._qkv_conv_setup,
        )
        historical_provenance = {
            "revision": SPECIALIZED_REVISION,
            "source": str(historical_source),
            "source_sha256": SPECIALIZED_SHA256,
            "resolved_original_a_config": str(original_a_config),
            "resolved_original_a_config_sha256": ORIGINAL_A_CONFIG_SHA256,
        }
    calls = {
        "baseline": make_call(fixture, specialized is not None, specialized),
        "candidate": make_call(fixture, True),
    }
    result = {
        "scope": "short-conv forward plus backward, Inductor fullgraph and selective AC",
        "fixture": str(args.fixture.resolve()),
        "fixture_receipt": receipt,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "file_sha256": {
            str(path): helpers.digest(path)
            for path in (
                helpers_path,
                Path(__file__).resolve(),
                Path(fla_conv.__file__),
                Path(fla_conv_kernels.__file__),
                Path(fla_custom_ops.__file__),
                Path(gdn_short_conv.__file__),
            )
        },
        "taps": fixture["conv_weight"].shape[-1],
        "heads": [fixture["num_heads"], fixture["num_v_heads"]],
        "packed": fixture["cu_seqlens"] is not None,
        "samples": args.samples,
        "graph_replays_per_sample": 20,
        "baseline": "historical specialized"
        if specialized is not None
        else "native FLA replay",
        "historical_specialized": historical_provenance,
    }
    result["comparison"] = compare(calls["baseline"](), calls["candidate"]())

    result["trace"] = {
        name: helpers.trace(call, output / f"{name}-trace.json")
        for name, call in calls.items()
    }
    names = {
        name: [kernel[0] for kernel in trace["kernels"]]
        for name, trace in result["trace"].items()
    }
    result["activation"] = {
        "baseline_pack": any("_qkv_bwd_pack_kernel" in n for n in names["baseline"]),
        "baseline_causal_backward": any(
            "causal_conv1d_bwd_kernel" in n for n in names["baseline"]
        ),
        "candidate_fused_backward": any(
            "_gdn_short_conv_backward_kernel" in n for n in names["candidate"]
        ),
        "candidate_replayed_fla_pack": any(
            "_qkv_bwd_pack_kernel" in n for n in names["candidate"]
        ),
    }
    if specialized is not None:
        result["activation"]["baseline_specialized_backward"] = (
            "linear_fixes::specialized_short_conv_backward"
            in operator_events(result["trace"]["baseline"])
        )
        result["activation"]["candidate_generalized_backward"] = (
            "torchtitan::gdn_short_conv_backward"
            in operator_events(result["trace"]["candidate"])
        )
    summary = output / "result.json"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    active = result["activation"]
    if specialized is not None:
        baseline_active = (
            active["baseline_specialized_backward"]
            and active["candidate_generalized_backward"]
            and not active["baseline_pack"]
            and not active["baseline_causal_backward"]
        )
    else:
        baseline_active = active["baseline_pack"] and active["baseline_causal_backward"]
    if (
        not baseline_active
        or not active["candidate_fused_backward"]
        or active["candidate_replayed_fla_pack"]
    ):
        raise RuntimeError(f"Unexpected short-conv kernel routing in {summary}")
    result["timing_us"] = helpers.pair_timing(calls, args.samples)
    result["memory"] = {name: helpers.memory(call) for name, call in calls.items()}
    summary.write_text(json.dumps(result, indent=2) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specialized-source", type=Path)
    parser.add_argument("--original-a-config", type=Path)
    parser.add_argument("--samples", type=int, default=7)
    main(parser.parse_args())
