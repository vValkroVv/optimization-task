"""Compare the three GDN changes together against the unfused native module."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

FLAGS = (
    "GIFT_GDN_DIRECT_FLA_BACKWARD",
    "GIFT_GDN_FUSE_SHORT_CONV_BWD",
    "GIFT_GDN_FUSE_OUTPUT_NORM_FP8",
)
MARKERS = (
    "torchtitan::gdn_fla_backward",
    "torchtitan::gdn_short_conv_backward",
    "torchtitan::gdn_output_norm_quantize",
)
CONFIG_SHA256 = "5e389eebaf3f09d0c9d40232afee8f0ef52ce40530ce2a3dc3510f009ba2d6aa"


def configure(resolved, fixture):
    from torchtitan.config.schema.models.common.norms.fused_gated_rmsnorm import (
        FusedRMSNormGatedConfig,
    )
    from torchtitan.models.common.attention.linear.gdn import GatedDeltaNet
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    model = resolved["model"]["config"]
    gdn = model["blocks"]["gdn_dense"]["attention"]
    train = resolved["training"]
    width = model["dim"]
    b, s = train["local_batch_size"], train["seq_len"]
    key_shape = (b, s, gdn["num_k_heads"], gdn["head_k_dim"])
    value_shape = (b, s, gdn["num_v_heads"], gdn["head_v_dim"])
    value_width = value_shape[-2] * value_shape[-1]
    packed_width = 2 * key_shape[-2] * key_shape[-1] + value_width
    if (
        fixture["x"].shape != (b, s, width)
        or fixture["grad_q"].shape != key_shape
        or fixture["grad_v"].shape != value_shape
        or fixture["proj_weight"].shape != (packed_width, width)
        or fixture["conv_weight"].shape != (packed_width, 1, gdn["conv_size"])
        or fixture["cu_seqlens"] is not None
    ):
        raise ValueError("Fixture geometry differs from the resolved dense crop")
    config = GatedDeltaNet.Config(
        dim=width,
        num_k_heads=gdn["num_k_heads"],
        num_v_heads=gdn["num_v_heads"],
        head_k_dim=gdn["head_k_dim"],
        head_v_dim=gdn["head_v_dim"],
        conv_size=gdn["conv_size"],
        allow_neg_eigval=gdn["allow_neg_eigval"],
        o_norm=FusedRMSNormGatedConfig(**gdn["o_norm"]),
        o_norm_output_multiplier=gdn["o_norm_output_multiplier"],
        z_proj=FP8Linear.Config(
            in_features=width, out_features=value_width, bias=False
        ),
        o_proj=FP8Linear.Config(
            in_features=value_width, out_features=width, bias=False
        ),
    )
    return config._copy(
        qkv=config.qkv._copy(
            qkv_proj=FP8Linear.Config(
                in_features=width, out_features=packed_width, bias=False
            ),
            eps=fixture["eps"],
            normalize_qk=fixture["normalize_qk"],
        )
    )


def build(config, switches):
    env = {name: str(int(name in switches)) for name in FLAGS}
    with patch.dict(os.environ, env):
        return config.build().cuda().to(torch.bfloat16)


@torch.no_grad()
def init(module, fixture, seed):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    for projection in (module.ba_proj, module.z_proj, module.o_proj):
        projection.weight.normal_(
            std=projection.weight.shape[-1] ** -0.5, generator=rng
        )
    module.qkv.qkv_proj.weight.copy_(fixture["proj_weight"])
    module.qkv.qkv_conv1d.weight.copy_(fixture["conv_weight"])
    module.o_norm.weight.zero_()
    module.A_log.fill_(1)
    module.dt_bias.zero_()


def run(module, x, grad, rope, ac_config, mode, markers=None):
    from torchtitan.distributed.activation_checkpoint import _apply_ac_to_block

    leaves = tuple(module.parameters())
    checkpointed = _apply_ac_to_block(module, ac_config, mode=mode)
    compiled = torch.compile(
        checkpointed, fullgraph=True, options={"triton.cudagraphs": False}
    )

    def value_and_grad():
        hidden = x.detach().requires_grad_(True)
        output = compiled(hidden, rope, None)
        return (output.detach(), *torch.autograd.grad(output, (hidden, *leaves), grad))

    warm = value_and_grad()
    if markers is None:
        return warm
    del warm
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        result = value_and_grad()
    active = {event.key for event in p.key_averages()} & set(MARKERS)
    if active != set(markers):
        raise RuntimeError(f"Expected {markers}, observed {sorted(active)}")
    return result


def compare(names, reference, actual):
    stats = {}
    for name, ref, out in zip(names, reference, actual, strict=True):
        if ref.shape != out.shape or ref.dtype != out.dtype:
            raise ValueError(f"Shape or dtype mismatch in {name}")
        if not (torch.isfinite(ref).all() and torch.isfinite(out).all()):
            raise ValueError(f"Non-finite values in {name}")
        left, right = ref.flatten(), out.flatten()
        count = different_bytes = 0
        max_abs = squared = squared_ref = 0.0
        for a, b in zip(left.split(1048576), right.split(1048576), strict=True):
            different_bytes += torch.count_nonzero(
                a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)
            ).item()
            delta = a.double() - b.double()
            count += torch.count_nonzero(delta).item()
            max_abs = max(max_abs, delta.abs().max().item())
            squared += delta.square().sum().item()
            squared_ref += a.double().square().sum().item()
        stats[name] = {
            "shape": list(ref.shape),
            "dtype": str(ref.dtype),
            "finite": True,
            "different_elements": count,
            "different_bytes": different_bytes,
            "max_abs": max_abs,
            "relative_l2": (squared / squared_ref) ** 0.5 if squared_ref else None,
        }
    return stats


def main(args):
    import yaml

    from torchtitan.config import ActivationCheckpointConfig

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one assigned H100")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("Use native FLA")
    spec = spec_from_file_location("gdn_component_helpers", args.helpers)
    helpers = module_from_spec(spec)
    spec.loader.exec_module(helpers)
    if helpers.digest(args.config) != CONFIG_SHA256:
        raise ValueError("Expected the original A resolved 10+10 configuration")
    resolved = yaml.safe_load(args.config.read_text())
    fixture, receipt = helpers.fixture_on_device(args.fixture)
    config = configure(resolved, fixture)
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    ac_config = ActivationCheckpointConfig(**resolved["activation_checkpoint"])
    mode = "selective"
    baseline = build(config, ())
    init(baseline, fixture, args.seed)
    names = ("output", "input", *(name for name, _ in baseline.named_parameters()))
    x = fixture["x"]
    upstream = fixture["grad_v"].flatten(2)
    with torch.no_grad():
        torch.testing.assert_close(
            baseline.qkv.qkv_proj(x), fixture["qkv"], atol=0, rtol=0
        )
    del fixture
    rope = torch.empty(0, device=x.device)
    reference = run(baseline, x, upstream, rope, ac_config, mode, markers=())
    repeat = run(baseline, x, upstream, rope, ac_config, mode)
    aa = compare(names, reference, repeat)
    del repeat

    def changed(switches, markers):
        candidate = build(config, switches)
        candidate.load_state_dict(baseline.state_dict())
        return run(candidate, x, upstream, rope, ac_config, mode, markers=markers)

    all_on = changed(FLAGS, MARKERS)
    result = {
        "scope": "actual GDN module with Inductor fullgraph and layer-0 selective AC",
        "data": "captured input, QKV and conv weights; seeded other projections; captured dV as dOutput",
        "seed": args.seed,
        "config_sha256": CONFIG_SHA256,
        "fixture": receipt,
        "harness_sha256": helpers.digest(Path(__file__)),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "baseline_repeat": aa,
        "all_three_vs_baseline": compare(names, reference, all_on),
        "candidate_activation": MARKERS,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if any(row["different_bytes"] for row in result["all_three_vs_baseline"].values()):
        short_conv = changed((FLAGS[1],), (MARKERS[1],))
        result["short_conv_only_vs_baseline"] = compare(names, reference, short_conv)
        result["all_three_vs_short_conv_only"] = compare(names, short_conv, all_on)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output, flush=True)
    if result["all_three_vs_baseline"]["output"]["different_bytes"]:
        raise RuntimeError(
            "The backward-only changes and FP8 producer changed GDN output"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2048)
    main(parser.parse_args())
