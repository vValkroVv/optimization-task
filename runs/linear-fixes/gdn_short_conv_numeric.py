"""Diagnose original native FLA versus generalized BF16 short-conv backward."""

import argparse
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import triton
import triton.language as tl


BATCH, TOKENS, HEADS, VALUE_HEADS, WIDTH, TAPS, BT = 2, 4096, 16, 32, 128, 4, 64
CHANNELS = (2 * HEADS + VALUE_HEADS) * WIDTH
FIELDS = ("acc", "norm", "rstd", "dot", "packed_dy", "post_silu")
DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float32,
    torch.float32,
    torch.bfloat16,
    torch.float32,
)


def sha256(path):
    with path.open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def comparison(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    delta = actual.float() - expected.float()
    rms = delta.square().mean().sqrt().item()
    reference_rms = expected.float().square().mean().sqrt().item()
    return {
        "dtype": str(actual.dtype),
        "elements": actual.numel(),
        "different_elements": (actual != expected).count_nonzero().item(),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "max_abs": delta.abs().max().item(),
        "rms": rms,
        "relative_rms": rms / reference_rms if reference_rms else None,
    }


def instrumented_kernel(source, output):
    code = source.read_text().split("@torch.library.custom_op", 1)[0]

    def replace(old, new):
        nonlocal code
        if code.count(old) != 1:
            raise ValueError(f"Unexpected source layout around {old!r}")
        code = code.replace(old, new)

    buffers = ["Dbg" + name.title().replace("_", "") for name in FIELDS]
    replace(
        "    EPS: tl.constexpr,\n):\n    valid",
        "    EPS: tl.constexpr,\n"
        + "".join(f"    {name},\n" for name in buffers)
        + "):\n    valid",
    )
    replace(
        "    grad = tl.load(\n",
        "    debug_offsets = (start + rows[:, None]) * QKV_DIM + columns[None, :]\n"
        "    tl.store(DbgAcc + debug_offsets, acc, valid)\n"
        "    grad = tl.load(\n",
    )
    replace(
        "            dot = tl.sum(grad * norm, axis=1)\n",
        "            dot = tl.sum(grad * norm, axis=1)\n"
        "            tl.store(DbgNorm + debug_offsets, norm, valid)\n"
        "            tl.store(DbgRstd + debug_offsets, rstd[:, None], valid)\n"
        "            tl.store(DbgDot + debug_offsets, dot[:, None], valid)\n",
    )
    replace(
        "    pre_silu = acc.to(QKV.dtype.element_ty).to(tl.float32)\n",
        "    tl.store(DbgPackedDy + debug_offsets, grad, valid)\n"
        "    pre_silu = acc.to(QKV.dtype.element_ty).to(tl.float32)\n",
    )
    replace(
        "    return tl.where(valid, grad, 0)\n",
        "    tl.store(DbgPostSilu + debug_offsets, grad, valid)\n"
        "    return tl.where(valid, grad, 0)\n",
    )
    replace(
        "    PartialWeight,\n    T,\n",
        "    PartialWeight,\n    DxFP32,\n"
        + "".join(
            f"    {base}{name},\n" for base in ("Main", "Halo") for name in buffers
        )
        + "    T,\n",
    )
    for base, indent in (("Main", "    "), ("Halo", "            ")):
        end = indent + "    EPS,\n" + indent + ")"
        replace(
            end,
            indent
            + "    EPS,\n"
            + "".join(indent + f"    {base}{name},\n" for name in buffers)
            + indent
            + ")",
        )
    replace(
        "        tl.store(Output + offsets, dx, valid)\n",
        "        tl.store(DxFP32 + offsets, dx, valid)\n"
        "        tl.store(Output + offsets, dx, valid)\n",
    )
    output.write_text(code)
    spec = spec_from_file_location("short_conv_numeric_instrumented", output)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._gdn_short_conv_backward_kernel


@triton.jit
def native_silu_derivative(
    Packed, Preactivation, Output, N: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dy = tl.load(Packed + offset, offset < N, 0).to(tl.float32)
    pre = tl.load(Preactivation + offset, offset < N, 0).to(tl.float32)
    # Match the expression and accumulation type in FLA causal_conv1d_bwd_kernel.
    sig = tl.sigmoid(pre)
    dy = dy * sig * (1 + pre * (1 - sig))
    tl.store(Output + offset, dy, offset < N)


def synthetic_crop():
    generator = torch.Generator(device="cuda").manual_seed(2048)

    def sample(shape):
        return (
            torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
            * 0.5
        )

    return {
        "qkv": sample((BATCH, TOKENS, CHANNELS)),
        "conv_weight": sample((CHANNELS, 1, TAPS)),
        "grad_q": sample((BATCH, TOKENS, HEADS, WIDTH)),
        "grad_k": sample((BATCH, TOKENS, HEADS, WIDTH)),
        "grad_v": sample((BATCH, TOKENS, VALUE_HEADS, WIDTH)),
        "num_heads": HEADS,
        "num_v_heads": VALUE_HEADS,
        "cu_seqlens": None,
        "eps": 2e-5,
        "normalize_qk": True,
    }


def replay_capture(kernel, x, weight, gradients, eps):
    dx = torch.empty_like(x)
    dx_fp32 = torch.empty_like(x, dtype=torch.float32)
    partials = x.new_empty((BATCH * TOKENS // BT, CHANNELS, TAPS), dtype=torch.float32)
    captures = tuple(
        {
            field: torch.empty_like(x, dtype=dtype)
            for field, dtype in zip(FIELDS, DTYPES, strict=True)
        }
        for _ in range(2)
    )
    compiled = kernel[(CHANNELS // WIDTH, TOKENS // BT, BATCH)](
        x,
        weight,
        *gradients,
        None,
        None,
        dx,
        partials,
        dx_fp32,
        *captures[0].values(),
        *captures[1].values(),
        TOKENS,
        CHANNELS,
        HEADS,
        VALUE_HEADS,
        WIDTH,
        WIDTH,
        WIDTH,
        TAPS,
        BT,
        4,
        True,
        eps,
        False,
        False,
        num_warps=4,
    )
    return (dx, partials.sum(0).to(weight.dtype)), dx_fp32, partials, captures, compiled


def native_stages(x, weight, gradients, eps):
    from fla.modules.conv.triton import ops
    from fla.ops.common.fused_conv_qkv import (
        _fused_qkv_conv1d_unpack_forward,
        _pack_qkv_backward_grad,
        fused_qkv_conv1d_unpack,
    )

    dims = {
        "num_heads": HEADS,
        "num_v_heads": VALUE_HEADS,
        "head_dim": WIDTH,
        "head_v_dim": WIDTH,
        "normalize_qk": True,
    }
    leaves = tuple(tensor.detach().requires_grad_() for tensor in (x, weight))
    public_qkv = fused_qkv_conv1d_unpack(*leaves, eps=eps, **dims)
    native_kernel = ops.causal_conv1d_bwd_kernel

    class BackwardCapture:
        def __getitem__(self, grid):
            def launch(**kwargs):
                self.partials = kwargs["dw"]
                self.compiled = native_kernel[grid](**kwargs)
                return self.compiled

            return launch

    capture = BackwardCapture()
    with patch.object(ops, "causal_conv1d_bwd_kernel", capture):
        public = torch.autograd.grad(public_qkv, leaves, gradients)
    assert torch.equal(capture.partials.sum(0).to(weight.dtype), public[1])
    rstd_q = x.new_empty((BATCH, TOKENS, HEADS), dtype=torch.float32)
    rstd_k = torch.empty_like(rstd_q)
    q, k, v = _fused_qkv_conv1d_unpack_forward(
        x,
        weight,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
        eps=eps,
        **dims,
    )
    packed = _pack_qkv_backward_grad(*gradients, q, k, rstd_q, rstd_k, bt=BT, **dims)
    preactivation, _ = ops.causal_conv1d_fwd(x, weight, bias=None, residual=None)
    post = torch.empty_like(x, dtype=torch.float32)
    native_silu_derivative[(triton.cdiv(x.numel(), 1024),)](
        packed, preactivation, post, x.numel(), 1024
    )

    def accumulate(dy):
        return ops.causal_conv1d_bwd(x=x, dy=dy, dht=None, weight=weight, BT=BT)[:2]

    return {
        "qkv": (q, k, v),
        "normalized": torch.stack((q, k), dim=3),
        "rstd": torch.stack((rstd_q, rstd_k), dim=3),
        "packed": packed,
        "preactivation": preactivation,
        "post_silu": post,
        "public": public,
        "partials": capture.partials,
        "compiled": capture.compiled,
        "replayed": accumulate(post),
        "accumulate": accumulate,
    }


def points_above(actual, expected, limit=16, strict=True):
    if strict:
        mask = ~torch.isclose(
            actual, expected, atol=3e-5, rtol=torch.finfo(actual.dtype).eps
        )
    else:
        mask = actual != expected
    count = mask.count_nonzero().item()
    if count > limit:
        raise ValueError(f"{count} points exceeds local diagnostic limit {limit}")
    return mask.nonzero().tolist()


def dx_point(point, x, weight, gradients, eps, native, fused, captures, dx_fp32):
    batch, token, channel = point
    segment, column = divmod(channel, WIDTH)
    head, role = divmod(segment, 4)
    features = slice(segment * WIDTH, (segment + 1) * WIDTH)
    last = min(token + TAPS, TOKENS)
    x0 = max(0, token - TAPS + 1)
    window = x[batch, x0:last, features].cpu().double()
    weights = weight[features].cpu().double()
    native_dx, fused_dx = (
        float(tensor[batch, token, channel])
        for tensor in (native["public"][0], fused[0])
    )
    result = {
        "point": point,
        "native": native_dx,
        "fused": fused_dx,
        "instrumented_before_dx_round": float(dx_fp32[batch, token, channel]),
        "taps": [],
    }

    def scalar(name, position, stage):
        return float(stage[name][position])

    for shift, row in enumerate(range(token, last)):
        position = batch, row, channel
        captured = captures[int(row // BT != token // BT)]
        start = max(0, row - TAPS + 1)
        samples = window[start - x0 : row - x0 + 1]
        acc = (samples * weights[:, -len(samples) :].T).sum(0)
        pre = acc[column].bfloat16().double()
        pre_sig = pre.sigmoid()
        dsilu = pre_sig * (1 + pre * (1 - pre_sig))
        if role < 2:
            silu = (acc * acc.sigmoid()).bfloat16().double()
            rstd = (silu.square().sum() + eps).rsqrt()
            norm = (silu * rstd).bfloat16().double()
            grad = gradients[role][batch, row, head].cpu().double()
            dot = (grad * norm).sum()
            before_round = (grad[column] - norm[column] * dot) * rstd
            packed = before_round.bfloat16().double()
            native_norm = float(native["normalized"][batch, row, head, role, column])
            native_rstd = float(native["rstd"][batch, row, head, role])
            specific = {
                "normalized": [
                    native_norm,
                    scalar("norm", position, captured),
                    float(norm[column]),
                ],
                "rstd": [native_rstd, scalar("rstd", position, captured), float(rstd)],
                "fp32_dot_vs_fp64": [scalar("dot", position, captured), float(dot)],
            }
        else:
            before_round = (
                gradients[2][batch, row, 2 * head + role - 2, column].cpu().double()
            )
            packed = before_round
            specific = {}
        native_post = float(native["post_silu"][position])
        fused_post = scalar("post_silu", position, captured)
        tap_weight = float(weights[column, TAPS - shift - 1])
        result["taps"].append(
            {
                "output_token": row,
                "halo": captured is captures[1],
                "weight": tap_weight,
                "preactivation_bf16": [
                    float(native["preactivation"][position]),
                    float(captured["acc"][position].bfloat16()),
                    float(pre),
                ],
                "packed_dy_bf16": [
                    float(native["packed"][position]),
                    scalar("packed_dy", position, captured),
                    float(packed),
                ],
                "fp64_packed_dy_before_cast": float(before_round),
                "post_silu_fp32_vs_fp64": [
                    native_post,
                    fused_post,
                    float(packed * dsilu),
                ],
                "contributions_fp64": [
                    native_post * tap_weight,
                    fused_post * tap_weight,
                    float(packed * dsilu) * tap_weight,
                ],
                **specific,
            }
        )
    totals = [sum(t["contributions_fp64"][i] for t in result["taps"]) for i in range(3)]
    result["fp64_lane_accumulation"] = totals
    result["fp64_lane_accumulation_rounded"] = torch.tensor(
        totals, dtype=torch.bfloat16
    ).tolist()
    return result


def dw_point(point, x, native, fused, captures, partials):
    channel, tap = point
    shift = TAPS - 1 - tap
    token = torch.arange(TOKENS - shift, device=x.device)
    crosses_tile = token // BT != (token + shift) // BT
    native_dy = native["post_silu"][:, shift:, channel]
    fused_dy = torch.where(
        crosses_tile,
        captures[1]["post_silu"][:, shift:, channel],
        captures[0]["post_silu"][:, shift:, channel],
    )
    activation = x[:, : TOKENS - shift, channel].double()
    native_partials = native["partials"][:, channel, tap]
    fused_partials = partials[:, channel, tap]
    return {
        "point": point,
        "native": float(native["public"][1][channel, tap]),
        "fused": float(fused[1][channel, tap]),
        "post_silu_inputs": comparison(fused_dy, native_dy),
        "fp32_partials": comparison(fused_partials, native_partials),
        "native_fp64_sum_of_partials": float(native_partials.double().sum()),
        "fused_fp64_sum_of_partials": float(fused_partials.double().sum()),
        "native_fp64_replay_from_post_silu": float((activation * native_dy).sum()),
        "fused_fp64_replay_from_post_silu": float((activation * fused_dy).sum()),
        "native_fp32_final_sum": float(native["partials"].sum(0)[channel, tap]),
        "fused_fp32_final_sum": float(partials.sum(0)[channel, tap]),
    }


def run_case(fixture, kernel, case, output):
    from torchtitan.models.common.attention.linear.gdn_short_conv import (
        _gdn_short_conv_backward_kernel,
        gdn_short_conv_backward,
    )

    x = fixture["qkv"].contiguous()
    weight = fixture["conv_weight"].reshape(CHANNELS, TAPS).contiguous()
    gradients = tuple(
        fixture[name].contiguous() for name in ("grad_q", "grad_k", "grad_v")
    )
    assert x.shape == (BATCH, TOKENS, CHANNELS)
    assert fixture["num_heads"] == HEADS and fixture["num_v_heads"] == VALUE_HEADS
    assert fixture["cu_seqlens"] is None and fixture["normalize_qk"]
    assert all(t.dtype == torch.bfloat16 for t in (x, weight, *gradients))
    eps = fixture["eps"]
    native = native_stages(x, weight, gradients, eps)
    fused = gdn_short_conv_backward(
        x,
        weight,
        *gradients,
        None,
        HEADS,
        VALUE_HEADS,
        WIDTH,
        WIDTH,
        True,
        eps,
    )
    measured, dx_fp32, partials, captures, compiled = replay_capture(
        kernel, x, weight, gradients, eps
    )

    # Captures are evidence only where they reproduce the uninstrumented result.
    def compare_gradients(pair, reference):
        return dict(
            zip(
                ("dx", "dw"),
                (comparison(a, b) for a, b in zip(pair, reference, strict=True)),
                strict=True,
            )
        )

    results = {
        "native_vs_fused": compare_gradients(fused, native["public"]),
        "native_staged_replay_vs_native": compare_gradients(
            native["replayed"], native["public"]
        ),
        "instrumented_vs_fused": compare_gradients(measured, fused),
        "instrumented_vs_native": compare_gradients(measured, native["public"]),
    }
    production_kernel = tuple(_gdn_short_conv_backward_kernel.device_caches.values())[
        0
    ][0]
    if case == "real":
        for tag, impl in (
            ("instrumented", compiled),
            ("fused", list(production_kernel.values())[-1]),
            ("native", native["compiled"]),
        ):
            (output / f"{tag}.ttgir").write_text(impl.asm["ttgir"])
            (output / f"{tag}.ptx").write_text(impl.asm["ptx"])
    main = captures[0]

    def normalized_view(tensor):
        return tensor.view(BATCH, TOKENS, HEADS, 4, WIDTH)[:, :, :, :2, :]

    results["stages"] = {
        "rounded_preactivation": comparison(
            main["acc"].bfloat16(), native["preactivation"]
        ),
        "rounded_normalized_qk": comparison(
            normalized_view(main["norm"]), native["normalized"]
        ),
        "fp32_rstd": comparison(normalized_view(main["rstd"])[..., 0], native["rstd"]),
        "rounded_packed_dy": comparison(main["packed_dy"], native["packed"]),
        "fp32_post_silu": comparison(main["post_silu"], native["post_silu"]),
    }
    halo_token = (
        torch.arange(1, TOKENS // BT, device=x.device)[:, None] * BT
        + torch.arange(TAPS - 1, device=x.device)
    ).flatten()
    halo = captures[1]
    results["halo_stages"] = {
        "rounded_preactivation": comparison(
            halo["acc"][:, halo_token].bfloat16(),
            native["preactivation"][:, halo_token],
        ),
        "rounded_normalized_qk": comparison(
            normalized_view(halo["norm"])[:, halo_token],
            native["normalized"][:, halo_token],
        ),
        "fp32_rstd": comparison(
            normalized_view(halo["rstd"])[:, halo_token, ..., 0],
            native["rstd"][:, halo_token],
        ),
        "rounded_packed_dy": comparison(
            halo["packed_dy"][:, halo_token], native["packed"][:, halo_token]
        ),
        "fp32_post_silu": comparison(
            halo["post_silu"][:, halo_token], native["post_silu"][:, halo_token]
        ),
    }
    fused_native_accumulation = native["accumulate"](main["post_silu"])
    results["replay_using_fused_main_derivative"] = {
        "vs_original": compare_gradients(fused_native_accumulation, native["public"]),
        "vs_fused": compare_gradients(fused_native_accumulation, fused),
    }
    points = points_above(fused[0], native["public"][0], strict=case != "real")
    results["dx_outliers"] = [
        dx_point(p, x, weight, gradients, eps, native, fused, captures, dx_fp32)
        for p in points
    ]
    dw_points = points_above(fused[1], native["public"][1], strict=False)
    results["dw_outliers"] = [
        dw_point(p, x, native, fused, captures, partials) for p in dw_points
    ]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (
        9,
        0,
    )
    from fla.modules.conv.triton import kernels as fla_conv
    from fla.ops.common import fused_conv_qkv as fla_qkv
    from torchtitan.models.common.attention.linear import gdn_short_conv

    args.output.mkdir(parents=True)
    generated = args.output / "instrumented_short_conv.py"
    kernel = instrumented_kernel(Path(gdn_short_conv.__file__), generated)
    receipt = json.loads(args.fixture.with_suffix(".json").read_text())
    assert sha256(args.fixture) == receipt["fixture_sha256"]
    fixture = torch.load(args.fixture, weights_only=True, map_location="cpu")
    for name in ("qkv", "conv_weight", "grad_q", "grad_k", "grad_v"):
        fixture[name] = fixture[name].cuda()
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "compiled_source_sha256": {
            str(path): sha256(Path(path))
            for path in (
                __file__,
                gdn_short_conv.__file__,
                fla_conv.__file__,
                fla_qkv.__file__,
                generated,
            )
        },
        "fixture": receipt,
        "synthetic_seed": 2048,
    }
    for case, data in (("real", fixture), ("synthetic", synthetic_crop())):
        result[case] = run_case(data, kernel, case, args.output)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            case,
            result[case]["native_vs_fused"],
            result[case]["instrumented_vs_fused"],
            flush=True,
        )


if __name__ == "__main__":
    main()
