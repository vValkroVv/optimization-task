"""Benchmark + accuracy check for the CuTe GatedNorm / ZeroCenteredGatedNorm kernels.

Implementations compared (per norm variant):
  zero-centered (weight applied as 1 + w):
    ref           copy of foundry `_compiled_zero_centered_gated_norm` (torch.compile, --ref-mode)
    cute          this package, gated_norm(x, 1 + w, ...) (autograd fwd+bwd)
    foundry-ref   llmfoundry.models.layers.norm.ZeroCenteredGatedNorm(norm_backend="compile")
    foundry-cute  llmfoundry.models.layers.norm.ZeroCenteredGatedNorm(norm_backend="cute")
  gated (weight applied as w):
    ref           copy of foundry `_compiled_gated_norm`
    cute          this package, gated_norm(x, w, ...)
    foundry-ref   llmfoundry.models.layers.norm.GatedNorm(norm_backend="compile")
    foundry-cute  llmfoundry.models.layers.norm.GatedNorm(norm_backend="cute")
  foundry-* rows need llmfoundry.models.layers.norm importable (flash_attn, inductor); they
  are skipped with a message otherwise, or with --no-foundry.

Timings (triton do_bench, median ms, L2 flushed between reps):
  fwd        training forward (grad enabled, saves for backward)
  fwd+bwd    forward + torch.autograd.grad w.r.t. x and all three params
  bwd        fwd+bwd - fwd
  %peak      effective bandwidth vs --peak-gbps using the MINIMAL traffic:
             fwd = 4 B/elem (read x, write out), fwd+bwd = 10 B/elem
             (+ read x, read dy, write dx)
  cute[kernel] calls gated_norm_fwd / gated_norm_bwd directly: prep + F1 + F2 and
  B1 + B1f + B2 + Wfin launches (plus their output allocations) without autograd.

Accuracy: relative L2 error ||a - g|| / ||g|| against an fp64 golden computed
from the same bf16 inputs, for out, dx, dweight, dgate_up, dgate_down.
--gate-scales controls the spread of gate logits: larger values push more
gates towards 0/1.

Usage (on the H100 box):
  python -m llmfoundry.models.ops.cute_kernels.gated_norm.bench
  python bench.py --shapes 16384x7168
  python bench.py --variants gated --skip-accuracy
"""

import argparse
import math
import os
import sys
import traceback

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from llmfoundry.models.ops.cute_kernels.gated_norm import (gated_norm as cute_gated_norm,
                                                               gated_norm_bwd, gated_norm_fwd)
except ModuleNotFoundError as exc:  # checkout not installed: import from the repo root
    if exc.name != "llmfoundry" and not (exc.name or "").startswith("llmfoundry."):
        raise
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, *[".."] * 5)))
    from llmfoundry.models.ops.cute_kernels.gated_norm import (gated_norm as cute_gated_norm,
                                                               gated_norm_bwd, gated_norm_fwd)

RANK = 16
EPS = 1e-6
LGW = 2.0


# ---------------------------------------------------------------------------
# Foundry references (copied from llmfoundry/models/layers/norm.py).
# ---------------------------------------------------------------------------
def _make_refs(mode):
    @torch.compile(mode=mode, dynamic=True)
    def ref_gated(x, weight, gate_up_weight, gate_down_weight, eps, lgw):
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + eps)
        output = (output * weight.float()).to(dtype=x.dtype)
        gate_hidden = F.linear(output, gate_up_weight.to(dtype=x.dtype))
        gate_hidden = F.silu(gate_hidden.float()).to(dtype=x.dtype)
        gate_hidden = F.linear(gate_hidden, gate_down_weight.to(dtype=x.dtype))
        gate = torch.sigmoid(gate_hidden.float()).to(dtype=x.dtype)
        return (output * float(lgw) * gate).to(dtype=x.dtype)

    @torch.compile(mode=mode, dynamic=True)
    def ref_zero_centered(x, weight, gate_up_weight, gate_down_weight, eps, lgw):
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + eps)
        output = (output * (1.0 + weight.float())).to(dtype=x.dtype)
        gate_hidden = F.linear(output, gate_up_weight.to(dtype=x.dtype))
        gate_hidden = F.silu(gate_hidden.float()).to(dtype=x.dtype)
        gate_hidden = F.linear(gate_hidden, gate_down_weight.to(dtype=x.dtype))
        gate = torch.sigmoid(gate_hidden.float()).to(dtype=x.dtype)
        return (output * float(lgw) * gate).to(dtype=x.dtype)

    return ref_gated, ref_zero_centered


def cute_zero_centered(x, weight, gate_up_weight, gate_down_weight, eps, lgw):
    """What ZeroCenteredGatedNorm(norm_backend="cute") does: the effective weight goes in as fp32."""
    return cute_gated_norm(x, 1.0 + weight.float(), gate_up_weight, gate_down_weight, eps, lgw)


def golden(x, w, gu, gd, dy, zero_centered):
    """fp64 math on the bf16 inputs, no intermediate rounding."""
    x64, w64, gu64, gd64 = (t.detach().double().requires_grad_() for t in (x, w, gu, gd))
    scale = (1.0 + w64) if zero_centered else w64
    y = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + EPS) * scale
    gate = torch.sigmoid(F.silu(y @ gu64.t()) @ gd64.t())
    out = LGW * y * gate
    grads = torch.autograd.grad(out, (x64, w64, gu64, gd64), dy.double())
    return (out.detach(), *grads)


# ---------------------------------------------------------------------------
# Implementations: each exposes .params (tensors to differentiate) and .forward
# ---------------------------------------------------------------------------
class FunctionalImpl:
    def __init__(self, name, fn, w, gu, gd):
        self.name, self.fn = name, fn
        self.params = [w, gu, gd]

    def forward(self, x):
        return self.fn(x, *self.params, EPS, LGW)


class ModuleImpl:
    """A foundry norm module (bf16 parameters) with the bench parameters copied in."""

    def __init__(self, name, module_cls, hidden, w, gu, gd, **kw):
        self.name = name
        m = module_cls(hidden_size=hidden, eps=EPS, layernorm_gating_weight=LGW,
                       device="cuda", **kw).to(torch.bfloat16)
        with torch.no_grad():
            m.weight.copy_(w)
            m.gate_up_projection.weight.copy_(gu)
            m.gate_down_projection.weight.copy_(gd)
        self.module = m
        self.params = [m.weight, m.gate_up_projection.weight, m.gate_down_projection.weight]

    def forward(self, x):
        return self.module(x)


def _load_foundry():
    try:
        import llmfoundry.models.layers.norm as norm
    except Exception as exc:  # noqa: BLE001 - flash_attn / inductor may be missing; report and go on
        return None, f"{type(exc).__name__}: {exc}"
    return norm, None


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _do_bench(fn):
    from triton.testing import do_bench
    fn()
    torch.cuda.synchronize()
    try:
        return do_bench(fn, warmup=25, rep=200, return_mode="median")
    except TypeError:  # older triton without return_mode
        return do_bench(fn, warmup=25, rep=200)


def bench_impl(impl, x, dy):
    inputs = [x] + impl.params

    def fwd():
        impl.forward(x)

    def fwd_bwd():
        torch.autograd.grad(impl.forward(x), inputs, dy)

    return _do_bench(fwd), _do_bench(fwd_bwd)


def bench_cute_kernels(x, w, gu, gd, dy, zero_centered):
    """gated_norm_fwd / gated_norm_bwd without the autograd graph (kernels + output allocs)."""
    H = x.shape[-1]
    x2d = x.detach().reshape(-1, H).contiguous()
    dy2d = dy.reshape(-1, H).contiguous()
    w_eff = (1.0 + w.detach().float()) if zero_centered else w.detach()
    gu_, gd_ = gu.detach(), gd.detach()
    grad_dtypes = (w.dtype, gu.dtype, gd.dtype)
    _, rho, p, prep = gated_norm_fwd(x2d, w_eff, gu_, gd_, EPS, LGW)
    t_fwd = _do_bench(lambda: gated_norm_fwd(x2d, w_eff, gu_, gd_, EPS, LGW))
    t_bwd = _do_bench(lambda: gated_norm_bwd(dy2d, x2d, rho, p, *prep, LGW, grad_dtypes))
    return t_fwd, t_bwd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def make_inputs(M, H, zero_centered, gate_scale, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, bf = "cuda", torch.bfloat16

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=dev, dtype=torch.float32)

    x = randn(M, H).to(bf).requires_grad_()
    dy = randn(M, H).to(bf)
    w = (0.1 * randn(H) if zero_centered else 1.0 + 0.1 * randn(H)).to(bf).requires_grad_()
    gu = (randn(RANK, H) / math.sqrt(H)).to(bf).requires_grad_()
    gd = (gate_scale * randn(H, RANK) / math.sqrt(RANK)).to(bf).requires_grad_()
    return x, dy, w, gu, gd


def build_impls(variant, H, w, gu, gd, foundry, refs):
    ref_gated, ref_zc = refs
    impls = []
    if variant == "zero-centered":
        impls.append(FunctionalImpl("ref", ref_zc, w, gu, gd))
        impls.append(FunctionalImpl("cute", cute_zero_centered, w, gu, gd))
        if foundry:
            impls.append(ModuleImpl("foundry-ref", foundry.ZeroCenteredGatedNorm, H, w, gu, gd))
            impls.append(ModuleImpl("foundry-cute", foundry.ZeroCenteredGatedNorm, H, w, gu, gd,
                                    norm_backend="cute"))
    else:
        impls.append(FunctionalImpl("ref", ref_gated, w, gu, gd))
        impls.append(FunctionalImpl("cute", cute_gated_norm, w, gu, gd))
        if foundry:
            impls.append(ModuleImpl("foundry-ref", foundry.GatedNorm, H, w, gu, gd))
            impls.append(ModuleImpl("foundry-cute", foundry.GatedNorm, H, w, gu, gd, norm_backend="cute"))
    return impls


def run_perf(args, foundry, refs):
    print("\n" + "=" * 100)
    print(f"PERFORMANCE  (median ms; %peak = minimal traffic / time / {args.peak_gbps:.0f} GB/s)")
    print("=" * 100)
    for variant in args.variants:
        for M, H in args.shapes:
            zc = variant == "zero-centered"
            x, dy, w, gu, gd = make_inputs(M, H, zc, gate_scale=1.0)
            elems = M * H

            def peak(t_ms, bytes_per_elem):
                return bytes_per_elem * elems / (t_ms * 1e-3) / 1e9 / args.peak_gbps * 100

            print(f"\n--- {variant}  M={M} H={H} ---")
            print(f"{'impl':<18}{'fwd':>10}{'fwd+bwd':>10}{'bwd':>10}{'fwd %pk':>10}{'f+b %pk':>10}")
            for impl in build_impls(variant, H, w, gu, gd, foundry, refs):
                try:
                    t_f, t_fb = bench_impl(impl, x, dy)
                except Exception:  # noqa: BLE001
                    print(f"{impl.name:<18} FAILED")
                    traceback.print_exc()
                    continue
                print(f"{impl.name:<18}{t_f:>10.4f}{t_fb:>10.4f}{t_fb - t_f:>10.4f}"
                      f"{peak(t_f, 4):>9.1f}%{peak(t_fb, 10):>9.1f}%")
                del impl
            try:
                t_kf, t_kb = bench_cute_kernels(x, w, gu, gd, dy, zc)
                print(f"{'cute[kernel]':<18}{t_kf:>10.4f}{t_kf + t_kb:>10.4f}{t_kb:>10.4f}"
                      f"{peak(t_kf, 4):>9.1f}%{peak(t_kf + t_kb, 10):>9.1f}%")
            except Exception:  # noqa: BLE001
                print(f"{'cute[kernel]':<18} FAILED")
                traceback.print_exc()
            del x, dy, w, gu, gd
            torch.cuda.empty_cache()


def _rel(a, g):
    a, g = a.double(), g.double()
    return ((a - g).norm() / g.norm().clamp_min(1e-30)).item()


def run_accuracy(args, foundry, refs):
    print("\n" + "=" * 100)
    print("ACCURACY  (relative L2 error vs fp64 golden; ref = foundry torch.compile bf16 baseline)")
    print("=" * 100)
    M, H = args.accuracy_shape
    for variant in args.variants:
        zc = variant == "zero-centered"
        for gate_scale in args.gate_scales:
            x, dy, w, gu, gd = make_inputs(M, H, zc, gate_scale)
            gold = golden(x, w, gu, gd, dy, zc)
            gate = torch.sigmoid(
                F.silu((x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + EPS)
                        * ((1 + w.double()) if zc else w.double())) @ gu.double().t())
                @ gd.double().t())
            frac_small = (gate < 0.05).float().mean().item() * 100
            frac_tiny = (gate < 0.002).float().mean().item() * 100
            del gate
            print(f"\n--- {variant}  M={M} H={H}  gate_scale={gate_scale}  "
                  f"(gates <0.05: {frac_small:.1f}%, <0.002: {frac_tiny:.2f}%) ---")
            print(f"{'impl':<14}{'out':>11}{'dx':>11}{'dweight':>11}{'dgate_up':>11}{'dgate_down':>11}")
            for impl in build_impls(variant, H, w, gu, gd, foundry, refs):
                try:
                    out = impl.forward(x)
                    grads = torch.autograd.grad(out, [x] + impl.params, dy)
                    res = [out.detach().clone()] + [g.clone() for g in grads]
                except Exception:  # noqa: BLE001
                    print(f"{impl.name:<14} FAILED")
                    traceback.print_exc()
                    continue
                errs = [_rel(a, g) for a, g in zip(res, gold)]
                print(f"{impl.name:<14}" + "".join(f"{e:>11.2e}" for e in errs))
                del impl, res, out, grads
            del x, dy, w, gu, gd, gold
            torch.cuda.empty_cache()


def _parse_shapes(s):
    return [tuple(int(v) for v in item.lower().split("x")) for item in s.split(",") if item]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", default="8192x1536,12288x7168,16384x7168",
                   help="comma-separated MxH (the three CuTe-tuned shapes by default)")
    p.add_argument("--variants", default="zero-centered,gated")
    p.add_argument("--accuracy-shape", default="4096x7168")
    p.add_argument("--gate-scales", default="1,4",
                   help="std multiplier of W_down; larger -> more saturated gates")
    p.add_argument("--peak-gbps", type=float, default=3350.0, help="H100 SXM HBM3 peak")
    p.add_argument("--ref-mode", default="max-autotune-no-cudagraphs",
                   help="torch.compile mode for the foundry reference (foundry uses this one)")
    p.add_argument("--skip-perf", action="store_true")
    p.add_argument("--skip-accuracy", action="store_true")
    p.add_argument("--no-foundry", action="store_true",
                   help="do not import llmfoundry.models.layers.norm (skip the foundry-* rows)")
    args = p.parse_args()
    args.shapes = _parse_shapes(args.shapes)
    args.accuracy_shape = _parse_shapes(args.accuracy_shape)[0]
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    args.gate_scales = [float(v) for v in args.gate_scales.split(",") if v]

    assert torch.cuda.is_available(), "needs a CUDA GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")

    foundry = None
    if not args.no_foundry:
        foundry, err = _load_foundry()
        if err:
            print(f"[skip] llmfoundry.models.layers.norm: {err}")

    refs = _make_refs(args.ref_mode)

    if not args.skip_accuracy:
        run_accuracy(args, foundry, refs)
    if not args.skip_perf:
        run_perf(args, foundry, refs)


if __name__ == "__main__":
    main()
