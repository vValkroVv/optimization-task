"""Checks for the CuTe GatedNorm package (llmfoundry.models.ops.cute_kernels.gated_norm).

Per shape (weight applied as w, i.e. the functional `gated_norm`):
  1. accuracy of out / dx / dweight / dgate_up / dgate_down vs an fp64 golden; with --prev, also
     the max difference to a previous implementation (a file exposing
     gated_norm_cute(x, w, gu, gd, eps, lgw)): 0 where both pick the same H split (e.g.
     M = 4096), tiny otherwise;
  2. determinism: two runs are bitwise identical (all outputs, incl. weight grads);
  3. batch invariance: the rows of an M = 1000 / 3000 call equal the same rows of the full
     call bitwise (out, dx);
  4. GPU time of fwd+bwd replayed from a CUDA graph (and of --prev, if given).

pytest runs 1-3 as tests over SHAPES (accuracy against loose TOLS); the script prints the
full report including the timings:
  python test_gated_norm.py
  python test_gated_norm.py --shapes 4096x7168 --prev /path/to/cute_gn.py
  pytest test_gated_norm.py
"""

import argparse
import importlib.util
import os
import sys
import traceback

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from llmfoundry.models.ops.cute_kernels.gated_norm import gated_norm
    from llmfoundry.models.ops.cute_kernels.gated_norm.bench import (EPS, LGW, _do_bench, _rel,
                                                                     golden, make_inputs)
except ImportError:  # plain `python test_gated_norm.py` from a checkout that is not installed
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, *[".."] * 6)))
    from llmfoundry.models.ops.cute_kernels.gated_norm import gated_norm
    from llmfoundry.models.ops.cute_kernels.gated_norm.bench import (EPS, LGW, _do_bench, _rel,
                                                                     golden, make_inputs)

SHAPES = [(4096, 7168), (4096, 1536), (4096, 512), (8192, 1536), (12288, 7168), (16384, 7168)]
SUB_ROWS = (1000, 3000)                      # batch-invariance sub-batches
M_SWEEP = (1, 63, 64, 65, 1000, 2049, 4097, 8192, 16384)   # partial tiles, every F2 bucket
NAMES = ["out", "dx", "dweight", "dgate_up", "dgate_down"]
# Relative L2 error vs fp64 golden that pytest tolerates. bf16 outputs / bf16 tensor-core
# operands sit around 1e-3; these are loose enough to pass rounding, tight enough to catch a
# wrong formula or tile (O(1)). Tighten once real numbers are on record.
TOLS = {"out": 1e-2, "dx": 1e-2, "dweight": 3e-2, "dgate_up": 3e-2, "dgate_down": 3e-2}


def step(fn, x, params, dy):
    """One fwd+bwd of fn(x, w, gu, gd, eps, lgw); returns [out, dx, dw, dgu, dgd] detached."""
    xs = x.detach().requires_grad_()
    ps = [p.detach().requires_grad_() for p in params]
    out = fn(xs, *ps, EPS, LGW)
    grads = torch.autograd.grad(out, [xs] + ps, dy)
    return [out.detach()] + list(grads)


def graph_ms(fn, x, params, dy):
    """fwd+bwd captured into a CUDA graph, median replay time in ms."""
    xs = x.detach().requires_grad_()
    ps = [p.detach().requires_grad_() for p in params]
    run = lambda: torch.autograd.grad(fn(xs, *ps, EPS, LGW), [xs] + ps, dy)  # noqa: E731
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    return _do_bench(g.replay)


def _load_module(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _inputs(M, H):
    x, dy, w, gu, gd = make_inputs(M, H, zero_centered=False, gate_scale=1.0)
    return x, dy, [w, gu, gd]


# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,H", SHAPES)
def test_accuracy_vs_fp64(M, H):
    x, dy, params = _inputs(M, H)
    res = step(gated_norm, x, params, dy)
    gold = golden(x, *params, dy, zero_centered=False)
    errs = {n: _rel(a, g) for n, a, g in zip(NAMES, res, gold)}
    bad = {n: e for n, e in errs.items() if not e < TOLS[n]}
    assert not bad, f"M={M} H={H}: rel err over tolerance: {bad} (all: {errs})"


@pytest.mark.parametrize("M,H", SHAPES)
def test_deterministic(M, H):
    x, dy, params = _inputs(M, H)
    res = step(gated_norm, x, params, dy)
    res2 = step(gated_norm, x, params, dy)
    differ = [n for n, a, b in zip(NAMES, res, res2) if not torch.equal(a, b)]
    assert not differ, f"M={M} H={H}: not bitwise reproducible: {differ}"


@pytest.mark.parametrize("M,H", [s for s in SHAPES if s[0] > min(SUB_ROWS)])
def test_batch_invariant(M, H):
    x, dy, params = _inputs(M, H)
    res = step(gated_norm, x, params, dy)
    for sub in SUB_ROWS:
        if sub >= M:
            continue
        rs = step(gated_norm, x[:sub], params, dy[:sub])
        assert torch.equal(rs[0], res[0][:sub]), f"M={M} H={H}: out rows differ for M={sub}"
        assert torch.equal(rs[1], res[1][:sub]), f"M={M} H={H}: dx rows differ for M={sub}"


def test_any_m_bounded_caches():
    """Any token count runs (M < 64, partial 64-row tiles) with correct results, and the
    per-M host state stays bounded: one compiled F2 per chunk bucket, one workspace buffer
    per direction, whatever the number of distinct M values (varlen, seq warmup)."""
    from llmfoundry.models.ops.cute_kernels.gated_norm import common, forward
    H = 1536
    n_buckets = len(forward.f2_chunk_options(H, common.num_sms(torch.device("cuda"))))
    plans0, ws0 = len(common.PLANS), len(common._WS)
    for M in M_SWEEP:
        x, dy, params = _inputs(M, H)
        res = step(gated_norm, x, params, dy)
        gold = golden(x, *params, dy, zero_centered=False)
        errs = {n: _rel(a, g) for n, a, g in zip(NAMES, res, gold)}
        bad = {n: e for n, e in errs.items() if not e < TOLS[n]}
        assert not bad, f"M={M} H={H}: rel err over tolerance: {bad} (all: {errs})"
    assert len(common.PLANS) - plans0 <= n_buckets + 1, "compile plans grow with M"
    assert len(common._WS) - ws0 <= 2, "workspace cache grows with M"


# ---------------------------------------------------------------------------
# script: full report
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default=",".join(f"{m}x{h}" for m, h in SHAPES))
    ap.add_argument("--prev", default=None,
                    help="path to a previous implementation exposing gated_norm_cute(...) to diff against")
    args = ap.parse_args()
    if hasattr(torch.autograd.graph, "set_override_stale_capture_stream"):
        torch.autograd.graph.set_override_stale_capture_stream(True)
    prev = _load_module(args.prev, "_t_prev_cute_gn") if args.prev else None

    for shape in args.shapes.split(","):
        M, H = (int(v) for v in shape.split("x"))
        print("\n" + "=" * 100 + f"\nM={M} H={H}")
        x, dy, params = _inputs(M, H)
        try:
            res = step(gated_norm, x, params, dy)
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            print("gated_norm FAILED")
            traceback.print_exc()
            continue

        # 1. accuracy (+ difference to the previous implementation)
        gold = golden(x, *params, dy, zero_centered=False)
        print(f"{'':<22}" + "".join(f"{n:>12}" for n in NAMES))
        print(f"{'rel err vs fp64':<22}" + "".join(f"{_rel(a, g):>12.2e}" for a, g in zip(res, gold)))
        old = None
        if prev is not None:
            old = step(prev.gated_norm_cute, x, params, dy)
            print(f"{'max|cute - prev|':<22}"
                  + "".join(f"{(a.float() - b.float()).abs().max().item():>12.1e}" for a, b in zip(res, old)))
        del gold

        # 2. determinism
        res2 = step(gated_norm, x, params, dy)
        same = all(torch.equal(a, b) for a, b in zip(res, res2))
        print(f"deterministic (2 runs bitwise equal): {same}")

        # 3. batch invariance (per-row outputs)
        for sub in SUB_ROWS:
            if sub >= M:
                continue
            rs = step(gated_norm, x[:sub], params, dy[:sub])
            inv = torch.equal(rs[0], res[0][:sub]) and torch.equal(rs[1], res[1][:sub])
            print(f"batch invariant (rows of M={sub} == rows of M={M}, out and dx bitwise): {inv}")

        # 4. time
        t = graph_ms(gated_norm, x, params, dy)
        line = f"fwd+bwd, CUDA graph: cute {t:.4f} ms"
        if prev is not None:
            line += f"   previous {graph_ms(prev.gated_norm_cute, x, params, dy):.4f} ms"
        print(line)
        del x, dy, res, res2, old
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
