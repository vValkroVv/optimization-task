# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Gloo parity tests for the sequence-parallel gather / reduce-scatter helpers.

Pins the contract the conformer halo SP (subtask C) relies on:

* ``reduce_scatter_along_dim`` delivers, on rank ``r``, the group SUM of every
  rank's ``r``-th slice along ``dim`` — for any ``dim``, not only ``dim=0``.
  Before the fix it always scattered flat dim-0 chunks: gloo aborted on any
  ``dim != 0`` (shape check), NCCL only checks numel and silently delivered
  flat chunk ``r`` — which coincides with the ``dim`` slice only when every
  dim before ``dim`` has size 1, i.e. scrambled rows for ``[B>1, S, H]``.
* ``gather_from_sequence_parallel_region`` forward == concat of the ranks'
  chunks along dim 1; backward (``tensor_parallel_output_grad=True``) == the
  SUM over ranks of the upstream grad restricted to the rank's own chunk.
* The split-backward mode (``tensor_parallel_output_grad=False``) yields
  local-slice-only grads — right for a replicated loss, the ~1/sp-diagonal
  bug for a sharded one — and ``gather_along_dim`` writes the local chunk in
  place into its input, so a view saved by its producing Function (the
  gigafsdp TyingGate case) fails at backward unless ``.clone()``d first.

The in-place side effects are pinned as observed on gloo; NCCL's copy-in is
autograd-visible too (it raised the production TyingGate error) but is only
verified on the GPU rerun.

Runs as ``torchrun --nproc_per_node=N
sber_tests/unit/cpu/test_sp_gather_parity.py --sp-size S [--backend nccl]`` (default gloo on CPU; nccl puts every tensor
on ``cuda:LOCAL_RANK`` — the GPU gate, since the silent-scramble mode is
NCCL-only); the pytest entry points launch the gloo variants themselves (on
a host without the training stack use ``pytest --noconftest``). The two
modules under test are always loaded by path so ``llmfoundry/__init__`` never
runs; the real ``composer.utils.dist`` is used when importable, otherwise a
minimal shim.
"""

import argparse
import importlib.util
import math
import os
import subprocess
import sys
import types

import pytest
import torch
import torch.distributed as dist

from sber_tests.test_utils.composer_dist_shim import install_composer_dist_shim

pytestmark = [pytest.mark.unit]

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_PARALLEL_DIR = os.path.join(_REPO_ROOT, "llmfoundry", "models", "parallel")

_B, _S, _H = 3, 5, 4


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_modules_under_test():
    """Return ``(parallel.utils, sequence.mappings, composer.utils.dist)``."""
    composer_dist = install_composer_dist_shim()

    for pkg in (
        "llmfoundry",
        "llmfoundry.models",
        "llmfoundry.models.parallel",
        "llmfoundry.models.parallel.sequence",
    ):
        stub = types.ModuleType(pkg)
        stub.__path__ = []
        sys.modules[pkg] = stub
    # utils first: mappings imports it by dotted name from sys.modules.
    utils = _load_by_path(
        "llmfoundry.models.parallel.utils",
        os.path.join(_PARALLEL_DIR, "utils.py"),
    )
    mappings = _load_by_path(
        "llmfoundry.models.parallel.sequence.mappings",
        os.path.join(_PARALLEL_DIR, "sequence", "mappings.py"),
    )
    return utils, mappings, composer_dist


def _make_sp_group(sp_size):
    """Consecutive blocks of ``sp_size`` ranks; every rank creates every group
    (required by new_group) and keeps the one it belongs to."""
    world = dist.get_world_size()
    assert world % sp_size == 0, (world, sp_size)
    my_group = None
    for start in range(0, world, sp_size):
        ranks = list(range(start, start + sp_size))
        group = dist.new_group(ranks=ranks)
        if dist.get_rank() in ranks:
            my_group = group
    return my_group


def _data(local_rank, group_id, shape, seed):
    """Deterministic tensor, every element distinct across ranks and groups
    (sums over <= 8 ranks stay exact integers in fp32)."""
    base = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    return base + 1000.0 * (local_rank + 1) + 100000.0 * (group_id + 1) + seed


def _own_chunk(full, local_rank, dim, chunk):
    return full.narrow(dim, local_rank * chunk, chunk)


class _Ctx:
    """Per-process test context (group, device, deterministic data)."""

    def __init__(self, utils, mappings, composer_dist, sp_size, device):
        self.utils = utils
        self.mappings = mappings
        self.composer_dist = composer_dist
        self.device = device
        self.group = _make_sp_group(sp_size)
        composer_dist._SEQUENCE_PARALLEL_GROUP = self.group
        self.sp = sp_size
        self.rank = self.group.rank()
        self.group_id = dist.get_rank() // sp_size

    def _tensors(self, shape, seed):
        return [
            _data(k, self.group_id, shape, seed).to(self.device) for k in range(self.sp)
        ]

    def inputs(self, shape, seed=0):
        return self._tensors(shape, seed)

    def upstream_grads(self, shape, seed=7):
        return self._tensors(shape, seed)


def _check_reduce_scatter_case(ctx, xs, dim):
    chunk = xs[0].size(dim) // ctx.sp
    expected = _own_chunk(sum(xs), ctx.rank, dim, chunk)
    x = xs[ctx.rank]
    before = x.clone()
    out = ctx.utils.reduce_scatter_along_dim(x, dim, ctx.group)
    assert torch.equal(x, before), "input was mutated"
    assert out.shape == expected.shape, (dim, out.shape, expected.shape)
    assert out.dtype == x.dtype and out.device == x.device
    assert out.is_contiguous(), dim
    assert torch.equal(out, expected), dim


def check_reduce_scatter_direct(ctx):
    """reduce_scatter_along_dim == SUM over ranks, sliced along ``dim`` at the
    rank's own chunk — for dim 0, 1 (B>1: the former bug), last, negative
    (incl. -ndim == 0), a non-contiguous transposed-view input and bf16."""
    sp = ctx.sp
    cases = [
        ((_B, _S * sp, _H), 1),
        ((_B * sp, _S, _H), 0),
        ((_B, _S, _H * sp), 2),
        ((_B, _S, _H * sp), -1),
        ((_B, _S * sp, _H), -2),
        ((_B * sp, _S, _H), -3),
    ]
    for shape, dim in cases:
        _check_reduce_scatter_case(ctx, ctx.inputs(shape), dim)

    xs = [x.transpose(1, 2) for x in ctx.inputs((_B, _S * sp, _H))]
    assert not xs[ctx.rank].is_contiguous()
    _check_reduce_scatter_case(ctx, xs, 2)

    # bf16 with small integers so the sums stay exact up to sp8 (< 256).
    xs = [
        (x.remainder(4) + 4 * k).to(torch.bfloat16)
        for k, x in enumerate(ctx.inputs((_B, _S * sp, _H)))
    ]
    _check_reduce_scatter_case(ctx, xs, 1)

    with pytest.raises(AssertionError, match="divisible"):
        ctx.utils.reduce_scatter_along_dim(
            torch.zeros(_B, _S * sp + 1, _H, device=ctx.device), 1, ctx.group
        )


def check_trivial_group_is_identity(ctx):
    """A size-1 SP group (sp_size == 1 in production) short-circuits both the
    helper and the Function: the input comes back as-is, backward is the
    upstream grad."""
    singleton = None
    for r in range(dist.get_world_size()):
        group = dist.new_group(ranks=[r])
        if r == dist.get_rank():
            singleton = group
    x = ctx.inputs((_B, _S, _H))[ctx.rank]
    assert ctx.utils.reduce_scatter_along_dim(x, 1, singleton) is x

    ctx.composer_dist._SEQUENCE_PARALLEL_GROUP = singleton
    try:
        leaf = x.clone().requires_grad_(True)
        y = ctx.mappings.gather_from_sequence_parallel_region(leaf.clone())
        assert torch.equal(y, x)
        grad = ctx.upstream_grads((_B, _S, _H))[ctx.rank]
        y.backward(grad)
        assert torch.equal(leaf.grad, grad)
    finally:
        ctx.composer_dist._SEQUENCE_PARALLEL_GROUP = ctx.group


def check_gather_forward(ctx):
    """(a) forward == concat of the ranks' chunks along dim 1."""
    xs = ctx.inputs((_B, _S, _H))
    y = ctx.mappings.gather_from_sequence_parallel_region(xs[ctx.rank].clone())
    assert y.shape == (_B, _S * ctx.sp, _H)
    assert torch.equal(y, torch.cat(xs, dim=1))


def _reference_reduce_scatter_grads(xs, grads):
    """Single-process adjoint of the concat: every rank's loss term sees the
    same gathered tensor, so d/dx_k = sum over ranks of that rank's upstream
    grad restricted to chunk k. Shares no code with the helpers under test."""
    leaves = [x.clone().requires_grad_(True) for x in xs]
    torch.cat(leaves, dim=1).backward(sum(grads))
    return [leaf.grad for leaf in leaves]


def check_gather_backward_reduce_scatter(ctx):
    """(b) backward with tensor_parallel_output_grad=True == SUM over ranks of
    the rank's own chunk of each rank's DISTINCT upstream grad."""
    xs = ctx.inputs((_B, _S, _H))
    grads = ctx.upstream_grads((_B, _S * ctx.sp, _H))
    reference = _reference_reduce_scatter_grads(xs, grads)

    x = xs[ctx.rank].clone().requires_grad_(True)
    y = ctx.mappings.gather_from_sequence_parallel_region(x.clone())
    y.backward(grads[ctx.rank])

    assert x.grad.shape == x.shape
    assert torch.equal(x.grad, reference[ctx.rank])
    own_slice = _own_chunk(grads[ctx.rank], ctx.rank, 1, _S)
    assert not torch.equal(x.grad, own_slice), "grads are not discriminating"


def check_split_backward_is_local_slice_only(ctx):
    """(c) regression contrast: with tensor_parallel_output_grad=False the
    backward is the plain split — the rank's own chunk of its OWN upstream
    grad, peers contribute nothing. Correct only when the downstream loss is
    replicated (the DPO callers); with a sharded loss it is the
    local-slice-only ~1/sp gradient the trainable vision adapter suffered
    from."""
    xs = ctx.inputs((_B, _S, _H))
    grads = ctx.upstream_grads((_B, _S * ctx.sp, _H))
    reference = _reference_reduce_scatter_grads(xs, grads)

    x = xs[ctx.rank].clone().requires_grad_(True)
    y = ctx.mappings.gather_from_sequence_parallel_region(x.clone(), False)
    y.backward(grads[ctx.rank])

    assert torch.equal(x.grad, _own_chunk(grads[ctx.rank], ctx.rank, 1, _S))
    assert not torch.equal(x.grad, reference[ctx.rank])


def check_raw_gather_drops_grads(ctx):
    """gather_along_dim called under grad mode (never done in production —
    every caller is inside a Function) is not a differentiable gather: its
    in-place all_gather copy replaces the tracked input's autograd history,
    so nothing reaches the leaf. Forward is still the correct concat."""
    xs = ctx.inputs((_B, _S, _H))
    grads = ctx.upstream_grads((_B, _S * ctx.sp, _H))

    leaf = xs[ctx.rank].clone().requires_grad_(True)
    y = ctx.utils.gather_along_dim(leaf * 1.0, 1, ctx.group)
    assert torch.equal(y, torch.cat(xs, dim=1))
    y.backward(grads[ctx.rank])
    assert torch.equal(leaf.grad, torch.zeros_like(leaf))


class _SavesOutputView(torch.autograd.Function):
    """Stand-in for the gigafsdp TyingGate: a custom Function that returns a
    view of its input and needs that output again in backward."""

    @staticmethod
    def forward(ctx, x):
        out = x.view_as(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        (out,) = ctx.saved_tensors
        return grad * torch.ones_like(out)


def check_inplace_hazard(ctx):
    """(d) the production call shape: gather_from_sequence_parallel_region on
    an encoder output that is a view saved by its producing Function. Un-cloned,
    the all_gather's in-place write bumps the view's version and backward
    fails on the saved-tensor check; the .clone()d view delivers the
    reduce-scatter grads through the producing Function."""
    xs = ctx.inputs((_B, _S, _H))
    grads = ctx.upstream_grads((_B, _S * ctx.sp, _H))
    reference = _reference_reduce_scatter_grads(xs, grads)

    leaf = xs[ctx.rank].clone().requires_grad_(True)
    view = _SavesOutputView.apply(leaf)
    version = view._version
    y = ctx.mappings.gather_from_sequence_parallel_region(view)
    assert view._version == version + 1, "expected the in-place write"
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.backward(grads[ctx.rank])

    leaf = xs[ctx.rank].clone().requires_grad_(True)
    view = _SavesOutputView.apply(leaf)
    y = ctx.mappings.gather_from_sequence_parallel_region(view.clone())
    y.backward(grads[ctx.rank])
    assert torch.equal(leaf.grad, reference[ctx.rank])


CHECKS = [
    check_reduce_scatter_direct,
    check_trivial_group_is_identity,
    check_gather_forward,
    check_gather_backward_reduce_scatter,
    check_split_backward_is_local_slice_only,
    check_raw_gather_drops_grads,
    check_inplace_hazard,
]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-size", type=int, required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    args = parser.parse_args(argv)

    if args.backend == "nccl":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    dist.init_process_group(args.backend)
    utils, mappings, composer_dist = load_modules_under_test()
    ctx = _Ctx(utils, mappings, composer_dist, args.sp_size, device)
    rank = dist.get_rank()
    for check in CHECKS:
        check(ctx)
        dist.barrier()
        print(
            f"[rank {rank} sp={ctx.sp} grp={ctx.group_id}] PASS {check.__name__}",
            flush=True,
        )
    dist.destroy_process_group()


def _sanitized_env():
    drop = {
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
    }
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in drop
        and not k.startswith("TORCHELASTIC_")
        and not k.startswith("PET_")
    }
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_torchrun(nproc, sp_size, backend="gloo"):
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        os.path.abspath(__file__),
        "--sp-size",
        str(sp_size),
        "--backend",
        backend,
    ]
    proc = subprocess.run(
        cmd,
        env=_sanitized_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    for check in CHECKS:
        assert proc.stdout.count(f"PASS {check.__name__}") == nproc, (
            check.__name__,
            proc.stdout,
        )


# (4, 2) is the only variant that proves subgroup isolation: a helper that
# reduces over the world instead of the SP group passes the full-world ones.
@pytest.mark.parametrize("nproc,sp_size", [(2, 2), (4, 2), (4, 4), (8, 8)])
def test_sp_gather_parity(nproc, sp_size):
    """2 ranks / sp2, 4 ranks as two sp2 subgroups, 4 ranks / sp4, 8 / sp8.

    gloo on CPU; on a GPU box run the file directly with ``--backend nccl``.
    """
    _run_torchrun(nproc, sp_size)


if __name__ == "__main__":
    main()
