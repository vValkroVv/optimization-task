#!/usr/bin/env python3
"""Exercise the staged owner overlay with two CPU Gloo workers."""

import contextlib
import importlib
import importlib.machinery
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.distributed as dist

from verify_transport import run_reductions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GFSDP = Path("gift/torchtitan/distributed/gfsdp")
PATCH = HERE / "owner-transport.patch"


class Caller:
    def wait_stream(self, _stream):
        pass

    def record_event(self):
        return SimpleNamespace(wait=lambda: None)


def populate(slot, rank, _special):
    for owner, shard in enumerate(slot.chunk(2)):
        index = torch.arange(shard.numel())
        value = ((index * 37 + rank * 11 + owner * 13) % 127 - 63).float()
        shard.copy_(value * (0.03125 if rank == 0 else 1.0))


def add_chunk(own, peer, destination, _n, _block):
    destination.copy_(own + peer.float())


class CpuKernel:
    def __init__(self, _kernel):
        pass

    def __getitem__(self, _grid):
        return add_chunk


def load(staged):
    sys.path.insert(0, str(ROOT / "gift"))
    importlib.import_module("torchtitan.distributed.gfsdp")

    if importlib.util.find_spec("triton") is None:
        triton = ModuleType("triton")
        triton.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)
        triton.language = ModuleType("triton.language")
        triton.language.constexpr = None
        triton.jit = CpuKernel
        triton.cdiv = lambda n, block: (n + block - 1) // block
        sys.modules.update({"triton": triton, "triton.language": triton.language})

    package = ModuleType("gfsdp_owner_cpu")
    package.__path__ = [str(staged / GFSDP), str(ROOT / GFSDP)]
    sys.modules[package.__name__] = package
    parts = [
        importlib.import_module(f"{package.__name__}.{part}")
        for part in ("_collectives", "_owner_transport", "_param_group", "_state")
    ]
    return parts


def expect_error(operation, message):
    try:
        operation()
    except RuntimeError as error:
        if message in str(error):
            return
        raise
    raise AssertionError(f"Expected {message}")


def verify_leases(params):
    cls = params.GFSDPParamGroup
    group = SimpleNamespace(
        _with_fqn=lambda label: label,
        _training_state=params.TrainingState.PRE_BACKWARD,
        _direct_expert_dw_lease=None,
        _owner_transport=object(),
        reduce_grads=True,
    )
    expect_error(lambda: cls.post_backward(group), "complete direct dW lease")

    releases = []
    flags = []
    group.device_handle = SimpleNamespace(current_stream=lambda: None)
    group.grad_buffer_ctx = SimpleNamespace(
        owner=group, release=lambda *_: releases.append(True)
    )
    group.fsdp_params = [
        SimpleNamespace(
            unsharded_param=SimpleNamespace(grad=None), unsharded_accumulated_grad=None
        )
        for _ in range(2)
    ]

    def incomplete():
        raise RuntimeError("incomplete projections")

    group._finish_direct_expert_dw = lambda lease: cls._finish_direct_expert_dw(
        group, lease
    )
    lease = SimpleNamespace(finish=incomplete, abort=lambda: flags.append("aborted"))
    group._direct_expert_dw_lease = lease
    expect_error(lambda: cls.post_backward(group), "incomplete projections")
    assert group._direct_expert_dw_lease is None
    assert releases == [True] and flags == ["aborted"]

    group._direct_expert_dw_lease = lease
    group.fsdp_params[0].unsharded_param.grad = torch.zeros(1)
    expect_error(lambda: cls.post_backward(group), "another gradient")
    assert group._direct_expert_dw_lease is None
    assert releases == [True, True] and flags == ["aborted", "aborted"]


def verify_order(collectives, owner, states, stream, device, group):
    shared_state = states.GFSDPStateContext()
    transport = owner.ExpertOwnerTransport(
        group, device, stream, lambda: shared_state.is_last_backward
    )
    shared_state.owner_transport = transport
    tensor = torch.empty(2 * owner.OWNER_ELEMENTS)
    output = tensor[: owner.OWNER_ELEMENTS]
    expect_error(
        lambda: transport.reduce(owner.FQNS[1], tensor, output, group, stream),
        "order",
    )
    expect_error(transport.finish_backward, "completed 0 of 7")
    peer = tensor.chunk(2)[1 - group.rank()]
    expect_error(
        lambda: transport.reduce(owner.FQNS[0], tensor, peer, group, stream),
        "overlaps the peer input",
    )
    params = [
        SimpleNamespace(sharded_param=SimpleNamespace(requires_grad=True, grad=None))
        for _ in range(2)
    ]
    expect_error(
        lambda: collectives.reduce_scatter(
            params,
            params[:1],
            output,
            tensor,
            group,
            None,
            stream,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            device,
            1.0,
            False,
            False,
            None,
            owner_transport=transport,
            owner_fqn=owner.FQNS[0],
        ),
        "Invalid BF16 owner reduction",
    )

    # The root callback may be queued from a child. It still sees the shared
    # order tracker, including intermediate-only loss backward callbacks.
    state = object.__new__(states.GFSDPState)
    state._state_ctx = shared_state
    state._comm_ctx = SimpleNamespace(post_forward_order=[])
    shared_state.is_last_backward = False
    state._root_post_backward_final_callback()
    expect_error(
        lambda: transport.reduce(owner.FQNS[0], tensor, output, group, stream),
        "order",
    )
    shared_state.is_last_backward = True
    expect_error(state._root_post_backward_final_callback, "completed 0 of 7")
    transport.index = len(owner.FQNS)
    state._root_post_backward_final_callback()
    assert transport.index == 0


def worker(staged, store_path, rank):
    collectives, owner, params, states = load(staged)
    owner.OWNER_ELEMENTS = 64
    owner._add_owner_chunk = CpuKernel(None)
    owner.require_runtime = lambda *_: None
    stream = Caller()
    owner.torch.cuda.stream = lambda _: contextlib.nullcontext()
    collectives._get_device_handle = lambda _: SimpleNamespace(
        stream=lambda _: contextlib.nullcontext(), current_stream=lambda: stream
    )

    def premul(factor):
        assert factor == 1.0
        return dist.ReduceOp.SUM

    torch.distributed._make_nccl_premul_sum = premul
    device = torch.device("cpu")
    store = dist.FileStore(store_path, 2)
    dist.init_process_group(
        "gloo", store=store, rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        group = dist.group.WORLD
        receipt = run_reductions(
            collectives, owner, populate, device, stream, rank, group
        )
        verify_leases(params)
        verify_order(collectives, owner, states, stream, device, group)
        print(receipt, flush=True)
    finally:
        dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory(prefix="ep4-owner-overlay.") as temporary:
        staged = Path(temporary)
        for line in PATCH.read_text().splitlines():
            if line.startswith("--- a/"):
                relative = Path(line.removeprefix("--- a/"))
                copied = staged / relative
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, copied)
        subprocess.run(["git", "apply", "--check", PATCH], cwd=staged, check=True)
        subprocess.run(["git", "apply", PATCH], cwd=staged, check=True)
        subprocess.run(["git", "apply", "-R", "--check", PATCH], cwd=staged, check=True)
        environment = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        if sys.platform == "darwin":
            environment["GLOO_SOCKET_IFNAME"] = "lo0"
        processes = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(__file__),
                    str(staged),
                    str(staged / "store"),
                    str(rank),
                ],
                env=environment,
            )
            for rank in range(2)
        ]
        try:
            statuses = [process.wait(timeout=45) for process in processes]
            if statuses != [0, 0]:
                raise SystemExit(f"CPU overlay worker exit statuses {statuses}")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        worker(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
    else:
        main()
