#!/usr/bin/env python3
"""Exercise the compact overlay with real Gloo traffic and CPU writer doubles."""

import contextlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PATCH = HERE / "owner-bf16-slot.patch"
DEFERRED = HERE.parent / "deferred-dw/deferred-dw.patch"
SHAPES = ((1, 256, 128), (1, 128, 128))
W13 = 32_768
OWNER = 24_576


class Stream:
    def __init__(self, log):
        self.log = log

    def wait_event(self, event):
        self.log.append(("wait_event", event))

    def wait_stream(self, stream):
        self.log.append(("wait_stream", stream))

    def record_event(self):
        event = object()
        self.log.append(("record_event", event))
        return event


class Kernel:
    def __init__(self, _fn=None):
        pass

    def __getitem__(self, _grid):
        def launch(local, remote, output, _n, accumulate, _block):
            reduced = local.float() + remote.float()
            output.copy_(output + reduced if accumulate else reduced)

        return launch


def load():
    # GFSDP decorators initialize Dynamo before installing the local Triton substitute.
    from torchtitan.distributed.gfsdp import _collectives, _state

    if importlib.util.find_spec("triton") is None:
        triton = ModuleType("triton")
        triton.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)
        triton.language = ModuleType("triton.language")
        triton.language.constexpr = None
        triton.jit = Kernel
        triton.cdiv = lambda n, block: (n + block - 1) // block
        sys.modules.update({"triton": triton, "triton.language": triton.language})

    from torchtitan.distributed.gfsdp import _owner_transport
    from torchtitan.ops.float8 import deep_gemm_handle, direct_expert_dw

    _owner_transport.OWNER_ELEMENTS = OWNER
    _owner_transport.OWNER_SHAPES = SHAPES
    _owner_transport._write_owner = Kernel()
    return _collectives, _owner_transport, _state, deep_gemm_handle, direct_expert_dw


def expect_error(operation, message, kind=RuntimeError):
    try:
        operation()
    except kind as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Expected {message}")


def make_group(state, fqn, pair, rank, *, direct=False):
    group = object.__new__(state.GFSDPParamGroup)
    group._module_fqn = fqn
    group.device = torch.device("cpu")
    group.mesh_info = object.__new__(state.DataParallelMeshInfo)
    group.mesh_info.shard_process_group = pair
    group.mesh_info.shard_mesh_size = pair.size()
    group.mesh_info.shard_mesh_rank = pair.rank()
    group._requires_grad = True
    group._direct_expert_dw_requested = direct
    group._direct_expert_dw_lease = None
    group._owner_transport = None
    group._bf16_cache_enabled = False
    group.post_forward_mesh_info = None
    group.reshard_after_backward = False
    group.orig_dtype, group.param_dtype, group.reduce_dtype = (
        torch.float32,
        torch.bfloat16,
        torch.float32,
    )
    group.reduce_grads = True
    group.gradient_divide_factor = 1.0
    group.force_sum_reduction_for_comms = False
    group.mp_policy = SimpleNamespace(bit32_acc_for_bit16_reduce_scatter=False)
    group.data_buffer_ctx = state.GFSDPBufferContext(
        state.GFSDPBufferContext.BufferType.ALL_GATHER
    )
    group.grad_buffer_ctx = state.GFSDPBufferContext(
        state.GFSDPBufferContext.BufferType.REDUCE_SCATTER
    )
    group.padded_unsharded_param_numel = {None: 2 * OWNER}
    group._unsharded_param_numels = {None: (W13, 2 * OWNER - W13)}
    local_lengths = (OWNER, 0) if rank == 0 else (W13 - OWNER, 2 * OWNER - W13)
    group._sharded_param_numels = {None: local_lengths}
    persistent = torch.zeros(OWNER)
    group._padded_sharded_param_grad = {None: persistent}
    group._all_gather_output = {}
    group._reduce_scatter_output = {}
    group._fp8_cache = object() if direct else None
    experts = SimpleNamespace()
    layout = SimpleNamespace(
        global_slices=[SimpleNamespace(shape=shape) for shape in SHAPES],
        local_numel=OWNER,
        global_numel=2 * OWNER,
        receive_sizes_by_dtype=lambda: {None: 8},
    )
    group._select_fp8_weight_cache_layout = lambda: layout if direct else None

    def param(name, shape, view):
        p = SimpleNamespace(
            _module_info=SimpleNamespace(param_name=name, module=experts),
            _orig_size=torch.Size(shape),
            sharded_param=SimpleNamespace(requires_grad=True, grad=None),
            sharded_param_grad=view,
            unsharded_param=SimpleNamespace(grad=None),
            unsharded_accumulated_grad=None,
        )
        p.init_unsharded_param = lambda _, grad: setattr(
            p, "_unsharded_accumulated_grad", grad.view(shape)
        )
        p.init_reduce_scatter_output = lambda output: setattr(
            p, "reduce_scatter_output", output
        )
        return p

    group.fsdp_params = [
        param(name, shape, view)
        for name, shape, view in zip(
            ("w13_E2FD", "w2_EDF"),
            ((1, 2, 128, 128), SHAPES[1]),
            persistent.split(local_lengths),
            strict=True,
        )
    ]
    group._all_gather_dtype_to_fsdp_params = {None: group.fsdp_params}
    return group


def initialize(state, experts, dense, head, stream, *, compact):
    root = state.GFSDPState()
    root._device = torch.device("cpu")
    root._auto_reshard_after_forward = False
    root._shard_order = None
    root._owner_transport_requested = compact
    root._fsdp_param_groups = [*experts, dense]
    root._state_ctx.retain_tail_weights = True
    root._comm_ctx.lazy_init = lambda _: None
    root._comm_ctx.all_gather_stream = root._comm_ctx.reduce_scatter_stream = stream
    head_state = state.GFSDPState()
    head_state._own_grad_buffer = True
    head_state._fsdp_param_groups = [head]
    root._state_ctx.all_states = [root, head_state]
    root._init_shared_state(False, None, None, None, None, None, None)
    return root


def verify_fake(direct):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        slot = torch.empty(2 * OWNER, dtype=torch.bfloat16, device="cuda")
        value = torch.empty(0, dtype=torch.float8_e4m3fn, device="cuda")
        scales = torch.empty((0, 0), device="cuda")
        counts = torch.empty(1, dtype=torch.int32, device="cuda")
        direct.write_direct_expert_dw(
            value, scales, value, scales, slot, 0, None, None, counts
        )
        direct.zero_direct_expert_dw(slot, W13, value)
    for name in ("write_direct_expert_dw", "zero_direct_expert_dw"):
        schema = getattr(torch.ops.torchtitan, name).default._schema
        assert [
            arg.name
            for arg in schema.arguments
            if arg.alias_info and arg.alias_info.is_write
        ] == ["slot"]


def worker(store_file, rank):
    collectives, owner, state, deep_gemm, direct = load()
    dist.init_process_group(
        "gloo",
        store=dist.FileStore(store_file, 2),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        pair = dist.group.WORLD
        dense_pg = dist.new_group([0, 1], backend="gloo")
        log = []
        compute, comm = Stream(log), Stream(log)
        device_handle = SimpleNamespace(
            current_stream=lambda: compute,
            stream=lambda _: contextlib.nullcontext(),
        )
        native_exchange = dist.all_to_all_single

        def export_fixture_pair(group):
            assert group is pair
            return [rank, rank + 4]

        def premul(factor):
            assert factor == 1.0
            return dist.ReduceOp.SUM

        def exchange(receive, send, **kwargs):
            assert send.dtype == receive.dtype == torch.bfloat16
            assert send.untyped_storage().data_ptr() == transport.slot_ptr
            assert (
                send.untyped_storage().data_ptr()
                != receive.untyped_storage().data_ptr()
            )
            assert kwargs["group"] is pair and kwargs["async_op"] is False
            log.append(("exchange", (send.storage_offset(), send.numel())))
            return native_exchange(receive, send, **kwargs)

        def fixture_value(destination):
            flat = destination.view(-1)
            flat.fill_(rank + 0.125 * (microbatch + index + 1))
            if microbatch < 2 and flat.numel() == W13:
                value = ((-(2**24), 0), (2**24, 1))[microbatch][rank]
                flat[0] = flat[OWNER] = value

        def native_writer(_a, _b, destination, ks, _ks_tensor):
            assert destination.dtype == torch.float32
            if sum(ks):
                fixture_value(destination)
            else:
                destination.zero_()

        def stock_writer(_a, _b, destination, ks, _ks_tensor, c):
            assert c is destination and destination.dtype == torch.bfloat16
            assert torch.count_nonzero(destination.view(torch.int16)) == 0
            assert sum(ks) > 0
            log.append(("stock_writer", destination.data_ptr()))
            fixture_value(destination)

        with (
            patch.object(owner, "require_runtime"),
            patch.object(
                dist, "get_process_group_ranks", side_effect=export_fixture_pair
            ),
            patch.object(
                dist, "_make_nccl_premul_sum", side_effect=premul, create=True
            ),
            patch.object(dist, "all_to_all_single", side_effect=exchange),
            patch.object(
                torch.cuda, "stream", side_effect=lambda _: contextlib.nullcontext()
            ),
            patch.object(torch.cuda, "current_stream", side_effect=lambda _: compute),
            patch.object(collectives, "_get_device_handle", return_value=device_handle),
            patch.object(
                direct, "_slot_key", side_effect=lambda slot: (0, slot.data_ptr())
            ),
            patch.object(direct, "_native_writer", native_writer),
            patch.object(
                deep_gemm,
                "_load_deep_gemm",
                return_value=SimpleNamespace(
                    k_grouped_fp8_gemm_nt_contiguous=stock_writer
                ),
            ),
        ):
            outputs = []
            for compact in (False, True):
                experts = [
                    make_group(state, fqn, pair, rank, direct=True)
                    for fqn in owner.FQNS
                ]
                dense = make_group(state, "dense", dense_pg, rank)
                head = make_group(state, "lm_head", dense_pg, rank)
                if compact:
                    wrong = make_group(state, "same-pair-fp32-fallback", pair, rank)
                    with patch.object(
                        state.GFSDPBufferContext, "lazy_init"
                    ) as allocate:
                        expect_error(
                            lambda: initialize(
                                state, experts, wrong, head, comm, compact=True
                            ),
                            "one complete native gradient context",
                        )
                        allocate.assert_not_called()
                        expect_error(
                            lambda: initialize(
                                state, experts[:-1], dense, head, comm, compact=True
                            ),
                            "seven pinned expert groups",
                        )
                        allocate.assert_not_called()
                root = initialize(state, experts, dense, head, comm, compact=compact)
                transport = root._state_ctx.owner_transport
                expert_ctx = experts[0].grad_buffer_ctx
                assert all(group.grad_buffer_ctx is expert_ctx for group in experts)
                assert dense.grad_buffer_ctx is not expert_ctx
                assert dense.grad_buffer_ctx is not head.grad_buffer_ctx
                slot_bytes = expert_ctx.buffers[None].untyped_storage().nbytes()
                assert slot_bytes == 2 * OWNER * (2 if compact else 4)
                assert dense.grad_buffer_ctx.buffers[None].nbytes == 2 * OWNER * 4
                assert head.grad_buffer_ctx.buffers[None].nbytes == 2 * OWNER * 4

                for group in experts:
                    group.device_handle = device_handle
                    group._init_unsharded_params()
                    group._init_reduce_scatter_output(None, group.fsdp_params)
                    # Intercept the CUDA device guard only. These are CPU views.
                    with patch.object(group, "device", torch.device("cuda", rank)):
                        group._init_direct_expert_dw()
                    slot = group._direct_expert_dw_slot
                    assert slot.dtype == (torch.bfloat16 if compact else torch.float32)
                    assert (None in group._reduce_scatter_output) != compact
                    if compact:
                        transport.bind_slot(slot)
                        assert transport.receive.untyped_storage().nbytes() == OWNER

                if compact:
                    root._state_ctx.is_last_backward = False
                    transport.finish_backward()
                    group = experts[0]
                    expect_error(
                        lambda: transport.reduce(
                            owner.FQNS[0],
                            group._direct_expert_dw_slot,
                            group._padded_sharded_param_grad[None],
                            pair,
                            comm,
                            accumulate=False,
                        ),
                        "order",
                    )
                    root._state_ctx.is_last_backward = True
                    group._training_state = state.TrainingState.PRE_BACKWARD
                    expect_error(group.post_backward, "complete direct dW lease")
                    expect_error(
                        lambda: direct.acquire_direct_expert_dw(
                            group._direct_expert_dw_slot, (0, W13), SHAPES
                        ),
                        "outside its lease policy",
                        ValueError,
                    )
                    group._acquire_direct_expert_dw(group._direct_expert_dw_slot)
                    expect_error(group.post_backward, "one write for each projection")
                    assert (
                        group.grad_buffer_ctx.owner
                        is group._direct_expert_dw_lease
                        is None
                    )

                results = []
                for microbatch in range(3):
                    for index, group in enumerate(experts):
                        slot = group._direct_expert_dw_slot
                        previous = expert_ctx.release_event
                        group._training_state = state.TrainingState.PRE_BACKWARD
                        group._acquire_direct_expert_dw(slot)
                        if compact:
                            # Acquire leaves the previous completion wait unsubmitted.
                            assert expert_ctx.release_event is previous
                            start = len(log)
                        rows = 0 if microbatch == 2 and rank == 1 else 128
                        ks = [rows]
                        ks_tensor = torch.tensor(ks, dtype=torch.int32)
                        for offset, (_, m, n) in zip((0, W13), SHAPES, strict=True):
                            if rows == 0 and index % 2 == 0:
                                direct._zero_direct_expert_dw(
                                    slot, offset, torch.empty(0)
                                )
                            else:
                                direct._write_direct_expert_dw(
                                    torch.empty(rows * m),
                                    torch.empty(m, rows // 128),
                                    torch.empty(rows * n),
                                    torch.empty(n, rows // 128),
                                    slot,
                                    offset,
                                    ks,
                                    None,
                                    ks_tensor,
                                )
                        if compact and previous is not None:
                            assert log[start] == ("wait_event", previous)
                            assert log[start:].count(("wait_event", previous)) == 1
                        group.post_backward()
                        assert (
                            expert_ctx.owner is None
                            and expert_ctx.release_event is not None
                        )
                        assert log[-1] == ("record_event", expert_ctx.release_event)
                        for p in group.fsdp_params:
                            assert p.sharded_param.grad is p.sharded_param_grad
                            assert p.sharded_param.grad.dtype == torch.float32
                        results.append(group._padded_sharded_param_grad[None].clone())
                    if compact:
                        transport.finish_backward()
                outputs.append(results)

        for native, compact in zip(*outputs, strict=True):
            assert torch.equal(native.view(torch.int32), compact.view(torch.int32))
        assert outputs[1][len(owner.FQNS)][0].view(torch.int32).item() == 0
        assert len([entry for entry in log if entry[0] == "exchange"]) == 2 * 3 * len(
            owner.FQNS
        )
        verify_fake(direct)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "bitwise_checks": len(outputs[0]),
                    "expert_groups": len(owner.FQNS),
                    "compact_receive_bytes": OWNER,
                    "dense_and_head_separate_fp32": True,
                    "fake_mutation_dispatch": True,
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory(prefix="verify-compact-owner.") as temporary:
        staged = Path(temporary)
        shutil.copytree(ROOT / "gift/torchtitan", staged / "gift/torchtitan")
        test_path = Path("gift/tests/unit_tests/test_direct_expert_dw.py")
        (staged / test_path.parent).mkdir(parents=True)
        shutil.copy2(ROOT / test_path, staged / test_path)
        subprocess.run(["git", "apply", DEFERRED], cwd=staged, check=True)
        affected = {
            staged / line[6:]
            for line in PATCH.read_text().splitlines()
            if line.startswith(("--- a/", "+++ b/"))
        }
        before = {path: path.read_bytes() if path.exists() else None for path in affected}
        for flags in (("--check",), (), ("-R", "--check"), ("-R",)):
            subprocess.run(["git", "apply", *flags, PATCH], cwd=staged, check=True)
        assert before == {
            path: path.read_bytes() if path.exists() else None for path in affected
        }
        subprocess.run(["git", "apply", PATCH], cwd=staged, check=True)
        env = dict(os.environ, PYTHONPATH=str(staged / "gift"), OMP_NUM_THREADS="1")
        if sys.platform == "darwin":
            env["GLOO_SOCKET_IFNAME"] = "lo0"
        processes = [
            subprocess.Popen(
                [sys.executable, "-B", __file__, str(staged / "store"), str(rank)],
                env=env,
            )
            for rank in range(2)
        ]
        try:
            statuses = [process.wait(timeout=75) for process in processes]
            if statuses != [0, 0]:
                raise SystemExit(f"Compact owner CPU worker exit statuses {statuses}")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        worker(sys.argv[1], int(sys.argv[2]))
    else:
        main()
