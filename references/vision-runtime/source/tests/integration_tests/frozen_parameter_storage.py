"""Distributed frozen-storage contracts for FSDP2 and GFSDP2."""

import argparse
import copy
from collections import Counter
import os
import sys
import unittest
from unittest.mock import patch
from contextlib import contextmanager, ExitStack
from tempfile import TemporaryDirectory
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.distributed.gfsdp import (
    fully_shard as g_fully_shard,
    GFSDPModule,
    MixedPrecisionPolicy as GMixedPrecisionPolicy,
)
from torchtitan.distributed.gfsdp._dcp import GFSDPCheckpointAdapter
from torchtitan.trainer import cast_frozen_parameters
import torchtitan.distributed.gfsdp._param_group as param_group_module


class Block(nn.Module):
    def __init__(self, mixed):
        super().__init__()
        self.base = nn.Linear(64, 64).requires_grad_(mixed == "trainable")
        if mixed is True or mixed == "mixed":
            self.adapter = nn.Linear(64, 64, bias=False)
        else:
            self.adapter = None

    def forward(self, x):
        y = self.base(x)
        if self.adapter is not None:
            y = y + 0.1 * self.adapter(x)
        return x + 0.1 * torch.tanh(y)


class Model(nn.Module):
    def __init__(self, mixed):
        super().__init__()
        self.input = nn.Linear(64, 64)
        block_modes = mixed if isinstance(mixed, tuple) else (mixed,) * 3
        self.blocks = nn.ModuleList(Block(mode) for mode in block_modes)
        self.output = nn.Linear(64, 16)
        self.register_buffer("fp32_buffer", torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def _full(tensor):
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def _build(initial, mixed, backend, mesh, reshard, ac, dtype, expert_mesh=None):
    model = Model(mixed).cuda()
    for param in model.parameters():
        param.data = param.data.to(dtype)
    cast_frozen_parameters(model)
    model.load_state_dict(initial)
    if ac:
        model.blocks = nn.ModuleList(
            checkpoint_wrapper(block) for block in model.blocks
        )
    shard = g_fully_shard if backend == "gfsdp2" else fully_shard
    policy_cls = GMixedPrecisionPolicy if backend == "gfsdp2" else MixedPrecisionPolicy
    policy = policy_cls(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    units = [model.input, *model.blocks, model.output]
    for unit in units:
        kwargs = {}
        if expert_mesh is not None and unit in model.blocks:
            block = getattr(unit, "_checkpoint_wrapped_module", unit)
            expert_params = {block.base.weight}
            if block.adapter is not None:
                expert_params.add(block.adapter.weight)
            kwargs["param_meshes"] = [(expert_mesh, expert_params)]
        shard(
            unit, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard, **kwargs
        )
    shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard)
    if backend == "gfsdp2":
        model.set_shard_order(units)
    for current, following in zip(units, units[1:]):
        current.set_modules_to_forward_prefetch([following])
        following.set_modules_to_backward_prefetch([current])
    return model


def _state(model, backend):
    if backend == "gfsdp2":
        return GFSDPCheckpointAdapter([model]).model_save_state_dict()
    return get_model_state_dict(model)


def _load(model, backend, path):
    if backend == "gfsdp2":
        GFSDPCheckpointAdapter([model]).load(
            {}, checkpoint_id=str(path), process_group=None, load_model=True
        )
    else:
        state = get_model_state_dict(model)
        dcp.load(state, checkpoint_id=path)
        set_model_state_dict(model, state)


@torch.no_grad()
def _assert_storage(model, dtype):
    frozen_bytes = 0
    for name, param in model.named_parameters():
        expected = dtype if param.requires_grad else torch.bfloat16
        assert param.dtype == expected, (name, param.dtype, expected)
        if not param.requires_grad:
            assert param.grad is None, name
            local = param.to_local()
            frozen_bytes += local.numel() * local.element_size()
    assert model.fp32_buffer.dtype == torch.float32
    for module in model.modules():
        if isinstance(module, GFSDPModule):
            for group in module._get_fsdp_state()._fsdp_param_groups:
                if not group.requires_grad:
                    assert group.grad_buffer_ctx is None
                    assert all(
                        value is None
                        for value in group._padded_sharded_param_grad.values()
                    )
                    assert all(
                        value.dtype == torch.bfloat16
                        for value in group._padded_sharded_param_data.values()
                    )
    return frozen_bytes


@contextmanager
def _checkpoint_dir(scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        paths = [None]
        if dist.get_rank() == 0:
            paths[0] = stack.enter_context(
                TemporaryDirectory(prefix="frozen-storage-", dir=scratch)
            )
        dist.broadcast_object_list(paths, src=0)
        yield Path(paths[0])
        dist.barrier()


def _fixture(
    mixed, backend, mesh, reshard=True, ac=False, dtype=torch.float32, expert_mesh=None
):
    torch.manual_seed(42)
    source = Model(mixed)
    initial = copy.deepcopy(source.state_dict())
    reference = copy.deepcopy(source).cuda()
    for param in reference.parameters():
        param.data = param.data.to(dtype)
    cast_frozen_parameters(reference)
    reference.load_state_dict(initial)
    model = _build(initial, mixed, backend, mesh, reshard, ac, dtype, expert_mesh)
    return initial, reference, model


def _assert_model_matches(model, reference):
    state = get_model_state_dict(model)
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(_full(state[name]), expected, rtol=0, atol=0)


def _train_and_compare(model, reference, dtype, checkpoint=None, backend=None):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    ref_optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=1e-3
    )
    trainable_before = {
        name: param.detach().clone()
        for name, param in reference.named_parameters()
        if param.requires_grad
    }
    frozen_before = {
        name: _full(param).detach().clone()
        for name, param in model.named_parameters()
        if not param.requires_grad
    }
    expected_bytes = sum(
        param.numel() * 2 // param.device_mesh.size()
        for param in model.parameters()
        if not param.requires_grad
    )
    for step in range(10):
        torch.manual_seed(100 + step)
        x = torch.randn(4, 64, device="cuda", requires_grad=True)
        xr = x.detach().clone().requires_grad_(True)
        target = torch.randn(4, 16, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (model(x).float() - target).square().mean()
            ref_loss = (reference(xr).float() - target).square().mean()
        torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
        loss.backward()
        ref_loss.backward()
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
        actual = {
            name.replace("._checkpoint_wrapped_module", ""): param
            for name, param in model.named_parameters()
        }
        for name, expected in reference.named_parameters():
            candidate = actual[name]
            if expected.requires_grad:
                torch.testing.assert_close(
                    _full(candidate.grad), expected.grad, rtol=0, atol=0
                )
            else:
                assert candidate.grad is None
        optimizer.step()
        ref_optimizer.step()
        assert _assert_storage(model, dtype) == expected_bytes
        assert all(p.requires_grad for p in optimizer.state)
        for param, state in optimizer.state.items():
            assert state["exp_avg"].dtype == param.dtype
            assert state["exp_avg_sq"].dtype == param.dtype
        if checkpoint is not None and step == 4:
            dcp.save(_state(model, backend), checkpoint_id=checkpoint)
            state_path = checkpoint / f"optimizer-rank{dist.get_rank()}.pt"
            torch.save(optimizer.state_dict(), state_path)
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(1)
            _load(model, backend, checkpoint)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=2e-3
            )
            assert not optimizer.state
            optimizer.load_state_dict(torch.load(state_path, weights_only=False))
            assert optimizer.param_groups[0]["lr"] == 1e-3
            assert len(optimizer.state) == sum(
                p.requires_grad for p in model.parameters()
            )
            assert all(state["step"].item() == 5 for state in optimizer.state.values())
            _assert_storage(model, dtype)
    _assert_model_matches(model, reference)
    assert any(
        not torch.equal(param, trainable_before[name])
        for name, param in reference.named_parameters()
        if param.requires_grad
    )
    for name, previous in frozen_before.items():
        torch.testing.assert_close(
            _full(dict(model.named_parameters())[name]), previous, rtol=0, atol=0
        )


def _assert_forward_lifecycle(model, reference):
    # PP warmup can queue several forwards before any backward. Consume them
    # out of order too, so retention cannot rely on reverse execution order.
    for iteration in range(2):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        inputs, ref_inputs, losses, ref_losses = [], [], [], []
        for microbatch in range(3):
            torch.manual_seed(200 + 3 * iteration + microbatch)
            x = torch.randn(4, 64, device="cuda", requires_grad=True)
            xr = x.detach().clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                actual = model(x)
                expected = reference(xr)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            inputs.append(x)
            ref_inputs.append(xr)
            losses.append(actual.float().square().mean())
            ref_losses.append(expected.float().square().mean())
        for index, microbatch in enumerate((1, 2, 0)):
            model.set_is_last_backward(index == 2)
            losses[microbatch].backward()
            ref_losses[microbatch].backward()
            torch.testing.assert_close(
                inputs[microbatch].grad, ref_inputs[microbatch].grad, rtol=0, atol=0
            )
        actual_params = {
            name.replace("._checkpoint_wrapped_module", ""): param
            for name, param in model.named_parameters()
        }
        for name, expected in reference.named_parameters():
            actual = actual_params[name]
            if expected.requires_grad:
                torch.testing.assert_close(
                    _full(actual.grad), expected.grad, rtol=0, atol=0
                )
            else:
                assert actual.grad is None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x in inputs:
                torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
        for module in model.modules():
            if isinstance(module, GFSDPModule):
                for group in module._get_fsdp_state()._fsdp_param_groups:
                    assert group.data_buffer_ctx.owner is None


@instantiate_parametrized_tests
class TestFrozenParameterStorageDistributed(unittest.TestCase):
    checkpoint_root: Path

    @property
    def world_size(self):
        return dist.get_world_size()

    def _mesh(self):
        return init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("fsdp",))

    @parametrize("dtype,trainable", [(torch.float32, True), (torch.bfloat16, False)])
    @torch.no_grad()
    def test_gfsdp2_constructor_memory_and_packed_values(self, dtype, trainable):
        mesh = self._mesh()
        model = nn.Module()
        shapes = [(3,), (2048, 2048), (0,), (5,)]
        for index, shape in enumerate(shapes):
            model.register_parameter(
                f"weight_{index}",
                nn.Parameter(
                    torch.full(shape, index + 1, dtype=dtype, device="cuda"),
                    requires_grad=trainable,
                ),
            )
        model.alias = nn.Module()
        model.alias.weight = model.weight_1
        total_numel = sum(param.numel() for param in model.parameters())
        shard_numel = (
            (total_numel + self.world_size * 8 - 1) // (self.world_size * 8) * 8
        )
        local_bytes = shard_numel * dtype.itemsize
        torch.cuda.synchronize()
        initial_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        g_fully_shard(model, mesh=mesh)
        torch.cuda.synchronize()
        construction_peak = torch.cuda.max_memory_allocated() - initial_allocated
        # Storage, optional gradients and the largest temporary local parameter
        # fit within three local shards. No allocation scales with the full unit.
        self.assertLessEqual(construction_peak, 3 * local_bytes + 1024**2)
        self.assertIs(model.weight_1, model.alias.weight)
        for index, shape in enumerate(shapes):
            param = getattr(model, f"weight_{index}")
            torch.testing.assert_close(
                _full(param).cpu(),
                torch.full(shape, index + 1, dtype=dtype),
                rtol=0,
                atol=0,
            )

    @parametrize("backend", ["fsdp2", "gfsdp2"])
    @parametrize(
        "mixed,reshard,ac,dtype",
        [
            (False, True, False, torch.float32),
            (True, True, False, torch.float32),
            (True, True, True, torch.float32),
            (True, False, True, torch.float32),
            (True, True, False, torch.bfloat16),
        ],
    )
    def test_numerics_gradients_and_storage(self, backend, mixed, reshard, ac, dtype):
        _, reference, model = _fixture(mixed, backend, self._mesh(), reshard, ac, dtype)
        _train_and_compare(model, reference, dtype)

    @parametrize("reshard", [True, None])
    @parametrize("ac", [False, True])
    @parametrize(
        "modes,expected_contexts",
        [
            (("frozen",) * 6 + ("trainable",) * 2, 2),
            (("frozen", "mixed", "trainable", "mixed"), 3),
        ],
    )
    def test_gfsdp2_data_pool_reuse(self, reshard, ac, modes, expected_contexts):
        _, reference, model = _fixture(
            modes, "gfsdp2", self._mesh(), reshard=reshard, ac=ac
        )
        units = [model.input, *model.blocks, model.output]
        for unit in units:
            unit.set_modules_to_backward_prefetch([])
        # Ten updates verify frozen middle-layer input gradients, full AC and
        # implicit backward / explicit forward prefetch against a reference.
        gathers = Counter()
        all_gather = param_group_module.all_gather

        def count_all_gather(group, *args, **kwargs):
            gathers[group] += 1
            return all_gather(group, *args, **kwargs)

        with patch.object(param_group_module, "all_gather", count_all_gather):
            _train_and_compare(model, reference, torch.float32)
        for index, unit in enumerate(units):
            expected = 10 if reshard is None and index >= len(units) - 2 else 20
            for group in unit._get_fsdp_state()._fsdp_param_groups:
                self.assertEqual(gathers[group], expected)
        if reshard is None:
            _assert_forward_lifecycle(model, reference)
        contexts = [
            [
                group.data_buffer_ctx
                for group in unit._get_fsdp_state()._fsdp_param_groups
            ]
            for unit in units
        ]
        self.assertEqual(
            len({ctx for unit in contexts for ctx in unit}), expected_contexts
        )
        for current, following in zip(contexts, contexts[1:]):
            self.assertEqual(len(set(current + following)), len(current + following))

    def test_gfsdp2_separate_parameter_mesh(self):
        mesh = self._mesh()
        expert_mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("efsdp", "ep"))[
            "efsdp"
        ]
        _, reference, model = _fixture(
            True, "gfsdp2", mesh, ac=True, expert_mesh=expert_mesh
        )
        with _checkpoint_dir(self.checkpoint_root) as checkpoint:
            _train_and_compare(
                model, reference, torch.float32, checkpoint=checkpoint, backend="gfsdp2"
            )

    @parametrize(
        "backend,separate_mesh",
        [("fsdp2", False), ("gfsdp2", False), ("gfsdp2", True)],
    )
    def test_old_fp32_checkpoint_load(self, backend, separate_mesh):
        mesh = self._mesh()
        expert_mesh = (
            init_device_mesh("cuda", (2, 2), mesh_dim_names=("efsdp", "ep"))["efsdp"]
            if separate_mesh
            else None
        )
        initial, reference, model = _fixture(
            True, backend, mesh, expert_mesh=expert_mesh
        )
        old_state = {
            name: value.flatten()
            if backend == "gfsdp2" and name != "fp32_buffer"
            else value
            for name, value in initial.items()
        }
        with _checkpoint_dir(self.checkpoint_root) as checkpoint:
            dcp.save(old_state, checkpoint_id=checkpoint)
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(1)
                model.fp32_buffer.zero_()
            _load(model, backend, checkpoint)
            _assert_model_matches(model, reference)
            _assert_storage(model, torch.float32)

    @parametrize("backend", ["fsdp2", "gfsdp2"])
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_checkpoint_resume(self, backend, dtype):
        _, reference, model = _fixture(
            True, backend, self._mesh(), ac=True, dtype=dtype
        )
        with _checkpoint_dir(self.checkpoint_root) as checkpoint:
            _train_and_compare(
                model, reference, dtype, checkpoint=checkpoint, backend=backend
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_folder", type=Path, required=True)
    args = parser.parse_args()
    TestFrozenParameterStorageDistributed.checkpoint_root = args.dump_folder

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        if dist.get_world_size() != 4:
            raise ValueError("Frozen storage integration requires four ranks")
        result = unittest.main(argv=[sys.argv[0]], verbosity=2, exit=False)
        if not result.result.wasSuccessful():
            raise SystemExit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
