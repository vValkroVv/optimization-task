"""Focused owner routing/state checks; NCCL is an explicit command-line opt-in.

CPU: PYTHONPATH=gift pytest -q gift/tests/unit_tests/test_row_owner_embedding.py
GPU (only after authorization): PYTHONPATH=gift torchrun --nproc-per-node=8 \
    gift/tests/unit_tests/test_row_owner_embedding.py --backend nccl --checkpoint-dir PATH
The GPU route runs the same actual owner and mixed GFSDP state paths; it does
not replace the model's native compile/SAC and production-shape gates.
"""

import argparse
from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.distributed.backends.gfsdp2 import BackendBufferStore
from torchtitan.distributed.gfsdp import fully_shard
from torchtitan.distributed.gfsdp._dcp import (
    GFSDPCheckpointAdapter,
    GFSDPOptimizerCheckpointState,
    _FlatCheckpointTensor,
)
from torchtitan.distributed.gfsdp._state import _get_module_fsdp_state
from torchtitan.distributed.gfsdp._tensor import RaggedShardDTensor
from torchtitan.distributed.gfsdp.row_owner_embedding import RowOwnerEmbedding
from torchtitan.distributed.utils import clip_grad_norm_
from torchtitan.config.schema.protocols.module import Ones
from torchtitan.models.common.embeddings.token_embeddings import Embedding
from torchtitan.protocols.module import Module


@pytest.fixture(autouse=True)
def _hide_cuda_in_cpu_tests(monkeypatch):
    # DCP's CPU writer otherwise selects an available CUDA stream even for
    # CPU payloads. Fresh spawned workers must see no GPUs during CPU checks.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


class _Model(nn.Module):
    def __init__(self, mesh, *, owner, optimization_stages=None):
        super().__init__()
        embedding = nn.Embedding(16, 8, device=mesh.device_type)
        self.tok_embeddings = (
            RowOwnerEmbedding.from_embedding(
                embedding, mesh, optimization_stages=optimization_stages
            )
            if owner
            else embedding
        )
        # An existing non-embedding GFSDP bucket supplies enough scratch,
        # analogous to the unchanged private lm_head bucket in the crop.
        self.probe = nn.Linear(8, 128, bias=False, device=mesh.device_type)
        ignored = {self.tok_embeddings.weight} if owner else None
        if not owner:
            fully_shard(self.tok_embeddings, mesh=mesh)
        fully_shard(self.probe, mesh=mesh, ignored_params=ignored)
        fully_shard(self, mesh=mesh, ignored_params=ignored)
        if owner:
            self.tok_embeddings.attach_comm_context(self)
        _get_module_fsdp_state(self)._lazy_init()

    def forward(self, ids):
        return self.probe(self.tok_embeddings(ids).float())


class _OptimizerState:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        return get_optimizer_state_dict(
            self.model,
            self.optimizer,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

    def load_state_dict(self, state):
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            state,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


def _full_weight(device):
    return (torch.arange(128, dtype=torch.float32, device=device) / 17).view(16, 8)


def _local_slice(full, owner):
    return full[owner.row_start : owner.row_start + owner.rows_per_owner].reshape(-1)


def _source_sum(ids, derivative, owner):
    source = torch.ops.aten.embedding_dense_backward.default(
        derivative.contiguous(),
        ids,
        16,
        -1,
        False,
    ).float()
    sources = [torch.empty_like(source) for _ in range(dist.get_world_size())]
    dist.all_gather(sources, source)
    # Explicit rank order is the implementation's FP32 SUM contract. A real
    # NCCL reduce-scatter may use a different FP32 tree; CUDA tests compare
    # the local BF16 boundary exactly, not a universal NCCL bitwise claim.
    total = torch.zeros_like(source)
    for contribution in sources:
        total.add_(contribution)
    return _local_slice(total, owner)


def _routing_checks(mesh, optimization_stages=None):
    model = _Model(mesh, owner=True, optimization_stages=optimization_stages)
    owner = model.tok_embeddings
    rank = dist.get_rank()
    device = owner.weight.device
    owner.weight._local_tensor.copy_(_local_slice(_full_weight(device), owner))
    parameter = owner.weight
    owner.to_empty(device=device)
    assert owner.weight is parameter
    owner.weight._local_tensor.copy_(_local_slice(_full_weight(device), owner))
    # Increasing later counts, two overlapping live invocations, original
    # ordering, boundaries, duplicates across requesters, empty and skew cases.
    cases = [
        torch.tensor([0], device=device),
        torch.tensor([15, 0, 3, 3, 15, rank % 16], device=device),
        torch.arange(16, device=device),
        torch.empty(0, dtype=torch.int64, device=device)
        if rank
        else torch.tensor([1, 1, 1], device=device),
        torch.empty(0, dtype=torch.int64, device=device),
    ]
    for ids in cases:
        owner.weight.grad = None
        first = owner(ids)
        second_ids = ids.flip(0)
        second = owner(second_ids)
        # The saved route owns its bytes through unrelated reuse, including
        # multiple live invocations followed by all their backwards.
        with owner._scratch(4096) as scratch:
            scratch.fill_(91)
        torch.testing.assert_close(
            first, _full_weight(device).bfloat16()[ids], rtol=0, atol=0
        )
        upstream = (
            torch.arange(first.numel(), device=device).reshape(first.shape) / 7
        ).bfloat16()
        other = torch.full_like(second, 0.3)
        # Two consumers stand in for the shared trunk/MTP derivative; the
        # exact incoming BF16 rounding is produced by ordinary autograd.
        leaf = first.detach().requires_grad_()
        torch.autograd.backward(
            (leaf * upstream, leaf * other),
            (torch.ones_like(first), torch.ones_like(first)),
        )
        combined = leaf.grad
        torch.autograd.backward(
            (first * upstream, first * other, second * other),
            (torch.ones_like(first), torch.ones_like(first), torch.ones_like(second)),
        )
        expected = _source_sum(ids, combined, owner) + _source_sum(
            second_ids, other, owner
        )
        torch.testing.assert_close(
            owner.weight.grad._local_tensor, expected, rtol=0, atol=0
        )
        _assert_gradient_not_borrowed(owner)
    # This case rejects occurrence gradients widened before source-local
    # BF16 rounding, even though that alternative would look more precise.
    ids = torch.tensor([0, 0], device=device)
    derivative = (
        torch.tensor([1.0, 2.0**-9], device=device, dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(2, 8)
        .contiguous()
    )
    native = torch.ops.aten.embedding_dense_backward.default(
        derivative, ids, 16, -1, False
    )
    wrong = torch.zeros((16, 8), device=device).index_add_(0, ids, derivative.float())
    assert not torch.equal(native.float(), wrong)
    owner.weight.grad = None
    owner(ids).backward(derivative)
    torch.testing.assert_close(
        owner.weight.grad._local_tensor,
        _source_sum(ids, derivative, owner),
        rtol=0,
        atol=0,
    )
    ids = torch.tensor([0, 0, 15, rank % 16], device=device)
    owner.weight.grad = None
    expected = torch.zeros_like(owner.weight._local_tensor)
    accumulated_pointer = None
    for microbatch in range(32):
        output = owner(ids)
        derivative = torch.full_like(output, (microbatch + 1) / 32)
        output.backward(derivative)
        pointer = owner.weight.grad._local_tensor.data_ptr()
        if accumulated_pointer is None:
            accumulated_pointer = pointer
        assert pointer == accumulated_pointer
        _assert_gradient_not_borrowed(owner)
        expected.add_(_source_sum(ids, derivative, owner))
    torch.testing.assert_close(
        owner.weight.grad._local_tensor, expected, rtol=0, atol=0
    )
    output = owner(ids)
    before = owner.weight.grad._local_tensor.clone()
    output.backward(torch.zeros_like(output))
    torch.testing.assert_close(owner.weight.grad._local_tensor, before, rtol=0, atol=0)
    # Every scratch byte can be reused after backward, including while a
    # persistent accumulated gradient exists. zero_grad(False) keeps storage.
    with owner._scratch(64) as scratch:
        scratch.fill_(173)
    torch.testing.assert_close(owner.weight.grad._local_tensor, before, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, fused=True)
    optimizer.zero_grad(set_to_none=False)
    output = owner(ids)
    derivative = torch.full_like(output, 0.25)
    output.backward(derivative)
    assert owner.weight.grad._local_tensor.data_ptr() == accumulated_pointer
    torch.testing.assert_close(
        owner.weight.grad._local_tensor,
        _source_sum(ids, derivative, owner),
        rtol=0,
        atol=0,
    )
    optimizer.zero_grad(set_to_none=True)
    assert owner.weight.grad is None
    owner(ids).backward(derivative)
    _assert_gradient_not_borrowed(owner)
    torch.testing.assert_close(
        owner.weight.grad._local_tensor,
        _source_sum(ids, derivative, owner),
        rtol=0,
        atol=0,
    )
    with torch.no_grad():
        torch.testing.assert_close(
            owner(ids), _full_weight(device).bfloat16()[ids], rtol=0, atol=0
        )
    for bad in (
        -1,
        16,
        torch.iinfo(torch.int64).min,
        torch.iinfo(torch.int64).max,
        "dtype",
        "device",
    ):
        invalid = torch.tensor([0], dtype=torch.int64, device=device)
        if rank == 0:
            if bad == "dtype":
                invalid = invalid.float()
            elif bad == "device":
                # Use meta to exercise metadata mismatch without a GPU copy.
                invalid = torch.empty(1, dtype=torch.int64, device="meta")
            else:
                invalid.fill_(bad)
        with pytest.raises(ValueError, match="on every rank"):
            owner(invalid)
    model.set_requires_gradient_sync(False)
    with pytest.raises(RuntimeError, match="no-sync"):
        owner(ids)
    model.set_requires_gradient_sync(True)

    # A direct VJP returns an owned Ragged gradient without changing .grad.
    owner.weight.grad = None
    output = owner(ids)
    derivative = torch.full_like(output, 1.25)
    expected = _source_sum(ids, derivative, owner)
    (gradient,) = torch.autograd.grad(
        output,
        owner.weight,
        derivative,
    )
    assert isinstance(gradient, RaggedShardDTensor)
    assert owner.weight.grad is None
    torch.testing.assert_close(gradient._local_tensor, expected, rtol=0, atol=0)
    _assert_gradient_not_borrowed(owner, gradient)

    existing_local = torch.full_like(owner.weight._local_tensor, 0.125)
    existing_grad = RaggedShardDTensor(
        existing_local, owner.weight._spec, requires_grad=False
    )
    owner.weight.grad = existing_grad
    output = owner(ids)
    (gradient,) = torch.autograd.grad(output.sum(), owner.weight)
    torch.testing.assert_close(
        gradient._local_tensor,
        _source_sum(ids, torch.ones_like(output), owner),
        rtol=0,
        atol=0,
    )
    assert owner.weight.grad is existing_grad
    torch.testing.assert_close(
        owner.weight.grad._local_tensor,
        torch.full_like(owner.weight._local_tensor, 0.125),
        rtol=0,
        atol=0,
    )
    _assert_gradient_not_borrowed(owner, gradient)

    observed_hook = {}

    def scale_parameter_gradient(value):
        assert isinstance(value, RaggedShardDTensor)
        observed_hook["gradient"] = value._local_tensor.clone()
        return RaggedShardDTensor(
            value._local_tensor * 2, value._spec, requires_grad=False
        )

    owner.weight.grad = None
    handle = owner.weight.register_hook(scale_parameter_gradient)
    try:
        output = owner(ids)
        output.sum().backward()
    finally:
        handle.remove()
    expected = _source_sum(ids, torch.ones_like(output), owner)
    torch.testing.assert_close(observed_hook["gradient"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        owner.weight.grad._local_tensor, expected * 2, rtol=0, atol=0
    )
    _assert_gradient_not_borrowed(owner)

    post_accumulate = {}

    def observe_post_accumulate(parameter):
        post_accumulate["gradient"] = parameter.grad._local_tensor.clone()

    owner.weight.grad = None
    handle = owner.weight.register_post_accumulate_grad_hook(observe_post_accumulate)
    try:
        output = owner(ids)
        output.sum().backward()
    finally:
        handle.remove()
    expected = _source_sum(ids, torch.ones_like(output), owner)
    torch.testing.assert_close(post_accumulate["gradient"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        owner.weight.grad._local_tensor, expected, rtol=0, atol=0
    )
    _assert_gradient_not_borrowed(owner)


def _assert_gradient_not_borrowed(owner, gradient=None):
    gradient = owner.weight.grad if gradient is None else gradient
    local = gradient._local_tensor
    for group in owner._sync_groups:
        if group.grad_buffer_ctx is not None:
            for buffer in group.grad_buffer_ctx.buffers.values():
                assert (
                    local.untyped_storage().data_ptr()
                    != buffer.untyped_storage().data_ptr()
                )


def _borrow_checks(mesh):
    model = _Model(mesh, owner=True)
    owner = model.tok_embeddings
    group = next(
        group for group in owner._sync_groups if group.grad_buffer_ctx is not None
    )
    context = group.grad_buffer_ctx._storage_ctx
    key, buffer = max(context.buffers.items(), key=lambda item: item[1].numel())
    storage = buffer.untyped_storage()
    inventory = (storage.data_ptr(), storage.nbytes())
    with owner._scratch(64) as scratch:
        assert scratch.untyped_storage().data_ptr() == inventory[0]
        scratch.fill_(19)
        assert context.conflicting_owner(group) is context
        with pytest.raises(RuntimeError, match="live owner"):
            with owner._scratch(64):
                pass
    assert context.owner is None
    assert (
        buffer.untyped_storage().data_ptr(),
        buffer.untyped_storage().nbytes(),
    ) == inventory
    region = context.region({key: 64}, {key: 0})
    region.owner = group
    with pytest.raises(RuntimeError, match="live owner"):
        with owner._scratch(64):
            pass
    region.owner = None
    with pytest.raises(RuntimeError, match="capacity"):
        with owner._scratch(storage.nbytes() + 1):
            pass
    if mesh.device_type != "cuda":
        return
    # Pending NCCL readers on another stream must finish before overwrite.
    reader, writer = torch.cuda.Stream(), torch.cuda.Stream()
    values = buffer.view(torch.float32)
    values.fill_(dist.get_rank() + 1)
    reader.wait_event(torch.cuda.current_stream().record_event())
    with torch.cuda.stream(reader):
        torch.cuda._sleep(20000000)
        dist.all_reduce(values, async_op=True).wait()
        snapshot = values.clone()
        pending = reader.record_event()
    context.release_event = pending
    group._post_reduce_event = pending
    with torch.cuda.stream(writer):
        with owner._scratch(buffer.numel()) as scratch:
            scratch.zero_()
        released = context.release_event
    # The root's older reduction completion must not erase the borrower's
    # newer write event, which a subsequent GFSDP reader still needs.
    group._wait_for_post_backward()
    assert context.release_event is released
    context.wait_for_release(torch.cuda.current_stream())
    after = values.clone()
    torch.cuda.synchronize()
    expected = dist.get_world_size() * (dist.get_world_size() + 1) // 2
    torch.testing.assert_close(
        snapshot, torch.full_like(snapshot, expected), rtol=0, atol=0
    )
    torch.testing.assert_close(after, torch.zeros_like(after), rtol=0, atol=0)


def _seed_gradients(model, value):
    for param in model.parameters():
        param.grad = torch.zeros_like(param).fill_(value)


def _state_checks(mesh, directory, optimization_stages=None):
    # The baseline saves flat GFSDP table/moments. The owner must load that
    # exact format and retain FQN, EMA, dense optimizer state and next update.
    baseline = _Model(mesh, owner=False)
    baseline_store = BackendBufferStore([baseline])
    baseline_adapter = GFSDPCheckpointAdapter([baseline])
    baseline_opt = torch.optim.AdamW(
        baseline.parameters(), lr=0.01, weight_decay=0.1, fused=True
    )
    for param in baseline.parameters():
        param._local_tensor.fill_(0.75)
    _seed_gradients(baseline, 0.4)
    baseline_opt.step()
    baseline_store.update(0.2)
    optim = GFSDPOptimizerCheckpointState(_OptimizerState(baseline, baseline_opt))
    checkpoint = str(directory) + "/baseline"
    dcp.save(
        {**baseline_adapter.model_save_state_dict(), "optimizer": optim},
        checkpoint_id=checkpoint,
    )
    for key, value in baseline_adapter.model_save_state_dict().items():
        if key == "tok_embeddings.weight":
            assert isinstance(value, _FlatCheckpointTensor) and value.shape == (128,)
    owner_model = _Model(mesh, owner=True, optimization_stages=optimization_stages)
    owner = owner_model.tok_embeddings
    owner_store = BackendBufferStore([owner_model])
    adapter = GFSDPCheckpointAdapter([owner_model])
    optimizer = torch.optim.AdamW(
        owner_model.parameters(), lr=0.01, weight_decay=0.1, fused=True
    )
    # Populate actual AdamW moments so DCP exposes load targets.
    _seed_gradients(owner_model, 0.0)
    optimizer.step()
    owner_optim = GFSDPOptimizerCheckpointState(_OptimizerState(owner_model, optimizer))
    adapter.load(
        {"optimizer": owner_optim},
        checkpoint_id=checkpoint,
        process_group=None,
        load_model=True,
    )
    for base, candidate in zip(
        baseline.parameters(), owner_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            base._local_tensor, candidate._local_tensor, rtol=0, atol=0
        )
        for moment in ("exp_avg", "exp_avg_sq", "step"):
            base_state = baseline_opt.state[base][moment]
            candidate_state = optimizer.state[candidate][moment]
            if isinstance(base_state, RaggedShardDTensor):
                base_state, candidate_state = (
                    base_state._local_tensor,
                    candidate_state._local_tensor,
                )
            torch.testing.assert_close(base_state, candidate_state, rtol=0, atol=0)
    # Mixed GFSDP-managed probe + excluded owner Ragged parameter clipping.
    _seed_gradients(baseline, 0.25)
    _seed_gradients(owner_model, 0.25)
    base_norm = clip_grad_norm_(baseline.parameters(), 0.5, ep_enabled=True)
    owner_norm = clip_grad_norm_(owner_model.parameters(), 0.5, ep_enabled=True)
    torch.testing.assert_close(base_norm, owner_norm, rtol=0, atol=0)
    baseline_opt.step()
    optimizer.step()
    for base, candidate in zip(
        baseline.parameters(), owner_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            base._local_tensor, candidate._local_tensor, rtol=0, atol=0
        )
    # Dense zero gradients still update untouched rows through old momentum
    # and weight decay. Compare with the original dense fused AdamW path.
    before = owner.weight._local_tensor.clone()
    _seed_gradients(baseline, 0.0)
    _seed_gradients(owner_model, 0.0)
    baseline_opt.step()
    optimizer.step()
    assert not torch.equal(before, owner.weight._local_tensor)
    for base, candidate in zip(
        baseline.parameters(), owner_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            base._local_tensor, candidate._local_tensor, rtol=0, atol=0
        )
    ids = torch.tensor([0, 15, 0], device=owner.weight.device)
    live = owner(ids).detach()
    previous_ema = owner.checkpoint_tensor(ema=True)._local_tensor.clone()
    expected_ema = previous_ema.lerp(owner.weight._local_tensor, 0.35)
    owner_store.update(0.35)
    ema = owner.checkpoint_tensor(ema=True)._local_tensor.clone()
    torch.testing.assert_close(ema, expected_ema, rtol=0, atol=0)
    main = owner.weight._local_tensor.clone()
    identity = owner.weight
    owner_store.swap()
    assert owner.weight is identity
    torch.testing.assert_close(owner.weight._local_tensor, ema, rtol=0, atol=0)
    # Fresh BF16 lookup follows the swapped backing, with no row cache.
    swapped = owner(ids)
    pieces = [torch.empty_like(ema) for _ in range(dist.get_world_size())]
    dist.all_gather(pieces, ema)
    expected_lookup = torch.cat(pieces).view(16, 8).bfloat16()[ids]
    torch.testing.assert_close(swapped, expected_lookup, rtol=0, atol=0)
    # Checkpoint main stays LIVE even during a temporary EMA evaluation swap.
    torch.testing.assert_close(
        adapter.model_save_state_dict()["tok_embeddings.weight"]._local_tensor,
        main,
        rtol=0,
        atol=0,
    )
    owner_store.swap()
    torch.testing.assert_close(owner(ids), live, rtol=0, atol=0)
    roundtrip = str(directory) + "/owner"
    dcp.save(
        {**adapter.model_save_state_dict(), "optimizer": owner_optim},
        checkpoint_id=roundtrip,
    )
    owner.weight._local_tensor.zero_()
    owner._ema_weight.zero_()
    optimizer.state[owner.weight]["exp_avg"]._local_tensor.zero_()
    optimizer.state[owner.weight]["exp_avg_sq"]._local_tensor.zero_()
    adapter.load(
        {"optimizer": owner_optim},
        checkpoint_id=roundtrip,
        process_group=None,
        load_model=True,
    )
    torch.testing.assert_close(owner.weight._local_tensor, main, rtol=0, atol=0)
    torch.testing.assert_close(owner._ema_weight, ema, rtol=0, atol=0)
    _seed_gradients(baseline, 0.125)
    _seed_gradients(owner_model, 0.125)
    baseline_opt.step()
    optimizer.step()
    for base, candidate in zip(
        baseline.parameters(), owner_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            base._local_tensor, candidate._local_tensor, rtol=0, atol=0
        )
    main = owner.weight._local_tensor.clone()
    # Old checkpoints lacking EMA must seed from loaded LIVE owner values.
    missing = str(directory) + "/missing-ema"
    old_state = adapter.model_save_state_dict()
    old_state.pop("ema_model")
    dcp.save(old_state, checkpoint_id=missing)
    owner.weight._local_tensor.zero_()
    owner._ema_weight.fill_(-3)
    adapter.load({}, checkpoint_id=missing, process_group=None, load_model=True)
    torch.testing.assert_close(owner._ema_weight, main, rtol=0, atol=0)


def _worker(rank, world, directory, backend="gloo", optimization_stages=None):
    torch.set_num_threads(1)
    device = "cuda" if backend == "nccl" else "cpu"
    if device == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        init_method="file://" + str(directory) + "/rendezvous",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        mesh = init_device_mesh(device, (world,), mesh_dim_names=("dp_shard",))
        print(
            f"row_owner_checks rank={rank} backend={dist.get_backend()} "
            f"mesh_device={mesh.device_type} cuda_initialized={torch.cuda.is_initialized()}",
            flush=True,
        )
        if device == "cpu":
            assert not torch.cuda.is_initialized()
        _borrow_checks(mesh)
        _routing_checks(mesh, optimization_stages)
        _state_checks(mesh, directory, optimization_stages)
        if device == "cpu":
            assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_cpu_gloo_routing_accumulation_and_state(tmp_path):
    mp.spawn(_worker, args=(2, str(tmp_path)), nprocs=2, join=True)


def test_unsupported_embedding_options_are_guarded(tmp_path):
    dist.init_process_group(
        "gloo", init_method="file://" + str(tmp_path / "single"), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))
        for kwargs in (
            {"sparse": True},
            {"padding_idx": 0},
            {"max_norm": 1.0},
            {"scale_grad_by_freq": True},
        ):
            with pytest.raises(ValueError, match="dense, unscaled"):
                RowOwnerEmbedding.from_embedding(nn.Embedding(16, 8, **kwargs), mesh)
        with pytest.raises(ValueError, match="FP32"):
            RowOwnerEmbedding.from_embedding(
                nn.Embedding(16, 8, dtype=torch.bfloat16), mesh
            )
        with pytest.raises(ValueError, match="aligned rows"):
            RowOwnerEmbedding.from_embedding(nn.Embedding(3, 3), mesh)
        with torch.device("meta"):
            original = Embedding.Config(
                num_embeddings=16, embedding_dim=8, param_init={"weight": Ones()}
            ).build()
        owner = RowOwnerEmbedding.from_embedding(original, mesh)
        assert owner.optimization_stages == ()
        assert RowOwnerEmbedding.from_embedding(
            original, mesh, optimize=True
        ).optimization_stages == (
            "compact",
            "owner_add",
            "owner_lookup",
            "routing",
        )
        assert RowOwnerEmbedding.from_embedding(
            original, mesh, optimization_stages=("routing", "compact")
        ).optimization_stages == ("compact", "routing")
        with pytest.raises(ValueError, match="unknown row-owner optimization"):
            RowOwnerEmbedding.from_embedding(
                original, mesh, optimization_stages=("unknown",)
            )
        root = Module()
        root.tok_embeddings = owner
        identity = owner.weight
        root.to_empty(device="cpu")
        root.init_states()
        assert owner.weight is identity and owner.weight.device.type == "cpu"
        torch.testing.assert_close(
            owner.weight._local_tensor, torch.ones(128), rtol=0, atol=0
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("nccl",))
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--optimize-embeddings", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    _worker(
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["WORLD_SIZE"]),
        args.checkpoint_dir,
        args.backend,
        (
            ("compact", "owner_add", "owner_lookup", "routing")
            if args.optimize_embeddings
            else None
        ),
    )
