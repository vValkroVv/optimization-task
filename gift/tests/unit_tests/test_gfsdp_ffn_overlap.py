"""Actual GFSDP accumulation and baseline-to-split DCP restoration.

The same test can run on CPU/Gloo or CUDA/NCCL through DTensorTestBase. Small
GQA/dense modules isolate these boundaries; the production FP8/GDN/MLA and
all-eight-rank schedule still require the native crop diagnostic.
"""

import shutil
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, CompileConfig
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.ffn_overlap import enable_ffn_overlap
from torchtitan.distributed.gfsdp._dcp import GFSDPCheckpointAdapter
from torchtitan.distributed.gfsdp._param_group import GFSDPBufferContext
from torchtitan.models.gift_moe._base.parallelize import apply_gfsdp
from tests.unit_tests.test_mtp import _make_tiny_decoder


def _canonical(name):
    return name.replace("_checkpoint_wrapped_module.", "")


@torch.no_grad()
def _full_parameters(model):
    return {
        _canonical(name): param.full_tensor().clone()
        for name, param in model.named_parameters()
    }


@torch.no_grad()
def _full_optimizer(optimizer):
    result = {}
    for name, value in optimizer.state_dict().items():
        if isinstance(value, torch.Tensor):
            result[name] = (
                value.full_tensor() if hasattr(value, "full_tensor") else value
            ).clone()
        else:
            result[name] = value
    return result


def _assert_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=key)
        else:
            assert left[key] == right[key], key


def _buffer_budget(model):
    storages = {}
    for state in model._get_fsdp_state()._state_ctx.all_states:
        for group in state._fsdp_param_groups:
            for kind, ctx in (
                ("data", group.data_buffer_ctx),
                ("grad", group.grad_buffer_ctx),
            ):
                if ctx is not None:
                    for tensor in ctx.buffers.values():
                        storage = tensor.untyped_storage()
                        storages[kind, storage.data_ptr()] = storage.nbytes()
    return sorted((kind, size) for (kind, _), size in storages.items())


@torch.no_grad()
def _assert_region_layout(parent, child):
    for attr in ("data_buffer_ctx", "grad_buffer_ctx"):
        left, right = getattr(parent, attr), getattr(child, attr)
        assert left._storage_ctx is right._storage_ctx
        assert not left.overlaps(right)
        for dtype in left.buffers.keys() & right.buffers.keys():
            lhs, rhs = left.buffers[dtype], right.buffers[dtype]
            assert lhs.storage_offset() == 0
            assert rhs.storage_offset() == lhs.numel()
    for group in (parent, child):
        world_size = group.mesh_info.shard_mesh_size
        for dtype, padded in group.padded_unsharded_param_numel.items():
            assert padded % (world_size * group._shard_alignment) == 0
            assert group._all_gather_output[dtype].numel() == padded
            assert group._reduce_scatter_input[dtype].numel() == padded
            assert group._all_gather_input[dtype].numel() == padded // world_size
            assert (
                group._padded_sharded_param_grad[dtype].numel() == padded // world_size
            )
            output = group._reduce_scatter_output[dtype]
            expected = group._reduce_scatter_input[dtype].chunk(world_size)[
                group.mesh_info.shard_mesh_rank
            ]
            assert output.data_ptr() == expected.data_ptr()
            assert output.numel() == expected.numel()
            for param in group._all_gather_dtype_to_fsdp_params[dtype]:
                # Region offsets do not alter each flat group's ragged shard
                # metadata, local parameter order, or collective rank slicing.
                assert param.sharded_param.to_local().numel() == param.sharded_numel
                assert param._sharded_param_grad_data.numel() == param.sharded_numel


class TestGFSDPFFNOverlap(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    @torch.no_grad()
    def test_region_budget_and_collective_rank_layout(self):
        baseline, _ = self._build(False)
        candidate, _ = self._build(True)
        assert _buffer_budget(candidate) == _buffer_budget(baseline)
        expected = _full_parameters(candidate)
        for unit in (candidate.layers["0"], candidate.mtp.layers["0"]):
            block = getattr(unit, "block", unit)
            (parent,) = unit._get_fsdp_state()._fsdp_param_groups
            (ffn,) = block.feed_forward._get_fsdp_state()._fsdp_param_groups
            _assert_region_layout(parent, ffn)
            # Exercise the actual GFSDP gather and reduction consumers, including
            # ragged local shards and padding, without CPU autograd's shared-
            # view version checks (Gloo writes bump versions unlike NCCL).
            for group in (parent, ffn):
                group.unshard()
                group.wait_for_unshard()
                for param in group.fsdp_params:
                    torch.testing.assert_close(
                        param.unsharded_param,
                        expected[_canonical(param._param_fqn)].to(torch.bfloat16),
                        rtol=0,
                        atol=0,
                    )
                    param.unsharded_param.grad = torch.full_like(
                        param.unsharded_param, self.rank + 1
                    )
                group.post_backward()
                group._wait_for_post_backward()
                for param in group.fsdp_params:
                    torch.testing.assert_close(
                        param.sharded_param.grad.to_local(),
                        torch.full_like(param._sharded_param_grad_data, 3),
                        rtol=0,
                        atol=0,
                    )

    @with_comms
    def test_whole_slot_waits_for_pending_region_collectives(self):
        if self.device_type != "cuda":
            self.skipTest("Requires real CUDA streams and NCCL asynchronous work")
        ctx = GFSDPBufferContext(GFSDPBufferContext.BufferType.REDUCE_SCATTER)
        ctx.lazy_init({None: 262144}, torch.device("cuda", self.rank))
        regions = [ctx.region({None: 131072}, {None: offset}) for offset in (0, 131072)]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        snapshots = []
        for region in regions:
            region.buffers[None].view(torch.float32).fill_(self.rank + 1)
        torch.cuda.synchronize()
        for region, stream in zip(regions, streams):
            with torch.cuda.stream(stream):
                # Keep actual collective readers pending while the whole-slot
                # writer is submitted on the default stream.
                torch.cuda._sleep(20000000)
                values = region.buffers[None].view(torch.float32)
                dist.all_reduce(values, async_op=True).wait()
                snapshots.append(values.clone())
                region.release_event = stream.record_event()
        ctx.wait_for_release(torch.cuda.current_stream())
        ctx.buffers[None].zero_()
        torch.cuda.synchronize()
        for snapshot in snapshots:
            torch.testing.assert_close(
                snapshot, torch.full_like(snapshot, 3), rtol=0, atol=0
            )

    def _build(self, split):
        torch._dynamo.reset()
        torch.manual_seed(2048)
        model = _make_tiny_decoder(1, n_layers=2).to(self.device_type)
        model.init_weights(buffer_device=torch.device(self.device_type))
        if split:
            enable_ffn_overlap(model.layers["0"])
            enable_ffn_overlap(model.mtp.layers["0"])
        apply_ac(
            model,
            ActivationCheckpointConfig(mode="selective", selective_save=[]),
            model_compile_enabled=True,
        )
        # AOTAutograd executes the checkpoint partition and its backward;
        # local Inductor parity is covered separately in test_ffn_overlap.py.
        apply_compile(model, CompileConfig(enable=True, backend="aot_eager"))
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        apply_gfsdp(
            model,
            mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )
        model.train()
        model.lazy_init()
        if self.device_type == "cpu":
            # Gloo cannot execute NCCL's PREMUL_SUM reduction. Keep factor=1
            # and all arithmetic unchanged using its supported SUM path.
            for state in model._get_fsdp_state()._state_ctx.all_states:
                for group in state._fsdp_param_groups:
                    group.force_sum_reduction_for_comms = True
        model.setup_ema_buffer()
        model.update_ema_buffer_after_load(force_update=True)
        optimizer = OptimizersContainer(
            OptimizersContainer.Config(implementation="fused"), model_parts=[model]
        )
        print(
            {
                "rank": self.rank,
                "device": self.device_type,
                "backend": dist.get_backend(),
                "split": split,
                "unique_pool_storages": _buffer_budget(model),
                "unique_pool_bytes": sum(size for _, size in _buffer_budget(model)),
            },
            flush=True,
        )
        return model, optimizer

    def _update(self, model, optimizer):
        optimizer.zero_grad()
        if self.device_type == "cuda":
            torch.cuda.synchronize()
            start_allocated = torch.cuda.memory_allocated()
            start_reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
        for microbatch in range(32):
            ids = (
                (torch.arange(3, device=self.device_type) + microbatch + self.rank) % 16
            ).view(1, 3)
            hidden = model(ids)
            logits = model.lm_head(hidden)
            loss = (
                logits[0].float().square().mean()
                + 0.3 * logits[1].float().square().mean()
            ) / 32
            loss.backward()
        if self.device_type == "cuda":
            torch.cuda.synchronize()
            print(
                {
                    "rank": self.rank,
                    "phase": "32mb_forward_backward_before_gradient_export",
                    "split": hasattr(model.layers["0"], "_ffn_overlap_regions"),
                    "start_allocated": start_allocated,
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                    "incremental_peak_allocated": torch.cuda.max_memory_allocated()
                    - start_allocated,
                    "start_reserved": start_reserved,
                    "peak_reserved": torch.cuda.max_memory_reserved(),
                    "scope": "tiny BF16 AOT/SAC; DCP test roots coexist; not fullcrop peak parity",
                },
                flush=True,
            )
        gradients = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                assert param.grad is not None
                assert param.grad.dtype is torch.float32
                gradients[_canonical(name)] = param.grad.full_tensor().clone()
        optimizer.step()
        model.update_ema_params(0.9)
        return gradients

    @with_comms
    def test_baseline_checkpoint_loads_split_layout_and_continues(self):
        baseline, baseline_optimizer = self._build(False)
        self._update(baseline, baseline_optimizer)
        original_parameters = _full_parameters(baseline)
        original_optimizer = _full_optimizer(baseline_optimizer)
        baseline.swap_ema_and_weights()
        original_ema = _full_parameters(baseline)
        baseline.swap_ema_and_weights()
        adapter = GFSDPCheckpointAdapter([baseline])
        path_holder = [
            str(
                Path(__file__).resolve().parents[3]
                / ".tmp"
                / f"ffn-overlap-dcp-{uuid.uuid4().hex}"
            )
            if self.rank == 0
            else None
        ]
        dist.broadcast_object_list(path_holder)
        checkpoint_path = path_holder[0]
        try:
            dcp.save(
                {
                    **adapter.model_save_state_dict(),
                    "optimizer": adapter.wrap_optimizer_state(baseline_optimizer),
                },
                checkpoint_id=checkpoint_path,
            )
            candidate, candidate_optimizer = self._build(True)
            assert _buffer_budget(candidate) == _buffer_budget(baseline)
            for unit in (candidate.layers["0"], candidate.mtp.layers["0"]):
                block = getattr(unit, "block", unit)
                (parent_group,) = unit._get_fsdp_state()._fsdp_param_groups
                (ffn_group,) = block.feed_forward._get_fsdp_state()._fsdp_param_groups
                _assert_region_layout(parent_group, ffn_group)
            candidate_adapter = GFSDPCheckpointAdapter([candidate])
            candidate_adapter.load(
                {
                    "optimizer": candidate_adapter.wrap_optimizer_state(
                        candidate_optimizer
                    )
                },
                checkpoint_id=checkpoint_path,
                process_group=None,
                load_model=True,
            )
            _assert_equal(original_parameters, _full_parameters(candidate))
            _assert_equal(original_optimizer, _full_optimizer(candidate_optimizer))
            candidate.swap_ema_and_weights()
            _assert_equal(original_ema, _full_parameters(candidate))
            candidate.swap_ema_and_weights()
            # Identical BF16 primitives under AOT eager are expected to match;
            # this assertion does not prescribe FP8/Inductor tolerances.
            expected_gradients = self._update(baseline, baseline_optimizer)
            events = []
            for label, unit in (
                ("layer0", candidate.layers["0"]),
                ("mtp", candidate.mtp.layers["0"]),
            ):
                block = getattr(unit, "block", unit)
                for suffix, owner in (("parent", unit), ("ffn", block.feed_forward)):
                    (group,) = owner._get_fsdp_state()._fsdp_param_groups
                    original = group.post_backward

                    def observed(
                        *args,
                        group=group,
                        original=original,
                        label=label,
                        suffix=suffix,
                    ):
                        ready = [
                            param.unsharded_param.grad is not None
                            for param in group.fsdp_params
                        ]
                        if any(ready):
                            assert all(ready)
                            events.append((label, suffix))
                        return original(*args)

                    group.post_backward = observed
            actual_gradients = self._update(candidate, candidate_optimizer)
            assert (
                events
                == [
                    ("mtp", "ffn"),
                    ("mtp", "parent"),
                    ("layer0", "ffn"),
                    ("layer0", "parent"),
                ]
                * 32
            )
            _assert_equal(expected_gradients, actual_gradients)
            _assert_equal(_full_parameters(baseline), _full_parameters(candidate))
            _assert_equal(
                _full_optimizer(baseline_optimizer),
                _full_optimizer(candidate_optimizer),
            )
            baseline.swap_ema_and_weights()
            candidate.swap_ema_and_weights()
            _assert_equal(_full_parameters(baseline), _full_parameters(candidate))
            baseline.swap_ema_and_weights()
            candidate.swap_ema_and_weights()
            for unit in (candidate.layers["0"], candidate.mtp.layers["0"]):
                block = getattr(unit, "block", unit)
                (parent_group,) = unit._get_fsdp_state()._fsdp_param_groups
                (ffn_group,) = block.feed_forward._get_fsdp_state()._fsdp_param_groups
                assert parent_group.data_buffer_ctx is not ffn_group.data_buffer_ctx
                assert parent_group.grad_buffer_ctx is not ffn_group.grad_buffer_ctx
        finally:
            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(checkpoint_path)
