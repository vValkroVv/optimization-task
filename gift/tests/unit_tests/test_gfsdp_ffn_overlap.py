"""Actual GFSDP accumulation and baseline-to-split DCP restoration.

The same test can run on CPU/Gloo or CUDA/NCCL through DTensorTestBase. Small
GQA/dense modules isolate these boundaries; the production FP8/GDN/MLA and
all-eight-rank schedule still require the native crop diagnostic.
"""

import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
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
from torchtitan.distributed.gfsdp._param_group import (
    GFSDPBufferContext,
    GFSDPParamGroup,
)
from torchtitan.distributed.gfsdp._state import _alias_buffer_region
from torchtitan.models.gift_moe._base.parallelize import apply_gfsdp
from torchtitan.models.common.ffn import FeedForward
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
        readers = [
            _alias_buffer_region(regions[0], {None: 65536}),
            regions[1],
        ]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        snapshots = []
        for region in readers:
            region.buffers[None].view(torch.float32).fill_(self.rank + 1)
        torch.cuda.synchronize()
        for region, stream in zip(readers, streams):
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

    @with_comms
    @torch.no_grad()
    def test_mtp_transition_aliases_bounded_existing_regions(self):
        control, _ = self._build(split=True, transition_fixture=True)
        candidate, _ = self._build(
            split=True, transition=True, transition_fixture=True
        )
        assert _buffer_budget(candidate) == _buffer_budget(control)

        root = candidate._get_fsdp_state()
        mtp_ffn = candidate.mtp.layers["0"].block.feed_forward
        source_group = mtp_ffn._get_fsdp_state()._fsdp_param_groups[0]
        source_pg = source_group.mesh_info.shard_process_group
        last_layer = candidate.layers[list(candidate.layers.keys())[-1]]
        data_target_groups = [
            group
            for group in last_layer._get_fsdp_state()._fsdp_param_groups
            if group.mesh_info.shard_process_group is source_pg
        ]
        norm_groups = [
            group
            for group in candidate.norm._get_fsdp_state()._fsdp_param_groups
            if group.mesh_info.shard_process_group is source_pg
        ]
        assert len(data_target_groups) == len(norm_groups) == 1
        data_target = data_target_groups[0]
        grad_target = norm_groups[0]
        assert source_group._post_backward_prefetch_target is data_target

        for target_group, source_ctx, attr, itemsize in (
            (
                data_target,
                source_group.data_buffer_ctx,
                "data_buffer_ctx",
                lambda dtype, group: (dtype or group.param_dtype or group.orig_dtype).itemsize,
            ),
            (
                grad_target,
                source_group.grad_buffer_ctx,
                "grad_buffer_ctx",
                lambda _dtype, group: (
                    group.reduce_dtype or group.param_dtype or group.orig_dtype
                ).itemsize,
            ),
        ):
            target_ctx = getattr(target_group, attr)
            assert source_ctx is not None
            assert target_ctx._storage_ctx is source_ctx._storage_ctx
            assert source_ctx.overlaps(target_ctx)
            for dtype, target_view in target_ctx.buffers.items():
                source_start, source_end = source_ctx._ranges[dtype]
                target_start, target_end = target_ctx._ranges[dtype]
                alignment = (
                    target_group.mesh_info.shard_mesh_size
                    * target_group._shard_alignment
                    * itemsize(dtype, target_group)
                )
                assert source_start <= target_start < target_end <= source_end
                assert target_view.storage_offset() == target_start
                assert target_start % alignment == 0
                assert (target_end - target_start) % alignment == 0
                assert (
                    target_view.untyped_storage().data_ptr()
                    == source_ctx.buffers[dtype].untyped_storage().data_ptr()
                )
                assert (
                    target_view.untyped_storage().nbytes()
                    == source_ctx._storage_ctx.buffers[dtype].numel()
                )

        with pytest.raises(RuntimeError, match="before the first forward"):
            candidate.set_buffer_region_transition(
                source=mtp_ffn,
                data_target=last_layer,
                grad_target=candidate.norm,
            )

        original_source_forward = source_group.post_forward_mesh_info
        original_target_forward = data_target.post_forward_mesh_info
        source_group.post_forward_mesh_info = None
        data_target.post_forward_mesh_info = None
        with pytest.raises(ValueError, match="share a data buffer"):
            root.validate_shared_state()
        source_group.post_forward_mesh_info = original_source_forward
        data_target.post_forward_mesh_info = original_target_forward

        source_group.reshard_after_backward = False
        with pytest.raises(ValueError, match="source must reshard after backward"):
            root.validate_shared_state()
        source_group.reshard_after_backward = True

        # Isolate the target-specific guard from the generic overlap check if
        # the FFN is configured to retain its forward data region.
        source_group.post_forward_mesh_info = original_target_forward
        data_target.post_forward_mesh_info = None
        with pytest.raises(ValueError, match="target must reshard after forward"):
            root.validate_shared_state()
        data_target.post_forward_mesh_info = original_target_forward
        source_group.post_forward_mesh_info = original_source_forward
        data_target.reshard_after_backward = False
        with pytest.raises(ValueError, match="target must reshard after backward"):
            root.validate_shared_state()
        data_target.reshard_after_backward = True

        source_group.reduce_grads = False
        root.validate_shared_state()
        grad_target.reduce_grads = False
        with pytest.raises(ValueError, match="no grad reduce"):
            root.validate_shared_state()
        source_group.reduce_grads = True
        grad_target.reduce_grads = True

    @with_comms
    @torch.no_grad()
    def test_parent_child_prefetch_does_not_duplicate_all_gather(self):
        import torchtitan.distributed.gfsdp._param_group as param_group_module

        model, _ = self._build(split=True)
        parent = model.mtp.layers["0"]
        ffn = parent.block.feed_forward
        parent_group = parent._get_fsdp_state()._fsdp_param_groups[0]
        ffn_group = ffn._get_fsdp_state()._fsdp_param_groups[0]
        original_all_gather = param_group_module.all_gather
        calls = []

        def count_all_gather(group, *args, **kwargs):
            calls.append(group)
            return original_all_gather(group, *args, **kwargs)

        with patch.object(param_group_module, "all_gather", count_all_gather):
            # The parent prefetch and the later generic child prefetch may
            # request the FFN more than once without enqueuing another gather.
            GFSDPParamGroup._prefetch_unshard(parent_group, "backward")
            GFSDPParamGroup._prefetch_unshard(ffn_group, "backward")
            GFSDPParamGroup._prefetch_unshard(ffn_group, "backward")
            parent_group.wait_for_unshard()
            ffn_group.wait_for_unshard()

        assert calls.count(parent_group) == len(
            parent_group._all_gather_dtype_to_fsdp_params
        )
        assert calls.count(ffn_group) == len(ffn_group._all_gather_dtype_to_fsdp_params)

    def _build(self, split, transition=False, transition_fixture=False):
        torch._dynamo.reset()
        torch.manual_seed(2048)
        model = _make_tiny_decoder(1, n_layers=2).to(self.device_type)
        if transition_fixture:
            mtp_ffn = model.mtp.layers["0"].block.feed_forward
            model.mtp.layers["0"].block.feed_forward = FeedForward.Config(
                dim=mtp_ffn.w1.in_features,
                hidden_dim=256,
            ).build()
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
            mtp_transition_prefetch=transition,
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
        baseline, baseline_optimizer = self._build(
            False, transition_fixture=True
        )
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
            candidate, candidate_optimizer = self._build(
                True, transition=True, transition_fixture=True
            )
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
