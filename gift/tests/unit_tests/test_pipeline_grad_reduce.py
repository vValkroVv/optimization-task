# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-backward grad reduce-scatter under PP: contract tests pinning the
semi-private torch behavior the feature relies on, plus FSDP/HSDP E2E on
multi- and single-stage schedules, chunked loss and shared-head paths."""

import copy
import gc
import types
import unittest
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    HSDPMeshInfo,
    ShardPlacementResult,
)
from torch.distributed.pipelining import PipelineStage
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import ChunkedCELoss, MTPLoss, cross_entropy_loss
from torchtitan.config import ParallelismConfig
from torchtitan.distributed.fsdp import _build_shard_units
from torchtitan.distributed.pipeline_grad_reduce import (
    _group_selected,
    _needs_fp32_accumulator,
    _release_reduce_scatter_keepalives,
    install_per_microbatch_grad_reduce,
)
from torchtitan.distributed.pipeline_parallel import _build_pipeline_schedule
from torchtitan.protocols.blocks import Segment

DIM = 16
NUM_MICROBATCHES = 4
MICROBATCH_SIZE = 2
NUM_STAGES = 4
PP_DEGREE = 2
FSDP_DEGREE = 2
REPLICATE_DEGREE = 2
SHARD_DEGREE = 2
NUM_DP_RANKS = REPLICATE_DEGREE * SHARD_DEGREE


def _loss_fn(out, target):
    return out.sum()


def _shard_dim0(full, num_shards, shard_rank):
    return torch.chunk(full, num_shards, dim=0)[shard_rank]


def _is_backward_wrapped(stage):
    return "backward_maybe_with_nosync" in stage.__dict__


def _iter_fsdp_params(model):
    state = fully_shard.state(model)
    for module_state in state._state_ctx.all_states:
        for group in module_state._fsdp_param_groups:
            yield from group.fsdp_params


class _StageLayer(nn.Module):
    def __init__(self, is_last, chunked_loss=False):
        super().__init__()
        self.linear = nn.Linear(DIM, DIM, bias=False)
        self.is_last = is_last
        if chunked_loss and is_last:
            self.norm = nn.LayerNorm(DIM)
            self.lm_head = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x):
        out = self.linear(x)
        if hasattr(self, "norm"):
            return self.norm(out)
        return out if self.is_last else torch.relu(out)


class _TwoGroupLayer(nn.Module):
    """One stage with two param groups of different kinds (dense + expert)."""

    def __init__(self, is_last):
        super().__init__()
        self.dense = nn.Linear(DIM, DIM, bias=False)
        self.expert = nn.Linear(DIM, DIM, bias=False)
        self.is_last = is_last

    def forward(self, x):
        out = self.expert(self.dense(x))
        return out if self.is_last else torch.relu(out)


class _ExpertSkipLayer(nn.Module):
    """A dense and an expert param group in one FSDP state, like a MoE block;
    ``skip_expert_calls`` leaves the expert out of those forward calls (an
    expert routed zero tokens), its post-backward hook still fires."""

    def __init__(self, is_last):
        super().__init__()
        self.dense = nn.Linear(DIM, DIM, bias=False)
        # the bias gives the expert group a different size: mispaired
        # collectives fail instead of silently mixing grads
        self.expert = nn.Linear(DIM, DIM)
        self.is_last = is_last
        self.forward_calls = 0
        self.skip_expert_calls = set()  # 1-based forward call numbers

    def forward(self, x, skip_expert=None):
        self.forward_calls += 1
        if skip_expert is None:
            skip_expert = self.forward_calls in self.skip_expert_calls
        out = self.dense(x)
        if not skip_expert:
            out = self.expert(out)
        return out if self.is_last else torch.relu(out)


def _make_batch(dp_index):
    g = torch.Generator().manual_seed(7000 + dp_index)
    return torch.randn(NUM_MICROBATCHES * MICROBATCH_SIZE, 4, DIM, generator=g)


def _reference_grads(base_layers, num_dp_ranks):
    """Pure-SUM grads of the unsharded model over all DP ranks' batches."""
    ref_layers = copy.deepcopy(base_layers)
    ref_model = nn.Sequential(*ref_layers)
    for dp_index in range(num_dp_ranks):
        out = ref_model(_make_batch(dp_index))
        if hasattr(ref_layers[-1], "lm_head"):
            labels = torch.zeros(out.shape[:-1], dtype=torch.long)
            cross_entropy_loss(ref_layers[-1].lm_head(out), labels).backward()
        else:
            _loss_fn(out, None).backward()
    return {
        (i, name): p.grad.clone()
        for i, layer in enumerate(ref_layers)
        for name, p in layer.named_parameters()
    }


class _PipelineBuilder:
    """Shared stage/schedule construction for the E2E suites."""

    def _build_stages(
        self, base_layers, mesh, pp_group, dp_mesh, *, num_stages=NUM_STAGES
    ):
        pp_rank = mesh["pp"].get_local_rank()
        stages_per_rank = num_stages // PP_DEGREE
        local_stage_idxs = [pp_rank + PP_DEGREE * s for s in range(stages_per_rank)]
        stages, stage_mods = [], {}
        for idx in local_stage_idxs:
            mod = copy.deepcopy(base_layers[idx]).to(self.device_type)
            if hasattr(mod, "lm_head"):
                segments = [
                    Segment(name, child, name == "linear", mod, name)
                    for name, child in mod.named_children()
                ]
                for unit in _build_shard_units(
                    segments,
                    reshard_after_forward=False,
                    output_reshard_after_forward=False,
                ):
                    fully_shard(unit.modules, mesh=dp_mesh, reshard_after_forward=False)
            fully_shard(mod, mesh=dp_mesh)  # 2D dp_mesh -> HSDP
            for child in mod.modules():
                if isinstance(child, FSDPModule):
                    child.set_gradient_divide_factor(1.0)
                    child.set_force_sum_reduction_for_comms(True)  # gloo
            stage_mods[idx] = mod
            stages.append(
                PipelineStage(
                    mod,
                    idx,
                    num_stages,
                    torch.device(self.device_type),
                    group=pp_group,
                )
            )
        return stages, stage_mods

    def _shard_with_expert_group(self, mod, dp_mesh):
        """HSDP-shard ``mod`` with its ``expert`` params in their own param
        group on their own process groups, as experts on the edp mesh are:
        the dp ranks transposed, so the expert group shards across replicas.
        Both groups live in one FSDP state, like a MoE block's."""
        expert_mesh = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size)
            .view(PP_DEGREE, REPLICATE_DEGREE, SHARD_DEGREE)
            .transpose(1, 2),
            mesh_dim_names=("pp", "edp_replicate", "edp_shard"),
        )["edp_replicate", "edp_shard"]
        expert_mesh_info = HSDPMeshInfo(
            mesh=expert_mesh, replicate_mesh_dim=0, shard_mesh_dim=1
        )
        expert_params = {id(p) for p in mod.expert.parameters()}

        def placement(param):
            if id(param) in expert_params:
                return ShardPlacementResult(
                    placement=Shard(0), mesh_info=expert_mesh_info
                )
            return None

        fully_shard(mod, mesh=dp_mesh, shard_placement_fn=placement)
        for group in fully_shard.state(mod)._fsdp_param_groups:
            # what apply_fsdp stamps
            group.grad_reduce_kind = (
                "moe" if group.mesh_info is expert_mesh_info else "dense"
            )
        mod.set_gradient_divide_factor(1.0)
        mod.set_force_sum_reduction_for_comms(True)  # gloo

    def _build_schedule(
        self,
        stages,
        *,
        loss_fn=_loss_fn,
        mode="full",
        schedule_name="Interleaved1F1B",
        ep_degree=1,
        num_microbatches=NUM_MICROBATCHES,
    ):
        parallelism = ParallelismConfig(
            pipeline_parallel_degree=PP_DEGREE,
            pipeline_parallel_schedule=schedule_name,
            pipeline_parallel_microbatch_size=MICROBATCH_SIZE,
            pipeline_parallel_grad_reduce=mode,
            expert_parallel_degree=ep_degree,
        )
        return _build_pipeline_schedule(
            parallelism=parallelism,
            local_batch_size=num_microbatches * MICROBATCH_SIZE,
            stages=stages,
            loss_fn=loss_fn,
        )


class TestCudaMemorySaving(DTensorTestBase):
    """Sharded accumulation must actually save the unsharded-gradient bytes."""

    @property
    def world_size(self):
        return FSDP_DEGREE

    @with_comms
    def test_cuda_memory_saving(self):
        if self.device_type != "cuda":
            self.skipTest("CUDA allocator statistics required")
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        dim, num_layers = 2048, 4
        gradient_bytes = num_layers * dim * dim * 4
        expected_saving = gradient_bytes * (1 - 1 / self.world_size)

        def measure(sync):
            gc.collect()
            torch.cuda.synchronize()
            model = nn.Sequential(
                *[nn.Linear(dim, dim, bias=False) for _ in range(num_layers)]
            ).to(self.device_type)
            for layer in model:
                fully_shard(layer, mesh=mesh, reshard_after_forward=False)
            fully_shard(model, mesh=mesh, reshard_after_forward=False)
            x = torch.randn(2, dim, device=self.device_type)
            # Materialize unsharded weights before counting gradient storage.
            out = model(x)
            # FSDP creates the groups lazily in the first forward.
            model.set_is_last_backward(False)
            model.set_reshard_after_backward(False)
            model.set_requires_gradient_sync(sync)
            torch.cuda.synchronize()
            start = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            for mb in range(NUM_MICROBATCHES):
                if mb:
                    out = model(x)
                out.sum().backward()
                torch.cuda.synchronize()
                _release_reduce_scatter_keepalives(model)
            resident = torch.cuda.memory_allocated() - start
            peak = torch.cuda.max_memory_allocated() - start
            grads = [
                (
                    param.sharded_param.grad.to_local()
                    if sync
                    else param.unsharded_param.grad
                )
                for param in _iter_fsdp_params(model)
            ]
            self.assertEqual(
                sum(grad.numel() * grad.element_size() for grad in grads),
                gradient_bytes // (self.world_size if sync else 1),
            )
            return resident, peak

        full_resident, full_peak = measure(False)
        shard_resident, shard_peak = measure(True)
        # Allocator residency also includes FSDP communication workspaces;
        # require at least the full-gradient-minus-shard saving checked above.
        self.assertGreaterEqual(
            full_resident - shard_resident, expected_saving - 1024 * 1024
        )
        self.assertGreater(full_peak - shard_peak, expected_saving / 4)


class TestInBackwardReduceContractHSDP(DTensorTestBase):
    """Pins the deferred all-reduce contract: mid-chunk backwards accumulate
    in the sharded partial buffer, the last one all-reduces the total."""

    @property
    def world_size(self):
        return NUM_DP_RANKS

    @staticmethod
    def _make_input(dp_index, microbatch_index):
        g = torch.Generator().manual_seed(1000 * dp_index + microbatch_index)
        return torch.randn(MICROBATCH_SIZE, DIM, generator=g)

    def _init_hsdp_model(self, mesh):
        torch.manual_seed(42)  # identical init on all ranks
        model = nn.Sequential(
            nn.Linear(DIM, DIM, bias=False), nn.Linear(DIM, DIM, bias=False)
        )
        ref_model = copy.deepcopy(model)
        model = model.to(self.device_type)
        # Multiple param groups exercise per-group reduce and both keepalives.
        fully_shard(model[0], mesh=mesh)
        fully_shard(model[1], mesh=mesh)
        fully_shard(model, mesh=mesh)
        for module in model.modules():
            if isinstance(module, FSDPModule):
                # Both setters are per-group, not recursive.
                module.set_gradient_divide_factor(1.0)  # repo invariant: SUM
                module.set_force_sum_reduction_for_comms(True)  # gloo
        return model, ref_model

    @with_comms
    def test_in_backward_hsdp_accumulation_matches_reference(self):
        mesh = init_device_mesh(
            self.device_type,
            (REPLICATE_DEGREE, SHARD_DEGREE),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        replicate_rank = mesh["dp_replicate"].get_local_rank()
        shard_rank = mesh["dp_shard"].get_local_rank()
        dp_index = replicate_rank * SHARD_DEGREE + shard_rank

        model, ref_model = self._init_hsdp_model(mesh)
        for r in range(NUM_DP_RANKS):
            for mb in range(NUM_MICROBATCHES):
                ref_model(self._make_input(r, mb)).sum().backward()
        expected = {n: p.grad.clone() for n, p in ref_model.named_parameters()}

        state = fully_shard.state(model)
        for mb in range(NUM_MICROBATCHES):
            last = mb == NUM_MICROBATCHES - 1
            out = model(self._make_input(dp_index, mb).to(self.device_type))
            # mid-pipe flags as the forced-sync wrapper sets them
            model.set_is_last_backward(False)
            model.set_reshard_after_backward(False)
            model.set_requires_gradient_sync(True)
            if not last:
                model.set_requires_all_reduce(False)
            out.sum().backward()

            groups_checked = 0
            for module in model.modules():
                if not isinstance(module, FSDPModule):
                    continue
                for group in module._get_fsdp_state()._fsdp_param_groups:
                    if not group.fsdp_params:
                        continue
                    groups_checked += 1
                    for fsdp_param in group.fsdp_params:
                        # No unsharded fp32 accumulation with sync on.
                        self.assertIsNone(fsdp_param.unsharded_accumulated_grad)
                        sharded_grad = fsdp_param.sharded_param.grad
                        if last:
                            self.assertIsInstance(sharded_grad, DTensor)
                            self.assertEqual(
                                sharded_grad.placements, (Replicate(), Shard(0))
                            )
                        else:
                            # deferred: totals live in the partial buffer
                            self.assertIsNone(sharded_grad)
                    if last:
                        # One all-reduce ran; partial buffer fully consumed.
                        self.assertIsNone(group._partial_reduce_output)
                        self.assertIsNotNone(group._all_reduce_state)
                    else:
                        self.assertIsNotNone(group._partial_reduce_output)
                        self.assertEqual(
                            group._partial_reduce_output.numel(),
                            sum(
                                p.sharded_param.to_local().numel()
                                for p in group.fsdp_params
                            ),
                            "partial buffer must stay sharded-size",
                        )
                        self.assertIsNone(group._all_reduce_state)
            self.assertGreater(groups_checked, 0)

            self.assertGreater(len(state._comm_ctx.reduce_scatter_states), 0)
            _release_reduce_scatter_keepalives(model)
            self.assertEqual(len(state._comm_ctx.reduce_scatter_states), 0)

        # reshard swaps the sharded params (with grads) into named_parameters
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        for name, param in model.named_parameters():
            torch.testing.assert_close(
                param.grad.to_local(),
                _shard_dim0(expected[name], SHARD_DEGREE, shard_rank).to(
                    self.device_type
                ),
                rtol=1e-5,
                atol=1e-6,
                msg=lambda m, name=name: f"grad mismatch for {name}: {m}",
            )


class TestPerMicrobatchGradReduceSchedule(_PipelineBuilder, DTensorTestBase):
    """E2E via _build_pipeline_schedule on a plain fsdp axis: pp=2 x fsdp=2,
    multi-stage (Interleaved1F1B) and single-stage (1F1B) schedules."""

    @property
    def world_size(self):
        return PP_DEGREE * FSDP_DEGREE

    def _make_base_layers(self, num_stages=NUM_STAGES):
        torch.manual_seed(42)  # identical init everywhere
        return [_StageLayer(is_last=(i == num_stages - 1)) for i in range(num_stages)]

    @with_comms
    def test_reduce_every_microbatch_two_steps(self):
        mesh = init_device_mesh(
            self.device_type,
            (PP_DEGREE, FSDP_DEGREE),
            mesh_dim_names=("pp", "fsdp"),
        )
        pp_rank = mesh["pp"].get_local_rank()
        fsdp_rank = mesh["fsdp"].get_local_rank()
        base_layers = self._make_base_layers()
        ref_grads = _reference_grads(base_layers, FSDP_DEGREE)
        stages, stage_mods = self._build_stages(
            base_layers, mesh, mesh["pp"].get_group(), mesh["fsdp"]
        )
        schedule = self._build_schedule(stages)
        # Re-install must be a no-op, not a second wrapper over the first.
        install_per_microbatch_grad_reduce(schedule)
        for stage in stages:
            self.assertTrue(_is_backward_wrapped(stage))

        x = _make_batch(fsdp_rank).to(self.device_type)
        target = torch.zeros(
            NUM_MICROBATCHES * MICROBATCH_SIZE, 4, DIM, device=self.device_type
        )
        # Two steps: catches counter/state bugs across schedule re-execution.
        for step in range(2):
            for mod in stage_mods.values():
                for p in mod.parameters():
                    p.grad = None
            if pp_rank == 0:
                schedule.step(x)
            else:
                schedule.step(target=target, losses=[])

            for idx, mod in stage_mods.items():
                for name, p in mod.named_parameters():
                    self.assertIsInstance(p.grad, DTensor)
                    self.assertEqual(p.grad.placements, (Shard(0),))
                    torch.testing.assert_close(
                        p.grad.to_local(),
                        _shard_dim0(ref_grads[(idx, name)], FSDP_DEGREE, fsdp_rank).to(
                            self.device_type
                        ),
                        rtol=1e-5,
                        atol=1e-6,
                        msg=lambda m, idx=idx, name=name, step=step: (
                            f"step {step} stage{idx}.{name}: {m}"
                        ),
                    )

    @with_comms
    def test_single_stage_1f1b_reduces_per_microbatch(self):
        """Single-stage schedules reach the hook via backward_one_chunk:
        reduce-scatter count must be microbatches, not one."""
        mesh = init_device_mesh(
            self.device_type,
            (PP_DEGREE, FSDP_DEGREE),
            mesh_dim_names=("pp", "fsdp"),
        )
        pp_rank = mesh["pp"].get_local_rank()
        fsdp_rank = mesh["fsdp"].get_local_rank()
        base_layers = self._make_base_layers(num_stages=PP_DEGREE)
        ref_grads = _reference_grads(base_layers, FSDP_DEGREE)
        stages, stage_mods = self._build_stages(
            base_layers,
            mesh,
            mesh["pp"].get_group(),
            mesh["fsdp"],
            num_stages=PP_DEGREE,
        )
        schedule = self._build_schedule(stages, schedule_name="1F1B")
        for stage in stages:
            self.assertTrue(_is_backward_wrapped(stage))

        x = _make_batch(fsdp_rank).to(self.device_type)
        target = torch.zeros(
            NUM_MICROBATCHES * MICROBATCH_SIZE, 4, DIM, device=self.device_type
        )
        for step in range(2):
            for mod in stage_mods.values():
                for p in mod.parameters():
                    p.grad = None
            with patch.object(
                dist, "reduce_scatter_single", wraps=dist.reduce_scatter_single
            ) as rs:
                if pp_rank == 0:
                    schedule.step(x)
                else:
                    schedule.step(target=target, losses=[])
            if step:
                # One param group per rank; "off" would show one call.
                self.assertEqual(rs.call_count, NUM_MICROBATCHES)
            for idx, mod in stage_mods.items():
                for name, p in mod.named_parameters():
                    self.assertIsInstance(p.grad, DTensor)
                    self.assertEqual(p.grad.placements, (Shard(0),))
                    torch.testing.assert_close(
                        p.grad.to_local(),
                        _shard_dim0(ref_grads[(idx, name)], FSDP_DEGREE, fsdp_rank).to(
                            self.device_type
                        ),
                        rtol=1e-5,
                        atol=1e-6,
                        msg=lambda m, idx=idx, name=name, step=step: (
                            f"step {step} stage{idx}.{name}: {m}"
                        ),
                    )

    @with_comms
    def test_selective_scope_reduces_only_matching_groups(self):
        """scope="dense"/"moe" puts only same-kind param groups on
        per-microbatch reduce; the other group keeps stock accumulation."""
        mesh = init_device_mesh(
            self.device_type,
            (PP_DEGREE, FSDP_DEGREE),
            mesh_dim_names=("pp", "fsdp"),
        )
        pp_rank = mesh["pp"].get_local_rank()
        fsdp_rank = mesh["fsdp"].get_local_rank()
        for scope in ("dense", "moe"):
            torch.manual_seed(42)
            base_layers = [
                _TwoGroupLayer(is_last=(i == PP_DEGREE - 1)) for i in range(PP_DEGREE)
            ]
            ref_grads = _reference_grads(base_layers, FSDP_DEGREE)
            mod = copy.deepcopy(base_layers[pp_rank]).to(self.device_type)
            for child, kind in ((mod.dense, "dense"), (mod.expert, "moe")):
                fully_shard(child, mesh=mesh["fsdp"])
                for group in child._get_fsdp_state()._fsdp_param_groups:
                    group.grad_reduce_kind = kind  # what apply_fsdp stamps
            fully_shard(mod, mesh=mesh["fsdp"])
            for child in mod.modules():
                if isinstance(child, FSDPModule):
                    child.set_gradient_divide_factor(1.0)
                    child.set_force_sum_reduction_for_comms(True)  # gloo
            stage = PipelineStage(
                mod,
                pp_rank,
                PP_DEGREE,
                torch.device(self.device_type),
                group=mesh["pp"].get_group(),
            )
            # ep_degree only satisfies config validation: the stages here are
            # hand-built, with group kinds stamped above.
            schedule = self._build_schedule(
                [stage], mode=scope, schedule_name="1F1B", ep_degree=2
            )
            self.assertTrue(_is_backward_wrapped(stage))

            x = _make_batch(fsdp_rank).to(self.device_type)
            target = torch.zeros(
                NUM_MICROBATCHES * MICROBATCH_SIZE, 4, DIM, device=self.device_type
            )
            for step in range(2):
                for p in mod.parameters():
                    p.grad = None
                with patch.object(
                    dist, "reduce_scatter_single", wraps=dist.reduce_scatter_single
                ) as rs:
                    if pp_rank == 0:
                        schedule.step(x)
                    else:
                        schedule.step(target=target, losses=[])
                if step:
                    # the selected group reduces per microbatch, the other
                    # accumulates unsharded and reduces once at REDUCE_GRAD
                    self.assertEqual(rs.call_count, NUM_MICROBATCHES + 1, scope)
                for name, p in mod.named_parameters():
                    self.assertIsInstance(p.grad, DTensor)
                    self.assertEqual(p.grad.placements, (Shard(0),))
                    torch.testing.assert_close(
                        p.grad.to_local(),
                        _shard_dim0(
                            ref_grads[(pp_rank, name)], FSDP_DEGREE, fsdp_rank
                        ).to(self.device_type),
                        rtol=1e-5,
                        atol=1e-6,
                        msg=lambda m, name=name, scope=scope: f"{scope} {name}: {m}",
                    )

    @with_comms
    def test_off_leaves_backward_unwrapped(self):
        mesh = init_device_mesh(
            self.device_type,
            (PP_DEGREE, FSDP_DEGREE),
            mesh_dim_names=("pp", "fsdp"),
        )
        stages, _ = self._build_stages(
            self._make_base_layers(), mesh, mesh["pp"].get_group(), mesh["fsdp"]
        )
        self._build_schedule(stages, mode="off")
        for stage in stages:
            self.assertFalse(_is_backward_wrapped(stage))


class TestFp32AccumulationBf16Params(_PipelineBuilder, DTensorTestBase):
    """bf16 params without a replicate axis: per-microbatch reduce must keep
    the cross-microbatch sum in the reduce dtype like stock does, instead of
    adding every reduce-scatter output onto a bf16 running sum."""

    num_microbatches = 64

    @property
    def world_size(self):
        return PP_DEGREE * FSDP_DEGREE

    def _relative_errors(self, mode):
        mesh = init_device_mesh(
            self.device_type,
            (PP_DEGREE, FSDP_DEGREE),
            mesh_dim_names=("pp", "fsdp"),
        )
        pp_rank = mesh["pp"].get_local_rank()
        fsdp_rank = mesh["fsdp"].get_local_rank()
        torch.manual_seed(42)
        base_layers = [
            _StageLayer(is_last=(i == PP_DEGREE - 1)).to(torch.bfloat16)
            for i in range(PP_DEGREE)
        ]
        batches = [
            # positive inputs keep every microbatch's grad the same sign, so
            # the sum outgrows each contribution (where bf16 sums lose bits)
            torch.rand(
                self.num_microbatches * MICROBATCH_SIZE,
                4,
                DIM,
                generator=torch.Generator().manual_seed(7000 + r),
            ).to(torch.bfloat16)
            for r in range(FSDP_DEGREE)
        ]
        # fp64 reference over the same bf16 weights and inputs
        reference = nn.Sequential(*copy.deepcopy(base_layers)).double()
        for batch in batches:
            reference(batch.double()).sum().backward()
        mod = copy.deepcopy(base_layers[pp_rank]).to(self.device_type)
        fully_shard(
            mod,
            mesh=mesh["fsdp"],
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        mod.set_gradient_divide_factor(1.0)
        mod.set_force_sum_reduction_for_comms(True)  # gloo
        stage = PipelineStage(
            mod,
            pp_rank,
            PP_DEGREE,
            torch.device(self.device_type),
            group=mesh["pp"].get_group(),
        )
        schedule = self._build_schedule(
            [stage],
            mode=mode,
            schedule_name="1F1B",
            num_microbatches=self.num_microbatches,
        )
        if pp_rank == 0:
            schedule.step(batches[fsdp_rank].to(self.device_type))
        else:
            schedule.step(
                target=torch.zeros(
                    self.num_microbatches * MICROBATCH_SIZE,
                    4,
                    DIM,
                    device=self.device_type,
                ),
                losses=[],
            )
        errors = []
        for name, p in mod.named_parameters():
            self.assertEqual(p.grad.dtype, torch.bfloat16)
            expected = _shard_dim0(
                reference[pp_rank].get_parameter(name).grad, FSDP_DEGREE, fsdp_rank
            )
            actual = p.grad.to_local().double().cpu()
            errors.append(((actual - expected).norm() / expected.norm()).item())
        return errors

    @with_comms
    def test_bf16_params_accumulate_like_stock(self):
        off = self._relative_errors("off")
        full = self._relative_errors("full")
        # Proves the check is sensitive: without the accumulator the bf16
        # running sum is visibly worse than stock.
        with patch(
            "torchtitan.distributed.pipeline_grad_reduce._needs_fp32_accumulator",
            return_value=False,
        ):
            bf16_sum = self._relative_errors("full")
        failure = ""
        try:
            for e_off, e_full, e_bf16 in zip(off, full, bf16_sum):
                # one extra bf16 rounding per contribution, not per running sum
                self.assertLess(e_full, 2 * e_off + 1e-3)
                self.assertGreater(e_bf16, 2 * e_full)
        except AssertionError as error:
            failure = f"off={off} full={full} bf16_sum={bf16_sum}: {error}"
        all_failures = [None] * self.world_size
        dist.all_gather_object(all_failures, failure)
        self.assertFalse(any(all_failures), "\n".join(filter(None, all_failures)))


class TestPerMicrobatchGradReduceHSDP(_PipelineBuilder, DTensorTestBase):
    """E2E on Interleaved1F1B: pp=2 x HSDP(dp_replicate=2, dp_shard=2)."""

    @property
    def world_size(self):
        return PP_DEGREE * NUM_DP_RANKS

    def _make_base_layers(self, chunked_loss=False):
        torch.manual_seed(42)  # identical init everywhere
        return [
            _StageLayer(is_last=(i == NUM_STAGES - 1), chunked_loss=chunked_loss)
            for i in range(NUM_STAGES)
        ]

    def _run_case(self, *, chunked_loss=False, mode="full"):
        mesh = init_device_mesh(
            self.device_type,
            (PP_DEGREE, REPLICATE_DEGREE, SHARD_DEGREE),
            mesh_dim_names=("pp", "dp_replicate", "dp_shard"),
        )
        pp_rank = mesh["pp"].get_local_rank()
        replicate_rank = mesh["dp_replicate"].get_local_rank()
        shard_rank = mesh["dp_shard"].get_local_rank()
        dp_index = replicate_rank * SHARD_DEGREE + shard_rank

        base_layers = self._make_base_layers(chunked_loss)
        ref_grads = {
            key: _shard_dim0(grad, SHARD_DEGREE, shard_rank).to(self.device_type)
            for key, grad in _reference_grads(base_layers, NUM_DP_RANKS).items()
        }
        stages, stage_mods = self._build_stages(
            base_layers,
            mesh,
            mesh["pp"].get_group(),
            mesh["dp_replicate", "dp_shard"],
        )
        loss_fn = _loss_fn
        if chunked_loss:
            loss_fn = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=2))
            loss_fn.reset_components()
            if NUM_STAGES - 1 in stage_mods:
                loss_fn.set_lm_head(stage_mods[NUM_STAGES - 1].lm_head)
        schedule = self._build_schedule(stages, loss_fn=loss_fn, mode=mode)
        for stage in stages:
            self.assertEqual(_is_backward_wrapped(stage), mode != "off")

        x = _make_batch(dp_index).to(self.device_type)
        target = torch.zeros(
            NUM_MICROBATCHES * MICROBATCH_SIZE,
            4,
            device=self.device_type,
            dtype=torch.long,
        )
        # Two steps: catches counter/state bugs across schedule re-execution.
        for step in range(2):
            for mod in stage_mods.values():
                for p in mod.parameters():
                    p.grad = None
            with (
                patch.object(
                    dist, "reduce_scatter_single", wraps=dist.reduce_scatter_single
                ) as rs,
                patch.object(dist, "all_reduce", wraps=dist.all_reduce) as ar,
            ):
                if pp_rank == 0:
                    schedule.step(x)
                else:
                    schedule.step(target=target, losses=[])
            if self.device_type == "cuda":
                torch.cuda.synchronize()
            failure = ""
            try:
                # Step 0 includes torch's shape probe; count only real microbatches.
                if step:
                    num_groups = 3 if chunked_loss and pp_rank == 1 else 2
                    head_reduces = (
                        NUM_MICROBATCHES if chunked_loss and pp_rank == 1 else 0
                    )
                    expected_rs = num_groups * (
                        NUM_MICROBATCHES if mode != "off" else 1
                    )
                    self.assertEqual(rs.call_count, expected_rs + head_reduces)
                    head_all_reduces = (
                        int(bool(head_reduces)) if mode != "off" else head_reduces
                    )
                    self.assertEqual(ar.call_count, num_groups + head_all_reduces)

                for idx, mod in stage_mods.items():
                    for child in mod.modules():
                        if not isinstance(child, FSDPModule):
                            continue
                        state = fully_shard.state(child)
                        self.assertEqual(state._comm_ctx.reduce_scatter_states, [])
                        for group in state._fsdp_param_groups:
                            self.assertIsNone(group._all_reduce_state)
                            self.assertIsNone(group._partial_reduce_output)
                            # armed only for the stage backward
                            self.assertFalse(group.reduce_scatter_unused_params)
                            for param in group.fsdp_params:
                                self.assertIsNone(param.unsharded_accumulated_grad)
                    for name, p in mod.named_parameters():
                        self.assertIsInstance(p.grad, DTensor)
                        self.assertEqual(p.grad.placements, (Replicate(), Shard(0)))
                        torch.testing.assert_close(
                            p.grad.to_local(),
                            ref_grads[(idx, name)],
                            rtol=1e-5,
                            # CPU/CUDA LayerNorm rounding is visible near zero.
                            atol=1e-5 if chunked_loss else 1e-6,
                            msg=lambda m, idx=idx, name=name, step=step: (
                                f"step {step} stage{idx}.{name}: {m}"
                            ),
                        )
            except AssertionError as error:
                failure = str(error)

            # raise on all ranks together: a lone raiser strands NCCL peers
            all_failures = [None] * self.world_size
            dist.all_gather_object(all_failures, failure)
            self.assertFalse(any(all_failures), "\n".join(filter(None, all_failures)))

    @with_comms
    def test_expert_skipped_on_one_rank_keeps_collectives_aligned(self):
        """An expert with no tokens in the last microbatch on one dp rank only
        must still reduce-scatter and all-reduce at the same point as on its
        peers: same collective sequence on every dp rank, no hang, grads
        equal to the reference."""
        mesh = init_device_mesh(
            self.device_type,
            (PP_DEGREE, REPLICATE_DEGREE, SHARD_DEGREE),
            mesh_dim_names=("pp", "dp_replicate", "dp_shard"),
        )
        pp_rank = mesh["pp"].get_local_rank()
        dp_index = (
            mesh["dp_replicate"].get_local_rank() * SHARD_DEGREE
            + mesh["dp_shard"].get_local_rank()
        )
        skip_dp_index = NUM_DP_RANKS - 1
        dp_mesh = mesh["dp_replicate", "dp_shard"]

        torch.manual_seed(42)
        base_layers = [
            _ExpertSkipLayer(is_last=(i == PP_DEGREE - 1)) for i in range(PP_DEGREE)
        ]
        reference = copy.deepcopy(base_layers)
        for r in range(NUM_DP_RANKS):
            batch = _make_batch(r)
            for mb in range(NUM_MICROBATCHES):
                out = batch[mb * MICROBATCH_SIZE : (mb + 1) * MICROBATCH_SIZE]
                skip = r == skip_dp_index and mb == NUM_MICROBATCHES - 1
                for layer in reference:
                    out = layer(out, skip_expert=skip)
                out.sum().backward()

        mod = copy.deepcopy(base_layers[pp_rank]).to(self.device_type)
        self._shard_with_expert_group(mod, dp_mesh)
        groups = fully_shard.state(mod)._fsdp_param_groups
        self.assertEqual(len(groups), 2)
        stage = PipelineStage(
            mod,
            pp_rank,
            PP_DEGREE,
            torch.device(self.device_type),
            group=mesh["pp"].get_group(),
        )
        schedule = self._build_schedule([stage], schedule_name="1F1B")

        x = _make_batch(dp_index).to(self.device_type)
        target = torch.zeros(
            NUM_MICROBATCHES * MICROBATCH_SIZE, 4, DIM, device=self.device_type
        )
        # Step 0 includes torch's shape probe; skip in step 1 only.
        for step in range(2):
            for p in mod.parameters():
                p.grad = None
            mod.forward_calls = 0
            if step and dp_index == skip_dp_index:
                mod.skip_expert_calls = {NUM_MICROBATCHES}
            calls = []
            original_rs, original_ar = dist.reduce_scatter_single, dist.all_reduce

            def rs(output, *args, **kwargs):
                calls.append(("RS", output.numel()))
                return original_rs(output, *args, **kwargs)

            def ar(tensor, *args, **kwargs):
                calls.append(("AR", tensor.numel()))
                return original_ar(tensor, *args, **kwargs)

            with (
                patch.object(dist, "reduce_scatter_single", rs),
                patch.object(dist, "all_reduce", ar),
            ):
                if pp_rank == 0:
                    schedule.step(x)
                else:
                    schedule.step(target=target, losses=[])
            if self.device_type == "cuda":
                torch.cuda.synchronize()

        # collective: gather on every rank before any assertion can raise
        full_grads = {
            name: p.grad.full_tensor().cpu() for name, p in mod.named_parameters()
        }
        failure = ""
        try:
            self.assertEqual(mod.forward_calls, NUM_MICROBATCHES)
            for group in groups:
                self.assertFalse(group.reduce_scatter_unused_params)
                self.assertIsNone(group._partial_reduce_output)
            for name, grad in full_grads.items():
                torch.testing.assert_close(
                    grad,
                    reference[pp_rank].get_parameter(name).grad,
                    rtol=1e-5,
                    atol=1e-5,  # GPU vs CPU fp32 summation order near zero
                    msg=lambda m, name=name: f"{name}: {m}",
                )
        except AssertionError as error:
            failure = f"rank {self.rank} (pp {pp_rank}, dp {dp_index}): {error}"
        sequences = [None] * self.world_size
        dist.all_gather_object(sequences, (pp_rank, calls))
        all_failures = [None] * self.world_size
        dist.all_gather_object(all_failures, failure)
        self.assertFalse(any(all_failures), "\n".join(filter(None, all_failures)))
        for stage_rank in range(PP_DEGREE):
            stage_sequences = [seq for r, seq in sequences if r == stage_rank]
            self.assertTrue(stage_sequences[0])
            for seq in stage_sequences[1:]:
                self.assertEqual(seq, stage_sequences[0])

    @with_comms
    def test_hsdp_grads_match_reference_two_steps(self):
        self._run_case()

    @with_comms
    def test_chunked_loss_grads_and_collectives(self):
        self._run_case(chunked_loss=True)

    @with_comms
    def test_off_chunked_loss_collectives(self):
        self._run_case(chunked_loss=True, mode="off")


class _SharedHeadStage(_StageLayer):
    def forward(self, inputs):
        hidden = super().forward(inputs)
        return torch.stack([hidden, 0.7 * hidden]) if hasattr(self, "norm") else hidden


class _ObservedMTPLoss(MTPLoss):
    in_head = False

    def _chunked_loss_and_grad(self, *args, **kwargs):
        self.in_head = True
        try:
            return super()._chunked_loss_and_grad(*args, **kwargs)
        finally:
            self.in_head = False


def _shared_head_batch(dp_rank, accumulation):
    generator = torch.Generator().manual_seed(7000 + dp_rank + 100 * accumulation)
    return (
        torch.randn(8, 4, DIM, generator=generator),
        torch.randint(DIM, (8, 4), generator=generator),
    )


def _integer_expert_layers(seed):
    """_ExpertSkipLayer stages with sparse ternary integer weights: every
    gradient term is an integer and, by the abs bound checked in the test,
    every partial sum stays below 2**24, so fp32 sums are exact in any order
    and reordered reductions must agree bitwise."""
    generator = torch.Generator().manual_seed(seed)
    layers = [
        _ExpertSkipLayer(is_last=(i == NUM_STAGES - 1)) for i in range(NUM_STAGES)
    ]
    with torch.no_grad():
        for layer in layers:
            for p in layer.parameters():
                values = torch.randint(-1, 2, p.shape, generator=generator)
                keep = torch.rand(p.shape, generator=generator) < 0.2
                p.copy_((values * keep).float())
    return layers


def _integer_batch(dp_index, accumulation):
    generator = torch.Generator().manual_seed(9000 + 10 * dp_index + accumulation)
    return torch.randint(
        -1, 2, (NUM_MICROBATCHES * MICROBATCH_SIZE, 4, DIM), generator=generator
    ).float()


class TestBitwiseAgainstStock(_PipelineBuilder, DTensorTestBase):
    """Exact arithmetic, so only the reduction plumbing can make a difference:
    every scope must reproduce stock ("off") gradients bitwise -- each
    microbatch counted once, on the right shard, all-reduced once -- over two
    optimizer steps of two accumulation chunks each, on Interleaved1F1B with
    pp=2 x HSDP(2x2), dense and expert groups per stage, and an expert skipped
    in one microbatch on one dp rank."""

    num_accumulation = 2
    num_iterations = 2
    skip_dp_index = NUM_DP_RANKS - 1

    @property
    def world_size(self):
        return PP_DEGREE * NUM_DP_RANKS

    def _skips(self, dp_index, accumulation):
        # one rank's expert gets no tokens in a middle and a last microbatch
        if dp_index != self.skip_dp_index:
            return set()
        return {1, NUM_MICROBATCHES - 1} if accumulation else {NUM_MICROBATCHES - 1}

    def _reference(self, layers, *, bound=False):
        """fp64 grads summed over all dp ranks; ``bound`` instead returns the
        per-element sum of |terms| (abs weights and inputs, no relu, no skip):
        an upper bound on every partial sum of any reduction order, and the
        largest activation magnitude under key ``"activation"``."""
        layers = copy.deepcopy(layers)
        max_activation = 0.0
        for layer in layers:
            layer.double()
            if bound:
                with torch.no_grad():
                    for p in layer.parameters():
                        p.abs_()
                layer.is_last = True  # no relu: its derivative is <= 1
        for r in range(NUM_DP_RANKS):
            for a in range(self.num_accumulation):
                batch = _integer_batch(r, a).double()
                skips = set() if bound else self._skips(r, a)
                for mb in range(NUM_MICROBATCHES):
                    out = batch[mb * MICROBATCH_SIZE : (mb + 1) * MICROBATCH_SIZE]
                    if bound:
                        out = out.abs()
                    for layer in layers:
                        out = layer(out, skip_expert=mb in skips)
                        max_activation = max(max_activation, out.abs().max().item())
                    out.sum().backward()
        if bound:
            return {
                "activation": max_activation,
                "grad": max(
                    p.grad.abs().max().item()
                    for layer in layers
                    for p in layer.parameters()
                ),
            }
        return {
            (i, name): p.grad
            for i, layer in enumerate(layers)
            for name, p in layer.named_parameters()
        }

    def _train_scope(self, mode, mesh, base_layers):
        pp_rank = mesh["pp"].get_local_rank()
        dp_index = (
            mesh["dp_replicate"].get_local_rank() * SHARD_DEGREE
            + mesh["dp_shard"].get_local_rank()
        )
        dp_mesh = mesh["dp_replicate", "dp_shard"]
        stage_idxs = [pp_rank + PP_DEGREE * s for s in range(NUM_STAGES // PP_DEGREE)]
        stage_mods, stages = {}, []
        for idx in stage_idxs:
            mod = copy.deepcopy(base_layers[idx]).to(self.device_type)
            self._shard_with_expert_group(mod, dp_mesh)
            stage_mods[idx] = mod
            stages.append(
                PipelineStage(
                    mod,
                    idx,
                    NUM_STAGES,
                    torch.device(self.device_type),
                    group=mesh["pp"].get_group(),
                )
            )
        schedule = self._build_schedule(stages, mode=mode, ep_degree=2)
        target = torch.zeros(
            NUM_MICROBATCHES * MICROBATCH_SIZE, 4, DIM, device=self.device_type
        )

        def step(x):
            for mod in stage_mods.values():
                mod.forward_calls = 0
            if pp_rank == 0:
                schedule.step(x)
            else:
                schedule.step(target=target, losses=[])

        # absorbs the first-step shape probe, so forward counts are exact
        step(_integer_batch(dp_index, 0).to(self.device_type))
        grads_per_iteration = []
        for _ in range(self.num_iterations):
            for mod in stage_mods.values():
                for p in mod.parameters():
                    p.grad = None
            for a in range(self.num_accumulation):
                for mod in stage_mods.values():
                    # forward calls are 1-based, microbatch indices 0-based
                    mod.skip_expert_calls = {mb + 1 for mb in self._skips(dp_index, a)}
                step(_integer_batch(dp_index, a).to(self.device_type))
                for mod in stage_mods.values():
                    mod.skip_expert_calls = set()
            grads_per_iteration.append(
                {
                    (idx, name): p.grad.full_tensor().cpu()
                    for idx, mod in stage_mods.items()
                    for name, p in mod.named_parameters()
                }
            )
        return grads_per_iteration

    @with_comms
    def test_scopes_reproduce_stock_bitwise(self):
        mesh = init_device_mesh(
            self.device_type,
            (PP_DEGREE, REPLICATE_DEGREE, SHARD_DEGREE),
            mesh_dim_names=("pp", "dp_replicate", "dp_shard"),
        )
        base_layers = _integer_expert_layers(seed=3)
        bounds = self._reference(base_layers, bound=True)
        # integer sums are exact in fp32 below 2**24, in any order
        self.assertLess(max(bounds.values()), 2**24, bounds)
        expected = {
            key: grad.float() for key, grad in self._reference(base_layers).items()
        }
        stock = self._train_scope("off", mesh, base_layers)
        results = {
            mode: self._train_scope(mode, mesh, base_layers)
            for mode in ("full", "dense", "moe")
        }
        failure = ""
        try:
            for it, grads in enumerate(stock):
                for key, grad in grads.items():
                    self.assertTrue(
                        torch.equal(grad, expected[key]), f"off iter {it} {key}"
                    )
            for mode, per_iteration in results.items():
                for it, grads in enumerate(per_iteration):
                    for key, grad in grads.items():
                        self.assertTrue(
                            torch.equal(grad, stock[it][key]),
                            f"{mode} iter {it} {key}: max |diff| "
                            f"{(grad - stock[it][key]).abs().max().item()}",
                        )
        except AssertionError as error:
            failure = f"rank {self.rank}: {error}"
        all_failures = [None] * self.world_size
        dist.all_gather_object(all_failures, failure)
        self.assertFalse(any(all_failures), "\n".join(filter(None, all_failures)))


class TestSharedHeadAccumulation(_PipelineBuilder, DTensorTestBase):
    world_size = 8

    @with_comms
    def test_two_optimizer_steps_two_schedules(self):
        mesh = init_device_mesh(
            self.device_type,
            (2, 2, 2),
            mesh_dim_names=("pp", "dp_replicate", "dp_shard"),
        )
        pp_rank = mesh["pp"].get_local_rank()
        shard_rank = mesh["dp_shard"].get_local_rank()
        dp_rank = mesh["dp_replicate"].get_local_rank() * 2 + shard_rank
        torch.manual_seed(42)
        layers = [_SharedHeadStage(is_last=i == 3, chunked_loss=True) for i in range(4)]
        reference = nn.Sequential(*copy.deepcopy(layers))
        stages, stage_mods = self._build_stages(
            layers, mesh, mesh["pp"].get_group(), mesh["dp_replicate", "dp_shard"]
        )
        loss_fn = _ObservedMTPLoss(MTPLoss.Config(num_chunks=2, loss_weights=[0.3]))
        if 3 in stage_mods:
            loss_fn.set_lm_head(stage_mods[3].lm_head)
        schedule = self._build_schedule(stages, loss_fn=loss_fn)
        optimizer = torch.optim.SGD(
            [p for mod in stage_mods.values() for p in mod.parameters()], lr=0.01
        )
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        batches = [
            tuple(t.to(self.device_type) for t in _shared_head_batch(dp_rank, a))
            for a in range(2)
        ]
        denominator = torch.tensor(256, device=self.device_type)
        for step in range(2):
            reference_optimizer.zero_grad(set_to_none=True)
            for peer in range(4):
                for accumulation in range(2):
                    x, y = _shared_head_batch(peer, accumulation)
                    hidden = reference(x)
                    shifted = torch.full_like(y, -100)
                    shifted[:, :-1] = y[:, 1:]
                    value = (
                        cross_entropy_loss(reference[-1].lm_head(hidden[0]), y)
                        + 0.3
                        * cross_entropy_loss(reference[-1].lm_head(hidden[1]), shifted)
                    ) / 256
                    value.backward()
            expected = {
                (i, name): parameter.grad.chunk(2, dim=0)[shard_rank].to(
                    self.device_type
                )
                for i, layer in enumerate(reference)
                for name, parameter in layer.named_parameters()
            }
            optimizer.zero_grad(set_to_none=True)
            loss_fn.reset_components()
            counts = {"RS": 0, "AR": 0}
            original_rs, original_ar = dist.reduce_scatter_single, dist.all_reduce

            def rs(*args, **kwargs):
                counts["RS"] += int(loss_fn.in_head)
                return original_rs(*args, **kwargs)

            def ar(*args, **kwargs):
                counts["AR"] += int(loss_fn.in_head)
                return original_ar(*args, **kwargs)

            with (
                patch.object(dist, "reduce_scatter_single", rs),
                patch.object(dist, "all_reduce", ar),
            ):
                for accumulation, (x, y) in enumerate(batches):
                    loss_fn.is_last_accumulation_step = accumulation == 1
                    if pp_rank == 0:
                        schedule.step(
                            x, loss_kwargs={"global_valid_tokens": denominator}
                        )
                    else:
                        schedule.step(
                            target=y,
                            losses=[],
                            loss_kwargs={"global_valid_tokens": denominator},
                        )
            if self.device_type == "cuda":
                torch.cuda.synchronize()
            failure = ""
            try:
                # Step 0 also executes a fully synchronized metadata loss.
                if pp_rank == 1 and step:
                    self.assertEqual(counts, {"RS": 8, "AR": 1})
                for i, mod in stage_mods.items():
                    for name, parameter in mod.named_parameters():
                        torch.testing.assert_close(
                            parameter.grad.to_local(),
                            expected[(i, name)],
                            rtol=1e-5,
                            atol=1e-5,
                            msg=f"step{step} {i}.{name}",
                        )
                    for child in mod.modules():
                        if not isinstance(child, FSDPModule):
                            continue
                        state = fully_shard.state(child)
                        self.assertEqual(state._comm_ctx.reduce_scatter_states, [])
                        for group in state._fsdp_param_groups:
                            self.assertIsNone(group._partial_reduce_output)
                            self.assertIsNone(group._all_reduce_state)
                            # armed only for the stage backward, never the head
                            self.assertFalse(group.reduce_scatter_unused_params)
                            for parameter in group.fsdp_params:
                                self.assertIsNone(parameter.unsharded_accumulated_grad)
            except (AssertionError, RuntimeError) as error:
                failure = str(error)
            all_failures = [None] * self.world_size
            dist.all_gather_object(all_failures, failure)
            self.assertFalse(any(all_failures), "\n".join(filter(None, all_failures)))
            optimizer.step()
            reference_optimizer.step()


class TestInstallValidation(unittest.TestCase):
    def test_non_pipeline_schedule_rejected(self):
        """Non-torch schedules raise; GPipe/1F1B/Interleaved all install."""
        with pytest.raises(ValueError, match="torch pipeline schedules"):
            install_per_microbatch_grad_reduce(object())

    def test_moe_scope_requires_expert_parallel(self):
        with pytest.raises(ValueError, match="expert_parallel_degree"):
            ParallelismConfig(
                pipeline_parallel_degree=2,
                pipeline_parallel_grad_reduce="moe",
            )

    def test_mixed_group_rejected_under_selective_scope(self):
        """A group holding both expert and dense params (ep=1) cannot be
        split by a selective scope."""
        group = types.SimpleNamespace(grad_reduce_kind="mixed", _module_fqn="blk")
        with pytest.raises(ValueError, match="cannot split"):
            _group_selected(group, "dense")

    def test_fp32_accumulator_only_where_bf16_would_accumulate(self):
        hsdp_mesh_info = HSDPMeshInfo.__new__(HSDPMeshInfo)  # isinstance only

        def group(orig=torch.bfloat16, mesh_info=None, kind="dense", head=False):
            module = nn.Linear(1, 1)
            if head:
                module._loss_managed_grad_reduce = True
            return types.SimpleNamespace(
                _orig_dtype=orig,
                _reduce_dtype=torch.float32,
                mesh_info=mesh_info,
                modules=(module,),
                grad_reduce_kind=kind,
                _module_fqn="layer",
            )

        self.assertTrue(_needs_fp32_accumulator(group(), "full"))
        self.assertTrue(_needs_fp32_accumulator(group(), "dense"))
        # fp32 params already accumulate in fp32
        self.assertFalse(_needs_fp32_accumulator(group(torch.float32), "full"))
        # HSDP: the deferred all-reduce partial lives in the reduce dtype
        self.assertFalse(
            _needs_fp32_accumulator(group(mesh_info=hsdp_mesh_info), "full")
        )
        self.assertFalse(_needs_fp32_accumulator(group(head=True), "full"))
        self.assertFalse(_needs_fp32_accumulator(group(kind="moe"), "dense"))

    def test_gfsdp_backend_rejects_grad_reduce(self):
        from torchtitan.distributed.backends.gfsdp2 import GFSDP2Backend

        with pytest.raises(ValueError, match="GFSDP2"):
            GFSDP2Backend().validate_parallelism(
                ParallelismConfig(
                    pipeline_parallel_degree=2,
                    pipeline_parallel_schedule="Interleaved1F1B",
                    pipeline_parallel_grad_reduce="full",
                )
            )

    def test_gfsdp_stage_rejected_by_installer(self):
        """GFSDPModule subclasses FSDPModule, so the isinstance gate alone
        would wrap it into torch-FSDP comm internals it does not expose."""
        from torchtitan.distributed.gfsdp import GFSDPModule
        from torchtitan.distributed.pipeline_grad_reduce import _reject_gfsdp_stage

        module = nn.Linear(4, 4)
        module.__class__ = type("GFSDPLinear", (GFSDPModule, nn.Linear), {})
        stage = types.SimpleNamespace(submod=module, stage_index=0)
        with pytest.raises(ValueError, match="gfsdp stages"):
            _reject_gfsdp_stage(stage)


if __name__ == "__main__":
    unittest.main()
