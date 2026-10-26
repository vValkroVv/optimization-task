# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared-head loss gradients with HSDP, tensor parallelism and tied weights."""

import copy
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    loss_parallel,
    parallelize_module,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import ChunkedCELoss, cross_entropy_loss, MTPLoss


class _TiedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens):
        hidden = self.embedding(tokens)
        return torch.stack([hidden, 0.7 * hidden])


class TestChunkedLossFSDP(DTensorTestBase):
    world_size = 4

    def _run_head(self, *, tp=False, weights=(0.3,)):
        torch.manual_seed(42)
        head = nn.Linear(8, 16, bias=False).to(self.device_type)
        reference = copy.deepcopy(head)
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dp_shard", "tp") if tp else ("dp_replicate", "dp_shard"),
        )
        if tp:
            parallelize_module(
                head, mesh["tp"], ColwiseParallel(use_local_output=False)
            )
        fully_shard(head, mesh=mesh["dp_shard"] if tp else mesh)
        head.set_gradient_divide_factor(1.0)
        head.set_force_sum_reduction_for_comms(True)
        # MTPLoss rejects empty loss_weights: zero MTP heads is plain chunked CE.
        criterion = (
            MTPLoss(MTPLoss.Config(num_chunks=2, loss_weights=list(weights)))
            if weights
            else ChunkedCELoss(ChunkedCELoss.Config(num_chunks=2))
        )
        criterion.set_lm_head(head)
        generator = torch.Generator().manual_seed(71)
        inputs = [
            torch.randn(len(weights) + 1, 2, 4, 8, generator=generator)
            for _ in range(2)
        ]
        labels = torch.randint(16, (2, 4), generator=generator)
        expected_inputs = []
        for value in inputs:
            # detach: on CPU ``.to`` is a no-op and ``requires_grad_`` would
            # mutate the shared input, breaking the no-grad eval pass below
            value = value.to(self.device_type).detach().requires_grad_(True)
            objective = cross_entropy_loss(
                reference(value[0]), labels.to(self.device_type)
            )
            for k, weight in enumerate(weights, 1):
                shifted = MTPLoss._shift_labels(labels, k).to(self.device_type)
                objective = objective + weight * cross_entropy_loss(
                    reference(value[k]), shifted
                )
            objective.backward()
            expected_inputs.append(value.grad)
        peers = 2 if tp else 4
        expected_weight = reference.weight.grad * peers
        observed = []
        with (
            patch.object(
                dist, "reduce_scatter_single", wraps=dist.reduce_scatter_single
            ) as rs,
            patch.object(dist, "all_reduce", wraps=dist.all_reduce) as ar,
        ):
            for index, value in enumerate(inputs):
                value = value.to(self.device_type)
                target = labels.to(self.device_type)
                if tp:
                    value = distribute_tensor(value, mesh["tp"], (Shard(2),))
                    target = distribute_tensor(target, mesh["tp"], (Replicate(),))
                value = value.detach().requires_grad_(True)
                with loss_parallel() if tp else nullcontext():
                    result = criterion(
                        value if weights else value[0],
                        target,
                        sync_all_reduce=index == 1,
                    )
                    result.backward()
                observed.append(value.grad.full_tensor() if tp else value.grad)
            rs_count, ar_count = rs.call_count, ar.call_count
        # Direct calls never reach the trainer/REDUCE_GRAD finalization that
        # waits on the async reduces; mirror it before reading grads.
        head.set_is_last_backward(True)
        fully_shard.state(head)._root_post_backward_final_callback()
        torch.testing.assert_close(
            head.weight.grad.full_tensor(), expected_weight, rtol=1e-5, atol=1e-5
        )
        for actual, expected in zip(observed, expected_inputs):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(rs_count, 2)
        if not tp:
            self.assertEqual(ar_count, 1)
        state = fully_shard.state(head)
        for group in state._fsdp_param_groups:
            self.assertIsNone(group._partial_reduce_output)
            for parameter in group.fsdp_params:
                self.assertIsNone(parameter.unsharded_accumulated_grad)
        # Evaluation must leave gradients and synchronization ready for training.
        saved_grad = head.weight.grad.clone()
        with torch.no_grad(), loss_parallel() if tp else nullcontext():
            hidden = inputs[0][0].to(self.device_type)
            target = labels.to(self.device_type)
            if tp:
                hidden = distribute_tensor(hidden, mesh["tp"], (Replicate(),))
                target = distribute_tensor(target, mesh["tp"], (Replicate(),))
            result = criterion(hidden, target)
        self.assertFalse(result.requires_grad)
        torch.testing.assert_close(head.weight.grad, saved_grad, rtol=0, atol=0)
        for group in state._fsdp_param_groups:
            self.assertTrue(group.reduce_grads)
            self.assertTrue(group.all_reduce_grads)

    @with_comms
    def test_hsdp_main_and_mtp(self):
        self._run_head()

    @with_comms
    def test_hsdp_zero_mtp(self):
        self._run_head(weights=())

    @with_comms
    def test_hsdp_zero_weight_last_mtp(self):
        self._run_head(weights=(0.3, 0.0))

    @with_comms
    def test_fsdp_tensor_parallel_sequence_sharded_input(self):
        self._run_head(tp=True)

    @with_comms
    def test_tied_embedding_head_keeps_per_prediction_sync(self):
        torch.manual_seed(42)
        model = _TiedHead().to(self.device_type)
        reference = copy.deepcopy(model)
        mesh = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("replicate", "shard")
        )
        fully_shard([model.embedding, model.head], mesh=mesh)
        fully_shard(model, mesh=mesh)
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.set_gradient_divide_factor(1.0)
                module.set_force_sum_reduction_for_comms(True)
        criterion = MTPLoss(MTPLoss.Config(num_chunks=2, loss_weights=[0.3]))
        criterion.set_lm_head(model.head)
        generator = torch.Generator().manual_seed(17)
        tokens = torch.randint(16, (2, 4), generator=generator).to(self.device_type)
        labels = torch.randint(16, (2, 4), generator=generator).to(self.device_type)
        ref_hidden = reference(tokens)
        shifted = MTPLoss._shift_labels(labels, 1)
        expected = cross_entropy_loss(
            reference.head(ref_hidden[0]), labels
        ) + 0.3 * cross_entropy_loss(reference.head(ref_hidden[1]), shifted)
        expected.backward()
        result = criterion(model(tokens), labels, sync_all_reduce=False)
        result.backward()
        torch.testing.assert_close(
            model.head.weight.grad.full_tensor(),
            reference.head.weight.grad * 4,
            rtol=1e-5,
            atol=1e-5,
        )
        state = fully_shard.state(model.head)
        for group in state._fsdp_param_groups:
            self.assertIsNone(group._partial_reduce_output)
