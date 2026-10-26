"""DTensor correctness for the MoE sequence auxiliary load-balancing loss.

Regression guard for the TP crash where ``sequence_aux_loss``'s ``scatter_add_``
mixed a plain tensor with a DTensor ``index``. Verifies the DTensor path (which
all-gathers only the sequence axis under ``local_map``) matches a plain-tensor
reference in both value and gradient across TP-only, EP+TP, and a full 3-D mesh.
"""

import unittest

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.schema.models.common.moe.const import ScoreFunc
from torchtitan.models.common.moe.losses import sequence_aux_loss
from torchtitan.models.common.regularization import AddAuxiliaryLoss

_TOP_K = 2
_COEF = 1e-4


class TestSequenceAuxLossDTensor(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def _run_one(self, *, full_dtensor: bool, tp_shard_l: bool, score_func: ScoreFunc):
        torch.manual_seed(0)
        if full_dtensor:
            mesh = init_device_mesh(
                self.device_type, (2, 2, 2), mesh_dim_names=("dp_shard", "cp", "tp")
            )
            # batch Shard(0); seq Shard(1) on cp; tp shards seq (EP+TP) or replicates.
            scores_pl = (Shard(0), Shard(1), Shard(1) if tp_shard_l else Replicate())
            # batch is a mesh axis -> the trailing mean reduces across ranks, so
            # reduction order differs from the single-tensor reference at ULP level.
            rtol = atol = 1e-6
        else:
            mesh = init_device_mesh(self.device_type, (8,), mesh_dim_names=("tp",))
            # tp shards the seq dim L (EP+TP) or replicates (TP-only). Batch is never
            # a mesh axis, so gathering L reproduces the reference exactly.
            scores_pl = (Shard(1) if tp_shard_l else Replicate(),)
            rtol = atol = 0.0

        B, L, E = 4, 16, 8
        gen = torch.Generator(device=self.device_type).manual_seed(42)
        # Continuous scores -> no top-k ties, so the discrete assignment is stable.
        global_scores = torch.rand(B, L, E, device=self.device_type, generator=gen)

        ref = global_scores.clone().detach().requires_grad_(True)
        ref_loss = sequence_aux_loss(
            ref, top_k=_TOP_K, score_func=score_func, coef=_COEF
        )
        ref_loss.backward()

        scores_dt = distribute_tensor(global_scores.clone(), mesh, scores_pl)
        scores_dt = scores_dt.detach().requires_grad_(True)
        loss_dt = sequence_aux_loss(
            scores_dt, top_k=_TOP_K, score_func=score_func, coef=_COEF
        )
        torch.testing.assert_close(
            loss_dt.full_tensor(), ref_loss, rtol=rtol, atol=atol
        )

        loss_dt.backward()
        torch.testing.assert_close(
            scores_dt.grad.full_tensor(), ref.grad, rtol=rtol, atol=atol
        )

    @with_comms
    def test_tp_only_sigmoid(self):
        self._run_one(
            full_dtensor=False, tp_shard_l=False, score_func=ScoreFunc.SIGMOID
        )

    @with_comms
    def test_ep_tp_sigmoid(self):
        self._run_one(full_dtensor=False, tp_shard_l=True, score_func=ScoreFunc.SIGMOID)

    @with_comms
    def test_tp_only_softmax(self):
        self._run_one(
            full_dtensor=False, tp_shard_l=False, score_func=ScoreFunc.SOFTMAX
        )

    @with_comms
    def test_full_dtensor_ep_tp_sigmoid(self):
        self._run_one(full_dtensor=True, tp_shard_l=True, score_func=ScoreFunc.SIGMOID)

    @with_comms
    def test_recorded_total_survives_the_partial_batch_mean(self):
        """The grad-only recording path, which ``full_tensor()`` above bypasses:
        a batch-sharded mean is ``Partial`` over dp, and the trainer's reduce over
        the loss mesh completes it."""
        mesh = init_device_mesh(
            self.device_type, (2, 2, 2), mesh_dim_names=("dp_shard", "cp", "tp")
        )
        gen = torch.Generator(device=self.device_type).manual_seed(42)
        scores = torch.rand(4, 16, 8, device=self.device_type, generator=gen)
        kwargs = dict(top_k=_TOP_K, score_func=ScoreFunc.SIGMOID, coef=_COEF)
        ref = sequence_aux_loss(scores.clone(), **kwargs)

        dt = distribute_tensor(
            scores.clone(), mesh, (Shard(0), Shard(1), Replicate())
        ).requires_grad_(True)
        aux = sequence_aux_loss(dt, **kwargs)
        self.assertTrue(
            any(p.is_partial() for p in aux.placements),
            f"a batch-sharded mean should stay Partial, got {aux.placements}",
        )

        # As the trainer does it: weigh by this rank's token share, record in
        # backward, then sum the totals over the loss mesh.
        reduced = ("dp_shard", "cp")
        share = 1.0 / (mesh["dp_shard"].size() * mesh["cp"].size())
        total = torch.zeros((), device=self.device_type)
        AddAuxiliaryLoss.apply(
            dt.sum(), aux, torch.full((), share, device=self.device_type), total
        ).backward()
        for axis in reduced:
            total = funcol.all_reduce(
                total, reduceOp=c10d.ReduceOp.SUM.name, group=mesh[axis]
            )
        torch.testing.assert_close(total, ref.detach(), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
