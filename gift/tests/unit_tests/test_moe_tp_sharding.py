# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TP sharding and checkpointing of the fused ``w13`` routed-expert weight.

``w13_E2FD`` stores gate and up as ``(E, 2, F, D)``. On the TP axis it is a
plain ``Shard(2)`` of the hidden dim: every rank holds matching gate/up rows,
the local shard folds into the ``(E, 2F, D)`` weight the kernels read, and
DCP can save and reshard it. A fused ``(E, 2F, D)`` weight would need a
strided shard for the same pairing, which DCP cannot describe as one chunk.
"""

import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.models.common.moe.sharding import GROUPED_EXPERTS_PARAM_LAYOUT
from torchtitan.protocols.sharding import resolve_placement


NUM_EXPERTS, HIDDEN, DIM, TOKENS = 2, 8, 6, 5


def _expert_ffn(x_TD, w13_2FD, w2_DF):
    """One expert's SwiGLU FFN, gate and up read as the two halves of w13."""
    h_T2F = x_TD @ w13_2FD.t()
    gate_TF, up_TF = h_T2F.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate_TF) * up_TF) @ w2_DF.t()


def _fold(w13_E2FD):
    """The ``(E, 2F, D)`` view the grouped GEMMs read, as ``GroupedExperts`` does."""
    num_experts, _, hidden, dim = w13_E2FD.shape
    return w13_E2FD.view(num_experts, 2 * hidden, dim)


def _weights():
    generator = torch.Generator().manual_seed(42)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    gate = randn(NUM_EXPERTS, HIDDEN, DIM)
    up = randn(NUM_EXPERTS, HIDDEN, DIM)
    w13_E2FD = torch.stack([gate, up], dim=1)
    w2_EDF = randn(NUM_EXPERTS, DIM, HIDDEN)
    x_TD = randn(TOKENS, DIM)
    return gate, up, w13_E2FD, w2_EDF, x_TD


class TestGroupedExpertsTPSharding(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cpu"

    def _shard(self, tensor, name):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("tp",))
        return distribute_tensor(
            tensor, mesh, [resolve_placement(GROUPED_EXPERTS_PARAM_LAYOUT[name])]
        )

    @with_comms
    def test_shard_keeps_gate_up_pairs_and_folds_for_free(self):
        gate, up, w13_E2FD, _, _ = _weights()
        local = self._shard(w13_E2FD, "w13_E2FD").to_local()

        rows = HIDDEN // self.world_size
        lo = self.rank * rows
        self.assertEqual(local.shape, (NUM_EXPERTS, 2, rows, DIM))
        torch.testing.assert_close(local[:, 0], gate[:, lo : lo + rows])
        torch.testing.assert_close(local[:, 1], up[:, lo : lo + rows])

        # The kernels read "[gate; up]" along the hidden dim, through a view.
        folded = _fold(local)
        self.assertEqual(folded.data_ptr(), local.data_ptr())
        gate_local, up_local = folded.chunk(2, dim=1)
        torch.testing.assert_close(gate_local, gate[:, lo : lo + rows])
        torch.testing.assert_close(up_local, up[:, lo : lo + rows])

    @with_comms
    def test_tp_matches_unsharded_ffn(self):
        """Colwise w13 + rowwise w2 + all-reduce reproduces the dense FFN."""
        _, _, w13_E2FD, w2_EDF, x_TD = _weights()
        reference = _expert_ffn(x_TD, _fold(w13_E2FD)[0], w2_EDF[0])

        w13_local = _fold(self._shard(w13_E2FD, "w13_E2FD").to_local())
        w2_local = self._shard(w2_EDF, "w2_EDF").to_local()
        partial = _expert_ffn(x_TD, w13_local[0], w2_local[0])
        dist.all_reduce(partial)
        torch.testing.assert_close(partial, reference)

    @with_comms
    def test_dcp_round_trip_reshards_tp_to_expert_parallel(self):
        """A TP checkpoint saves, and loads back under an EP-style Shard(0)."""
        _, _, w13_E2FD, _, _ = _weights()
        sharded = self._shard(w13_E2FD, "w13_E2FD")

        # One directory for both ranks; torch's with_temp_dir moves between
        # releases, so rank 0 picks it and broadcasts.
        checkpoint_dir = [tempfile.mkdtemp() if self.rank == 0 else None]
        dist.broadcast_object_list(checkpoint_dir, src=0)
        try:
            dcp.save({"w13_E2FD": sharded}, checkpoint_id=checkpoint_dir[0])
            loaded = distribute_tensor(
                torch.zeros_like(w13_E2FD), sharded.device_mesh, [Shard(0)]
            )
            dcp.load({"w13_E2FD": loaded}, checkpoint_id=checkpoint_dir[0])
            torch.testing.assert_close(loaded.full_tensor(), w13_E2FD)
        finally:
            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(checkpoint_dir[0], ignore_errors=True)
