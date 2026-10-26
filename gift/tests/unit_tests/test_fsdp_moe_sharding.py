# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.models.llama4.model import Llama4Model
from torchtitan.models.llama4.parallelize import apply_fsdp


def _build_llama4_model(num_experts: int = 8) -> Llama4Model:
    """Build a tiny Llama4Model with a configurable number of experts."""
    from tests.unit_tests.config_helpers import load_model_spec

    config = load_model_spec(
        "llama4/debugmodel.yaml",
        {
            "config": {
                "layers": [{"block": "moe"}] * 4,
                "blocks": {
                    "moe": {
                        "moe": {
                            "experts": {
                                "num_experts": num_experts,
                                "token_dispatcher": {"type": "all_to_all"},
                            }
                        }
                    }
                },
            }
        },
    ).model
    return Llama4Model(config)


def _expert_shard_dims(model: Llama4Model) -> dict[str, int | None]:
    """Shard dim of each expert param in the first MoE layer (None if unsharded)."""
    for layer in model.layers.values():
        if layer.moe_enabled:
            dims: dict[str, int | None] = {}
            for name, param in layer.moe.experts.named_parameters():
                shard_dims = [
                    p.dim
                    for p in getattr(param, "placements", ())
                    if isinstance(p, Shard)
                ]
                dims[name] = shard_dims[0] if shard_dims else None
            return dims
    raise AssertionError("model has no MoE layer")


# Off the expert axis, w13 (E, 2, F, D) must skip the size-2 gate/up dim and
# shard the hidden dim; w2 (E, D, F) shards its first non-expert dim.
_OFF_EXPERT_AXIS = {"w13_E2FD": 2, "w2_EDF": 1}
_ON_EXPERT_AXIS = {"w13_E2FD": 0, "w2_EDF": 0}


class TestApplyFsdpMoESharding(DTensorTestBase):
    """Test apply_fsdp expert sharding behavior with ep_degree=1 and ep_degree>1."""

    @property
    def world_size(self):
        return 8

    @with_comms
    def test_no_ep_fsdp_gt_num_experts_shards_off_expert_axis(self):
        """ep_degree=1, fsdp_size(8) > num_experts(4) → shard another dim."""
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_llama4_model(num_experts=4).to(self.device_type)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=1,
        )

        self.assertEqual(_expert_shard_dims(model), _OFF_EXPERT_AXIS)

    @with_comms
    def test_no_ep_fsdp_le_num_experts_shards_dim0(self):
        """ep_degree=1, fsdp_size(8) <= num_experts(8) → Shard(0)."""
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_llama4_model(num_experts=8).to(self.device_type)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=1,
        )

        self.assertEqual(_expert_shard_dims(model), _ON_EXPERT_AXIS)

    @with_comms
    def test_with_ep_fsdp_gt_num_experts_shards_off_expert_axis(self):
        """ep_degree=2, efsdp*ep(8) > num_experts(4) → shard another dim."""
        # edp_mesh: 2D mesh [efsdp=4, ep=2], dp_mesh: 1D mesh [8]
        edp_mesh = init_device_mesh(
            self.device_type, (4, 2), mesh_dim_names=("efsdp", "ep")
        )
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_llama4_model(num_experts=4).to(self.device_type)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=2,
            edp_mesh=edp_mesh,
        )

        self.assertEqual(_expert_shard_dims(model), _OFF_EXPERT_AXIS)


if __name__ == "__main__":
    unittest.main()
