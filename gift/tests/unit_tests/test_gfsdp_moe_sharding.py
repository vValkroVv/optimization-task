"""Under GFSDP2, an activation-checkpointed block stays one sharding unit.

Routed experts shard over the expert-FSDP axis and the rest of their block over
the data-parallel one. ``apply_ac`` runs before ``apply_gfsdp`` and rebinds each
block to a ``CheckpointWrapper``, so sharding the experts with a second, nested
``fully_shard`` put GFSDP hooks on a module *inside* the checkpointed region.
There they re-fire on recompute and emit a different sequence of dispatcher ops
than the original forward, which selective activation checkpointing rejects with
"encountered during backward but not found in storage".

``param_meshes`` keeps both parameter sets in one unit on the wrapper instead.
These tests pin the structure that fixes it: no sharding state below a
checkpoint boundary, and the experts in their own group on their own mesh.
"""

import unittest

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.gfsdp._state import _get_module_fsdp_state
from tests.unit_tests.config_helpers import load_model_spec
from torchtitan.models.gift_moe._base.parallelize import apply_gfsdp

_EP_DEGREE = 2


def _moe_blocks(model):
    """Yield the ``(wrapper, block)`` pairs of every MoE layer."""
    for layer in model.layers.values():
        block = getattr(layer, "_checkpoint_wrapped_module", layer)
        if getattr(block, "moe_enabled", False):
            yield layer, block


def _states_below(module):
    """Yield the GFSDP states of every module strictly inside ``module``."""
    for submodule in module.modules():
        if submodule is module:
            continue
        if (state := _get_module_fsdp_state(submodule)) is not None:
            yield submodule, state


class TestApplyGFSDPMoESharding(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def _shard(self, *, ep_degree: int):
        dp_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        # The expert mesh has to span the whole world, so build it 2D and take
        # the ``efsdp`` slice -- which is the shape ``apply_gfsdp`` expects.
        edp_mesh = (
            init_device_mesh(
                self.device_type,
                (self.world_size // ep_degree, ep_degree),
                mesh_dim_names=("efsdp", "ep"),
            )["efsdp"]
            if ep_degree > 1
            else None
        )
        model = (
            load_model_spec("gift_moe/debugmodel.yaml")
            .model.build()
            .to(self.device_type)
        )
        apply_ac(model, ActivationCheckpointConfig(mode="selective"))
        expert_param_counts = [
            len(list(block.moe.experts.parameters())) for _, block in _moe_blocks(model)
        ]
        apply_gfsdp(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=ep_degree,
            edp_mesh=edp_mesh,
        )
        return model, dp_mesh, edp_mesh, expert_param_counts

    @with_comms
    def test_activation_checkpointing_is_applied_at_all(self):
        """Guard the premise: without a wrapper the rest proves nothing."""
        model = (
            load_model_spec("gift_moe/debugmodel.yaml")
            .model.build()
            .to(self.device_type)
        )
        apply_ac(model, ActivationCheckpointConfig(mode="selective"))

        wrappers = [m for m in model.modules() if isinstance(m, CheckpointWrapper)]

        self.assertTrue(wrappers)
        self.assertTrue(any(True for _ in _moe_blocks(model)))

    @with_comms
    def test_no_sharding_state_inside_a_checkpointed_region(self):
        """The whole point: a nested unit here is what broke selective AC."""
        model, _, _, _ = self._shard(ep_degree=_EP_DEGREE)

        for module in model.modules():
            if not isinstance(module, CheckpointWrapper):
                continue
            nested = [fqn for fqn, _ in _states_below(module)]
            self.assertEqual(nested, [], f"sharding state inside {module}")

    @with_comms
    def test_experts_get_a_sibling_group_on_the_expert_mesh(self):
        model, dp_mesh, edp_mesh, expert_param_counts = self._shard(
            ep_degree=_EP_DEGREE
        )

        for (wrapper, _), num_expert_params in zip(
            _moe_blocks(model), expert_param_counts, strict=True
        ):
            state = _get_module_fsdp_state(wrapper)
            self.assertIsNotNone(state)
            dense_group, expert_group = state._fsdp_param_groups

            self.assertIs(dense_group.mesh_info.mesh, dp_mesh)
            self.assertIs(expert_group.mesh_info.mesh, edp_mesh)
            self.assertEqual(len(expert_group.fsdp_params), num_expert_params)

    @with_comms
    def test_without_expert_parallelism_a_block_keeps_one_group(self):
        """``ep_degree=1`` never took the nested path, and must not change."""
        model, dp_mesh, _, _ = self._shard(ep_degree=1)

        for wrapper, _ in _moe_blocks(model):
            state = _get_module_fsdp_state(wrapper)
            self.assertIsNotNone(state)
            self.assertEqual(len(state._fsdp_param_groups), 1)
            self.assertIs(state._fsdp_param_groups[0].mesh_info.mesh, dp_mesh)

    @with_comms
    def test_lazy_init_accepts_the_sibling_groups(self):
        """Sibling groups must not land in one data buffer.

        ``validate_shared_state`` refuses groups that share a buffer but are
        unsharded together, and buffers are pooled per shard process group --
        so this passing is what says the two axes really are pooled apart.
        """
        model, _, _, _ = self._shard(ep_degree=_EP_DEGREE)

        model.lazy_init()

        for wrapper, _ in _moe_blocks(model):
            state = _get_module_fsdp_state(wrapper)
            dense_group, expert_group = state._fsdp_param_groups
            self.assertIsNot(dense_group.data_buffer_ctx, expert_group.data_buffer_ctx)
            self.assertNotEqual(dense_group._module_fqn, expert_group._module_fqn)
            self.assertIn("efsdp", expert_group._module_fqn)


if __name__ == "__main__":
    unittest.main()
