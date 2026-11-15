"""Config and GFSDP wiring for the opt-in row-owner input embedding."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from tests.unit_tests.config_helpers import load_model_spec
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.config.decode import decode
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.gfsdp._state import _get_module_fsdp_state
from torchtitan.distributed.gfsdp.fully_shard import GFSDPModule
from torchtitan.distributed.gfsdp.row_owner_embedding import RowOwnerEmbedding
from torchtitan.models.gift_moe._base.parallelize import (
    _validate_row_owner_request,
    parallelize_gift_moe,
)


def test_row_owner_control_decodes_independently_and_defaults_off():
    defaults = ParallelismConfig()
    assert defaults.gfsdp_row_owner_embeddings is False
    assert defaults.gfsdp_split_layer0_ffn is False
    assert defaults.gfsdp_split_mtp_ffn is False

    owner_only = decode(
        ParallelismConfig,
        {"gfsdp_row_owner_embeddings": True},
        path="parallelism",
    )
    assert owner_only.gfsdp_row_owner_embeddings is True
    assert owner_only.gfsdp_split_layer0_ffn is False
    assert owner_only.gfsdp_split_mtp_ffn is False


def _validation_inputs(**overrides):
    parallelism = ParallelismConfig(
        fsdp_backend="gfsdp2", gfsdp_row_owner_embeddings=True
    )
    parallel_dims = SimpleNamespace(tp=1, pp=1, cp=1, dp_replicate=1)
    training = TrainingConfig()
    embedding_weight = torch.nn.Parameter(torch.empty(2, 2))
    model = SimpleNamespace(
        enable_weight_tying=False,
        tok_embeddings=SimpleNamespace(weight=embedding_weight),
        lm_head=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(2, 2))),
    )
    for key, value in overrides.items():
        if key.startswith("parallelism__"):
            setattr(parallelism, key.removeprefix("parallelism__"), value)
        elif key.startswith("parallel_dims__"):
            setattr(parallel_dims, key.removeprefix("parallel_dims__"), value)
        elif key.startswith("training__"):
            setattr(training, key.removeprefix("training__"), value)
        else:
            setattr(model, key, value)
    return model, parallel_dims, training, parallelism


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parallelism__fsdp_backend": "fsdp2"}, "fsdp_backend='gfsdp2'"),
        ({"parallel_dims__tp": 2}, "TP/PP/CP1"),
        ({"parallel_dims__pp": 2}, "TP/PP/CP1"),
        ({"parallel_dims__cp": 2}, "TP/PP/CP1"),
        ({"parallel_dims__dp_replicate": 2}, "HSDP replicas"),
        ({"training__enable_cpu_offload": True}, "CPU offload"),
        ({"enable_weight_tying": True}, "untied input weights"),
        ({"tok_embeddings": None}, "requires tok_embeddings"),
    ],
)
def test_enabled_row_owner_rejects_unsupported_modes(overrides, message):
    model, parallel_dims, training, parallelism = _validation_inputs(**overrides)
    with pytest.raises(ValueError, match=message):
        _validate_row_owner_request(
            model,
            parallel_dims=parallel_dims,
            training=training,
            parallelism=parallelism,
        )


class TestGFSDPRowOwnerIntegration(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _parallelize(self, *, row_owner: bool, split_layer0_ffn: bool = False):
        parallelism = ParallelismConfig(
            fsdp_backend="gfsdp2",
            data_parallel_shard_degree=self.world_size,
            gfsdp_row_owner_embeddings=row_owner,
            gfsdp_split_layer0_ffn=split_layer0_ffn,
        )
        model = load_model_spec("gift_moe/debugmodel.yaml").model.build().to(
            self.device_type
        )
        # This checkout falls back to a CUDA mesh name when no accelerator is
        # available; pin the test's actual distributed mesh to CPU/Gloo.
        with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
            parallel_dims = ParallelDims.from_config(
                parallelism, world_size=self.world_size
            )
            parallelize_gift_moe(
                model,
                parallel_dims=parallel_dims,
                training=TrainingConfig(seq_len=8),
                parallelism=parallelism,
                compile_config=CompileConfig(enable=False),
                ac_config=ActivationCheckpointConfig(mode="none"),
                dump_folder=".",
            )
        return model

    @with_comms
    def test_owner_exclusion_and_ffn_chain_keep_default_embedding_chain(self):
        ordinary = self._parallelize(row_owner=False)
        self.assertIsInstance(ordinary.tok_embeddings, GFSDPModule)
        ordinary_root = _get_module_fsdp_state(ordinary)
        self.assertIsNotNone(ordinary_root._shard_order)
        self.assertIs(
            ordinary_root._shard_order[0],
            _get_module_fsdp_state(ordinary.tok_embeddings),
        )

        model = self._parallelize(row_owner=True, split_layer0_ffn=True)
        owner = model.tok_embeddings
        self.assertIsInstance(owner, RowOwnerEmbedding)
        self.assertNotIsInstance(owner, GFSDPModule)
        self.assertEqual(tuple(owner.weight.shape), (2048, 256))
        self.assertEqual(owner.weight.dtype, torch.float32)
        self.assertIn("tok_embeddings.weight", dict(model.named_parameters()))
        self.assertTrue(
            all(
                param.sharded_param is not owner.weight
                for group in owner._sync_groups
                for param in group.fsdp_params
            )
        )
        self.assertIs(
            owner._comm_ctx,
            model._get_fsdp_state()._comm_ctx,
        )

        root_state = _get_module_fsdp_state(model)
        self.assertIsNotNone(root_state._shard_order)
        self.assertIs(
            root_state._shard_order[0],
            _get_module_fsdp_state(model.layers["0"]),
        )
        self.assertIs(
            root_state._shard_order[1],
            _get_module_fsdp_state(model.layers["0"].feed_forward),
        )
