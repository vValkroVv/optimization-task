import pytest
import torch
from pathlib import Path
from collections import OrderedDict
from unittest.mock import patch

from mappers import map_gigafsdp_to_fsdp
from mapper_gigafsdp_to_monolith import (
    check_buffer_params,
    _normalize_module_name,
    _normalize_key_prefix,
)


@pytest.fixture
def gigafsdp_checkpoint_dir(temp_dir, simple_config):
    checkpoint_dir = temp_dir / "checkpoint"
    checkpoint_dir.mkdir()

    meta = {
        "fsdp_group_size": 2,
        "main_parameter_shard_bounaries": [512 * 512],
        "main_parameter_orig_shapes": [[[512, 512]]],
        "main_shards_fqns": [["layers.0.self_attn.q_proj.weight"]],
        "main_shards_numels": [[512 * 512]],
    }

    for rank in range(2):
        shard = {
            "state": {
                "model": {
                    "model.meta": meta,
                    "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 512),
                },
                "integrations": {
                    "huggingface": {"model": {"config": {"content": simple_config}}}
                },
            }
        }
        torch.save(shard, checkpoint_dir / f"__{rank}_0.distcp")

    return checkpoint_dir


@pytest.mark.unit
class TestBufferParams:

    def test_valid_buffer(self):
        tensors = [torch.tensor([1.0, 2.0])] * 3
        name = "model.layers.0._gate.expert_bias"

        result_name, result_tensor = check_buffer_params(tensors, name)

        assert result_name == "model.layers.0._gate.expert_bias"
        assert torch.equal(result_tensor, tensors[0])

    def test_non_buffer(self):
        tensors = [torch.randn(10, 10)] * 3
        name = "model.layers.0.weight"

        result_name, result_tensor = check_buffer_params(tensors, name)

        assert result_name is None
        assert result_tensor is None


@pytest.mark.unit
class TestNormalizeFunctions:

    def test_normalize_module_name(self):
        assert _normalize_module_name("model._module.layer") == "model.layer"
        assert _normalize_module_name("model.root_module.layer") == "model.layer"
        assert _normalize_module_name("model.layer") == "model.layer"

    def test_normalize_key_prefix(self):
        assert _normalize_key_prefix("model.model.layers.0") == "model.layers.0"
        assert _normalize_key_prefix("model.layers.0") == "model.layers.0"


@pytest.mark.unit
class TestGigafsdpToFsdpMoENormalization:

    def test_single_shard_moe_weights_are_normalized_in_regular_path(
        self, simple_config, mock_args
    ):
        config = simple_config.copy()
        config.update(
            {
                "hidden_size": 4,
                "num_routed_experts": 2,
                "use_new_expert_weight_layout": True,
            }
        )
        hidden_size = config["hidden_size"]
        num_experts = config["num_routed_experts"]
        intermediate_size = 3

        weight1_legacy = torch.arange(
            num_experts * intermediate_size * hidden_size, dtype=torch.float32
        ).reshape(num_experts * intermediate_size, hidden_size)
        weight2_legacy = torch.arange(
            num_experts * intermediate_size * hidden_size, dtype=torch.float32
        ).reshape(hidden_size, num_experts * intermediate_size)

        expected_weight1 = (
            weight1_legacy.view(num_experts, intermediate_size, hidden_size)
            .mT.contiguous()
            .reshape(hidden_size, -1)
        )
        expected_weight2 = weight2_legacy.mT.contiguous().reshape(-1, hidden_size)

        state_dicts = [
            {
                "state": {
                    "model": {"model.meta": {"dummy": True}},
                    "integrations": {
                        "huggingface": {"model": {"config": {"content": config}}}
                    },
                }
            }
        ]
        unsharded_model = OrderedDict(
            {
                "model.layers.0.block_sparse_moe.experts.weight1": weight1_legacy.clone(),
                "model.layers.0.block_sparse_moe.experts.weight2": weight2_legacy.clone(),
            }
        )

        with patch("mappers.load_gigafsdp_checkpoint", return_value=state_dicts), patch(
            "mapper_gigafsdp_to_monolith.unshard_gigafsdp_model",
            return_value={1: [unsharded_model]},
        ):
            result = map_gigafsdp_to_fsdp("/tmp/fake.distcp", args=mock_args)

        model = result["state"]["model"]
        assert torch.equal(
            model["model.layers.0.block_sparse_moe.experts.weight1"],
            expected_weight1,
        )
        assert torch.equal(
            model["model.layers.0.block_sparse_moe.experts.weight2"],
            expected_weight2,
        )
