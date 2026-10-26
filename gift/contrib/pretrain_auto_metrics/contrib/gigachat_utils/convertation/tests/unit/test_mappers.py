import pytest
import torch
from unittest.mock import patch, MagicMock, Mock
from collections import OrderedDict
import argparse
import copy

from mappers import (
    map_base_to_packed_attn,
    map_packed_to_base_attn,
    remove_fsdp_optimizer,
    map_tp_to_tp,
    map_ema_to_weights,
)


@pytest.mark.unit
class TestAttentionConversions:

    def test_base_to_packed_basic(self, simple_fsdp_checkpoint, mock_args):
        result = map_base_to_packed_attn(simple_fsdp_checkpoint, args=mock_args)
        model = result["state"]["model"]

        assert "model.layers.0.self_attn.qkv_proj.weight" in model
        assert "model.layers.1.self_attn.qkv_proj.weight" in model

        assert "model.layers.0.self_attn.q_proj.weight" not in model
        assert "model.layers.0.self_attn.k_proj.weight" not in model
        assert "model.layers.0.self_attn.v_proj.weight" not in model

    def test_packed_to_base_basic(self, simple_fsdp_checkpoint, mock_args):
        packed = map_base_to_packed_attn(simple_fsdp_checkpoint, args=mock_args)

        result = map_packed_to_base_attn(packed, args=mock_args)
        model = result["state"]["model"]

        assert "model.layers.0.self_attn.q_proj.weight" in model
        assert "model.layers.0.self_attn.k_proj.weight" in model
        assert "model.layers.0.self_attn.v_proj.weight" in model

        assert "model.layers.0.self_attn.qkv_proj.weight" not in model

    def test_roundtrip_preserves_weights(self, simple_fsdp_checkpoint, mock_args):
        original = copy.deepcopy(simple_fsdp_checkpoint)

        packed = map_base_to_packed_attn(original, args=mock_args)
        restored = map_packed_to_base_attn(packed, args=mock_args)

        orig_q = original["state"]["model"]["model.layers.0.self_attn.q_proj.weight"]
        rest_q = restored["state"]["model"]["model.layers.0.self_attn.q_proj.weight"]

        assert torch.allclose(orig_q, rest_q, atol=1e-6)


@pytest.mark.unit
class TestOptimizerOps:

    def test_remove_optimizer(self, simple_fsdp_checkpoint, mock_args):
        result = remove_fsdp_optimizer(simple_fsdp_checkpoint, args=mock_args)
        assert "optimizers" not in result["state"]
        assert "model" in result["state"]

    def test_optimizer_already_removed(self, simple_fsdp_checkpoint, mock_args):
        del simple_fsdp_checkpoint["state"]["optimizers"]

        with pytest.raises(KeyError):
            remove_fsdp_optimizer(simple_fsdp_checkpoint, args=mock_args)


@pytest.mark.unit
class TestTpConversions:

    def test_tp_same_size(self, simple_fsdp_checkpoint, mock_args):
        mock_args.input_tp_size = 1
        mock_args.output_tp_size = 1

        result = map_tp_to_tp(simple_fsdp_checkpoint, args=mock_args)

        assert result == simple_fsdp_checkpoint


@pytest.mark.unit
class TestEmaHandling:

    def test_delete_ema_keeps_original(self, simple_fsdp_checkpoint, mock_args):
        model = simple_fsdp_checkpoint["state"]["model"]
        original_q = model["model.layers.0.self_attn.q_proj.weight"].clone()

        for key in list(model.keys()):
            if "weight" in key:
                model[key + ".ema"] = model[key].clone() * 2.0

        mock_args.use_ema = False
        result = map_ema_to_weights(simple_fsdp_checkpoint, args=mock_args)

        result_q = result["state"]["model"]["model.layers.0.self_attn.q_proj.weight"]
        assert torch.allclose(original_q, result_q)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
