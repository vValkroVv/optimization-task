"""Fused QKV works on Qwen3 MoE layers through sharding and HF conversion."""

import torch

from tests.unit_tests.config_helpers import load_model_spec
from torchtitan.models.common.attention import FusedQKVLinear, QKVLinear
from torchtitan.models.qwen3.sharding import set_qwen3_sharding_config
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter


def _moe_model_config(*, fuse_qkv: bool):
    """A debugmodel_moe-shaped config, two layers, with the chosen Q/K/V."""
    block = {"moe": {"router": {"top_k": 2}}}
    if fuse_qkv:
        block["attention"] = {"qkv_linear": {"type": "fused"}}
    config = {"blocks": {"moe": block}, "layers": [{"block": "moe", "repeat": 2}]}
    return load_model_spec("qwen3/debugmodel_moe.yaml", {"config": config}).model


def test_moe_layers_default_to_separate_qkv():
    for layer in _moe_model_config(fuse_qkv=False).layers:
        assert isinstance(layer.attention.qkv_linear, QKVLinear.Config)


def test_moe_layers_fuse_qkv_matches_the_dense_builder():
    for layer in _moe_model_config(fuse_qkv=True).layers:
        qkv = layer.attention.qkv_linear
        assert isinstance(qkv, FusedQKVLinear.Config)
        # (n_heads + 2 * n_kv_heads) * head_dim, as FusedQKVLinear validates.
        assert qkv.wqkv.out_features == (16 + 2 * 8) * 128
        assert qkv.wqkv.in_features == 256


def test_fused_moe_model_shards_and_runs():
    config = _moe_model_config(fuse_qkv=True)
    set_qwen3_sharding_config(
        config, loss_parallel=False, enable_sp=True, enable_ep=True
    )
    # Attention is sharded on MoE blocks too, via the fused projection.
    assert config.layers[0].attention.qkv_linear.wqkv.sharding_config is not None

    model = config.build()
    tokens = torch.randint(0, config.tok_embeddings.num_embeddings, (2, 16))
    assert model(tokens).shape == (2, 16, config.lm_head.out_features)


def test_fused_moe_state_dict_round_trips_through_hf():
    config = _moe_model_config(fuse_qkv=True)
    adapter = Qwen3StateDictAdapter(config, None)
    assert adapter.fuse_qkv

    state_dict = config.build().state_dict()
    fused_key = "layers.0.attention.qkv_linear.wqkv.weight"
    assert fused_key in state_dict

    hf = adapter.to_hf(state_dict)
    # The fused weight is split back into the three HF projections, and the
    # expert weights survive alongside it.
    for proj in ("q", "k", "v"):
        assert f"model.layers.0.self_attn.{proj}_proj.weight" in hf
    assert any("mlp.experts" in key for key in hf)

    assert torch.equal(adapter.from_hf(hf)[fused_key], state_dict[fused_key])
