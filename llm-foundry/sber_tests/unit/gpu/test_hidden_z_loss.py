# Copyright 2024 LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
"""Minimal end-to-end GPU tests for hidden z-loss on REAL modules.

Four checks that matter:
  1. runs        — an instrumented real module does forward+backward on CUDA;
  2. layerwise   — a `layerwise` coef fires only on the targeted layer_idx;
  3. projections — per-projection (up vs gate) hooks are independently selective;
  4. mtp         — `hidden_z_loss_mtp_coef` lands on the real MTP head and its
                   projection scalars are collected under `mtp<idx>_<proj>`.

Run: `bash run_unit_gpu_tests.sh sber_tests/unit/gpu/test_hidden_z_loss.py`.
"""

import pytest
import torch
from transformers import PretrainedConfig

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.layers.mtp import LlamaDecoderMTPBlock
from llmfoundry.models.ops.included_proj_z_loss import collect_included_proj_z_losses

pytestmark = [pytest.mark.unit, pytest.mark.requires_gpu]


class _Qwen3NextMockConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=512,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_use_short_conv=True,
        linear_sp_implementation="all2all",
        linear_gating_type="gated_rmsnorm",
        linear_sigmoid_gate_scale=1.0,
        linear_attn_o_norm_eps=None,
        linear_conv_init_std=0.006,
        linear_use_legacy_qkvz_layout=False,
        rms_norm_eps=1e-5,
        init_device="cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_use_short_conv = linear_use_short_conv
        self.linear_sp_implementation = linear_sp_implementation
        self.linear_gating_type = linear_gating_type
        self.linear_sigmoid_gate_scale = linear_sigmoid_gate_scale
        self.linear_attn_o_norm_eps = linear_attn_o_norm_eps
        self.linear_conv_init_std = linear_conv_init_std
        self.linear_use_legacy_qkvz_layout = linear_use_legacy_qkvz_layout
        self.rms_norm_eps = rms_norm_eps
        self.init_device = init_device


def _build_attn(qkv_coef=0.0, layerwise=None, layer_idx=0):
    from llmfoundry.models.layers.attention import Qwen3NextGatedDeltaNet

    cfg = _Qwen3NextMockConfig(
        hidden_z_loss_included_proj_coef={
            "qkv": {"base": qkv_coef, "layerwise": layerwise or {}},
        },
    )
    m = Qwen3NextGatedDeltaNet(cfg, layer_idx=layer_idx)
    m.reset_parameters()
    return m.to("cuda").to(torch.bfloat16)


def _build_mlp(up=0.0, gate=0.0, layer_idx=0):
    from llmfoundry.models.layers.ffn import LlamaMLP

    cfg = PretrainedConfig(
        hidden_size=256,
        intermediate_size=512,
        hidden_act="silu",
        init_device="cuda",
        fused_mlp=False,
        fused_mlp_checkpoint_lvl=0,
        pretraining_tp=1,
        num_hidden_layers=4,
        init_type="giga",
        hidden_z_loss_included_proj_coef={
            "up_mlp": {"base": up},
            "gate_mlp": {"base": gate},
        },
    )
    return LlamaMLP(cfg, layer_idx=layer_idx).to("cuda").to(torch.bfloat16)


def _dense_mtp_config(**overrides):
    base = dict(
        init_device="cuda",
        vocab_size=256, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=8,
        intermediate_size=64, dense_intermediate_size=256, first_k_dense_replace=1,
        attention_type="LlamaLatentAttention", linear_attention_type=None,
        full_attention_layers=[0, 1, 2, 3],
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        kv_lora_rank=32, q_lora_rank=48,
        norm_type="GatedNorm", layernorm_type="pre_post", rms_norm_eps=1e-6,
        rope_theta=100000, init_type="deepseek",
        use_mtp=True, mtp_predictor_num=1, mtp_block_type="dense", tp_size=1,
    )
    base.update(overrides)
    return GigaMixConfig(**base)


# 1. everything runs: instrumented real module does fwd+bwd, z-loss active ------

def test_runs_forward_backward_with_zloss():
    m = _build_attn(qkv_coef=1e-2, layer_idx=0)
    m.train()
    x = torch.randn(1, 128, m.hidden_size, device="cuda", dtype=torch.bfloat16)
    out, _, _ = m(x)
    out.float().pow(2).mean().backward()          # backward completes
    loss = m._included_proj_z_loss.get("qkv")
    assert loss is not None and torch.isfinite(loss).all() and loss.item() >= 0.0
    assert m.qkv_proj.weight.grad is not None      # z-loss is in-graph


# 2. works layerwise: coef fires only on the targeted layer --------------------

def test_layerwise_selectivity():
    lw = {2: 5e-3}
    x = None
    on = _build_attn(qkv_coef=0.0, layerwise=lw, layer_idx=2)   # in override
    off = _build_attn(qkv_coef=0.0, layerwise=lw, layer_idx=1)  # base 0 -> off
    assert on._qkv_z_loss_coef == pytest.approx(5e-3)
    assert off._qkv_z_loss_coef == 0.0
    for m, should_fire in [(on, True), (off, False)]:
        m.train()
        x = torch.randn(1, 128, m.hidden_size, device="cuda", dtype=torch.bfloat16)
        m(x)
        got = m._included_proj_z_loss.get("qkv")
        assert (got is not None) == should_fire


# 3. works for individual projections: up vs gate are independent --------------

def test_per_projection_selective():
    mlp = _build_mlp(up=1e-2, gate=0.0)
    mlp.train()
    x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16)
    mlp(x)
    assert mlp._included_proj_z_loss.get("up_mlp") is not None
    assert mlp._included_proj_z_loss.get("gate_mlp") is None

    both = _build_mlp(up=1e-2, gate=1e-2)
    both.train()
    both(x)
    s = both._included_proj_z_loss
    assert s.get("up_mlp") is not None and s.get("gate_mlp") is not None


# 4. works for MTP: coefs land on the real head; scalars collected namespaced --

def test_mtp_head_applies_and_fires():
    cfg = _dense_mtp_config(
        hidden_z_loss_mtp_coef={
            "attn": 1e-7, "dense_mlp": 1e-7, "up_mlp": 1e-2, "gate_mlp": 1e-2,
        },
    )
    block = LlamaDecoderMTPBlock(cfg).to("cuda").to(torch.bfloat16)
    dl = block.mtp_layers[0].decoder_layer
    # coefs applied onto the REAL MTP submodules
    assert dl._hidden_z_loss_attn_coef == 1e-7
    assert dl._hidden_z_loss_dense_mlp_coef == 1e-7
    assert dl.mlp._up_mlp_z_loss_coef == 1e-2
    assert dl.mlp._gate_mlp_z_loss_coef == 1e-2
    assert dl._z_loss_log_prefix == "mtp0_"
    # real forward of the head's MLP -> projection hooks fire, namespaced collect
    dl.mlp.train()
    x = torch.randn(2, 16, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    dl.mlp(x)
    keys = set(collect_included_proj_z_losses(block).keys())
    assert "mtp0_up_mlp" in keys and "mtp0_gate_mlp" in keys
