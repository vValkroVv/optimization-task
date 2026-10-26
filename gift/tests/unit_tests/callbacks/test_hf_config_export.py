# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The exported HF ``config.json`` describes the model the weights came from.

Two formats: a plain MLA/MoE/RMSNorm GIFT exports as ``deepseek_v3`` (stock
transformers / opensource sglang); anything DeepSeek-V3 cannot express — GDN
layers, gated attention, gated or sandwich norms, MLA latent scaling, logit
soft-cap — exports as ``gigachat3_5`` with the fields PAM's HF modeling and
sglang's ``gigachat35.py`` read. A wrong field here loads fine and computes a
different model, which is exactly what autometrics would then score.
"""

import math
from torchtitan.config.schema.models.common.attention.linear.gdn import (
    GatedDeltaNetConfig,
)
from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig
from dataclasses import replace
from types import SimpleNamespace

import pytest

from torchtitan.components.checkpoint import hf_remote_code_files, resolve_hf_export
from tests.unit_tests.config_helpers import load_model_spec
from torchtitan.models.common import NoScaling, YarnScaling
from torchtitan.models.gift_moe._base.hf_config import build_hf_config
from torchtitan.models.gift_moe._base.state_dict_adapter import GiftMoEStateDictAdapter
from torchtitan.models.gift_moe.gift_3_5.state_dict_adapter import (
    GiftHybridStateDictAdapter,
)

# What sglang's gigachat35.py reads from the config (models/gigachat35.py +
# configs/hybrid_gdn.py); a config missing one of these does not even load.
_SGLANG_REQUIRED = {
    "hidden_size",
    "num_hidden_layers",
    "n_routed_experts",
    "vocab_size",
    "rms_norm_eps",
    "q_lora_rank",
    "num_nextn_predict_layers",
    "layer_types",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
}


def _base():
    return load_model_spec("gift_moe/debugmodel.yaml").model


def _hybrid(num_mtp: int = 0):
    mtp = {"mtp": {"layers": [{"block": "mtp", "repeat": num_mtp}]}} if num_mtp else {}
    return load_model_spec("gift_moe/gift_3_5/debugmodel.yaml", {"config": mtp}).model


def _gift4():
    return load_model_spec("gift_moe/gift_4/debugmodel.yaml").model


# --- format selection ------------------------------------------------------ #


def test_plain_mla_moe_exports_deepseek_v3_and_needs_no_remote_code():
    hf = GiftMoEStateDictAdapter.hf_config(_base())
    assert hf["model_type"] == "deepseek_v3"
    assert hf["architectures"] == ["DeepseekV3ForCausalLM"]
    assert "auto_map" not in hf and "layer_types" not in hf
    assert hf_remote_code_files(hf) == ()


def test_hybrid_exports_gigachat3_5_with_auto_map_and_remote_code():
    hf = GiftHybridStateDictAdapter.hf_config(_hybrid())
    assert hf["model_type"] == "gigachat3_5"
    assert hf["architectures"] == ["Gigachat35ForCausalLM"]
    assert hf["auto_map"]["AutoModelForCausalLM"] == (
        "modeling_gigachat3_5.Gigachat35ForCausalLM"
    )
    assert hf_remote_code_files(hf) == (
        "configuration_gigachat3_5.py",
        "modeling_gigachat3_5.py",
    )


def test_gift4_soft_cap_is_exported_and_forces_gigachat():
    hf = GiftHybridStateDictAdapter.hf_config(_gift4())
    assert hf["model_type"] == "gigachat3_5"
    assert (
        hf["final_logit_softcapping"] is not None and hf["final_logit_softcapping"] > 0
    )


# --- the hybrid pattern ---------------------------------------------------- #


def test_layer_types_follow_the_attention_classes_of_the_model():
    model = _hybrid()
    hf = GiftHybridStateDictAdapter.hf_config(model)
    expected_full = [
        i
        for i, layer in enumerate(model.layers)
        if isinstance(layer.attention, MLAConfig)
    ]
    assert hf["full_attention_layers"] == expected_full
    assert hf["layer_types"] == [
        "full_attention" if i in expected_full else "linear_attention"
        for i in range(len(model.layers))
    ]
    assert hf["num_hidden_layers"] == len(model.layers)
    assert hf["linear_attention_type"] == "Qwen3NextGatedDeltaNet"


def test_gdn_dims_and_gating_come_from_the_gdn_config():
    model = _hybrid()
    gdn = next(
        layer.attention
        for layer in model.layers
        if isinstance(layer.attention, GatedDeltaNetConfig)
    )
    hf = GiftHybridStateDictAdapter.hf_config(model)
    assert hf["linear_num_key_heads"] == gdn.num_k_heads
    assert hf["linear_num_value_heads"] == gdn.num_v_heads
    assert hf["linear_key_head_dim"] == gdn.head_k_dim
    assert hf["linear_value_head_dim"] == gdn.head_v_dim
    assert hf["linear_conv_kernel_dim"] == gdn.conv_size
    # gift_3_5: sigmoid gate, zero-centered, output x2 -> HF's scale on norm output.
    assert hf["linear_gating_type"] == "gated_rmsnorm_sigmoid_zero_centered"
    assert hf["linear_sigmoid_gate_scale"] == gdn.o_norm_output_multiplier
    assert hf["linear_use_legacy_qkvz_layout"] is False


def test_gated_sandwich_norms_select_the_gigachat_norm_fields():
    hf = GiftHybridStateDictAdapter.hf_config(_hybrid())
    assert hf["norm_type"] == "ZeroCenteredGatedNorm"
    assert hf["layernorm_type"] == "pre_post"
    assert hf["layernorm_gating_weight"] == 2.0
    assert hf["gated_attention"] is True
    assert hf["use_mla_scaling_factor"] is True


def test_gift_4_exports_plain_gated_norms():
    hf = GiftHybridStateDictAdapter.hf_config(_gift4())
    assert hf["norm_type"] == "GatedNorm"
    assert hf["linear_gating_type"] == "gated_rmsnorm_sigmoid"


def test_derived_dims_match_gigachat35config_init():
    # Gigachat35Config.__init__ recomputes these from the parts; writing the
    # same values keeps a hand-written and a round-tripped config identical.
    hf = GiftHybridStateDictAdapter.hf_config(_hybrid())
    assert hf["qk_head_dim"] == hf["qk_nope_head_dim"] + hf["qk_rope_head_dim"]
    assert hf["head_dim"] == hf["qk_rope_head_dim"]
    assert hf["num_key_value_heads"] == hf["num_attention_heads"]
    assert hf["rope_interleave"] is True


def test_every_field_sglang_reads_is_present():
    hf = GiftHybridStateDictAdapter.hf_config(_hybrid())
    assert _SGLANG_REQUIRED <= set(hf), _SGLANG_REQUIRED - set(hf)


def test_mtp_heads_export_their_count_and_ffn_kind():
    hf = GiftHybridStateDictAdapter.hf_config(_hybrid(num_mtp=1))
    assert hf["num_nextn_predict_layers"] == 1
    assert hf["nextn_is_sparse"] is False  # gift_3_5 heads take a dense FFN
    without = GiftHybridStateDictAdapter.hf_config(_hybrid())
    assert without["num_nextn_predict_layers"] == 0
    assert "nextn_is_sparse" not in without


def test_optional_gigachat_round_trip_through_pam_config_class():
    """If PAM's remote code is checked out, its constructor must accept our
    config and derive the same layer_types / dims."""
    model = _hybrid(num_mtp=1)
    try:
        _, code_dir = resolve_hf_export(GiftHybridStateDictAdapter, model, "")
    except ValueError:
        pytest.skip("contrib/pretrain_auto_metrics is not checked out")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "configuration_gigachat3_5", f"{code_dir}/configuration_gigachat3_5.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hf = GiftHybridStateDictAdapter.hf_config(model)
    hf.pop("auto_map")
    cfg = module.Gigachat35Config(**hf)
    assert cfg.layer_types == hf["layer_types"]
    assert cfg.full_attention_layers == hf["full_attention_layers"]
    assert cfg.qk_head_dim == hf["qk_head_dim"] and cfg.head_dim == hf["head_dim"]


# --- refusing to guess ----------------------------------------------------- #


def test_mixed_norm_classes_are_refused():
    model = _hybrid()
    layers = list(model.layers)
    from torchtitan.models.common.norms import RMSNorm

    layers[0] = replace(
        layers[0], attention_norm=RMSNorm.Config(normalized_shape=model.dim)
    )
    # ``layers`` is derived (init=False), so the edit goes onto the fresh copy.
    model.layers = layers
    with pytest.raises(ValueError, match="norm class differs"):
        build_hf_config(model)


def test_dense_layers_must_be_a_prefix():
    model = _base()
    layers = list(model.layers)
    layers[0], layers[-1] = layers[-1], layers[0]  # move the dense layer to the end
    model.layers = layers
    with pytest.raises(ValueError, match="dense prefix"):
        build_hf_config(model)


def test_remote_code_is_reported_missing_before_the_run_starts(tmp_path):
    model = _hybrid()
    with pytest.raises(ValueError, match="modeling_gigachat3_5.py"):
        resolve_hf_export(GiftHybridStateDictAdapter, model, str(tmp_path))
    for name in ("configuration_gigachat3_5.py", "modeling_gigachat3_5.py"):
        (tmp_path / name).write_text("")
    _, code_dir = resolve_hf_export(GiftHybridStateDictAdapter, model, str(tmp_path))
    assert code_dir == str(tmp_path)


# --- rope / q_lora, on the 3B flavour that trains with YaRN ---------------- #


def _yarn_get_mscale(scale: float, mscale: float) -> float:
    """HF's helper, verbatim: modeling_deepseek.yarn_get_mscale."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def test_yarn_training_exports_yarn_scaling_that_reproduces_the_softmax_scale():
    model = load_model_spec("gift_moe/3b.yaml").model
    hf = GiftMoEStateDictAdapter.hf_config(model)
    scaling = hf["rope_scaling"]
    assert scaling is not None and scaling["type"] == "yarn"
    rope = model.rope.scaling
    assert isinstance(rope, YarnScaling)
    assert scaling["factor"] == rope.rope_factor
    assert scaling["original_max_position_embeddings"] == rope.original_seq_len
    mscale = model.layers[0].attention.mscale
    hf_correction = _yarn_get_mscale(scaling["factor"], scaling["mscale_all_dim"]) ** 2
    ours = (0.1 * mscale * math.log(rope.rope_factor) + 1.0) ** 2
    assert hf_correction == pytest.approx(ours, rel=1e-12)


def test_unscaled_rope_exports_none_and_qlora_zero_exports_none():
    model = _base()
    hf = GiftMoEStateDictAdapter.hf_config(model)
    if isinstance(model.rope.scaling, NoScaling):
        assert hf["rope_scaling"] is None
    attention = model.layers[0].attention
    assert hf["q_lora_rank"] == (attention.q_lora_rank or None)


def test_inert_yarn_is_unrepresentable():
    with pytest.raises(ValueError, match="rope_factor"):
        YarnScaling(rope_factor=1.0, original_seq_len=4096)


def test_legacy_namespace_doubles_are_no_longer_accepted():
    # hf_config branches on the real config classes now; a duck-typed double
    # is refused instead of producing a config for a model that does not exist.
    fake = SimpleNamespace(layers=[SimpleNamespace(attention=object(), moe=None)])
    with pytest.raises(ValueError, match="no HF export"):
        build_hf_config(fake)
