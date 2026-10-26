"""Round-trip tests for the reverse converter (HF safetensors → local monolith).

Strategy: run the forward per-layer transform (`_DeepseekPerLayerTransformer.
rename_and_split`, the local→HF mapper) and then the reverse (`HFToLocalPerLayer
Mapper.map_chunk`), and assert the tensors and keys round-trip exactly.

All tensors are created in bf16 up front; every transform below is a pure
reshape/permute/cat/transpose (value-preserving), so exact equality holds.

Covers GigaChat 3.0 (DeepSeek MLA) and GigaChat 3.5 (Qwen3NextGatedDeltaNet,
KimiDeltaAttention, both qkvz layouts), MoE (ggemm + sonic), and MTP heads.
"""
import pytest
import torch

from transformers_utils.deepseek import _DeepseekPerLayerTransformer
from mapper_hf_to_deepseek_moe import (
    HFToLocalPerLayerMapper,
    conv_order_to_interleaved_qkv_conv,
    convert_hf_config_to_local,
    merge_kimi_qkv,
    resolve_mtp_block_type_from_hf_config,
    split_conv_order_to_separate,
    split_in_proj_qkvz,
)


def _t(*shape):
    """Deterministic bf16 tensor (bf16-exact values, so round-trips compare exactly)."""
    n = 1
    for s in shape:
        n *= s
    return torch.arange(n, dtype=torch.float32).reshape(*shape).to(torch.bfloat16)


def _assert_same(a, b, name=""):
    assert a.shape == b.shape, f"{name}: shape {tuple(a.shape)} != {tuple(b.shape)}"
    assert torch.equal(a, b), f"{name}: values differ"


def _assert_roundtrip(local_in, local_out):
    """Assert two local state dicts have identical keys and tensors."""
    assert set(local_in.keys()) == set(local_out.keys()), (
        f"key mismatch:\n only in forward-input: {sorted(set(local_in) - set(local_out))}\n"
        f" only in reverse-output: {sorted(set(local_out) - set(local_in))}"
    )
    for k in local_in:
        _assert_same(local_in[k], local_out[k], k)


# Linear-attention head layout used across the 3.5 tests.
_LIN = dict(
    linear_num_key_heads=2,
    linear_num_value_heads=4,   # g = 2
    linear_key_head_dim=4,
    linear_value_head_dim=4,
)


def _base_cfg(**overrides):
    """Minimal config carrying every dim the forward/reverse mappers read."""
    cfg = {
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "qk_rope_head_dim": 4,
        "qk_nope_head_dim": 4,
        "v_head_dim": 4,
        "kv_lora_rank": 8,
        "q_lora_rank": 8,
        "n_routed_experts": 2,
        "first_k_dense_replace": 1,
        "moe_intermediate_size": 8,
        "linear_conv_kernel_dim": 4,
    }
    cfg.update(overrides)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# A. Isolated tensor-math round-trips (forward helper ↔ reverse free function)
# ═══════════════════════════════════════════════════════════════════════════

class TestLinearAttentionMath:
    def _tr(self, **extra):
        cfg = _base_cfg(linear_attention_type="Qwen3NextGatedDeltaNet", **_LIN, **extra)
        return _DeepseekPerLayerTransformer(cfg)

    def test_qkvz_split_inverts_forward_pack(self):
        tr = self._tr()
        n_h, n_v = _LIN["linear_num_key_heads"], _LIN["linear_num_value_heads"]
        d_k, d_v = _LIN["linear_key_head_dim"], _LIN["linear_value_head_dim"]
        g = n_v // n_h
        H = 16
        qkv = _t(n_h * (2 * d_k + g * d_v), H)
        z = _t(n_h * (g * d_v), H)

        # forward: accumulate qkv then z -> in_proj_qkvz
        base = "model.layers.1"
        out = tr._maybe_accumulate_and_return_qkvz(base, "self_attn", "qkv", qkv)
        assert out == {}
        out = tr._maybe_accumulate_and_return_qkvz(base, "self_attn", "z", z)
        fused = out[f"{base}.self_attn.in_proj_qkvz.weight"]

        qkv2, z2 = split_in_proj_qkvz(fused, n_h, n_v, d_k, d_v)
        _assert_same(qkv, qkv2, "qkv_proj")
        _assert_same(z, z2, "z_proj")

    def test_conv_order_split_inverts_forward_reorder(self):
        tr = self._tr()
        n_h, n_v = _LIN["linear_num_key_heads"], _LIN["linear_num_value_heads"]
        d_k, d_v = _LIN["linear_key_head_dim"], _LIN["linear_value_head_dim"]
        g = n_v // n_h
        K = 4
        qkv_conv = _t(n_h * (2 * d_k + g * d_v), 1, K)

        conv = tr._interleaved_qkv_to_conv_order(qkv_conv)
        qkv_conv2 = conv_order_to_interleaved_qkv_conv(conv, n_h, n_v, d_k, d_v)
        _assert_same(qkv_conv, qkv_conv2, "qkv_conv1d")

    def test_legacy_conv_split_inverts_cat(self):
        # forward pack_conv_weights does cat([q, k, v], dim=0)
        n_h, n_v = _LIN["linear_num_key_heads"], _LIN["linear_num_value_heads"]
        d_k, d_v = _LIN["linear_key_head_dim"], _LIN["linear_value_head_dim"]
        K = 4
        q = _t(n_h * d_k, 1, K)
        k = _t(n_h * d_k, 1, K)
        v = _t(n_v * d_v, 1, K)
        conv = torch.cat([q, k, v], dim=0)
        q2, k2, v2 = split_conv_order_to_separate(conv, n_h, n_v, d_k, d_v)
        _assert_same(q, q2, "q_conv1d")
        _assert_same(k, k2, "k_conv1d")
        _assert_same(v, v2, "v_conv1d")

    def test_kimi_merge_inverts_forward_split(self):
        cfg = _base_cfg(linear_attention_type="KimiDeltaAttention", **_LIN)
        tr = _DeepseekPerLayerTransformer(cfg)
        n_h, n_v = _LIN["linear_num_key_heads"], _LIN["linear_num_value_heads"]
        d_k, d_v = _LIN["linear_key_head_dim"], _LIN["linear_value_head_dim"]
        g = n_v // n_h
        H = 16
        qkv = _t(n_h * (2 * d_k + g * d_v), H)
        q, k, v = tr._interleaved_qkv_to_kimi_order(qkv)
        qkv2 = merge_kimi_qkv(q, k, v, n_h, n_v, d_k, d_v)
        _assert_same(qkv, qkv2, "qkv_proj")


# ═══════════════════════════════════════════════════════════════════════════
# B. Config back-conversion
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigBackConversion:
    def test_gigachat_fields_inverted(self):
        hf = {
            "model_type": "gigachat3_5",
            "architectures": ["Gigachat35ForCausalLM"],
            "auto_map": {"AutoConfig": "x"},
            "linear_attention_type": "Qwen3NextGatedDeltaNet",
            "full_attention_layers": [0],
            "gated_attention": True,
            "use_shared_expert_sigmoid": True,
            "layernorm_type": "pre_post",
            "n_routed_experts": 8,
            "moe_intermediate_size": 128,
            "intermediate_size": 512,
            "aux_loss_alpha": 0.001,
            "final_logit_softcapping": 30.0,
            "q_lora_rank": None,
            "num_nextn_predict_layers": 1,
            "nextn_is_sparse": True,
        }
        local = convert_hf_config_to_local(hf, moe_layout="ggemm")

        assert local["model_type"] == "giga_mix"
        assert "architectures" not in local and "auto_map" not in local
        # gigachat markers survive
        assert local["gated_attention"] is True
        assert local["use_shared_expert_sigmoid"] is True
        assert local["layernorm_type"] == "pre_post"
        assert local["linear_attention_type"] == "Qwen3NextGatedDeltaNet"
        assert local["full_attention_layers"] == [0]
        # field inversions
        assert local["num_routed_experts"] == 8
        assert local["dense_intermediate_size"] == 512
        assert local["intermediate_size"] == 128          # per-expert
        assert local["router_aux_loss_coef"] == 0.001
        assert local["lm_head_logit_softcapping"] == 30.0
        assert local["q_lora_rank"] == 0                  # None -> 0
        assert local["use_mtp"] is True
        assert local["mtp_predictor_num"] == 1
        assert local["mtp_block_type"] == "moe"           # nextn_is_sparse True
        assert "nextn_is_sparse" not in local

    def test_mtp_block_type_dense(self):
        hf = {"model_type": "gigachat3_5", "num_nextn_predict_layers": 1, "nextn_is_sparse": False}
        local = convert_hf_config_to_local(hf)
        assert local["mtp_block_type"] == "dense"

    @pytest.mark.parametrize("nextn_is_linear", [True, False])
    def test_nextn_is_linear_maps_to_use_linear_attention_in_mtp(self, nextn_is_linear):
        hf = {
            "model_type": "gigachat3_5",
            "num_hidden_layers": 4,
            "linear_attention_type": "Qwen3NextGatedDeltaNet",
            "full_attention_layers": [1, 3],
            "num_nextn_predict_layers": 1,
            "nextn_is_linear": nextn_is_linear,
        }
        local = convert_hf_config_to_local(hf)
        assert local["use_linear_attention_in_mtp"] is nextn_is_linear
        assert "nextn_is_linear" not in local

    def test_absent_nextn_is_linear_infers_from_anchor(self):
        hf = {
            "model_type": "gigachat3_5",
            "num_hidden_layers": 4,
            "linear_attention_type": "Qwen3NextGatedDeltaNet",
            "full_attention_layers": [1],
            "num_nextn_predict_layers": 1,
        }
        assert convert_hf_config_to_local(hf)["use_linear_attention_in_mtp"] is True

    def test_nextn_is_linear_dropped_without_mtp(self):
        hf = {"model_type": "gigachat3_5", "num_nextn_predict_layers": 0, "nextn_is_linear": True}
        local = convert_hf_config_to_local(hf)
        assert "nextn_is_linear" not in local
        assert "use_linear_attention_in_mtp" not in local

    @pytest.mark.parametrize("nextn_is_sparse,expected", [(None, "moe"), (True, "moe"), (False, "dense")])
    def test_mtp_block_type_matches_tensor_layout(self, nextn_is_sparse, expected):
        hf = {"model_type": "gigachat3_5", "num_nextn_predict_layers": 1}
        if nextn_is_sparse is not None:
            hf["nextn_is_sparse"] = nextn_is_sparse

        tensor_layout = resolve_mtp_block_type_from_hf_config(hf)
        assert tensor_layout == expected
        assert convert_hf_config_to_local(hf, mtp_block_type=tensor_layout)["mtp_block_type"] == expected
        # and the same when the caller leaves the resolve to the config converter
        assert convert_hf_config_to_local(hf)["mtp_block_type"] == expected

    def test_sonic_layout_markers(self):
        local = convert_hf_config_to_local({"model_type": "gigachat3_5"}, moe_layout="sonicmoe")
        assert local["moe_kernel"] == "sonic"
        assert local["moe_impl"] == "SonicGMMMoeBlock"

    def test_deepseek_strips_model_type(self):
        hf = {"model_type": "deepseek_v3", "n_routed_experts": 4}
        local = convert_hf_config_to_local(hf)
        assert "model_type" not in local
        assert local["num_routed_experts"] == 4


# ═══════════════════════════════════════════════════════════════════════════
# C. Full per-layer map_chunk round-trips (validates naming + routing)
# ═══════════════════════════════════════════════════════════════════════════

def _forward(cfg, local_sd):
    return _DeepseekPerLayerTransformer(cfg).rename_and_split(local_sd)


def _reverse(cfg, hf_sd, moe_layout="ggemm", mtp_block_type="moe"):
    return HFToLocalPerLayerMapper(cfg, moe_layout=moe_layout, mtp_block_type=mtp_block_type).map_chunk(hf_sd)


def _moe_experts_local(prefix, E, H, I):
    """Build fused local MoE expert weights (same 2D shapes for ggemm/sonic)."""
    # weight1: (H, 2*I*E); weight2: (I*E, H)
    return {
        f"{prefix}block_sparse_moe.experts.weight1": _t(H, 2 * I * E),
        f"{prefix}block_sparse_moe.experts.weight2": _t(I * E, H),
    }


class TestGlobalTensors:
    """embed / lm_head / final norm.

    `lm_head` is the odd one out: it lives outside `GigaMixModel`, so in a local
    monolith it keeps the Composer-level `model.` prefix that `_normalize_key_prefix`
    never strips (it only rewrites `model.model.` → `model.`), while HF has it bare.
    """

    def test_globals_roundtrip(self):
        cfg = _base_cfg()
        H = cfg["hidden_size"]
        local = {
            "model.embed_tokens.weight": _t(100, H),
            "model.lm_head.weight": _t(100, H),
            "model.norm.weight": _t(H),
        }
        hf = _forward(cfg, local)
        assert "lm_head.weight" in hf and "model.lm_head.weight" not in hf
        back = _reverse(cfg, hf)
        _assert_roundtrip(local, back)

    def test_mtp_clones_of_globals_dropped(self):
        """The forward pass replicates embed/lm_head into every MTP head; the reverse
        must drop those clones and keep only the two base tensors."""
        cfg = _base_cfg(num_hidden_layers=2, num_nextn_predict_layers=1)
        H = cfg["hidden_size"]
        local = {
            "model.embed_tokens.weight": _t(100, H),
            "model.lm_head.weight": _t(100, H),
        }
        hf = _forward(cfg, local)
        assert "model.layers.2.embed_tokens.weight" in hf
        assert "model.layers.2.shared_head.head.weight" in hf
        back = _reverse(cfg, hf)
        _assert_roundtrip(local, back)


class TestMlaMoeRoundTrip:
    """GigaChat 3.0 / DeepSeek MLA + MoE (regression guard for the existing path)."""

    def _mla_attn_local(self, prefix, cfg):
        H = cfg["hidden_size"]
        nh = cfg["num_attention_heads"]
        dkr, dkn, vd = cfg["qk_rope_head_dim"], cfg["qk_nope_head_dim"], cfg["v_head_dim"]
        kv_r, q_r = cfg["kv_lora_rank"], cfg["q_lora_rank"]
        a = prefix + "self_attn._attn."
        return {
            a + "dq_proj.weight": _t(q_r, H),
            a + "q_norm.weight": _t(q_r),
            a + "uq_proj.weight": _t(nh * (dkn + dkr), q_r),
            a + "dkv_proj.weight": _t(kv_r, H),
            a + "kr_proj.weight": _t(dkr, H),
            a + "kv_norm.weight": _t(kv_r),
            a + "uk_proj.weight": _t(nh * dkn, kv_r),
            a + "uv_proj.weight": _t(nh * vd, kv_r),
            a + "o_proj.weight": _t(H, nh * vd),
        }

    @pytest.mark.parametrize("layout", ["ggemm", "sonicmoe"])
    def test_mla_moe_layer_roundtrip(self, layout):
        moe_kernel = "sonic" if layout == "sonicmoe" else "g_gemm"
        cfg = _base_cfg(moe_kernel=moe_kernel)
        H, E, I = cfg["hidden_size"], cfg["n_routed_experts"], cfg["moe_intermediate_size"]
        prefix = "model.layers.1."   # >= first_k_dense_replace -> MoE

        local = {}
        local.update(self._mla_attn_local(prefix, cfg))
        local[prefix + "self_attn.input_layernorm.weight"] = _t(H)
        local[prefix + "self_attn.post_attention_layernorm.weight"] = _t(H)
        local[prefix + "self_attn._gate.gate.weight"] = _t(E, H)
        local[prefix + "self_attn._gate.expert_bias"] = _t(E)
        local.update(_moe_experts_local(prefix, E, H, I))

        hf = _forward(cfg, local)
        back = _reverse(cfg, hf, moe_layout=layout)
        _assert_roundtrip(local, back)


class TestGatedNorms:
    """gigachat3_5 layernorm_type='pre_post' with ZeroCenteredGatedNorm: every
    norm carries `gate_up_projection`/`gate_down_projection` submodules. Local
    and HF use the same names, so the reverse leaves them unchanged."""

    @staticmethod
    def _gated(base, dim, H, g=8):
        return {
            base + ".weight": _t(dim),
            base + ".gate_up_projection.weight": _t(g, H),
            base + ".gate_down_projection.weight": _t(H, g),
        }

    def test_gated_norms_roundtrip(self):
        cfg = _base_cfg(layernorm_type="pre_post", gated_attention=True)
        H = cfg["hidden_size"]
        E, I = cfg["n_routed_experts"], cfg["moe_intermediate_size"]
        q_r, kv_r = cfg["q_lora_rank"], cfg["kv_lora_rank"]
        prefix = "model.layers.1."   # MoE layer

        local = {}
        local.update(TestMlaMoeRoundTrip()._mla_attn_local(prefix, cfg))
        local[prefix + "self_attn._attn.gate_proj.weight"] = _t(
            cfg["num_attention_heads"] * cfg["v_head_dim"], H)
        # gated attention norms (overwrite the plain q_norm/kv_norm with gated ones)
        local.update(self._gated(prefix + "self_attn._attn.q_norm", q_r, H))
        local.update(self._gated(prefix + "self_attn._attn.kv_norm", kv_r, H))
        # gated decoder norms (MoE layer 1): post_self_attn on the AttnGate
        # (self_attn); post_feedforward at the decoder-layer top level.
        local.update(self._gated(prefix + "self_attn.input_layernorm", H, H))
        local.update(self._gated(prefix + "self_attn.post_attention_layernorm", H, H))
        local.update(self._gated(prefix + "self_attn.post_self_attn_layernorm", H, H))
        local.update(self._gated(prefix + "post_feedforward_layernorm", H, H))
        # gate + experts + final norm
        local[prefix + "self_attn._gate.gate.weight"] = _t(E, H)
        local[prefix + "self_attn._gate.expert_bias"] = _t(E)
        local.update(_moe_experts_local(prefix, E, H, I))
        local.update(self._gated("model.norm", H, H))
        local["model.embed_tokens.weight"] = _t(100, H)

        hf = _forward(cfg, local)
        back = _reverse(cfg, hf)
        _assert_roundtrip(local, back)


class TestGatedDeltaNetRoundTrip:
    def _gdn_local(self, prefix, legacy):
        n_h, n_v = _LIN["linear_num_key_heads"], _LIN["linear_num_value_heads"]
        d_k, d_v = _LIN["linear_key_head_dim"], _LIN["linear_value_head_dim"]
        H, K = 16, 4
        key_dim, value_dim = n_h * d_k, n_v * d_v
        a = prefix + "self_attn._attn."
        sd = {
            a + "ba_proj.weight": _t(2 * n_v, H),
            a + "o_proj.weight": _t(H, value_dim),
            a + "o_norm.weight": _t(d_v),
            a + "A_log": _t(n_v),
            a + "dt_bias": _t(n_v),
        }
        if legacy:
            sd[a + "qkvz_proj.weight"] = _t(2 * key_dim + 2 * value_dim, H)
            sd[a + "q_conv1d.weight"] = _t(key_dim, 1, K)
            sd[a + "k_conv1d.weight"] = _t(key_dim, 1, K)
            sd[a + "v_conv1d.weight"] = _t(value_dim, 1, K)
        else:
            sd[a + "qkv_proj.weight"] = _t(2 * key_dim + value_dim, H)
            sd[a + "z_proj.weight"] = _t(value_dim, H)
            sd[a + "qkv_conv1d.weight"] = _t(2 * key_dim + value_dim, 1, K)
        return sd

    @pytest.mark.parametrize("legacy", [False, True])
    def test_gdn_layer_roundtrip(self, legacy):
        # layer 0 is full-softmax (MLA), layer 1 is the linear layer under test.
        cfg = _base_cfg(
            linear_attention_type="Qwen3NextGatedDeltaNet",
            full_attention_layers=[0],
            linear_use_legacy_qkvz_layout=legacy,
            **_LIN,
        )
        prefix = "model.layers.1."
        local = self._gdn_local(prefix, legacy)
        # For the legacy layout the forward first packs the separate convs.
        if legacy:
            from base_mappers import DeepseekV3Mapper
            hf = DeepseekV3Mapper(cfg).map(local)
        else:
            hf = _forward(cfg, local)
        back = _reverse(cfg, hf)
        _assert_roundtrip(local, back)


class TestKimiDeltaRoundTrip:
    def test_kda_layer_roundtrip(self):
        n_h, n_v = _LIN["linear_num_key_heads"], _LIN["linear_num_value_heads"]
        d_k, d_v = _LIN["linear_key_head_dim"], _LIN["linear_value_head_dim"]
        H, K = 16, 4
        k_dim, v_dim = n_h * d_k, n_v * d_v
        gate_dim = n_v * d_k  # KDA dt_bias / f_proj.1 out-dim
        cfg = _base_cfg(
            linear_attention_type="KimiDeltaAttention",
            full_attention_layers=[0],
            **_LIN,
        )
        prefix = "model.layers.1."
        a = prefix + "self_attn._attn."
        local = {
            a + "qkv_proj.weight": _t(2 * k_dim + v_dim, H),
            a + "qkv_conv1d.weight": _t(2 * k_dim + v_dim, 1, K),
            a + "f_proj.0.weight": _t(d_v, H),
            a + "f_proj.1.weight": _t(gate_dim, d_v),
            a + "g_proj.0.weight": _t(d_v, H),
            a + "g_proj.1.weight": _t(v_dim, d_v),
            a + "b_proj.weight": _t(n_v, H),
            a + "o_norm.weight": _t(d_v),
            a + "o_proj.weight": _t(H, v_dim),
            a + "A_log": _t(1, 1, n_v, 1),
            a + "dt_bias": _t(gate_dim),
        }
        hf = _forward(cfg, local)
        back = _reverse(cfg, hf)
        _assert_roundtrip(local, back)


class TestMtpRoundTrip:
    """MTP head: embed/lm_head clones dropped, eh_proj half-swap, enorm/hnorm/norm."""

    def test_mtp_dense_attention_roundtrip(self):
        cfg = _base_cfg(num_hidden_layers=1, num_nextn_predict_layers=1)
        H = cfg["hidden_size"]
        nh = cfg["num_attention_heads"]
        dkr, dkn, vd = cfg["qk_rope_head_dim"], cfg["qk_nope_head_dim"], cfg["v_head_dim"]
        kv_r, q_r = cfg["kv_lora_rank"], cfg["q_lora_rank"]
        # MTP head 0 lives at mtp_block.mtp_layers.0.decoder_layer.* locally.
        m = "model.mtp_block.mtp_layers.0.decoder_layer."
        a = m + "self_attn.base_attn._attn."
        local = {
            m + "self_attn.mtp_input_proj.weight": _t(H, 2 * H),
            m + "self_attn.mtp_token_layernorm.weight": _t(H),
            m + "self_attn.mtp_hidden_layernorm.weight": _t(H),
            "model.mtp_block.mtp_norms.0.weight": _t(H),
            a + "dq_proj.weight": _t(q_r, H),
            a + "q_norm.weight": _t(q_r),
            a + "uq_proj.weight": _t(nh * (dkn + dkr), q_r),
            a + "dkv_proj.weight": _t(kv_r, H),
            a + "kr_proj.weight": _t(dkr, H),
            a + "kv_norm.weight": _t(kv_r),
            a + "uk_proj.weight": _t(nh * dkn, kv_r),
            a + "uv_proj.weight": _t(nh * vd, kv_r),
            a + "o_proj.weight": _t(H, nh * vd),
        }
        hf = _forward(cfg, local)
        # The forward clones embed_tokens / lm_head into the MTP head; the reverse
        # must drop those clones (they duplicate the base tensors).
        assert "model.layers.1.embed_tokens.weight" not in hf  # no embed in this chunk
        back = _reverse(cfg, hf, mtp_block_type="moe")
        _assert_roundtrip(local, back)

    @pytest.mark.parametrize("block_type", ["moe", "dense"])
    def test_mtp_norms_roundtrip(self, block_type):
        """layernorm_type='pre_post'. In the moe block the two post-norms have different
        owners: post_self_attn on the AttnGate (self_attn.base_attn), post_feedforward on
        the decoder layer itself — a sibling of block_sparse_moe, not a child. The dense
        block (LlamaDecoderLayer) owns all four at its top level."""
        cfg = _base_cfg(num_hidden_layers=1, num_nextn_predict_layers=1,
                        layernorm_type="pre_post")
        H = cfg["hidden_size"]
        m = "model.mtp_block.mtp_layers.0.decoder_layer."
        gate = "self_attn.base_attn." if block_type == "moe" else ""
        local = {
            m + gate + "input_layernorm.weight": _t(H),
            m + gate + "post_attention_layernorm.weight": _t(H),
            m + gate + "post_self_attn_layernorm.weight": _t(H),
            m + "post_feedforward_layernorm.weight": _t(H),
        }
        hf = _forward(cfg, local)
        back = _reverse(cfg, hf, mtp_block_type=block_type)
        _assert_roundtrip(local, back)

    def test_mtp_shared_expert_sigmoid_roundtrip(self):
        """shared_expert_sigmoid belongs to the AttnGate, like shared_experts. It must be
        caught before the dense-MLP branch, which would misplace it under self_attn.mlp.
        Only reachable for moe — the dense block has no AttnGate."""
        cfg = _base_cfg(num_hidden_layers=1, num_nextn_predict_layers=1)
        H = cfg["hidden_size"]
        m = "model.mtp_block.mtp_layers.0.decoder_layer."
        local = {m + "self_attn.base_attn.shared_expert_sigmoid.weight": _t(1, H)}
        hf = _forward(cfg, local)
        assert "model.layers.1.mlp.shared_expert_sigmoid.weight" in hf
        back = _reverse(cfg, hf, mtp_block_type="moe")
        _assert_roundtrip(local, back)
