import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

CONVERTATION_DIR = Path(__file__).resolve().parents[2]
if str(CONVERTATION_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERTATION_DIR))

from models.gigachat_slm.configuration_gigachat_slm import GigachatSLMConfig
from models.gigachat_slm import modeling_gigachat_slm as modeling
from models.gigachat_slm.modeling_gigachat_slm import (
    GigachatSLMDynamicCache,
    GigachatSLMForCausalLM,
    GigachatShortConv,
    _apply_xsa,
)

ALL_SCONV_SITES = {"q": True, "k": True, "v": True, "attn_out": True, "ffn_out": True}


@pytest.fixture(autouse=True)
def _fp32_default_dtype():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


TINY_MODEL = dict(
    vocab_size=97,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=256,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    attn_implementation="eager",
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)
SLIDING_LAYERS = dict(sliding_window=4, layer_types=["sliding_attention", "full_attention"])
FULL_LAYERS = dict(layer_types=["full_attention", "full_attention"])

# Embedded HF config of an llm-foundry `giga_mix_causal_lm` SLM checkpoint (training field names)
TRAINING_CONFIG = {
    "model_type": "giga_mix",
    "attention_type": "LlamaAttention",
    "hidden_size": 640,
    "intermediate_size": 2560,
    "dense_intermediate_size": 2560,
    "head_dim": 128,
    "num_attention_heads": 10,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "vocab_size": 128256,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "apply_qk_norm": True,
    "use_qk_gain": False,
    "norm_type": "LlamaRMSNorm",
    "layernorm_type": "pre",
    "linear_attention_type": None,
    "first_k_dense_replace": 2,
    "num_routed_experts": 64,
    "sliding_window": 128,
    "sliding_window_layer_types": ["sliding_attention", "full_attention"],
    "use_mtp": True,
    "mtp_block_type": "dense",
    "mtp_predictor_num": 1,
    "mtp_recurrent": False,
    "gated_attention": True,
    "use_xsa": True,
    "sconv": dict(ALL_SCONV_SITES),
    "sconv_kernel_size": 3,
    "tie_word_embeddings": False,
    "tied_modules_groups": [["model.embed_tokens", "lm_head"]],
    "lm_head_logit_softcapping": None,
    "use_cache": False,
    "_name_or_path": "some/tokenizer",
}


def _args():
    return argparse.Namespace(torch_dtype="bfloat16")


def _build_config(**overrides):
    from base_mappers import GigachatSLMMapper

    return GigachatSLMMapper().build_config({**TRAINING_CONFIG, **overrides}, _args())


def _mapper(**overrides):
    from base_mappers import GigachatSLMMapper

    mapper = GigachatSLMMapper()
    mapper.ensure_config({**TRAINING_CONFIG, **overrides}, _args())
    return mapper


def _tiny_config(**overrides):
    return GigachatSLMConfig(**{**TINY_MODEL, **SLIDING_LAYERS, **overrides})


def _full_features_config(**overrides):
    return GigachatSLMConfig(
        **{
            **TINY_MODEL,
            **SLIDING_LAYERS,
            "gated_attention": True,
            "use_xsa": True,
            "sconv": dict(ALL_SCONV_SITES),
            "sconv_kernel_size": 3,
            **overrides,
        }
    )


def _randomize(model, seed=0):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn(param.shape, generator=generator, dtype=param.dtype)
            if "norm" in name:
                param.copy_(1.0 + 0.2 * noise)
            elif "conv1d" in name:
                param.copy_(0.5 * noise)
            else:
                param.copy_(0.2 * noise)
    return model


def _padded_batch(seq_len=9, pad_lens=(3, 0), seed=0):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(3, TINY_MODEL["vocab_size"], (len(pad_lens), seq_len), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    for row, pad_len in enumerate(pad_lens):
        attention_mask[row, :pad_len] = 0
        input_ids[row, :pad_len] = TINY_MODEL["pad_token_id"]
    return input_ids, attention_mask


# --- pure-torch reference formulas (llm-foundry semantics) ---


def _ref_short_conv(x, weight, mask=None):
    x = x.float()
    if mask is not None:
        x = x * mask[:, :, None].float()
    kernel_size, seq_len, channels = weight.shape[-1], x.shape[1], x.shape[2]
    conv = F.conv1d(x.transpose(1, 2), weight.float(), padding=kernel_size - 1, groups=channels)
    return x + conv[..., :seq_len].transpose(1, 2)


def _ref_xsa(attn, value, eps):
    unit = value / value.norm(dim=-1, keepdim=True).clamp_min(eps)
    return attn - (attn * unit).sum(dim=-1, keepdim=True) * unit


def _ref_rms_norm(x, weight, eps):
    x = x.float()
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def _ref_rope(x, positions, theta):
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def _reference_logits(model, input_ids, attention_mask):
    cfg = model.config
    sd = {k: v.detach().float() for k, v in model.state_dict().items()}
    sconv = cfg.sconv or {}
    bsz, seq_len = input_ids.shape
    num_heads, num_kv_heads, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    eps = cfg.rms_norm_eps
    mask = attention_mask.float()
    positions = torch.arange(seq_len)
    causal = positions[None, :] <= positions[:, None]

    hidden = sd["model.embed_tokens.weight"][input_ids]
    for layer_idx in range(cfg.num_hidden_layers):
        p = f"model.layers.{layer_idx}."
        x = _ref_rms_norm(hidden, sd[p + "input_layernorm.weight"], eps)
        q = x @ sd[p + "self_attn.q_proj.weight"].T
        k = x @ sd[p + "self_attn.k_proj.weight"].T
        v = x @ sd[p + "self_attn.v_proj.weight"].T
        gate = x @ sd[p + "self_attn.gate_proj.weight"].T if cfg.gated_attention else None
        if sconv.get("q"):
            q = _ref_short_conv(q, sd[p + "self_attn.q_sconv.conv1d.weight"], mask)
        if sconv.get("k"):
            k = _ref_short_conv(k, sd[p + "self_attn.k_sconv.conv1d.weight"], mask)
        if sconv.get("v"):
            v = _ref_short_conv(v, sd[p + "self_attn.v_sconv.conv1d.weight"], mask)

        q = q.view(bsz, seq_len, num_heads, head_dim)
        k = k.view(bsz, seq_len, num_kv_heads, head_dim)
        v = v.view(bsz, seq_len, num_kv_heads, head_dim)
        q = _ref_rms_norm(q, sd[p + "self_attn.q_norm.weight"], eps)
        k = _ref_rms_norm(k, sd[p + "self_attn.k_norm.weight"], eps)
        q = _ref_rope(q, positions, cfg.rope_theta)
        k = _ref_rope(k, positions, cfg.rope_theta)
        k = k.repeat_interleave(num_heads // num_kv_heads, dim=2)
        v = v.repeat_interleave(num_heads // num_kv_heads, dim=2)

        scores = torch.einsum("bthd,bshd->bhts", q, k) * head_dim**-0.5
        allowed = causal[None, None] & mask.bool()[:, None, None, :]
        if cfg.layer_types[layer_idx] == "sliding_attention":
            allowed = allowed & ((positions[:, None] - positions[None, :]) < cfg.sliding_window)[None, None]
        attn = torch.softmax(scores.masked_fill(~allowed, torch.finfo(torch.float32).min), dim=-1)
        # flash-attn varlen pads its output with zeros at padded query positions
        out = torch.einsum("bhts,bshd->bthd", attn, v) * mask[:, :, None, None]
        if cfg.use_xsa:
            out = _ref_xsa(out, v, eps)
        out = out.reshape(bsz, seq_len, num_heads * head_dim)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        out = out @ sd[p + "self_attn.o_proj.weight"].T
        if sconv.get("attn_out"):
            out = _ref_short_conv(out, sd[p + "self_attn.attn_out_sconv.conv1d.weight"], mask)
        hidden = hidden + out

        x = _ref_rms_norm(hidden, sd[p + "post_attention_layernorm.weight"], eps)
        mlp = (F.silu(x @ sd[p + "mlp.gate_proj.weight"].T) * (x @ sd[p + "mlp.up_proj.weight"].T)) @ sd[
            p + "mlp.down_proj.weight"
        ].T
        if sconv.get("ffn_out"):
            mlp = _ref_short_conv(mlp, sd[p + "ffn_out_sconv.conv1d.weight"], mask)
        hidden = hidden + mlp

    hidden = _ref_rms_norm(hidden, sd["model.norm.weight"], eps)
    return hidden @ sd["lm_head.weight"].T


def _training_tensors(cfg, tied=True):
    """Tiny checkpoint with the real llm-foundry parameter names (trunk + dense MTP block)."""
    hidden, inter, kernel = cfg["hidden_size"], cfg["intermediate_size"], cfg["sconv_kernel_size"]
    q_size = cfg["num_attention_heads"] * cfg["head_dim"]
    kv_size = cfg["num_key_value_heads"] * cfg["head_dim"]
    torch.manual_seed(0)

    def attention(prefix):
        return {
            prefix + "q_proj.weight": torch.randn(q_size, hidden),
            prefix + "k_proj.weight": torch.randn(kv_size, hidden),
            prefix + "v_proj.weight": torch.randn(kv_size, hidden),
            prefix + "o_proj.weight": torch.randn(hidden, q_size),
            prefix + "gate_proj.weight": torch.randn(q_size, hidden),
            prefix + "q_sconv.conv1d.weight": torch.randn(q_size, 1, kernel),
            prefix + "k_sconv.conv1d.weight": torch.randn(kv_size, 1, kernel),
            prefix + "v_sconv.conv1d.weight": torch.randn(kv_size, 1, kernel),
            prefix + "attn_out_sconv.conv1d.weight": torch.randn(hidden, 1, kernel),
            prefix + "q_norm.weight": torch.ones(cfg["head_dim"]),
            prefix + "k_norm.weight": torch.ones(cfg["head_dim"]),
        }

    def mlp(prefix):
        return {
            prefix + "gate_proj.weight": torch.randn(inter, hidden),
            prefix + "up_proj.weight": torch.randn(inter, hidden),
            prefix + "down_proj.weight": torch.randn(hidden, inter),
        }

    embedding = torch.randn(cfg["vocab_size"], hidden)
    tensors = {
        "model.embed_tokens.weight": embedding,
        "model.norm.weight": torch.ones(hidden),
        "model.lm_head.weight": embedding.clone() if tied else torch.randn(cfg["vocab_size"], hidden),
    }
    for layer in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        tensors.update(attention(p + "self_attn._attn."))
        tensors.update(mlp(p + "self_attn.mlp."))
        tensors[p + "self_attn.input_layernorm.weight"] = torch.ones(hidden)
        tensors[p + "self_attn.post_attention_layernorm.weight"] = torch.ones(hidden)
        tensors[p + "ffn_out_sconv.conv1d.weight"] = torch.randn(hidden, 1, kernel)
    mtp = "model.mtp_block.mtp_layers.0.decoder_layer."
    tensors.update(attention(mtp + "self_attn.base_attn."))
    tensors.update(mlp(mtp + "mlp."))
    tensors[mtp + "self_attn.mtp_input_proj.weight"] = torch.randn(hidden, 2 * hidden)
    tensors[mtp + "self_attn.mtp_token_layernorm.weight"] = torch.ones(hidden)
    tensors[mtp + "self_attn.mtp_hidden_layernorm.weight"] = torch.ones(hidden)
    tensors[mtp + "input_layernorm.weight"] = torch.ones(hidden)
    tensors[mtp + "post_attention_layernorm.weight"] = torch.ones(hidden)
    tensors["model.mtp_block.mtp_norms.0.weight"] = torch.ones(hidden)
    return tensors


class TestGigachatSLMConfig:
    def test_sconv_sites_are_normalized(self):
        assert _tiny_config(sconv={"q": True, "k": False, "ffn_out": True}, sconv_kernel_size=3).sconv == {
            "q": True,
            "ffn_out": True,
        }
        assert _tiny_config(sconv={"q": False, "v": False}).sconv is None
        assert _tiny_config(sconv={}).sconv is None

    def test_layer_types_default_to_full_attention_and_drop_sliding_window(self):
        cfg = GigachatSLMConfig(**TINY_MODEL, sliding_window=4)
        assert cfg.layer_types == ["full_attention", "full_attention"]
        assert cfg.sliding_window is None

    def test_config_roundtrip_keeps_feature_fields(self, tmp_path):
        cfg = _tiny_config(
            gated_attention=True, use_xsa=True, sconv={"q": True, "ffn_out": True}, sconv_kernel_size=2,
            num_nextn_predict_layers=1,
        )
        cfg.save_pretrained(tmp_path)
        loaded = GigachatSLMConfig.from_pretrained(tmp_path)

        assert loaded.gated_attention is True and loaded.use_xsa is True
        assert loaded.sconv == {"q": True, "ffn_out": True} and loaded.sconv_kernel_size == 2
        assert loaded.layer_types == ["sliding_attention", "full_attention"] and loaded.sliding_window == 4
        assert loaded.num_nextn_predict_layers == 1


class TestGigachatSLMMapperConfig:
    def test_exports_attention_features_and_architecture(self):
        cfg = _build_config()

        assert isinstance(cfg, GigachatSLMConfig)
        assert cfg.model_type == "gigachat_slm" and cfg.architectures == ["GigachatSLMForCausalLM"]
        assert cfg.gated_attention is True and cfg.use_xsa is True
        assert cfg.sconv == ALL_SCONV_SITES and cfg.sconv_kernel_size == 3
        assert cfg.head_dim == 128 and cfg.num_key_value_heads == 2 and cfg.intermediate_size == 2560
        assert cfg.layer_types == ["sliding_attention", "full_attention"] and cfg.sliding_window == 128
        assert cfg.num_nextn_predict_layers == 1
        assert cfg.tie_word_embeddings is True
        assert cfg.use_cache is True
        assert cfg.dtype == torch.bfloat16
        assert cfg._auto_map_for_save == {
            "AutoConfig": "configuration_gigachat_slm.GigachatSLMConfig",
            "AutoModel": "modeling_gigachat_slm.GigachatSLMModel",
            "AutoModelForCausalLM": "modeling_gigachat_slm.GigachatSLMForCausalLM",
        }

    def test_rejects_missing_kernel_size_when_sconv_is_enabled(self):
        source = {k: v for k, v in TRAINING_CONFIG.items() if k != "sconv_kernel_size"}
        from base_mappers import GigachatSLMMapper

        with pytest.raises(ValueError, match="sconv_kernel_size"):
            GigachatSLMMapper().build_config(source, _args())

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"attention_type": "LlamaPackedAttention"}, "LlamaAttention"),
            ({"use_qk_gain": True}, "use_qk_gain"),
            ({"attention_bias": True}, "attention_bias"),
            ({"layernorm_type": "post"}, "layernorm_type"),
            ({"norm_type": "GatedNorm"}, "norm_type"),
            ({"linear_attention_type": "Qwen3NextGatedDeltaNet", "full_attention_layers": [1]}, "linear_attention_type"),
            ({"first_k_dense_replace": 1}, "MoE layers"),
            ({"mtp_block_type": "moe"}, "mtp_block_type"),
            ({"head_dim": None, "attention_hidden_size": 1280}, "partial rotary"),
            ({"final_logit_softcapping": 30.0}, "final_logit_softcapping"),
            ({"sconv": {"q": True, "mlp": True}}, "Unknown sconv sites"),
            ({"sconv_kernel_size": 5}, "sconv_kernel_size"),
            ({"layer_types": ["full_attention", "full_attention"]}, "differ"),
            ({"sliding_window": None}, "sliding_window"),
            ({"sliding_window_layer_types": ["sliding_attention"] * 3}, "num_hidden_layers"),
        ],
    )
    def test_rejects_unsupported_training_configs(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            _build_config(**overrides)

    def test_sliding_window_layer_types_become_layer_types(self):
        saved = _build_config().to_dict()
        assert saved["layer_types"] == ["sliding_attention", "full_attention"]
        assert saved["sliding_window"] == 128
        assert "sliding_window_layer_types" not in saved

    def test_tied_modules_groups_tie_the_embeddings(self):
        assert _build_config().tie_word_embeddings is True
        assert _build_config(tied_modules_groups=[["model.embed_tokens", "lm_head"], ["a", "b"]]).tie_word_embeddings is True

    def test_plain_dense_checkpoint_without_attention_features(self):
        # Plain dense GQA checkpoint: no features at all (keys absent), stray `mtp_block_type: moe`
        # with `use_mtp: False`, EMA handled upstream, pad_token_id absent
        from base_mappers import GigachatSLMMapper

        source = {
            "attention_type": "LlamaAttention", "hidden_size": 1024,
            "intermediate_size": 3072, "dense_intermediate_size": 3072, "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "attention_hidden_size": None,
            "apply_qk_norm": True, "attention_bias": False, "norm_type": "LlamaRMSNorm", "layernorm_type": "pre",
            "linear_attention_type": None, "first_k_dense_replace": 28, "num_routed_experts": 64,
            "sliding_window": None, "sliding_window_layer_types": None, "use_mtp": False, "mtp_block_type": "moe",
            "mtp_predictor_num": 1, "gated_attention": False, "tie_word_embeddings": False,
            "tied_modules_groups": [["model.embed_tokens", "lm_head"]], "lm_head_logit_softcapping": None,
            "rope_theta": 10000.0, "rope_scaling": None, "vocab_size": 128256, "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6, "use_cache": False, "bos_token_id": 1, "eos_token_id": 2,
        }
        cfg = GigachatSLMMapper().build_config(source, _args())

        assert cfg.model_type == "gigachat_slm"
        assert (cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) == (28, 16, 8, 128)
        assert cfg.layer_types == ["full_attention"] * 28 and cfg.sliding_window is None
        assert cfg.num_nextn_predict_layers == 0
        assert cfg.tie_word_embeddings is True
        assert cfg.sconv is None and cfg.sconv_kernel_size is None
        assert cfg.gated_attention is False and cfg.use_xsa is False
        assert cfg.pad_token_id == 2 and cfg.use_cache is True

    def test_exported_fields_match_template(self):
        from utils.hf_model_specs import GIGACHAT_SLM_SPEC

        template = json.loads((GIGACHAT_SLM_SPEC.template_dir / "config.json").read_text())
        saved = _build_config().to_dict()
        assert set(template) - {"auto_map"} <= set(saved)

    def test_build_config_is_a_fixed_point_of_its_own_output(self):
        mapper = _mapper()
        reloaded = mapper.build_config(mapper.cfg.to_dict(), _args())
        assert reloaded.to_dict() == mapper.cfg.to_dict()

    @patch("base_mappers._save_config_and_tokenizer", return_value=(None, None, None))
    def test_save_meta_vendors_slm_runtime_files(self, mock_save, tmp_path):
        mapper = _mapper()

        mapper.save_meta(dict(TRAINING_CONFIG), _args(), str(tmp_path))

        mock_save.assert_called_once()
        assert (tmp_path / "configuration_gigachat_slm.py").is_file()
        assert (tmp_path / "modeling_gigachat_slm.py").is_file()
        assert not (tmp_path / "modeling_gigachat3_5_dense.py").exists()


class TestGigachatSLMMapperKeys:
    # hidden 640, heads 10, kv heads 2, head_dim 128: q 1280 rows, k/v 256 rows
    def test_maps_trunk_layer_keys(self):
        mapper = _mapper(tied_modules_groups=None)
        prefix = "model.layers.1.self_attn."
        tensors = {
            prefix + "_attn.q_proj.weight": torch.randn(1280, 640),
            prefix + "_attn.k_proj.weight": torch.randn(256, 640),
            prefix + "_attn.v_proj.weight": torch.randn(256, 640),
            prefix + "_attn.o_proj.weight": torch.randn(640, 1280),
            prefix + "_attn.gate_proj.weight": torch.randn(1280, 640),
            prefix + "_attn.q_norm.weight": torch.ones(128),
            prefix + "_attn.k_norm.weight": torch.ones(128),
            prefix + "_attn.q_sconv.conv1d.weight": torch.randn(1280, 1, 3),
            prefix + "_attn.k_sconv.conv1d.weight": torch.randn(256, 1, 3),
            prefix + "_attn.v_sconv.conv1d.weight": torch.randn(256, 1, 3),
            prefix + "_attn.attn_out_sconv.conv1d.weight": torch.randn(640, 1, 3),
            prefix + "input_layernorm.weight": torch.ones(640),
            prefix + "post_attention_layernorm.weight": torch.ones(640),
            prefix + "mlp.gate_proj.weight": torch.randn(2560, 640),
            prefix + "mlp.up_proj.weight": torch.randn(2560, 640),
            prefix + "mlp.down_proj.weight": torch.randn(640, 2560),
            "model.layers.1.ffn_out_sconv.conv1d.weight": torch.randn(640, 1, 3),
            "model.embed_tokens.weight": torch.randn(16, 640),
            "model.lm_head.weight": torch.randn(16, 640),
            "model.norm.weight": torch.ones(640),
        }

        result = mapper.map(tensors)

        hf = "model.layers.1."
        assert set(result) == {
            hf + "self_attn.q_proj.weight", hf + "self_attn.k_proj.weight", hf + "self_attn.v_proj.weight",
            hf + "self_attn.o_proj.weight", hf + "self_attn.gate_proj.weight",
            hf + "self_attn.q_norm.weight", hf + "self_attn.k_norm.weight",
            hf + "self_attn.q_sconv.conv1d.weight", hf + "self_attn.k_sconv.conv1d.weight",
            hf + "self_attn.v_sconv.conv1d.weight", hf + "self_attn.attn_out_sconv.conv1d.weight",
            hf + "input_layernorm.weight", hf + "post_attention_layernorm.weight",
            hf + "mlp.gate_proj.weight", hf + "mlp.up_proj.weight", hf + "mlp.down_proj.weight",
            hf + "ffn_out_sconv.conv1d.weight",
            "model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
        }
        assert result[hf + "self_attn.q_sconv.conv1d.weight"].shape == (1280, 1, 3)
        assert result[hf + "self_attn.k_sconv.conv1d.weight"].shape == (256, 1, 3)
        assert all(tensor.dtype == torch.bfloat16 for tensor in result.values())
        torch.testing.assert_close(result["lm_head.weight"], tensors["model.lm_head.weight"].to(torch.bfloat16))
        assert mapper.flush() == {}

    @pytest.mark.parametrize("param", ["weight", "bias"])
    def test_splits_fused_qkv(self, param):
        mapper = _mapper(tied_modules_groups=None)
        fused = torch.randn(1280 + 2 * 256, 640) if param == "weight" else torch.randn(1280 + 2 * 256)

        result = mapper.map({f"model.layers.0.self_attn._attn.qkv_proj.{param}": fused})

        assert set(result) == {f"model.layers.0.self_attn.{proj}_proj.{param}" for proj in "qkv"}
        torch.testing.assert_close(result[f"model.layers.0.self_attn.q_proj.{param}"], fused[:1280].to(torch.bfloat16))
        torch.testing.assert_close(result[f"model.layers.0.self_attn.k_proj.{param}"], fused[1280:1536].to(torch.bfloat16))
        torch.testing.assert_close(result[f"model.layers.0.self_attn.v_proj.{param}"], fused[1536:].to(torch.bfloat16))

    def test_maps_dense_mtp_block_to_deepseek_layout(self):
        mapper = _mapper()
        prefix = "model.mtp_block.mtp_layers.0.decoder_layer."
        input_proj = torch.randn(640, 1280)
        tensors = {
            prefix + "self_attn.base_attn.q_proj.weight": torch.randn(1280, 640),
            prefix + "self_attn.base_attn.k_norm.weight": torch.ones(128),
            prefix + "self_attn.base_attn.gate_proj.weight": torch.randn(1280, 640),
            prefix + "self_attn.base_attn.q_sconv.conv1d.weight": torch.randn(1280, 1, 3),
            prefix + "self_attn.base_attn.k_sconv.conv1d.weight": torch.randn(256, 1, 3),
            prefix + "self_attn.base_attn.v_sconv.conv1d.weight": torch.randn(256, 1, 3),
            prefix + "self_attn.base_attn.attn_out_sconv.conv1d.weight": torch.randn(640, 1, 3),
            prefix + "self_attn.mtp_input_proj.weight": input_proj,
            prefix + "self_attn.mtp_token_layernorm.weight": torch.ones(640),
            prefix + "self_attn.mtp_hidden_layernorm.weight": torch.ones(640),
            prefix + "input_layernorm.weight": torch.ones(640),
            prefix + "post_attention_layernorm.weight": torch.ones(640),
            prefix + "mlp.down_proj.weight": torch.randn(640, 2560),
            "model.mtp_block.mtp_norms.0.weight": torch.ones(640),
        }

        result = mapper.map(tensors)

        # num_hidden_layers == 2, so the single MTP block becomes layer 2
        mtp = "model.layers.2."
        assert set(result) == {
            mtp + "self_attn.q_proj.weight", mtp + "self_attn.k_norm.weight", mtp + "self_attn.gate_proj.weight",
            mtp + "self_attn.q_sconv.conv1d.weight", mtp + "self_attn.k_sconv.conv1d.weight",
            mtp + "self_attn.v_sconv.conv1d.weight", mtp + "self_attn.attn_out_sconv.conv1d.weight",
            mtp + "eh_proj.weight", mtp + "enorm.weight", mtp + "hnorm.weight",
            mtp + "input_layernorm.weight", mtp + "post_attention_layernorm.weight", mtp + "mlp.down_proj.weight",
            mtp + "shared_head.norm.weight",
        }
        # training input is [prev_hidden, embedding]; DeepSeek eh_proj expects [embedding, prev_hidden]
        expected = torch.cat([input_proj[:, 640:], input_proj[:, :640]], dim=1).to(torch.bfloat16)
        torch.testing.assert_close(result[mtp + "eh_proj.weight"], expected)
        assert result[mtp + "self_attn.attn_out_sconv.conv1d.weight"].shape == (640, 1, 3)

    @pytest.mark.parametrize(
        "key",
        [
            "model.layers.0.self_attn._attn.qk_gain.gain",
            "model.mtp_block.mtp_layers.0.decoder_layer.self_attn.base_attn.qk_gain.gain",
        ],
    )
    def test_unknown_attn_wrapper_key_is_an_error(self, key):
        with pytest.raises(ValueError, match="Unmapped attention wrapper key"):
            _mapper().map({key: torch.zeros(10)})

    def test_tied_embeddings_drop_lm_head(self):
        mapper = _mapper()
        embedding = torch.randn(16, 640)

        result = mapper.map({"model.embed_tokens.weight": embedding, "model.lm_head.weight": embedding.clone()})

        assert set(result) == {"model.embed_tokens.weight"}

    def test_tied_embeddings_require_equal_tensors_in_the_same_chunk(self):
        mapper = _mapper()
        with pytest.raises(ValueError, match="differ"):
            mapper.map({"model.embed_tokens.weight": torch.randn(16, 640), "model.lm_head.weight": torch.randn(16, 640)})
        with pytest.raises(ValueError, match="same chunk"):
            mapper.map({"lm_head.weight": torch.randn(16, 640)})


class TestGigachatSLMComponents:
    @pytest.mark.parametrize("kernel_size", [2, 3, 4])
    def test_short_conv_chunked_with_state_equals_full(self, kernel_size):
        torch.manual_seed(10 + kernel_size)
        conv = GigachatShortConv(hidden_size=5, kernel_size=kernel_size)
        x = torch.randn(3, 11, 5)
        mask = torch.ones(3, 11, dtype=torch.long)
        mask[1, :4] = 0

        full, _ = conv(x, mask=mask)
        state = None
        chunks = []
        for start, end in ((0, 4), (4, 6), (6, 7), (7, 11)):
            y, state = conv(x[:, start:end], mask=mask[:, start:end], conv_state=state)
            chunks.append(y)

        torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-6, rtol=1e-6)

    def test_xsa_grouped_equals_repeat_kv_reference(self):
        torch.manual_seed(1)
        bsz, seq_len, num_heads, num_kv_heads, head_dim = 2, 5, 4, 2, 8
        groups = num_heads // num_kv_heads
        attn = torch.randn(bsz, seq_len, num_heads, head_dim)
        value_kv = torch.randn(bsz, seq_len, num_kv_heads, head_dim)
        value_repeated = value_kv.repeat_interleave(groups, dim=2)

        out = _apply_xsa(attn, value_kv, groups, eps=1e-6)

        torch.testing.assert_close(out, _ref_xsa(attn, value_repeated, 1e-6), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close((out * value_repeated).sum(-1), torch.zeros(bsz, seq_len, num_heads), atol=1e-5, rtol=0)
        # head h belongs to kv head h // groups, exactly as `repeat_kv` lays the heads out
        torch.testing.assert_close(value_repeated[..., 1, :], value_kv[..., 0, :])
        torch.testing.assert_close(value_repeated[..., 2, :], value_kv[..., 1, :])


class TestGigachatSLMReferenceSemantics:
    @pytest.mark.parametrize("layers", [SLIDING_LAYERS, FULL_LAYERS], ids=["sliding", "full"])
    def test_all_features_match_reference_with_padding(self, layers):
        cfg = _full_features_config(**layers)
        model = _randomize(GigachatSLMForCausalLM(cfg)).eval()
        input_ids, attention_mask = _padded_batch(seq_len=9, pad_lens=(3, 0))

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
        expected = _reference_logits(model, input_ids, attention_mask)

        torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)

    def test_left_padding_matches_unpadded_sequence(self):
        model = _randomize(GigachatSLMForCausalLM(_full_features_config()), seed=5).eval()
        pad_len = 4
        input_ids, attention_mask = _padded_batch(seq_len=10, pad_lens=(pad_len, 0), seed=5)

        with torch.no_grad():
            padded = model(input_ids, attention_mask=attention_mask).logits[0, pad_len:]
            unpadded = model(input_ids[:1, pad_len:]).logits[0]

        torch.testing.assert_close(padded, unpadded, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "features",
        [
            dict(gated_attention=True),
            dict(use_xsa=True),
            dict(sconv={"q": True, "v": True}, sconv_kernel_size=3),
            dict(sconv={"ffn_out": True}, sconv_kernel_size=3),
            dict(sconv={"attn_out": True}, sconv_kernel_size=3, use_xsa=True),
        ],
    )
    def test_single_features_match_reference(self, features):
        cfg = _tiny_config(**features)
        model = _randomize(GigachatSLMForCausalLM(cfg), seed=3).eval()
        input_ids, attention_mask = _padded_batch(seq_len=8, pad_lens=(2, 0), seed=3)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
        expected = _reference_logits(model, input_ids, attention_mask)

        if not cfg.sconv:
            # without short convs the padded query rows are not zeroed, as in the stock model
            valid = attention_mask.bool()
            torch.testing.assert_close(logits[valid], expected[valid], atol=1e-4, rtol=1e-4)
        else:
            torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)


class TestGigachatSLMBackwardCompat:
    def test_features_off_equals_stock_qwen3(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        cfg = _tiny_config()
        assert cfg.gated_attention is False and cfg.use_xsa is False and cfg.sconv is None
        model = _randomize(GigachatSLMForCausalLM(cfg)).eval()
        stock = Qwen3ForCausalLM(Qwen3Config(**TINY_MODEL, use_sliding_window=True, **SLIDING_LAYERS)).eval()
        stock.load_state_dict(model.state_dict())
        assert set(model.state_dict()) == set(stock.state_dict())
        input_ids, attention_mask = _padded_batch(seq_len=9, pad_lens=(3, 0))

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
            expected = stock(input_ids, attention_mask=attention_mask).logits
            cached = model(input_ids, attention_mask=attention_mask, use_cache=True)

        torch.testing.assert_close(logits, expected, atol=0, rtol=0)
        assert type(cached.past_key_values).__name__ == "DynamicCache"


class TestGigachatSLMConvCache:
    def _model(self):
        return _randomize(GigachatSLMForCausalLM(_full_features_config()), seed=7).eval()

    def test_stepwise_decode_matches_full_recompute(self):
        model = self._model()
        input_ids, attention_mask = _padded_batch(seq_len=12, pad_lens=(4, 0), seed=7)

        with torch.no_grad():
            full = model(input_ids, attention_mask=attention_mask).logits
            chunks = [(0, 5), (5, 7)] + [(t, t + 1) for t in range(7, 12)]
            cache = None
            logits = []
            for start, end in chunks:
                out = model(
                    input_ids[:, start:end],
                    attention_mask=attention_mask[:, :end],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                logits.append(out.logits)
            stepwise = torch.cat(logits, dim=1)

        assert isinstance(cache, GigachatSLMDynamicCache)
        assert cache.get_seq_length() == 12
        # the sliding layer really uses the windowed cache layer
        assert [layer.is_sliding for layer in cache.layers] == [True, False]
        expected_sites = {(layer, site) for layer in range(2) for site in ALL_SCONV_SITES}
        assert set(cache.conv_states) == expected_sites
        assert cache.conv_states[(0, "q")].shape == (2, 2, model.config.num_attention_heads * model.config.head_dim)
        valid = attention_mask.bool()
        torch.testing.assert_close(stepwise[valid], full[valid], atol=1e-5, rtol=1e-5)

    def test_beam_search_reorders_conv_states(self):
        model = self._model()
        input_ids, attention_mask = _padded_batch(seq_len=6, pad_lens=(1, 0), seed=9)
        kwargs = dict(max_new_tokens=4, num_beams=2, do_sample=False, pad_token_id=0, early_stopping=True)

        with_cache = model.generate(input_ids, attention_mask=attention_mask, use_cache=True, **kwargs)
        without_cache = model.generate(input_ids, attention_mask=attention_mask, use_cache=False, **kwargs)

        assert torch.equal(with_cache, without_cache)

    def test_padding_mask_reaches_the_convs_but_not_the_attention_kernel(self, monkeypatch):
        model = self._model()
        input_ids, attention_mask = _padded_batch(seq_len=6, pad_lens=(2, 0))
        seen_masks, attention_kwargs = [], []

        original_sconv, original_attention = modeling._apply_sconv, modeling.eager_attention_forward

        def recording_sconv(sconv, hidden_states, mask, *args, **kwargs):
            seen_masks.append(mask)
            return original_sconv(sconv, hidden_states, mask, *args, **kwargs)

        def recording_attention(*args, **kwargs):
            attention_kwargs.append(set(kwargs))
            return original_attention(*args, **kwargs)

        monkeypatch.setattr(modeling, "_apply_sconv", recording_sconv)
        monkeypatch.setattr(modeling, "eager_attention_forward", recording_attention)

        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)

        assert len(seen_masks) == 2 * len(ALL_SCONV_SITES)
        assert all(torch.equal(mask, attention_mask) for mask in seen_masks)
        assert len(attention_kwargs) == 2
        assert not any("padding_mask" in kwargs for kwargs in attention_kwargs)


class TestGigachatSLMGenerationCache:
    def _model(self, **overrides):
        return _randomize(GigachatSLMForCausalLM(_full_features_config(**overrides)), seed=11).eval()

    def test_generate_builds_a_conv_cache_with_sliding_layers(self):
        model = self._model()
        input_ids, attention_mask = _padded_batch(seq_len=5, pad_lens=(1, 0))

        out = model.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=2, do_sample=False, pad_token_id=0,
            return_dict_in_generate=True,
        )

        cache = out.past_key_values
        assert type(cache) is GigachatSLMDynamicCache
        assert [layer.is_sliding for layer in cache.layers] == [True, False]
        assert cache.offloading is False
        assert set(cache.conv_states) == {(layer, site) for layer in range(2) for site in ALL_SCONV_SITES}

    @pytest.mark.parametrize("mode", ["ASSISTED_GENERATION", "CONTRASTIVE_SEARCH"])
    def test_cache_rollback_modes_are_rejected_up_front(self, mode):
        from transformers import GenerationConfig
        from transformers.generation import GenerationMode

        model = self._model()
        with pytest.raises(ValueError, match="rollback"):
            model._prepare_cache_for_generation(GenerationConfig(), {}, GenerationMode[mode], 1, 16)


TINY_TRAINING_CONFIG = {
    **TRAINING_CONFIG,
    "hidden_size": 32,
    "intermediate_size": 64,
    "dense_intermediate_size": 64,
    "head_dim": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 97,
    "sliding_window": 4,
}


def _export_checkpoint(tmp_path, source, tied=True):
    from safetensors.torch import save_file

    from base_mappers import GigachatSLMMapper
    from utils.file_utils import _atomic_write_json, vendor_runtime_code
    from utils.hf_model_specs import GIGACHAT_SLM_SPEC

    mapper = GigachatSLMMapper()
    mapper.ensure_config(source, _args())
    cfg_dict = mapper.cfg.to_dict()
    cfg_dict["auto_map"] = cfg_dict.pop("_auto_map_for_save")
    _atomic_write_json(str(tmp_path / "config.json"), cfg_dict)
    vendor_runtime_code(str(tmp_path), GIGACHAT_SLM_SPEC)

    mapped = mapper.map(_training_tensors(source, tied=tied))
    save_file({k: v.contiguous() for k, v in mapped.items()}, str(tmp_path / "model.safetensors"))
    return mapper, mapped


class TestGigachatSLMRemoteCodeRoundtrip:
    def test_tied_checkpoint_loads_with_auto_model(self, tmp_path):
        from transformers import AutoModelForCausalLM

        _, mapped = _export_checkpoint(tmp_path, TINY_TRAINING_CONFIG, tied=True)
        assert (tmp_path / "modeling_gigachat_slm.py").is_file()
        assert (tmp_path / "configuration_gigachat_slm.py").is_file()
        assert "lm_head.weight" not in mapped
        assert any(key.startswith("model.layers.2.") for key in mapped)

        model, loading_info = AutoModelForCausalLM.from_pretrained(
            tmp_path, trust_remote_code=True, output_loading_info=True, dtype=torch.float32
        )

        assert loading_info["missing_keys"] == []
        assert loading_info["unexpected_keys"] == []
        assert type(model).__name__ == "GigachatSLMForCausalLM"
        assert model.config.tie_word_embeddings is True
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        assert model.config.gated_attention is True and model.config.use_xsa is True
        assert model.config.sconv == ALL_SCONV_SITES and model.config.sconv_kernel_size == 3
        assert model.config.layer_types == ["sliding_attention", "full_attention"] and model.config.sliding_window == 4
        assert model.config.num_nextn_predict_layers == 1
        assert len(model.model.layers) == 2

        layer = model.model.layers[0]
        assert layer.self_attn.gate_proj is not None
        assert layer.self_attn.q_sconv.conv1d.weight.shape == (32, 1, 3)
        assert layer.self_attn.k_sconv.conv1d.weight.shape == (16, 1, 3)
        assert layer.ffn_out_sconv.conv1d.weight.shape == (32, 1, 3)
        torch.testing.assert_close(
            layer.self_attn.gate_proj.weight.float(), mapped["model.layers.0.self_attn.gate_proj.weight"].float()
        )

        input_ids, attention_mask = _padded_batch(seq_len=6, pad_lens=(2, 0))
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
        assert logits.shape == (2, 6, 97)
        assert torch.isfinite(logits).all()
        generated = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=3, do_sample=False, pad_token_id=0)
        assert generated.shape == (2, 9)

