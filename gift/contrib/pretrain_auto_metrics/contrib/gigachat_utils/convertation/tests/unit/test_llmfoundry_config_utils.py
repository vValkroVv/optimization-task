"""
Unit tests for convertation/utils/llmfoundry_config_utils.py
"""

import os
import sys
import json
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root or from convertation/
# ---------------------------------------------------------------------------
_CONVERTATION_DIR = Path(__file__).resolve().parents[2]
_UTILS_DIR = _CONVERTATION_DIR / "utils"
for _d in (_CONVERTATION_DIR, _UTILS_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from llmfoundry_config_utils import (
    _find_llmfoundry_yaml,
    extract_hybrid_fields_from_llmfoundry,
    enrich_hf_config_from_llmfoundry_yaml,
    is_gigachat_model,
    is_hybrid_model,
    _get_model_params,
    _GIGACHAT35_BOOL_MARKERS,
    _DEEPSEEK_LAYERNORM_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yaml_cfg(params: dict, model_key: str = "params"):
    """
    Build a minimal OmegaConf DictConfig that mimics llm_foundry_config.yaml.
    Falls back to a plain dict if omegaconf is not available.
    """
    try:
        from omegaconf import OmegaConf as om
        raw = {"model": {model_key: params}}
        return om.create(raw)
    except ImportError:
        # Return a plain dict-like object for environments without omegaconf
        class _DictLike(dict):
            def get(self, k, default=None):
                return super().get(k, default)
        return _DictLike({"model": _DictLike({model_key: _DictLike(params)})})


def _make_yaml_file(params: dict, directory: str, filename: str = "llm_foundry_config.yaml") -> str:
    """Write a minimal llm_foundry_config.yaml to a temp directory."""
    import yaml
    content = {"model": {"params": params}}
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        yaml.dump(content, f)
    return path


# ---------------------------------------------------------------------------
# Tests: _find_llmfoundry_yaml
# ---------------------------------------------------------------------------

class TestFindLlmfoundryYaml:
    def test_finds_in_artifacts_subdir(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        yaml_file = artifacts / "llm_foundry_config.yaml"
        yaml_file.write_text("model: {}")

        result = _find_llmfoundry_yaml(tmp_path)
        assert result == yaml_file

    def test_finds_in_parent_artifacts(self, tmp_path):
        checkpoint = tmp_path / "ep0_ba1000"
        checkpoint.mkdir()
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        yaml_file = artifacts / "llm_foundry_config.yaml"
        yaml_file.write_text("model: {}")

        result = _find_llmfoundry_yaml(checkpoint)
        assert result == yaml_file

    def test_finds_in_parent_dir_for_hf_checkpoint(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        checkpoint = artifacts / "hf_checkpoint"
        checkpoint.mkdir()
        yaml_file = artifacts / "llm_foundry_config.yaml"
        yaml_file.write_text("model: {}")

        result = _find_llmfoundry_yaml(checkpoint)
        assert result == yaml_file

    def test_finds_directly_in_dir(self, tmp_path):
        yaml_file = tmp_path / "llm_foundry_config.yaml"
        yaml_file.write_text("model: {}")

        result = _find_llmfoundry_yaml(tmp_path)
        assert result == yaml_file

    def test_returns_none_when_not_found(self, tmp_path):
        result = _find_llmfoundry_yaml(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _get_model_params
# ---------------------------------------------------------------------------

class TestGetModelParams:
    def test_extracts_from_model_params(self):
        cfg = _make_yaml_cfg({"hidden_size": 4096, "num_hidden_layers": 32})
        params = _get_model_params(cfg)
        assert params["hidden_size"] == 4096
        assert params["num_hidden_layers"] == 32

    def test_extracts_from_model_llm_params(self):
        try:
            from omegaconf import OmegaConf as om
            cfg = om.create({"model": {"llm": {"params": {"hidden_size": 7168}}}})
        except ImportError:
            pytest.skip("omegaconf not available")
        params = _get_model_params(cfg)
        assert params["hidden_size"] == 7168

    def test_extracts_from_model_config_overrides(self):
        cfg = _make_yaml_cfg(
            {
                "attention_type": "LlamaLatentSparseAttention",
                "dense_attention_layers": [3, 7],
            },
            model_key="config_overrides",
        )

        params = _get_model_params(cfg)

        assert params["attention_type"] == "LlamaLatentSparseAttention"
        assert params["dense_attention_layers"] == [3, 7]

    def test_returns_empty_for_missing_params(self):
        try:
            from omegaconf import OmegaConf as om
            cfg = om.create({"model": {}})
        except ImportError:
            pytest.skip("omegaconf not available")
        params = _get_model_params(cfg)
        assert params == {}


# ---------------------------------------------------------------------------
# Tests: extract_hybrid_fields_from_llmfoundry
# ---------------------------------------------------------------------------

class TestExtractHybridFields:
    def test_extracts_dense_attention_layers(self):
        cfg = _make_yaml_cfg(
            {
                "dense_attention_layers": [3, 7],
                "indexer_norm_type": "LlamaRMSNorm",
            }
        )

        result = extract_hybrid_fields_from_llmfoundry(cfg)

        assert result["dense_attention_layers"] == [3, 7]
        assert result["indexer_norm_type"] == "LlamaRMSNorm"

    def test_extracts_linear_attention_type(self):
        cfg = _make_yaml_cfg({
            "linear_attention_type": "Qwen3NextGatedDeltaNet",
            "full_attention_layers": [0, 2, 4],
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 64,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 8,
            "linear_num_value_heads": 16,
            "linear_use_short_conv": True,
        })
        result = extract_hybrid_fields_from_llmfoundry(cfg)

        assert result["linear_attention_type"] == "Qwen3NextGatedDeltaNet"
        assert result["full_attention_layers"] == [0, 2, 4]
        assert result["linear_key_head_dim"] == 64
        assert result["linear_value_head_dim"] == 64
        assert result["linear_conv_kernel_dim"] == 4
        assert result["linear_num_key_heads"] == 8
        assert result["linear_num_value_heads"] == 16
        assert result["linear_use_short_conv"] is True

    @pytest.mark.parametrize("flag", [True, False])
    def test_extracts_use_linear_attention_in_mtp(self, flag):
        cfg = _make_yaml_cfg({
            "linear_attention_type": "Qwen3NextGatedDeltaNet",
            "use_mtp": True,
            "mtp_predictor_num": 1,
            "use_linear_attention_in_mtp": flag,
        })
        result = extract_hybrid_fields_from_llmfoundry(cfg)

        assert result["use_linear_attention_in_mtp"] is flag

    def test_omits_use_linear_attention_in_mtp_when_absent(self):
        cfg = _make_yaml_cfg({
            "linear_attention_type": "Qwen3NextGatedDeltaNet",
            "use_mtp": True,
            "mtp_predictor_num": 1,
        })
        result = extract_hybrid_fields_from_llmfoundry(cfg)

        assert "use_linear_attention_in_mtp" not in result

    def test_sets_model_type_for_hybrid(self):
        cfg = _make_yaml_cfg({"linear_attention_type": "Qwen3NextGatedDeltaNet"})
        result = extract_hybrid_fields_from_llmfoundry(cfg)
        assert result["model_type"] == "gigachat3_5"
        assert result["architectures"] == ["Gigachat35ForCausalLM"]
        assert "auto_map" in result
        assert "AutoConfig" in result["auto_map"]
        assert "AutoModel" in result["auto_map"]
        assert "AutoModelForCausalLM" in result["auto_map"]

    def test_no_model_type_override_for_plain_deepseek(self):
        cfg = _make_yaml_cfg({
            "num_routed_experts": 64,
            "norm_topk_prob": True,
        })
        result = extract_hybrid_fields_from_llmfoundry(cfg)
        assert "model_type" not in result
        assert result.get("n_routed_experts") == 64

    def test_renames_num_routed_experts(self):
        cfg = _make_yaml_cfg({"num_routed_experts": 256})
        result = extract_hybrid_fields_from_llmfoundry(cfg)
        assert result["n_routed_experts"] == 256
        assert "num_routed_experts" not in result

    def test_renames_router_aux_loss_coef(self):
        cfg = _make_yaml_cfg({"router_aux_loss_coef": 0.001})
        result = extract_hybrid_fields_from_llmfoundry(cfg)
        assert result["aux_loss_alpha"] == 0.001

    def test_dense_intermediate_size_overrides_intermediate_size(self):
        cfg = _make_yaml_cfg({
            "intermediate_size": 2048,
            "dense_intermediate_size": 18432,
        })
        result = extract_hybrid_fields_from_llmfoundry(cfg)
        assert result["intermediate_size"] == 18432

    def test_moe_intermediate_size_set_from_intermediate_size(self):
        cfg = _make_yaml_cfg({"intermediate_size": 2048})
        result = extract_hybrid_fields_from_llmfoundry(cfg)
        assert result.get("moe_intermediate_size") == 2048

    def test_returns_empty_for_empty_params(self):
        try:
            from omegaconf import OmegaConf as om
            cfg = om.create({"model": {}})
        except ImportError:
            pytest.skip("omegaconf not available")
        result = extract_hybrid_fields_from_llmfoundry(cfg)
        assert result == {}

    def test_passthrough_moe_fields(self):
        cfg = _make_yaml_cfg({
            "attention_type": "LlamaLatentAttention",
            "num_krope_heads": 1,
            "norm_topk_prob": True,
            "first_k_dense_replace": 3,
            "n_shared_experts": 1,
            "norm_type": "ZeroCenteredGatedNorm",
            "layernorm_type": "pre",
            "layernorm_gating_weight": 2.0,
            "use_shared_expert_sigmoid": True,
            "use_mla_scaling_factor": True,
            "rms_norm_eps": 1e-20,
            "scoring_func": "sigmoid",
            "aux_loss_free": True,
            "indexer_rotate_activation_enabled": True,
        })
        result = extract_hybrid_fields_from_llmfoundry(cfg)
        assert result["attention_type"] == "LlamaLatentAttention"
        assert result["num_krope_heads"] == 1
        assert result["norm_topk_prob"] is True
        assert result["first_k_dense_replace"] == 3
        assert result["n_shared_experts"] == 1
        assert result["norm_type"] == "ZeroCenteredGatedNorm"
        assert result["layernorm_type"] == "pre"
        assert result["layernorm_gating_weight"] == 2.0
        assert result["use_shared_expert_sigmoid"] is True
        assert result["use_mla_scaling_factor"] is True
        assert result["rms_norm_eps"] == 1e-20
        assert result["scoring_func"] == "sigmoid"
        assert result["aux_loss_free"] is True
        assert result["indexer_rotate_activation_enabled"] is True

    def test_passthrough_dense_attention_and_mtp_fields(self):
        params = {
            "head_dim": 8,
            "attention_hidden_size": 32,
            "apply_qk_norm": False,
            "attention_bias": False,
            "mtp_block_type": "dense",
            "mtp_recurrent": True,
        }
        result = extract_hybrid_fields_from_llmfoundry(_make_yaml_cfg(params))
        assert {key: result[key] for key in params} == params

    def test_extracts_optimizer_eps(self):
        cfg = _make_yaml_cfg({"linear_attention_type": "Qwen3NextGatedDeltaNet"})
        cfg["optimizer"] = {"eps": 1e-20}

        result = extract_hybrid_fields_from_llmfoundry(cfg)

        assert result["optimizer_eps"] == 1e-20


# ---------------------------------------------------------------------------
# Tests: enrich_hf_config_from_llmfoundry_yaml
# ---------------------------------------------------------------------------

class TestEnrichHfConfigFromLlmfoundryYaml:
    def test_enriches_with_hybrid_fields(self, tmp_path):
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not available")

        yaml_content = {
            "model": {
                "params": {
                    "linear_attention_type": "Qwen3NextGatedDeltaNet",
                    "full_attention_layers": [0, 2, 4],
                    "linear_key_head_dim": 64,
                }
            }
        }
        yaml_path = tmp_path / "llm_foundry_config.yaml"
        import yaml as _yaml
        with open(yaml_path, "w") as f:
            _yaml.dump(yaml_content, f)

        hf_cfg = {
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
        }
        result = enrich_hf_config_from_llmfoundry_yaml(hf_cfg, yaml_path=str(yaml_path))

        assert result["linear_attention_type"] == "Qwen3NextGatedDeltaNet"
        assert result["full_attention_layers"] == [0, 2, 4]
        assert result["linear_key_head_dim"] == 64
        # model_type should be overridden to hybrid
        assert result["model_type"] == "gigachat3_5"

    def test_does_not_overwrite_core_fields(self, tmp_path):
        try:
            import yaml as _yaml
        except ImportError:
            pytest.skip("pyyaml not available")

        yaml_content = {"model": {"params": {"linear_attention_type": "X"}}}
        yaml_path = tmp_path / "llm_foundry_config.yaml"
        with open(yaml_path, "w") as f:
            _yaml.dump(yaml_content, f)

        hf_cfg = {
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
        result = enrich_hf_config_from_llmfoundry_yaml(hf_cfg, yaml_path=str(yaml_path))

        # Core fields must not be overwritten
        assert result["hidden_size"] == 7168
        assert result["num_hidden_layers"] == 61
        assert result["bos_token_id"] == 1
        assert result["eos_token_id"] == 2

    def test_warns_and_returns_unchanged_when_yaml_missing(self, tmp_path):
        hf_cfg = {"model_type": "deepseek_v3"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = enrich_hf_config_from_llmfoundry_yaml(
                hf_cfg, yaml_path=str(tmp_path / "nonexistent.yaml")
            )
        assert result == {"model_type": "deepseek_v3"}
        assert any("not found" in str(warning.message).lower() for warning in w)

    def test_warns_and_returns_unchanged_when_no_path_given(self):
        hf_cfg = {"model_type": "deepseek_v3"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = enrich_hf_config_from_llmfoundry_yaml(hf_cfg)
        assert result == {"model_type": "deepseek_v3"}
        assert any(issubclass(warning.category, RuntimeWarning) for warning in w)

    def test_auto_discovers_yaml_from_checkpoint_path(self, tmp_path):
        try:
            import yaml as _yaml
        except ImportError:
            pytest.skip("pyyaml not available")

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        yaml_content = {"model": {"params": {"linear_attention_type": "GatedDeltaNet"}}}
        with open(artifacts / "llm_foundry_config.yaml", "w") as f:
            _yaml.dump(yaml_content, f)

        hf_cfg = {"model_type": "deepseek_v3"}
        result = enrich_hf_config_from_llmfoundry_yaml(hf_cfg, checkpoint_path=str(tmp_path))
        assert result["linear_attention_type"] == "GatedDeltaNet"


# ---------------------------------------------------------------------------
# Tests: is_gigachat_model
# ---------------------------------------------------------------------------

# Realistic giga_mix / DeepSeek training config — has GigaChat3.5 fields
# but with disabled (False / RMSNorm) values. Must NOT be detected as GigaChat3.5.
_GIGA_MIX_CONFIG = {
    "model_type": "giga_mix",
    "aux_loss_free": True,
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "rope_interleave": True,         # HF DeepseekV3Attention default
    "num_krope_heads": 1,            # also appears in DeepSeek training configs
    "attention_type": "LlamaLatentAttention",
    "linear_attention_type": None,
    "gated_attention": False,        # feature present but disabled
    "use_mla_scaling_factor": False,
    "use_shared_expert_sigmoid": False,
    "layernorm_type": "pre",         # RMSNorm-family variant — not GigaChat3.5
    "layernorm_gating_weight": 2,
}

# Minimal realistic GigaChat3.5 full-softmax config (no hybrid layers).
_GIGACHAT35_FULLMLA_CONFIG = {
    "model_type": "gigachat3_5",
    "gated_attention": True,
    "use_mla_scaling_factor": True,
    "layernorm_type": "ZeroCenteredGatedNorm",
    "linear_attention_type": None,
}


class TestIsGigachatModel:
    # --- positive: model_type path ---

    def test_true_for_gigachat35_model_type(self):
        assert is_gigachat_model({"model_type": "gigachat3_5"}) is True

    def test_true_for_full_gigachat35_config(self):
        assert is_gigachat_model(_GIGACHAT35_FULLMLA_CONFIG) is True

    # --- positive: linear_attention_type path ---

    def test_true_for_hybrid_linear_attention_type(self):
        assert is_gigachat_model({"linear_attention_type": "Qwen3NextGatedDeltaNet"}) is True

    def test_true_for_hybrid_gated_delta_net(self):
        assert is_gigachat_model({"linear_attention_type": "GatedDeltaNet"}) is True

    # --- positive: boolean marker fields (must be truthy) ---

    @pytest.mark.parametrize("field", sorted(_GIGACHAT35_BOOL_MARKERS))
    def test_true_for_each_bool_marker_truthy(self, field):
        assert is_gigachat_model({field: True}) is True

    # --- positive: layernorm_type non-RMSNorm ---

    def test_true_for_zerocent_gated_norm(self):
        assert is_gigachat_model({"layernorm_type": "ZeroCenteredGatedNorm"}) is True

    def test_true_for_unknown_layernorm_type(self):
        assert is_gigachat_model({"layernorm_type": "SomeCustomNorm"}) is True

    # --- negative: pure DeepSeek V3 ---

    def test_false_for_deepseek_v3_model_type(self):
        assert is_gigachat_model({"model_type": "deepseek_v3"}) is False

    def test_false_for_giga_mix_config(self):
        # giga_mix carries GigaChat3.5 fields with disabled values — must not fire.
        assert is_gigachat_model(_GIGA_MIX_CONFIG) is False

    def test_false_for_empty_dict(self):
        assert is_gigachat_model({}) is False

    # --- false-positive guards ---

    def test_false_for_rope_interleave_only(self):
        assert is_gigachat_model({"rope_interleave": True}) is False

    def test_false_for_num_krope_heads_only(self):
        assert is_gigachat_model({"num_krope_heads": 1}) is False

    def test_false_for_layernorm_gating_weight_only(self):
        # layernorm_gating_weight alone is not a marker (removed from markers).
        assert is_gigachat_model({"layernorm_gating_weight": 2.0}) is False

    def test_false_for_bool_markers_set_to_false(self):
        assert is_gigachat_model({f: False for f in _GIGACHAT35_BOOL_MARKERS}) is False

    def test_false_for_layernorm_type_rmsnorm(self):
        assert is_gigachat_model({"layernorm_type": "rmsnorm"}) is False

    def test_false_for_layernorm_type_llama_rmsnorm(self):
        assert is_gigachat_model({"layernorm_type": "LlamaRMSNorm"}) is False

    def test_false_for_layernorm_type_pre(self):
        # "pre" is the value present in giga_mix configs — must not trigger.
        assert is_gigachat_model({"layernorm_type": "pre"}) is False

    def test_false_for_layernorm_type_none(self):
        assert is_gigachat_model({"layernorm_type": None}) is False

    def test_false_for_none_linear_attention_type(self):
        assert is_gigachat_model({"linear_attention_type": None}) is False

    # --- is_gigachat_model vs is_hybrid_model orthogonality ---

    def test_gigachat_fullmla_is_not_hybrid(self):
        cfg = {"model_type": "gigachat3_5", "linear_attention_type": None}
        assert is_gigachat_model(cfg) is True
        assert is_hybrid_model(cfg) is False

    def test_hybrid_gigachat_is_both(self):
        cfg = {"model_type": "gigachat3_5", "linear_attention_type": "GatedDeltaNet"}
        assert is_gigachat_model(cfg) is True
        assert is_hybrid_model(cfg) is True


# ---------------------------------------------------------------------------
# Tests: is_hybrid_model
# ---------------------------------------------------------------------------

class TestIsHybridModel:
    def test_true_for_linear_attention_type(self):
        assert is_hybrid_model({"linear_attention_type": "Qwen3NextGatedDeltaNet"}) is True

    def test_false_for_gigachat_without_linear_attention(self):
        # GigaChat3.5 can be full-MLA; that's not hybrid attention.
        assert is_hybrid_model({"model_type": "gigachat3_5"}) is False

    def test_false_for_giga_mix_without_linear_attention(self):
        assert is_hybrid_model({"model_type": "giga_mix"}) is False

    def test_false_for_plain_deepseek(self):
        assert is_hybrid_model({"model_type": "deepseek_v3"}) is False

    def test_false_for_empty_dict(self):
        assert is_hybrid_model({}) is False

    def test_false_for_none_linear_attention_type(self):
        assert is_hybrid_model({"linear_attention_type": None}) is False
