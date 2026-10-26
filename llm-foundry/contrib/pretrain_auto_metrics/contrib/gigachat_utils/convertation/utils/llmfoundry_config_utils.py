"""
Utilities for reading llm_foundry_config.yaml and mapping GigaMixConfig parameters
to HuggingFace config fields for HF checkpoint generation.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from omegaconf import OmegaConf as om, DictConfig
    _OMEGACONF_AVAILABLE = True
except ImportError:
    _OMEGACONF_AVAILABLE = False


# ---------------------------------------------------------------------------
# GigaMixConfig → HF config field mappings
# ---------------------------------------------------------------------------

# Fields that are passed through with the same name
_PASSTHROUGH_FIELDS = (
    "gated_attention",
    "attention_distillation",
    "norm_type",
    "indexer_norm_type",
    "layernorm_type",
    "layernorm_gating_weight",
    "use_shared_expert_sigmoid",
    "use_mla_scaling_factor",
    "rms_norm_eps",
    "linear_attention_type",
    "full_attention_layers",
    "dense_attention_layers",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_use_short_conv",
    "indexer_rotate_activation_enabled",
    # Dense attention
    "head_dim",
    "attention_hidden_size",
    "apply_qk_norm",
    "attention_bias",
    # MoE / routing
    "norm_topk_prob",
    "first_k_dense_replace",
    "n_shared_experts",
    "scoring_func",
    "moe_router_num_groups",
    "moe_router_group_topk",
    "moe_router_routed_scaling_factor",
    "aux_loss_free",
    "moe_router_bias_update_rate",
    # MTP
    "use_mtp",
    "mtp_predictor_num",
    "mtp_block_type",
    "mtp_recurrent",
    "use_linear_attention_in_mtp",
    # Rope
    "attention_type",
    "num_krope_heads",
    "rope_interleave",
    "rope_scaling",
    "rope_theta",
    # SwiGLU activation clipping bound (hard clamp; 0.0 = disabled)
    "swiglu_limit",
)

# Boolean fields that identify GigaChat3.5 when truthy.
# DeepSeek training configs may carry these fields too but set them to False.
_GIGACHAT35_BOOL_MARKERS: frozenset = frozenset({
    "gated_attention",
    "attention_distillation",
    "use_shared_expert_sigmoid",
    "use_mla_scaling_factor",
    # NOTE: rope_interleave excluded — HF DeepseekV3Attention uses it by default.
    # NOTE: num_krope_heads excluded — also appears in DeepSeek training configs.
})

# layernorm_type values that belong to DeepSeek / plain RMSNorm models.
# Any other non-None value indicates a GigaChat3.5-specific norm variant.
_DEEPSEEK_LAYERNORM_TYPES: frozenset = frozenset({
    "rmsnorm",
    "LlamaRMSNorm",
    "pre",  # papaya_mix / DeepSeek training configs use this value
})

# Fields that need renaming: llmfoundry_name → hf_name
_RENAME_FIELDS: Dict[str, str] = {
    "num_routed_experts": "n_routed_experts",
    "router_aux_loss_coef": "aux_loss_alpha",
    "dense_intermediate_size": "intermediate_size",  # only if present
}

# Rename fields that apply only to GigaChat3.5 configs.
_GIGACHAT35_RENAME_FIELDS: Dict[str, str] = {
    "lm_head_logit_softcapping": "final_logit_softcapping",
    "indexer_n_heads": "index_n_heads",
    "indexer_head_dim": "index_head_dim",
    "indexer_topk": "index_topk",
}


def _find_llmfoundry_yaml(checkpoint_path: Union[str, Path]) -> Optional[Path]:
    """
    Locate llm_foundry_config.yaml relative to a checkpoint directory.

    Tries the following locations in order:
      1. <checkpoint_path>/artifacts/llm_foundry_config.yaml
      2. <checkpoint_path>/../llm_foundry_config.yaml
      3. <checkpoint_path>/../artifacts/llm_foundry_config.yaml
      4. <checkpoint_path>/llm_foundry_config.yaml
    """
    checkpoint_path = Path(checkpoint_path)
    candidates = [
        checkpoint_path / "artifacts" / "llm_foundry_config.yaml",
        checkpoint_path.parent / "llm_foundry_config.yaml",
        checkpoint_path.parent / "artifacts" / "llm_foundry_config.yaml",
        checkpoint_path / "llm_foundry_config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_llmfoundry_yaml(yaml_path: Union[str, Path]) -> DictConfig:
    """Load and return the llm_foundry_config.yaml as an OmegaConf DictConfig."""
    if not _OMEGACONF_AVAILABLE:
        raise ImportError("omegaconf is required to load llm_foundry_config.yaml")
    with open(yaml_path) as f:
        return om.load(f)


def _get_model_params(yaml_cfg: "DictConfig") -> Dict[str, Any]:
    """
    Extract the flat model params dict from the llm_foundry_config.yaml.

    Handles both:
      - model.params  (legacy GigaMixCausalLM style)
      - model.config_overrides  (Papaya/GigaMix Composer style)
      - model.llm.params  (GigaTTS style)
    """
    model_cfg = yaml_cfg.get("model", {})
    if "params" in model_cfg:
        params = model_cfg["params"]
    elif "config_overrides" in model_cfg:
        params = model_cfg["config_overrides"]
    elif "llm" in model_cfg and "params" in model_cfg["llm"]:
        params = model_cfg["llm"]["params"]
    else:
        return {}

    # Convert OmegaConf to plain dict so callers don't need to import omegaconf
    if _OMEGACONF_AVAILABLE and isinstance(params, DictConfig):
        return dict(om.to_container(params, resolve=True))
    return dict(params)


def extract_hybrid_fields_from_llmfoundry(
    yaml_cfg: "DictConfig",
) -> Dict[str, Any]:
    """
    Extract hybrid-attention-specific and other HF-relevant fields from
    llm_foundry_config.yaml model params.

    Returns a dict of HF config fields that should be overlaid on top of
    the embedded Composer HF config.
    """
    params = _get_model_params(yaml_cfg)
    if not params:
        return {}

    result: Dict[str, Any] = {}

    # Pass-through fields (same name in GigaMixConfig and HF config)
    for field in _PASSTHROUGH_FIELDS:
        if field in params and params[field] is not None:
            result[field] = params[field]

    # Renamed fields
    for llmf_name, hf_name in _RENAME_FIELDS.items():
        if llmf_name in params and params[llmf_name] is not None:
            # Only add if not already present under the HF name
            if hf_name not in result:
                result[hf_name] = params[llmf_name]

    # Special case: dense_intermediate_size → intermediate_size
    # Only override if dense_intermediate_size is explicitly set
    if "dense_intermediate_size" in params and params["dense_intermediate_size"] is not None:
        result["intermediate_size"] = params["dense_intermediate_size"]

    # Special case: moe_intermediate_size (keep as-is if present)
    if "intermediate_size" in params and "moe_intermediate_size" not in result:
        # In GigaMixConfig, intermediate_size is the MoE expert size
        result.setdefault("moe_intermediate_size", params["intermediate_size"])

    optimizer_cfg = yaml_cfg.get("optimizer", {})
    optimizer_eps = optimizer_cfg.get("eps") if optimizer_cfg else None
    if optimizer_eps is not None:
        result["optimizer_eps"] = optimizer_eps

    # Determine model_type for hybrid models
    if result.get("linear_attention_type") is not None:
        result["model_type"] = "gigachat3_5"
        result["architectures"] = ["Gigachat35ForCausalLM"]
        result["auto_map"] = {
            "AutoConfig": "configuration_gigachat3_5.Gigachat35Config",
            "AutoModel": "modeling_gigachat3_5.Gigachat35Model",
            "AutoModelForCausalLM": "modeling_gigachat3_5.Gigachat35ForCausalLM",
        }

    return result


def enrich_hf_config_from_llmfoundry_yaml(
    hf_cfg_dict: Dict[str, Any],
    checkpoint_path: Union[str, Path, None] = None,
    yaml_path: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """
    Overlay hybrid-specific fields from llm_foundry_config.yaml onto an
    existing HF config dict.

    Args:
        hf_cfg_dict: The HF config dict to enrich (modified in-place and returned).
        checkpoint_path: Path to the gigafsdp checkpoint directory. Used to
            auto-locate llm_foundry_config.yaml if yaml_path is not given.
        yaml_path: Explicit path to llm_foundry_config.yaml. Takes priority
            over checkpoint_path.

    Returns:
        The enriched hf_cfg_dict (same object, modified in-place).
    """
    # Locate the yaml
    resolved_yaml: Optional[Path] = None
    if yaml_path is not None:
        resolved_yaml = Path(yaml_path)
        if not resolved_yaml.is_file():
            warnings.warn(
                f"llm_foundry_config.yaml not found at explicit path: {yaml_path}. "
                "Skipping config enrichment.",
                RuntimeWarning,
            )
            return hf_cfg_dict
    elif checkpoint_path is not None:
        resolved_yaml = _find_llmfoundry_yaml(checkpoint_path)
        if resolved_yaml is None:
            warnings.warn(
                f"llm_foundry_config.yaml not found near checkpoint: {checkpoint_path}. "
                "Skipping config enrichment.",
                RuntimeWarning,
            )
            return hf_cfg_dict
    else:
        warnings.warn(
            "Neither checkpoint_path nor yaml_path provided. "
            "Skipping config enrichment from llm_foundry_config.yaml.",
            RuntimeWarning,
        )
        return hf_cfg_dict

    try:
        yaml_cfg = load_llmfoundry_yaml(resolved_yaml)
    except Exception as e:
        warnings.warn(
            f"Failed to load llm_foundry_config.yaml from {resolved_yaml}: {e}. "
            "Skipping config enrichment.",
            RuntimeWarning,
        )
        return hf_cfg_dict

    extra_fields = extract_hybrid_fields_from_llmfoundry(yaml_cfg)
    params = _get_model_params(yaml_cfg)

    if is_gigachat_model(hf_cfg_dict) or is_gigachat_model(extra_fields):
        for llmf_name, hf_name in _GIGACHAT35_RENAME_FIELDS.items():
            if llmf_name in params and params[llmf_name] is not None:
                if hf_name not in extra_fields:
                    extra_fields[hf_name] = params[llmf_name]

    if not extra_fields:
        return hf_cfg_dict

    # Overlay: fields from llm_foundry_config.yaml take priority for hybrid-specific
    # fields, but we don't overwrite core architecture fields that are already correct
    _CORE_FIELDS = {
        "vocab_size", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads",
        "bos_token_id", "eos_token_id", "pad_token_id",
        "torch_dtype",
    }
    for k, v in extra_fields.items():
        if k not in _CORE_FIELDS:
            hf_cfg_dict[k] = v

    return hf_cfg_dict


def is_gigachat_model(hf_cfg_dict: Dict[str, Any]) -> bool:
    """Return True if the config represents a GigaChat3.5 model (hybrid or full-softmax variant).

    Detection is based on:
    1. model_type == "gigachat3_5"
    2. DSA indexer config is present (indexer_topk or index_topk is not None)
    3. linear_attention_type is not None (hybrid layers present)
    4. Any boolean GigaChat3.5 marker field is truthy
       (gated_attention, attention_distillation, use_shared_expert_sigmoid, use_mla_scaling_factor)
    5. layernorm_type is set to a non-RMSNorm value (not "rmsnorm" / "LlamaRMSNorm")
    """
    if hf_cfg_dict.get("model_type") == "gigachat3_5":
        return True
    if any(
        hf_cfg_dict.get(field) is not None
        for field in ("indexer_topk", "index_topk")
    ):
        return True
    if hf_cfg_dict.get("linear_attention_type") is not None:
        return True
    if any(hf_cfg_dict.get(f) for f in _GIGACHAT35_BOOL_MARKERS):
        return True
    layernorm_type = hf_cfg_dict.get("layernorm_type")
    if layernorm_type is not None and layernorm_type not in _DEEPSEEK_LAYERNORM_TYPES:
        return True
    return False


def is_hybrid_model(hf_cfg_dict: Dict[str, Any]) -> bool:
    """Return True if the model uses a hybrid (linear + full-softmax) attention layout.

    GigaChat3.5 can be trained in pure full-MLA mode (linear_attention_type=None),
    so is_hybrid_model=False does NOT imply the model is DeepSeek.
    Use is_gigachat_model() to test for the GigaChat3.5 architecture family.
    """
    return hf_cfg_dict.get("linear_attention_type") is not None
