"""
Initialize a random-weight HF checkpoint from an llm-foundry config.

Reads an llm-foundry YAML config, maps it to a HuggingFace config,
instantiates the model with random weights, and saves as a checkpoint
compatible with:
  - transformers AutoModelForCausalLM.from_pretrained (via auto_map + custom code)
  - sglang/vllm (Gigachat35ForCausalLM / DeepseekV3ForCausalLM architecture)

By default, the script uses the custom sglang implementation from this
repository and selects the config by --model_type. With
--use_custom_sglang False, the script imports the open-source sglang
installed in the runtime environment and loads the base config from
--model_tag, which must match the model tag used by that sglang.

Supports two initialization modes:
  - **cpu** (default): Instantiate the full model on CPU, then save.
    Simple but requires enough CPU RAM for the entire model.
  - **cuda**: Layer-by-layer GPU initialization via meta device.
    Only one layer's worth of GPU memory is needed at a time,
    making it possible to initialize very large models (700B+)
    on a single GPU.  Requires the ``accelerate`` package (falls back
    to ``torch.device("meta")`` if unavailable).

The key mapping logic is based on:
  contrib/gigachat-utils/convertation/mapper_deepseek_moe_to_hf.py
  contrib/gigachat-utils/convertation/models/gigachat3_5/

Usage:
    python init_random_hf_model.py \
        --config /path/to/llm_foundry_config.yaml \
        --output_dir /path/to/output_hf_checkpoint \
        --model_type gigachat35

    # Use the open-source sglang installed in the runtime image.
    python init_random_hf_model.py \
        --config /path/to/llm_foundry_config.yaml \
        --output_dir /path/to/output_hf_checkpoint \
        --use_custom_sglang False \
        --model_tag deepseek-ai/DeepSeek-V3.1

    # GPU-accelerated initialization for large models
    python init_random_hf_model.py \
        --config /path/to/llm_foundry_config.yaml \
        --output_dir /path/to/output_hf_checkpoint \
        --model_type gigachat35 \
        --device cuda
"""

import json
import os
import math
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import argparse
import torch
import yaml


# ---------------------------------------------------------------------------
# Paths to modeling code in the llm-foundry submodule
# ---------------------------------------------------------------------------
GIGACHAT_MODELS_DIR = (
    Path(__file__).resolve().parents[1]
    / "contrib"
    / "gigachat-utils"
    / "convertation"
    / "models"
    / "gigachat3_5"
)

DEEPSEEK_MODELS_DIR = (
    Path(__file__).resolve().parents[1]
    / "contrib"
    / "gigachat-utils"
    / "convertation"
    / "models"
    / "DeepSeekV3"
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_llm_foundry_config(config_path: str) -> dict:
    """Load llm-foundry YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_common_hf_config(
    overrides: dict, llm_config: dict, include_mtp: bool = False
) -> dict:
    """Build the common part of the HF config dict shared by both architectures.

    Field mapping follows ``mapper_deepseek_moe_to_hf.py``:
      - llm-foundry ``intermediate_size``      -> HF ``moe_intermediate_size``
      - llm-foundry ``dense_intermediate_size`` -> HF ``intermediate_size``
      - llm-foundry ``num_routed_experts``      -> HF ``n_routed_experts``
      - llm-foundry ``router_aux_loss_coef``    -> HF ``aux_loss_alpha``
      - llm-foundry ``moe_router_routed_scaling_factor`` -> HF ``routed_scaling_factor``
      - llm-foundry ``moe_router_num_groups``   -> HF ``n_group``
      - llm-foundry ``moe_router_group_topk``   -> HF ``topk_group``
      - llm-foundry ``q_lora_rank == 0``        -> HF ``q_lora_rank = None``
    """
    cfg = {}

    # --- Renamed mappings (shared by both architectures) ---
    # llm-foundry intermediate_size -> HF moe_intermediate_size
    cfg["moe_intermediate_size"] = overrides.get("intermediate_size")
    # llm-foundry dense_intermediate_size -> HF intermediate_size
    cfg["intermediate_size"] = overrides.get(
        "dense_intermediate_size", overrides.get("intermediate_size")
    )
    # num_routed_experts -> n_routed_experts
    cfg["n_routed_experts"] = overrides.get("num_routed_experts")
    # router_aux_loss_coef -> aux_loss_alpha
    cfg["aux_loss_alpha"] = overrides.get("router_aux_loss_coef", 0.001)
    # moe_router_routed_scaling_factor -> routed_scaling_factor
    cfg["routed_scaling_factor"] = overrides.get("moe_router_routed_scaling_factor", 1)

    # --- Group routing ---
    g_topk = overrides.get("moe_router_group_topk")
    n_groups = overrides.get("moe_router_num_groups")
    if g_topk is None and n_groups is None:
        cfg["n_group"] = 1
        cfg["topk_group"] = 1
    else:
        cfg["n_group"] = n_groups if n_groups is not None else 1
        cfg["topk_group"] = g_topk if g_topk is not None else 1

    # --- q_lora_rank: 0 means None (no LoRA for q) ---
    q_lora_rank = overrides.get("q_lora_rank")
    cfg["q_lora_rank"] = None if q_lora_rank == 0 else q_lora_rank

    # --- MTP (multi-token prediction) ---
    use_mtp = overrides.get("use_mtp", False)
    if include_mtp and use_mtp:
        cfg["num_nextn_predict_layers"] = overrides.get("mtp_predictor_num", 1)
    else:
        cfg["num_nextn_predict_layers"] = 0

    # --- rope_scaling ---
    rope_scaling = overrides.get("rope_scaling")
    if isinstance(rope_scaling, str):
        rope_scaling = {"type": rope_scaling}
    cfg["rope_scaling"] = rope_scaling

    # --- Token IDs ---
    cfg["eos_token_id"] = llm_config.get("eos_token_id", 2)
    cfg["bos_token_id"] = 1
    cfg["pad_token_id"] = llm_config.get("eos_token_id", 2)

    # --- sglang/vllm compatibility fields ---
    cfg["topk_method"] = "noaux_tc"
    cfg["scoring_func"] = "sigmoid"
    # moe_layer_freq: every N-th layer after first_k_dense_replace is MoE.
    # 1 = every layer is MoE (standard DeepSeek-V3 behavior).
    cfg["moe_layer_freq"] = 1
    # ep_size: expert parallelism size (1 = no EP).
    cfg["ep_size"] = 1

    return cfg


def build_gigachat35_hf_config_dict(
    llm_config: dict, include_mtp: bool = False
) -> dict:
    """Build a Gigachat35Config-compatible dict from llm-foundry config."""
    overrides = llm_config["model"]["config_overrides"]
    cfg = _build_common_hf_config(overrides, llm_config, include_mtp)

    # --- Direct mappings (same name in both llm-foundry and HF) ---
    _direct_keys = [
        "vocab_size",
        "max_position_embeddings",
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "attention_dropout",
        "hidden_act",
        "initializer_range",
        "tie_word_embeddings",
        "use_cache",
        "attention_bias",
        "attention_type",
        "num_krope_heads",
        "norm_type",
        "layernorm_type",
        "gated_attention",
        "use_shared_expert_sigmoid",
        "use_mla_scaling_factor",
        "n_shared_experts",
        "num_experts_per_tok",
        "norm_topk_prob",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "rope_interleave",
        "first_k_dense_replace",
        # Linear attention (hybrid architecture) fields
        "linear_attention_type",
        "full_attention_layers",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_use_short_conv",
        "attention_distillation",
        "linear_gating_type",
        "linear_sigmoid_gate_scale",
        "linear_attn_o_norm_eps",
        "linear_use_legacy_qkvz_layout",
        "swiglu_limit",
    ]
    for key in _direct_keys:
        if key in overrides:
            cfg[key] = overrides[key]

    # Ensure compatibility with sglang's hybrid_gdn.py which expects full_attention_layer_ids
    if (
        "full_attention_layers" in overrides
        and "full_attention_layer_ids" not in overrides
    ):
        cfg["full_attention_layer_ids"] = overrides["full_attention_layers"]

    # --- Architecture & model type ---
    cfg["model_type"] = "gigachat3_5"
    cfg["architectures"] = ["Gigachat35ForCausalLM"]
    cfg["torch_dtype"] = "bfloat16"

    return cfg


def build_deepseek_v3_hf_config_dict(
    llm_config: dict, include_mtp: bool = False
) -> dict:
    """Build a DeepseekV3Config-compatible dict from llm-foundry config.

    DeepseekV3Config does NOT have Gigachat-specific fields:
      norm_type, layernorm_type, gated_attention, use_shared_expert_sigmoid,
      use_mla_scaling_factor, rope_interleave, num_krope_heads, attention_type.
    These are silently ignored if present in the llm-foundry config.
    """
    overrides = llm_config["model"]["config_overrides"]
    cfg = _build_common_hf_config(overrides, llm_config, include_mtp)

    # --- Direct mappings (same name in both llm-foundry and HF) ---
    # Only keys that DeepseekV3Config.__init__ accepts
    _direct_keys = [
        "vocab_size",
        "max_position_embeddings",
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "attention_dropout",
        "hidden_act",
        "initializer_range",
        "tie_word_embeddings",
        "use_cache",
        "attention_bias",
        "n_shared_experts",
        "num_experts_per_tok",
        "norm_topk_prob",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "first_k_dense_replace",
    ]
    for key in _direct_keys:
        if key in overrides:
            cfg[key] = overrides[key]

    # --- Architecture & model type ---
    cfg["model_type"] = "deepseek_v3"
    cfg["architectures"] = ["DeepseekV3ForCausalLM"]
    cfg["torch_dtype"] = "bfloat16"

    return cfg


def build_hf_config_dict(
    llm_config: dict,
    model_type: str = "gigachat35",
    include_mtp: bool = False,
    use_custom_sglang: bool = True,
    model_tag: str = None,
) -> dict:
    """Build HF config dict for either custom or open-source sglang."""
    if not use_custom_sglang:
        return build_open_source_sglang_hf_config_dict(
            llm_config=llm_config,
            model_tag=model_tag,
            include_mtp=include_mtp,
        )

    if model_type == "gigachat35":
        return build_gigachat35_hf_config_dict(llm_config, include_mtp)
    elif model_type == "deepseek_v3":
        return build_deepseek_v3_hf_config_dict(llm_config, include_mtp)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Supported: gigachat35, deepseek_v3"
        )


def _iter_hf_config_targets(config_dict: dict) -> Iterable[dict]:
    """Yield top-level and common nested text config dicts."""
    yielded = set()
    stack = [config_dict]
    while stack:
        target = stack.pop()
        if not isinstance(target, dict) or id(target) in yielded:
            continue
        yielded.add(id(target))
        yield target

        for nested_key in ("text_config", "llm_config", "language_config"):
            nested = target.get(nested_key)
            if isinstance(nested, dict):
                stack.append(nested)
                nested_config = nested.get("config")
                if isinstance(nested_config, dict):
                    stack.append(nested_config)


def _set_if_present(target: dict, key: str, value: Any) -> bool:
    if key in target and value is not None:
        target[key] = value
        return True
    return False


def _apply_llm_foundry_overrides_to_hf_config_dict(
    hf_config_dict: dict,
    llm_config: dict,
    include_mtp: bool = False,
) -> None:
    """Overlay llm-foundry config values onto an existing HF config.

    This keeps the model family, architecture names, and any sglang-specific
    config fields from ``model_tag`` intact, while replacing dimensions and
    MoE parameters that are present in the llm-foundry config.
    """
    overrides = llm_config["model"]["config_overrides"]

    for target in _iter_hf_config_targets(hf_config_dict):
        for key, value in overrides.items():
            if key in target and value is not None:
                if key == "q_lora_rank" and value == 0:
                    target[key] = None
                else:
                    target[key] = value

        if "moe_intermediate_size" in target:
            _set_if_present(
                target, "moe_intermediate_size", overrides.get("intermediate_size")
            )

        if "intermediate_size" in target:
            _set_if_present(
                target,
                "intermediate_size",
                overrides.get(
                    "dense_intermediate_size", overrides.get("intermediate_size")
                ),
            )

        _set_if_present(target, "n_routed_experts", overrides.get("num_routed_experts"))
        _set_if_present(target, "num_experts", overrides.get("num_routed_experts"))
        _set_if_present(
            target, "num_local_experts", overrides.get("num_routed_experts")
        )
        _set_if_present(target, "aux_loss_alpha", overrides.get("router_aux_loss_coef"))
        _set_if_present(
            target,
            "routed_scaling_factor",
            overrides.get("moe_router_routed_scaling_factor"),
        )
        _set_if_present(target, "n_group", overrides.get("moe_router_num_groups"))
        _set_if_present(target, "topk_group", overrides.get("moe_router_group_topk"))

        q_lora_rank = overrides.get("q_lora_rank")
        if "q_lora_rank" in target and q_lora_rank is not None:
            target["q_lora_rank"] = None if q_lora_rank == 0 else q_lora_rank

        rope_scaling = overrides.get("rope_scaling")
        if rope_scaling is not None and "rope_scaling" in target:
            target["rope_scaling"] = (
                {"type": rope_scaling}
                if isinstance(rope_scaling, str)
                else rope_scaling
            )

        if "num_nextn_predict_layers" in target:
            target["num_nextn_predict_layers"] = (
                overrides.get("mtp_predictor_num", 1)
                if include_mtp and overrides.get("use_mtp", False)
                else 0
            )

        _set_if_present(target, "eos_token_id", llm_config.get("eos_token_id"))
        _set_if_present(target, "pad_token_id", llm_config.get("eos_token_id"))


def _find_hf_config_target_with_key(config_dict: dict, key: str) -> dict:
    for target in _iter_hf_config_targets(config_dict):
        if key in target:
            return target
    raise KeyError(key)


def _import_open_source_sglang_config_registry() -> None:
    import sglang.srt.utils.hf_transformers_utils  # noqa: F401


def build_open_source_sglang_hf_config_dict(
    llm_config: dict,
    model_tag: str,
    include_mtp: bool = False,
) -> dict:
    """Build config from a model tag for the installed open-source sglang."""
    if not model_tag:
        raise ValueError("--model_tag is required when --use_custom_sglang False")

    _import_open_source_sglang_config_registry()

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_tag, trust_remote_code=True)
    hf_config_dict = hf_config.to_dict()
    _apply_llm_foundry_overrides_to_hf_config_dict(
        hf_config_dict,
        llm_config=llm_config,
        include_mtp=include_mtp,
    )
    hf_config_dict.pop("auto_map", None)

    return hf_config_dict


def estimate_model_size_mb(cfg: dict) -> float:
    """Rough estimate of model size in MB (assuming bfloat16, 2 bytes/param)."""
    h = cfg.get("hidden_size", 0)
    n_layers = cfg.get("num_hidden_layers", 0)
    n_experts = cfg.get("n_routed_experts", 0)
    moe_inter = cfg.get("moe_intermediate_size", 0)
    n_shared = cfg.get("n_shared_experts", 0)
    inter = cfg.get("intermediate_size", 0)
    vocab = cfg.get("vocab_size", 0)
    first_k = cfg.get("first_k_dense_replace", 0)
    n_heads = cfg.get("num_attention_heads", 1)
    q_lora = cfg.get("q_lora_rank") or h
    kv_lora = cfg.get("kv_lora_rank") or h
    qk_nope = cfg.get("qk_nope_head_dim", 128)
    qk_rope = cfg.get("qk_rope_head_dim", 64)
    v_dim = cfg.get("v_head_dim", 128)

    bytes_per_param = 2  # bfloat16

    # Embedding + LM head
    emb = 2 * vocab * h

    # Attention per layer (rough)
    attn_per_layer = (
        h * q_lora
        + q_lora * n_heads * (qk_nope + qk_rope)
        + h * (kv_lora + qk_rope)
        + kv_lora * n_heads * (qk_nope + v_dim)
        + n_heads * v_dim * h
    )

    # Dense MLP per layer
    dense_mlp = 3 * h * inter

    # MoE per layer
    moe_per_layer = (
        n_experts * 3 * h * moe_inter
        + h * n_experts  # router
        + 3 * h * (moe_inter * n_shared)  # shared experts
    )

    total_params = (
        emb
        + first_k * (attn_per_layer + dense_mlp)
        + (n_layers - first_k) * (attn_per_layer + moe_per_layer)
        + h  # final norm
    )

    return total_params * bytes_per_param / (1024 * 1024)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def copy_modeling_files(output_dir: Path, model_type: str = "gigachat35") -> None:
    """Copy custom modeling code to the checkpoint directory.

    These files are required for ``from_pretrained(..., trust_remote_code=True)``
    to work with the model_type.
    """
    if model_type == "gigachat35":
        files_to_copy = [
            "configuration_gigachat3_5.py",
            "modeling_gigachat3_5.py",
            "__init__.py",
        ]
        models_dir = GIGACHAT_MODELS_DIR
    elif model_type == "deepseek_v3":
        files_to_copy = [
            "configuration_deepseek.py",
            "modeling_deepseek.py",
        ]
        models_dir = DEEPSEEK_MODELS_DIR
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    for fname in files_to_copy:
        src = models_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"  Copied {fname}")
        else:
            print(f"  WARNING: {fname} not found at {src}")
            raise FileNotFoundError(
                f"Required modeling file not found: {src}. "
                f"Cannot create {model_type} checkpoint without it."
            )


# ---------------------------------------------------------------------------
# GPU layer-by-layer initialization helpers
# ---------------------------------------------------------------------------


def _gpu_init_module(module, initializer_range: float = 0.02):
    """Initialize all parameters in a module after ``to_empty()`` on GPU.

    More comprehensive than the default HF ``_init_weights`` which only
    handles ``nn.Linear`` and ``nn.Embedding``.  Also handles norm layers
    (weight → 1.0) and any remaining parameters (normal init).
    """
    import torch.nn as nn

    initialized = set()  # ids of already-initialized parameters

    # 1) Linear layers: weight ~ N(0, std), bias = 0
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            sub.weight.data.normal_(mean=0.0, std=initializer_range)
            initialized.add(id(sub.weight))
            if sub.bias is not None:
                sub.bias.data.zero_()
                initialized.add(id(sub.bias))

    # 2) Embedding layers: weight ~ N(0, std), padding_idx row = 0
    for sub in module.modules():
        if isinstance(sub, nn.Embedding):
            sub.weight.data.normal_(mean=0.0, std=initializer_range)
            initialized.add(id(sub.weight))
            if sub.padding_idx is not None:
                sub.weight.data[sub.padding_idx].zero_()

    # 3) Norm layers: weight = 1, bias = 0
    for sub in module.modules():
        cls_name = type(sub).__name__.lower()
        if "norm" in cls_name and not isinstance(sub, (nn.Linear, nn.Embedding)):
            if (
                hasattr(sub, "weight")
                and sub.weight is not None
                and id(sub.weight) not in initialized
            ):
                sub.weight.data.fill_(1.0)
                initialized.add(id(sub.weight))
            if (
                hasattr(sub, "bias")
                and sub.bias is not None
                and id(sub.bias) not in initialized
            ):
                sub.bias.data.zero_()
                initialized.add(id(sub.bias))

    # 4) Catch-all: any remaining parameters get normal init
    for param in module.parameters():
        if id(param) not in initialized:
            param.data.normal_(mean=0.0, std=initializer_range)


def _save_model_gpu_layer_by_layer(
    model,
    output_path: Path,
    initializer_range: float = 0.02,
    gpu_device: str = "cuda:0",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Save a meta-device model layer-by-layer via GPU materialization.

    For each component (embed, decoder layer, norm, lm_head):

      1. Materialize on GPU with ``to_empty(gpu_device)``
      2. Initialize random weights via ``_gpu_init_module``
      3. Save state dict to a safetensors shard
      4. Move back to meta device and free GPU memory

    Only one layer's worth of GPU memory is needed at a time.
    """
    from safetensors.torch import save_file

    # Collect (prefix, module) pairs to process
    components = []

    base_model = getattr(model, "model", None)
    if base_model is not None:
        # Embed tokens
        if hasattr(base_model, "embed_tokens"):
            components.append(("model.embed_tokens", base_model.embed_tokens))

        # Decoder layers
        if hasattr(base_model, "layers"):
            for i, layer in enumerate(base_model.layers):
                components.append((f"model.layers.{i}", layer))

        # MTP / nextn predict layers
        if hasattr(base_model, "nextn_predict_layers"):
            for i, layer in enumerate(base_model.nextn_predict_layers):
                components.append((f"model.nextn_predict_layers.{i}", layer))

        # Final norm
        if hasattr(base_model, "norm"):
            components.append(("model.norm", base_model.norm))

    # LM head
    if hasattr(model, "lm_head"):
        components.append(("lm_head", model.lm_head))

    if not components:
        raise RuntimeError(
            "No model components found. Expected model.model.{embed_tokens, layers, norm} "
            "and model.lm_head. Check that the model was created correctly on meta device."
        )

    n_shards = len(components)
    n_digits = max(5, len(str(n_shards)))
    shard_fmt = f"model-{{:0{n_digits}d}}-of-{{:0{n_digits}d}}.safetensors"

    weight_map = {}
    total_size = 0

    for idx, (prefix, module) in enumerate(components, start=1):
        shard_name = shard_fmt.format(idx, n_shards)
        print(f"  [{idx}/{n_shards}] {prefix} → {shard_name}")

        # 1) Materialize on GPU (creates uninitialized tensors on gpu_device)
        module.to_empty(device=gpu_device)
        module.to(dtype=torch_dtype)

        # 2) Initialize weights with random values
        _gpu_init_module(module, initializer_range)

        # 3) Collect state dict with full parameter names (clone to CPU)
        state = {}
        for k, v in module.state_dict().items():
            full_key = f"{prefix}.{k}"
            state[full_key] = v.cpu().clone()

        # 4) Save shard
        save_file(state, str(output_path / shard_name))

        # 5) Track in weight map and accumulate total size
        for full_key, tensor in state.items():
            weight_map[full_key] = shard_name
            total_size += tensor.nelement() * tensor.element_size()

        # 6) Free GPU memory — move module back to meta
        module.to_empty(device="meta")
        del state
        torch.cuda.empty_cache()

    # Write index JSON (model.safetensors.index.json)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = output_path / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Index written to {index_path}")
    print(f"  Total: {n_shards} shards, {total_size / (1024**3):.2f} GB")


@contextmanager
def _init_empty_weights_fallback():
    """Fallback for ``accelerate.init_empty_weights`` when accelerate is not installed.

    Uses ``torch.device("meta")`` context manager which makes all tensor
    creation happen on the meta device (no memory allocated).
    """
    with torch.device("meta"):
        yield


@contextmanager
def _temporary_default_dtype(dtype: torch.dtype):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


def _resolve_checkpoint_dtype(torch_dtype: str) -> torch.dtype:
    normalized = torch_dtype.lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(
        f"Unsupported checkpoint dtype {torch_dtype!r}; "
        "random HF checkpoints are created in bfloat16."
    )


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def init_and_save_model(
    config: str,
    output_dir: str,
    tokenizer_path: str = None,
    torch_dtype: str = "bfloat16",
    seed: int = 42,
    include_mtp: bool = False,
    use_custom_sglang: bool = True,
    model_type: str = "gigachat35",
    model_tag: str = None,
    device: str = "cpu",
    uniform_moe=False,
    expand_context_len_to=None,
) -> None:
    """Initialize a random-weight model and save as HF checkpoint.

    Args:
        config: Path to llm_foundry_config.yaml.
        output_dir: Output directory for HF checkpoint.
        tokenizer_path: Path to tokenizer (optional; will try config if not provided).
        torch_dtype: Torch dtype for model weights (default: bfloat16).
        seed: Random seed for weight initialization (default: 42).
        include_mtp: Include MTP (multi-token prediction) layers in the checkpoint.
        use_custom_sglang: Use the custom sglang implementation from this repo.
            If False, use open-source sglang from the current environment.
        model_type: Custom sglang config type: "gigachat35" or "deepseek_v3".
            Ignored when use_custom_sglang is False.
        model_tag: Model tag for the open-source sglang config. Required when
            use_custom_sglang is False and ignored with a warning otherwise.
        device: Device for initialization - "cpu" (default) or "cuda".
            When "cuda" (or "cuda:N"), uses layer-by-layer GPU initialization
            via meta device: the model is created on meta device (no memory),
            then each layer is materialized on GPU one at a time, initialized,
            saved to a safetensors shard, and freed.  Only one layer's worth
            of GPU memory is needed at a time.
    """

    if not isinstance(use_custom_sglang, bool):
        raise TypeError("--use_custom_sglang must be True or False")
    if use_custom_sglang and model_tag is not None:
        print(
            "WARNING: --model_tag is used only when --use_custom_sglang False; ignoring."
        )

    use_gpu = device.startswith("cuda")

    if use_gpu and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is not available.")
        print("Falling back to CPU initialization.")
        use_gpu = False
        device = "cpu"

    print("=" * 60)
    print("Initializing random-weight HF checkpoint")
    print(f"Use custom sglang: {use_custom_sglang}")
    if use_custom_sglang:
        print(f"Model type: {model_type}")
    else:
        print(f"Model tag: {model_tag}")
    print(f"Device: {device}")
    if use_gpu:
        print("Mode: layer-by-layer GPU (meta device → materialize → save → free)")
    print("from llm-foundry config")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and map config
    # ------------------------------------------------------------------
    print("\n[1/6] Loading llm-foundry config...")
    llm_config = load_llm_foundry_config(config)
    hf_config_dict = build_hf_config_dict(
        llm_config,
        model_type=model_type,
        include_mtp=include_mtp,
        use_custom_sglang=use_custom_sglang,
        model_tag=model_tag,
    )
    dtype = _resolve_checkpoint_dtype(torch_dtype)
    hf_config_dict["torch_dtype"] = "bfloat16"

    est_size = estimate_model_size_mb(hf_config_dict)
    print(f"  Run: {llm_config.get('run_name', 'unknown')}")
    print(f"  Architectures: {hf_config_dict.get('architectures')}")
    print(f"  HF model_type: {hf_config_dict.get('model_type')}")
    if use_custom_sglang:
        print(f"  Model type: {model_type}")
    print(f"  hidden_size={hf_config_dict.get('hidden_size')}")
    print(f"  num_hidden_layers={hf_config_dict.get('num_hidden_layers')}")
    print(f"  n_routed_experts={hf_config_dict.get('n_routed_experts')}")
    print(f"  num_experts_per_tok={hf_config_dict.get('num_experts_per_tok')}")
    print(f"  moe_intermediate_size={hf_config_dict.get('moe_intermediate_size')}")
    print(f"  n_shared_experts={hf_config_dict.get('n_shared_experts')}")
    print(f"  Estimated model size: ~{est_size:.0f} MB ({est_size / 1024:.1f} GB)")

    # ------------------------------------------------------------------
    # 2. Create output directory
    # ------------------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Copy modeling files
    # ------------------------------------------------------------------
    if use_custom_sglang:
        print("\n[2/6] Copying custom modeling code...")
        copy_modeling_files(output_path, model_type=model_type)
    else:
        print("\n[2/6] Skipping custom modeling code for open-source sglang.")

    # ------------------------------------------------------------------
    # 4. Write config.json with auto_map
    # ------------------------------------------------------------------
    print("\n[3/6] Writing config.json...")
    if use_custom_sglang:
        if model_type == "gigachat35":
            hf_config_dict["auto_map"] = {
                "AutoConfig": "configuration_gigachat3_5.Gigachat35Config",
                "AutoModelForCausalLM": "modeling_gigachat3_5.Gigachat35ForCausalLM",
            }
        elif model_type == "deepseek_v3":
            hf_config_dict["auto_map"] = {
                "AutoConfig": "configuration_deepseek.DeepseekV3Config",
                "AutoModel": "modeling_deepseek.DeepseekV3Model",
                "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
            }

    if uniform_moe:
        if use_custom_sglang:
            hf_config_dict["architectures"] = [
                hf_config_dict["architectures"][0] + "_UniformMoE"
            ]
            print(
                f"[WARNING] using {hf_config_dict['architectures']} architecture, make sure it exists in sglang"
            )
        else:
            print(
                "WARNING: --uniform_moe is used only when --use_custom_sglang True; ignoring."
            )

    if expand_context_len_to is not None:
        context_target = _find_hf_config_target_with_key(
            hf_config_dict,
            "max_position_embeddings",
        )
        previous_pos_emb = context_target["max_position_embeddings"]
        factor = math.ceil(expand_context_len_to / previous_pos_emb)
        context_target["rope_scaling"] = {
            "type": "dynamic",
            "factor": factor,
            "original_max_position_embeddings": previous_pos_emb,
            "rope_type": "dynamic",
        }
        context_target["max_position_embeddings"] = previous_pos_emb * factor

    config_json_path = output_path / "config.json"
    with open(config_json_path, "w") as f:
        json.dump(hf_config_dict, f, indent=2)
    print(f"  Written to {config_json_path}")

    # ------------------------------------------------------------------
    # 5. Instantiate model
    # ------------------------------------------------------------------
    from transformers import AutoConfig, AutoModelForCausalLM

    if not use_custom_sglang:
        _import_open_source_sglang_config_registry()

    hf_config = AutoConfig.from_pretrained(
        str(output_path),
        trust_remote_code=True,
    )
    print(f"  Config class: {type(hf_config).__name__}")

    if use_gpu:
        # --- GPU path: create on meta device, save layer-by-layer ---
        print("\n[4/6] Instantiating model on meta device (GPU layer-by-layer)...")

        # Try accelerate first, fall back to torch.device("meta")
        try:
            from accelerate import init_empty_weights

            print("  Using accelerate.init_empty_weights()")
        except ImportError:
            init_empty_weights = _init_empty_weights_fallback
            print("  accelerate not installed, using torch.device('meta') fallback")

        with _temporary_default_dtype(dtype), init_empty_weights():
            if not use_custom_sglang:
                model = AutoModelForCausalLM.from_config(
                    hf_config,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
            elif model_type == "gigachat35":
                models_parent = str(GIGACHAT_MODELS_DIR.parent)
                if models_parent not in sys.path:
                    sys.path.insert(0, models_parent)
                from gigachat3_5.configuration_gigachat3_5 import Gigachat35Config
                from gigachat3_5.modeling_gigachat3_5 import Gigachat35ForCausalLM

                cfg = Gigachat35Config(**hf_config_dict)
                model = Gigachat35ForCausalLM(cfg)
            elif model_type == "deepseek_v3":
                models_parent = str(DEEPSEEK_MODELS_DIR.parent)
                if models_parent not in sys.path:
                    sys.path.insert(0, models_parent)
                from DeepSeekV3.configuration_deepseek import DeepseekV3Config
                from DeepSeekV3.modeling_deepseek import DeepseekV3ForCausalLM

                cfg = DeepseekV3Config(**hf_config_dict)
                model = DeepseekV3ForCausalLM(cfg)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model class: {type(model).__name__}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Dtype: {dtype}")
        print(f"  Device: meta (will materialize layer-by-layer on {device})")

    else:
        # --- CPU path: existing behavior ---
        print("\n[4/6] Instantiating model on CPU...")
        torch.manual_seed(seed)

        try:
            with _temporary_default_dtype(dtype), torch.device("cpu"):
                model = AutoModelForCausalLM.from_config(
                    hf_config,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
                model = model.to(dtype)
        except Exception as e:
            if not use_custom_sglang:
                raise

            print(f"  from_config with trust_remote_code failed: {e}")
            print("  Falling back to direct import...")

            if model_type == "gigachat35":
                # Fallback: import the custom classes directly via sys.path
                models_parent = str(GIGACHAT_MODELS_DIR.parent)
                if models_parent not in sys.path:
                    sys.path.insert(0, models_parent)

                from gigachat3_5.configuration_gigachat3_5 import Gigachat35Config
                from gigachat3_5.modeling_gigachat3_5 import Gigachat35ForCausalLM

                with _temporary_default_dtype(dtype):
                    cfg = Gigachat35Config(**hf_config_dict)
                    model = Gigachat35ForCausalLM(cfg)
                    model = model.to(dtype)

            elif model_type == "deepseek_v3":
                # Fallback: import the custom classes directly via sys.path
                models_parent = str(DEEPSEEK_MODELS_DIR.parent)
                if models_parent not in sys.path:
                    sys.path.insert(0, models_parent)

                from DeepSeekV3.configuration_deepseek import DeepseekV3Config
                from DeepSeekV3.modeling_deepseek import DeepseekV3ForCausalLM

                with _temporary_default_dtype(dtype):
                    cfg = DeepseekV3Config(**hf_config_dict)
                    model = DeepseekV3ForCausalLM(cfg)
                    model = model.to(dtype)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model class: {type(model).__name__}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Dtype: {dtype}")

    # ------------------------------------------------------------------
    # 6. Save model weights
    # ------------------------------------------------------------------
    if use_gpu:
        print("\n[5/6] Saving model weights (GPU layer-by-layer)...")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        initializer_range = getattr(hf_config, "initializer_range", 0.02)
        _save_model_gpu_layer_by_layer(
            model, output_path, initializer_range, gpu_device=device, torch_dtype=dtype
        )
        print(f"  Saved to {output_path}")
        del model
    else:
        print("\n[5/6] Saving model weights...")
        model.save_pretrained(str(output_path), safe_serialization=True)
        print(f"  Saved to {output_path}")
        del model

    # ------------------------------------------------------------------
    # 7. Handle tokenizer
    # ------------------------------------------------------------------
    print("\n[6/6] Handling tokenizer...")
    tok_path = tokenizer_path
    if not tok_path:
        # Try to get tokenizer path from the llm-foundry config
        tok_path = (
            llm_config.get("tokenizer", {}).get("name")
            or llm_config.get("tokenizer_path")
            or llm_config.get("model", {}).get("pretrained_model_name_or_path")
        )
        if not tok_path and not use_custom_sglang:
            tok_path = model_tag

    if tok_path and (os.path.exists(tok_path) or not use_custom_sglang):
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            tokenizer.save_pretrained(str(output_path))
            print(f"  Tokenizer saved from {tok_path}")
        except Exception as e:
            print(f"  WARNING: Failed to save tokenizer: {e}")
    else:
        print("  No tokenizer path provided or found in config.")
        print("  For sglang/vllm, specify --tokenizer_path separately at runtime.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Done! HF checkpoint saved to: {output_path}")
    print("=" * 60)
    print("\nUsage with transformers:")
    print("  from transformers import AutoModelForCausalLM")
    print(
        f"  model = AutoModelForCausalLM.from_pretrained('{output_path}', trust_remote_code=True)"
    )
    print("\nUsage with sglang:")
    if model_type == "gigachat35":
        print(
            f"  python -m sglang.launch_server --model-path {output_path} --trust-remote-code"
        )
    else:
        print(
            f"  python -m sglang.launch_server --model-path {output_path} --trust-remote-code"
        )
    print("\nUsage with vllm:")
    if model_type == "gigachat35":
        print(
            f"  python -m vllm.entrypoints.openai.api_server --model {output_path} --trust-remote-code"
        )
    else:
        print(
            f"  python -m vllm.entrypoints.openai.api_server --model {output_path} --trust-remote-code"
        )


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value!r}")


def _optional_int(value):
    if value is None or value == "" or value.lower() == "none":
        return None
    return int(value)


def _argparse_main():
    parser = argparse.ArgumentParser(
        description="Initialize a random-weight HF checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_mtp", type=_str_to_bool, default=False)
    parser.add_argument("--use_custom_sglang", type=_str_to_bool, default=True)
    parser.add_argument("--model_type", default="gigachat35")
    parser.add_argument("--model_tag", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--uniform_moe", type=_str_to_bool, default=False)
    parser.add_argument("--expand_context_len_to", type=_optional_int, default=None)
    args = parser.parse_args()
    init_and_save_model(**vars(args))


if __name__ == "__main__":
    _argparse_main()
