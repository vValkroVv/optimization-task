"""The ``config.json`` for a GIFT model exported to HuggingFace format.

Two serving formats exist for the family, and the choice mirrors what PAM's
converter (``pretrain_auto_metrics/contrib/gigachat_utils/convertation``,
``is_gigachat_model``) does for llm-foundry checkpoints:

- ``deepseek_v3`` — a plain MLA + MoE stack with plain RMSNorm. Served by
  stock ``DeepseekV3ForCausalLM`` (opensource sglang).
- ``gigachat3_5`` — anything the DeepSeek-V3 config cannot express: GDN
  linear-attention layers, gated attention, gated / sandwich norms, the MLA
  latent scaling factor, logit soft-capping. Served by ``Gigachat35ForCausalLM``
  from the custom sglang tree; needs ``auto_map`` plus the vendored
  ``configuration_gigachat3_5.py`` / ``modeling_gigachat3_5.py`` next to the
  weights, which the checkpointer copies (``checkpoint.hf_remote_code_dir``).

Field semantics were taken from ``configuration_gigachat3_5.py`` /
``modeling_gigachat3_5.py`` and sglang's ``models/gigachat35.py``; anything the
model config cannot express unambiguously raises instead of guessing — an
exported model that loads but computes something else is the failure mode
this module exists to prevent.
"""

from __future__ import annotations

from typing import Any

from torchtitan.config.schema.components.lora import FrozenConfig

from torchtitan.config.schema.models.common.embeddings.rope.scaling import YarnScaling
from torchtitan.config.schema.models.common.attention.linear.gated_delta import (
    GatedDeltaAttentionConfig,
)
from torchtitan.config.schema.models.common.attention.linear.gdn import (
    GatedDeltaNetConfig,
)
from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig
from torchtitan.config.schema.models.common.attention.linear.const import (
    GatedNormActivation,
)
from torchtitan.config.schema.models.common.norms.gated_norm import GatedRMSNormConfig
from torchtitan.config.schema.models.common.norms.rmsnorm import RMSNormConfig

__all__ = [
    "build_hf_config",
    "GIGACHAT_MODEL_TYPE",
]

GIGACHAT_MODEL_TYPE = "gigachat3_5"
DEEPSEEK_MODEL_TYPE = "deepseek_v3"

# ``auto_map`` is what makes an HF directory loadable without the class being
# in ``transformers``: it names the modules that must sit next to the weights,
# which is where the checkpointer gets the file list from.
GIGACHAT_AUTO_MAP = {
    "AutoConfig": "configuration_gigachat3_5.Gigachat35Config",
    "AutoModel": "modeling_gigachat3_5.Gigachat35Model",
    "AutoModelForCausalLM": "modeling_gigachat3_5.Gigachat35ForCausalLM",
}

# HF hardcodes the gated norm's low rank (``gate_up_projection: hidden x 16``).
_GATED_NORM_RANK = 16
_LINEAR_ATTENTION_TYPE = "Qwen3NextGatedDeltaNet"


def _same(values: list[Any], what: str) -> Any:
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise ValueError(
                f"hf_config: {what} differs between layers ({first!r} vs {value!r}); "
                "the HF config describes one value for the whole model."
            )
    return first


def _unwrapped(config: Any) -> Any:
    """Freezing preserves the inner module's computation and HF contract."""
    while isinstance(config, FrozenConfig):
        config = config.inner
    return config


def _norm_kind(norm: Any, where: str) -> tuple[str, float | None]:
    """HF ``norm_type`` and ``layernorm_gating_weight`` for one norm config."""
    norm = _unwrapped(norm)
    if isinstance(norm, RMSNormConfig):
        return "LlamaRMSNorm", None
    if isinstance(norm, GatedRMSNormConfig):
        if norm.gate_rank != _GATED_NORM_RANK:
            raise ValueError(
                f"hf_config: {where} uses GatedRMSNorm with gate_rank={norm.gate_rank}, "
                f"but the HF Gigachat35 gated norm hardcodes rank {_GATED_NORM_RANK}."
            )
        kind = "ZeroCenteredGatedNorm" if norm.zero_centered else "GatedNorm"
        return kind, float(norm.gating_weight)
    raise ValueError(
        f"hf_config: {where} is {type(norm).__qualname__}, which has no HF "
        "norm_type (LlamaRMSNorm / GatedNorm / ZeroCenteredGatedNorm)."
    )


def _mla_fields(attention: MLAConfig) -> dict[str, Any]:
    return {
        "num_attention_heads": attention.n_heads,
        "num_key_value_heads": attention.n_heads,
        "q_lora_rank": attention.q_lora_rank or None,
        "kv_lora_rank": attention.kv_lora_rank,
        "qk_nope_head_dim": attention.qk_nope_head_dim,
        "qk_rope_head_dim": attention.qk_rope_head_dim,
        "v_head_dim": attention.v_head_dim,
        # Derived the way Gigachat35Config.__init__ derives them, so a config
        # written here and one round-tripped through the class agree.
        "qk_head_dim": attention.qk_nope_head_dim + attention.qk_rope_head_dim,
        "head_dim": attention.qk_rope_head_dim,
        "mscale": attention.mscale,
        "gated_attention": attention.gate is not None,
        "use_mla_scaling_factor": bool(attention.use_mla_scaling_factor),
    }


def _linear_gating_type(gdn: GatedDeltaAttentionConfig) -> str:
    activation = gdn.o_norm.activation
    if activation == GatedNormActivation.SIGMOID:
        return (
            "gated_rmsnorm_sigmoid_zero_centered"
            if gdn.o_norm.zero_centered
            else "gated_rmsnorm_sigmoid"
        )
    if gdn.o_norm.zero_centered:
        raise ValueError(
            "hf_config: GDN o_norm is zero-centered with a SiLU/Swish gate; HF "
            "Gigachat35 only offers zero-centering for the sigmoid gate "
            "(linear_gating_type=gated_rmsnorm_sigmoid_zero_centered)."
        )
    if gdn.o_norm_output_multiplier != 1.0:
        raise ValueError(
            "hf_config: GDN o_norm_output_multiplier != 1 with a SiLU/Swish gate; "
            "HF applies linear_sigmoid_gate_scale only to the sigmoid gating types."
        )
    return "gated_rmsnorm"


def _gdn_fields(gdn_configs: list[GatedDeltaAttentionConfig]) -> dict[str, Any]:
    for gdn in gdn_configs:
        if not isinstance(gdn, GatedDeltaNetConfig):
            raise ValueError(
                f"hf_config: linear attention {type(gdn).__qualname__} has no HF "
                f"export yet; only GatedDeltaNet ({_LINEAR_ATTENTION_TYPE}) does."
            )
    fields = {
        "linear_attention_type": _LINEAR_ATTENTION_TYPE,
        "linear_num_key_heads": _same(
            [g.num_k_heads for g in gdn_configs], "GDN num_k_heads"
        ),
        "linear_num_value_heads": _same(
            [g.num_v_heads for g in gdn_configs], "GDN num_v_heads"
        ),
        "linear_key_head_dim": _same(
            [g.head_k_dim for g in gdn_configs], "GDN head_k_dim"
        ),
        "linear_value_head_dim": _same(
            [g.head_v_dim for g in gdn_configs], "GDN head_v_dim"
        ),
        "linear_conv_kernel_dim": _same(
            [g.conv_size for g in gdn_configs], "GDN conv_size"
        ),
        "linear_gating_type": _same(
            [_linear_gating_type(g) for g in gdn_configs], "GDN gating type"
        ),
        # HF: ``output = norm(o, z) * linear_sigmoid_gate_scale`` — the same
        # place gift's ``o = o_norm(o, z) * o_norm_output_multiplier`` scales.
        "linear_sigmoid_gate_scale": _same(
            [float(g.o_norm_output_multiplier) for g in gdn_configs],
            "GDN o_norm_output_multiplier",
        ),
        "linear_attn_o_norm_eps": _same(
            [g.o_norm.eps for g in gdn_configs], "GDN o_norm.eps"
        ),
        # The only in_proj_qkvz layout HF implements is the fused one we write.
        "linear_use_legacy_qkvz_layout": False,
        "linear_use_short_conv": True,
    }
    return fields


def _rope_scaling(model_config: Any, attention: MLAConfig) -> dict[str, Any] | None:
    scaling = model_config.rope.scaling
    # ``scaling`` is a dataclass and present-means-active (YarnScaling rejects
    # rope_factor <= 1), so the type check alone decides.
    if not isinstance(scaling, YarnScaling):
        return None
    return {
        "type": "yarn",
        "factor": scaling.rope_factor,
        "beta_fast": scaling.beta_fast,
        "beta_slow": scaling.beta_slow,
        "original_max_position_embeddings": scaling.original_seq_len,
        "mscale": attention.mscale,
        "mscale_all_dim": attention.mscale,
    }


def build_hf_config(model_config: Any) -> dict[str, Any]:
    """The ``config.json`` describing what the adapter's ``to_hf`` writes."""
    layers = list(model_config.layers)
    if not layers:
        raise ValueError("hf_config: model has no layers")

    mla = [
        _unwrapped(layer.attention)
        for layer in layers
        if isinstance(_unwrapped(layer.attention), MLAConfig)
    ]
    gdn = [
        _unwrapped(layer.attention)
        for layer in layers
        if isinstance(_unwrapped(layer.attention), GatedDeltaAttentionConfig)
    ]
    stray = [
        type(_unwrapped(layer.attention)).__qualname__
        for layer in layers
        if not isinstance(
            _unwrapped(layer.attention), (MLAConfig, GatedDeltaAttentionConfig)
        )
    ]
    if stray:
        raise ValueError(
            f"hf_config: attention types {sorted(set(stray))} have no HF export "
            "(MLA and GatedDeltaNet do)."
        )
    if not mla:
        raise ValueError(
            "hf_config: model has no MLA layer; both HF formats describe MLA "
            "attention (the GDN layers ride alongside it)."
        )
    mtp_heads = model_config.mtp.layers if model_config.mtp is not None else []
    mtp_blocks = [_unwrapped(head.block) for head in mtp_heads]
    for index, block in enumerate(mtp_blocks):
        if not isinstance(_unwrapped(block.attention), MLAConfig):
            raise ValueError(
                f"hf_config: mtp.layers.{index}.block.attention must be MLA; "
                "HF MTP heads have no per-head attention-type selector."
            )
    export_blocks = layers + mtp_blocks

    attention_fields = _same(
        [_mla_fields(a) for a in mla]
        + [_mla_fields(_unwrapped(block.attention)) for block in mtp_blocks],
        "MLA attention config",
    )
    mla_reference = mla[0]

    moe_layers = [layer for layer in layers if layer.moe is not None]
    dense_layers = [layer for layer in layers if layer.moe is None]
    if not moe_layers:
        raise ValueError(
            "hf_config: model has no MoE layer; both HF formats are MoE configs"
        )
    if not dense_layers:
        raise ValueError(
            "hf_config: model has no dense layer, so intermediate_size (the dense "
            "FFN width, kept apart from moe_intermediate_size) cannot be determined"
        )
    dense_indices = [i for i, layer in enumerate(layers) if layer.moe is None]
    if dense_indices != list(range(len(dense_indices))):
        raise ValueError(
            f"hf_config: dense layers sit at {dense_indices}; HF only expresses a "
            "dense prefix (first_k_dense_replace)."
        )
    moe = moe_layers[0].moe
    router, experts, shared = moe.router, moe.experts, moe.shared_experts
    dense_ffn_dim = _same(
        [
            layer.feed_forward.w1.out_features
            for layer in export_blocks
            if layer.moe is None
        ],
        "dense FFN width",
    )
    if shared is not None and shared.hidden_dim % experts.hidden_dim:
        raise ValueError(
            "hf_config: shared expert hidden_dim must be an integer multiple of "
            f"routed expert hidden_dim ({experts.hidden_dim}), got {shared.hidden_dim}; "
            "HF represents the shared width as n_shared_experts."
        )
    num_shared = shared.hidden_dim // experts.hidden_dim if shared is not None else 0
    # These HF fields are shared by trunk MoEs and sparse MTP heads alike.
    _same(
        [
            (
                layer.moe.experts.hidden_dim,
                layer.moe.experts.num_experts,
                layer.moe.shared_experts.hidden_dim if layer.moe.shared_experts else 0,
                layer.moe.router.top_k,
                layer.moe.router.route_norm,
                layer.moe.router.route_scale,
                layer.moe.router.score_func,
                layer.moe.router.num_expert_groups or 1,
                layer.moe.router.num_limited_groups or 1,
                layer.moe.router_aux_loss_coef,
            )
            for layer in export_blocks
            if layer.moe is not None
        ],
        "MoE expert/router config",
    )
    # One HF field for the whole model: every SwiGLU must clip alike.
    swiglu_limit = _same(
        [
            layer.feed_forward.swiglu_limit
            for layer in export_blocks
            if layer.moe is None
        ]
        + [
            layer.moe.experts.swiglu_limit
            for layer in export_blocks
            if layer.moe is not None
        ]
        + [
            layer.moe.shared_experts.swiglu_limit
            for layer in export_blocks
            if layer.moe is not None and layer.moe.shared_experts is not None
        ],
        "SwiGLU swiglu_limit",
    )

    # Norms: one class for the whole model (HF picks it with one field).
    norm_sites: list[tuple[str, Any]] = [("model.norm", model_config.norm)]
    layernorm_types: list[str] = []
    for index, layer in enumerate(layers):
        norm_sites.append((f"layers.{index}.attention_norm", layer.attention_norm))
        norm_sites.append((f"layers.{index}.ffn_norm", layer.ffn_norm))
        post_attn = getattr(layer, "post_attn_norm", None)
        post_ffn = getattr(layer, "post_ffn_norm", None)
        if (post_attn is None) != (post_ffn is None):
            raise ValueError(
                f"hf_config: layers.{index} has only one of post_attn_norm / "
                "post_ffn_norm; HF layernorm_type=pre_post needs both."
            )
        layernorm_types.append("pre_post" if post_attn is not None else "pre")
        if post_attn is not None:
            norm_sites.append((f"layers.{index}.post_attn_norm", post_attn))
            norm_sites.append((f"layers.{index}.post_ffn_norm", post_ffn))
    # MTP heads export into the same norm fields as the stack.

    for index, head in enumerate(mtp_heads):
        where = f"mtp.layers.{index}"
        for name in ("token_norm", "hidden_norm", "final_norm"):
            norm_sites.append((f"{where}.{name}", getattr(head, name)))
        block = _unwrapped(head.block)
        post_attn = getattr(block, "post_attn_norm", None)
        post_ffn = getattr(block, "post_ffn_norm", None)
        if (post_attn is None) != (post_ffn is None):
            raise ValueError(
                f"hf_config: {where}.block has only one of post_attn_norm / "
                "post_ffn_norm; HF layernorm_type=pre_post needs both."
            )
        layernorm_types.append("pre_post" if post_attn is not None else "pre")

        for name in ("attention_norm", "ffn_norm", "post_attn_norm", "post_ffn_norm"):
            norm = getattr(block, name, None)
            if norm is not None:
                norm_sites.append((f"{where}.block.{name}", norm))
        attention = _unwrapped(block.attention)
        if isinstance(attention, MLAConfig):
            norm_sites.append((f"{where}.block.attention.kv_norm", attention.kv_norm))
            if attention.q_norm is not None:
                norm_sites.append((f"{where}.block.attention.q_norm", attention.q_norm))
    for index, attention in enumerate(mla):
        norm_sites.append((f"mla[{index}].kv_norm", attention.kv_norm))
        if attention.q_norm is not None:
            norm_sites.append((f"mla[{index}].q_norm", attention.q_norm))
    norm_kinds = [_norm_kind(norm, where) for where, norm in norm_sites]
    norm_type = _same([kind for kind, _ in norm_kinds], "norm class")
    gating_weight = _same([w for _, w in norm_kinds], "gated-norm gating_weight")
    layernorm_type = _same(layernorm_types, "sandwich norms")
    rms_norm_eps = _same([norm.eps for _, norm in norm_sites], "norm eps")

    logit_softcap = float(getattr(model_config.lm_head, "logit_softcap", 0.0) or 0.0)

    mtp = model_config.mtp
    num_mtp = len(mtp.layers) if mtp is not None else 0
    nextn_is_sparse = None
    if num_mtp:
        nextn_is_sparse = _same(
            [head.block.moe is not None for head in mtp.layers], "MTP head FFN kind"
        )

    full_attention_layers = [
        i
        for i, layer in enumerate(layers)
        if isinstance(_unwrapped(layer.attention), MLAConfig)
    ]
    is_gigachat = (
        bool(gdn)
        or attention_fields["gated_attention"]
        or attention_fields["use_mla_scaling_factor"]
        or norm_type != "LlamaRMSNorm"
        or layernorm_type != "pre"
        or logit_softcap > 0.0
    )

    if not is_gigachat and nextn_is_sparse is False:
        raise ValueError(
            "hf_config: dense MTP heads have no representation in the "
            "deepseek_v3 format; it has no nextn_is_sparse selector."
        )

    config: dict[str, Any] = {
        "model_type": GIGACHAT_MODEL_TYPE if is_gigachat else DEEPSEEK_MODEL_TYPE,
        "architectures": [
            "Gigachat35ForCausalLM" if is_gigachat else "DeepseekV3ForCausalLM"
        ],
        # The checkpointer overwrites this with its export_dtype; it matters
        # only for a config written by hand.
        "torch_dtype": "bfloat16",
        # Shape.
        "vocab_size": model_config.vocab_size,
        "hidden_size": model_config.dim,
        "num_hidden_layers": len(layers),
        "hidden_act": "silu",
        "max_position_embeddings": model_config.rope.max_seq_len,
        "rms_norm_eps": rms_norm_eps,
        "tie_word_embeddings": False,
        "attention_bias": False,
        # MLA.
        "num_attention_heads": attention_fields["num_attention_heads"],
        "num_key_value_heads": attention_fields["num_key_value_heads"],
        "q_lora_rank": attention_fields["q_lora_rank"],
        "kv_lora_rank": attention_fields["kv_lora_rank"],
        "qk_nope_head_dim": attention_fields["qk_nope_head_dim"],
        "qk_rope_head_dim": attention_fields["qk_rope_head_dim"],
        "v_head_dim": attention_fields["v_head_dim"],
        "qk_head_dim": attention_fields["qk_head_dim"],
        "head_dim": attention_fields["head_dim"],
        # MoE.
        "intermediate_size": dense_ffn_dim,
        "moe_intermediate_size": experts.hidden_dim,
        "n_routed_experts": experts.num_experts,
        "n_shared_experts": num_shared,
        "num_experts_per_tok": router.top_k,
        "first_k_dense_replace": len(dense_layers),
        "moe_layer_freq": 1,
        "norm_topk_prob": router.route_norm,
        "routed_scaling_factor": router.route_scale,
        "scoring_func": router.score_func.value,
        "n_group": router.num_expert_groups or 1,
        "topk_group": router.num_limited_groups or 1,
        "topk_method": "noaux_tc",
        "aux_loss_alpha": moe.router_aux_loss_coef,
        # RoPE: gift's MLA rotates complex pairs, which is HF's interleaved
        # layout (``rope_interleave``; sglang: ``is_neox_style = not it``).
        "rope_theta": model_config.rope.theta,
        "rope_scaling": _rope_scaling(model_config, mla_reference),
        "rope_interleave": True,
        # MTP heads live at layer indices [num_hidden_layers, +num_nextn).
        "num_nextn_predict_layers": num_mtp,
    }
    if not is_gigachat:
        return config

    config.update(
        {
            "auto_map": dict(GIGACHAT_AUTO_MAP),
            "attention_type": "LlamaLatentAttention",
            "num_krope_heads": 1,
            "attention_distillation": False,
            "gated_attention": attention_fields["gated_attention"],
            "use_mla_scaling_factor": attention_fields["use_mla_scaling_factor"],
            "use_shared_expert_sigmoid": False,
            "norm_type": norm_type,
            "layernorm_type": layernorm_type,
            "layernorm_gating_weight": gating_weight
            if gating_weight is not None
            else 2.0,
            "swiglu_limit": swiglu_limit,
            "final_logit_softcapping": logit_softcap if logit_softcap > 0.0 else None,
            # Hybrid pattern: ``layer_types`` is what HF forward and sglang
            # dispatch on; ``full_attention_layers`` is what HF's constructor
            # derives it from. Both are written so they cannot disagree.
            "full_attention_layers": full_attention_layers,
            "layer_types": [
                "full_attention" if i in full_attention_layers else "linear_attention"
                for i in range(len(layers))
            ],
        }
    )
    if gdn:
        config.update(_gdn_fields(gdn))
    else:
        config.update(
            {
                "linear_attention_type": None,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "linear_gating_type": "gated_rmsnorm",
                "linear_sigmoid_gate_scale": 1.0,
                "linear_attn_o_norm_eps": None,
                "linear_use_legacy_qkvz_layout": False,
                "linear_use_short_conv": True,
            }
        )
    if nextn_is_sparse is not None:
        config["nextn_is_sparse"] = nextn_is_sparse
    return config
