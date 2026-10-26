# from transformers.models.deepseek_v3.configuration_deepseek_v3 impAort DeepseekV3Config
from typing import Optional
from transformers.configuration_utils import PretrainedConfig



class Gigachat35Config(PretrainedConfig):
    model_type = "gigachat3_5"
    
    def __init__(
        self,
        vocab_size=129280,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        num_hidden_layers=61,
        nextn_is_sparse=None,
        nextn_is_linear=None,
        num_attention_heads=128,
        num_key_value_heads=128,
        n_shared_experts=1,
        n_routed_experts=256,
        routed_scaling_factor=2.5,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        n_group=8,
        topk_group=4,
        num_experts_per_tok=8,
        first_k_dense_replace=3,
        norm_topk_prob=True,
        aux_loss_alpha=0.001,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-20,
        optimizer_eps=1e-20,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_interleave=True,
        attention_bias=False,
        attention_dropout=0.0,
        attention_type="LlamaLatentAttention",
        num_krope_heads=1,
        norm_type="LlamaRMSNorm",
        indexer_norm_type="LlamaRMSNorm",
        layernorm_type="pre",
        layernorm_gating_weight=2.0,
        gated_attention=False,
        use_shared_expert_sigmoid=False,
        use_mla_scaling_factor=False,
        attention_distillation=False,
        linear_attention_type=None,
        full_attention_layers=None,
        dense_attention_layers: Optional[list[int]] = None,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4, 
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_use_short_conv=True,
        linear_gating_type: str = "gated_rmsnorm",
        linear_sigmoid_gate_scale: float = 1.0,
        linear_attn_o_norm_eps: Optional[float] = None,
        linear_use_legacy_qkvz_layout: bool = False,
        swiglu_limit: float = 0.0,
        final_logit_softcapping: Optional[float] = None,
        index_n_heads: Optional[int] = None,
        index_head_dim: Optional[int] = None,
        index_topk: Optional[int] = None,
        indexer_rotate_activation_enabled: Optional[bool] = None,
        **kwargs,
    ):
        if layernorm_type is None:
            layernorm_type = "pre"

        if layernorm_type not in {"pre", "post", "pre_post"}:
            raise ValueError(
                "layernorm_type must be one of {'pre', 'post', 'pre_post'}, "
                f"got {layernorm_type!r}"
            )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.nextn_is_sparse = nextn_is_sparse
        self.nextn_is_linear = nextn_is_linear
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.rope_interleave = rope_interleave

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.optimizer_eps = optimizer_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.num_krope_heads = num_krope_heads
        self.norm_type = norm_type
        self.indexer_norm_type = indexer_norm_type
        self.layernorm_type = layernorm_type
        self.layernorm_gating_weight = layernorm_gating_weight
        self.gated_attention = gated_attention
        self.use_shared_expert_sigmoid = use_shared_expert_sigmoid
        self.use_mla_scaling_factor = use_mla_scaling_factor
        
        _supported_linear_attention_types = (
            None,
            "Qwen3NextGatedDeltaNet",
            "KimiDeltaAttention",
        )
        if linear_attention_type not in _supported_linear_attention_types:
            raise ValueError(
                "linear_attention_type must be one of "
                f"{_supported_linear_attention_types}, got {linear_attention_type!r}"
            )

        self.attention_distillation = attention_distillation
        self.linear_attention_type = linear_attention_type
        self.full_attention_layers = full_attention_layers or []
        if self.linear_attention_type is not None:
            self.layer_types = [
                "full_attention" if i in self.full_attention_layers else "linear_attention" \
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.full_attention_layers = list(range(self.num_hidden_layers))
            self.layer_types = ["full_attention" for _ in range(self.num_hidden_layers)]

        normalized_dense_attention_layers = (
            [] if dense_attention_layers is None else dense_attention_layers
        )
        if not isinstance(normalized_dense_attention_layers, list):
            raise ValueError(
                "dense_attention_layers must be a list of unique layer indices, "
                f"got {dense_attention_layers!r}."
            )
        if any(
            isinstance(layer_idx, bool) or not isinstance(layer_idx, int)
            for layer_idx in normalized_dense_attention_layers
        ):
            raise ValueError(
                "dense_attention_layers must contain only integer layer indices, "
                f"got {normalized_dense_attention_layers!r}."
            )
        if len(set(normalized_dense_attention_layers)) != len(
            normalized_dense_attention_layers
        ):
            raise ValueError(
                "dense_attention_layers must contain unique layer indices, "
                f"got {normalized_dense_attention_layers!r}."
            )
        if (
            normalized_dense_attention_layers
            and attention_type != "LlamaLatentSparseAttention"
        ):
            raise ValueError(
                "dense_attention_layers is supported only with "
                "attention_type='LlamaLatentSparseAttention'."
            )
        invalid_dense_layer_ids = set(normalized_dense_attention_layers) - set(
            range(self.num_hidden_layers)
        )
        if invalid_dense_layer_ids:
            raise ValueError(
                "dense_attention_layers contains out-of-range layer indices: "
                f"{sorted(invalid_dense_layer_ids)} for {self.num_hidden_layers} layers."
            )
        non_full_dense_layer_ids = set(normalized_dense_attention_layers) - set(
            self.full_attention_layers
        )
        if non_full_dense_layer_ids:
            raise ValueError(
                "dense_attention_layers must be a subset of full_attention_layers; "
                f"invalid indices: {sorted(non_full_dense_layer_ids)}."
            )
        self.dense_attention_layers = normalized_dense_attention_layers
        # `layer_types` is always derived from `full_attention_layers` — drop any
        # value coming through **kwargs (e.g. a stale list persisted in a prior
        # `config.json`) so the base `PretrainedConfig.__init__` does not
        # overwrite the freshly computed list via setattr. Same for the derived
        # `full_attention_layers` in the non-linear branch.
        kwargs.pop("layer_types", None)
        if self.linear_attention_type is None:
            kwargs.pop("full_attention_layers", None)
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_use_short_conv = linear_use_short_conv

        # Gating/normalization applied to the Qwen3NextGatedDeltaNet output.
        # KimiDeltaAttention always uses its sigmoid gated RMSNorm path.
        # Supported values:
        #   - "gated_rmsnorm": per-head FusedRMSNormGated with SiLU activation.
        #   - "gated_rmsnorm_sigmoid": per-head FusedRMSNormGated with sigmoid activation.
        #   - "gated_rmsnorm_sigmoid_zero_centered": same as above, but with the
        #     zero-centered (1 + weight) parameterization.
        #   - "sigmoid_gate": element-wise sigmoid gate using the projected `z`,
        #     no normalization.
        _supported_gating_types = (
            "gated_rmsnorm",
            "gated_rmsnorm_sigmoid",
            "gated_rmsnorm_sigmoid_zero_centered",
            "sigmoid_gate",
        )
        if linear_gating_type not in _supported_gating_types:
            raise ValueError(
                f"linear_gating_type must be one of {_supported_gating_types}, "
                f"got {linear_gating_type!r}"
            )
        self.linear_gating_type = linear_gating_type
        self.linear_sigmoid_gate_scale = linear_sigmoid_gate_scale
        self.linear_attn_o_norm_eps = linear_attn_o_norm_eps
        self.linear_use_legacy_qkvz_layout = linear_use_legacy_qkvz_layout

        # DeepSeek-style hard SwiGLU activation clipping bound. 0.0 disables.
        # When > 0:  gate.clamp(max=L) (upper-only) and up.clamp(-L, L) (symmetric).
        # Used by both dense MLP and MoE experts on the inference side.
        self.swiglu_limit = swiglu_limit
        self.final_logit_softcapping = final_logit_softcapping
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.indexer_rotate_activation_enabled = indexer_rotate_activation_enabled

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def _use_pre_layernorm(self) -> bool:
        return self.layernorm_type in {"pre", "pre_post"}

    @property
    def _use_post_layernorm(self) -> bool:
        return self.layernorm_type in {"post", "pre_post"}

    def is_dsa_attention_layer(self, layer_idx: int) -> bool:
        """Return whether a full-attention layer owns and uses a DSA indexer."""
        return (
            self.index_topk is not None
            and layer_idx in self.full_attention_layers
            and layer_idx not in self.dense_attention_layers
        )


Gigachat35Config.register_for_auto_class("AutoConfig")
__all__ = ["Gigachat35Config"]
