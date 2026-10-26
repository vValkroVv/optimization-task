from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class Gigachat35DenseConfig(PretrainedConfig):
    """GigaChat3.5 with dense MLP and grouped-query full attention, optionally hybrid with linear attention."""

    model_type = "gigachat3_5_dense"

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=10240,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=None,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.006,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        attention_type="LlamaAttention",
        gated_attention=False,
        norm_type="LlamaRMSNorm",
        layernorm_type="pre",
        layernorm_gating_weight=2.0,
        linear_attention_type=None,
        full_attention_layers=None,
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
        num_nextn_predict_layers: int = 0,
        **kwargs,
    ):
        if attention_type != "LlamaAttention":
            raise ValueError(
                f"Gigachat35DenseConfig supports attention_type='LlamaAttention', got {attention_type!r}"
            )

        if layernorm_type is None:
            layernorm_type = "pre"
        if layernorm_type not in {"pre", "post", "pre_post"}:
            raise ValueError(
                "layernorm_type must be one of {'pre', 'post', 'pre_post'}, "
                f"got {layernorm_type!r}"
            )

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

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.gated_attention = gated_attention
        self.norm_type = norm_type
        self.layernorm_type = layernorm_type
        self.layernorm_gating_weight = layernorm_gating_weight

        self.linear_attention_type = linear_attention_type
        self.full_attention_layers = full_attention_layers or []
        if self.linear_attention_type is not None:
            self.layer_types = [
                "full_attention" if i in self.full_attention_layers else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.full_attention_layers = list(range(self.num_hidden_layers))
            self.layer_types = ["full_attention" for _ in range(self.num_hidden_layers)]
        # Derived above; a stale value from a saved config.json must not override it via kwargs
        kwargs.pop("layer_types", None)

        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_use_short_conv = linear_use_short_conv
        self.linear_gating_type = linear_gating_type
        self.linear_sigmoid_gate_scale = linear_sigmoid_gate_scale
        self.linear_attn_o_norm_eps = linear_attn_o_norm_eps
        self.linear_use_legacy_qkvz_layout = linear_use_legacy_qkvz_layout

        self.swiglu_limit = swiglu_limit
        self.final_logit_softcapping = final_logit_softcapping
        self.num_nextn_predict_layers = num_nextn_predict_layers

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


Gigachat35DenseConfig.register_for_auto_class("AutoConfig")
__all__ = ["Gigachat35DenseConfig"]
