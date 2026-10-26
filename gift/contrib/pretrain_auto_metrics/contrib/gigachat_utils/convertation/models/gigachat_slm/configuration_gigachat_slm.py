from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

def normalize_sconv_sites(sconv) -> Optional[dict]:
    sconv_sites = ("q", "k", "v", "attn_out", "ffn_out")
    if sconv is None:
        return None
    if not isinstance(sconv, dict):
        raise ValueError(f"sconv must be a dict of site flags, got {type(sconv).__name__}")
    unknown_sites = sorted(set(sconv) - set(sconv_sites))
    if unknown_sites:
        raise ValueError(f"Unknown sconv sites {unknown_sites}; valid: {sconv_sites}")
    enabled = {site: True for site in sconv_sites if sconv.get(site)}
    return enabled or None


class GigachatSLMConfig(PretrainedConfig):
    """Dense SLM with `attention_type: LlamaAttention`, SWA, xsa, sconv, attn gate."""

    model_type = "gigachat_slm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
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
        sliding_window: Optional[int] = None,
        layer_types: Optional[list] = None,
        gated_attention: bool = False,
        use_xsa: bool = False,
        sconv: Optional[dict] = None,
        sconv_kernel_size: Optional[int] = None,
        num_nextn_predict_layers: int = 0,
        **kwargs,
    ):
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
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        layer_types = list(layer_types)
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types must have one entry per layer: {len(layer_types)} != num_hidden_layers={num_hidden_layers}"
            )
        unknown_layer_types = sorted(set(layer_types) - {"full_attention", "sliding_attention"})
        if unknown_layer_types:
            raise ValueError(
                f"Unsupported layer_types values {unknown_layer_types}; "
                "expected only full_attention / sliding_attention"
            )
        has_sliding_layers = "sliding_attention" in layer_types
        if has_sliding_layers:
            if isinstance(sliding_window, bool) or not isinstance(sliding_window, int) or sliding_window <= 0:
                raise ValueError(
                    f"sliding_attention layers need a positive integer sliding_window, got {sliding_window!r}"
                )
        else:
            sliding_window = None
        self.layer_types = layer_types
        self.sliding_window = sliding_window

        self.gated_attention = gated_attention
        self.use_xsa = use_xsa
        self.sconv = normalize_sconv_sites(sconv)
        if self.sconv:
            if not isinstance(sconv_kernel_size, int) or not 2 <= sconv_kernel_size <= 4:
                raise ValueError(
                    f"sconv sites {sorted(self.sconv)} are enabled, so sconv_kernel_size must be an integer "
                    f"in [2, 4], got {sconv_kernel_size!r}"
                )
        self.sconv_kernel_size = sconv_kernel_size
        self.num_nextn_predict_layers = num_nextn_predict_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


GigachatSLMConfig.register_for_auto_class("AutoConfig")
__all__ = ["GigachatSLMConfig"]
