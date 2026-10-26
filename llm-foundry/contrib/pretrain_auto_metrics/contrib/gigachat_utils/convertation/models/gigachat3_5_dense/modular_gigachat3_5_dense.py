from typing import Callable, Optional

import torch
from torch import nn

from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg
from ...models.gigachat3_5.modeling_gigachat3_5 import (
    LINEAR_ATTENTION_CLASS_REGISTRY,
    Gigachat35DecoderLayer,
    Gigachat35DynamicCache,
    Gigachat35ForCausalLM,
    Gigachat35MLP,
    Gigachat35Model,
    _apply_sigmoid_gate_fp32,
    apply_rotary_pos_emb,
    build_norm,
    eager_attention_forward,
)
from ...models.llama.modeling_llama import LlamaAttention, LlamaPreTrainedModel
from .configuration_gigachat3_5_dense import Gigachat35DenseConfig


class Gigachat35DenseAttention(LlamaAttention):
    """Grouped-query attention with the optional sigmoid output gate of GigaChat3.5."""

    def __init__(self, config: Gigachat35DenseConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.gated_attention = config.gated_attention
        if self.gated_attention:
            self.gate_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=False,
            )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.gated_attention:
            attn_output = _apply_sigmoid_gate_fp32(attn_output, self.gate_proj(hidden_states))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gigachat35DenseMLP(Gigachat35MLP):
    pass


class Gigachat35DenseDynamicCache(Gigachat35DynamicCache):
    def __init__(self, config: Gigachat35DenseConfig):
        self.layer_types = config.layer_types
        self.transformer_layers = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        linear_layers = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]
        self.last_linear_layer = linear_layers[-1] if linear_layers else None

        # Initialized to None, lazily allocated by the layers (multi-gpu inference)
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]
        self.index_key_cache = [None for _ in range(config.num_hidden_layers)]  # touched by the inherited reorder_cache

    @property
    def has_previous_state(self):
        return self.last_linear_layer is not None and self.conv_states[self.last_linear_layer] is not None


class Gigachat35DenseDecoderLayer(Gigachat35DecoderLayer):
    def __init__(self, config: Gigachat35DenseConfig, layer_idx: int):
        # Not super().__init__(): the parent builds MLA attention and MoE
        GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.mlp = Gigachat35DenseMLP(config)

        self._use_pre_layernorm = config._use_pre_layernorm
        self._use_post_layernorm = config._use_post_layernorm
        if self._use_pre_layernorm:
            self.input_layernorm = build_norm(config, config.hidden_size)
            self.post_attention_layernorm = build_norm(config, config.hidden_size)
        if self._use_post_layernorm:
            self.post_self_attn_layernorm = build_norm(config, config.hidden_size)
            self.post_feedforward_layernorm = build_norm(config, config.hidden_size)

        self.layer_type = config.layer_types[layer_idx]
        if layer_idx in config.full_attention_layers:
            self.self_attn = Gigachat35DenseAttention(config=config, layer_idx=layer_idx)
        else:
            try:
                linear_attn_cls = LINEAR_ATTENTION_CLASS_REGISTRY[config.linear_attention_type]
            except KeyError as e:
                raise ValueError(
                    f"Unsupported linear_attention_type: {config.linear_attention_type!r}"
                ) from e
            self.self_attn = linear_attn_cls(config=config, layer_idx=layer_idx)


# Llama base: Gigachat35PreTrainedModel._init_weights would pull the MoE router into this file
class Gigachat35DensePreTrainedModel(LlamaPreTrainedModel):
    _can_compile_fullgraph = False


class Gigachat35DenseModel(Gigachat35Model):
    pass


class Gigachat35DenseForCausalLM(Gigachat35ForCausalLM):
    pass


__all__ = [
    "Gigachat35DenseForCausalLM",
    "Gigachat35DenseModel",
    "Gigachat35DensePreTrainedModel",
]
