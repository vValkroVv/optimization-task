from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationConfig, GenerationMode
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ...models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_gigachat_slm import GigachatSLMConfig


def _mtp_keys_to_ignore_on_load(config: GigachatSLMConfig) -> list[str]:
    """Checkpoints keep the dense MTP blocks as `model.layers.{num_hidden_layers + i}`; the model has no MTP head."""
    if config.num_nextn_predict_layers <= 0:
        return []
    indices = "|".join(str(config.num_hidden_layers + i) for i in range(config.num_nextn_predict_layers))
    return [rf"^(?:model\.)?layers\.(?:{indices})\."]


class GigachatSLMDynamicCache(DynamicCache):
    """DynamicCache that also carries the short-conv input tails."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_states: dict[tuple[int, str], torch.Tensor] = {}

    def get_conv_state(self, layer_idx: int, site: str) -> Optional[torch.Tensor]:
        return self.conv_states.get((layer_idx, site))

    def update_conv_state(self, layer_idx: int, site: str, state: torch.Tensor) -> None:
        self.conv_states[(layer_idx, site)] = state

    def reorder_cache(self, beam_idx: torch.LongTensor):
        super().reorder_cache(beam_idx)
        for key, state in self.conv_states.items():
            self.conv_states[key] = state.index_select(0, beam_idx.to(state.device))

    def batch_repeat_interleave(self, repeats: int):
        super().batch_repeat_interleave(repeats)
        for key, state in self.conv_states.items():
            self.conv_states[key] = state.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        super().batch_select_indices(indices)
        for key, state in self.conv_states.items():
            self.conv_states[key] = state[indices, ...]

    def crop(self, max_length: int):
        if self.conv_states:
            raise NotImplementedError(
                "GigachatSLMDynamicCache cannot crop short-conv states; cache rollback "
                "(assisted generation, contrastive search) is not supported with sconv."
            )
        super().crop(max_length)

    def reset(self):
        super().reset()
        self.conv_states.clear()


class GigachatShortConv(nn.Module):
    """Depthwise causal conv1d with a residual shortcut, `y = x + conv1d(x)`."""

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        if mask is not None:
            x = x * mask[:, :, None].to(x.dtype)
        if conv_state is None:
            conv_state = x.new_zeros(x.shape[0], self.kernel_size - 1, x.shape[2])
        padded = torch.cat([conv_state, x], dim=1)
        weight = self.conv1d.weight.to(torch.float32)
        conv = F.conv1d(padded.transpose(1, 2), weight, groups=self.hidden_size).transpose(1, 2)
        new_state = padded[:, padded.shape[1] - (self.kernel_size - 1):].clone()
        return (x + conv).to(input_dtype), new_state


def _build_sconv(config: GigachatSLMConfig, site: str, channels: int) -> Optional[GigachatShortConv]:
    if site not in (config.sconv or {}):
        return None
    return GigachatShortConv(channels, config.sconv_kernel_size)


def _apply_sconv(
    sconv: Optional[GigachatShortConv],
    hidden_states: torch.Tensor,
    mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache],
    layer_idx: int,
    site: str,
) -> torch.Tensor:
    if sconv is None:
        return hidden_states
    has_conv_cache = isinstance(past_key_values, GigachatSLMDynamicCache)
    conv_state = past_key_values.get_conv_state(layer_idx, site) if has_conv_cache else None
    hidden_states, conv_state = sconv(hidden_states, mask=mask, conv_state=conv_state)
    if has_conv_cache:
        past_key_values.update_conv_state(layer_idx, site, conv_state)
    return hidden_states


def _apply_xsa(
    attn_output: torch.Tensor,
    self_value: torch.Tensor,
    num_key_value_groups: int,
    eps: float,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = attn_output.shape
    num_kv_heads = self_value.shape[2]
    unit = F.normalize(self_value.float(), dim=-1, eps=eps).unsqueeze(-2)
    grouped = attn_output.float().reshape(batch_size, seq_len, num_kv_heads, num_key_value_groups, head_dim)
    projection = (grouped * unit).sum(dim=-1, keepdim=True) * unit
    return (grouped - projection).reshape(batch_size, seq_len, num_heads, head_dim)


class GigachatSLMRMSNorm(Qwen3RMSNorm):
    pass


class GigachatSLMMLP(Qwen3MLP):
    pass


class GigachatSLMRotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class GigachatSLMAttention(Qwen3Attention):
    """Qwen3 attention (GQA + qk-norm + optional sliding window) with the llm-foundry attn features."""

    def __init__(self, config: GigachatSLMConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.gated_attention = config.gated_attention
        self.use_xsa = config.use_xsa
        self.rms_norm_eps = config.rms_norm_eps
        q_size = config.num_attention_heads * self.head_dim
        kv_size = config.num_key_value_heads * self.head_dim
        self.gate_proj = (
            nn.Linear(config.hidden_size, q_size, bias=config.attention_bias) if self.gated_attention else None
        )
        self.q_sconv = _build_sconv(config, "q", q_size)
        self.k_sconv = _build_sconv(config, "k", kv_size)
        self.v_sconv = _build_sconv(config, "v", kv_size)
        self.attn_out_sconv = _build_sconv(config, "attn_out", config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        gate = self.gate_proj(hidden_states) if self.gate_proj is not None else None

        query_states = _apply_sconv(self.q_sconv, query_states, padding_mask, past_key_values, self.layer_idx, "q")
        key_states = _apply_sconv(self.k_sconv, key_states, padding_mask, past_key_values, self.layer_idx, "k")
        value_states = _apply_sconv(self.v_sconv, value_states, padding_mask, past_key_values, self.layer_idx, "v")

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        self_value_states = value_states
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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        if padding_mask is not None:
            attn_output = attn_output * padding_mask[:, :, None, None].to(attn_output.dtype)
        if self.use_xsa:
            attn_output = _apply_xsa(
                attn_output, self_value_states.transpose(1, 2), self.num_key_value_groups, self.rms_norm_eps
            )

        attn_output = attn_output.reshape(*input_shape, -1)
        if gate is not None:
            attn_output = attn_output.float() * torch.sigmoid(gate.float())
        attn_output = self.o_proj(attn_output.to(hidden_states.dtype).contiguous())
        attn_output = _apply_sconv(
            self.attn_out_sconv, attn_output, padding_mask, past_key_values, self.layer_idx, "attn_out"
        )
        return attn_output, attn_weights


class GigachatSLMDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: GigachatSLMConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.ffn_out_sconv = _build_sconv(config, "ffn_out", config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            padding_mask=padding_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _apply_sconv(
            self.ffn_out_sconv, hidden_states, padding_mask, past_key_values, self.layer_idx, "ffn_out"
        )
        hidden_states = residual + hidden_states
        return hidden_states


class GigachatSLMPreTrainedModel(Qwen3PreTrainedModel):
    _can_compile_fullgraph = False


class GigachatSLMModel(Qwen3Model):
    def __init__(self, config: GigachatSLMConfig):
        super().__init__(config)
        self.uses_conv_cache = bool(config.sconv)
        self._keys_to_ignore_on_load_unexpected = _mtp_keys_to_ignore_on_load(config)

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.uses_conv_cache:
            if use_cache and past_key_values is None:
                past_key_values = GigachatSLMDynamicCache(config=self.config)
            elif past_key_values is not None and not isinstance(past_key_values, GigachatSLMDynamicCache):
                raise TypeError(
                    "Short convolutions need their conv states cached: pass a GigachatSLMDynamicCache "
                    f"(or no cache at all) instead of {type(past_key_values).__name__}."
                )
            if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
                seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
                kwargs["padding_mask"] = attention_mask[:, -seq_len:]
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )


class GigachatSLMForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: GigachatSLMConfig):
        super().__init__(config)
        self._keys_to_ignore_on_load_unexpected = _mtp_keys_to_ignore_on_load(config)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        generation_mode: GenerationMode,
        batch_size: int,
        max_cache_length: int,
    ) -> bool:
        uses_conv_cache = bool(self.config.sconv)
        if uses_conv_cache and generation_mode in (
            GenerationMode.ASSISTED_GENERATION,
            GenerationMode.CONTRASTIVE_SEARCH,
        ):
            raise ValueError(
                f"{generation_mode.value} needs cache rollback (`Cache.crop`), which GigachatSLMDynamicCache does "
                "not support for the short-conv states; use greedy, sampling or beam search instead."
            )
        if uses_conv_cache and generation_config.cache_implementation not in (None, "dynamic", "dynamic_full", "offloaded"):
            raise ValueError(
                f"cache_implementation={generation_config.cache_implementation!r} is not supported with short convs; "
                "use 'dynamic', 'dynamic_full' or 'offloaded'."
            )
        user_defined_cache = model_kwargs.get("past_key_values") is not None
        result = super()._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )
        cache = model_kwargs.get("past_key_values")
        if uses_conv_cache and not user_defined_cache and type(cache) is DynamicCache:
            cache_kwargs = {} if generation_config.cache_implementation == "dynamic_full" else {"config": self.config}
            model_kwargs["past_key_values"] = GigachatSLMDynamicCache(**cache_kwargs, offloading=cache.offloading)
        return result


__all__ = [
    "GigachatSLMForCausalLM",
    "GigachatSLMModel",
    "GigachatSLMPreTrainedModel",
]
