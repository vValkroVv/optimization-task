import math
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextGatedDeltaNet,
    Qwen3NextDynamicCache,
    apply_mask_to_padding_states,
)
from ...models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3RMSNorm,
    DeepseekV3MLP,
    DeepseekV3TopkRouter,
    DeepseekV3MoE,
    DeepseekV3DecoderLayer,
    DeepseekV3Model,
    DeepseekV3ForCausalLM,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
    yarn_get_mscale,
)
from ...models.gemma.modeling_gemma import GemmaRMSNorm
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...masking_utils import create_causal_mask
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg
from ...cache_utils import Cache, DynamicCache
from ...processing_utils import Unpack
from .configuration_gigachat3_5 import Gigachat35Config

try:
    from fla.modules import FusedRMSNormGated
except ImportError:
    FusedRMSNormGated = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_update, causal_conv1d_fn = None, None


def _apply_sigmoid_gate_fp32(hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (hidden_states.float() * torch.sigmoid(gate.float())).to(hidden_states.dtype)


def _apply_linear_attention_padding_mask(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    seq_len = hidden_states.shape[1]
    if attention_mask is not None and attention_mask.dim() == 2:
        if attention_mask.shape[1] > seq_len:
            attention_mask = attention_mask[:, -seq_len:]
        if attention_mask.shape[1] == seq_len and attention_mask.shape[1] > 1:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
    return hidden_states


def _torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


class Gigachat35GatedDeltaRMSNorm(nn.Module):
    """Pure-PyTorch fallback for `fla.modules.FusedRMSNormGated`.

    Supports the activation/zero-centered variants used by
    `Gigachat35GatedDeltaNetWrapper` so the model can run without FLA installed.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        activation: str = "silu",
        zero_centered: bool = False,
        **kwargs,
    ):
        super().__init__()
        if activation not in ("silu", "sigmoid"):
            raise ValueError(
                f"Unsupported activation {activation!r} for Gigachat35GatedDeltaRMSNorm; "
                "expected 'silu' or 'sigmoid'."
            )
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation
        self.zero_centered = zero_centered

    def forward(self, hidden_states: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight = self.weight.float()
        if self.zero_centered:
            weight = 1.0 + weight
        hidden_states = weight * hidden_states
        if gate is not None:
            gate_f32 = gate.to(torch.float32)
            if self.activation == "silu":
                hidden_states = hidden_states * F.silu(gate_f32)
            else:
                hidden_states = hidden_states * torch.sigmoid(gate_f32)
        return hidden_states.to(input_dtype)


def _build_gated_delta_norm(
    head_v_dim: int,
    eps: float,
    activation: str,
    zero_centered: bool = False,
) -> nn.Module:
    if FusedRMSNormGated is not None:
        return FusedRMSNormGated(
            head_v_dim,
            eps=eps,
            activation=activation,
            zero_centered=zero_centered,
        )
    return Gigachat35GatedDeltaRMSNorm(
        head_v_dim,
        eps=eps,
        activation=activation,
        zero_centered=zero_centered,
    )


class Gigachat35RMSNorm(DeepseekV3RMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__(hidden_size, eps=eps)


class Gigachat35ZeroCenteredRMSNorm(GemmaRMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__(dim, eps=eps)


class Gigachat35ZeroCenteredGatedNorm(Gigachat35ZeroCenteredRMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        layernorm_gating_weight: float = 2.0,
        **kwargs,
    ):
        super().__init__(hidden_size, eps=eps)
        self.layernorm_gating_weight = layernorm_gating_weight
        self.gate_up_projection = nn.Linear(hidden_size, 16, bias=False)
        self.gate_down_projection = nn.Linear(16, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = super().forward(hidden_states)
        gate_hidden = F.linear(hidden_states.float(), self.gate_up_projection.weight.float())
        gate_hidden = F.silu(gate_hidden)
        gate_hidden = F.linear(gate_hidden, self.gate_down_projection.weight.float())
        return (
            hidden_states.float()
            * self.layernorm_gating_weight
            * torch.sigmoid(gate_hidden)
        ).to(hidden_states.dtype)

class Gigachat35GatedNorm(Gigachat35RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        layernorm_gating_weight: float = 2.0,
        **kwargs,
    ):
        super().__init__(hidden_size, eps=eps)
        self.layernorm_gating_weight = layernorm_gating_weight
        self.gate_up_projection = nn.Linear(hidden_size, 16, bias=False)
        self.gate_down_projection = nn.Linear(16, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = super().forward(hidden_states)
        gate_hidden = F.linear(hidden_states.float(), self.gate_up_projection.weight.float())
        gate_hidden = F.silu(gate_hidden)
        gate_hidden = F.linear(gate_hidden, self.gate_down_projection.weight.float())
        return (
            hidden_states.float()
            * self.layernorm_gating_weight
            * torch.sigmoid(gate_hidden)
        ).to(hidden_states.dtype)


def resolve_norm_class(norm_type: str) -> type[nn.Module]:
    if norm_type == "LlamaRMSNorm":
        return Gigachat35RMSNorm
    if norm_type == "ZeroCenteredRMSNorm":
        return Gigachat35ZeroCenteredRMSNorm
    if norm_type == "ZeroCenteredGatedNorm":
        return Gigachat35ZeroCenteredGatedNorm
    if norm_type == "GatedNorm":
        return Gigachat35GatedNorm
    raise ValueError(
        "Unknown norm type "
        f"`{norm_type}`. Available values: LlamaRMSNorm, ZeroCenteredGatedNorm, ZeroCenteredRMSNorm, GatedNorm."
    )


def build_norm(config: Gigachat35Config, hidden_size: int) -> nn.Module:
    norm_class = resolve_norm_class(getattr(config, "norm_type", "LlamaRMSNorm"))
    return norm_class(
        hidden_size,
        eps=config.rms_norm_eps,
        layernorm_gating_weight=getattr(config, "layernorm_gating_weight", 2.0),
    )


class Gigachat35DynamicCache(Qwen3NextDynamicCache):
    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

            if self.conv_states[layer_idx] is not None:
                conv_state = self.conv_states[layer_idx]
                device = conv_state[0].device if isinstance(conv_state, (tuple, list)) else conv_state.device
                beam_idx = beam_idx.to(device)
                if isinstance(conv_state, (tuple, list)):
                    self.conv_states[layer_idx] = tuple(state.index_select(0, beam_idx) for state in conv_state)
                else:
                    self.conv_states[layer_idx] = conv_state.index_select(0, beam_idx)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx)


class Gigachat35GatedDeltaNetWrapper(Qwen3NextGatedDeltaNet):
    """Qwen3NextGatedDeltaNet variant with configurable output gating.

    Mirrors the GigaMix `linear_gating_type` flag: the value/gate path can use
    one of four flavours, matching the upstream llm-foundry attention module.

    Inherits the conv / delta-rule plumbing from `Qwen3NextGatedDeltaNet`; we
    only swap the output normalization in `__init__` and replace the final
    gating step inside `forward`.
    """

    def __init__(self, config: Gigachat35Config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.linear_gating_type = getattr(config, "linear_gating_type", "gated_rmsnorm")
        self.linear_sigmoid_gate_scale = getattr(config, "linear_sigmoid_gate_scale", 1.0)

        o_norm_eps = getattr(config, "linear_attn_o_norm_eps", None)
        if o_norm_eps is None:
            o_norm_eps = config.rms_norm_eps

        # Drop the silu-only `norm` built by the parent and rebuild it
        # according to the requested gating type so that
        # gated_rmsnorm / gated_rmsnorm_sigmoid / gated_rmsnorm_sigmoid_zero_centered
        # all map to the right gated norm flavour, and `sigmoid_gate`
        # has no normalization at all.
        if hasattr(self, "norm"):
            del self.norm

        if self.linear_gating_type == "gated_rmsnorm":
            self.norm = _build_gated_delta_norm(self.head_v_dim, o_norm_eps, activation="silu")
        elif self.linear_gating_type == "gated_rmsnorm_sigmoid":
            self.norm = _build_gated_delta_norm(self.head_v_dim, o_norm_eps, activation="sigmoid")
        elif self.linear_gating_type == "gated_rmsnorm_sigmoid_zero_centered":
            self.norm = _build_gated_delta_norm(
                self.head_v_dim, o_norm_eps, activation="sigmoid", zero_centered=True,
            )
        elif self.linear_gating_type == "sigmoid_gate":
            self.norm = None
        else:
            raise ValueError(
                f"Unknown linear_gating_type: {self.linear_gating_type!r}. "
                "Supported: 'gated_rmsnorm', 'gated_rmsnorm_sigmoid', "
                "'gated_rmsnorm_sigmoid_zero_centered', 'sigmoid_gate'."
            )

    def _apply_output_gate(self, core_attn_out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Apply the configured gating to the per-head linear-attention output.

        Both `core_attn_out` and `z` are shaped `(b, t, h, d)`.
        Returns a tensor shaped `(b, t, h * d)` ready to be fed into `out_proj`.
        """
        b, t = core_attn_out.shape[0], core_attn_out.shape[1]

        if self.linear_gating_type == "gated_rmsnorm":
            o_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z_flat = z.reshape(-1, z.shape[-1])
            output = self.norm(o_flat, z_flat)
        elif self.linear_gating_type in ("gated_rmsnorm_sigmoid", "gated_rmsnorm_sigmoid_zero_centered"):
            o_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z_flat = z.reshape(-1, z.shape[-1])
            output = self.norm(o_flat, z_flat)
            if self.linear_sigmoid_gate_scale != 1.0:
                output = output * self.linear_sigmoid_gate_scale
        else:
            # "sigmoid_gate": no norm, element-wise sigmoid over the flattened value_dim
            o_flat = core_attn_out.reshape(b, t, -1).contiguous()
            z_flat = z.reshape(b, t, -1).contiguous()
            output = _apply_sigmoid_gate_fp32(o_flat, z_flat * self.linear_sigmoid_gate_scale)

        return output.reshape(b, t, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional["Gigachat35DynamicCache"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Extra keyword arguments (`position_ids`, `past_key_values`,
        # `use_cache`, `position_embeddings`, …) are accepted for call-site
        # compatibility with `Gigachat35Attention.forward` but are unused here.
        # `past_key_values` (if provided) is treated as an alias for `cache_params`.
        if cache_params is None and "past_key_values" in kwargs:
            cache_params = kwargs["past_key_values"]
        # Body mirrors `Qwen3NextGatedDeltaNet.forward`; only the trailing
        # gating step is replaced with `_apply_output_gate` so that the four
        # `linear_gating_type` flavours behave correctly.
        hidden_states = _apply_linear_attention_padding_mask(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape

        has_previous_state = (
            cache_params is not None
            and cache_params.has_previous_state
            and cache_position is not None
        )
        use_recurrent_kernel = has_previous_state and seq_len == 1

        recurrent_state = None
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba,
        )
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if has_previous_state:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_recurrent_kernel:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state if has_previous_state else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = self._apply_output_gate(core_attn_out, z)
        return self.out_proj(core_attn_out)


class Gigachat35KimiDeltaAttentionWrapper(nn.Module):
    """Kimi Delta Attention with the same decoder-layer interface as GDN."""

    def __init__(self, config: Gigachat35Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = "silu"

        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"linear_num_value_heads={self.num_v_heads} must be divisible by "
                f"linear_num_key_heads={self.num_k_heads} for KimiDeltaAttention."
            )
        if self.head_k_dim != self.head_v_dim:
            raise ValueError(
                "KimiDeltaAttention currently expects linear_key_head_dim == "
                "linear_value_head_dim."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.q_conv1d = nn.Conv1d(
            in_channels=self.key_dim,
            out_channels=self.key_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.key_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.k_conv1d = nn.Conv1d(
            in_channels=self.key_dim,
            out_channels=self.key_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.key_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.v_conv1d = nn.Conv1d(
            in_channels=self.value_dim,
            out_channels=self.value_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.value_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16)
            ).view(1, 1, -1, 1)
        )
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_v_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_v_dim, self.value_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.value_dim, dtype=torch.float32))
        self.b_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_v_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_v_dim, self.value_dim, bias=False)
        self.o_norm = _build_gated_delta_norm(
            self.head_v_dim,
            config.rms_norm_eps,
            activation="sigmoid",
        )
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or _torch_causal_conv1d_update
        try:
            from fla.ops.kda import chunk_kda, fused_recurrent_kda
        except ImportError:
            chunk_kda, fused_recurrent_kda = None, None
        self.chunk_kda = chunk_kda
        self.recurrent_kda = fused_recurrent_kda

    def fix_query_key_value_ordering(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = torch.split(
            qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.view(query.size(0), query.size(1), self.num_k_heads, self.head_k_dim)
        key = key.view(key.size(0), key.size(1), self.num_k_heads, self.head_k_dim)
        value = value.view(value.size(0), value.size(1), self.num_v_heads, self.head_v_dim)
        return query.contiguous(), key.contiguous(), value.contiguous()

    def _project_conv_qkv(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional["Gigachat35DynamicCache"],
        use_cached_conv_state: bool,
    ) -> torch.Tensor:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if use_cached_conv_state:
            q_state, k_state, v_state = cache_params.conv_states[self.layer_idx]
            query = self.causal_conv1d_update(
                query,
                q_state,
                self.q_conv1d.weight.squeeze(1),
                self.q_conv1d.bias,
                self.activation,
            )
            key = self.causal_conv1d_update(
                key,
                k_state,
                self.k_conv1d.weight.squeeze(1),
                self.k_conv1d.bias,
                self.activation,
            )
            value = self.causal_conv1d_update(
                value,
                v_state,
                self.v_conv1d.weight.squeeze(1),
                self.v_conv1d.bias,
                self.activation,
            )
            return torch.cat([query, key, value], dim=1).transpose(1, 2)

        if cache_params is not None:
            cache_params.conv_states[self.layer_idx] = (
                F.pad(query, (self.conv_kernel_size - query.shape[-1], 0)),
                F.pad(key, (self.conv_kernel_size - key.shape[-1], 0)),
                F.pad(value, (self.conv_kernel_size - value.shape[-1], 0)),
            )

        if self.causal_conv1d_fn is not None:
            query = self.causal_conv1d_fn(
                x=query,
                weight=self.q_conv1d.weight.squeeze(1),
                bias=self.q_conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
            key = self.causal_conv1d_fn(
                x=key,
                weight=self.k_conv1d.weight.squeeze(1),
                bias=self.k_conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
            value = self.causal_conv1d_fn(
                x=value,
                weight=self.v_conv1d.weight.squeeze(1),
                bias=self.v_conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            query = F.silu(self.q_conv1d(query)[:, :, : hidden_states.shape[1]])
            key = F.silu(self.k_conv1d(key)[:, :, : hidden_states.shape[1]])
            value = F.silu(self.v_conv1d(value)[:, :, : hidden_states.shape[1]])
        return torch.cat([query, key, value], dim=1).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional["Gigachat35DynamicCache"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.chunk_kda is None or self.recurrent_kda is None:
            raise ImportError("KimiDeltaAttention requires fla.ops.kda kernels.")

        if cache_params is None and "past_key_values" in kwargs:
            cache_params = kwargs["past_key_values"]

        padding_mask = (
            attention_mask
            if attention_mask is not None and attention_mask.dim() == 2
            else None
        )
        hidden_states = _apply_linear_attention_padding_mask(hidden_states, padding_mask)

        batch_size, seq_len, _ = hidden_states.shape
        has_previous_state = (
            cache_params is not None
            and cache_params.has_previous_state
            and cache_position is not None
        )
        use_recurrent_kernel = has_previous_state and seq_len == 1
        recurrent_state = (
            cache_params.recurrent_states[self.layer_idx]
            if cache_params is not None
            else None
        )

        qkv = self._project_conv_qkv(hidden_states, cache_params, has_previous_state)
        query, key, value = self.fix_query_key_value_ordering(qkv)
        del qkv

        g = self.f_b_proj(self.f_a_proj(hidden_states)).view(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        beta = self.b_proj(hidden_states).sigmoid()

        if self.num_k_heads != self.num_v_heads:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)

        if not use_recurrent_kernel:
            core_attn_out, last_recurrent_state = self.chunk_kda(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state if has_previous_state else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_kda(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        output_gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        output = self.o_norm(
            core_attn_out.reshape(-1, self.head_v_dim),
            output_gate.reshape(-1, self.head_v_dim),
        )
        output = output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return _apply_linear_attention_padding_mask(output, padding_mask)


LINEAR_ATTENTION_CLASS_REGISTRY = {
    "Qwen3NextGatedDeltaNet": Gigachat35GatedDeltaNetWrapper,
    "KimiDeltaAttention": Gigachat35KimiDeltaAttentionWrapper,
}


class Gigachat35Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gigachat35Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True

        self.use_mla_scaling_factor = bool(getattr(config, "use_mla_scaling_factor", False))
        q_hidden_dim = config.hidden_size if self.q_lora_rank is None else self.q_lora_rank
        kv_hidden_dim = config.hidden_size if self.kv_lora_rank is None else self.kv_lora_rank
        if self.use_mla_scaling_factor:
            self.alpha_q = math.sqrt(config.hidden_size / q_hidden_dim)
            self.alpha_kv = math.sqrt(config.hidden_size / kv_hidden_dim)
        else:
            self.alpha_q = 1.0
            self.alpha_kv = 1.0

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                config.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = build_norm(config, config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = build_norm(config, self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

        self.gated_attention = getattr(config, "gated_attention", False)
        if self.gated_attention:
            self.gate_proj = nn.Linear(
                config.hidden_size,
                self.num_heads * self.v_head_dim,
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
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.alpha_q * self.q_proj(hidden_states)
        else:
            q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q_states = self.q_b_proj(self.alpha_q * q_latent)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(k_pass)
        k_pass = self.kv_b_proj(self.alpha_kv * k_pass).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

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

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.gated_attention:
            attn_output = _apply_sigmoid_gate_fp32(attn_output, self.gate_proj(hidden_states))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gigachat35TopkRouter(DeepseekV3TopkRouter):
    pass


class Gigachat35MLP(DeepseekV3MLP):
    """DeepseekV3MLP + optional hard SwiGLU activation clipping (DeepSeek style).

    When ``config.swiglu_limit > 0``, gate is clamped to ``<= L``
    (upper-only, since silu makes large negative values vanish anyway) and up
    to ``[-L, L]`` (symmetric) before the SwiGLU product. ``swiglu_limit = 0.0``
    disables clipping (default).
    """

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        L = float(getattr(self.config, "swiglu_limit", 0.0))
        if L > 0:
            gate = gate.clamp(max=L)
            up = up.clamp(-L, L)
        return self.down_proj(self.act_fn(gate) * up)


class Gigachat35MoE(DeepseekV3MoE):
    def __init__(self, config: Gigachat35Config):
        super().__init__(config)
        self.use_shared_expert_sigmoid = bool(getattr(config, "use_shared_expert_sigmoid", False))
        if self.use_shared_expert_sigmoid:
            self.shared_expert_sigmoid = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        shared_states = self.shared_experts(residuals)
        if self.use_shared_expert_sigmoid:
            shared_states = _apply_sigmoid_gate_fp32(
                shared_states,
                self.shared_expert_sigmoid(residuals),
            )
        hidden_states = hidden_states + shared_states
        return hidden_states


class Gigachat35DecoderLayer(DeepseekV3DecoderLayer):
    def __init__(self, config: Gigachat35Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = Gigachat35MoE(config)
        else:
            self.mlp = Gigachat35MLP(config)

        layernorm_type = getattr(config, "layernorm_type", "pre")
        self._use_pre_layernorm = layernorm_type in {"pre", "pre_post"}
        self._use_post_layernorm = layernorm_type in {"post", "pre_post"}

        if self._use_pre_layernorm:
            self.input_layernorm = build_norm(config, config.hidden_size)
            self.post_attention_layernorm = build_norm(config, config.hidden_size)

        if self._use_post_layernorm:
            self.post_self_attn_layernorm = build_norm(config, config.hidden_size)
            self.post_feedforward_layernorm = build_norm(config, config.hidden_size)

        self.layer_type = config.layer_types[layer_idx]
        if layer_idx in config.full_attention_layers:
            self.self_attn = Gigachat35Attention(config=config, layer_idx=layer_idx)
        else:
            try:
                linear_attn_cls = LINEAR_ATTENTION_CLASS_REGISTRY[config.linear_attention_type]
            except KeyError as e:
                raise ValueError(
                    f"Unsupported linear_attention_type: {config.linear_attention_type!r}"
                ) from e
            self.self_attn = linear_attn_cls(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        if self._use_pre_layernorm:
            hidden_states = self.input_layernorm(hidden_states)

        # Linear Attention
        if self.layer_type == "linear_attention":
            linear_attention_mask = padding_mask
            if linear_attention_mask is None and attention_mask is not None and attention_mask.dim() == 2:
                linear_attention_mask = attention_mask
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=linear_attention_mask,
            )
        # Full Attention
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        if self._use_post_layernorm:
            hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if self._use_pre_layernorm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self._use_post_layernorm:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Gigachat35Model(DeepseekV3Model):
    def __init__(self, config: Gigachat35Config):
        super().__init__(config)
        self.norm = build_norm(config, config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        padding_mask = attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                padding_mask=padding_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Gigachat35ForCausalLM(DeepseekV3ForCausalLM):
    config_class = Gigachat35Config

    def __init__(self, config: Gigachat35Config):
        super().__init__(config)

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
        **kwargs: Unpack[TransformersKwargs],
    ):
        return_dict = kwargs.pop("return_dict", None)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if use_cache and (past_key_values is None or isinstance(past_key_values, DynamicCache)):
            past_key_values = Gigachat35DynamicCache(config=self.config)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if (final_logit_softcapping := self.config.final_logit_softcapping) is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
