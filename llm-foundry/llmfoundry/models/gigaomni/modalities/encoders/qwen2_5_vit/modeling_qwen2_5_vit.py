import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import accumulate
from math import inf
from typing import Dict, List, Literal, Optional, Tuple
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel

from ...base_component import BaseOmniComponent
from .configuration_qwen2_5_vit import Qwen2_5VisionConfig


logging.basicConfig(
    format="%(asctime)s | %(levelname)s |  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

try:
    from flash_attn.layers.rotary import apply_rotary_emb
    from flash_attn import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    logger.warning(
        "Flash attention is not installed in your environment. "
        "Qwen2_5ViTForVisualLanguageModeling will use eager attention instead."
    )
    apply_rotary_emb = None
    flash_attn_varlen_func = None
    FLASH_ATTN_AVAILABLE = False

try:
    from composer.utils.dist import get_sp_group_size, get_sp_group_rank
    from llmfoundry.models.parallel.sequence.mappings import (
        gather_from_sequence_parallel_region_on_batch_size_dimension,
    )
except ImportError:
    logger.info(
        "Encountered exception while importing functions for training with sequence parallelism. "
        "It's normal behaviour if you are loading hf model."
    )

    def get_sp_group_size() -> int:
        return 1


class Qwen2_5RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def reset_parameters(self) -> None:
        self.weight.data.fill_(1.0)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2_5MLP(nn.Module):
    def __init__(self, config: Qwen2_5VisionConfig, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


class Qwen2_5PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class Qwen2_5RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(
        self,
        seqlen: int,
        device: torch.device,
    ) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )
        seq = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(seq, inv_freq)
        return freqs


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(-2).float().to(q.device)
    sin = sin.unsqueeze(-2).float().to(q.device)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def apply_rope_fa2(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous().to(q.device)
    sin = sin.chunk(2, dim=-1)[0].contiguous().to(q.device)
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_length, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_length, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, seq_length, head_dim
    )


class Qwen2_5FA2Attention(nn.Module):
    def __init__(self, config: Qwen2_5VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q, k, v = (
            qkv.reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        assert position_embeddings is not None, "position_embeddings must be provided."
        cos, sin = position_embeddings
        q, k = apply_rope_fa2(
            q.unsqueeze(0), k.unsqueeze(0), cos.to(q.device), sin.to(q.device)
        )
        q, k = q.squeeze(0), k.squeeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=0.0,
            causal=False,
        )
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5EagerAttention(nn.Module):
    def __init__(self, config: Qwen2_5VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q, k, v = (
            qkv.reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        assert position_embeddings is not None, "position_embeddings must be provided."
        cos, sin = position_embeddings
        q, k = apply_rope(q, k, cos.to(q.device), sin.to(q.device))
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        dropout = 0.0 if not self.training else self.attention_dropout
        q_chunks = torch.split(q, lengths, dim=2)
        k_chunks = torch.split(k, lengths, dim=2)
        v_chunks = torch.split(v, lengths, dim=2)
        attn_outputs = []
        for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
            key_states = repeat_kv(k_chunk, self.num_key_value_groups)
            value_states = repeat_kv(v_chunk, self.num_key_value_groups)
            attn_weights = (
                torch.matmul(q_chunk, key_states.transpose(2, 3)) * self.scaling
            )
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                q_chunk.dtype
            )
            attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)
            attn_outputs.append(
                torch.matmul(attn_weights, value_states).transpose(1, 2)
            )
        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5VisionBlock(nn.Module):
    def __init__(
        self,
        config: Qwen2_5VisionConfig,
        attn_implementation: Literal[
            "eager", "flash_attention_2"
        ] = "flash_attention_2",
    ):
        super().__init__()
        self.norm1 = Qwen2_5RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2_5RMSNorm(config.hidden_size, eps=1e-6)
        attn_impl = config.attn_impl or attn_implementation
        self.attn = (
            Qwen2_5FA2Attention(config)
            if attn_impl == "flash_attention_2" and FLASH_ATTN_AVAILABLE
            else Qwen2_5EagerAttention(config)
        )
        self.mlp = Qwen2_5MLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5VisionModel(PreTrainedModel):
    config_class = Qwen2_5VisionConfig
    config: Qwen2_5VisionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5VisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def __init__(self, config: Qwen2_5VisionConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_embed = Qwen2_5PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5RotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [Qwen2_5VisionBlock(config) for _ in range(config.depth)]
        )

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max().item()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size, device=grid_thw.device)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def _get_window_index(
        self, grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Build a flattened patch reorder index and cumulative window sequence lengths.

        Args:
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                Per-sample ``(T, H, W)`` patch-grid sizes before spatial merging.

        Returns:
            Tuple containing:
                - `window_index`: flattened indices that group merged patches by local ViT window.
                - `cu_window_seqlens`: cumulative token counts per window after applying
                  `self.spatial_merge_unit`.
        """
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values (`torch.Tensor` of shape `(seq_len, patch_dim)`):
                Flattened raw patches from the Qwen image processor; typically
                ``patch_dim = in_channels * temporal_patch_size * patch_size**2`` (e.g. 1176).
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                Per-image ``(T, H, W)`` patch-grid sizes (same convention as HF Qwen2-VL).

        Returns:
            Tuple of ``pixel_embeds`` `(seq_len, hidden_size)` and ``window_index`` for the adapter.
        """
        pixel_embeds = self.patch_embed(pixel_values)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self._get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=pixel_embeds.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        seq_len, _ = pixel_embeds.size()
        pixel_embeds = pixel_embeds.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        pixel_embeds = pixel_embeds[window_index, :, :]
        pixel_embeds = pixel_embeds.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            pixel_embeds = blk(
                pixel_embeds,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return pixel_embeds, window_index


class Qwen2_5ViTForVisualLanguageModeling(Qwen2_5VisionModel, BaseOmniComponent):
    def forward(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, ...]:  # type: ignore
        sp_group_size = get_sp_group_size() or 1
        pixel_values, grid_thw, n_patches_per_modality, n_images_per_modality = (
            self._concat_inputs(inputs)
        )
        total_real_patch_count = sum(n_patches_per_modality)
        if sp_group_size > 1 and self.training:
            pixel_values_per_split, grid_thw_per_split, paddings_per_split = (
                self._divide_with_padding(
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    sp_group_size=sp_group_size,
                )
            )
            sp_rank = get_sp_group_rank()
            local_pixel_values = pixel_values_per_split[sp_rank]
            local_grid_thw = grid_thw_per_split[sp_rank]
            local_pixel_embeds, _ = super().forward(
                pixel_values=local_pixel_values, grid_thw=local_grid_thw
            )
            window_index, _ = self._get_window_index(grid_thw)
            pixel_embeds = self._gather_sp_pixel_embeds(
                pixel_embeds=local_pixel_embeds, paddings_per_split=paddings_per_split
            )
            pixel_embeds = pixel_embeds[:total_real_patch_count]
        else:
            pixel_embeds, window_index = super().forward(
                pixel_values=pixel_values, grid_thw=grid_thw
            )
        return self._split_outputs(
            pixel_embeds,
            grid_thw,
            n_patches_per_modality,
            n_images_per_modality,
        )

    def _gather_sp_pixel_embeds(
        self,
        pixel_embeds: torch.Tensor,
        paddings_per_split: List[int],
    ) -> torch.Tensor:
        sp_group_size = get_sp_group_size()
        sp_rank = get_sp_group_rank()
        local_padding = paddings_per_split[sp_rank]
        local_padding_pixel_embeds = torch.zeros(
            size=(local_padding, pixel_embeds.shape[1]),
            device=pixel_embeds.device,
            dtype=pixel_embeds.dtype,
        )
        pixel_embeds = torch.cat(
            [pixel_embeds, local_padding_pixel_embeds], dim=0
        ).contiguous()
        pixel_embeds = gather_from_sequence_parallel_region_on_batch_size_dimension(
            pixel_embeds.clone(), tensor_parallel_output_grad=True
        )
        num_features_per_sp_rank = pixel_embeds.shape[0] // sp_group_size
        real_features_idxs = []
        for sp_rank, local_padding in zip(range(sp_group_size), paddings_per_split):
            offset = num_features_per_sp_rank * sp_rank
            real_features_local_idxs = torch.arange(
                num_features_per_sp_rank - local_padding, device=pixel_embeds.device
            )
            real_features_idxs.append(offset + real_features_local_idxs)
        real_features_idxs = torch.cat(real_features_idxs)
        pixel_embeds = pixel_embeds[real_features_idxs]
        return pixel_embeds

    def _divide_with_padding(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        sp_group_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        num_images = grid_thw.shape[0]
        patch_counts_per_image = grid_thw.prod(1).tolist()
        pixel_values_per_image = list(pixel_values.split(patch_counts_per_image, dim=0))
        grid_thw_per_image = list(grid_thw.split(1, dim=0))
        if num_images < sp_group_size:
            pad_size = sp_group_size - num_images
            min_num_patches_side = self.spatial_merge_size
            min_num_patches_total = self.spatial_merge_unit
            min_embed_dim = (
                self.patch_size
                * self.patch_size
                * self.config.in_channels
                * self.config.temporal_patch_size
            )
            pixel_values_pad = [
                torch.zeros(
                    min_num_patches_total,
                    min_embed_dim,
                    device=pixel_values.device,
                    dtype=pixel_values.dtype,
                )
                for _ in range(pad_size)
            ]
            grid_thw_pad = [
                torch.tensor(
                    [[1, min_num_patches_side, min_num_patches_side]],
                    dtype=grid_thw.dtype,
                    device=grid_thw.device,
                )
                for _ in range(pad_size)
            ]
            pixel_values_per_image.extend(pixel_values_pad)
            grid_thw_per_image.extend(grid_thw_pad)
        n_patches_per_image = [
            pixel_values.shape[0] for pixel_values in pixel_values_per_image
        ]
        assert len(n_patches_per_image) >= sp_group_size, (
            f"Number of image after padding must be greater or equal to number of SP ranks, got: {len(n_patches_per_image)=}, {sp_group_size=}"
        )
        sp_partitions = self._build_sp_partitions(n_patches_per_image, sp_group_size)
        pixel_values_per_split = []
        grid_thw_per_split = []
        n_patches_per_split = []
        for start, end in sp_partitions:
            local_pixel_values = pixel_values_per_image[start:end]
            local_rank_grid_thw = grid_thw_per_image[start:end]
            if local_pixel_values:
                local_pixel_values = torch.cat(local_pixel_values, dim=0)
                local_rank_grid_thw = torch.cat(local_rank_grid_thw, dim=0)
            else:
                local_pixel_values = torch.zeros(
                    0,
                    pixel_values.shape[1],
                    device=pixel_values.device,
                    dtype=pixel_values.dtype,
                )
                local_rank_grid_thw = torch.zeros(
                    0, grid_thw.shape[1], device=grid_thw.device, dtype=grid_thw.dtype
                )
            local_n_patches = int(local_pixel_values.shape[0])
            pixel_values_per_split.append(local_pixel_values)
            grid_thw_per_split.append(local_rank_grid_thw)
            n_patches_per_split.append(local_n_patches)
        max_n_patches = max(n_patches_per_split)
        paddings_per_split = [
            max_n_patches - local_n_patches for local_n_patches in n_patches_per_split
        ]
        return pixel_values_per_split, grid_thw_per_split, paddings_per_split

    def _concat_inputs(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        pixel_values = []
        grid_thw = []
        n_patches_per_modality = []
        n_images_per_modality = []
        for modality_input in inputs:
            assert "pixel_values" in modality_input, (
                f"`pixel_values` is required, but not found in input_item: {modality_input.keys()}"
            )
            assert "grid_thw" in modality_input, (
                f"`grid_thw` is required, but not found in input_item: {modality_input.keys()}"
            )
            modality_pixel_values = modality_input["pixel_values"]
            modality_grid_thw = modality_input["grid_thw"]
            pixel_values.append(modality_pixel_values)
            grid_thw.append(modality_grid_thw)
            modality_n_patches = int(modality_grid_thw.prod(1).sum().item())
            modality_n_images = int(modality_grid_thw.shape[0])
            n_patches_per_modality.append(modality_n_patches)
            n_images_per_modality.append(modality_n_images)
        pixel_values = torch.cat(pixel_values, dim=0)
        grid_thw = torch.cat(grid_thw, dim=0)
        return pixel_values, grid_thw, n_patches_per_modality, n_images_per_modality

    def _split_outputs(
        self,
        pixel_embeds: torch.Tensor,
        grid_thw: torch.Tensor,
        n_patches_per_modality: List[int],
        n_images_per_modality: List[int],
    ) -> Tuple[torch.Tensor, ...]:
        pixel_values_per_modality = torch.split(
            pixel_embeds, n_patches_per_modality, dim=0
        )
        grid_thw_per_modality = torch.split(grid_thw, n_images_per_modality, dim=0)
        window_index_per_modality = []
        for modality_grid_thw in grid_thw_per_modality:
            window_index, _ = self._get_window_index(modality_grid_thw)
            window_index_per_modality.append(window_index)
        result = []
        for modality_pixel_values, modality_window_index in zip(
            pixel_values_per_modality, window_index_per_modality
        ):
            result.append(modality_pixel_values.contiguous())
            result.append(modality_window_index.contiguous())
        return tuple(result)

    def unwrap_tuple_output(
        self, inputs: List[Dict[str, torch.Tensor]], outputs: Tuple[torch.Tensor, ...]
    ) -> List[Dict[str, torch.Tensor]]:
        return [
            {"pixel_embeds": outputs[i * 2], "window_index": outputs[i * 2 + 1]}
            for i in range(len(inputs))
        ]

    @staticmethod
    def _build_sp_partitions(
        n_patches_per_image: List[int], n_ranks: int
    ) -> List[Tuple[int, int]]:
        """
        Build NON-EMPTY partition if images into given number of ranks.
        Number of images is always greater or equal to n_ranks.
        """
        n_images = len(n_patches_per_image)
        prefix_sums = [0, *accumulate(n_patches_per_image)]
        total_patches = prefix_sums[-1]
        target_load = total_patches / n_ranks
        dp = [[inf] * (n_ranks + 1) for _ in range(n_images + 1)]
        prev_cut = [[-1] * (n_ranks + 1) for _ in range(n_images + 1)]
        dp[0][0] = 0.0
        for image_count in range(1, n_images + 1):
            for ranks_used in range(1, min(image_count, n_ranks) + 1):
                best_cost = inf
                best_prev = -1
                prev_min = ranks_used - 1
                prev_max = image_count - 1
                for prev_end in range(prev_min, prev_max + 1):
                    segment_sum = prefix_sums[image_count] - prefix_sums[prev_end]
                    cost = (
                        dp[prev_end][ranks_used - 1] + (segment_sum - target_load) ** 2
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = prev_end
                dp[image_count][ranks_used] = best_cost
                prev_cut[image_count][ranks_used] = best_prev
        segments = []
        image_count = n_images
        ranks_used = n_ranks
        while ranks_used > 0:
            start = prev_cut[image_count][ranks_used]
            segments.append((start, image_count))
            image_count = start
            ranks_used -= 1
        segments.reverse()
        return segments

    def get_giga_fsdp_modules_to_wrap_with_names(
        self, prefix: str = ""
    ) -> List[Tuple[torch.nn.Module, str]]:
        modules_to_wrap_with_names = [
            (self.patch_embed, "patch_embed"),
        ] + [(blk, f"blocks.{i}") for i, blk in enumerate(self.blocks)]
        if prefix and not prefix.endswith("."):
            prefix += "."
        return [(m, prefix + name) for m, name in modules_to_wrap_with_names]

    def get_giga_fsdp_modules_for_activation_checkpointing(
        self,
    ) -> List[torch.nn.Module]:
        return list(self.blocks)
