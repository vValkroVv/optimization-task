import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import accumulate
from math import inf
from typing import Dict, List, Optional, Tuple
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel

from ...base_component import BaseOmniComponent
from .configuration_qwen3_vit import Qwen3VisionConfig

logger = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    logger.warning(
        "Flash attention is not installed. The Qwen3 vision encoder will use SDPA."
    )
    apply_rotary_emb = None
    flash_attn_varlen_func = None
    FLASH_ATTN_AVAILABLE = False

try:
    from composer.utils.dist import get_sp_group_rank, get_sp_group_size
    from llmfoundry.models.parallel.sequence.mappings import (
        gather_from_sequence_parallel_region_on_batch_size_dimension,
    )
except ImportError:
    logger.info(
        "Encountered exception while importing functions for training with sequence parallelism. "
        "It is expected when loading the HF model outside the training environment."
    )

    def get_sp_group_size() -> int:
        return 1


class Qwen3VisionMLP(nn.Module):
    def __init__(self, config: Qwen3VisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
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


class Qwen3VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer(
            "inv_freq", torch.empty(dim // 2, dtype=torch.float32), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(
                    0, self.dim, 2, device=self.inv_freq.device, dtype=torch.float32
                )
                / self.dim
            )
        )
        self.inv_freq.copy_(inv_freq)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(seq, self.inv_freq)


def apply_rotary_pos_emb_vision_fa2(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert q.ndim == k.ndim == 3, (
        "Qwen3 FA2 rotary expects Q/K shaped [tokens, heads, head_dim]"
    )
    assert q.shape == k.shape, f"Q/K shapes must match, got {q.shape} and {k.shape}"
    assert q.device == k.device, (
        f"Q/K devices must match, got {q.device} and {k.device}"
    )
    assert cos.shape == sin.shape, (
        f"cos/sin shapes must match, got {cos.shape} and {sin.shape}"
    )
    assert cos.shape == (q.shape[0], q.shape[-1] // 2), (
        f"Expected cos/sin shape {(q.shape[0], q.shape[-1] // 2)}, got {cos.shape}"
    )
    q_dtype, k_dtype = q.dtype, k.dtype
    cos = cos.contiguous().to(device=q.device, dtype=torch.float32)
    sin = sin.contiguous().to(device=q.device, dtype=torch.float32)
    q = (
        apply_rotary_emb(q.unsqueeze(0).float(), cos, sin, interleaved=False)
        .squeeze(0)
        .to(q_dtype)
    )
    k = (
        apply_rotary_emb(k.unsqueeze(0).float(), cos, sin, interleaved=False)
        .squeeze(0)
        .to(k_dtype)
    )
    return q, k


def apply_rotary_pos_emb_vision_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3 vision RoPE without materializing full-width angles."""
    assert q.ndim == k.ndim == 3, (
        "Qwen3 SDPA rotary expects Q/K shaped [tokens, heads, head_dim]"
    )
    assert q.shape == k.shape, f"Q/K shapes must match, got {q.shape} and {k.shape}"
    assert q.device == k.device, (
        f"Q/K devices must match, got {q.device} and {k.device}"
    )
    assert cos.shape == sin.shape, (
        f"cos/sin shapes must match, got {cos.shape} and {sin.shape}"
    )
    assert cos.shape == (q.shape[0], q.shape[-1] // 2), (
        f"Expected cos/sin shape {(q.shape[0], q.shape[-1] // 2)}, got {cos.shape}"
    )
    q_dtype, k_dtype = q.dtype, k.dtype
    cos = cos.to(device=q.device, dtype=torch.float32).unsqueeze(1)
    sin = sin.to(device=q.device, dtype=torch.float32).unsqueeze(1)
    q_first, q_second = q.float().chunk(2, dim=-1)
    k_first, k_second = k.float().chunk(2, dim=-1)
    q = torch.cat(
        (q_first * cos - q_second * sin, q_second * cos + q_first * sin),
        dim=-1,
    ).to(q_dtype)
    k = torch.cat(
        (k_first * cos - k_second * sin, k_second * cos + k_first * sin),
        dim=-1,
    ).to(k_dtype)
    return q, k


class Qwen3VisionAttention(nn.Module):
    def __init__(self, config: Qwen3VisionConfig) -> None:
        super().__init__()
        self.use_flash_attention = FLASH_ATTN_AVAILABLE
        self.num_heads = config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sequence_lengths: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        assert position_embeddings is not None, "position_embeddings must be provided"
        if self.use_flash_attention:
            q, k = apply_rotary_pos_emb_vision_fa2(q, k, *position_embeddings)
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            output = flash_attn_varlen_func(
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
        else:
            q, k = apply_rotary_pos_emb_vision_sdpa(q, k, *position_embeddings)
            if sequence_lengths is None:
                sequence_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            output_chunks = []
            for q_chunk, k_chunk, v_chunk in zip(
                torch.split(q, sequence_lengths, dim=0),
                torch.split(k, sequence_lengths, dim=0),
                torch.split(v, sequence_lengths, dim=0),
            ):
                output = F.scaled_dot_product_attention(
                    q_chunk.transpose(0, 1).unsqueeze(0),
                    k_chunk.transpose(0, 1).unsqueeze(0),
                    v_chunk.transpose(0, 1).unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=False,
                )
                output_chunks.append(output.squeeze(0).transpose(0, 1))
            output = torch.cat(output_chunks, dim=0)
        return self.proj(output.reshape(seq_length, -1).contiguous())


class Qwen3VisionBlock(nn.Module):
    def __init__(self, config: Qwen3VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VisionAttention(config)
        self.mlp = Qwen3VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sequence_lengths: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            sequence_lengths=sequence_lengths,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3VisionPositionEmbedding(nn.Module):
    def __init__(self, config: Qwen3VisionConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.num_position_embeddings, config.hidden_size)
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.spatial_merge_size = config.spatial_merge_size
        self.initializer_range = config.initializer_range
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.initializer_range)

    def forward(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_sizes = grid_thw.tolist()
        index_parts = [[] for _ in range(4)]
        weight_parts = [[] for _ in range(4)]
        for _, height, width in grid_sizes:
            height_positions = torch.linspace(0, self.num_grid_per_side - 1, height)
            width_positions = torch.linspace(0, self.num_grid_per_side - 1, width)
            height_floor, width_floor = height_positions.int(), width_positions.int()
            height_ceil = (height_floor + 1).clip(max=self.num_grid_per_side - 1)
            width_ceil = (width_floor + 1).clip(max=self.num_grid_per_side - 1)
            height_delta, width_delta = (
                height_positions - height_floor,
                width_positions - width_floor,
            )
            base_height = height_floor * self.num_grid_per_side
            base_height_ceil = height_ceil * self.num_grid_per_side
            indices = [
                (base_height[:, None] + width_floor[None, :]).flatten(),
                (base_height[:, None] + width_ceil[None, :]).flatten(),
                (base_height_ceil[:, None] + width_floor[None, :]).flatten(),
                (base_height_ceil[:, None] + width_ceil[None, :]).flatten(),
            ]
            weights = [
                ((1 - height_delta)[:, None] * (1 - width_delta)[None, :]).flatten(),
                ((1 - height_delta)[:, None] * width_delta[None, :]).flatten(),
                (height_delta[:, None] * (1 - width_delta)[None, :]).flatten(),
                (height_delta[:, None] * width_delta[None, :]).flatten(),
            ]
            for corner in range(4):
                index_parts[corner].append(indices[corner])
                weight_parts[corner].append(weights[corner])
        indices = torch.stack([torch.cat(parts) for parts in index_parts]).to(
            self.weight.device
        )
        weights = torch.stack([torch.cat(parts) for parts in weight_parts]).to(
            self.weight.device, self.weight.dtype
        )
        corner_embeddings = self.weight[indices] * weights.unsqueeze(-1)
        interpolated = (
            corner_embeddings[0]
            + corner_embeddings[1]
            + corner_embeddings[2]
            + corner_embeddings[3]
        )
        per_image = interpolated.split(
            [height * width for _, height, width in grid_sizes]
        )
        output = []
        merge_size = self.spatial_merge_size
        for embeddings, (num_frames, height, width) in zip(per_image, grid_sizes):
            embeddings = embeddings.repeat(num_frames, 1).view(
                num_frames,
                height // merge_size,
                merge_size,
                width // merge_size,
                merge_size,
                -1,
            )
            output.append(embeddings.permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
        return torch.cat(output)


def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    return F.pad(cu_seqlens, (1, 0), value=0)


class Qwen3VisionModel(PreTrainedModel):
    config_class = Qwen3VisionConfig
    config: Qwen3VisionConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3VisionBlock"]
    _supports_flash_attn = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, Qwen3VisionPositionEmbedding):
            module.reset_parameters()
        else:
            super()._init_weights(module)

    def __init__(self, config: Qwen3VisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size**2
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.patch_embed = Qwen3VisionPatchEmbed(config)
        self.pos_embed = Qwen3VisionPositionEmbedding(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [Qwen3VisionBlock(config) for _ in range(config.depth)]
        )
        self.post_init()

    def to_empty(self, *, device: torch.device, recurse: bool = True):
        module = super().to_empty(device=device, recurse=recurse)
        if recurse:
            self.rotary_pos_emb.reset_parameters()
        return module

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        position_ids = torch.empty(
            (int(torch.prod(grid_thw, dim=1).sum().item()), 2),
            dtype=torch.long,
            device=freq_table.device,
        )
        offset = 0
        for num_frames, height, width in grid_thw:
            merged_height, merged_width = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_height, device=freq_table.device)
            block_cols = torch.arange(merged_width, device=freq_table.device)
            intra_rows = torch.arange(merge_size, device=freq_table.device)
            intra_cols = torch.arange(merge_size, device=freq_table.device)
            row_ids = (
                block_rows[:, None, None, None] * merge_size
                + intra_rows[None, None, :, None]
            )
            col_ids = (
                block_cols[None, :, None, None] * merge_size
                + intra_cols[None, None, None, :]
            )
            row_ids = row_ids.expand(
                merged_height, merged_width, merge_size, merge_size
            ).reshape(-1)
            col_ids = col_ids.expand(
                merged_height, merged_width, merge_size, merge_size
            ).reshape(-1)
            coordinates = torch.stack((row_ids, col_ids), dim=-1)
            if num_frames > 1:
                coordinates = coordinates.repeat(num_frames, 1)
            position_ids[offset : offset + coordinates.shape[0]] = coordinates
            offset += coordinates.shape[0]
        return freq_table[position_ids].flatten(1)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        pixel_embeds = self.patch_embed(pixel_values)
        grid_thw = grid_thw.to(pixel_embeds.device)
        pixel_embeds = pixel_embeds + self.pos_embed(grid_thw).to(pixel_embeds.dtype)
        rotary_pos_emb = self.rot_pos_emb(grid_thw).reshape(pixel_embeds.shape[0], -1)
        position_embeddings = (rotary_pos_emb.cos(), rotary_pos_emb.sin())
        cu_seqlens = get_vision_cu_seqlens(grid_thw)
        sequence_lengths = None
        if any(not block.attn.use_flash_attention for block in self.blocks):
            sequence_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        for block in self.blocks:
            pixel_embeds = block(
                pixel_embeds,
                cu_seqlens=cu_seqlens,
                sequence_lengths=sequence_lengths,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return pixel_embeds


class Qwen3ViTForVisualLanguageModeling(Qwen3VisionModel, BaseOmniComponent):
    def forward(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, ...]:
        sp_group_size = get_sp_group_size() or 1
        pixel_values, grid_thw, patch_counts = self._concat_inputs(inputs)
        total_patch_count = sum(patch_counts)
        if sp_group_size > 1 and self.training:
            pixel_splits, grid_splits, paddings = self._divide_with_padding(
                pixel_values, grid_thw, sp_group_size
            )
            rank = get_sp_group_rank()
            local_embeddings = super().forward(
                pixel_values=pixel_splits[rank], grid_thw=grid_splits[rank]
            )
            pixel_embeds = self._gather_sp_pixel_embeds(local_embeddings, paddings)[
                :total_patch_count
            ]
        else:
            pixel_embeds = super().forward(pixel_values=pixel_values, grid_thw=grid_thw)
        return self._split_outputs(pixel_embeds, patch_counts)

    def _gather_sp_pixel_embeds(
        self, pixel_embeds: torch.Tensor, paddings_per_split: List[int]
    ) -> torch.Tensor:
        sp_group_size = get_sp_group_size()
        rank = get_sp_group_rank()
        local_padding = paddings_per_split[rank]
        padding = torch.zeros(
            local_padding,
            pixel_embeds.shape[1],
            device=pixel_embeds.device,
            dtype=pixel_embeds.dtype,
        )
        gathered = gather_from_sequence_parallel_region_on_batch_size_dimension(
            torch.cat([pixel_embeds, padding], dim=0).contiguous(),
            tensor_parallel_output_grad=True,
        )
        features_per_rank = gathered.shape[0] // sp_group_size
        real_indices = []
        for rank_index, padding_size in enumerate(paddings_per_split):
            offset = features_per_rank * rank_index
            real_indices.append(
                offset
                + torch.arange(features_per_rank - padding_size, device=gathered.device)
            )
        return gathered[torch.cat(real_indices)]

    def _divide_with_padding(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        sp_group_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        patch_counts = grid_thw.prod(1).tolist()
        pixel_per_image = list(pixel_values.split(patch_counts, dim=0))
        grid_per_image = list(grid_thw.split(1, dim=0))
        if len(pixel_per_image) < sp_group_size:
            pad_images = sp_group_size - len(pixel_per_image)
            patch_dim = (
                self.patch_size
                * self.patch_size
                * self.in_channels
                * self.temporal_patch_size
            )
            pixel_per_image.extend(
                [
                    torch.zeros(
                        self.spatial_merge_unit,
                        patch_dim,
                        device=pixel_values.device,
                        dtype=pixel_values.dtype,
                    )
                    for _ in range(pad_images)
                ]
            )
            grid_per_image.extend(
                [
                    torch.tensor(
                        [[1, self.spatial_merge_size, self.spatial_merge_size]],
                        device=grid_thw.device,
                        dtype=grid_thw.dtype,
                    )
                    for _ in range(pad_images)
                ]
            )
        image_patch_counts = [image.shape[0] for image in pixel_per_image]
        partitions = self._build_sp_partitions(image_patch_counts, sp_group_size)
        pixel_splits, grid_splits, split_counts = [], [], []
        for start, end in partitions:
            pixel_splits.append(torch.cat(pixel_per_image[start:end], dim=0))
            grid_splits.append(torch.cat(grid_per_image[start:end], dim=0))
            split_counts.append(pixel_splits[-1].shape[0])
        max_count = max(split_counts)
        return pixel_splits, grid_splits, [max_count - count for count in split_counts]

    def _concat_inputs(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        assert inputs, "Qwen3 encoder requires at least one modality input"
        for modality_input in inputs:
            assert "pixel_values" in modality_input, (
                f"pixel_values required but not in input keys: {list(modality_input.keys())}"
            )
            assert "grid_thw" in modality_input, (
                f"grid_thw required but not in input keys: {list(modality_input.keys())}"
            )
        pixel_values = [modality_input["pixel_values"] for modality_input in inputs]
        grids = [modality_input["grid_thw"] for modality_input in inputs]
        patch_counts = [int(grid.prod(1).sum().item()) for grid in grids]
        return torch.cat(pixel_values), torch.cat(grids), patch_counts

    @staticmethod
    def _split_outputs(
        pixel_embeds: torch.Tensor, patch_counts: List[int]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(
            output.contiguous()
            for output in torch.split(pixel_embeds, patch_counts, dim=0)
        )

    def unwrap_tuple_output(
        self, inputs: List[Dict[str, torch.Tensor]], outputs: Tuple[torch.Tensor, ...]
    ) -> List[Dict[str, torch.Tensor]]:
        return [{"pixel_embeds": output} for output in outputs]

    @staticmethod
    def _build_sp_partitions(
        patches_per_image: List[int], n_ranks: int
    ) -> List[Tuple[int, int]]:
        assert n_ranks > 0, "Sequence parallel group size must be positive"
        assert len(patches_per_image) >= n_ranks, (
            "Each sequence parallel rank requires at least one image"
        )
        n_images = len(patches_per_image)
        prefix_sums = [0, *accumulate(patches_per_image)]
        target_load = prefix_sums[-1] / n_ranks
        costs = [[inf] * (n_ranks + 1) for _ in range(n_images + 1)]
        previous = [[-1] * (n_ranks + 1) for _ in range(n_images + 1)]
        costs[0][0] = 0.0
        for end in range(1, n_images + 1):
            for ranks in range(1, min(end, n_ranks) + 1):
                for start in range(ranks - 1, end):
                    segment = prefix_sums[end] - prefix_sums[start]
                    candidate = costs[start][ranks - 1] + (segment - target_load) ** 2
                    if candidate < costs[end][ranks]:
                        costs[end][ranks] = candidate
                        previous[end][ranks] = start
        partitions = []
        end, ranks = n_images, n_ranks
        while ranks:
            start = previous[end][ranks]
            partitions.append((start, end))
            end, ranks = start, ranks - 1
        return list(reversed(partitions))

    def get_giga_fsdp_modules_to_wrap_with_names(
        self, prefix: str = ""
    ) -> List[Tuple[torch.nn.Module, str]]:
        modules_to_wrap_with_names = [
            (self.patch_embed, "patch_embed"),
            (self.pos_embed, "pos_embed"),
            *[(block, f"blocks.{index}") for index, block in enumerate(self.blocks)],
        ]
        if prefix and not prefix.endswith("."):
            prefix += "."
        return [(module, prefix + name) for module, name in modules_to_wrap_with_names]

    def get_giga_fsdp_modules_for_activation_checkpointing(
        self,
    ) -> List[torch.nn.Module]:
        return list(self.blocks)
