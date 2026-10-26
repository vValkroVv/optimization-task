import torch
import torch.nn as nn

from typing import Dict, List, Tuple

from ..base_component import BaseOmniComponent


class Qwen2_5RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
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


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2_5RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class VisionQwen2_5Adapter(nn.Module, BaseOmniComponent):
    main_input_name: str = "pixel_embeds"
    output_name: str = "pixel_embeds"

    def __init__(
        self,
        llm_hidden_size: int,
        ve_hidden_size: int = 1280,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.patch_merger = Qwen2_5_VLPatchMerger(
            dim=llm_hidden_size,
            context_dim=ve_hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

    def forward(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, ...]:
        pixel_embeds, window_index, merged_sizes = self._concat_inputs(inputs)
        adapter_outputs = self._forward_adapter(pixel_embeds, window_index)
        return self._split_outputs(adapter_outputs, merged_sizes)

    def _concat_inputs(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        for inp in inputs:
            assert "pixel_embeds" in inp, (
                f"`pixel_embeds` required but not in input keys: {list(inp.keys())}"
            )
            assert "window_index" in inp, (
                f"`window_index` required but not in input keys: {list(inp.keys())}"
            )
        merged_sizes = [inp["window_index"].shape[0] for inp in inputs]
        pixel_embeds = torch.cat([inp["pixel_embeds"] for inp in inputs], dim=0)
        offset = 0
        rebuilt_indices = []
        for inp, m in zip(inputs, merged_sizes):
            rebuilt_indices.append(inp["window_index"] + offset)
            offset += m
        window_index = torch.cat(rebuilt_indices, dim=0)
        return pixel_embeds, window_index, merged_sizes

    def _forward_adapter(
        self, pixel_embeds: torch.Tensor, window_index: torch.Tensor
    ) -> torch.Tensor:
        merged = self.patch_merger(pixel_embeds)
        reverse_indices = torch.argsort(window_index)
        return merged[reverse_indices, :]

    def _split_outputs(
        self, adapter_outputs: torch.Tensor, merged_sizes: List[int]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(
            chunk.contiguous()
            for chunk in torch.split(adapter_outputs, merged_sizes, dim=0)
        )

    def unwrap_tuple_output(
        self, inputs: List[Dict[str, torch.Tensor]], outputs: Tuple[torch.Tensor, ...]
    ) -> List[Dict[str, torch.Tensor]]:
        return [{"pixel_embeds": o} for o in outputs]
