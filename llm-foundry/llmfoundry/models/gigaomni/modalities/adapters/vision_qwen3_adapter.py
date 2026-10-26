import torch
import torch.nn as nn

from typing import Dict, List, Tuple, Any

from ..base_component import BaseOmniComponent


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size * spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm
        self.initializer_range = initializer_range
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size)
        self._init_rules()
        self.reset_parameters()

    def _init_rules(self) -> None:
        def raise_init_error(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "Qwen3 patch merger child modules must be initialized by the parent"
            )

        for module in (self.norm, self.linear_fc1, self.linear_fc2):
            module.skip_init = True
            module.reset_parameters = raise_init_error

    def reset_parameters(self) -> None:
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.normal_(self.linear_fc1.weight, mean=0.0, std=self.initializer_range)
        nn.init.zeros_(self.linear_fc1.bias)
        nn.init.normal_(self.linear_fc2.weight, mean=0.0, std=self.initializer_range)
        nn.init.zeros_(self.linear_fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(
            x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x
        ).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionQwen3Adapter(nn.Module, BaseOmniComponent):
    main_input_name: str = "pixel_embeds"
    output_name: str = "pixel_embeds"

    def __init__(
        self,
        llm_hidden_size: int,
        ve_hidden_size: int = 1152,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.patch_merger = Qwen3VLVisionPatchMerger(
            hidden_size=ve_hidden_size,
            out_hidden_size=llm_hidden_size,
            spatial_merge_size=spatial_merge_size,
            use_postshuffle_norm=use_postshuffle_norm,
            initializer_range=initializer_range,
        )

    def forward(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, ...]:
        pixel_embeds, sizes = self._concat_inputs(inputs)
        outputs = self.patch_merger(pixel_embeds)
        return self._split_outputs(outputs, sizes)

    def _concat_inputs(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[int]]:
        assert inputs, "Qwen3 adapter requires at least one encoder output"
        for inp in inputs:
            assert "pixel_embeds" in inp, (
                f"pixel_embeds required but not in input keys: {list(inp.keys())}"
            )
        sizes = [inp["pixel_embeds"].shape[0] for inp in inputs]
        concatenated_inputs = torch.cat([inp["pixel_embeds"] for inp in inputs])
        return concatenated_inputs, sizes

    def _split_outputs(
        self, outputs: torch.Tensor, sizes: List[int]
    ) -> Tuple[torch.Tensor, ...]:
        merge_unit = self.spatial_merge_size**2
        assert not any(size % merge_unit for size in sizes), (
            "Qwen3 adapter inputs must contain complete spatial groups"
        )
        split_sizes = [size // merge_unit for size in sizes]
        return tuple(
            output.contiguous() for output in torch.split(outputs, split_sizes)
        )

    def unwrap_tuple_output(
        self, inputs: List[Dict[str, torch.Tensor]], outputs: Tuple[torch.Tensor, ...]
    ) -> List[Dict[str, torch.Tensor]]:
        return [{"pixel_embeds": output} for output in outputs]
