import torch
import torch.nn as nn

from typing import List, Dict, Any, Optional, Tuple

from ..base_component import BaseOmniComponent
from .video_token_reduction import VideoTokenReduction


class VisionPixelShuffleAdapter(nn.Module, BaseOmniComponent):
    main_input_name: str = "pixel_embeds"
    output_name: str = "pixel_embeds"

    def __init__(
        self,
        ve_hidden_size: int,
        llm_hidden_size: int,
        downsample_ratio: float,
        use_video_tokens_reduction: bool = False,
        video_tokens_reduction_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.ve_hidden_size = ve_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.downsample_ratio = downsample_ratio
        self.use_video_tokens_reduction = use_video_tokens_reduction

        if self.use_video_tokens_reduction:
            assert video_tokens_reduction_params, (
                "video_tokens_reduction_params must be provided for video token reduction."
            )
            self.video_token_reduction = VideoTokenReduction(
                **video_tokens_reduction_params
            )

        self.module_f = nn.Sequential(
            nn.LayerNorm(ve_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                ve_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, ...]:
        pixel_embeds, sizes = self._concat_inputs(inputs)
        adapter_outputs = self._forward_adapter(pixel_embeds)
        splitted_adapter_outputs = self._split_outputs(adapter_outputs, sizes)

        if not self.use_video_tokens_reduction:
            return splitted_adapter_outputs

        splitted_adapter_outputs_with_reduction = []
        for input, output in zip(inputs, splitted_adapter_outputs):
            if "input_ids" in input:
                splitted_adapter_outputs_with_reduction.extend(
                    self.video_token_reduction(
                        pixel_embeds=output,
                        input_ids=input["input_ids"],
                        labels=input["labels"],
                        attention_mask=input["attention_mask"],
                        num_crops_per_frame=input["num_crops_per_frame"],
                    )
                )
            else:
                splitted_adapter_outputs_with_reduction.append(output)

        return tuple(splitted_adapter_outputs_with_reduction)

    def _forward_adapter(self, x: torch.Tensor) -> torch.Tensor:
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, -1)
        x = self._pixel_shuffle(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.module_f(x)
        return x

    def _pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.reshape(
            n, w, int(h * self.downsample_ratio), int(c / self.downsample_ratio)
        )
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.reshape(
            n,
            int(h * self.downsample_ratio),
            int(w * self.downsample_ratio),
            int(c / (self.downsample_ratio * self.downsample_ratio)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _concat_inputs(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        for input_item in inputs:
            assert "pixel_embeds" in input_item, (
                f"`pixel_embeds` is required, but not found in input_item: {input_item.keys()}"
            )

        pixel_embeds_list = [input_item["pixel_embeds"] for input_item in inputs]
        meta = [{"size": len(pixel_embeds)} for pixel_embeds in pixel_embeds_list]

        pixel_embeds = torch.cat(pixel_embeds_list, dim=0)
        return pixel_embeds, meta

    def _split_outputs(
        self, adapter_outputs: torch.Tensor, meta: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, ...]:
        adapter_outputs_list = torch.split(
            adapter_outputs, [item["size"] for item in meta], dim=0
        )
        return tuple(adapter_outputs_list)

    def unwrap_tuple_output(
        self, inputs: List[Dict[str, torch.Tensor]], outputs: Tuple[torch.Tensor, ...]
    ) -> List[Dict[str, torch.Tensor]]:
        # NOTE (Sbr): This is a hack to handle the case of video token reduction.
        # In this case, inputs will contain pixel_embeds, input_ids, labels, attention_mask and text_mask.
        outputs_idx = 0
        unwrapped_outputs = []
        for input in inputs:
            if self.use_video_tokens_reduction and "input_ids" in input:
                unwrapped_outputs.append(
                    {
                        "pixel_embeds": outputs[outputs_idx],
                        "input_ids": outputs[outputs_idx + 1],
                        "labels": outputs[outputs_idx + 2],
                        "attention_mask": outputs[outputs_idx + 3],
                        "text_mask": outputs[outputs_idx + 4],
                    }
                )
                outputs_idx += 5
            else:
                unwrapped_outputs.append({"pixel_embeds": outputs[outputs_idx]})
                outputs_idx += 1
        return unwrapped_outputs
