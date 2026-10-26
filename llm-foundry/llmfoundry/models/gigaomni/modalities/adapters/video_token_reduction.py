import warnings

import torch
import torch.nn as nn

from typing import Optional, Tuple, Union


class VideoTokenReduction(nn.Module):
    def __init__(
        self,
        num_frames: int,
        avg_tokens_per_frame: int,
        frame_aspect_ratio: str,
        video_image_token_id: int,
    ):
        """Video frames token reduction: patchwise compression by frame difference.

        Args:
            num_frames: Unused, kept for existing configs; the frame count comes
                from the batch.
            avg_tokens_per_frame: Average number of tokens per frame, all crops of
                the frame included.
            frame_aspect_ratio: Aspect ratio of the frame. With "pad" a budget of a
                whole frame is a config error; anyres frames it covers stay whole.
            video_image_token_id: Token for the video image.
        """
        super().__init__()

        self.avg_tokens_per_frame = avg_tokens_per_frame
        self.frame_aspect_ratio = frame_aspect_ratio
        self.video_image_token_id = video_image_token_id

    def forward(
        self,
        pixel_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        num_crops_per_frame: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, Optional[torch.Tensor]], ...]:
        return self.apply_compression_video_token(
            pixel_embeds=pixel_embeds,
            input_ids=input_ids,
            num_crops_per_frame=num_crops_per_frame,
            labels=labels,
            attention_mask=attention_mask,
        )

    def apply_compression_video_token(
        self,
        pixel_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        num_crops_per_frame: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, Optional[torch.Tensor]], ...]:
        """Keep `avg_tokens_per_frame x T` video tokens, the first frame included.

        Args:
            pixel_embeds: (T * C, P, D) crops of the sample's videos, frame-major.
            input_ids: (1, L) ids with T * C * P video placeholders.
            num_crops_per_frame: (1,) crops per frame C, 0 when the videos of the
                sample differ.
            labels: (1, L), filtered like `input_ids`.
            attention_mask: (1, L), filtered like `input_ids`.
        """
        TC, P, D = pixel_embeds.shape

        # dummy video case, just return
        if TC == 1:
            return pixel_embeds, input_ids, labels, attention_mask, None

        assert input_ids.shape[0] == 1, (
            "Token reduction works only with batch_size = 1, got"
            f" {input_ids.shape[0]=}; set device_train_microbatch_size = 1"
        )

        C = int(num_crops_per_frame.item())
        assert C >= 1, (
            "Token reduction needs the same number of crops per frame for every video"
            " of the sample; keep one video per sample or use frame_aspect_ratio = pad."
        )
        T, PC = TC // C, P * C

        assert self.frame_aspect_ratio != "pad" or self.avg_tokens_per_frame < PC, (
            f"Got avg_tokens_per_frame = {self.avg_tokens_per_frame} >= number of"
            f" tokens in one frame = {PC}, nothing to reduce!"
        )

        num_to_keep = self.avg_tokens_per_frame * T

        if num_to_keep < PC:
            warnings.warn(
                "Token reduction budget avg_tokens_per_frame x frames is below one"
                " frame, such short videos keep their first frame only."
            )

        # The crops of a frame fold into its patch axis: frames compare with frames.
        pixel_embeds = pixel_embeds.view(T, PC, D)
        compression_mask = self._get_patchwise_compression_mask(
            pixel_embeds, num_to_keep
        )

        pixel_embeds, input_ids, labels, attention_mask, text_mask = (
            self._compress_video_tokens(
                compression_mask, pixel_embeds, input_ids, labels, attention_mask
            )
        )

        input_ids = input_ids.unsqueeze(0)

        if labels is not None:
            labels = labels.unsqueeze(0)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)

        return pixel_embeds, input_ids, labels, attention_mask, text_mask

    @torch.no_grad()
    def _get_patchwise_compression_mask(self, pixel_embeds, num_to_keep):
        """Keep the first frame and the most changed patches of the next frames.

        Args:
            pixel_embeds: torch.Tensor of shape (T, PC, D).
            num_to_keep: Number of patches to keep, the first frame included; a
                budget below one frame keeps the first frame only, a budget of T
                frames keeps everything.

        Returns:
            bool mask (T, PC).
        """
        T, PC, D = pixel_embeds.shape
        diffs = (pixel_embeds[1:] - pixel_embeds[:-1]).abs().mean(dim=-1).flatten()
        k = min(max(num_to_keep - PC, 0), diffs.numel())

        mask = torch.zeros_like(diffs, dtype=torch.bool)
        mask[diffs.topk(k).indices] = True
        first_frame = torch.ones(PC, dtype=torch.bool, device=mask.device)

        return torch.cat([first_frame, mask]).view(T, PC)

    def _compress_video_tokens(
        self,
        compression_mask: torch.Tensor,  # (T, PC)
        pixel_embeds: torch.Tensor,  # (T, PC, D)
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        pixel_embeds = pixel_embeds[compression_mask]

        video_selected = input_ids == self.video_image_token_id
        assert video_selected.sum() == compression_mask.numel(), (
            f"Got {video_selected.sum()} video tokens from input_ids before token"
            f" reduction, but compression mask has values for"
            f" {compression_mask.numel()} tokens!"
        )

        text_mask = torch.logical_not(video_selected)

        text_mask[video_selected] = compression_mask.flatten()

        input_ids = input_ids[text_mask]

        if attention_mask is not None:
            attention_mask = attention_mask[text_mask]

        if labels is not None:
            labels = labels[text_mask]
            labels[input_ids == self.video_image_token_id] = -100

        return pixel_embeds, input_ids, labels, attention_mask, text_mask
