from __future__ import annotations

import torch
import torch.nn as nn

from typing import Dict, List, Tuple


from ..base_component import BaseOmniComponent


class StackingSubsampler(nn.Module):
    def __init__(self, subsampling_factor: int) -> None:
        super().__init__()

        self.subsampling_factor = subsampling_factor

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/subsampling.py#L25
        b, t, h = x.size()

        mask = torch.arange(t, device=x.device).unsqueeze(0).unsqueeze(-1).expand(b, t, h)
        mask = mask >= lengths.view(b, 1, 1)
        x = x.clone()
        x[mask] = 0

        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        lengths = torch.div(lengths + self.subsampling_factor - 1, self.subsampling_factor, rounding_mode='floor')
        return x, lengths


class AudioLinearAdapterWithStackingSubsampler(nn.Module, BaseOmniComponent):
    def __init__(
        self,
        ae_hidden_size: int,
        llm_hidden_size: int,
        subsampling_factor: int,
    ) -> None:
        super().__init__()

        self.ae_hidden_size = ae_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.subsampling_factor = subsampling_factor

        self.subsampler = StackingSubsampler(self.subsampling_factor)
        self.lin_proj = nn.Linear(self.ae_hidden_size * self.subsampling_factor, self.llm_hidden_size)

    def forward(self, inputs: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
        spectrograms, spectrogram_lengths, sizes_list = self._concat_inputs(inputs)
        spectrograms, spectrogram_lengths = self.subsampler(spectrograms, spectrogram_lengths)
        spectrograms = self.lin_proj(spectrograms)
        return self._split_outputs(spectrograms, spectrogram_lengths, sizes_list)

    def _concat_inputs(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for input_item in inputs:
            assert "spectrograms" in input_item and "spectrogram_lengths" in input_item, (
                f"`spectrograms` and `spectrogram_lengths` is required, but not found in input_item: {input_item.keys()}"
            )

        sizes_list = [len(input_item["spectrogram_lengths"]) for input_item in inputs]
        spectrograms = torch.cat([input_item["spectrograms"] for input_item in inputs], dim=0)
        spectrogram_lengths = torch.cat([input_item["spectrogram_lengths"] for input_item in inputs], dim=0)
        return spectrograms, spectrogram_lengths, sizes_list

    def _split_outputs(self, spectrograms: torch.Tensor, spectrogram_lengths: torch.Tensor, sizes_list: List[int]) -> Tuple[torch.Tensor, ...]:
        spectrograms = self._reshape_spectrograms_to_unbatched(spectrograms, spectrogram_lengths)
        spectrogram_lengths_splitted = torch.split(spectrogram_lengths, sizes_list)
        spectrogram_sizes = [lengths.sum().item() for lengths in spectrogram_lengths_splitted]
        spectrograms_splitted = torch.split(spectrograms, spectrogram_sizes)

        outputs = []
        for spec, lengths in zip(spectrograms_splitted, spectrogram_lengths_splitted):
            outputs.extend([spec, lengths])
        return tuple(outputs)

    def _reshape_spectrograms_to_unbatched(
        self,
        spectrograms: torch.Tensor,
        spectrogram_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([
            spectrogram[:length]
            for spectrogram, length in zip(spectrograms.unbind(dim=0), spectrogram_lengths)
        ], dim=0)

    def unwrap_tuple_output(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        outputs: Tuple[torch.Tensor, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        return [{"audio_embeds": outputs[idx]} for idx in range(0, len(outputs), 2)]
