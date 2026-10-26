"""Batch containers exchanged between the kiwi dataloader and the kiwi model.

TODO: this is the model's input contract, not dataloader internals. Move it
into the kiwi model package once that model lands in this repository, and
leave the dataloader importing it from there.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import accumulate

import torch

from .constants import ENCODER_MODALITY_MAPPING, ENCODER_ORDER
from .magi_ranges import MagiRanges


class MirroredTensor:
    """A tensor wrapper that's on CPU and GPU at the same time."""

    __slots__ = ("cpu", "cuda", "device")

    def __init__(self, data: torch.Tensor, device: torch.device | str | None = None):
        self.cpu: torch.Tensor = data.detach().cpu()
        self.cuda: torch.Tensor | None = None
        self.device: torch.device | None = None
        if device is not None:
            self.to(device)

    def to(self, device: torch.device | str) -> MirroredTensor:
        device = torch.device(device)
        if device.type == "cpu":
            return self
        if self.device != device:
            self.cuda = self.cpu.to(device)
            self.device = device
        return self


@dataclass
class ModalityData:
    tokens: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    target_mask: torch.Tensor
    span_ids: torch.Tensor

    @classmethod
    def merge(cls, *inputs: ModalityData) -> ModalityData:
        if len(inputs) == 1:
            return inputs[0]
        tokens = torch.cat([x.tokens for x in inputs])
        position_ids = torch.cat([x.position_ids for x in inputs])
        cu_parts = [inputs[0].cu_seqlens]
        for inp in inputs[1:]:
            cu_parts.append(inp.cu_seqlens[1:] + cu_parts[-1][-1])
        target_mask = torch.cat([x.target_mask for x in inputs])
        span_ids = torch.cat([x.span_ids for x in inputs])
        return cls(tokens, position_ids, torch.cat(cu_parts), target_mask, span_ids)

    def to(self, device: torch.device | str | None = None) -> ModalityData:
        return ModalityData(
            self.tokens.to(device),
            self.position_ids.to(device),
            self.cu_seqlens.to(device),
            self.target_mask.to(device),
            self.span_ids.to(device),
        )


@dataclass
class ModelBatchData:
    position_ids: torch.Tensor
    magi_ranges: MagiRanges
    span_properties: list[set[str]]
    text: ModalityData | None = None
    semantic_visual: ModalityData | None = None
    semantic_audio: ModalityData | None = None
    pixel_visual: ModalityData | None = None
    pixel_audio: ModalityData | None = None
    gen_visual: ModalityData | None = None
    gen_audio: ModalityData | None = None

    def to(self, device: torch.device | str | None = None) -> ModelBatchData:
        values = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, "to"):
                value = value.to(device)
            values[f.name] = value
        return ModelBatchData(**values)

    def num_tokens(self) -> int:
        """Total packed sequence length of the batch."""
        return int(self.position_ids.shape[0])


class BatchLayout:
    ZERO_TIMESTEPS = torch.empty(0)

    def __init__(self, **inputs: ModalityData | None) -> None:
        self.field_sizes: dict[str, int] = {}
        self.field_offsets: dict[str, tuple[int, int]] = {}
        self.field_num_seqs: dict[str, int] = {}
        pos = 0
        for name in ENCODER_ORDER:
            modality = inputs.get(name)
            if modality is not None:
                size = modality.tokens.shape[0]
                self.field_sizes[name] = size
                self.field_offsets[name] = (pos, pos + size)
                self.field_num_seqs[name] = modality.cu_seqlens.shape[0] - 1
                pos += size

    def merge_encoder_inputs(
        self, **modality_fields: ModalityData | None
    ) -> ModalityData | None:
        parts: list[ModalityData] = []
        for name in ENCODER_ORDER:
            data = modality_fields.get(name)
            if data is not None:
                parts.append(data)
        if not parts:
            return None
        return ModalityData.merge(*parts)

    def get_encoder_timesteps(
        self,
        device: torch.device | str | None = None,
        **modality_fields: torch.Tensor | None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for name in ENCODER_ORDER:
            timesteps = modality_fields.get(name)
            if timesteps is None or name not in self.field_num_seqs:
                continue
            elif timesteps is self.ZERO_TIMESTEPS:
                parts.append(torch.zeros(self.field_num_seqs[name], device=device))
            else:
                parts.append(timesteps)
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    def build_model_sequence(
        self, encoder_outputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.cat(
            [encoder_outputs[name] for name in ENCODER_ORDER if name in encoder_outputs]
        )

    def get_cu_modlens(
        self, num_parallel_layers: int, device: torch.device | str
    ) -> MirroredTensor:
        # Assign each distinct modality a fixed slot in order of first appearance
        # in ENCODER_ORDER (text -> 0, visual -> 1, audio -> 2, ...).
        modality_slot: dict[str, int] = {}
        for name in ENCODER_ORDER:
            mod = ENCODER_MODALITY_MAPPING[name]
            if mod not in modality_slot:
                modality_slot[mod] = len(modality_slot)

        # One entry per parallel layer. A modality whose natural slot is beyond the
        # available layers collapses into the last layer.
        modlens_list: list[int] = [0] * num_parallel_layers
        for name in ENCODER_ORDER:
            size = self.field_sizes.get(name, 0)
            if not size:
                continue
            slot = min(
                modality_slot[ENCODER_MODALITY_MAPPING[name]], num_parallel_layers - 1
            )
            modlens_list[slot] += size

        modlens = torch.tensor(
            list(accumulate(modlens_list, initial=0)), dtype=torch.int32
        )
        return MirroredTensor(modlens, device=device)

    def slice_sequence(
        self, hidden_states: torch.Tensor, field_name: str
    ) -> torch.Tensor:
        start, end = self.field_offsets[field_name]
        return hidden_states[start:end]
