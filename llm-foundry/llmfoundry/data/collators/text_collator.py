import torch

from typing import Dict, Mapping, Optional

from composer.utils import dist

from llmfoundry.data.collators.base_collator import BaseCollator
from llmfoundry.data.processors.base_processor import DataModality


class TextCollator(BaseCollator):
    data_modality = DataModality.TEXT

    def __init__(
        self,
        pad_token_id: int = 0,
        scale: float = 1.0,
        left_padding: bool = False,
        return_attention_mask: Optional[bool] = None,
    ):
        """``return_attention_mask=None`` sends a mask only for left padding.

        Under causal attention no real token attends to a trailing pad, so an
        end-padded row needs no mask, and without one the language model skips
        the unpad path (a host sync per attention layer). The pads then cost
        attention compute and enter losses averaged over every position (hidden
        z-loss); ``return_attention_mask: true`` keeps them out.
        """
        self.pad_token_id = pad_token_id
        self.left_padding = left_padding
        self.return_attention_mask = (
            left_padding if return_attention_mask is None else return_attention_mask
        )
        self.divider = dist.get_tp_sp_group_size() or 1
        assert self.divider * scale == int(self.divider * scale), (
            "Devider is expected to be an int value but provided scale makes it non-int. "
            f"Scale={scale}, divider={self.divider}, divider * scale={self.divider * scale}."
        )
        self.divider = int(self.divider * scale)

    def _pad_seq(
        self, seq: torch.Tensor, max_length: int, pad_token_id: int
    ) -> torch.Tensor:
        left_pad, right_pad = 0, max_length - len(seq)

        if self.left_padding:
            left_pad, right_pad = right_pad, left_pad

        seq = torch.nn.functional.pad(seq, (left_pad, right_pad), value=pad_token_id)

        return seq

    def __call__(self, examples):
        input_ids = [sample["input_ids"] for sample in examples]
        labels = [sample["labels"] for sample in examples]
        attention_masks = []

        if self.return_attention_mask:
            for sample, ids in zip(examples, input_ids):
                attention_mask = sample.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(ids, dtype=torch.bool)
                assert attention_mask.ndim == 1 and len(attention_mask) == len(ids), (
                    "attention_mask must be one-dimensional and match input_ids length"
                )
                attention_masks.append(attention_mask.to(dtype=torch.bool))
        elif any(sample.get("attention_mask") is not None for sample in examples):
            raise ValueError(
                "A sample carries its own attention_mask, but the collator sends "
                "none (return_attention_mask is off for end padding by default); "
                "set return_attention_mask: true."
            )

        max_length = max(len(ids) for ids in input_ids)
        max_length = ((max_length + self.divider - 1) // self.divider) * self.divider

        # pad to max length
        input_ids = [
            self._pad_seq(ids, max_length, self.pad_token_id) for ids in input_ids
        ]
        labels = [self._pad_seq(label, max_length, -100) for label in labels]

        batch = {
            "input_ids": torch.stack(input_ids, dim=0),
            "labels": torch.stack(labels, dim=0),
        }

        if self.return_attention_mask:
            attention_masks = [
                self._pad_seq(mask, max_length, False) for mask in attention_masks
            ]
            batch["attention_mask"] = torch.stack(attention_masks, dim=0)

        return batch

    def split_batch(self, batch: Mapping, microbatch_size: int):
        """Split ``batch`` into microbatches cut to their own longest row.

        ``__call__`` pads the whole device batch to its longest row, so without
        the cut a microbatch of a short sample would carry the longest one's
        padding into the model. The cut keeps a multiple of ``divider``.
        """
        assert batch["input_ids"].shape[0] % microbatch_size == 0, (
            f"Batch size {batch['input_ids'].shape[0]} is not divisible by microbatch size {microbatch_size}!"
        )

        num_microbatches = batch["input_ids"].shape[0] // microbatch_size

        input_ids_split = batch["input_ids"].split(microbatch_size)
        labels_split = batch["labels"].split(microbatch_size)

        microbatches = [
            {
                "input_ids": input_ids_split[i],
                "labels": labels_split[i],
            }
            for i in range(num_microbatches)
        ]

        if "attention_mask" in batch:
            attention_mask_split = batch["attention_mask"].split(microbatch_size)
            for i in range(num_microbatches):
                microbatches[i]["attention_mask"] = attention_mask_split[i]

        return [self._trim_padding(microbatch) for microbatch in microbatches]

    def _trim_padding(
        self, microbatch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Drop the trailing columns that are padding in every row.

        A pad is ``pad_token_id`` with label ``-100``, so a token with that id
        keeps its column when it carries a label. A left-padded or pad-only
        microbatch is returned as is.
        """
        is_pad = (microbatch["input_ids"] == self.pad_token_id) & (
            microbatch["labels"] == -100
        )
        length = is_pad.shape[1]
        positions = torch.arange(1, length + 1, device=is_pad.device)
        real_length = int(torch.where(is_pad.all(dim=0), 0, positions).max())
        trimmed_length = -(-real_length // self.divider) * self.divider

        if real_length == 0 or trimmed_length >= length:
            return microbatch

        return {
            key: value[:, :trimmed_length].contiguous()
            for key, value in microbatch.items()
        }
