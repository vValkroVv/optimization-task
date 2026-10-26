import torch

from typing import Dict, List, Mapping, Optional, Union

from llmfoundry.data.collators.base_collator import BaseCollator
from llmfoundry.data.processors.base_processor import DataModality


class VideoCollator(BaseCollator):
    data_modality = DataModality.VIDEO

    def __call__(
        self, examples: List[Optional[Dict[str, torch.Tensor]]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        n_crops_per_sample = [
            example["pixel_values"].shape[0] if example else 0 for example in examples
        ]
        if sum(n_crops_per_sample) == 0:
            return {}

        processor_fallback_count = sum(
            example["processor_fallback_count"] if example else 0
            for example in examples
        )

        # Read by the token reduction only.
        num_crops_per_frame = [
            example["num_crops_per_frame"] if example else 0 for example in examples
        ]

        return {
            "pixel_values": torch.cat(
                [example["pixel_values"] for example in examples if example]
            ),
            "n_crops_per_sample": torch.tensor(n_crops_per_sample, dtype=torch.long),
            "num_crops_per_frame": torch.tensor(num_crops_per_frame, dtype=torch.long),
            "processor_fallback_count": processor_fallback_count,
        }

    @staticmethod
    def split_batch(batch: Mapping, microbatch_size: int):
        assert batch["input_ids"].shape[0] % microbatch_size == 0, (
            f"Batch size {batch['input_ids'].shape[0]} is not divisible by microbatch size {microbatch_size}!"
        )

        num_microbatches = batch["input_ids"].shape[0] // microbatch_size

        data_modality_str = VideoCollator.data_modality.value
        if data_modality_str not in batch["modality_inputs"]:
            return [{} for _ in range(num_microbatches)]

        modality_batch = batch["modality_inputs"][data_modality_str]

        num_crops_per_image_split = modality_batch["n_crops_per_sample"].split(
            microbatch_size
        )
        num_crops_per_microbatch = [
            num_crops.sum().item() for num_crops in num_crops_per_image_split
        ]
        pixel_values_split = modality_batch["pixel_values"].split(
            num_crops_per_microbatch
        )
        num_crops_per_frame_split = modality_batch["num_crops_per_frame"].split(
            microbatch_size
        )

        return [
            {
                data_modality_str: {
                    "pixel_values": pixel_values_split[i],
                    "n_crops_per_sample": num_crops_per_image_split[i],
                    "num_crops_per_frame": num_crops_per_frame_split[i],
                }
            }
            if num_crops_per_microbatch[i] > 0
            else {}
            for i in range(num_microbatches)
        ]
