from typing import Mapping, Optional

from llmfoundry.data.collators.base_collator import BaseCollator
from llmfoundry.data.collators.text_collator import TextCollator
from llmfoundry.data.processors.base_processor import DataModality
from llmfoundry.data.text_data import FlashAttnCollatorWrapper


class TextFlashAttnCollator(BaseCollator):
    data_modality = DataModality.TEXT

    def __init__(
        self,
        scale: float = 1.0,
        left_padding: bool = False,
        return_attention_mask: Optional[bool] = None,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        boundary_mode: str = "auto",
    ):
        self.base_collator = TextCollator(
            pad_token_id=pad_token_id,
            scale=scale,
            left_padding=left_padding,
            return_attention_mask=return_attention_mask,
        )

        self.wrapped = FlashAttnCollatorWrapper(
            base_collator=self.base_collator,
            eos_token=eos_token_id,
            bos_token=bos_token_id,
            pad_token=pad_token_id,
            boundary_mode=boundary_mode,
        )

    def __call__(self, examples):
        return self.wrapped(examples)

    def split_batch(self, batch: Mapping, microbatch_size: int):
        position_ids = batch.pop("position_ids")
        position_ids_split = position_ids.split(microbatch_size)

        splitted_batch = self.base_collator.split_batch(batch, microbatch_size)

        for i in range(len(splitted_batch)):
            length = splitted_batch[i]["input_ids"].shape[1]
            splitted_batch[i]["position_ids"] = position_ids_split[i][
                :, :length
            ].contiguous()

        return splitted_batch
