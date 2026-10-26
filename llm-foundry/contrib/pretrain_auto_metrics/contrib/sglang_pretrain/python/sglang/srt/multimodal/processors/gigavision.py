from __future__ import annotations


import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.gigavision import GigaVisionForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class GigaVisionImageProcessor(BaseMultimodalProcessor):
    models = [GigaVisionForCausalLM]
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # _processor is GigaVisionProcessor from hf checkpoint
        # _processor.tokenizer should be a valid field
        # it is set while loading with `from_pretrained`
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # transformers>=5 renamed PIL-backend image processors (loaded with
        # use_fast=False / --disable-fast-image-processor), e.g.
        # CLIPImageProcessor -> CLIPImageProcessorPil, while checkpoint remote
        # code (mm_utils.get_crop_size) dispatches on the v4 class name.
        # Re-class the instance under the old name; behavior is unchanged.
        image_processor = getattr(self._processor, "image_processor", None)
        cls_name = type(image_processor).__name__
        if image_processor is not None and cls_name.endswith("Pil"):
            image_processor.__class__ = type(
                cls_name[: -len("Pil")], (type(image_processor),), {}
            )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=self._processor.image_token_id,
        ).build(self._processor)

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        # Load data, set one multimodal token per image
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
        )

        # no images data, return None
        if base_output.images is None or len(base_output.images) == 0:
            return None

        if isinstance(input_text, str):
            input_ids = self._processor.tokenizer(
                input_text, return_tensors="pt"
            )["input_ids"].flatten()
        else:
            input_ids = torch.as_tensor(input_text)

        # Logic from GigaVisionProcessor.preprocess
        input_ids, _, image_tensors = self._processor.preprocess(
            input_ids=input_ids, pil_images=base_output.images
        )
        # Prepare multimodal items in sglang format
        image_offsets = self.get_mm_items_offset(
            input_ids=input_ids, mm_token_id=self.mm_tokens.image_token_id
        )
        items = [
            MultimodalDataItem(
                feature=image_tensors,
                modality=Modality.IMAGE,
                offsets=image_offsets,
            )
        ]

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=items,
            im_token_id=self.mm_tokens.image_token_id,
        )
