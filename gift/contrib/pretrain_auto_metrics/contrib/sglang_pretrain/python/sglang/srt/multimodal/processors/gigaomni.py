from __future__ import annotations

import uuid
import torch

from pathlib import Path
from transformers import AutoProcessor
from typing import Any, Dict, List, Optional

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.gigaomni import GigaOmniForCausalLM
from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)


class GigaOmniMultiModalProcessor(BaseMultimodalProcessor):
    models = [GigaOmniForCausalLM]
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.processors = self._load_processors()
        # any checkpoint processor carries the tokenizer, fall back to the model one
        self.tokenizer = next(
            (p.tokenizer for p in self.processors.values() if hasattr(p, "tokenizer")),
            self._processor.tokenizer,
        )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=self.processors["image"].image_token_id if "image" in self.processors else None,
            video_token_id=self.processors["video"].video_image_token_id if "video" in self.processors else None,
            audio_token_id=self.processors["audio"].audio_token_id if "audio" in self.processors else None,
        ).build(self)

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[MultimodalProcessorOutput]:
        video_data = getattr(request_obj, "video_data", None)
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=getattr(self.processors.get("audio"), "sample_rate", None),
        )

        if not base_output.images and not base_output.videos and not base_output.audios:
            return None

        input_ids = self._tokenize_input(base_output.input_text)

        outputs = []
        for modality_str, processor in self.processors.items():
            files = self._build_files_for_processor(modality_str, base_output)

            if not files:
                continue

            processor_output = processor(files=files, input_ids=input_ids)
            input_ids = processor_output["text"]["input_ids"]
            outputs.append((processor, processor_output))

        # offsets must be computed on final input_ids, later processors shift positions
        items = [
            self._build_mm_data_item_for_processor_output(input_ids, processor, processor_output)
            for processor, processor_output in outputs
        ]

        return MultimodalProcessorOutput(
            mm_items=items,
            input_ids=input_ids.tolist(),
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
        )

    def _load_processors(self):
        processors = {}

        assert self.server_args.trust_remote_code, "Gigaomni processors require --trust-remote-code"
        processors_dir = Path(self.server_args.model_path) / "processors"
        assert processors_dir.exists(), f"Processors directory {processors_dir} does not exist."

        for processor_dir in processors_dir.iterdir():
            if not processor_dir.is_dir():
                continue

            processor = AutoProcessor.from_pretrained(
                processor_dir, trust_remote_code=self.server_args.trust_remote_code
            )
            assert hasattr(processor, "data_modality"), f"Processor {processor_dir} has no data_modality attribute."
            processors[processor.data_modality.value] = processor

        return processors

    def _tokenize_input(self, input_text) -> list[int]:
        if isinstance(input_text, str):
            return self._processor.tokenizer(input_text, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].flatten()
        return torch.as_tensor(input_text).flatten()

    @classmethod
    def _get_content_for_modality(cls, modality_str: str, base_output: BaseMultiModalProcessorOutput) -> Any:
        if modality_str == "image":
            return base_output.images
        elif modality_str == "video":
            return base_output.videos
        elif modality_str == "audio":
            return base_output.audios
        else:
            raise ValueError(f"Modality `{modality_str}` is not supported right now. Please add support for it in `_get_content_for_modality`.")

    @classmethod
    def _generate_file_id(cls, modality_str: str, index: int) -> str:
        file_id_namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        return str(uuid.uuid5(file_id_namespace, f"{modality_str}_{index}"))

    @classmethod
    def _build_files_for_processor(cls, modality_str: str, base_output: BaseMultiModalProcessorOutput) -> List[Dict[str, Any]]:
        contents = cls._get_content_for_modality(modality_str, base_output)
        return [{
            "id": cls._generate_file_id(modality_str, idx),
            "name": f"file",
            "type": modality_str,
            "content": content,
            "embed": True,
        } for idx, content in enumerate(contents)]

    def _build_mm_data_item_for_processor_output(self, input_ids: List[int], processor: AutoProcessor, processor_output: Dict[str, Any]) -> MultimodalDataItem:
        modality_str = processor.data_modality.value

        offsets = self.get_mm_items_offset(
            input_ids=input_ids,
            mm_token_id=self.mm_tokens.get_token_id_by_modality(Modality.from_str(modality_str))
        )

        if "processor_fallback_count" in processor_output[modality_str]:
            del processor_output[modality_str]["processor_fallback_count"]

        return MultimodalDataItem(
            modality=Modality.from_str(modality_str),
            hash=self._hash_processor_output(processor_output[modality_str]),
            feature=processor_output[modality_str],
            offsets=offsets,
            model_specific_data=processor_output[modality_str],
        )

    def _hash_processor_output(self, processor_output: Dict[str, Any]) -> str:
        combined_tensor = torch.tensor([], device="cpu", dtype=torch.float32)
        for _, value in processor_output.items():
            if isinstance(value, torch.Tensor):
                combined_tensor = torch.cat([combined_tensor, value.flatten().to(device="cpu", dtype=torch.float32)])
        return hash_feature(combined_tensor)
