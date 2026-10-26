import io
import cv2
import torch
import logging
import numpy as np

from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union, Dict, Any, Literal, Tuple
from transformers import BaseImageProcessor, AutoProcessor, AutoImageProcessor

from .base_processor import BaseProcessor, DataModality
from .mm_utils import (
    expand2square,
    process_anyres_image,
    process_anyres_image_onevision,
    process_qwen_image,
    get_crop_size,
    smart_resize,
    duplicate_modality_tokens,
    get_anyres_image_grid_shape,
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s |  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class GigaOmniVisionProcessor(BaseProcessor):
    data_modality = DataModality.IMAGE
    attributes = []
    valid_kwargs = [
        "image_processor",
        "image_aspect_ratio",
        "num_image_tokens_per_crop",
        "image_grid_pinpoints",
        "image_token_id",
        "qwen_max_image_tokens",
        "tokenizer",
    ]

    def __init__(
        self,
        image_processor: Optional[Union[str, BaseImageProcessor]] = None,
        image_aspect_ratio: Literal[
            "anyres", "anyres_onevision", "qwen", "pad"
        ] = "anyres",
        num_image_tokens_per_crop: Optional[int] = None,
        image_grid_pinpoints: Union[
            str, List[Tuple[int, int]]
        ] = "[[448, 448], [448, 896], [896, 448], [896, 896], [1344, 448], [448, 1344]]",
        image_token_id: int = 128005,
        qwen_max_image_tokens: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(image_processor, BaseImageProcessor) or image_processor is None:
            self.image_processor = image_processor
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(image_processor)

        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints
        self.image_token_id = image_token_id

        if image_aspect_ratio == "qwen":
            assert (
                num_image_tokens_per_crop == 1 or num_image_tokens_per_crop is None
            ), (
                f"When image_aspect_ratio is 'qwen', num_image_tokens_per_crop must be 1 or None, got {num_image_tokens_per_crop}"
            )
            num_image_tokens_per_crop = 1
            if qwen_max_image_tokens is not None:
                assert qwen_max_image_tokens >= 4, (
                    f"qwen_max_image_tokens must be at least 4, got {qwen_max_image_tokens}"
                )
        else:
            assert qwen_max_image_tokens is None, (
                "qwen_max_image_tokens can only be used when image_aspect_ratio is 'qwen'"
            )
            assert num_image_tokens_per_crop and num_image_tokens_per_crop > 0, (
                f"num_image_tokens_per_crop must be provided and greater than 0 when image_aspect_ratio is not 'qwen', got {num_image_tokens_per_crop}"
            )

        self.num_image_tokens_per_crop = num_image_tokens_per_crop
        self.qwen_max_image_tokens = qwen_max_image_tokens

        self.auto_map = {
            "AutoProcessor": "gigaomni_vision_processor.GigaOmniVisionProcessor",
        }

    def __call__(
        self,
        files: Optional[List[Dict[str, Any]]] = None,
        dataset_sample: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess images and insert image tokens into the text input_ids and labels.
        You can provide dataset_sample in the form of:
        {
            "text": {
                "input_ids": torch.LongTensor,
                "labels": torch.LongTensor
            }
        } or input_ids and labels as separate arguments.
        """
        self._processor_fallback_count = 0

        file_contents = self._prepare_file_contents(files)
        dataset_sample = self._prepare_dataset_sample(
            dataset_sample, input_ids, labels, attention_mask
        )

        if len(file_contents) == 0:
            return dataset_sample

        preprocessed = self._preprocess_images(file_contents)
        image_lengths = [
            num_crops * self.num_image_tokens_per_crop
            for num_crops in preprocessed["num_crops_per_image"]
        ]

        dataset_sample["text"]["input_ids"], dataset_sample["text"]["labels"] = (
            duplicate_modality_tokens(
                dataset_sample["text"]["input_ids"],
                dataset_sample["text"].get("labels", None),
                self.image_token_id,
                image_lengths,
            )
        )

        dataset_sample[self.data_modality.value] = {
            "pixel_values": preprocessed["pixel_values"],
            "processor_fallback_count": self._processor_fallback_count,
        }

        if self.image_aspect_ratio == "qwen":
            dataset_sample[self.data_modality.value]["grid_thw"] = preprocessed[
                "grid_thw"
            ]

        return dataset_sample

    def get_num_tokens(
        self,
        files: Optional[List[Dict[str, Any]]] = None,
        hw_grid: Optional[List[Tuple[int, int]]] = None,
    ) -> int:
        assert files or hw_grid, "Either files or hw_grid must be provided!"

        image_sizes = []
        if files is not None:
            image_sizes = [image.size for image in self._prepare_file_contents(files)]

        if hw_grid is not None:
            image_sizes += [(hw[0], hw[1]) for hw in hw_grid]

        return sum(
            self._get_num_crops(image_size) * self.num_image_tokens_per_crop
            for image_size in image_sizes
        )

    def save_pretrained(self, save_directory: str, **kwargs):  # type: ignore
        if self.image_processor is not None:
            self.image_processor.save_pretrained(save_directory, **kwargs)  # type: ignore

        output = super().save_pretrained(save_directory, **kwargs)
        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):  # type: ignore
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        image_processor: BaseImageProcessor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        processor.image_processor: BaseImageProcessor = image_processor
        return processor

    def _prepare_file_contents(
        self, files: Optional[List[Dict[str, Any]]] = None
    ) -> List[Image.Image]:
        if not files:
            return []

        file_contents = []
        for file in files:
            if file["type"] != self.data_modality.value:
                continue

            if "content" in file and isinstance(file["content"], Image.Image):
                image = file["content"]
            elif "content" in file and isinstance(file["content"], bytes):
                image = self._read_image_from_bytes(file["content"])
            else:
                image = self._read_image_from_path(file["path"])

            file_contents.append(image)

        return file_contents

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            return bg
        return image.convert("RGB")

    def _get_black_image(self) -> Image.Image:
        crop_size = get_crop_size(self.image_processor)
        return Image.new("RGB", (crop_size, crop_size))

    def _read_image_from_bytes(self, raw_bytes: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(raw_bytes)) as image:
                return self._to_rgb(image.copy())
        except (OSError, UnidentifiedImageError):
            logger.warning(
                "Image can't be loaded with Image.open(), using cv2.imdecode() instead!"
            )

        try:
            np_image = cv2.imdecode(
                np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(np.uint8(np_image))
        except Exception as exc:
            logger.error(
                f"Image can't be loaded with cv2.imdecode(), returning black image! Error:\n{exc}"
            )
            self._processor_fallback_count += 1

        return self._get_black_image()

    def _read_image_from_path(self, image_path: str) -> Image.Image:
        try:
            with Image.open(image_path) as image:
                return self._to_rgb(image.copy())
        except (OSError, UnidentifiedImageError):
            logger.warning(
                f"Image {image_path} can't be loaded with Image.open(), using cv2.imread() instead!"
            )

        try:
            np_image = cv2.imread(image_path)
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)  # type: ignore
            return Image.fromarray(np.uint8(np_image))
        except Exception as exc:
            logger.error(
                f"Image {image_path} can't be loaded with cv2.imread(), returning black image! Error:\n{exc}"
            )
            self._processor_fallback_count += 1

        return self._get_black_image()

    def _preprocess_single_image(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """Crop tensor `[num_crops, C, H, W]`, crop count and (qwen only) grid_thw for one image."""
        grid_thw_i = None

        if self.image_aspect_ratio == "pad":
            image = expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean)
            )  # type: ignore
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ]  # type: ignore
        if self.image_aspect_ratio == "anyres":
            image_tensor = process_anyres_image(
                image, self.image_processor, self.image_grid_pinpoints
            )
        if self.image_aspect_ratio == "anyres_onevision":
            image_tensor = process_anyres_image_onevision(
                image, self.image_processor, self.image_grid_pinpoints
            )
        if self.image_aspect_ratio == "qwen":
            qwen_max_pixels = None
            if self.qwen_max_image_tokens is not None:
                factor = (
                    self.image_processor.patch_size * self.image_processor.merge_size
                )
                qwen_max_pixels = self.qwen_max_image_tokens * factor**2
            image_tensor, num_crops, grid_thw_i = process_qwen_image(
                image, self.image_processor, max_pixels=qwen_max_pixels
            )

        if self.image_aspect_ratio != "qwen":
            num_crops = image_tensor.shape[0]

        return image_tensor, num_crops, grid_thw_i

    def _preprocess_images(
        self, images: List[Image.Image], num_threads: int = 1
    ) -> Dict[str, Any]:
        if num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                preprocessed = list(pool.map(self._preprocess_single_image, images))
        else:
            preprocessed = [self._preprocess_single_image(image) for image in images]

        assert preprocessed, "Image tensors is empty, it shouldn't be empty"

        image_tensors, num_crops_per_image, grids_thw = zip(*preprocessed)
        grids_thw = [grid_thw for grid_thw in grids_thw if grid_thw is not None]
        return {
            # bf16 halves the pixel bytes that cross worker -> trainer (/dev/shm,
            # pinned memory, H2D); under amp_bf16 the patch-embed conv rounds its
            # input to bf16 anyway, so the encoder sees the same values
            "pixel_values": torch.vstack(image_tensors).to(torch.bfloat16),
            "num_crops_per_image": list(num_crops_per_image),
            "grid_thw": torch.cat(grids_thw, dim=0) if grids_thw else None,
        }

    def _get_num_crops(self, image_size: Tuple[int, int]) -> int:
        if self.image_aspect_ratio == "pad":
            return 1

        if self.image_aspect_ratio in ("anyres", "anyres_onevision"):
            patch_size = get_crop_size(self.image_processor)
            shape_x, shape_y = get_anyres_image_grid_shape(
                image_size, self.image_grid_pinpoints, patch_size
            )
            return 1 + shape_x * shape_y

        if self.image_aspect_ratio == "qwen":
            width, height = image_size
            factor = self.image_processor.patch_size * self.image_processor.merge_size  # type: ignore
            qwen_max_pixels = None
            if self.qwen_max_image_tokens is not None:
                qwen_max_pixels = self.qwen_max_image_tokens * factor**2
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=factor,
                max_pixels=qwen_max_pixels,
            )
            return (resized_height // factor) * (resized_width // factor)

        raise ValueError(f"Unknown image_aspect_ratio: {self.image_aspect_ratio}")


GigaOmniVisionProcessor.register_for_auto_class(AutoProcessor)
