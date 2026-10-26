import re
import ast
import math
import torch
import numpy as np
import typing as tp

from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import BaseImageProcessor


_QWEN_IMAGE_MIN_TOKEN_NUM = 4
_QWEN_IMAGE_MAX_TOKEN_NUM = 16384
_QWEN_MAX_ASPECT_RATIO = 200


def select_best_resolution(
    original_size: Tuple[int, int], possible_resolutions: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = (0, 0)
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = (
            int(original_width * scale),
            int(original_height * scale),
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def get_crop_size(processor: BaseImageProcessor) -> int:
    if processor.__class__.__name__ == "CLIPImageProcessor":
        return processor.crop_size["height"]  # type: ignore
    if processor.__class__.__name__ == "SiglipImageProcessor":
        return processor.size["height"]  # type: ignore
    if processor.__class__.__name__ in {
        "Qwen2VLImageProcessor",
        "Qwen2VLImageProcessorFast",
    }:
        return processor.merge_size * processor.patch_size  # type: ignore
    raise ValueError(f"Unknown image_processor class: {processor.__class__.__name__}")


def process_anyres_image_onevision(
    image: Image.Image,
    processor: BaseImageProcessor,
    grid_pinpoints: Union[List[Tuple[int, int]], str],
):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (Union[List[Tuple[int, int]], str]): A list of possible resolutions or a string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    patch_size = get_crop_size(processor)

    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        OV_PATCH_SIZES = (224, 336, 384, 448, 512)
        assert patch_size in OV_PATCH_SIZES, f"patch_size should be in {OV_PATCH_SIZES}"

        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints: List[Tuple[int, int]] = [
            (i * patch_size, j * patch_size)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]

    if isinstance(grid_pinpoints, list):
        possible_resolutions = grid_pinpoints
    elif isinstance(grid_pinpoints, str):
        possible_resolutions = ast.literal_eval(grid_pinpoints)

    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, patch_size)

    image_original_resize = image.resize((patch_size, patch_size))

    image_patches = [image_original_resize] + patches
    image_patches = [
        processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0]  # type: ignore
        for image_patch in image_patches
    ]

    return torch.stack(image_patches, dim=0)


def process_anyres_image(
    image: Image.Image,
    processor: BaseImageProcessor,
    grid_pinpoints: Union[List[Tuple[int, int]], str],
):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (Union[List[Tuple[int, int]], str]): A list of possible resolutions or a string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if isinstance(grid_pinpoints, list):
        possible_resolutions = grid_pinpoints
    elif isinstance(grid_pinpoints, str):
        possible_resolutions = ast.literal_eval(grid_pinpoints)

    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patch_size = get_crop_size(processor)
    patches = divide_to_patches(image_padded, patch_size)
    image_original_resize = image.resize((patch_size, patch_size))
    image_patches = [image_original_resize] + patches
    image_patches = [
        processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0]  # type: ignore
        for image_patch in image_patches
    ]

    return torch.stack(image_patches, dim=0)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


DTYPE2ENCODING_MAPPING_DICT = {
    np.uint16: 0,
    np.uint32: 1,
    np.int64: 2,
}

ENCODING2DTYPE_MAPPING_DICT = {v: k for k, v in DTYPE2ENCODING_MAPPING_DICT.items()}


def get_decoded_str_from_deserialized_bytes(deserialized_bytes):
    decoded_string = "".join([chr(code) for code in deserialized_bytes]).strip()
    return decoded_string


def read_binary_tokenized_sample(
    max_seq_len, sample: Dict[str, Any], field: str = "tokens"
):
    # take dtype or use np.int64
    sample_dtype_code = sample["dtype"] if "dtype" in sample else 2
    assert sample_dtype_code in ENCODING2DTYPE_MAPPING_DICT, (
        f"Use unsupported dtype code{sample_dtype_code}"
    )
    tokens_dtype = ENCODING2DTYPE_MAPPING_DICT[sample_dtype_code]
    return torch.from_numpy(
        np.frombuffer(sample[field], dtype=tokens_dtype)[:max_seq_len]
        .copy()
        .astype(np.int64)
    )


def duplicate_modality_tokens(
    input_ids: torch.Tensor,
    labels: tp.Optional[torch.Tensor],
    modality_token_id: int,
    modality_lengths: tp.Union[torch.Tensor, tp.List[int]],
) -> tp.Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Duplicates tokens with `modality_token_id` in input_ids and labels using `modality_lengths`.
    Args:
        input_ids (torch.Tensor): Input tokens with shape [seq_len].
        labels (Optional[torch.Tensor]): Input labels with shape [seq_len].
        modality_token_id (int): Id of token that will be replicated.
        modality_lengths (Union[torch.Tensor, list[int]]): Sequence of number of times that
            each token with `modality_token_id` will be replicated.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: input_ids and labels with duplicated tokens.
    """
    if isinstance(modality_lengths, torch.Tensor):
        modality_lengths = modality_lengths.tolist()

    assert isinstance(modality_lengths, list) and not isinstance(
        modality_lengths[0], list
    ), "`modality_lengths` must be 1-d array!"

    modality_token_positions = (
        (input_ids == modality_token_id).nonzero(as_tuple=True)[0].tolist()
    )

    err_msg = (
        f"Number of occurances of `modality_token_id={modality_token_id}` in input_ids must be equal to "
        f"modality_lenghts size but got {len(modality_token_positions)} and {len(modality_lengths)}!"
    )
    assert len(modality_token_positions) == len(modality_lengths), err_msg

    input_ids = duplicate_tokens_on_positions(
        input_ids,
        modality_lengths,
        modality_token_positions,
    )

    if labels is not None:
        labels = duplicate_tokens_on_positions(
            labels,
            modality_lengths,
            modality_token_positions,
        )

    return input_ids, labels


def duplicate_tokens_on_positions(
    tokens: torch.Tensor,
    duplication_factors: tp.List[int],
    positions: tp.List[int],
) -> torch.Tensor:
    """
    Duplicates tokens with `modality_token_id` in the token sequence using `token_id_to_duplicate_positions` and `lengths`.
    """
    assert len(duplication_factors) == len(positions), (
        "Lenght of `duplication_factors` must be equal to lenght of `positions`"
    )

    token_ids_to_duplicate = tokens[positions]
    tokens_sequeses = [
        torch.full(
            (dup_factor,), token_to_dup, dtype=tokens.dtype, device=tokens.device
        )
        for dup_factor, token_to_dup in zip(duplication_factors, token_ids_to_duplicate)
    ]

    return insert_tokens_sequences_on_positions(tokens, tokens_sequeses, positions)


def insert_tokens_sequences_on_positions(
    tokens: torch.Tensor,
    tokens_sequences: tp.List[torch.Tensor],
    positions: tp.List[int],
    remove_tokens_on_positions: bool = True,
) -> torch.Tensor:
    """
    Inserts `tokens_sequences` into the `tokens` at the specified `positions`.
    """
    assert len(tokens_sequences) == len(positions), (
        "Length of `tokens_sequences` must be equal to length of `positions`"
    )

    result_tokens = []
    prev_position = 0

    for position, tokens_sequence in zip(positions, tokens_sequences):
        result_tokens.append(
            torch.cat([tokens[prev_position:position], tokens_sequence])
        )
        prev_position = position + int(remove_tokens_on_positions)

    result_tokens.append(tokens[prev_position:])
    result_tokens = torch.cat(result_tokens)
    return result_tokens


def predict_spec_length(init_length: int, projector_subsampling_factor: int = 4) -> int:
    kernel_size = 5
    left_padding = (kernel_size - 1) // 2
    right_padding = (kernel_size - 1) // 2
    stride = 2
    ceil_mode = False
    repeat_num = 2
    encoded_length = calc_length(
        lengths=torch.tensor([init_length]),
        all_paddings=left_padding + right_padding,
        kernel_size=kernel_size,
        stride=stride,
        ceil_mode=ceil_mode,
        repeat_num=repeat_num,
    )[0]
    projected_length = math.ceil(encoded_length / projector_subsampling_factor)
    return projected_length


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


def seconds_to_hms(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_inter_timings_and_create_times(
    times: tp.List[str],
    duration: int,
) -> tp.List[int]:
    duration_mapper = {"s": 1, "m": 60, "h": 3600}
    set_times = set()
    for timing in times:
        if (
            not isinstance(timing, str)
            or len(timing) < 2
            or timing[-1] not in duration_mapper
        ):
            raise ValueError(f"Invalid timing value: {timing}")
        step = int(timing[:-1]) * duration_mapper[timing[-1]]
        if step <= 0:
            raise ValueError(f"Timing interval must be positive, got: {timing}")
        set_times |= set(range(0, duration, step))
    return sorted(list(set_times))


def _round_by_factor(number: Union[int | float], factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: Union[int | float], factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: Union[int | float], factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    """
    This function defines Qwen3-VL preprocessing logic for images.

    NOTE (sbr): Original implentation raises an error if image aspect ration exceeds max image aspect ratio,
        however, it causes our pretrain to fail (we have at least one image with 500+ aspect ratio), so logic
        was adjusted to resize image to match max aspect ratio.

    Original code can be found here: [GitHub](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)
    """
    max_pixels = (
        max_pixels if max_pixels is not None else _QWEN_IMAGE_MAX_TOKEN_NUM * factor**2
    )
    min_pixels = (
        min_pixels if min_pixels is not None else _QWEN_IMAGE_MIN_TOKEN_NUM * factor**2
    )

    assert max_pixels >= min_pixels, (
        f"max_pixels ({max_pixels}) must be >= min_pixels ({min_pixels})"
    )
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > _QWEN_MAX_ASPECT_RATIO:
        reshape_min_size = math.ceil(max(height, width) / _QWEN_MAX_ASPECT_RATIO)
        if height < reshape_min_size:
            height = reshape_min_size
        if width < reshape_min_size:
            width = reshape_min_size
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    # NOTE (avtereshchenko): После округления до кратности factor и после
    # корректировок по числу пикселей итоговый размер снова может выйти за
    # ограничение Qwen по aspect ratio > 200 и упасть в processor. Здесь
    # повторно поджимаем длинную сторону, чтобы сохранить допустимый ratio.
    if max(h_bar, w_bar) / min(h_bar, w_bar) > _QWEN_MAX_ASPECT_RATIO:
        short_side = min(h_bar, w_bar)
        long_side = max(h_bar, w_bar)
        safe_long_side = max(
            factor, _floor_by_factor(short_side * _QWEN_MAX_ASPECT_RATIO, factor)
        )
        safe_long_side = min(safe_long_side, long_side)
        if h_bar >= w_bar:
            h_bar = safe_long_side
            w_bar = short_side
        else:
            h_bar = short_side
            w_bar = safe_long_side
    return h_bar, w_bar


def process_qwen_image(
    image: Image.Image,
    processor: BaseImageProcessor,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Resize and preprocess a PIL image for the Qwen vision encoder.

    Args:
        image: Input PIL image (must already be in RGB mode).
        processor: Qwen2VLImageProcessor instance.
        min_pixels: Optional minimum pixel count override for smart_resize.
        max_pixels: Optional maximum pixel count override for smart_resize.

    Returns:
        Tuple of (pixel_values, num_crops, grid_thw).
    """
    factor = processor.patch_size * processor.merge_size  # type: ignore
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )
    image = image.resize((resized_width, resized_height))
    image_dict = processor([image], return_tensors="pt", do_resize=False)  # type: ignore
    num_crops = (
        image_dict["image_grid_thw"][0].prod() // (processor.merge_size**2)
    ).item()  # type: ignore
    pixel_values = image_dict["pixel_values"]
    grid_thw = image_dict["image_grid_thw"][0].reshape(1, 3)
    return pixel_values, num_crops, grid_thw
