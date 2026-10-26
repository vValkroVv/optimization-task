# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .model import Qwen3VLModel
from .parallelize import parallelize_qwen3_vl

__all__ = [
    "parallelize_qwen3_vl",
    "Qwen3VLModel",
    "QWEN3_VL_SPECIAL_TOKENS",
]


QWEN3_VL_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}
