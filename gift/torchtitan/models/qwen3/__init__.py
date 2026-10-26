# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .model import Qwen3Model
from .parallelize import parallelize_qwen3

__all__ = [
    "parallelize_qwen3",
    "Qwen3Model",
]
