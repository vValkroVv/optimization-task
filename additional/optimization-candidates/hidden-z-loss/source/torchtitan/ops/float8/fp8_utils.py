# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import nn


def check_dim_for_fp8(tensor: torch.Tensor) -> bool:
    """True when the leading dims and last dim are both 128-aligned (TN GEMM)."""
    return math.prod(tensor.size()[:-1]) % 128 == 0 and tensor.size()[-1] % 128 == 0


def pad_x_to_y_divisible(x: int, y: int) -> int:
    return (y - x % y) % y


def pad_tensor_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    pad_tuple: tuple[int, ...] = ()

    if not check_dim_for_fp8(tensor):
        pad_last_dim = pad_x_to_y_divisible(tensor.size()[-1], 128)
        pad_tuple += (0, pad_last_dim)
        if len(tensor.size()) > 1:
            if len(tensor.size()) == 2 or math.prod(tensor.size()[:-1]) % 128 != 0:
                pad_pen_dim = pad_x_to_y_divisible(tensor.size()[-2], 128)
                pad_tuple += (0, pad_pen_dim)
        return nn.functional.pad(tensor, pad_tuple, value=0.0)
    return tensor


def unpad_tensor_if_needed(
    tensor: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    if tensor.size() != shape:
        if len(tensor.size()) > 1:
            h, w = shape[-2:]
            return tensor[..., :h, :w].contiguous()
        return tensor[: shape[-1]].contiguous()
    return tensor


@torch._dynamo.allow_in_graph
class SharedActivationQuantize(torch.autograd.Function):
    """Quantize an activation once for every fp8 consumer (``w1`` / ``w3``).

    ``allow_in_graph`` for the same reason as ``FP8Gemm``: the body runs at
    AOTAutograd trace time, and the wrapper it returns is a traceable subclass
    Dynamo carries to the consuming ``FP8Linear``.
    """

    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        from torchtitan.ops.float8._cudacherry import quantize_vec128

        ctx.orig_shape = inp.shape
        padded = pad_tensor_if_needed(inp)
        quantized_inp = quantize_vec128(padded, pack=True)
        quantized_inp.orig_shape = ctx.orig_shape
        return quantized_inp

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return unpad_tensor_if_needed(grad_output, ctx.orig_shape)
