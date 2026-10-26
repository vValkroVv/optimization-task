# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Contains zero nan grads funtions."""

__all__ = ['enable_zero_nan_grads']

from typing import no_type_check, Tuple
from packaging import version

import torch
import re

from torch.distributed.fsdp._common_utils import _FSDPState
from composer.utils.misc import get_torch_version

NUM_MODULE_GARDS_ZEROED_OUT = None
NUM_MODULE_GARDS_PROCESSED = None

TORCH_VERSION = re.match(r"^(\d+\.\d+\.\d+(\.[a-z]+)?)", torch.__version__).group()


def zero_nan_grads(grad):
    is_good = torch.isfinite(grad).all()
    global NUM_MODULE_GARDS_ZEROED_OUT, NUM_MODULE_GARDS_PROCESSED
    if NUM_MODULE_GARDS_ZEROED_OUT is None:
        NUM_MODULE_GARDS_ZEROED_OUT = torch.tensor([0], device=is_good.device)
        NUM_MODULE_GARDS_PROCESSED = torch.tensor([0], device=is_good.device)
    NUM_MODULE_GARDS_ZEROED_OUT += (1-is_good.long())
    NUM_MODULE_GARDS_PROCESSED += 1
    grad = torch.where(is_good, grad, 0)
    return grad

def reduce_scatter_hook(state, grad, output):
    r"""
    This FSDP communication hook implements ``reduce_scatter`` modified algorithm that zero nans in gradients for
    sharded FSDP strategies and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): An unsharded gradient for the local batch that needs to be
        communicated across ranks.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
    # Average grad by pre-division factor.
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)

    grad = zero_nan_grads(grad)
    handle = torch.distributed.reduce_scatter_tensor(
        output, grad, group=state.process_group, async_op=True
    )
    handle.wait()
    # Average grad's shard by post-division factor.
    if state.gradient_postdivide_factor > 1:
        output.div_(state.gradient_postdivide_factor)

@no_type_check
def _get_reduce_scatter_tensors(
    state: _FSDPState, unsharded_grad: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the input and output tensors to reduce-scatter, respectively.
    """
    unsharded_grad = zero_nan_grads(unsharded_grad)
    chunks = list(unsharded_grad.chunk(state.world_size))
    numel_to_pad = state.world_size * chunks[0].numel() - unsharded_grad.numel()
    padded_unsharded_grad = (
        torch.nn.functional.pad(unsharded_grad, [0, numel_to_pad]) if numel_to_pad > 0 else unsharded_grad
    )
    new_sharded_grad = torch.empty_like(chunks[0])  # padded
    return padded_unsharded_grad, new_sharded_grad

def enable_zero_nan_grads():
    if version.parse(TORCH_VERSION) in [version.parse("2.4"), version.parse("2.5.1"), version.parse("2.6")]:
        import torch.distributed.fsdp._runtime_utils
        torch.distributed.fsdp._runtime_utils._get_reduce_scatter_tensors = _get_reduce_scatter_tensors
    else:
        raise RuntimeError(f'To use zero_nan_grads callback with Composer, you must use torch 2.4, 2.5.1 or 2.6. Your version = {torch.__version__}')
