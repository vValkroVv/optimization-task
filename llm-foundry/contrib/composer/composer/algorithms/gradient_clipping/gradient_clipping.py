# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core gradient clipping classes and functions."""

from __future__ import annotations

import re
import logging
from typing import Iterable, Optional, Union, List
import math
import functools
import torch
import torch.distributed as torch_dist
from packaging import version

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.models import ComposerModel
import composer.utils.dist as dist
from composer.utils import TimerType, gigatimer

log = logging.getLogger(__name__)

__all__ = ['GradientClipping', 'apply_gradient_clipping']

from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import _get_grad_norm

@torch.no_grad()
def tp_clip_grad_norm(
    model: FSDP, max_norm: Union[float, int], norm_type: Union[float, int] = 2.0
) -> torch.Tensor:
    """
    Clips the gradient norm of all parameters. The norm is computed over
    all parameters' gradients as viewed as a single vector, and the
    gradients are modified in-place.

    Args:
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'``
            for infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).

    .. note:: If every FSDP instance uses ``NO_SHARD``, meaning that no
        gradients are sharded across ranks, then you may directly use
        :param model:
        :func:`torch.nn.utils.clip_grad_norm_`.

    .. note:: If at least some FSDP instance uses a sharded strategy (i.e.
        one other than ``NO_SHARD``), then you should use this method
        instead of :func:`torch.nn.utils.clip_grad_norm_` since this method
        handles the fact that gradients are sharded across ranks.

    .. note:: The total norm returned will have the "largest" dtype across
        all parameters/gradients as defined by PyTorch's type promotion
        semantics. For example, if *all* parameters/gradients use a low
        precision dtype, then the returned norm's dtype will be that low
        precision dtype, but if there exists at least one parameter/
        gradient using FP32, then the returned norm's dtype will be FP32.

    .. warning:: This needs to be called on all ranks since it uses
        collective communications.
    """
    tp_rank = dist.get_tp_group_rank()
    tp_world_size = dist.get_tp_group_size()
    # Only on zero Tensor Parallel rank duplicated params are calculating
    zero_tp_rank = tp_rank == 0

    assert tp_rank is not None, "tp_clip_grad_norm() function should work only with Tensor Parallel"

    _lazy_init(model, model)
    if not model._is_root:
        raise RuntimeError(
            "`clip_grad_norm_()` should only be called on the root FSDP instance"
        )
    model._assert_state(TrainingState.IDLE)
    # If every FSDP instance uses `NO_SHARD`, then we can directly use
    # the normal `nn.utils` one targeting local gradients
    all_no_shard = all(
        not handle.uses_sharded_strategy for handle in model._all_handles
    )
    if all_no_shard:
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm, norm_type
        )
    # Otherwise, there exists some FSDP instance using a sharded strategy,
    # where sharded and non-sharded parameters must be handled separately
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    sharded_params = set()
    nonsharded_params = set()  # `NO_SHARD` or not FSDP-managed
    model_parallel_ignore_params = set()
    model_parallel_params = set()
    grads: List[torch.Tensor] = []

    handle_use_orig_params = []
    for handle in model._all_handles:
        target_set = (
            sharded_params if handle.uses_sharded_strategy else nonsharded_params
        )
        if handle._use_orig_params:
            handle_use_orig_params.append(True)
            for param in handle.flat_param._params:
                if getattr(param, "tensor_model_parallel", False) or zero_tp_rank:
                    target_set.add(param)
                else:
                    model_parallel_ignore_params.add(param)

                if getattr(param, "tensor_model_parallel", False):
                    model_parallel_params.add(param)
        else:
            handle_use_orig_params.append(False)
            target_set.add(handle.flat_param)
            if handle.flat_param.grad is not None:
                grads.append(handle.flat_param.grad)

    # (Sber) We should ensure that all params has same format
    use_orig_params = handle_use_orig_params[0]
    assert all(x == use_orig_params for x in handle_use_orig_params)

    for param in model.parameters():
        if use_orig_params and param.grad is not None:
            grads.append(param.grad)

        not_fsdp_managed = (
            param not in sharded_params and param not in nonsharded_params and param not in model_parallel_ignore_params
        )
        if not_fsdp_managed:
            if use_orig_params:
                if getattr(param, "tensor_model_parallel", False) or zero_tp_rank:
                    nonsharded_params.add(param)
                else:
                    model_parallel_ignore_params.add(param)

                if getattr(param, "tensor_model_parallel", False):
                    model_parallel_params.add(param)
            else:
                nonsharded_params.add(param)
                if param.grad is not None:
                    grads.append(param.grad)
    # Compute local norms (forced to be in FP32)
    local_sharded_norm = _get_grad_norm(sharded_params, norm_type).to(
        model.compute_device
    )
    local_nonsharded_norm = _get_grad_norm(nonsharded_params, norm_type).to(
        model.compute_device
    )
    # Warn if mp_world_size > 1 but model_parallel_params is empty
    if tp_world_size > 1 and len(model_parallel_params) == 0:
        Warning("tp_world_size > 1 but model_parallel_params is empty, are model parallel params correctly marked?")

    # Reconstruct the total gradient norm depending on the norm type
    if norm_type == math.inf:
        total_norm = torch.maximum(local_sharded_norm, local_nonsharded_norm)
        handle = torch_dist.all_reduce(
            total_norm, op=torch_dist.ReduceOp.MAX, group=model.process_group, async_op=True
        )
        handle.wait()
        handle = torch_dist.all_reduce(
            total_norm, op=torch_dist.ReduceOp.MAX, group=dist.get_tp_group(), async_op=True
        )
        handle.wait()
    else:
        total_norm = local_sharded_norm**norm_type
        handle = torch_dist.all_reduce(total_norm, group=model.process_group, async_op=True)
        handle.wait()
        # All-reducing the local non-sharded norm would count it an extra
        # world-size-many times
        total_norm += local_nonsharded_norm**norm_type
        handle = torch_dist.all_reduce(
            total_norm, group=dist.get_tp_group(), async_op=True
        )
        handle.wait()
        total_norm = total_norm ** (1.0 / norm_type)
    if model.cpu_offload.offload_params:
        total_norm = total_norm.cpu()

    clip_coef = max_norm / (total_norm + 1e-6)
    # Multiplying by the clamped coefficient is meaningless when it is
    # equal to 1, but it avoids the host-device sync that would result from
    # `if clip_coef < 1`
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grad in grads:
        grad.detach().mul_(clip_coef_clamped.to(grad.device, grad.dtype))
    # Use the "largest" dtype by type promotion semantics to use the same
    # dtype as if we did not force local norm computation to be in FP32
    if len(grads) == 0:
        # If this rank has no gradients, then we must default to FP32
        # unless we use additional communication, which we prefer to avoid
        # since `clip_grad_norm_()` is called in the training loop
        Warning(
            f"Called FSDP.clip_grad_norm_() on rank {dist.get_global_rank()} with no "
            "gradients -- returning the total norm in the default dtype "
            f"{total_norm.dtype}"
        )  # warn since this is generally unexpected
        return total_norm
    total_norm_dtype = functools.reduce(
        lambda dtype1, dtype2: torch.promote_types(dtype1, dtype2),
        [grad.dtype for grad in grads],
    )
    return total_norm.to(total_norm_dtype)

@gigatimer(timer_level=0)
def apply_gradient_clipping(model: Union[ComposerModel, torch.nn.Module], clipping_type: str, clipping_threshold: float,
                            fsdp_enabled: bool, gigafsdp_enabled: bool) -> Optional[torch.Tensor]:
    """Clips all gradients in model based on specified clipping_type.

    Args:
        model (ComposerModel or torch.nn.Module): The model that we want to apply gradient clipping.
        clipping_type ('adaptive', 'norm', 'value'): String denoting which type of
            gradient clipping to do. The options are: 'norm', which clips the gradient norm
            and uses `torch.nn.utils.clip_grad_norm_`, 'value', which clips gradient at
            a specified value and uses `torch.nn.utils.clip_grad_value_`, and 'adaptive',
            which clips all gradients based on gradient norm:parameter norm ratio using
            composer.algorithms.gradient_clipping.gradient_clipping._apply_agc.
        clipping_threshold (float, optional): Specifies what value to clip the gradients
            to (for 'value'), what values to clip the gradient norms to (for 'norm'), and
            threshold by which if grad_norm / weight_norm is greater than this threshold then
            scale gradients by this threshold * (weight_norm / grad_norm) (for 'adaptive').
        fsdp_enabled (bool): Bool of if the model is a FSDP model or not.
    """
    total_norm = None
    if fsdp_enabled:
        torch_version = re.match(r"^(\d+\.\d+\.\d+(\.[a-z, 0-9]+)?)", torch.__version__).group()
        if version.parse(torch_version) < version.parse('1.13.0'):
            raise RuntimeError('To use FSDP with Composer, you must use torch>=1.13.0.')
        from torch.distributed.fsdp import FullyShardedDataParallel

        for module in model.modules():
            if isinstance(module, FullyShardedDataParallel) and module.check_is_root():
                if clipping_type == 'norm':
                    norm_type = 2
                elif clipping_type == 'value':
                    norm_type = float('inf')
                else:
                    raise ValueError(f"clipping type must be 'norm' or 'value' with FSDP not {clipping_type}")
                if dist.get_tp_group() is None:
                    total_norm = module.clip_grad_norm_(max_norm=clipping_threshold, norm_type=norm_type)
                else:
                    assert version.parse(torch_version) < version.parse('2.5'), 'To use Tensor Parallel clip_grad_norm, you must use torch<2.5.'
                    total_norm = tp_clip_grad_norm(model=module, max_norm=clipping_threshold, norm_type=norm_type)
    elif gigafsdp_enabled:
        assert clipping_type == 'norm', "GigaFSDP only support norm grad clip"
        from giga_fsdp import GigaFSDP
        for module in model.children():
            if isinstance(module, GigaFSDP):
                total_norm = module.clip_grad_norm_(max_norm=clipping_threshold, norm_type=2)
            else:
                parameters = module.parameters()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=clipping_threshold)
    else:
        parameters = model.parameters()
        if clipping_type == 'adaptive':
            _apply_agc(parameters, clipping_threshold=clipping_threshold)
        elif clipping_type == 'norm':
            total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=clipping_threshold)
        elif clipping_type == 'value':
            torch.nn.utils.clip_grad_value_(parameters, clip_value=clipping_threshold)
        else:
            raise ValueError(f"clipping_type must be 'adaptive', 'norm', or 'value' not {clipping_type} ")
    return total_norm


def _apply_agc(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    clipping_threshold: float,
) -> None:
    """Clips all gradients in model based on ratio of gradient norms to parameter norms.
    Args:
        parameters (torch.Tensor or Iterable[torch.Tensor]): The parameters to of the
            model for whose gradients we will clip
        clipping_threshold (float, optional): The largest acceptable ratio between grad
            norms and parameter norms before clipping is done.
    """
    for param in parameters:
        if param.grad is None:
            continue

        # Detach weights and gradients, so the clipping operation is not added to
        # computational graph.
        weights = param.detach()
        grad = param.grad.detach()

        # Get clipped version of gradients.
        clipped_grad_coeff = _get_clipped_gradient_coeff(weights, grad, clipping_threshold=clipping_threshold)

        # Copy clipped gradients into param.grad attribute, so they can be accessed by
        # optimizer.
        grad.mul_(clipped_grad_coeff)


class GradientClipping(Algorithm):
    """Clips all gradients in model based on specified clipping_type.

    Runs on ``Event.AFTER_TRAIN_BATCH``.

    Example:
         .. testcode::

            from composer.algorithms import GradientClipping
            from composer.trainer import Trainer
            gc = GradientClipping(clipping_type='norm', clipping_threshold=0.1)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[gc],
                optimizers=[optimizer]
            )

    Args:
        clipping_type ('adaptive', 'norm', 'value'): String denoting which type of
            gradient clipping to do. The options are: 'norm', which clips the gradient norm
            and uses `torch.nn.utils.clip_grad_norm_`, 'value', which clips gradient at
            a specified value and uses `torch.nn.utils.clip_grad_value_`, and 'adaptive',
            which clips all gradients based on gradient norm:parameter norm ratio using
            composer.algorithms.gradient_clipping.gradient_clipping._apply_agc.
        clipping_threshold (float, optional): Specifies what value to clip the gradients
            to (for 'value'), what values to clip the gradient norms to (for 'norm'), and
            threshold by which if grad_norm / weight_norm is greater than this threshold then
            scale gradients by this threshold * (weight_norm / grad_norm) (for 'adaptive').
    """

    def __init__(self, clipping_type: str, clipping_threshold: float):
        self.clipping_type = clipping_type
        self.clipping_threshold = clipping_threshold
        self.last_total_norm: Optional[torch.Tensor] = None

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.INIT, Event.AFTER_TRAIN_BATCH]

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        if event == Event.AFTER_TRAIN_BATCH:
            for callback in state.callbacks:
                hook = getattr(callback, "before_gradient_clipping", None)
                if callable(hook):
                    hook(state, logger)
                    
            self.last_total_norm = apply_gradient_clipping(model=state.model,
                                    clipping_type=self.clipping_type,
                                    clipping_threshold=self.clipping_threshold,
                                    fsdp_enabled=state.fsdp_enabled,
                                    gigafsdp_enabled=state.gigafsdp_enabled)


def _get_clipped_gradient_coeff(weights: torch.Tensor, grad: torch.Tensor, clipping_threshold: float = 0.01):
    """Clips all gradients in model based on ratio of gradient norms to parameter norms.

    Gradients whose norms exceed

    .. math:: weight_norm * clipping_threshold

    are scaled down by

    .. math:: (weight_norm / grad_norm) * clipping_threshold.

    Args:
        weights (torch.Tensor): Tensor of weights (parameters) from the model.
        grad (torch.Tensor): Tensor of gradients of the loss with respect to the weights.
        clipping_threshold (float, optional): The largest acceptable ratio between grad
            norms and parameter norms before clipping is done.

    Return:
        clipped_grad_coeff (torch.Tensor): Coefficient of same shape as grad_norm equal to
            (weight_norm / grad_norm) * clipping_threshold for gradients whose norms
            that exceed weight_norm * clipping_threshold and one otherwise.
    """

    # Compute and clamp grad and weight norms.
    w_norm = _unitwise_norm(weights)
    grad_norm = _unitwise_norm(grad)

    # Gradients whose norms are greater than weight_norm * clipping_threhsold are
    # scaled down by (weight_norm * clipping_threhsold) / grad_norm.
    max_norm = w_norm.mul_(clipping_threshold)
    clipped_grad_coeff = max_norm.div_(grad_norm).nan_to_num_(nan=1.0).clamp_(max=1.0)

    return clipped_grad_coeff


def _unitwise_norm(tensor: torch.Tensor):
    """Implements unitwise norm as described in Brock et al, 2021.

    For 0D scalars of shape [], we trivially normalize with dim=0 which essentially returns the absolute value of the scalar.
    For 1D *.bias weights of shape [out_features], we normalize across entire vector -> dim=0.
    For 2D torch.nn.Linear weights of shape [out_features, in_features]: we normalize across in_features -> dim = 1
    For 4D torch.nn.Conv2d weights [out_channels, in_channels, kernel_height, kernel_width]:
        we normalize across [in_channels, kernel_height, kernel_width] -> dim = (1, 2, 3).
    If a 3D parameter were somehow in your model, we would normalize buy the last two dimensions -> dim = (1,2).

    Args:
        tensor (torch.Tensor): A parameter or gradient of the model.

    Returns:
        The appropriate L2 norm of the parameter or gradient as described above.
    """
    # 0D for scalars, 1D for bias vectors.
    if tensor.ndim <= 1:
        dim = 0
        keepdim = False
    # 2D corresponds to MLPs and 4D corresponds to ConvNets.
    else:
        dim = tuple(range(1, tensor.ndim))
        keepdim = True
    # L2 Norm.
    return torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=keepdim)
