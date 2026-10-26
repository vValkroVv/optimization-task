# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

import torch
import math

from composer.core import Callback, State
from composer.functional.triton_kernels import slice_chunks, slice_square_sums_batch
from composer.loggers import Logger
from composer.optim.decoupled_weight_decay import step_counts
from composer.utils import dist, TimerType, callback_timer

__all__ = ['OptimizerMonitor']


class OptimizerMonitor(Callback):
    """Computes and logs the L2 norm of gradients as well as any optimizer-specific metrics implemented in the optimizer's `report_per_parameter_metrics` method.

    L2 norms are calculated after the reduction of gradients across GPUs. This function iterates over the parameters of
    the model and may cause a reduction in throughput while training large models. In order to ensure the
    correctness of the norm, this function should be called after gradient unscaling in cases where gradients are scaled.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import OptimizerMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[OptimizerMonitor()],
            ... )

    The metrics are logged by the :class:`.Logger` to the following keys as described below. `grad_l2_norm` and `layer_grad_l2_norm` are
    logged in addition to metrics logged by the optimizer's `report_per_parameter_metrics` method. For convenience we have listed
    the metrics logged by DecoupledAdamW below.

    +-----------------------------------------------+-----------------------------------------------------+
    | Key                                           | Logged data                                         |
    +===============================================+=====================================================+
    |                                               | L2 norm of the gradients of all parameters in       |
    | ``l2_norm/grad/global``                       | the model on the :attr:`.Event.AFTER_TRAIN_BATCH`   |
    |                                               | event.                                              |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms                                 |
    | ``l2_norm/grad/LAYER_NAME``                   |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of Adam first moment after      |
    | ``l2_norm/moment/LAYER_NAME``                 |  calling optimizer step.                            |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of parameter weights            |
    | ``l2_norm/param/LAYER_NAME``                  |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of the step                     |
    | ``l2_norm/update/LAYER_NAME``                 |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    """

    def __init__(self, log_optimizer_metrics: bool = True, batch_log_interval: int = 10, log_pre_clip_gradients: bool = False):
        self.log_optimizer_metrics = log_optimizer_metrics
        self.batch_log_interval = batch_log_interval
        self.log_pre_clip_gradients = log_pre_clip_gradients
        self._pre_clip_logging = False
        self.rank_local_observer = None
        self._projections_by_shard = {}
        self._projection_chunks_by_shard = {}

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def batch_end(self, state: State, logger: Logger):
        should_log = state.timestamp.batch.value % self.batch_log_interval == 0
        if not should_log and self.rank_local_observer is None:
            return

        norm = 0.0
        optimizer_metrics = {}
        grad_prefix = 'l2_norm/pre_clip_grad' if self._pre_clip_logging else 'l2_norm/grad'
        log_optimizer_metrics = self.log_optimizer_metrics and not self._pre_clip_logging #to avoid logging optmizer-state metrics
        should_reduce_metrics = self.log_optimizer_metrics or self._pre_clip_logging

        named_parameters = [(name, p) for name, p in state.model.named_parameters()
                            if p.grad is not None and p.requires_grad]
        # An optimizer that reports all parameters at once reduces them together.
        batch_reporter = getattr(state.optimizers[0], 'report_metrics_for_parameters', None)
        metric_reporter = getattr(state.optimizers[0], 'report_per_parameter_metrics', None)
        if callable(batch_reporter) and log_optimizer_metrics:
            optimizer_metrics.update(batch_reporter(named_parameters, optimizer_metrics))
        elif callable(metric_reporter) and log_optimizer_metrics:
            for name, p in named_parameters:
                optimizer_metrics.update(metric_reporter(p, name, optimizer_metrics))
        for name, p in named_parameters:
            # Always log grad norm as a default metric if it's not specified
            if f'{grad_prefix}/{name}' not in optimizer_metrics:
                param_grad_norm = torch.linalg.vector_norm(p.grad)
                optimizer_metrics[f'{grad_prefix}/{name}'] = param_grad_norm



        if (state.fsdp_enabled or state.gigafsdp_enabled) and dist.get_world_size() > 0 and should_reduce_metrics:
            # If FSDP is enabled, the optimizer state lives on different ranks and must be reduced
            # and combined before we can compute metrics.
            # Each metric has a different way of being reduced, so the optimizer is responsible for implementing
            # the reduction process.
            # It occurs first via a pre-reduce, where the metric on each rank is modified and prepared
            # then an all-reduce where the modified metric on each rank is combined into the correct metric across all ranks.
            #
            # For example, L2 norms are squared on each rank before we apply all_reduce(SUM) and take the sqrt on each rank
            self.add_per_projection_metrics(state, optimizer_metrics, grad_prefix=grad_prefix, log_optimizer_metrics=log_optimizer_metrics)

            if self.rank_local_observer is not None:
                self.rank_local_observer(optimizer_metrics)
            if not should_log:
                return

            pre_reduce_metrics = getattr(state.optimizers[0], 'pre_reduce_metrics', None)
            if callable(pre_reduce_metrics):
                optimizer_metrics = pre_reduce_metrics(optimizer_metrics)

            dist_reduce_metrics = getattr(state.optimizers[0], 'dist_reduce_metrics', None)
            if callable(dist_reduce_metrics):
                optimizer_metrics = dist_reduce_metrics(optimizer_metrics)

        else:
            if self.rank_local_observer is not None:
                self.rank_local_observer(optimizer_metrics)
            if not should_log:
                return

        for metric in optimizer_metrics:
            if metric.startswith('l2_norm') and state.gigafsdp_enabled:
                optimizer_metrics[metric] /= (state.model.model._meta_info['hybrid_group_count']) ** 0.5

                if "_layernorm_parameter" in metric:
                    tp_size = dist.get_tp_group_size() or 1
                    optimizer_metrics[metric] /= tp_size ** 0.5

            if metric.startswith(grad_prefix):
                norm += optimizer_metrics[metric]**2

        optimizer_metrics[f'{grad_prefix}/global'] = norm**0.5

        for metric in optimizer_metrics:
            if isinstance(optimizer_metrics[metric], torch.Tensor):
                optimizer_metrics[metric] = optimizer_metrics[metric].item()
        logger.log_metrics(optimizer_metrics)

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def add_per_projection_metrics(self, state: State, optimizer_metrics: dict, grad_prefix: str = 'l2_norm/grad', log_optimizer_metrics: bool = True):
        """Slice each GigaFSDP flatten shard back into individual projection params
        and compute per-projection l2 norms from the local shard. The norms are
        squared+reduced later by the optimizer's pre/dist_reduce_metrics, so here we
        only emit the *local* contribution (norm over the part of the param that
        lives on this rank).
        """
        optimizer = state.optimizers[0]
        # Adam hyperparams for reconstructing the update tensor
        group = optimizer.param_groups[0]

        assert all(k in group for k in ('lr', 'eps', 'betas', 'weight_decay')), (
            f'{type(self).__name__} per-projection metrics require an Adam-like '
            f'optimizer (lr/eps/betas/weight_decay in the param group); '
            f'got {type(optimizer).__name__}.'
        )
        
        lr = group['lr']
        eps = group['eps']
        beta1, beta2 = group['betas']
        weight_decay = group['weight_decay']
        initial_lr = group.get('initial_lr', lr)
        decay_factor = (lr / initial_lr) if initial_lr else 1.0

        gigafsdp = getattr(state.model, 'model', None)
        if gigafsdp is None or not hasattr(gigafsdp, '_module'):
            return
        steps = step_counts(optimizer) if log_optimizer_metrics else {}
        # CUDA shards are reduced together after the loop.
        batched = []

        for _, stm in gigafsdp._module.named_modules():
            st = getattr(stm, 'super_tensor', None)
            if st is None:
                continue

            st = stm.super_tensor
            if not st.is_trainable or st.shard is None:
                continue

            flatten = stm.flatten            # this rank's shard (nn.Parameter)
            grad = flatten.grad

            exp_avg = exp_avg_sq = None
            step_size, sqrt_bias_correction2 = 0.0, 1.0
            if log_optimizer_metrics:
                opt_state = optimizer.state.get(flatten, None)
                if opt_state is None or 'exp_avg' not in opt_state:
                    continue

                exp_avg = opt_state['exp_avg']
                exp_avg_sq = opt_state['exp_avg_sq']
                step = steps[flatten] if flatten in steps else opt_state['step'].item()
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1
                sqrt_bias_correction2 = math.sqrt(bias_correction2)

            fqns, bounds = self._projections(flatten, st)
            if not fqns:
                continue  # no projection is on this rank
            if flatten.is_cuda:
                batched.append((flatten, grad, exp_avg, exp_avg_sq, self._projection_chunks(flatten, bounds),
                                step_size, sqrt_bias_correction2, eps, -weight_decay * decay_factor, fqns))
            else:
                if log_optimizer_metrics:
                    # full update tensor for this shard (same formula as report_per_parameter_metrics)
                    denom = (exp_avg_sq.float().sqrt() / sqrt_bias_correction2).add_(eps)
                    shard_update = step_size * exp_avg.float().div(denom)
                    shard_update.add_(flatten.float(), alpha=-weight_decay * decay_factor)
                    param_norms = torch._foreach_norm([flatten.data[lo:hi] for lo, hi in bounds])
                    update_norms = torch._foreach_norm([shard_update[lo:hi] for lo, hi in bounds])
                if grad is not None:
                    grad_norms = torch._foreach_norm([grad[lo:hi] for lo, hi in bounds])
                self._add_projection_norms(
                    optimizer_metrics, fqns, grad, param_norms if log_optimizer_metrics else None,
                    update_norms if log_optimizer_metrics else None, grad_norms if grad is not None else None,
                    grad_prefix,
                )
        if batched:
            norms = slice_square_sums_batch([entry[:9] for entry in batched]).sqrt()
            first = 0
            for flatten, grad, *_, fqns in batched:
                shard = norms[first:first + len(fqns)]
                first += len(fqns)
                self._add_projection_norms(
                    optimizer_metrics, fqns, grad, shard[:, 0].to(flatten.dtype) if log_optimizer_metrics else None,
                    shard[:, 1].float() if log_optimizer_metrics else None,
                    None if grad is None else shard[:, 2].to(grad.dtype), grad_prefix,
                )

    @staticmethod
    def _add_projection_norms(optimizer_metrics: dict, fqns: list, grad, param_norms, update_norms, grad_norms,
                              grad_prefix: str) -> None:
        """The norms of a shard's projections: parameter and update norms unless they are None, and the
        gradient norms if the shard has a gradient."""
        for index, fqn in enumerate(fqns):
            if param_norms is not None:
                optimizer_metrics[f'l2_norm/param/{fqn}'] = param_norms[index]
                optimizer_metrics[f'l2_norm/update/{fqn}'] = update_norms[index]
            if grad is not None:
                optimizer_metrics[f'{grad_prefix}/{fqn}'] = grad_norms[index]

    def _projections(self, flatten: torch.Tensor, st) -> tuple[list[str], list[tuple[int, int]]]:
        """The projections intersecting this rank's shard, in ``param_fqn_to_offset`` order, and
        their slices of the shard; built once per shard."""
        cached = self._projections_by_shard.get(flatten)
        if cached is not None:
            return cached
        shard_size = st.shard_size
        shard_start = st.rank * shard_size
        shard_end = shard_start + shard_size
        fqns, bounds = [], []
        for fqn, offset in st.param_fqn_to_offset.items():
            numel = math.prod(st.param_fqn_to_shape[fqn])
            # intersection of this param's global range with this rank's shard
            lo = max(offset, shard_start) - shard_start
            hi = min(offset + numel, shard_end) - shard_start
            if hi > lo:
                fqns.append(fqn)
                bounds.append((lo, hi))
        cached = (fqns, bounds)
        self._projections_by_shard[flatten] = cached
        return cached

    def _projection_chunks(self, flatten: torch.Tensor, bounds: list[tuple[int, int]]) -> tuple:
        """The projection slices' ``slice_chunks`` on the shard's device, built once per shard."""
        chunks = self._projection_chunks_by_shard.get(flatten)
        if chunks is None:
            chunks = self._projection_chunks_by_shard[flatten] = slice_chunks(bounds, flatten.device)
        return chunks

    def before_gradient_clipping(self, state: State, logger: Logger):
        if not self.log_pre_clip_gradients:
            return
        self._pre_clip_logging = True
        try:
            self.batch_end(state, logger)
        finally:
            self._pre_clip_logging = False