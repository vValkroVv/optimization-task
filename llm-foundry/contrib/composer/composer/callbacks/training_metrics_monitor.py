# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

import torch

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist, TimerType, callback_timer
import composer.utils.zero_nan_grads


__all__ = ['TrainingMetricsMonitor']

class TrainingMetricsMonitor(Callback):
    """ Here we consider metrics and statistics regarding the stability of learning

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import TrainingMetricsMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[TrainingMetricsMonitor()],
            ... )

    The metrics are logged by the :class:`.Logger` to the following keys as described below.

    +-----------------------------------------------+-----------------------------------------------------+
    | Key                                           | Logged data                                         |
    +===============================================+=====================================================+
    |                                               |                                                     |
    | ``training/skip_updates_total``               | Calculate absolute module updates skips             |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               |                                                     |
    | ``training/skip_updates_ratio``               | Calculate average module updates skips              |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               |                                                     |
    | ``training/zeroed_module_grads``              | Calculate absolute nodule zeroed grads              |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               |                                                     |
    | ``training/zeroed_module_grads_ratio``        | Calculate average module zeroed skips               |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    """

    def __init__(self, batch_log_interval: int = 10):
        self.batch_log_interval = batch_log_interval
        self.rank_local_observer = None
    
    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def batch_end(self, state: State, logger: Logger):
        if self.rank_local_observer is not None:
            optimizer = state.optimizers[0]
            self.rank_local_observer((
                getattr(optimizer, 'skip_updates', None),
                getattr(optimizer, 'total_updates', None),
                composer.utils.zero_nan_grads.NUM_MODULE_GARDS_ZEROED_OUT,
                composer.utils.zero_nan_grads.NUM_MODULE_GARDS_PROCESSED,
            ))
        if state.timestamp.batch.value % self.batch_log_interval != 0:
            return

        training_metrics = {}

        # skip_updates count
        assert len(state.optimizers) == 1, "Not implemetned logic only for multiple optimizers"
        skip_updates = getattr(state.optimizers[0], 'skip_updates', None)
        total_updates = getattr(state.optimizers[0], 'total_updates', None)
        if skip_updates is not None and total_updates is not None:
            updates_stats = torch.tensor(
                [skip_updates, total_updates],
                device=torch.cuda.current_device(),
            )
            training_metrics["training/skip_updates_total"] = updates_stats[0]
            training_metrics["training/skip_updates_ratio"] = updates_stats[0] / updates_stats[1]
            zeroed_module_grads = composer.utils.zero_nan_grads.NUM_MODULE_GARDS_ZEROED_OUT
            processed_module_grads = composer.utils.zero_nan_grads.NUM_MODULE_GARDS_PROCESSED
            if zeroed_module_grads is not None and processed_module_grads is not None:
                zeroed_stats = torch.stack(
                    (
                        zeroed_module_grads.reshape(()),
                        processed_module_grads.reshape(()),
                    ),
                )
                if dist.get_world_size() > 1:
                    dist.all_reduce(zeroed_stats, reduce_operation='SUM')
                training_metrics["training/zeroed_module_grads"] = zeroed_stats[0]
                training_metrics["training/zeroed_module_grads_ratio"] = (
                    zeroed_stats[0] / zeroed_stats[1]
                )
            
            # set counters to default (0 or None)
            setattr(state.optimizers[0], 'skip_updates', 0)
            setattr(state.optimizers[0], 'total_updates', 0)
            composer.utils.zero_nan_grads.NUM_MODULE_GARDS_ZEROED_OUT = None
            composer.utils.zero_nan_grads.NUM_MODULE_GARDS_PROCESSED = None

        for metric in training_metrics:
            if isinstance(training_metrics[metric], torch.Tensor):
                training_metrics[metric] = training_metrics[metric].item()
        logger.log_metrics(training_metrics)
