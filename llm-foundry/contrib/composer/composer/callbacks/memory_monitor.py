# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log memory usage during training."""
import logging
import math
import warnings
from typing import Callable, Dict, Optional, Union

import torch.cuda

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import TimerType, callback_timer

log = logging.getLogger(__name__)

__all__ = ['MemoryMonitor']


class MemoryMonitor(Callback):
    """Logs the memory usage of the model.

    This callback calls the torch memory stats API for CUDA (see :func:`torch.cuda.memory_stats`)
    on the :attr:`.Event.AFTER_TRAIN_BATCH` and reports different memory statistics.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import MemoryMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[MemoryMonitor()],
            ... )

    The memory statistics are logged by the :class:`.Logger` to the following keys as
    described below.

    +--------------------------+-------------------------------------------------------------+
    | Key                      | Logged data                                                 |
    +==========================+=============================================================+
    |                          | Several memory usage statistics                             |
    | ``memory/{statistic}``   | are logged on                                               |
    |                          | :attr:`.Event.AFTER_TRAIN_BATCH` event.                     |
    +--------------------------+-------------------------------------------------------------+

    The following statistics are recorded:

    +------------------------+-------------------------------------------------------------------------------------------+
    | Statistic              | Description                                                                               |
    +========================+===========================================================================================+
    | current_allocated_mem  | Current amount of allocated memory in gibibytes.                                          |
    +------------------------+-------------------------------------------------------------------------------------------+
    | current_active_mem     | Current amount of active memory in gibibytes at the time of recording.                    |
    +------------------------+-------------------------------------------------------------------------------------------+
    | current_inactive_mem   | Current amount of inactive, non-releaseable memory in gibibytes at the time of recording. |
    +------------------------+-------------------------------------------------------------------------------------------+
    | current_reserved_mem   | Current amount of reserved memory in gibibytes at the time of recording.                  |
    +------------------------+-------------------------------------------------------------------------------------------+
    | peak_allocated_mem     | Peak amount of allocated memory in gibibytes.                                             |
    +------------------------+-------------------------------------------------------------------------------------------+
    | peak_active_mem        | Peak amount of active memory in gibibytes at the time of recording.                       |
    +------------------------+-------------------------------------------------------------------------------------------+
    | peak_inactive_mem      | Peak amount of inactive, non-releaseable memory in gibibytes at the time of recording.    |
    +------------------------+-------------------------------------------------------------------------------------------+
    | peak_reserved_mem      | Peak amount of reserved memory in gibibytes at the time of recording.                     |
    +------------------------+-------------------------------------------------------------------------------------------+
    | alloc_retries          | Number of failed cudaMalloc calls that result in a cache flush and retry.                 |
    +------------------------+-------------------------------------------------------------------------------------------+
    | non_torch_mem          | Device memory in use beyond what torch reserved, in gibibytes: NCCL buffers, the CUDA     |
    |                        | context, and whatever else allocates around the caching allocator. Device-wide and        |
    |                        | rank-local; libraries allocate lazily, so read it after a few batches.                    |
    +------------------------+-------------------------------------------------------------------------------------------+
    | total_peak_mem         | Peak allocated by torch plus non_torch_mem, in gibibytes: what a step needs on            |
    |                        | the device, not counting free blocks the caching allocator holds aside.                   |
    +------------------------+-------------------------------------------------------------------------------------------+

    .. note::
        Memory usage monitoring is only supported for GPU devices.

    Args:
        memory_keys (Dict[str, str], optional): A dict specifying memory statistics to log. Keys
            are the names of memory statistics to log from `torch.cuda.memory_stats()`, and values
            are the names they will be logged under. If not provided, the above statistics are
            logged. Defaults to None.
        reset_peak_per_batch (bool, optional): If True, calls
            :func:`torch.cuda.reset_peak_memory_stats` at the start of each batch so the logged
            ``peak_*`` statistics reflect the peak *within the batch* rather than the cumulative
            peak since process start. Defaults to True.
    """

    def __init__(
        self,
        memory_keys: Optional[Dict[str, str]] = None,
        reset_peak_per_batch: bool = True,
    ) -> None:
        self.rank_local_observer = None
        self.memory_keys = memory_keys
        self.reset_peak_per_batch = reset_peak_per_batch

    def init(self, state: State, logger: Logger) -> None:
        # Not relying on `torch.cuda.is_available()` since the model could be on CPU.
        model_device = next(state.model.parameters()).device

        if model_device.type not in ('cuda', 'meta'):
            warnings.warn(f'The memory monitor only works on CUDA devices, but the model is on {model_device.type}.')

    def before_train_batch(self, state: State, logger: Logger):
        if not self.reset_peak_per_batch:
            return
        model_device = next(state.model.parameters()).device
        if model_device.type != 'cuda':
            return
        torch.cuda.reset_peak_memory_stats()

    @callback_timer(timer_type=TimerType.CPU, timer_level=3)
    def after_train_batch(self, state: State, logger: Logger):
        memory_report = {}

        model_device = next(state.model.parameters()).device
        if model_device.type != 'cuda':
            return

        memory_report = _get_memory_report(self.memory_keys, self.rank_local_observer)

        logger.log_metrics({f'memory/{mem_stat}': val for (mem_stat, val) in memory_report.items()})


_MEMORY_KEYS = {
    'allocated_bytes.all.current': 'current_allocated_mem',
    'active_bytes.all.current': 'current_active_mem',
    'inactive_split_bytes.all.current': 'current_inactive_mem',
    'reserved_bytes.all.current': 'current_reserved_mem',
    'allocated_bytes.all.peak': 'peak_allocated_mem',
    'active_bytes.all.peak': 'peak_active_mem',
    'inactive_split_bytes.all.peak': 'peak_inactive_mem',
    'reserved_bytes.all.peak': 'peak_reserved_mem',
    'num_alloc_retries': 'alloc_retries',
}


def _gib(nbytes: float) -> float:
    """Bytes as gibibytes, five significant digits."""
    gib = nbytes / (1 << 30)
    return round(gib, 4 - int(math.floor(math.log10(abs(gib))))) if gib else 0.0


def _get_memory_report(
    memory_keys: Optional[Dict[str, str]] = None,
    rank_local_observer: Optional[Callable[[Dict[str, int], int, int], None]] = None,
) -> Dict[str, Union[int, float]]:
    memory_stats = torch.cuda.memory_stats()
    memory_keys = memory_keys or _MEMORY_KEYS

    memory_report = {
        name: _gib(memory_stats[torch_name]) if 'bytes' in torch_name else memory_stats[torch_name]
        for torch_name, name in memory_keys.items()
        if torch_name in memory_stats
    }

    free, total = torch.cuda.mem_get_info()
    if rank_local_observer is not None:
        rank_local_observer(memory_stats, free, total)
    non_torch = total - free - memory_stats.get('reserved_bytes.all.current', 0)
    memory_report['non_torch_mem'] = _gib(non_torch)
    memory_report['total_peak_mem'] = _gib(memory_stats.get('allocated_bytes.all.peak', 0) + non_torch)

    return memory_report
