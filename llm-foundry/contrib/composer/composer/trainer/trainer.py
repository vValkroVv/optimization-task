# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Train models."""

from __future__ import annotations

import collections.abc
import contextlib
import datetime
import itertools
import logging
import os
import random
import re
import tempfile
import textwrap
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import (Any, Callable, ContextManager, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple,
                    Union, cast)

from packaging import version
from composer.utils import reproducibility
from composer.utils.misc import get_torch_version

import coolname
import torch
import torch.distributed
import torch.nn as nn
import torch.utils.data
from packaging import version
from torch.amp.grad_scaler import GradScaler

if get_torch_version() < version.parse("2.4"):
    from torch.amp.grad_scaler import _refresh_per_optimizer_state

from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import Metric

from composer.callbacks import CheckpointSaver, OptimizerMonitor
from composer.core import (Algorithm, AlgorithmPass, Batch, Callback, DataSpec, Engine, Evaluator, Event, Precision,
                           PyTorchScheduler, State, Time, Timestamp, TimeUnit, TrainerMode, ensure_data_spec,
                           ensure_evaluator, ensure_time, get_precision_context, validate_eval_automicrobatching)
from composer.devices import Device, DeviceCPU, DeviceGPU, DeviceMPS, DeviceTPU
from composer.loggers import (ConsoleLogger, Logger, LoggerDestination, MosaicMLLogger, ProgressBarLogger,
                              RemoteUploaderDownloader, WandBLogger)
from composer.loggers.mosaicml_logger import MOSAICML_ACCESS_TOKEN_ENV_VAR, MOSAICML_PLATFORM_ENV_VAR
from composer.models import ComposerModel
from composer.optim import ComposerScheduler, DecoupledSGDW, compile_composer_scheduler
from composer.profiler import Profiler
from composer.trainer._scale_schedule import scale_pytorch_scheduler
from composer.trainer._scaler import ClosureGradScaler
from composer.trainer.dist_strategy import (prepare_fsdp_module,
                                            set_fsdp_default)
from composer.utils import (ExportFormat, MissingConditionalImportError, ObjectStore, Transform, checkpoint, dist,
                            ensure_tuple, export_with_logger, extract_hparams, format_name_with_dist,
                            get_composer_env_dict, get_device, get_file, is_tpu_installed, map_collection,
                            maybe_create_object_store_from_uri, maybe_create_remote_uploader_downloader_from_uri,
                            model_eval_mode, parse_uri, reproducibility, using_torch_2_1_1, enable_zero_nan_grads)
from composer.datasets.utils import MultiTokenEOSCriteria
from composer.utils.misc import get_torch_version
from streaming import StreamingDataset
from torch.utils.data import Dataset

if is_tpu_installed():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

log = logging.getLogger(__name__)

__all__ = ['Trainer']

# syntax to shorten the Scheduler type annotations
Scheduler = Union[ComposerScheduler, PyTorchScheduler]


def _raise_missing_argument_exception(arg_name: str):
    raise ValueError((f'{arg_name} is a required argument and must be specified when constructing the '
                      f'{Trainer.__name__} or when calling {Trainer.__name__}.{Trainer.fit.__name__}(). '
                      f'To fix, please specify `{arg_name}` via {Trainer.__name__}({arg_name}=...) or '
                      f'{Trainer.__name__}.{Trainer.fit.__name__}({arg_name}=...).'))


def _scale_max_duration_by_ssr(
    scale_schedule_ratio: float,
    orig_max_duration: Optional[Time[int]],
) -> Optional[Time[int]]:
    if orig_max_duration is None:
        return None
    max_duration = cast(Time[int], orig_max_duration * scale_schedule_ratio)
    log.info(f'max_duration changed from {orig_max_duration} to {max_duration}')
    if max_duration.value == 0:
        raise ValueError('Scale schedule has reduced the max_duration to 0. Set a higher ratio or use more epochs.')
    return max_duration


def _get_default_scheduler_frequency(schedulers: Optional[Union[Scheduler, Sequence[Scheduler]]]):
    has_pytorch_scheduler = any(isinstance(scheduler, PyTorchScheduler) for scheduler in ensure_tuple(schedulers))
    if has_pytorch_scheduler:
        log.info(('Stepping schedulers every epoch, as a PyTorch scheduler was provided. '
                  'The trainer cannot automatically convert the parameters (e.g. step_size, T_max) of the '
                  'PyTorch scheduler to be in terms of batches. If the PyTorch scheduler should be stepped '
                  'every batch, set `step_schedulers_every_batch=True`.'))
        return TimeUnit.EPOCH
    else:
        log.info(('Stepping schedulers every batch. '
                  'To step schedulers every epoch, set `step_schedulers_every_batch=False`.'))
        return TimeUnit.BATCH


def _filter_metrics(metrics: Dict[str, Metric], metric_names: Optional[List[str]]) -> Dict[str, Metric]:
    """Filter the metrics based on the given metric_names as regex strings (e.g. 'Accuracy', 'f1' for 'BinaryF1Score', 'Top-.' for 'Top-1 Accuracy' and 'Top-2 Accuracy', etc). If no metric_names are provided, all metrics will be returned."""
    metrics = deepcopy(metrics)
    if metric_names is None:
        return metrics
    filtered_metrics = {}
    for name, metric in metrics.items():
        if any(re.match(f'.*{metric_name}.*', name, re.IGNORECASE) for metric_name in metric_names):
            filtered_metrics[name] = metric
    return filtered_metrics


def _validate_precision(precision: Precision, device: Device):
    if isinstance(device, DeviceCPU) and precision != Precision.FP32:
        raise ValueError(f'{precision} is not supported for CPU training.')


def _compile_schedulers(
    schedulers: Optional[Union[Scheduler, Sequence[Scheduler]]],
    state: State,
    scale_schedule_ratio: float,
) -> List[PyTorchScheduler]:
    compiled_schedulers = []
    for scheduler in ensure_tuple(schedulers):
        if isinstance(scheduler, PyTorchScheduler):
            scale_pytorch_scheduler(scheduler, scale_schedule_ratio)
            compiled_schedulers.append(scheduler)
        else:  # it's a composer scheduler
            compiled_schedulers.append(compile_composer_scheduler(scheduler, state, scale_schedule_ratio))

    return compiled_schedulers


def _set_evaluator_interval_and_subset_num_batches(
    evaluators: Sequence[Evaluator],
    eval_interval: Union[int, str, Time, Callable[[State, Event], bool]],
    subset_num_batches: int,
):
    # convert eval_dataloader to `List[Evaluator]`
    for evaluator in evaluators:
        if evaluator.subset_num_batches is None:
            evaluator.subset_num_batches = subset_num_batches
        if evaluator.eval_interval is None:
            evaluator.eval_interval = eval_interval
        eval_dataloader = evaluator.dataloader.dataloader
        if isinstance(eval_dataloader, collections.abc.Sized) and (evaluator.subset_num_batches is None or
                                                                   evaluator.subset_num_batches == -1):
            try:
                dataloader_len = len(eval_dataloader)
            except TypeError:
                dataloader_len = None
            if dataloader_len == None:
                raise ValueError('eval_subset_num_batches must be set when using an infinite sized '
                                 'eval_dataloader where length is `None`. Otherwise, evaluation will '
                                 'run forever and never terminate.')


def _is_auto_microbatching(device_train_microbatch_size: Optional[Union[int, str]], device: Device):
    if device_train_microbatch_size == 'auto':
        warnings.warn(("`device_train_microbatch_size='auto'` may potentially fail with unexpected "
                       'CUDA errors. Auto microbatching attempts to catch CUDA Out of Memory errors '
                       'and adjust the batch size, but it is possible CUDA will be put into an '
                       'irrecoverable state due to PyTorch bugs, e.g. integer overflow. In this case, '
                       'please manually set device_train_microbatch_size explicitly to an integer '
                       'instead.'))
        if not isinstance(device, DeviceGPU):
            raise ValueError(
                'Can only use adaptive device_train_microbatch_size on GPU. Please set device_train_microbatch_size >= 1.'
            )
        return True
    else:
        return False


def _get_initial_device_train_microbatch_size(device_train_microbatch_size: Optional[Union[int, str]],
                                              auto_microbatching: bool,
                                              train_dataloader: Optional[Iterable]) -> Optional[int]:
    """Sets initial value of device_train_microbatch_size.

    If auto_microbatching, sets initial `device_train_microbatch_size` to per rank batch size. If
    `train_dataloader` is not set yet, returns None and this function will be called again when
    `train_dataloader` is set, such as when `fit()` is called.
    """
    if device_train_microbatch_size is None or auto_microbatching:
        # Return None, this function will be called again when `train_dataloader` is set
        if train_dataloader is None:
            return None
        try:
            batch_size = getattr(train_dataloader, 'batch_size')
        except AttributeError as e:
            # Error message when `device_train_microbatch_size` is None
            # Note: This code path will be removed after `auto` is made default
            if device_train_microbatch_size is None:
                raise ValueError(
                    '`device_train_microbatch_size` must be set when `state.train_dataloader` does not have a `batch_size` attribute.'
                ) from e
            # Error message when `device_train_microbatch_size` is 'auto'
            raise AttributeError(
                "`device_train_microbatch_size='auto'` requires the `state.train_dataloader` to have a `batch_size` attribute."
            ) from e
        return batch_size
    elif isinstance(device_train_microbatch_size, int):
        return device_train_microbatch_size
    else:
        raise ValueError("device_train_microbatch_size must be an int or ``'auto'``")


def _ignore_dataset_state(
    ignore_keys: Optional[Union[List[str], Callable[[Dict], None]]]
) -> Union[List[str], Callable[[Dict], None]]:
    """Extends ``ignore_keys`` so that the dataset state is not loaded from the checkpoint."""
    dataset_state_key = 'state/dataset_state'
    if ignore_keys is None:
        return [dataset_state_key]
    if callable(ignore_keys):

        def ignore_keys_and_dataset_state(state_dict: Dict) -> None:
            ignore_keys(state_dict)  # pyright: ignore[reportGeneralTypeIssues]
            checkpoint.glob_filter([dataset_state_key])(state_dict)

        return ignore_keys_and_dataset_state
    return list(ignore_keys) + [dataset_state_key]


def _is_cuda_oom(e: RuntimeError):
    """Determines if error is CUDA Out of Memory and if auto_microbatching is enabled."""
    if 'CUDA out of memory' in str(e):
        return True
    # With batch_norm, large batch sizes sometimes result in cuDNN instead of Cuda OOMs.
    if 'cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.' in str(
            e):
        warnings.warn('Encountered "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in '
                      'a non-contiguous input." This can happen when the batch_size is too large for the GPU so auto '
                      'auto_microbatching will rerun with a smaller microbatch size value, but there may be a user '
                      'error with non-contiguous inputs.')
        return True
    return False


def _adjust_device_train_microbatch_size(state: State):
    """Adjust device_train_microbatch_size if we encounter OOM.

    Args:
        state (State): State of trainer.
    """
    # If any rank hit CUDA OOM, update device_train_microbatch_size and retry. Raise runtime error
    # if training 1 sample at a time still resulted in CUDA out of memory.
    assert state.device_train_microbatch_size is not None
    if state.device_train_microbatch_size == 1:
        raise RuntimeError(('CUDA out of memory. The train loop failed with an internal microbatch of size 1.'
                            'The GPU does not have enough memory to process even 1 sample during train.'))
    else:
        original_microbatch_size = state.device_train_microbatch_size
        state.device_train_microbatch_size = max(int(original_microbatch_size / 2), 1)
        warnings.warn(
            RuntimeWarning('CUDA out of memory detected. Train microbatch size will be decreased from '
                           f'{original_microbatch_size} -> {state.device_train_microbatch_size}.'))
    # Clear gradients in case failure happened during backwards pass
    if hasattr(state, 'outputs'):
        del state.outputs
    if hasattr(state, 'loss'):
        del state.loss
    for optimizer in state.optimizers:
        optimizer.zero_grad(set_to_none=not state.gigafsdp_enabled)
    if state.scaler is not None:
        # it appears as if this is done by default
        if get_torch_version() < version.parse("2.4"):
            state.scaler._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
    torch.cuda.empty_cache()


def _adjust_device_eval_microbatch_size(evaluator: Evaluator):
    """Adjust device_eval_microbatch_size if we encounter OOM.

    Args:
        evaluator (State): Current evaluator
    """
    # If any rank hit CUDA OOM, update device_eval_microbatch_size and retry. Raise runtime error
    # if evaluating 1 sample at a time still resulted in CUDA out of memory.
    assert evaluator.device_eval_microbatch_size is not None
    if evaluator.device_eval_microbatch_size == 1:
        raise RuntimeError(('CUDA out of memory. The eval loop failed with an internal microbatch of size 1.'
                            'The GPU does not have enough memory to process even 1 sample during eval.'))
    else:
        original_microbatch_size = evaluator.device_eval_microbatch_size
        evaluator.device_eval_microbatch_size = max(int(original_microbatch_size / 2), 1)
        warnings.warn(
            RuntimeWarning('CUDA out of memory detected. Train microbatch size will be decreased from '
                           f'{original_microbatch_size} -> {evaluator.device_eval_microbatch_size}.'))
    torch.cuda.empty_cache()


def _distribute_and_get_random_seed(seed: Optional[int], device: Device):
    if seed is None:
        seed = reproducibility.get_random_seed()

    # Ensure that each process has a seed = rank_zero_seed + global_rank
    # This "deterministically different" seed behavior is required to be able
    # to restore seeds when resuming form checkpoints, since only the
    # `rank_zero_seed` is stored on state.
    if seed < 0 or seed > reproducibility.MAX_SEED:
        raise ValueError(f'Invalid seed: {seed}. It must be on [0; 2**32 - 1)')

    # using int64 to prevent overflow
    rank_zero_seed = device.tensor_to_device(torch.tensor([seed], dtype=torch.int64))
    if dist.get_world_size() > 1:
        dist.broadcast(rank_zero_seed, src=0)
    rank_zero_seed = rank_zero_seed.item()
    assert isinstance(rank_zero_seed, int)
    local_seed = rank_zero_seed + dist.get_global_rank()
    tp_sp_group = dist.get_tp_sp_group()
    if tp_sp_group is not None:
        local_seed = device.tensor_to_device(torch.tensor([local_seed], dtype=torch.int64))
        tp_sp_group.broadcast(local_seed, root=0)
        local_seed = local_seed.item()
    return rank_zero_seed, local_seed


def _get_precision_context(precision: Precision, precision_config: Optional[Dict[str, Any]]):
    return get_precision_context(precision, precision_config)


def _generate_run_name() -> str:
    # change coolname randomness for different names with same seed
    coolname.replace_random(random.Random(os.urandom(128)))
    # prefixing with the time so experiments sorted alphabetically will have the latest experiment last
    generated_run_name = str(int(time.time())) + '-' + coolname.generate_slug(2)
    run_name_list = [generated_run_name]
    # ensure all ranks have the same experiment name
    dist.broadcast_object_list(run_name_list)
    generated_run_name = run_name_list[0]
    return generated_run_name


def create_empty_batch_like(batch: Any) -> Any:
    """Create an empty batch with the same structure as the input batch.

    Args:
        batch: The batch to copy structure from

    Returns:
        An empty batch with the same structure
    """
    # todo: вынести из трейнера
    if isinstance(batch, torch.Tensor):
        return torch.empty_like(batch)
    elif isinstance(batch, (list, tuple)):
        return [create_empty_batch_like(x) for x in batch]
    elif isinstance(batch, dict):
        return {k: create_empty_batch_like(v) for k, v in batch.items()}
    elif any(isinstance(batch, cls) for cls in [str, int, float, MultiTokenEOSCriteria]):
        return batch
    else:
        raise ValueError(f"Unknown batch type = {type(batch)}")


def recursive_split(batch_d: Any, split_num: int, dim: int=0, split_batch_fn: Optional[Callable] = None) -> List[Any]:
    """Recursively splits tensors in the batch along the dimension.

    Args:
        batch_d: The batch dictionary with tensors to split
        split_num: Number of splits to make

    Returns:
        List of split batch dictionaries
    """
    # todo: вынести из трейнера
    if split_num == 1:
        return [batch_d]

    if isinstance(batch_d, dict) and "modality_inputs" in batch_d:
        # NOTE (Sbr): For GigaOmni model we use data_spec.split_batch method to correctly split
        # modality inputs for training with DynamicSP
        assert split_batch_fn is not None, "split_batch_fn must not be None for GigaOmni model training with DynamicSP!"
        return split_batch_fn(batch_d, batch_d["input_ids"].shape[0] // split_num)

    if isinstance(batch_d, torch.Tensor):
        # Calculate the split size
        batch_size = batch_d.shape[0]
        split_size = batch_size // split_num
        remainder = batch_size % split_num

        # Create split sizes list (handling uneven divisions)
        split_sizes = [split_size + (1 if i < remainder else 0) for i in range(split_num)]

        # Split the tensor along the 0th dimension
        return torch.split(batch_d, split_sizes, dim=dim)

    elif isinstance(batch_d, (list, tuple)):
        # Split each element in the list/tuple
        split_results = [recursive_split(x, split_num, dim) for x in batch_d]
        # Transpose the results to group by split index
        return [
            [split_results[j][i] for j in range(len(split_results))]
            for i in range(split_num)
        ]

    elif isinstance(batch_d, dict):
        # Split each value in the dictionary
        split_results = {k: recursive_split(v, split_num, dim) for k, v in batch_d.items()}
        # Group by split index
        return [
            {k: split_results[k][i] for k in split_results}
            for i in range(split_num)
        ]

    elif any(isinstance(batch_d, cls) for cls in [str, int, float, MultiTokenEOSCriteria]):
        # For primitive types, duplicate them for each split
        return [batch_d] * split_num

    else:
        raise ValueError(f"Unknown batch type = {type(batch_d)}")


class Trainer:
    """Train models with Composer algorithms.

    The trainer supports models with :class:`~composer.models.base.ComposerModel` instances.
    The :class:`.Trainer` is highly customizable and can support a wide variety of workloads.
    See the :doc:`training guide</trainer/using_the_trainer>` for more information.

    Example
    --------

    Train a model and save a checkpoint:

    .. testcode::

        import os
        from composer import Trainer

        ### Create a trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            max_duration="1ep",
            eval_dataloader=eval_dataloader,
            optimizers=optimizer,
            schedulers=scheduler,
            device="cpu",
            eval_interval="1ep",
            save_folder="checkpoints",
            save_filename="ep{epoch}.pt",
            save_interval="1ep",
            save_overwrite=True,
        )

        # Fit and run evaluation for 1 epoch.
        # Save a checkpoint after 1 epoch as specified during trainer creation.
        trainer.fit()

    Load the checkpoint and resume training:

    .. testcode::

        # Get the saved checkpoint filepath
        checkpoint_path = trainer.saved_checkpoints.pop()

        # Create a new trainer with the `load_path` argument set to the checkpoint path.
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            max_duration="2ep",
            eval_dataloader=eval_dataloader,
            optimizers=optimizer,
            schedulers=scheduler,
            device="cpu",
            eval_interval="1ep",
            load_path=checkpoint_path,
        )

        # Continue training and running evaluation where the previous trainer left off
        # until the new max_duration is reached.
        # In this case it will be one additional epoch to reach 2 epochs total.
        trainer.fit()


    Args:
        model (ComposerModel): The model to train. Can be user-defined or one of the models included
            with Composer.

            .. seealso:: :mod:`composer.models` for models built into Composer.
        train_dataloader (Iterable | DataSpec | dict, optional): The dataloader, :class:`.DataSpec`,
            or dict of :class:`.DataSpec` kwargs for the training data. In order to specify custom
            preprocessing steps on each data batch, specify a :class:`.DataSpec` instead of a dataloader.
            It is recommended that the dataloader, whether specified directly or as part of a :class:`.DataSpec`,
            should be a :class:`torch.utils.data.DataLoader`.

            .. note:: The ``train_dataloader`` should yield per-rank batches. Each per-rank batch
                will then be further divided based on the ``device_train_microbatch_size`` parameter. For example, if the
                desired optimization batch size is ``2048`` and training is happening across 8 GPUs, then each
                ``train_dataloader`` should yield a batch of size ``2048 / 8 = 256``. If ``device_train_microbatch_size = 128``,
                then the per-rank batch will be divided into ``256 / 128 = 2`` microbatches of size ``128``.

            If ``train_dataloader`` is not specified when constructing the trainer, it must be specified when invoking
            :meth:`.Trainer.fit`.
        train_dataloader_label (str, optional): The label for the train dataloader. (default: ``'train'``)

            This label is used to index the training metrics in
            :attr:`.State.train_metrics`.

            This parameter has no effect if ``train_dataloader`` is not specified.
        train_subset_num_batches (int, optional): If specified, finish every epoch early after training
            on this many batches. This parameter has no effect if it is greater than ``len(train_dataloader)``.
            If ``-1``, then the entire dataloader will be iterated over. (default: ``-1``)

            When using the profiler, it can be helpful to set this parameter to the length of the profile schedule.
            This setting will end each epoch early to avoid additional training that will not be profiled.

            This parameter is ignored if ``train_dataloader`` is not specified.
        spin_dataloaders (bool, optional): If ``True``, dataloaders will be spun up to the current timestamp
            by skipping samples which have already been trained on. If a dataloader has a way to resume from
            the current batch without spinning, this will be a no-op. This ensures dataloaders continue from
            the same batch when resuming training. (default: ``True``)

            .. note:: Spinning dataloaders can be potentially very slow but is required to skip samples which
                have already been trained on. If it is acceptable to repeat samples when resuming training,
                it is possible to resume faster by setting ``spin_dataloaders=False``. This may have severe
                performance implications and is generally not recommended unless you confidently understand the
                implications.
        max_duration (Time | str | int, optional): The maximum duration to train. Can be an integer, which will be
            interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), or a :class:`.Time` object.

            If ``max_duration`` is not specified when constructing the trainer, ``duration`` must be specified when invoking
            :meth:`.Trainer.fit`.
        algorithms (Algorithm | Sequence[Algorithm], optional): The algorithms to use during training. If ``None``, then
            no algorithms will be used. (default: ``None``)

            .. seealso:: :mod:`composer.algorithms` for the different algorithms built into Composer.
        algorithm_passes ([AlgorithmPass | Tuple[AlgorithmPass, int] | Sequence[AlgorithmPass | Tuple[AlgorithmPass, int]], optional):
            Optional list of passes to change order in which algorithms are applied. These passes are merged with the
            default passes specified in :class:`.Engine`. If ``None``, then no additional passes will be used.
            (default: ``None``)

            .. seealso:: :class:`composer.core.Engine` for more information.
        optimizers (torch.optim.Optimizer, optional): The optimizer.
            If ``None``, will be set to ``DecoupledSGDW(model.parameters(), lr=0.1)``. (default: ``None``)

            .. seealso:: :mod:`composer.optim` for the different optimizers built into Composer.
        schedulers (PyTorchScheduler | ComposerScheduler | Sequence[PyTorchScheduler | ComposerScheduler], optional):
            The learning rate schedulers. If ``[]`` or ``None``, the learning rate will be constant.
            (default: ``None``).

            .. seealso:: :mod:`composer.optim.scheduler` for the different schedulers built into Composer.
        scale_schedule_ratio (float, optional): Ratio by which to scale the training duration and learning rate
            schedules. (default: ``1.0``)

            E.g., ``0.5`` makes the schedule take half as many epochs and ``2.0`` makes it take twice as
            many epochs. ``1.0`` means no change.

            This parameter has no effect if ``schedulers`` is not specified.

            .. note ::

                Training for less time, while rescaling the learning rate schedule,
                is a strong baseline approach to speeding up training. E.g., training
                for half duration often yields minor accuracy degradation,
                provided that the learning rate schedule is also rescaled to take half as long.

                To see the difference, consider training for half as long using a cosine
                annealing learning rate schedule. If the schedule is not rescaled,
                training ends while the learning rate is still ~0.5 of the initial LR.
                If the schedule is rescaled with ``scale_schedule_ratio``, the LR schedule
                would finish the entire cosine curve, ending with a learning rate near zero.
        step_schedulers_every_batch (bool, optional): By default, native
            `PyTorch schedulers <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
            are updated every epoch, while :doc:`Composer Schedulers</trainer/schedulers>` are updated every step.
            Setting this to ``True`` will force schedulers to be stepped every batch,
            while ``False`` means schedulers stepped every epoch. ``None`` indicates the default behavior.
            (default: ``None``)
        eval_dataloader (Iterable | DataLoader | DataSpec | Evaluator | Sequence[Evaluator], optional): The :class:`.Iterable`,
            :class:`.DataLoader`, :class:`.DataSpec`, :class:`.Evaluator`, or sequence of evaluators for the evaluation data.

            To evaluate one or more specific metrics across one or more datasets, pass in an
            :class:`.Evaluator`. If a :class:`.DataLoader`, :class:`.DataSpec`, or :class:`.Iterable` is passed in, then all
            metrics returned by ``model.get_metrics()`` will be used during evaluation. If a :class:`.Evaluator`
            is specified in a list, all eval dataloaders must be :class:`.Evaluator` instances.
            ``None`` results in no evaluation. (default: ``None``)
        eval_interval (int | str | Time | (State, Event) -> bool, optional): Specifies how frequently to run evaluation.
            An integer, which will be interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), a :class:`.Time`
            object, or a callable.
            Defaults to ``1`` (evaluate every epoch).

            If an integer (in epochs), :class:`.Time` string, or :class:`.Time` instance, the evaluator will be run
            with this frequency. :class:`.Time` strings or :class:`.Time` instances must have units of
            :attr:`.TimeUnit.BATCH` or :attr:`.TimeUnit.EPOCH`.

            Set to ``0`` to disable evaluation.

            If a callable, it should take two arguments (:class:`.State`, :class:`.Event`) and return a bool
            representing whether the evaluator should be invoked. The event will be either :attr:`.Event.BATCH_END`
            or :attr:`.Event.EPOCH_END`.

            This ``eval_interval`` will apply to any :class:`.Evaluator` in ``eval_dataloader`` that does not specify
            an ``eval_interval`` or if a dataloader is passed in directly. This parameter has no effect if
            ``eval_dataloader`` is not specified.

            When specifying time string or integer for the ``eval_interval``, the evaluator(s) are also run at the ``Event.FIT_END`` if it doesn't
            evenly divide the training duration.

        eval_subset_num_batches (int, optional): If specified, evaluate on this many batches. Defaults to ``-1``,
            which means to iterate over the entire dataloader.

            This parameter has no effect if ``eval_dataloader`` is not specified, it is greater than
            ``len(eval_dataloader)``, or ``eval_dataloader`` is an :class:`.Evaluator` and ``subset_num_batches``
            was specified as part of the :class:`.Evaluator`.
        callbacks (Callback | Sequence[Callback], optional): The callbacks to run during training. If ``None``,
            then no callbacks will be run. (default: ``None``).

            .. seealso:: :mod:`composer.callbacks` for the different callbacks built into Composer.
        loggers (LoggerDestination | Sequence[LoggerDestination], optional): The destinations to log training information to.

            .. seealso:: :mod:`composer.loggers` for the different loggers built into Composer.
        run_name (str, optional): A name for this training run. If not specified, the env var
            `COMPOSER_RUN_NAME` or `RUN_NAME` will be used if set. Otherwise, the timestamp will be
            combined with a :doc:`coolname <coolname:index>`, e.g. ``1654298855-electric-zebra``.
        progress_bar (bool): Whether to show a progress bar. (default: ``True``)
        log_to_console (bool): Whether to print logging statements to the console. (default: ``False``)
        console_stream (TextIO | str, optional): The stream to write to. If a string, it can either be
            ``'stdout'`` or ``'stderr'``. (default: :attr:`sys.stderr`)
        console_log_interval (int | str | Time, optional): Specifies how frequently to log metrics to console.
            An integer, which will be interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), a :class:`.Time`
            object, or a callable. (default: ``1ba``)
            Defaults to ``1ba`` (log metrics every batch).

            If an integer (in epochs), :class:`.Time` string, or :class:`.Time` instance, the metrics will be logged
            with this frequency. :class:`.Time` strings or :class:`.Time` instances must have units of
            :attr:`.TimeUnit.BATCH` or :attr:`.TimeUnit.EPOCH`.

            Set to ``0`` to disable metrics logging to console.
        log_traces (bool): Whether to log traces or not. (default: ``False``)
        auto_log_hparams (bool): Whether to automatically extract hyperparameters. (default: ``False``)
        load_path (str, optional):  The path format string to an existing checkpoint file.

            It can be a path to a file on the local disk, a URL, or if ``load_object_store`` is set, the object name
            for a checkpoint in a cloud bucket. If a URI is specified, ``load_object_store`` does not need to be set.

            If ``None`` then no checkpoint will be loaded. (default: ``None``)
        load_object_store (Union[ObjectStore, LoggerDestination], optional): If the ``load_path`` is in an
            object store (i.e. AWS S3 or Google Cloud Storage), an instance of :class:`.ObjectStore` or
            :class:`.LoggerDestination` which will be used to retrieve the checkpoint. Otherwise, if the
            checkpoint is a local filepath, set to ``None``. Also, it can be ``None`` if the ``load_path`` is
            an S3 URI because the appropriate object store will be automatically constructed in that case.
            Ignored if ``load_path`` is ``None``.
            (default: ``None``)

            Example:

            .. testsetup::

                import composer.trainer

                composer.trainer.trainer.checkpoint.load_checkpoint = lambda *args, **kwargs: None

            .. testcode::

                from composer import Trainer
                from composer.utils import LibcloudObjectStore

                # Create the object store provider with the specified credentials
                creds = {"key": "object_store_key",
                         "secret": "object_store_secret"}
                store = LibcloudObjectStore(provider="s3",
                                            container="my_container",
                                            provider_kwargs=creds)

                checkpoint_path = "./path_to_the_checkpoint_in_object_store"

                # Create a trainer which will load a checkpoint from the specified object store
                trainer = Trainer(
                    model=model,
                    train_dataloader=train_dataloader,
                    max_duration="10ep",
                    eval_dataloader=eval_dataloader,
                    optimizers=optimizer,
                    schedulers=scheduler,
                    device="cpu",
                    eval_interval="1ep",
                    load_path=checkpoint_path,
                    load_object_store=store,
                )
        load_weights_only (bool, optional): Whether or not to only restore the weights from the checkpoint without
            restoring the associated state. Ignored if ``load_path`` is ``None``. (default: ``False``)
        load_strict_model_weights (bool, optional): Ensure that the set of weights in the checkpoint and model must exactly match.
            Ignored if ``load_path`` is ``None``. (default: ``True``)
        load_progress_bar (bool, optional): Display the progress bar for downloading the checkpoint.
            Ignored if ``load_path`` is either ``None`` or a local file path. (default: ``True``)
        load_ignore_keys (List[str] | (Dict) -> None, optional): A list of paths for the ``state_dict`` of the checkpoint,
            which, when provided, will be ignored from the state_dict before a checkpoint is loaded. Each path is a list
            of strings specifying the keys to index into ``state_dict`` joined together with `/` as a separator (as PyTorch
            uses `.` in parameter names). If a prefix is provided, all children are also ignored (see Example 2).
            See :mod:`composer.core.state` for the structure of state_dict.

            Example 1: ``load_ignore_keys = ["state/model/layer1.weights", "state/model/layer1.bias"]`` would ignore
            layer 1 weights and bias.

            Example 2: ``load_ignore_keys = ["state/model/*"]`` would ignore the entire model, which would have the same
            effect as the previous example if there was only 1 layer.

            Example 3: ``load_ignore_keys = ["state/model/layer*.weights"]`` would ignore all weights in the model.

            Example 4: ``load_ignore_keys = ["state/rank_zero_seed", "rng"]`` would reset all randomness when
            loading the checkpoint.

            If a callable, it should take one argument which is the state_dict. The callable is free to arbitrarily modify
            the state_dict before it is loaded.

            (default: ``None``)
        load_force_new_epoch (bool, optional): Whether to start a new epoch with a fresh dataset state when loading
            the checkpoint given by ``load_path``. Use it when training continues from a checkpoint with a changed
            train dataset, or when the dataset state has to be reset for any other reason.

            When enabled, the dataset state is excluded from the checkpoint (as with
            ``load_ignore_keys = ["state/dataset_state"]``), the in-epoch counters of the timestamp
            (``batch_in_epoch``, ``sample_in_epoch``, ``token_in_epoch``) are zeroed, the ``epoch`` is incremented,
            and the train dataset shuffle seed is set back to the configured one, so that the new pass over the data
            has a different shuffle. The model, optimizers and the cumulative counters are loaded as usual.

            Like ``load_ignore_keys``, this applies only to the initial load from ``load_path`` and is disabled when
            ``autoresume`` picks up a checkpoint of the current run, so that a restarted run resumes exactly where it
            left off. Ignored if ``load_path`` is ``None`` or ``load_weights_only`` is ``True``.

            (default: ``False``)
        load_exclude_algorithms (List[str], optional): A list of algorithm names to exclude from loading.
            By default, algorithms with `required_on_load=True` which were enabled when training the loaded
            checkpoint are automatically applied unless they conflict with a user specified algorithm. These
            algorithms often change the model, and not applying them could result in certain layers not having
            weights loaded.

            Example 1: ``load_exclude_algorithms = ["BlurPool"]`` would exclude BlurPool from loading.

            Example 2: ``load_exclude_algorithms = ["FusedLayerNorm", "Alibi"]`` would exclude FusedLayerNorm and Alibi from loading.

            (default: ``None``)
        save_folder (str, optional): Format string for the folder where checkpoints are saved.
            If ``None``, checkpoints will not be saved. Can also be a URI for S3 paths only.
            In the case of an S3 URI, the appropriate `~.RemoteUploader` object will be created
            automatically. (default: ``None``)

            .. seealso:: :class:`~.CheckpointSaver`

            .. note::

                For fine-grained control on checkpoint saving (e.g. to save different types of checkpoints
                at different intervals), leave this parameter as ``None``, and instead pass
                instance(s) of :class:`~.CheckpointSaver` directly as ``callbacks``.
        save_filename (str, optional): A format string describing how to name checkpoints.
            This parameter has no effect if ``save_folder`` is ``None``.
            (default: ``"ep{epoch}-ba{batch}-rank{rank}.pt"``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_latest_filename (str, optional): A format string for the name of a symlink
            (relative to ``save_folder``) that points to the last saved checkpoint.
            This parameter has no effect if ``save_folder`` is ``None``.
            To disable symlinking, set this to ``None``. (default: ``"latest-rank{rank}.pt"``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_overwrite (bool, optional): Whether existing checkpoints should be overridden.
            This parameter has no effect if ``save_folder`` is None. (default: ``False``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_interval (Time | str | int | (State, Event) -> bool): A :class:`Time`, time-string, integer (in epochs),
            or a function that takes (state, event) and returns a boolean whether a checkpoint should be saved.
            This parameter has no effect if ``save_folder`` is ``None``. (default: ``'1ep'``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_weights_only (bool, optional): Whether to save only the model weights instead of the entire training
            state. This parameter has no effect if ``save_folder`` is ``None``. (default: ``False``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_num_checkpoints_to_keep (int, optional): The number of checkpoints to keep locally. The oldest checkpoints
            are removed first. Set to ``-1`` to keep all checkpoints locally. (default: ``-1``)

            Checkpoints will be removed after they have been uploaded. For example, when this callback
            is used in conjunction with the :class:`.RemoteUploaderDownloader`, set this
            parameter to ``0`` to immediately delete checkpoints from the local disk after they have been uploaded to
            the object store.

            This parameter only controls how many checkpoints are kept locally; checkpoints are not deleted from
            remote file systems.
        save_metrics (bool, optional): Whether to save the metrics. By default, metrics are not saved to checkpoint
            as state usually does not need to be preserved and inconsistent state can cause issues when loading.
            (default: ``False``)
        compute_checkpoint_hashes (bool, optional): If True, will compute hashes for checkpoints it saves/loads
            and check that loaded checkpoints are not corrupted.
        autoresume (bool, optional): Whether or not to enable autoresume, which allows for stopping and resuming
            training. This allows use of spot instances, as the training run is now fault tolerant.  This parameter
            requires ``save_folder`` and ``run_name`` to be specified and ``save_overwrite`` to be ``False``.
            (default: ``False``)

            When enabled, the save_folder is checked for checkpoints of the format ``"{save_folder}/{save_latest_filename}"``,
            which are loaded to continue training. If no local checkpoints are found, each logger is checked for potential
            remote checkpoints named ``"{save_folder}/{save_latest_filename}"``. Finally, if no logged checkpoints are found, ``load_path`` is
            used to load a checkpoint if specified. This should only occur at the start of a run using autoresume.

            For example, to run a fine-tuning run on a spot instance, ``load_path`` would be set to the original
            weights and an object store logger would be added. In the original run, ``load_path`` would be used
            to get the starting checkpoint. For any future restarts, such as due to the spot instance being killed,
            the loggers would be queried for the latest checkpoint the object store logger would be downloaded and
            used to resume training.
        fsdp_config (Dict[str, Any], optional): Configuration for FSDP.
            See :doc:`FSDP Documentation </notes/distributed_training>` for more details.
            To use FSDP with default values, set to the empty dictionary ``{}``. To
            disable FSDP, set to ``None``. (default: ``None``)
        fsdp_auto_wrap (bool, optional): option to let trainer wrap the module, or if
            the module is already wrapped outside, allow the user to disable auto-wrapping.

        device (Device | str, optional): The device to use for training, which can be ``'cpu'``, ``'gpu'``,
            ``'tpu'``, or ``'mps'``. (default: ``None``)

            The default behavior sets the device to ``'gpu'`` if CUDA is available, and otherwise ``'cpu'``.
        precision (Precision | str, optional): Numerical precision to use for training. One of ``fp32``, ``amp_bf16``
            or ``amp_fp16`` (recommended). (default: ``Precision.FP32`` if training on CPU; ``Precision.AMP_FP16`` if
            training on GPU)
        precision_config (Optional[Dict[str, Any]]): The config for FP8 scaling strategy. See parameters for
            `DelayedScaling <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html?highlight=delayedscaling#transformer_engine.common.recipe.DelayedScaling>`_.
        device_train_microbatch_size (Union[int, str), optional): The number of samples to process on each device per
            microbatch during training. Gradients are summed over the microbatches per device. If set to ``auto``,
            dynamically decreases device_train_microbatch_size if microbatch is too large for GPU. (default: ``None``)

            .. note:: This is implemented by taking the batch yielded by the ``train_dataloader`` and splitting
                it into sections of size ``device_train_microbatch_size``. If the batch size of the dataloader
                is not divisible by ``device_train_microbatch_size``, the last section will be potentially smaller.
        seed (int, optional): The seed used in randomization. If ``None``, then a random seed
            will be created. (default: ``None``)

            .. note:: In order to get reproducible results, call the
                :func:`.seed_all` function at the start of your script with the seed
                passed to the trainer. This will ensure any initialization done before the trainer init
                (ex. model weight initialization) also uses the provided seed.

            .. seealso:: :mod:`composer.utils.reproducibility` for more details on reproducibility.
        deterministic_mode (bool, optional): Run the model deterministically. (default: ``False``)

            .. note:: This is an experimental feature. Performance degradations expected. Certain Torch modules may
                not have deterministic implementations, which will result in a crash.

            .. note:: In order to get reproducible results, call the
                :func:`.configure_deterministic_mode` function at the start of your script.
                This will ensure any initialization done before the trainer init also runs deterministically.

            .. seealso:: :mod:`composer.utils.reproducibility` for more details on reproducibility.
        dist_timeout (float, optional): Timeout, in seconds, for initializing the distributed process group.
            (default: ``300.0``)

        profiler (Profiler, optional): The profiler, if profiling should be enabled. (default: ``None``)

            .. seealso::

                See the :doc:`Profiling Guide </trainer/performance_tutorials/profiling>` for
                additional information.
        python_log_level (str, optional): The Python log level to use for log statements in the :mod:`composer`
            module. (default: ``None``). If it is ``None``, python logging will not be configured (i.e.
            ``logging.basicConfig`` won't be called).

            .. seealso:: The :mod:`logging` module in Python.
        compile_config (Dict[str, Any], optional): Configuration for torch compile. Only supported with PyTorch 2.0 or higher.
            Checkout [`torch.compile`](https://pytorch.org/get-started/pytorch-2.0/) for more details.
            To use torch compile with default values, set it to empty dictionary ``{}``.
            To use torch compile with custom config, set to a dictionary such as ``{'mode': 'max-autotune'}``.
            To disable torch compile, set to ``None``. (default: ``None``)
        zero_nan_grads (bool): enables hook that zero grads with nans

    Attributes:
        state (State): The :class:`.State` object used to store training state.
        evaluators (List[Evaluator]): The :class:`.Evaluator` objects to use for validation
            during training.
        logger (Logger): The :class:`.Logger` used for logging.
        engine (Engine): The :class:`.Engine` used for running callbacks and algorithms.
    """

    def __init__(
        self,
        *,
        # The Model
        model: ComposerModel,

        # Train Dataloader
        train_dataloader: Optional[Union[Iterable, DataSpec, Dict[str, Any]]] = None,
        train_dataloader_label: str = 'train',
        train_subset_num_batches: int = -1,
        spin_dataloaders: bool = True,

        # Global Train batch size
        global_train_batch_size: Union[int, str] = -1,

        # Stopping Condition
        max_duration: Optional[Union[int, str, Time]] = None,

        # Algorithms
        algorithms: Optional[Union[Algorithm, Sequence[Algorithm]]] = None,

        # Engine Pass Registration
        algorithm_passes: Optional[Union[AlgorithmPass, Tuple[AlgorithmPass, int],
                                         Sequence[Union[AlgorithmPass, Tuple[AlgorithmPass, int]]]]] = None,

        # Optimizers and Scheduling
        optimizers: Optional[torch.optim.Optimizer] = None,
        schedulers: Optional[Union[ComposerScheduler, PyTorchScheduler, Sequence[Union[ComposerScheduler,
                                                                                       PyTorchScheduler]]]] = None,
        scale_schedule_ratio: float = 1.0,
        step_schedulers_every_batch: Optional[bool] = None,

        # Evaluators
        eval_dataloader: Optional[Union[Iterable, DataSpec, Evaluator, Sequence[Evaluator]]] = None,
        eval_interval: Union[int, str, Time, Callable[[State, Event], bool]] = 1,
        eval_subset_num_batches: int = -1,

        # Callbacks and Logging
        callbacks: Optional[Union[Callback, Sequence[Callback]]] = None,
        loggers: Optional[Union[LoggerDestination, Sequence[LoggerDestination]]] = None,
        run_name: Optional[str] = None,
        progress_bar: bool = True,
        log_to_console: bool = False,
        console_stream: Union[str, TextIO] = 'stderr',
        console_log_interval: Union[int, str, Time] = '1ba',
        log_traces: bool = False,
        auto_log_hparams: bool = False,

        # Load Checkpoint
        load_path: Optional[str] = None,  # typing might be off
        teacher_load_path: Optional[str] = None,
        load_object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
        load_weights_only: bool = False,
        load_strict_model_weights: bool = True,
        load_progress_bar: bool = True,
        load_ignore_keys: Optional[Union[List[str], Callable[[Dict], None]]] = None,
        load_force_new_epoch: bool = False,
        load_exclude_algorithms: Optional[List[str]] = None,

        # Save Checkpoint
        save_folder: Optional[str] = None,
        save_filename: str = 'ep{epoch}-ba{batch}-rank{rank}.pt',
        save_latest_filename: Optional[str] = 'latest-rank{rank}.pt',
        save_overwrite: bool = False,
        save_interval: Union[str, int, Time, Callable[[State, Event], bool]] = '1ep',
        stable_save_interval: Union[None, str, int, Time, Callable[[State, Event], bool]] = None,
        stable_save_postfix: str = ".stable",
        save_weights_only: bool = False,
        save_lora_adapter_only: bool = False,
        save_num_checkpoints_to_keep: int = -1,
        save_metrics: bool = False,
        compute_checkpoint_hashes: bool = True,
        save_checkpoint_async: bool = False,

        # Graceful Resumption
        autoresume: bool = False,

        fsdp_config: Optional[Dict[str, Any]] = None,
        fsdp_auto_wrap: bool = True,

        # System/Numerics
        device: Optional[Union[str, Device]] = None,
        precision: Optional[Union[str, Precision]] = None,
        precision_config: Optional[Dict[str, Any]] = None,
        device_train_microbatch_size: Optional[Union[int, str]] = None,

        # Reproducibility
        seed: Optional[int] = None,
        deterministic_mode: bool = False,

        # Distributed Training
        dist_timeout: float = 300.0,

        # Profiling
        profiler: Optional[Profiler] = None,

        # Python logging
        python_log_level: Optional[str] = None,

        # compile config for PyTorch 2.0 or higher
        compile_config: Optional[Dict[str, Any]] = None,

        # zero grads containing nans
        zero_nan_grads: bool = False,
    ):
        if not using_torch_2_1_1():
            raise ValueError(
                f'Please use torch version >= 2.1.1. You have torch version {torch.__version__}')

        tp_zero_rank = dist.get_tp_group_rank()
        if zero_nan_grads:
            enable_zero_nan_grads()

        self.global_train_batch_size = int(global_train_batch_size)

        self.auto_log_hparams = auto_log_hparams
        self.python_log_level = python_log_level
        if self.python_log_level is not None:
            logging.basicConfig(
                # Example of format string
                # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: composer.trainer.trainer: Using precision Precision.FP32
                # Including the PID and thread name to help with debugging dataloader workers and callbacks that spawn background
                # threads / processes
                format=
                f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s'
            )
            logging.getLogger('composer').setLevel(self.python_log_level.upper())

        algorithms = list(ensure_tuple(algorithms))

        # Device
        device = get_device(device)

        # Precision
        if precision is None:
            precision = Precision.AMP_FP16 if isinstance(device, DeviceGPU) else Precision.FP32
        elif isinstance(precision, str):
            precision = Precision(precision)
        _validate_precision(precision, device)

        # check if provided model is compiled or not
        is_model_compiled = False

        from torch._dynamo import OptimizedModule
        if isinstance(model, OptimizedModule):
            log.warning(f'Provided `model` is already compiled with `torch.compile`. Ignoring ' +
                        f'parameter `compile_config` if provided. If you would like `Trainer` ' +
                        f'to takes care of model compilation, provide a not-compiled model and ' +
                        f'`compile_config` parameter.')
            # The `torch.compile` function returns an object of type `torch._dynamo.OptimizedModule`
            # which wraps the original `nn.Module` object and later patches its forward method to
            # optimized `self.forward` method.
            is_model_compiled = True
            compiled_model = model._orig_mod
            if not isinstance(compiled_model, ComposerModel):
                raise ValueError(f'Provided `model` must be a subclass of ComposerModel. ' +
                                    f'Instead found as type `{type(compiled_model)}`')
            compiled_model.forward = model.dynamo_ctx(
                compiled_model.forward)  # pyright: ignore [reportGeneralTypeIssues]
            model = compiled_model

        # Microbatching
        auto_microbatching = _is_auto_microbatching(device_train_microbatch_size, device=device)
        if auto_microbatching and profiler:
            raise ValueError("`device_train_microbatch_size='auto'` is not compatible with the profiler. It is "
                             "recommended to run a mini-run with `device_train_microbatch_size='auto'` to identify "
                             'the optimal device_train_microbatch_size value and then manually specify that in a '
                             'second run with profiler.')
        self.first_batch_complete = False
        # If auto_microbatching is True or `device_train_microbatch_size` is not specified, the microbatch size
        # will be determined when dataloader is specified. train_dataloader is parsed after `Event.INIT` or in
        # fit()
        device_train_microbatch_size = _get_initial_device_train_microbatch_size(device_train_microbatch_size,
                                                                                 auto_microbatching, None)

        assert not isinstance(device_train_microbatch_size, str)

        # Distributed
        if fsdp_config is not None or dist.get_world_size() > 1:
            # FSDP require torch.distributed to be initialized, even if the world size is 1
            # And torch.distributed is always required for multi-rank training
            # ---
            # NOTE (Sbr): This is a second time we try to initialize dist after `train.py`,
            # but we just return here as it is already initialized.
            _sp_size = fsdp_config.get("sp_size", 1)
            if isinstance(_sp_size, list):
                self.dynamic_sp = _sp_size
                self.dynamic_sp = sorted(self.dynamic_sp, key=lambda x: x["max_seq_len"])
            elif isinstance(_sp_size, int):
                self.dynamic_sp = [{"sp_size": _sp_size, "max_seq_len": -1}]
            else:
                raise ValueError(f"Invalid fsdp_config.sp_size: {_sp_size=}")

            dist.initialize_dist(
                device,
                tp_size=fsdp_config.get("tp_size", 1),
                ep_size=fsdp_config.get("ep_size", 1),
                sp_size=_sp_size,
                timeout=dist_timeout
            )

        # Reproducibility
        rank_zero_seed, seed = _distribute_and_get_random_seed(seed, device)
        # If hparams is used to create the Trainer this function is called twice
        # which is okay because all runs with the hparams codepath will do this
        reproducibility.seed_all(seed)
        if deterministic_mode:
            reproducibility.configure_deterministic_mode()

        # Optimizers and Schedulers
        if optimizers is None:
            try:
                optimizers = DecoupledSGDW(model.parameters(), lr=0.1)
                # hard-coding the optimizer in the warning, as repr(optimizers) would print an annoying, multi-line warning
                warnings.warn(('No optimizer was specified. Defaulting to '
                               f"{type(optimizers).__name__}(lr={optimizers.defaults['lr']})"))
            except ValueError as e:
                if 'optimizer got an empty parameter list' in str(e):
                    warnings.warn(
                        'No optimizer was specified, and the model does not have parameters. Skipping auto-creating optimizer.'
                    )
                else:
                    raise

        if optimizers is not None:
            num_optimizers = len(ensure_tuple(optimizers))
            if num_optimizers != 1:
                raise NotImplementedError(f'Only one optimizer is supported; found {num_optimizers} optimizers')

        # Move the model and optimizers to the device
        if fsdp_config is None:
            # check if model is already on tpu
            if isinstance(device, DeviceTPU) and 'xla' not in str(next(model.parameters()).device):
                raise ValueError(
                    'Use model.to(xm.xla_device()) to set the model to the TPU before providing to the trainer.')
            else:
                model = device.module_to_device(model)
                # Move any remaining optimizer parameters onto the device
                # It is possible that optimizer initialize created some internal tensors on CPU
                # that need to be moved onto GPU.
            optimizers = map_collection(optimizers, device.optimizer_to_device)

        # Run Name
        run_name = os.getenv('COMPOSER_RUN_NAME', None) if run_name is None else run_name
        run_name = os.getenv('RUN_NAME', None) if run_name is None else run_name
        if run_name is None:
            if autoresume:
                raise ValueError('When autoresume=True, the `run_name` must be specified.')
            run_name = _generate_run_name()
        log.info('Run name: %s', run_name)

        # Create the State
        self.state = State(
            rank_zero_seed=rank_zero_seed,
            algorithms=algorithms,
            model=model,
            device=device,
            callbacks=callbacks,
            device_train_microbatch_size=device_train_microbatch_size,
            auto_microbatching=auto_microbatching,
            precision=precision,
            precision_config=precision_config,
            optimizers=optimizers,
            run_name=run_name,
            save_metrics=save_metrics,
            compute_checkpoint_hashes=compute_checkpoint_hashes,
            fsdp_config=set_fsdp_default(fsdp_config) if fsdp_config is not None else None,
            fsdp_auto_wrap=fsdp_auto_wrap,
        )

        # Console Logging
        loggers = list(ensure_tuple(loggers))

        # Profiler
        if profiler is not None:
            warnings.warn('The profiler is enabled. Using the profiler adds additional overhead when training.')
            self.state.profiler = profiler
            for remote_uri in profiler.remote_filenames:
                remote_ud = maybe_create_remote_uploader_downloader_from_uri(uri=remote_uri, loggers=loggers)
                if remote_ud is not None:
                    loggers.append(remote_ud)
            self.state.profiler.bind_to_state(self.state)

        if progress_bar and log_to_console:
            warnings.warn(
                'Setting both `progress_bar` and `log_to_console` both to True is not recommended and will'
                'lead to duplicate logs and weird formatting issues. Please set one of them to False for a better logging experience.'
            )

        if any(isinstance(x, ProgressBarLogger) for x in loggers):
            warnings.warn(
                Warning((
                    f'Specifying the {ProgressBarLogger.__name__} via `loggers` is not recommended as '
                    'any values set for the following Trainer arguments will be ignored: `progress_bar`, `console_stream`, or `log_traces`. '
                    'The recommended way of enabling a progress bar is to set `progress_bar` to True instead of '
                    f'constructing a {ProgressBarLogger.__name__} instance.')))
        else:
            if progress_bar:
                loggers.append(ProgressBarLogger(stream=console_stream, log_traces=log_traces))

        # Console Logging
        if any(isinstance(x, ConsoleLogger) for x in loggers):
            warnings.warn(
                Warning((
                    f'Specifying the {ConsoleLogger.__name__} via `loggers` is not recommended as '
                    'any values set for the following Trainer arguments will be ignored: `log_to_console`, `console_stream`, `log_traces`, and `console_log_interval`. '
                    'The recommended way of enabling a console logging is to set `log_to_console` to True instead of '
                    f'constructing a {ConsoleLogger.__name__} instance.')))
        else:
            if log_to_console:
                loggers.append(
                    ConsoleLogger(stream=console_stream, log_interval=console_log_interval, log_traces=log_traces))

        # MosaicML Logger
        # Keep MosaicML logger above the RemoteUploaderDownloader so that fit end is reported before the final checkpoint begins uploading
        if os.environ.get(MOSAICML_PLATFORM_ENV_VAR, 'false').lower() == 'true' and os.environ.get(
                MOSAICML_ACCESS_TOKEN_ENV_VAR) is not None and not any(isinstance(x, MosaicMLLogger) for x in loggers):
            log.info('Detected run on MosaicML platform. Adding MosaicMLLogger to loggers.')
            mosaicml_logger = MosaicMLLogger()
            loggers.append(mosaicml_logger)

        if save_folder is not None:
            remote_ud = maybe_create_remote_uploader_downloader_from_uri(save_folder, loggers)
            if remote_ud is not None:
                loggers.append(remote_ud)

        # Logger
        self.logger = Logger(state=self.state, destinations=loggers)

        if save_latest_filename is not None:
            remote_ud_has_format_string = [
                isinstance(logger_destination, RemoteUploaderDownloader) and
                logger_destination.file_path_format_string != '{remote_file_name}'
                for logger_destination in self.logger.destinations
            ]
            if any(remote_ud_has_format_string):
                raise ValueError(
                    'Specifying a `file_path_format_string` to a `RemoteUploaderDownloader` is not currently supported while using `save_latest_filename`. '
                    'Please specify the path formatting via `save_folder`, `save_filename`, and `save_latest_filename`')

        # Callbacks
        self.state.callbacks[:] = list(cast(List[Callback], loggers)) + self.state.callbacks

        # Checkpoint Saving
        self._checkpoint_saver = None
        latest_remote_file_name = None
        if save_folder is not None:
            if save_weights_only:
                log.info(
                    'save_weights_only=True now also saves metadata and integrations! Please adjust your workflow accordingly.'
                )

            _, _, parsed_save_folder = parse_uri(save_folder)

            # If user passes a URI with s3:// and a bucket_name, but no other
            # path then we assume they just want their checkpoints saved directly in their
            # bucket.
            if parsed_save_folder == '':
                folder = '.'
                remote_file_name = save_filename
                latest_remote_file_name = save_latest_filename

            # If they actually specify a path, then we use that for their local save path
            # and we prefix save_filename with that path for remote_file_name.
            else:
                folder = parsed_save_folder
                remote_file_name = str(Path(parsed_save_folder) / Path(save_filename))
                if save_latest_filename is not None:
                    latest_remote_file_name = str(Path(parsed_save_folder) / Path(save_latest_filename))
                else:
                    latest_remote_file_name = None

            self._checkpoint_saver = CheckpointSaver(
                folder=folder,
                filename=save_filename,
                remote_file_name=remote_file_name,
                latest_filename=save_latest_filename,
                latest_remote_file_name=latest_remote_file_name,
                overwrite=save_overwrite,
                weights_only=save_weights_only,
                save_lora_adapter_only=save_lora_adapter_only,
                save_interval=save_interval,
                stable_save_interval=stable_save_interval,
                stable_save_postfix=stable_save_postfix,
                num_checkpoints_to_keep=save_num_checkpoints_to_keep,
                async_save=save_checkpoint_async,
            )
            self.state.callbacks.append(self._checkpoint_saver)

        # The Engine
        self.engine = Engine(state=self.state, logger=self.logger, algorithm_passes=algorithm_passes)

        # Set the logger
        self.state.model.logger = self.logger

        # Run Event.INIT
        self.engine.run_event(Event.INIT)

        # Log hparams.
        if self.auto_log_hparams:
            self.local_hparams = extract_hparams(locals())
            self.logger.log_hyperparameters(self.local_hparams)

        # Log composer version
        composer_env_dict = get_composer_env_dict()
        self.logger.log_hyperparameters({'composer_version': composer_env_dict['composer_version']})
        self.logger.log_hyperparameters({'composer_commit_hash': str(composer_env_dict['composer_commit_hash'])})

        # Log gpus and nodes.
        device_name = self.state.device.__class__.__name__.lstrip('Device').lower()
        self.logger.log_hyperparameters({
            'num_nodes': int(dist.get_world_size() / dist.get_local_world_size()),
            f'num_{device_name}s_per_node': dist.get_local_world_size(),
            'node_name': os.environ.get('NODENAME', 'unknown because NODENAME environment variable not set')
        })

        if not isinstance(self.state.model, ComposerModel):
            raise ValueError('Provided model must be a subclass of ComposerModel.')

        # After running Event.INIT, then set the "optional" elements of state that could be passed in on FIT instead of INIT
        # Setting these attributes here ensures that algorithms do not depend on unavailable attributes during Event.INIT

        # Metrics and Evaluators
        # Set state.train_metrics and state.eval_metrics here to allow callbacks / algs to potentially
        # change the model, which could change what metrics are computed
        self.state.train_metrics = deepcopy(self.state.model.get_metrics(is_train=True))
        self.state.eval_metrics = {}
        if eval_dataloader is None:
            evaluators: List[Evaluator] = []
        else:
            eval_metrics = deepcopy(self.state.model.get_metrics(is_train=False))
            model_metric_names = [str(k) for k in eval_metrics.keys()]
            eval_dataloader = ensure_tuple(eval_dataloader)

            evaluator_types = [isinstance(evaluator, Evaluator) for evaluator in eval_dataloader]
            if any(evaluator_types) and not all(evaluator_types):
                raise ValueError('Mixing Evaluator with other classes is not allowed, please wrap'
                                 'all other classes with the Evaluator class. These are the classes'
                                 'that were detected:' + str([type(evaluator) for evaluator in eval_dataloader]))

            evaluators = [
                ensure_evaluator(evaluator, default_metric_names=model_metric_names) for evaluator in eval_dataloader
            ]
            # match metric names to model metrics
            self.state.eval_metrics = {
                evaluator.label: _filter_metrics(eval_metrics, evaluator.metric_names) for evaluator in evaluators
            }

            _set_evaluator_interval_and_subset_num_batches(
                evaluators=evaluators,
                eval_interval=eval_interval,
                subset_num_batches=eval_subset_num_batches,
            )

            for evaluator in evaluators:
                validate_eval_automicrobatching(evaluator.auto_microbatching, self.state.device)
        if len(evaluators) == 0:
            if eval_subset_num_batches != -1:
                raise ValueError(
                    f'Specifying `eval_subset_num_batches={eval_subset_num_batches}` without an `eval_dataloader` '
                    'has no effect. If trying to run an evaluator, make sure `eval_dataloader` is specified. '
                    'Otherwise, set `eval_subset_num_batches` to default value -1.')
            if eval_interval != 0 and eval_interval != 1 and not os.getenv("SBR_IGNORE_EVAL_INTERVAL", False):
                raise ValueError(
                    f'Specifying `eval_interval={eval_interval}` without an `eval_dataloader` has no effect. '
                    'If trying to run an evaluator, make sure `eval_dataloader` is specified. Otherwise, '
                    'set `eval_interval` to 0 or default value 1.')

        self.state.evaluators = evaluators

        # Train Dataloader
        self._train_data_spec = None if train_dataloader is None else ensure_data_spec(train_dataloader)
        if self._train_data_spec is not None:
            self.state.set_dataloader(self._train_data_spec.dataloader, train_dataloader_label,
                                      train_subset_num_batches)
            if isinstance(self.state.device, DeviceTPU):
                self.state.train_dataloader = pl.MpDeviceLoader(self.state.dataloader, xm.xla_device())
            else:
                self.state.train_dataloader = self.state.dataloader
            self.state.device_train_microbatch_size = _get_initial_device_train_microbatch_size(
                self.state.device_train_microbatch_size, self.state.auto_microbatching, self.state.train_dataloader)
        self.spin_dataloaders = spin_dataloaders

        # Max Duration
        if max_duration is not None:
            self.state.max_duration = ensure_time(max_duration, TimeUnit.EPOCH)

        self.logger.log_hyperparameters({'rank_zero_seed': rank_zero_seed})

        # Schedulers
        self.state.schedulers = _compile_schedulers(schedulers, self.state, scale_schedule_ratio)
        if scale_schedule_ratio != 1.0:
            if len(self.state.schedulers) == 0:
                raise ValueError('Specifying `scale_schedule_ratio` without `schedulers` has no effect.')
            self.state.max_duration = _scale_max_duration_by_ssr(scale_schedule_ratio, self.state.max_duration)

        if step_schedulers_every_batch is None:
            self._scheduler_step_frequency = _get_default_scheduler_frequency(schedulers)
        else:
            self._scheduler_step_frequency = TimeUnit.BATCH if step_schedulers_every_batch else TimeUnit.EPOCH

        # Some algorithms require specific settings
        self._backwards_create_graph = any((x.backwards_create_graph for x in self.state.algorithms))
        self._find_unused_parameters = any((x.find_unused_parameters for x in self.state.algorithms))

        # Suppressing GradScaler warnings as they are always created
        # self._use_grad_scaling() will raise a RuntimeError if grad scaling is not available when it is required
        warnings.filterwarnings(action='ignore', message='torch.cuda.amp.GradScaler')
        self.state.scaler = ClosureGradScaler() if self._use_closures() else GradScaler()

        if self.state.fsdp_config is not None:
            if version.parse(torch.__version__) < version.parse('1.13.0'):
                raise RuntimeError('To use FSDP with Composer, you must use torch>=1.13.0.')
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            # This state should never be reached, but we raise a ValueError just in case
            if self._use_closures() and self.state.precision == Precision.AMP_FP16:
                raise ValueError(f'Using closures and precision {self.state.precision} is not supported'
                                 f' with FSDP. Please use another optimizer or precision type.')
            self.state.scaler = ShardedGradScaler()

        # suppressing FSDP warning when auto grad accum exits the forward pass before completing
        warnings.filterwarnings(action='ignore', message='Forward order differs from that of the first iteration')

        # If using DDP, we need to wrap the ComposerModel but store a reference to the
        # original model for functions like `eval_forward`, `get_metrics`, etc.
        self._original_model = self.state.model

        # If using PyTorch DDP, the model must be loaded before it is wrapped with DDP.
        # If using FSDP, the model must be wrapped and then loaded unless loading a monolith
        # checkpoint on rank 0 only, in which case the model be loaded before it is wrapped.

        # FSDP wrap if not using monolith checkpoint on rank 0 only
        if self.state.fsdp_config is not None and fsdp_auto_wrap and not self.state.load_fsdp_monolith_rank0_only:
            with reproducibility.seed_context(self.state.rank_zero_seed):
                prepare_fsdp_module(model, optimizers, self.state.fsdp_config, precision, device, auto_microbatching)

        # Run Event.BEFORE_LOAD
        self.engine.run_event(Event.BEFORE_LOAD)
        # Load Checkpoint
        self._rng_state = None
        # If autoresume is enabled, first check for existing checkpoints to load
        if autoresume:
            log.info('Searching for a previous checkpoint to autoresume')
            error_message = ''
            if save_folder is None:
                error_message += 'The `save_folder` must be specified when autoresume is enabled. '
            if save_overwrite:
                error_message += textwrap.dedent(
                    'The flag `save_overwrite` must be False when autoresume is enabled as autoresume always loads the '
                    'latest existing checkpoint in `save_folder`. ')
            if save_latest_filename is None:
                error_message += 'The `save_latest_filename` must be specified so autoresume knows where to load checkpoints from. '
            if run_name is None:
                error_message += 'The `run_name` must be specified when using autoresume so Event.INIT is run with the correct run name. '
            if error_message != '':
                raise ValueError(error_message)
            assert save_folder is not None
            assert save_latest_filename is not None

            remote_ud_has_multiple_concurrent_uploads = [
                isinstance(logger_destination, RemoteUploaderDownloader) and
                logger_destination._num_concurrent_uploads != 1 for logger_destination in self.logger.destinations
            ]
            if any(remote_ud_has_multiple_concurrent_uploads):
                raise ValueError(
                    'Multiple concurrent uploads is not currently supported when using autoresume. Please set `num_concurrent_uploads` to 1 '
                    'for all `RemoteUploaderDownloader` instances.')
            assert latest_remote_file_name is not None
            if self.state.fsdp_elastic_sharded_enabled:
                ar_object_store = maybe_create_object_store_from_uri(save_folder)
                # Symlink is on object store.
                if ar_object_store is not None:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        local_symlink_file = str(Path(temp_dir) / Path('autoresume.symlink'))
                        formatted_latest_remote_file_name = format_name_with_dist(latest_remote_file_name,
                                                                                  self.state.run_name) + '.symlink'
                        rank0_formatted_latest_remote_file_name = dist.all_gather_object(
                            formatted_latest_remote_file_name)[0]
                        try:
                            ar_object_store.download_object(rank0_formatted_latest_remote_file_name, local_symlink_file)
                            with open(local_symlink_file, 'r') as f:
                                real_path = f.read()
                                log.debug(f'Read path {real_path} from symlink file')
                            autoresume_checkpoint_path = ar_object_store.get_uri(real_path)
                        except FileNotFoundError:
                            autoresume_checkpoint_path = None
                # Symlink is local.
                else:
                    save_latest_filename = format_name_with_dist(save_latest_filename, self.state.run_name)
                    filenames_list = dist.all_gather_object(save_latest_filename)

                    tp_size = dist.get_tp_group_size() or 0
                    ep_size = dist.get_ep_group_size() or 0

                    assert tp_size % 2 == 0 and ep_size % 2 == 0

                    if dist.get_tp_group() is None and dist.get_ep_group() is None:
                        rank0_save_idx = 0
                    elif dist.get_tp_group() is None and dist.get_ep_group() is not None:
                        rank0_save_idx = dist.get_ep_group_rank()
                    elif dist.get_tp_group() is not None and dist.get_ep_group() is None:
                        rank0_save_idx = dist.get_tp_group_rank()
                    elif dist.get_tp_group() is not None and dist.get_ep_group() is not None:
                        rank0_save_idx = (
                            dist.get_tp_group_rank()
                            if tp_size > ep_size
                            else dist.get_ep_group_rank()
                        )
                    else:
                        raise NotImplementedError("Call the support line: +7 (926) 941 63 38")

                    rank0_save_latest_filename = filenames_list[rank0_save_idx]
                    save_folder = format_name_with_dist(save_folder, self.state.run_name)
                    latest_checkpoint_path = os.path.join(save_folder, rank0_save_latest_filename)

                    if os.path.exists(latest_checkpoint_path):
                        latest_checkpoint_path = os.path.join(os.path.dirname(latest_checkpoint_path),
                                                              os.readlink(latest_checkpoint_path))
                        autoresume_checkpoint_path = latest_checkpoint_path
                    else:
                        autoresume_checkpoint_path = None

            # Standard non-elastic codepath for autoresume.
            else:
                autoresume_checkpoint_path = self._get_autoresume_checkpoint(
                    save_folder=save_folder,
                    save_latest_filename=save_latest_filename,
                    save_latest_remote_file_name=latest_remote_file_name,
                    loggers=loggers,
                    load_progress_bar=load_progress_bar)

            # Found latest checkpoint path, load that instead
            if autoresume_checkpoint_path:
                load_path = autoresume_checkpoint_path
                # Disable object_store since _get_autoresume_checkpoint will download the checkpoint
                # To the save folder, if needed.
                load_object_store = None
                # Set load arguments to defaults as this applies only to the initial non-autoresume
                # load. We do not reset `load_strict_model_weights` for models with frozen layers.
                load_weights_only = False
                load_ignore_keys = None
                load_force_new_epoch = False
                load_exclude_algorithms = None
                log.info('Autoresuming training from checkpoint')
            else:
                log.info('No previous autoresume checkpoint found')
        if load_force_new_epoch:
            if load_path is None:
                warnings.warn('`load_force_new_epoch` is ignored as no checkpoint is loaded from `load_path`.')
                load_force_new_epoch = False
            elif load_weights_only:
                warnings.warn('`load_force_new_epoch` is ignored as `load_weights_only` is set, so neither the '
                              'timestamp nor the dataset state is loaded from the checkpoint.')
                load_force_new_epoch = False

        # Actually load the checkpoint from potentially updated arguments
        if load_path is not None:
            if load_force_new_epoch:
                # The dataset state of the checkpoint describes the position inside the interrupted epoch of the
                # previous dataset, so it must not be loaded. `State.force_new_epoch()` below resets the rest.
                load_ignore_keys = _ignore_dataset_state(load_ignore_keys)
            if load_object_store is None:
                load_object_store = maybe_create_object_store_from_uri(load_path)
            if isinstance(load_object_store, WandBLogger):
                import wandb
                if wandb.run is None:
                    load_object_store.init(self.state, self.logger)

            _, _, parsed_load_path = parse_uri(load_path)

            if teacher_load_path is not None:
                _, _, parsed_teacher_load_path = parse_uri(teacher_load_path)
            else:
                parsed_teacher_load_path = None

            self._rng_state = checkpoint.load_checkpoint(
                state=self.state,
                logger=self.logger,
                path=parsed_load_path,
                teacher_path=parsed_teacher_load_path,
                object_store=load_object_store,
                load_weights_only=load_weights_only,
                strict_model_weights=load_strict_model_weights,
                progress_bar=load_progress_bar,
                ignore_keys=load_ignore_keys,
                exclude_algorithms=load_exclude_algorithms,
                algorithm_passes=self.engine.algorithm_passes,
            )
            self.state.run_name = run_name

            if load_force_new_epoch:
                self.state.force_new_epoch()

        # FSDP wrap if model is not yet wrapped and FSDP is enabled. This can happen if
        # load_fsdp_monolith_rank0_only=True but no checkpoint was loaded.
        if not self.state.fsdp_enabled and self.state.fsdp_config is not None and self.state.fsdp_auto_wrap and self.state.load_fsdp_monolith_rank0_only:
            prepare_fsdp_module(model, optimizers, self.state.fsdp_config, precision, device, auto_microbatching)

        self.engine.run_event(Event.AFTER_LOAD)

        # reseed here. This helps with a couple of issues:
        # 1. rng state may change at Event.INIT/Event.AFTER_LOAD. For example, if an algorithm
        # creates a new module and module parameters are initialized randomly, rng state will
        # change. This reseeding nullifies such effects.
        # 2. While resuming from a checkpoint, we want to spin dataloader and bring it back to the
        # same state as at the time of the checkpoint. Therefore, spinning needs to start from the
        # same rng state as in the original run.
        log.info(f'Setting seed to {self.state.seed}')
        reproducibility.seed_all(self.state.seed)

        # The model would need to be torch.compile()'d after being wrapped in a distributed strategy
        # to take advantage of any graph breaks.
        if not is_model_compiled:
            if self.state.model.enable_compiler:
                self.state.model.compile_model_wrap(compile_config or dict())
            elif compile_config is not None:
                compiled_model = torch.compile(  # pyright: ignore [reportGeneralTypeIssues]
                    self.state.model, **compile_config)
                self.state.model = compiled_model._orig_mod
                self.state.model.forward = compiled_model.dynamo_ctx(self.state.model.forward)
                is_model_compiled = True
                # update local_hparams to ensure the `is_model_compiled` is set correctly for
                # debugging purpose and for unit test.
                if self.auto_log_hparams:
                    self.local_hparams['is_model_compiled'] = is_model_compiled

    @property
    def saved_checkpoints(self) -> List[str]:
        """Returns list of saved checkpoints.
        """
        if self._checkpoint_saver is None:
            return []
        return self._checkpoint_saver.saved_checkpoints

    def _try_checkpoint_download(self, latest_checkpoint_path: str, save_latest_remote_file_name: str,
                                 loggers: Sequence[LoggerDestination], load_progress_bar: bool) -> None:
        """Attempts to download the checkpoint from the logger destinations."""
        log.debug(
            f'Trying to download {save_latest_remote_file_name} to {latest_checkpoint_path} on rank {dist.get_global_rank()}'
        )
        for logger in loggers:
            try:
                # Fetch from logger. If it succeeds, stop trying the rest of the loggers
                get_file(
                    path=save_latest_remote_file_name,
                    destination=latest_checkpoint_path,
                    object_store=logger,
                    overwrite=True,
                    progress_bar=load_progress_bar,
                )
                break
            except (NotImplementedError, FileNotFoundError):
                log.info(f'Checkpoint not found in: {logger}')
                # Ignore errors caused by no checkpoint saved with logger
                pass

    def _get_autoresume_checkpoint(
        self,
        save_folder: str,
        save_latest_filename: str,
        save_latest_remote_file_name: str,
        loggers: Sequence[LoggerDestination],
        load_progress_bar: bool,
    ) -> Optional[str]:
        """Determines the load path when using autoresume.

        First, check the ``save_folder`` for the latest checkpoint.
        If no latest checkpoint is found locally, then check each logger for the latest checkpoint, and download
        it to the ``save_folder``.

        Returns:
            Optional[str]: The path to the latest checkpoint, if found, otherwise None.
        """
        save_latest_filename = format_name_with_dist(save_latest_filename, self.state.run_name)
        save_folder = format_name_with_dist(save_folder, self.state.run_name)
        save_latest_remote_file_name = format_name_with_dist(save_latest_remote_file_name, self.state.run_name)
        latest_checkpoint_path = os.path.join(save_folder, save_latest_filename)

        log.info(
            f'Looking for autoresume checkpoint: {save_latest_remote_file_name} (remote), {latest_checkpoint_path} (local)'
        )

        if self.state.fsdp_sharded_state_dict_enabled:
            # If latest checkpoint is not saved locally, try to fetch from loggers
            if not os.path.exists(latest_checkpoint_path):
                log.debug(f'Attempting to download the checkpoint on to rank {dist.get_global_rank()}')
                os.makedirs(save_folder, exist_ok=True)
                self._try_checkpoint_download(latest_checkpoint_path, save_latest_remote_file_name, loggers,
                                              load_progress_bar)

            # List of whether the checkpoint exists on each rank
            latest_checkpoint_exists = dist.all_gather_object(os.path.exists(latest_checkpoint_path))

            if all(latest_checkpoint_exists):  # All paths exist, so return the path.
                return latest_checkpoint_path
            # Require all ranks to have their own local checkpoint if we wish to restore from it for
            # fsdp + sharding
            elif any(latest_checkpoint_exists):  # Some but not all exist, which is very bad.
                missing_ranks = [n for (n, exist) in enumerate(latest_checkpoint_exists) if not exist]
                mode ='FSDP sharding'
                raise RuntimeError(f'{mode} was enabled, but checkpoints missing on ranks: {missing_ranks}')
            else:  # None of the paths exists, so no autoresume necessary.
                return None
        else:

            def broadcast_object_list(object_list):
                tp_zero_rank = dist.get_tp_group_rank()
                if tp_zero_rank is None:
                    dist.broadcast_object_list(object_list, src=0)
                else:
                    # NOTE (Sbr): Double `dist` to get torch's `dist` module.
                    dist.dist.broadcast_object_list(object_list, src=tp_zero_rank, group=dist.get_fsdp_group())

            # broadcast the local checkpoint path to all ranks
            latest_checkpoint_path_list = [os.path.abspath(latest_checkpoint_path)]
            broadcast_object_list(latest_checkpoint_path_list)
            latest_checkpoint_path = latest_checkpoint_path_list[0]

            # broadcast the remote checkpoint path to all ranks
            save_latest_remote_file_name_list = [save_latest_remote_file_name]
            broadcast_object_list(save_latest_remote_file_name_list)
            save_latest_remote_file_name = save_latest_remote_file_name_list[0]

            cpid = os.getpid()
            crank = dist.get_local_rank()
            log.debug(f"cpid = {cpid}, crank = {crank}")

            # # try to download the checkpoint on local rank 0 of all nodes
            # if dist.get_local_rank() == 0 and not os.path.exists(latest_checkpoint_path):
            #     log.debug(f'Attempting to download the checkpoint {save_latest_remote_file_name} on to all nodes')
            #     os.makedirs(save_folder, exist_ok=True)
            #     self._try_checkpoint_download(latest_checkpoint_path, save_latest_remote_file_name, loggers,
            #                                   load_progress_bar)

            # signal_file_path = os.path.join(os.path.dirname(latest_checkpoint_path),
            #                                 f'.node_{dist.get_node_rank()}_local_rank0_completed_autoresume')
            # if dist.get_local_rank() == 0:
            #     os.makedirs(os.path.dirname(signal_file_path), exist_ok=True)
            #     with open(signal_file_path, 'wb') as f:
            #         f.write(b'local_rank0_completed_autoresume')

            # # Avoid the collective call until the local rank zero has finished trying to download the checkpoint
            # # so that we don't timeout for large downloads. This syncs all processes on the node
            # with dist.local_rank_zero_download_and_wait(signal_file_path):
            #     # Then, wait to ensure every node has finished downloading the checkpoint
            #     dist.barrier()

            # if dist.get_local_rank() == 0:
            #     os.remove(signal_file_path)
            with dist.run_local_rank_zero_first():
                if not os.path.exists(latest_checkpoint_path):
                    log.debug(f'Attempting to download the checkpoint {save_latest_remote_file_name} on to all nodes')
                    os.makedirs(save_folder, exist_ok=True)
                    self._try_checkpoint_download(latest_checkpoint_path, save_latest_remote_file_name, loggers,
                                                load_progress_bar)
            #dist.barrier()

            # At this point the rank 0 filepath should exist on all ranks if the download succeeded
            # list of whether the checkpoint exists on each rank
            latest_checkpoint_exists = dist.all_gather_object(os.path.exists(latest_checkpoint_path))

            log.debug(
                f'Checkpoint {latest_checkpoint_path} exists on rank {dist.get_global_rank()}? {os.path.exists(latest_checkpoint_path)}'
            )

            if not latest_checkpoint_exists[0]:
                # If the checkpoint doesn't exist on rank 0, don't crash, so the initial autoresume run can succeed
                return None
            elif not all(latest_checkpoint_exists):
                raise RuntimeError('Downloading the checkpoint to all nodes failed')

            return latest_checkpoint_path

    def fit(
        self,
        *,
        # Train Dataloader
        train_dataloader: Optional[Union[Iterable, DataSpec, Dict[str, Any]]] = None,
        train_dataloader_label: str = 'train',
        train_subset_num_batches: Optional[int] = None,
        spin_dataloaders: Optional[bool] = None,

        # Timing
        duration: Optional[Union[int, str, Time[int]]] = None,
        reset_time: bool = False,

        # Schedulers
        schedulers: Optional[Union[ComposerScheduler, PyTorchScheduler, Sequence[Union[ComposerScheduler,
                                                                                       PyTorchScheduler]]]] = None,
        scale_schedule_ratio: float = 1.0,
        step_schedulers_every_batch: Optional[bool] = None,

        # Evaluation
        eval_dataloader: Optional[Union[Iterable, DataSpec, Evaluator, Sequence[Evaluator]]] = None,
        eval_subset_num_batches: int = -1,
        eval_interval: Union[int, str, Time, Callable[[State, Event], bool]] = 1,

        # Numerics
        device_train_microbatch_size: Optional[Union[int, str]] = None,
        precision: Optional[Union[str, Precision]] = None,
    ):
        """Train the model.

        The Composer :class:`.Trainer` supports multiple calls to :meth:`.fit`. Any arguments specified during
        the call to :meth:`.fit` will override the values specified when constructing the :class:`.Trainer`.
        All arguments are optional, with the following exceptions:

        *   The ``train_dataloader`` must be specified here if not provided when constructing the :class:`.Trainer`.
        *   The ``duration`` must be specified here if not provided when constructing the :class:`.Trainer`,
            or if this is a subsequent call to :meth:`.fit`.

        For example, the following are equivalent:

        .. testcode::

            # The `train_dataloader` and `duration` can be specified
            # when constructing the Trainer
            trainer_1 = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                max_duration="1ep",
            )
            trainer_1.fit()

            # Or, these arguments can be specified on `fit()`
            trainer_2 = Trainer(model)
            trainer_2.fit(
                train_dataloader=train_dataloader,
                duration="1ep"
            )

        When invoking :meth:`.fit` for a subsequent time, either ``reset_time`` or ``duration`` must be specified.
        Otherwise, it is ambiguous for how long to train.

        *   If ``reset_time`` is True, then :meth:`.fit` will train for the same amount of time as the previous
            call (or for ``duration`` if that parameter is also specified). The :attr:`.State.timestamp` will be reset,
            causing :class:`.ComposerScheduler` and :class:`.Algorithm` instances to start from the beginning, as if it
            is a new training run. Model gradients, optimizer states, and native PyTorch schedulers will not be reset.

        *   If ``reset_time`` is False, then :meth:`.fit` will train for the amount of time specified by
            ``duration``. The :attr:`.State.max_duration` will be incremented by ``duration``.

        For example:

        .. testcode::

            # Construct the trainer
            trainer = Trainer(max_duration="1ep")

            # Train for 1 epoch
            trainer.fit()
            assert trainer.state.timestamp.epoch == "1ep"

            # Reset the time to 0, then train for 1 epoch
            trainer.fit(reset_time=True)
            assert trainer.state.timestamp.epoch == "1ep"

            # Train for another epoch (2 epochs total)
            trainer.fit(duration="1ep")
            assert trainer.state.timestamp.epoch == "2ep"

            # Train for another batch (2 epochs + 1 batch total)
            # It's OK to switch time units!
            trainer.fit(duration="1ba")
            assert trainer.state.timestamp.epoch == "2ep"
            assert trainer.state.timestamp.batch_in_epoch == "1ba"

            # Reset the time, then train for 3 epochs
            trainer.fit(reset_time=True, duration="3ep")
            assert trainer.state.timestamp.epoch == "3ep"

        Args:
            train_dataloader (Iterable | DataSpec | Dict[str, Any], optional): See :class:`.Trainer`.
            train_dataloader_label (str, optional): See :class:`.Trainer`.
            train_subset_num_batches (int, optional): See :class:`.Trainer`.
            spin_dataloaders (bool, optional): See :class:`.Trainer`.
            reset_time (bool): Whether to reset the :attr:`.State.timestamp` to zero values. Defaults to False.

                If ``True``, the timestamp will be zeroed out, causing :class:`.ComposerScheduler` and
                :class:`.Algorithm` instances to start from the beginning, as if it is a new training run. The model
                will be trained for ``duration``, if specified, or for :attr:`.State.max_duration`, which would have
                been provided when constructing the :class:`.Trainer` or by a previous call to :meth:`.fit`.

                .. note::

                    Model gradients, optimizer states, and native PyTorch schedulers will not be reset.

                If ``False`` (the default), training time will be incremented from where the previous call to
                :meth:`.fit` finished (or from zero, if a new training run).
                The :attr:`~.State.max_duration` will be incremented by the ``duration`` parameter.

            duration (Time[int] | str | int, optional): The duration to train. Can be an integer, which will be
                interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), or a :class:`.Time` object.

                If ``reset_time`` is False (the default), then :attr:`.State.max_duration` will be converted
                into the same units as this parameter (if necessary), and then the max duration incremented by the
                value of this parameter.

                If ``reset_time`` is True, then :attr:`.State.max_duration` will be set to this parameter.

            optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional): See :class:`.Trainer`.
            schedulers (PyTorchScheduler | ComposerScheduler | Sequence[PyTorchScheduler | ComposerScheduler], optional): See :class:`.Trainer`.
            scale_schedule_ratio (float, optional): See :class:`.Trainer`.
            step_schedulers_every_batch (bool, optional): See :class:`.Trainer`.
            eval_dataloader (Iterable | DataSpec | Evaluator | Sequence[Evaluator], optional): See :class:`.Trainer`.
            eval_subset_num_batches (int, optional): See :class:`.Trainer`.
            eval_interval (int | str | Time | (State, Event) -> bool, optional): See :class:`.Trainer`.
            device_train_microbatch_size (int | str, optional): See :class:`.Trainer`.
            precision (Precision | str, optional): See :class:`.Trainer`.
        """
        # Check Optimizer
        if len(self.state.optimizers) == 0:
            raise ValueError(f'No optimizer was specified when constructing the Trainer. As the '
                             'model had no parameters, SGD was not created by default. This trainer '
                             'object can only be used to evaluate or predict. Please specify a model '
                             'with parameters and an optimizer for training.')

        # Train Dataloader
        if train_dataloader is not None:
            self._train_data_spec = ensure_data_spec(train_dataloader)
            self.state.set_dataloader(self._train_data_spec.dataloader, train_dataloader_label)
            self.state.train_dataloader = self.state.dataloader
            self.state.device_train_microbatch_size = _get_initial_device_train_microbatch_size(
                self.state.device_train_microbatch_size, self.state.auto_microbatching, self.state.train_dataloader)
        if self._train_data_spec is None:
            _raise_missing_argument_exception('train_dataloader')
        if train_subset_num_batches is not None:
            self.state.dataloader_len = train_subset_num_batches
        if spin_dataloaders is not None:
            self.spin_dataloaders = spin_dataloaders

        # Reset Time
        if reset_time:
            self.state.timestamp = Timestamp()

        # Max Duration
        if duration is not None:
            duration = ensure_time(duration, TimeUnit.EPOCH)
            # Effectively increment the max duration (if not resetting the Time)
            # or set the max_duration (if resetting the time -- self.state.timestamp.get(duration.unit) will be 0)
            # It is important to set the duration, rather than incrementing it, as ``duration`` could be in
            # different units than ``max_duration``
            self.state.max_duration = duration + self.state.timestamp.get(duration.unit)

        if self.state.max_duration is None:
            _raise_missing_argument_exception('max_duration')

        if self.state.dataloader_len is None and self.state.max_duration.unit == TimeUnit.EPOCH:
            raise ValueError(
                ('max_duration cannot be specified in epochs when using an infinite dataloader. Please either '
                 'provide a dataloader with a length, specify max_duration in batches, samples, or tokens, or provide '
                 'train_subset_num_batches.'))

        if self.state.max_duration <= self.state.timestamp.get(self.state.max_duration.unit) and not reset_time:
            raise ValueError(
                (f'The max_duration ({self.state.max_duration}) is less than or equal to the elapsed training duration '
                 f'({self.state.timestamp.get(self.state.max_duration.unit)}). No training would occur. '
                 'Please provide the `duration` or specify `reset_time=True` in Trainer.fit().'))

        # Scale Schedule Ratio and Schedulers
        if scale_schedule_ratio != 1.0:
            # Not scaling the schedulers if the ratio is 1.0 in case if the scheduler cannot be scaled
            # (e.g. a custom LambdaLR). However, since 1.0 implies no scaling, it is still possible
            # to train with it.
            self.state.max_duration = _scale_max_duration_by_ssr(scale_schedule_ratio, self.state.max_duration)
        if schedulers is not None:
            self.state.schedulers = _compile_schedulers(schedulers, self.state, scale_schedule_ratio)

            if step_schedulers_every_batch is None:
                self._scheduler_step_frequency = _get_default_scheduler_frequency(schedulers)
            else:
                self._scheduler_step_frequency = TimeUnit.BATCH if step_schedulers_every_batch else TimeUnit.EPOCH
        else:
            if scale_schedule_ratio != 1.0:
                raise ValueError('Specifying `scale_schedule_ratio` without `schedulers` has no effect.')

            if step_schedulers_every_batch is not None:
                raise ValueError('Specifying `step_schedulers_every_batch` without `schedulers` has no effect.')

            if step_schedulers_every_batch is not None:
                raise ValueError('Specifying `step_schedulers_every_batch` without `schedulers` has no effect.')

        # Evaluators
        if eval_dataloader is not None:
            # Need to use the `original_model` rather than `state.model`, as `state.model`
            # could be DDP wrapped.
            eval_metrics = self._original_model.get_metrics(is_train=False)
            metric_names = [str(k) for k in eval_metrics.keys()]
            eval_dataloader = ensure_tuple(eval_dataloader)

            evaluator_types = [isinstance(evaluator, Evaluator) for evaluator in eval_dataloader]
            if any(evaluator_types) and not all(evaluator_types):
                raise ValueError('Mixing Evaluator with other classes is not allowed, please wrap'
                                 'all other classes with the Evaluator class. These are the classes'
                                 'that were detected:' + str([type(evaluator) for evaluator in eval_dataloader]))

            evaluators = [
                ensure_evaluator(evaluator, default_metric_names=metric_names) for evaluator in eval_dataloader
            ]

            # match metric names to model metrics
            self.state.eval_metrics = {
                evaluator.label: _filter_metrics(eval_metrics, evaluator.metric_names) for evaluator in evaluators
            }

            _set_evaluator_interval_and_subset_num_batches(
                evaluators=evaluators,
                eval_interval=eval_interval,
                subset_num_batches=eval_subset_num_batches,
            )

            for evaluator in evaluators:
                validate_eval_automicrobatching(evaluator.auto_microbatching, self.state.device)

            if len(evaluators) == 0:
                if eval_subset_num_batches != -1:
                    raise ValueError('Specifying `eval_subset_num_batches` without an `eval_dataloader` has no effect.')
                if eval_interval != 1:
                    raise ValueError('Specifying `eval_interval` without an `eval_dataloader` has no effect.')

            self.state.evaluators = evaluators

        # Microbatching
        if device_train_microbatch_size is not None:
            self.state.auto_microbatching = _is_auto_microbatching(device_train_microbatch_size,
                                                                   device=self.state.device)
            if self.state.auto_microbatching and self.state.profiler:
                raise ValueError("`device_train_microbatch_size='auto'` is not compatible with the profiler. It is "
                                 "recommended to run a mini-run with `device_train_microbatch_size='auto'` to identify "
                                 'the optimal device_train_microbatch_size value and then manually specify that in a '
                                 'second run with profiler.')
            self.state.device_train_microbatch_size = _get_initial_device_train_microbatch_size(
                device_train_microbatch_size, self.state.auto_microbatching, self.state.train_dataloader)

        # Precision
        if precision is not None:
            if Precision(precision) != self.state.precision:
                precision = Precision(precision)
                _validate_precision(precision, self.state.device)
                self.state.precision = precision

            # update scaler since precision was provided
            self.state.scaler = ClosureGradScaler() if self._use_closures() else GradScaler()

        self.first_batch_complete = False
        self._train_loop()

    def close(self):
        """Shutdown the trainer.

        .. seealso:: :meth:`.Engine.close` for additional information.
        """
        self.engine.close()
        dist.barrier()

    def _ensure_metrics_device_and_dtype(self, metrics: Dict[str, Metric]):
        for name, metric in metrics.items():
            # Safety check to ensure the metric and data are on the same device. Normally not
            # needed because the metric is automatically on the same device as the model.
            # See https://torchmetrics.readthedocs.io/en/latest/pages/overview.html for details.
            metrics[name] = self.state.device.module_to_device(metric)

        return metrics

    def _compute_and_log_metrics(self, dataloader_label: str, metrics: Dict[str, Metric]):
        """Computes metrics, logs the results, and updates the state with the deep-copied metrics.

        Args:
            dataloader_label (str): The dataloader label.
            metrics (Dict[str, Metric]): The metrics to compute.
        """
        metrics = deepcopy(metrics)

        # log computed metrics
        computed_metrics = {}
        for metric_name, metric in metrics.items():
            computed_metrics[metric_name] = metric.compute()

        self.logger.log_metrics(
            {f'metrics/{dataloader_label}/{name}': val for (name, val) in computed_metrics.items()},)

        # store metric instances
        for metric_name, metric in metrics.items():
            assert isinstance(metric, Metric)
            if dataloader_label == 'train':
                self.state.train_metrics[metric_name] = metric
                self.state.train_metric_values[metric_name] = computed_metrics[metric_name]
            else:
                if dataloader_label not in self.state.eval_metrics:
                    self.state.eval_metrics[dataloader_label] = {}
                self.state.eval_metrics[dataloader_label][metric_name] = metric
                self.state.eval_metric_values[metric_name] = computed_metrics[metric_name]

    def _spin_dataloaders_to_cur_epoch(self):
        """Spin the dataloaders to restore sampler state for current epoch.

        Only one batch must be loaded to seed the sampler's generator. since only the first batch is being loaded, the
        dataloader may not be completely iterated through.
        """
        log.debug('Spinning the dataloaders')

        # spin the evaluator dataloaders once to initialize its sampler deterministically
        # so it does not affect any other RNG reads
        eval_state = self.state.dataset_resumption.get('eval', {})
        for evaluator in self.state.evaluators:
            dataloader = evaluator.dataloader.dataloader
            if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(0)
            if evaluator.label not in eval_state:
                for _ in dataloader:
                    break

        # spin the train dataloader's sampler to get to the state of the desired epoch
        dataloader = self.state.dataloader
        assert dataloader is not None, 'train dataloader is set on state after FIT_START'
        if 'train' not in self.state.dataset_resumption:
            for epoch in range(int(self.state.timestamp.epoch)):
                if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                    dataloader.sampler.set_epoch(epoch)
                for _ in dataloader:
                    break

    def _accumulate_time_across_ranks(
        self,
        num_samples: int,
        num_tokens: int,
        batch_time: datetime.timedelta,
    ) -> Tuple[int, int, datetime.timedelta]:
        """Accumulate the number of samples and tokens across ranks.

        Returns a (num_samples, num_tokens, batch_time) tuple.
        """
        # Samples and tokens should be summed
        # Batch time should be the value from rank 0
        sample_token_tensor = self.state.device.tensor_to_device(
            torch.tensor([num_samples, num_tokens], dtype=torch.int))
        dist.all_reduce(sample_token_tensor, reduce_operation='SUM')
        batch_time_tensor = self.state.device.tensor_to_device(
            torch.tensor([batch_time.total_seconds()], dtype=torch.float32))
        dist.broadcast(batch_time_tensor, src=0)
        batch_time = datetime.timedelta(seconds=batch_time_tensor[0].cpu().item())
        tp_sp_group_size = dist.get_tp_sp_group_size() or 1

        return int(sample_token_tensor[0].cpu().item()) // tp_sp_group_size, int(sample_token_tensor[1].cpu().item()) // tp_sp_group_size, batch_time

    def _train_loop(self) -> None:
        """Run training for the specified number of epochs and log results."""
        # print training start
        log.info('Using precision %s', self.state.precision)
        self.logger.log_hyperparameters(
            {'enabled_algorithms/' + algo.__class__.__name__: True for algo in self.state.algorithms})
        assert self.state.dataloader is not None, 'dataloader is set in __init__() or fit()'
        assert self._train_data_spec is not None, 'The train data spec is set in __init__() or fit()'
        assert self.state.scaler is not None, 'scaler should have been set in __init__()'

        self.engine.run_event(Event.FIT_START)

        use_grad_scaling = self._use_grad_scaling(self.state.precision, self.state.scaler)

        if self.spin_dataloaders:
            self._spin_dataloaders_to_cur_epoch()

        if self.state.timestamp.batch_in_epoch == 0 and self._rng_state is not None:
            # Only restore the rng state here if the step in the current epoch is zero.
            reproducibility.load_rng_state(self._rng_state)
            self._rng_state = None

        self.state.model.train()
        finished_epoch_early = False
        last_wct = datetime.datetime.now()

        if self.state.max_duration is None:
            # This is essentially just a type check, as max_duration should always be
            # asserted to be not None when Trainer.fit() is called
            raise RuntimeError('max_duration must be specified when initializing the Trainer')

        while self.state.timestamp < self.state.max_duration:
            if int(self.state.timestamp.batch_in_epoch) == 0:
                self.engine.run_event(Event.EPOCH_START)
                self.logger.log_metrics({'time/epoch': self.state.timestamp.epoch.value})

            dataloader = self.state.dataloader
            if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(int(self.state.timestamp.epoch))

            assert isinstance(dataloader.dataset, (StreamingDataset, Dataset))
            if not isinstance(dataloader.dataset, StreamingDataset):
                log.warning("The `StreamingDataset` format was expected for the dataloader.dataset, " +
                            "but `Dataset` is used instead. If you're using dummy data, just ignore this message.")
            assert isinstance(dataloader, DataLoader)

            # calculating the index on which we should spin
            batch_idx = 0
            if self.spin_dataloaders and 'train' not in self.state.dataset_resumption and int(self.state.timestamp.batch_in_epoch) - 1 > 0:
                batch_idx = int(self.state.timestamp.batch_in_epoch) - 1
                device_batch_size = dataloader.batch_size
                samples_per_device = int(self.state.timestamp.sample_in_epoch) // dist.get_world_size() - device_batch_size

                if dist.get_tp_sp_group_size() is None:
                    # Tensor parallel disabled
                    dataloader.dataset.yield_index = samples_per_device
                else:
                    # Tensor parallel enabled
                    dataloader.dataset.yield_index = samples_per_device // dist.get_tp_sp_group_size()

            log.debug('Starting training loop')

            for self.state.batch in self._iter_dataloader(TrainerMode.TRAIN):
                # Spin dataloader forward unless dataloader handles internally with dataset_resumption
                if self.spin_dataloaders and 'train' not in self.state.dataset_resumption and batch_idx < int(
                        self.state.timestamp.batch_in_epoch):
                    # Restore the RNG state immediately before the next batch is yielded from the dataloader
                    if batch_idx + 1 == int(self.state.timestamp.batch_in_epoch) and self._rng_state is not None:
                        reproducibility.load_rng_state(self._rng_state)
                        self._rng_state = None
                    batch_idx += 1

                    self.engine.run_marker_only_event(Event.AFTER_DATALOADER)
                    continue

                self.state.batch = self.state.device.batch_to_device(self.state.batch)
                self.state.batch = self._train_data_spec.device_transforms(self.state.batch)

                # move after_dataloader event here to fix sequence warmup seq_lenth truncete before calculating rank_num_tokens
                self.engine.run_event(Event.AFTER_DATALOADER)

                rank_num_samples = self._train_data_spec.get_num_samples_in_batch(self.state.batch)
                rank_num_tokens = self._train_data_spec.get_num_tokens_in_batch(self.state.batch)


                self.engine.run_event(Event.BATCH_START)

                # Log time values
                self.logger.log_metrics({
                    'time/batch': self.state.timestamp.batch.value,
                    'time/sample': self.state.timestamp.sample.value,
                    'time/batch_in_epoch': self.state.timestamp.batch_in_epoch.value,
                    'time/sample_in_epoch': self.state.timestamp.sample_in_epoch.value,
                })
                if rank_num_tokens > 0:
                    self.logger.log_metrics({'time/token': self.state.timestamp.token.value})
                    self.logger.log_metrics({'time/token_in_epoch': self.state.timestamp.token_in_epoch.value})

                total_loss_dict = self._train_batch(use_grad_scaling)

                if use_grad_scaling:
                    self.state.scaler.update()

                # total_loss_dict can be None if gradient scaling failed
                if total_loss_dict is not None:
                    map_collection(total_loss_dict, dist.all_reduce)
                    total_loss_dict = {
                        k: loss.cpu().item() / dist.get_world_size() for k, loss in total_loss_dict.items()
                    }
                    self.state.total_loss_dict = total_loss_dict
                    self.logger.log_metrics(total_loss_dict)

                # The scheduler step.step() and compute_and_log_metrics() are going to be included in the
                # next batch's wall clock time. The time accumulation must be done here so schedulers
                # have the latest timing information

                now = datetime.datetime.now()

                batch_time = now - last_wct
                if self.state.rank_local_training_observer is not None:
                    self.state.rank_local_training_observer(
                        rank_num_samples, rank_num_tokens, batch_time.total_seconds(),
                    )

                total_num_samples, total_num_tokens, batch_time = self._accumulate_time_across_ranks(
                    rank_num_samples,
                    rank_num_tokens,
                    batch_time,
                )

                # `now` is actually in the past, but want to include the time it takes to perform this reduction
                last_wct = now

                if self.state.train_metrics is not None:
                    self._compute_and_log_metrics(
                        dataloader_label='train',
                        metrics=self.state.train_metrics,
                    )

                self.state.previous_timestamp = self.state.timestamp
                self.state.timestamp = self.state.timestamp.to_next_batch(
                    samples=total_num_samples,
                    tokens=total_num_tokens,
                    duration=batch_time,
                )

                if self._scheduler_step_frequency == TimeUnit.BATCH:
                    for scheduler in self.state.schedulers:
                        scheduler.step()
                self.engine.run_event(Event.BATCH_END)

                # NOTE: checkpointing has been moved to *before* evaluation
                self.engine.run_event(Event.BATCH_CHECKPOINT)
                self._run_checkpoint_save_events(blocking=False)

                # Pause the timing during evaluation
                # Evaluation time is tracked separately in state.eval_timestamp
                duration = datetime.datetime.now() - last_wct
                self._run_evaluators(Event.BATCH_END)
                last_wct = datetime.datetime.now() - duration

                self.logger.log_timers()

                if self.state.timestamp >= self.state.max_duration:
                    # If max_duration is specified in batches, samples, or tokens, and
                    # and the max_duration is reached mid-epoch, then break out of the dataloader
                    # to finish the epoch early and finish training.
                    finished_epoch_early = True
                    break
                batch_idx += 1

            if not finished_epoch_early or self.state.dataloader_len == self.state.timestamp.batch_in_epoch:
                # Trigger the epoch end events if the dataloader was exhausted.
                # This happens if the "break" did not trigger above, or if it
                # did (e.g. duration specified in samples/batches/tokens), but it is still
                # the end of the dataloader (i.e. next(dataloader) would raise StopIteration)
                if self.state.train_metrics is not None:
                    self.state.train_metrics = self._ensure_metrics_device_and_dtype(self.state.train_metrics)
                    self._compute_and_log_metrics(
                        dataloader_label='train',
                        metrics=self.state.train_metrics,
                    )

                self.state.previous_timestamp = self.state.timestamp
                self.state.timestamp = self.state.timestamp.to_next_epoch()

                if self._scheduler_step_frequency == TimeUnit.EPOCH:
                    for scheduler in self.state.schedulers:
                        scheduler.step()

                self.engine.run_event(Event.EPOCH_END)

                # NOTE: checkpointing has been moved to *before* evaluation
                self.engine.run_event(Event.EPOCH_CHECKPOINT)
                self._run_checkpoint_save_events(blocking=False)

                # Pause the timing during evaluation
                # Evaluation time is tracked separately in state.eval_timestamp
                duration = datetime.datetime.now() - last_wct
                self._run_evaluators(Event.EPOCH_END)
                last_wct = datetime.datetime.now() - duration

        # Log final time values
        self.logger.log_metrics({
            'time/epoch': self.state.timestamp.epoch.value,
            'time/batch': self.state.timestamp.batch.value,
            'time/sample': self.state.timestamp.sample.value,
            'time/batch_in_epoch': self.state.timestamp.batch_in_epoch.value,
            'time/sample_in_epoch': self.state.timestamp.sample_in_epoch.value,
        })
        if self.state.previous_timestamp is not None and self.state.timestamp.token.value - self.state.previous_timestamp.token.value > 0:
            self.logger.log_metrics({'time/token': self.state.timestamp.token.value})
            self.logger.log_metrics({'time/token_in_epoch': self.state.timestamp.token_in_epoch.value})

        self._run_checkpoint_save_events(blocking=True)
        self.engine.run_event(Event.FIT_END)
        self._run_evaluators(Event.FIT_END)
        self.logger.log_timers(force=True)

    def _run_checkpoint_save_events(self, *, blocking: bool):
        """Run callbacks for checkpoint saves completed on the main training thread."""
        checkpoint_savers = [
            callback for callback in self.state.callbacks if isinstance(callback, CheckpointSaver)
        ]
        if not checkpoint_savers:
            return

        if blocking:
            for checkpoint_saver in checkpoint_savers:
                checkpoint_saver.wait_for_async_checkpoint()

        for checkpoint_saver in checkpoint_savers:
            for checkpoint_save in checkpoint_saver.drain_completed_checkpoint_saves():
                checkpoint_saver._set_current_checkpoint_save(checkpoint_save)
                try:
                    self.engine.run_event(Event.AFTER_CHECKPOINT_SAVE)
                finally:
                    checkpoint_saver._set_current_checkpoint_save(None)

    def _eval_train_metrics(self, device_batch):
        assert self._train_data_spec is not None, 'The train data spec should be set on __init__ or fit()'
        assert self.state.train_metrics is not None, 'The train metrics should be set on __init__ or fit()'

        with torch.no_grad(),\
                model_eval_mode(self.state.model),\
                _get_precision_context(self.state.precision, self.state.precision_config):
            eval_outputs = self._original_model.eval_forward(device_batch, self.state.outputs)
            for metric in self.state.train_metrics.values():
                self._original_model.update_metric(
                    device_batch,
                    eval_outputs,
                    metric,
                )

    def _run_evaluators(self, event: Event):
        """Runs evaluators periodically during training."""
        evaluators_executing = []
        for evaluator in self.state.evaluators:
            assert evaluator.eval_interval is not None, 'eval_interval should have been set on __init__() or fit()'
            assert evaluator.subset_num_batches is not None, 'subset_num_batches should have been set on __init__() or fit()'
            evaluators_executing.append(evaluator.eval_interval(self.state, event))
        if not any(evaluators_executing):
            return

        self.engine.run_event(Event.EVAL_BEFORE_ALL)
        for index, evaluator in enumerate(self.state.evaluators):
            if evaluators_executing[index]:
                self._eval_loop(
                    evaluator=evaluator,
                    subset_num_batches=evaluator.subset_num_batches,
                    metrics=self.state.eval_metrics[evaluator.label],
                )

        self.engine.run_event(Event.EVAL_AFTER_ALL)

    def _train_batch(self, use_grad_scaling: bool) -> Dict[str, torch.Tensor]:
        """Compute loss by training on a full batch of data.

        Adaptively change microbatch size if enabled to maximize GPU usage.

        Args:
            use_grad_scaling (bool): Enables gradient scaling.

        Returns:
            Dict[str, torch.Tensor]: a dictionary containing the total loss and individual losses if available.
        """
        assert self._train_data_spec is not None, 'The train data spec should be set on __init__ or fit()'

        # Cache the device batch, because `self.state.batch` gets overridden in microbatching loop.
        # Any in-place changes to a microbatch will be reflected in the device batch.
        device_batch = self.state.batch

        # Retry until we successfully complete training and return loss
        while True:
            # Reset train_metrics on every batch
            # Placing reset here ensures that if auto grad accum catches an OOM, incomplete metric state is cleared
            if self.state.train_metrics is not None:
                for metric in self.state.train_metrics.values():
                    metric.reset()

            total_loss_dict = {'loss/train/total': self.state.device.tensor_to_device(torch.zeros(size=(1,)))}
            found_cuda_oom = 0  # int since bool BOR not supported on all torch.distributed backends
            try:
                assert self.state.scaler is not None
                assert self.state.device_train_microbatch_size is not None
                microbatches = self._train_data_spec.split_batch(device_batch, self.state.device_train_microbatch_size)
                if self._use_closures():
                    for optimizer in self.state.optimizers:
                        if use_grad_scaling:
                            self.state.scaler.step(optimizer,
                                                    closure=lambda loss_dict=total_loss_dict, **kwargs: self.
                                                    _train_microbatches(microbatches, loss_dict, **kwargs))
                        else:
                            optimizer.step(closure=lambda loss_dict=total_loss_dict, **kwargs: self._train_microbatches(
                                    microbatches, loss_dict, **kwargs).item())
                else:
                    self._train_microbatches(microbatches, total_loss_dict)
                    for optimizer in self.state.optimizers:
                        if use_grad_scaling:
                            self.state.scaler.step(optimizer)
                        else:
                            if isinstance(self.state.device, DeviceTPU):
                                xm.optimizer_step(optimizer, barrier=True)
                            else:
                                optimizer.step()
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    log.warning(f"Rank {dist.get_global_rank()} OOM'd with {dist.get_sp_group_size()=} {dist.get_tp_sp_group_size()=} {self.state.batch.get('local_max_seq_len')=}")
                    raise
                elif self.state.auto_microbatching:
                    if _is_cuda_oom(e):
                        log.debug((f"Rank {dist.get_global_rank()} OOM'd."))
                        found_cuda_oom = 1
                    elif ('cuda' in str(e).lower() or 'c10' in str(e).lower()):
                        raise RuntimeError(
                            textwrap.dedent(
                                'Encountered non-addressable cuda error while using auto microbatching. '
                                'If this repeatedly occurs, set `device_train_microbatch_size` manually.')) from e
                else:
                    raise

            if self.state.auto_microbatching:
                all_ranks_finished = False
                while not all_ranks_finished:
                    # Propagate across all ranks if any rank hit CUDA OOM
                    found_cuda_oom_tensor = self.state.device.tensor_to_device(
                        torch.tensor([found_cuda_oom], dtype=torch.uint8))
                    dist.all_reduce(found_cuda_oom_tensor, reduce_operation='MAX')
                    found_cuda_oom = found_cuda_oom_tensor.item()
                    # Check if any rank is still not done with the batch. This may happen if only a
                    # subset of ranks OOM, leaving some batches still in the forward pass
                    all_ranks_finished_tensor = self.state.device.tensor_to_device(torch.tensor([1], dtype=torch.uint8))
                    dist.all_reduce(all_ranks_finished_tensor, reduce_operation='MIN')
                    all_ranks_finished = all_ranks_finished_tensor.item() == 1
                if found_cuda_oom == 1:
                    _adjust_device_train_microbatch_size(self.state)
                    # Skip return and rerun after handling oom
                    continue
            # Log microbatch and return loss if we've completed without OOMing.
            assert self.state.device_train_microbatch_size is not None
            self.logger.log_metrics({'trainer/device_train_microbatch_size': self.state.device_train_microbatch_size})
            self.logger.log_metrics({'trainer/global_train_batch_size': self.global_train_batch_size})
            accum_batch_size = self.state.device_train_microbatch_size * dist.get_world_size() // (dist.get_tp_sp_group_size() or 1)
            self.logger.log_metrics({'trainer/gradient_accumulation_steps': self.global_train_batch_size // accum_batch_size})
            self.logger.log_metrics({'trainer/num_gpus': dist.get_world_size()})
            self.first_batch_complete = True
            return total_loss_dict

    def _train_microbatches(self,
                            microbatches: Sequence[Batch],
                            total_loss_dict: Dict[str, torch.Tensor],
                            ) -> torch.Tensor:
        """Iterate over microbatches and compute the loss that will be used to step the optimizer.

        Args:
            microbatches (Sequence[Batch]): The microbatches which make up the batch.
            total_loss_dict (Dict[str, torch.tensor]): Dictionary containing individual losses and their sum aggregated across all
                microbatches.
        """

        context = contextlib.nullcontext

        assert self._train_data_spec is not None

        with context():
            self.engine.run_event(Event.BEFORE_TRAIN_BATCH)

            assert self.state.optimizers is not None
            assert self.state.scaler is not None

            use_grad_scaling = self._use_grad_scaling(self.state.precision, self.state.scaler)

            for optimizer in self.state.optimizers:
                try:
                    optimizer.zero_grad(set_to_none=not self.state.gigafsdp_enabled)
                except TypeError:
                    optimizer.zero_grad()

            # Tracker for gradient accumulation
            current_batch_size = sum([self._train_data_spec.get_num_samples_in_batch(batch) for batch in microbatches])
            # Cache batch, which will be overwritten by microbatches. Restore after microbatches complete
            current_batch = self.state.batch

            for microbatch_idx, self.state.batch in enumerate(microbatches):
                is_final_microbatch = microbatch_idx + 1 == len(microbatches)
                microbatch_loss_dict = self._train_microbatch(use_grad_scaling, current_batch_size, is_final_microbatch)
                # Aggregate each loss in microbatch_loss_dict into total_loss_dict
                for k, microbatch_loss in microbatch_loss_dict.items():
                    loss_key = f'loss/train/{k}'
                    if loss_key not in total_loss_dict:
                        total_loss_dict[loss_key] = self.state.device.tensor_to_device(torch.zeros(size=(1,)))
                    total_loss_dict[loss_key] += microbatch_loss

            # Restore batch
            self.state.batch = current_batch

            # Unscale gradients before `Event.AFTER_TRAIN_BATCH`
            if use_grad_scaling:
                for optimizer in ensure_tuple(self.state.optimizers):
                    self.state.scaler.unscale_(optimizer)

            self.engine.run_event(Event.AFTER_TRAIN_BATCH)

            return total_loss_dict['loss/train/total']

    def _train_microbatch(self, use_grad_scaling: bool, current_batch_size: int,
                         is_final_microbatch: bool) -> Dict[str, torch.Tensor]:
        """Train and compute the loss of ``state.batch``, which is assumed to be a single microbatch.

        Args:
            use_grad_scaling (bool): Whether to use gradient scaling.
            current_batch_size (int): The current batch size.
            minibatch_num_samples (int): Number of samples in the minibatch.
            is_final_microbatch (bool): If current microbatch is the last one.
        """
        assert self.state.scaler is not None
        assert self._train_data_spec is not None

        # Cache the device batch, because `self.state.batch` gets overridden in microbatching loop
        device_batch = deepcopy(self.state.batch)

        microbatch_num_samples = self._train_data_spec.get_num_samples_in_batch(self.state.batch)

        sync_context = contextlib.nullcontext()

        with sync_context:
            # Forward pass
            self.engine.run_event(Event.BEFORE_FORWARD)

            with _get_precision_context(self.state.precision, self.state.precision_config):
                self.state.outputs = self.state.model(self.state.batch)

            self.engine.run_event(Event.AFTER_FORWARD)

            # Check if other ranks OOMed after forward pass when using auto microbatching. This may
            # happen when close to memory limit or with uneven memory usage across ranks
            if self.state.auto_microbatching:
                # Check if any other rank hit an OOM
                found_cuda_oom_tensor = self.state.device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
                dist.all_reduce(found_cuda_oom_tensor, reduce_operation='MAX')
                found_cuda_oom = found_cuda_oom_tensor.item()
                # Signal current rank is still in batch
                all_ranks_finished_tensor = self.state.device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
                dist.all_reduce(all_ranks_finished_tensor, reduce_operation='MIN')

                if found_cuda_oom == 1:
                    raise RuntimeError('CUDA out of memory encountered on a different rank')

            # Loss
            self.engine.run_event(Event.BEFORE_LOSS)

            with _get_precision_context(self.state.precision, self.state.precision_config):
                self.state.loss = self._original_model.loss(self.state.outputs, self.state.batch)

            assert self.state.loss is not None
            self.engine.run_event(Event.AFTER_LOSS)

            # Backward Pass
            self.engine.run_event(Event.BEFORE_BACKWARD)

            microbatch_loss_dict = {}
            # If total loss key is present, copy loss
            if isinstance(self.state.loss, dict) and ('total' in self.state.loss):
                microbatch_loss = self.state.loss['total']  # type: ignore
                microbatch_loss_dict = self.state.loss.copy()
            # If total loss key is not present, sum individual losses
            else:
                microbatch_loss = self.state.device.tensor_to_device(torch.zeros(size=(1,)))
                for loss in ensure_tuple(self.state.loss):
                    microbatch_loss.add_(loss.mean())

                # Copy the loss if it is a dictionary
                if isinstance(self.state.loss, dict):
                    microbatch_loss_dict = self.state.loss.copy()
                # If not, create a dictionary with generic loss names
                elif len(ensure_tuple(self.state.loss)) > 1:
                    microbatch_loss_dict = {f'loss{i}': loss for i, loss in enumerate(ensure_tuple(self.state.loss))}

                # Include total loss
                microbatch_loss_dict['total'] = microbatch_loss

            # For each loss to log: detach, clone, mean, then multiply by (microbatch size) / (batch size)
            for k, loss in microbatch_loss_dict.items():
                microbatch_loss_dict[k] = loss.detach().clone().mean() * (microbatch_num_samples / current_batch_size)

            if use_grad_scaling:
                microbatch_loss = cast(torch.Tensor, self.state.scaler.scale(microbatch_loss))

            if not self.state.gigafsdp_enabled:
                # Scale loss based on the number of samples in the microbatch to maintain gradient numerics
                microbatch_loss.mul_(microbatch_num_samples / current_batch_size)

            microbatch_loss.backward(create_graph=self._backwards_create_graph, retain_graph=False)

            self.engine.run_event(Event.AFTER_BACKWARD)

            # Use microbatch outputs to update training metrics
            if self.state.train_metrics is not None and len(self.state.train_metrics) != 0:
                self.state.train_metrics = self._ensure_metrics_device_and_dtype(self.state.train_metrics)
                self._eval_train_metrics(device_batch)

        return microbatch_loss_dict

    def predict(
        self,
        dataloader: Union[DataLoader, DataSpec],
        subset_num_batches: int = -1,
        *,
        return_outputs: bool = True,
    ):
        """Output model prediction on the provided data.

        There are two ways to access the prediction outputs.

        1.  With ``return_outputs`` set to True, the batch predictions will be collected into a list and returned.
        2.  Via a custom callback, which can be used with ``return_outputs`` set to False.

            This technique can be useful if collecting all the outputs from the dataloader would exceed available memory,
            and you want to write outputs directly to files. For example:

            .. testsetup::

                predict_dl = train_dataloader

            .. testcode::

                import os
                import torch

                from torch.utils.data import DataLoader

                from composer import Trainer, Callback
                from composer.loggers import Logger

                class PredictionSaver(Callback):
                    def __init__(self, folder: str):
                        self.folder = folder
                        os.makedirs(self.folder, exist_ok=True)

                    def predict_batch_end(self, state: State, logger: Logger) -> None:
                        name = f'batch_{int(state.predict_timestamp.batch)}.pt'
                        filepath = os.path.join(self.folder, name)
                        torch.save(state.outputs, filepath)

                        # Also upload the files
                        logger.upload_file(remote_file_name=name, file_path=filepath)

                trainer = Trainer(
                    ...,
                    callbacks=PredictionSaver('./predict_outputs'),
                )

                trainer.predict(predict_dl, return_outputs=False)

                print(sorted(os.listdir('./predict_outputs')))

            .. testoutput::

                ['batch_1.pt', ...]

        Args:
            dataloader (DataLoader | DataSpec): The :class:`.DataLoader` or
                :class:`.DataSpec` for the prediction data.
            subset_num_batches (int, optional): If specified, only perform model prediction
                on this many batches. This parameter has no effect if it is greater than ``len(dataloader)``.
                If ``-1``, then the entire loader will be iterated over. (default: ``-1``)
            return_outputs (bool, optional): If True (the default), then prediction outputs will be (recursively)
                moved to cpu and accumulated into a list. Otherwise, prediction outputs are discarded after each
                batch.

        Returns:
            List: A list of batch outputs, if ``return_outputs`` is True. Otherwise, an empty list.

        """
        if isinstance(dataloader, DataSpec):
            data_spec = dataloader
        else:
            data_spec = DataSpec(dataloader)

        # Bind the dataloader to the state, but be able to restore the previous dataloader afterwards
        original_dataloader = self.state.dataloader
        original_dataloader_label = self.state.dataloader_label
        original_dataloader_len = self.state.dataloader_len
        self.state.set_dataloader(data_spec.dataloader, 'predict', subset_num_batches)
        assert self.state.dataloader is not None, 'Already set the dataloader'

        # Reset the predict timestamp
        self.state.predict_timestamp = Timestamp()

        last_wct = datetime.datetime.now()

        outputs = []
        cpu_device = DeviceCPU()

        # NOTE (Sbr): Set batch gather group size to TP group size on predict
        # as we disable SP processing on eval and predict for now.
        # ---
        # TODO (Sbr): Fix this back to TP-SP group size when the issue is fixed.
        batch_gather_group_size = dist.get_tp_group_size()

        with torch.no_grad(), model_eval_mode(self.state.model):

            self.engine.run_event(Event.PREDICT_START)

            for batch_idx, self.state.batch in enumerate(self._iter_dataloader(TrainerMode.PREDICT)):
                # Move the batch onto the device
                self.state.batch = self.state.device.batch_to_device(self.state.batch)

                # Perform any device transforms
                if data_spec.device_transforms is not None:
                    self.state.batch = data_spec.device_transforms(self.state.batch)

                # Count the batch size and num tokens before any events run
                rank_num_samples = data_spec.get_num_samples_in_batch(self.state.batch)
                rank_num_tokens = data_spec.get_num_tokens_in_batch(self.state.batch)

                if batch_gather_group_size is None or batch_idx % batch_gather_group_size == 0:
                    self.engine.run_event(Event.PREDICT_BATCH_START)

                self.engine.run_event(Event.PREDICT_BEFORE_FORWARD)
                with _get_precision_context(self.state.precision, self.state.precision_config):
                    self.state.outputs = self.state.model(self.state.batch)
                self.engine.run_event(Event.PREDICT_AFTER_FORWARD)

                if return_outputs:
                    outputs.append(cpu_device.batch_to_device(self.state.outputs))

                if batch_gather_group_size is None or (batch_idx + 1) % batch_gather_group_size == 0:
                    now = datetime.datetime.now()
                    batch_time = now - last_wct

                    total_num_samples, total_num_tokens, batch_time = self._accumulate_time_across_ranks(
                        num_samples=rank_num_samples,
                        num_tokens=rank_num_tokens,
                        batch_time=batch_time,
                    )

                    last_wct = now

                    self.state.predict_timestamp = self.state.predict_timestamp.to_next_batch(samples=total_num_samples,
                                                                                            tokens=total_num_tokens,
                                                                                            duration=batch_time)

                    self.engine.run_event(Event.PREDICT_BATCH_END)

            self.engine.run_event(Event.PREDICT_END)

        # Restore the dataloader
        self.state.set_dataloader(original_dataloader, original_dataloader_label)
        if original_dataloader_len is not None:
            self.state.dataloader_len = original_dataloader_len

        return outputs

    def eval(
        self,
        eval_dataloader: Optional[Union[Iterable, DataSpec, Evaluator, Sequence[Evaluator]]] = None,
        subset_num_batches: int = -1,
    ):
        """Run evaluation loop.

        Results are stored in ``trainer.state.eval_metrics``. The ``eval_dataloader`` can be provided to
        either the eval() method or during training init().

        Examples:
        .. testcode::

            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                max_duration="2ep",
                device="cpu",
            )

            trainer.fit()

            # run eval
            trainer.eval(
                eval_dataloader=eval_dataloader,
            )

        Or, if the ``eval_dataloader`` is provided during init:

        .. testcode::

            trainer = Trainer(
                model=model,
                eval_dataloader=eval_dataloader,
                train_dataloader=train_dataloader,
                max_duration="2ep",
                device="cpu",
            )

            trainer.fit()

            # eval_dataloader already provided:
            trainer.eval()

        For multiple metrics or dataloaders, use :class:`.Evaluator` to provide
        identifier names. For example, to run the GLUE task:

        .. code:: python

            from composer.core import Evaluator
            from composer.models.nlp_metrics import BinaryF1Score

            glue_mrpc_task = Evaluator(
                label='glue_mrpc',
                dataloader=mrpc_dataloader,
                metric_names=['BinaryF1Score', 'MulticlassAccuracy']
            )

            glue_mnli_task = Evaluator(
                label='glue_mnli',
                dataloader=mnli_dataloader,
                metric_names=['MulticlassAccuracy']
            )

            trainer = Trainer(
                ...,
                eval_dataloader=[glue_mrpc_task, glue_mnli_task],
                ...
            )

        The metrics used are defined in your model's ``get_metrics()`` method. For more information,
        see :doc:`/trainer/evaluation`.

        .. note::

            If evaluating with multiple GPUs using a DistributedSampler with `drop_last=False`, the last
            batch will contain duplicate samples, which may affect metrics. To avoid this, as long as
            the dataset passed to the DistributedSampler has a length defined, Composer will correctly
            drop duplicate samples.

        Args:
            eval_dataloader (Iterable | DataLoader | DataSpec | Evaluator | Sequence[Evaluator], optional): Dataloaders
                for evaluation.  If not provided, defaults to using the
                ``eval_dataloader`` provided to the trainer init().
            subset_num_batches (int, optional): Evaluate on this many batches. Default to ``-1`` (the entire
                dataloader. Can also be provided in the trainer.__init__() as ``eval_subset_num_batches``.

        """
        self.engine.run_event(Event.EVAL_STANDALONE_START)

        if eval_dataloader is not None:
            eval_passed_in = True
            eval_metrics = deepcopy(self._original_model.get_metrics(is_train=False))
            metric_names = [str(k) for k in eval_metrics.keys()]

            eval_dataloader = ensure_tuple(eval_dataloader)

            evaluator_types = [isinstance(evaluator, Evaluator) for evaluator in eval_dataloader]
            if any(evaluator_types) and not all(evaluator_types):
                raise ValueError('Mixing Evaluator with other classes is not allowed, please wrap'
                                 'all other classes with the Evaluator class. These are the classes'
                                 'that were detected:' + str([type(evaluator) for evaluator in eval_dataloader]))

            evaluators = [
                ensure_evaluator(evaluator, default_metric_names=metric_names) for evaluator in eval_dataloader
            ]

            if self.state.eval_metrics:
                for evaluator in evaluators:
                    if evaluator.label in self.state.eval_metrics:
                        warnings.warn(
                            f'eval_dataloader label \'{evaluator.label}\' was already provided in '
                            'trainer initialization. Existing data for that label will be overwritten. '
                            'To prevent this in the future, assign unique label names.',
                            category=UserWarning)

            # match metric names to model metrics
            log.info(f'Added {[e.label for e in evaluators]} to eval_metrics.')
            self.state.eval_metrics.update({e.label: _filter_metrics(eval_metrics, e.metric_names) for e in evaluators})

            _set_evaluator_interval_and_subset_num_batches(
                evaluators=evaluators,
                eval_interval='1ep',  # ignored
                subset_num_batches=subset_num_batches,
            )

            for evaluator in evaluators:
                validate_eval_automicrobatching(evaluator.auto_microbatching, self.state.device)

            self.state.evaluators.extend(evaluators)  # Add evaluators to state.evaluators
        else:
            eval_passed_in = False
            if not self.state.evaluators:
                raise ValueError('eval_dataloader must be provided to either Trainer init() or eval().')
            evaluators = self.state.evaluators

        for evaluator in evaluators:
            eval_subset_num_batches = evaluator.subset_num_batches if subset_num_batches == -1 else subset_num_batches
            self._eval_loop(
                evaluator=evaluator,
                metrics=self.state.eval_metrics[evaluator.label],
                subset_num_batches=eval_subset_num_batches,
            )
            if eval_passed_in:
                self.state.evaluators.remove(evaluator)  # Remove them from state once eval is finished.

        self.engine.run_event(Event.EVAL_STANDALONE_END)

    def _eval_loop(
        self,
        evaluator: Evaluator,
        metrics: Dict[str, Metric],
        subset_num_batches: Optional[int] = None,
    ):
        """Evaluate the model and log appropriate metrics.

        Args:
            evaluator (Evaluator): The evaluator to use for evaluation.
            metrics (Dict[str, Metric]): Dictionary mapping metric names to metrics to evaluate against.
            subset_num_batches (int, optional): If specified, evaluate on this many batches. Defaults to ``-1``,
                which means to iterate over the entire dataloader.
        """
        if subset_num_batches is None:
            subset_num_batches = -1

        if os.environ.get('SAVE_LOGITS_FOLDER', None):
            saved_logits = False

        # back up the original dataloader on the state, so we can restore it after evaluation is finished
        original_dataloader = self.state.dataloader
        original_dataloader_label = self.state.dataloader_label
        original_num_batches = self.state.dataloader_len

        # Unpack data_spec
        data_spec = evaluator.dataloader

        # Reset the eval timestamp
        self.state.eval_timestamp = Timestamp()

        last_wct = datetime.datetime.now()

        # NOTE (Sbr): Set batch gather group size to TP group size on eval
        # as we disable SP processing on eval for now.
        # ---
        # TODO (Sbr): Fix this back to TP-SP group size when the issue is fixed.
        batch_gather_group_size = dist.get_tp_group_size()

        with torch.no_grad(), model_eval_mode(self.state.model):
            self.state.set_dataloader(data_spec.dataloader, evaluator.label, subset_num_batches)
            assert self.state.dataloader is not None, 'Dataloader is not set.'

            self.engine.run_event(Event.EVAL_START)

            metrics = self._ensure_metrics_device_and_dtype(metrics)

            for metric in metrics.values():
                metric.reset()

            dataloader = self.state.dataloader
            dist_sampler = None
            drop_last = None
            dataset_len = None
            last_batch = False
            if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                # The distributed sampler uses `set_epoch` to set the random seed
                # Because evaluation can run on each batch, we use the batch to seed the sampler
                # so each evaluation will get a proper shuffle.
                # The epoch provided to `set_epoch` need not be sequential, so this is fine.
                dist_sampler = dataloader.sampler
                dist_sampler.set_epoch(int(self.state.timestamp.batch))
                drop_last = dataloader.drop_last
                # Only compute the dataset length if drop_last is False, as otherwise we don't need
                # to remove any duplicate samples.
                if drop_last == False:
                    try:
                        dataset_len = len(dist_sampler.dataset)  # type: ignore
                    except AttributeError:
                        warnings.warn("DistributedSampler's dataset does not have length defined. When "
                                      '`drop_last=False`, metrics may be incorrect, as DistributedSampler '
                                      'duplicates samples to make the dataset divisible by world size. To '
                                      'fix this, provide a dataset with a length attribute to the '
                                      'DistributedSampler to correctly drop duplicate samples.')

            for batch_idx, self.state.batch in enumerate(self._iter_dataloader(TrainerMode.EVAL)):
                self.state.batch = self.state.device.batch_to_device(self.state.batch)
                if data_spec.device_transforms is not None:
                    self.state.batch = data_spec.device_transforms(self.state.batch)

                # Count the batch size and num tokens before any events run
                rank_num_samples = data_spec.get_num_samples_in_batch(self.state.batch)
                rank_num_tokens = data_spec.get_num_tokens_in_batch(self.state.batch)

                # If using a distributed sampler, keep track of last_batch for metrics update
                if dist_sampler is not None and drop_last == False and dataset_len is not None:
                    batch_num_samples_tensor = self.state.device.tensor_to_device(torch.tensor(rank_num_samples))
                    dist.all_reduce(batch_num_samples_tensor, reduce_operation='SUM')
                    batch_num_samples = batch_num_samples_tensor.item()
                    last_batch = self.state.eval_timestamp.sample + batch_num_samples >= dataset_len

                # Run eval event in case of no additioanl (TP or SP) parallelism or when we encounter
                # the first sample in a batch (that we gathered in TP or/and SP case).
                if batch_gather_group_size is None or batch_idx % batch_gather_group_size == 0:
                    self.engine.run_event(Event.EVAL_BATCH_START)

                # Cache the device batch, because `self.state.batch` gets overridden in microbatching loop
                device_batch = self.state.batch
                # Retry until we successfully complete evaluation
                while True:
                    # Note: We use uint8 instead of bool as BOR is not supported on all torch.distributed backends
                    found_cuda_oom = 0
                    try:
                        microbatches = data_spec.split_batch(device_batch, evaluator.device_eval_microbatch_size)
                        for i, self.state.batch in enumerate(microbatches):
                            last_microbatch = i == len(microbatches) - 1
                            skip_metric_update = False
                            # Distributed samplers pad batches to be the same size. If using a
                            # distributed sampler and on last batch, remove the padding
                            if dist_sampler is not None and drop_last == False and dataset_len is not None and last_batch and last_microbatch:
                                # How many samples are padded samples
                                padding = dist_sampler.total_size - dataset_len

                                if batch_gather_group_size is None:
                                    # Truncate only on those ranks where padding would be present
                                    truncate = dist.get_global_rank() >= dist.get_world_size() - padding
                                else:
                                    truncate = True
                                    first_padded_batch_group = (dist.get_world_size() - padding) // dist.get_fsdp_group_size()
                                    batch_group = batch_idx % batch_gather_group_size
                                    if batch_group < first_padded_batch_group:
                                        truncate = False
                                    elif batch_group == first_padded_batch_group:
                                        first_padded_in_minibatch_group = (dist.get_world_size() - padding) % dist.get_fsdp_group_size()
                                        if dist.get_fsdp_group_rank() < first_padded_in_minibatch_group:
                                            truncate = False

                                if truncate:
                                    # NOTE (Sbr): Could not have padded more than 1 sample.
                                    rank_num_samples -= 1
                                    num_samples_in_microbatch = data_spec.get_num_samples_in_batch(self.state.batch)
                                    # Skip updating metric if batch is only padded samples
                                    if num_samples_in_microbatch == 1:
                                        skip_metric_update = True
                                    # Remove padded samples from batch
                                    else:
                                        self.state.batch = data_spec.split_batch(self.state.batch,
                                                                                 num_samples_in_microbatch - 1)[0]

                            self.engine.run_event(Event.EVAL_BEFORE_FORWARD)

                            with _get_precision_context(self.state.precision, self.state.precision_config):
                                self.state.outputs = self._original_model.eval_forward(self.state.batch, inner_microbatch_size=evaluator.device_eval_microbatch_size)

                            self.engine.run_event(Event.EVAL_AFTER_FORWARD)

                            if os.environ.get('SAVE_LOGITS_FOLDER', None) and not saved_logits and dist.get_global_rank() == 0:
                                batch = self.state.timestamp.batch.value
                                torch.save(self.state.outputs, f"{os.environ['SAVE_LOGITS_FOLDER'][:-3]}-{batch}.pt")
                                saved_logits = True

                            # Skip metric update if batch is only padded samples. We do this after
                            # forward as all models must run forward for FSDP.
                            if skip_metric_update:
                                continue

                            # Run in same precision context to avoid NaNs
                            with _get_precision_context(self.state.precision, self.state.precision_config):
                                if isinstance(self.state.device, DeviceMPS):
                                    # torchmetrics math has numerical errors on M1 devices
                                    # running the compute on CPU instead
                                    if isinstance(self.state.outputs, Mapping):
                                        outputs = {}
                                        for k, v in self.state.outputs.items():
                                            if isinstance(v, torch.Tensor):
                                                outputs[k] = v.cpu()
                                            else:
                                                outputs[k] = v
                                    else:
                                        outputs = self.state.outputs.cpu()
                                else:
                                    outputs = self.state.outputs

                                for metric in metrics.values():
                                    self._original_model.update_metric(
                                        self.state.batch,
                                        outputs,
                                        metric,
                                    )

                    except RuntimeError as e:
                        if evaluator.auto_microbatching and _is_cuda_oom(e):
                            log.debug((f"Rank {dist.get_global_rank()} OOM'd."))
                            found_cuda_oom = 1
                        elif self.state.auto_microbatching and ('cuda' in str(e).lower() or 'c10' in str(e).lower()):
                            raise ValueError(
                                textwrap.dedent(
                                    'Encountered non-addressable cuda error while using auto microbatching. '
                                    'If this repeatedly occurs, set `device_eval_microbatch_size` manually.')) from e
                        else:
                            raise

                    if evaluator.auto_microbatching:
                        # Propagate across all ranks if any rank hit CUDA OOM
                        found_cuda_oom = self.state.device.tensor_to_device(
                            torch.tensor([found_cuda_oom], dtype=torch.uint8))
                        dist.all_reduce(found_cuda_oom, reduce_operation='MAX')
                        if found_cuda_oom.item() == 1:
                            _adjust_device_eval_microbatch_size(evaluator)
                            # Skip return and rerun after handling oom
                            continue
                        # Log device_eval_microbatch_size if auto_microbatching is enabled
                        self.logger.log_metrics({
                            f'trainer/{evaluator.label}/device_eval_microbatch_size':
                                evaluator.device_eval_microbatch_size
                        })
                    # Break if we've successfully completed eval without OOMing.
                    break

                # Run batch end even in case of no additional parallelism (TP and/or SP)
                # or when we encounter the last sample in a batch (gathered from TP-SP ranks).
                if batch_gather_group_size is None or (batch_idx + 1) % batch_gather_group_size == 0:
                    now = datetime.datetime.now()
                    batch_time = now - last_wct

                    total_num_samples, total_num_tokens, batch_time = self._accumulate_time_across_ranks(
                        num_samples=rank_num_samples,
                        num_tokens=rank_num_tokens,
                        batch_time=batch_time,
                    )

                    self.state.eval_timestamp = self.state.eval_timestamp.to_next_batch(
                        samples=total_num_samples,
                        tokens=total_num_tokens,
                        duration=batch_time,
                    )

                    last_wct = now

                    self.engine.run_event(Event.EVAL_BATCH_END)

            self._compute_and_log_metrics(dataloader_label=evaluator.label, metrics=metrics)

            self.engine.run_event(Event.EVAL_END)

        self.state.set_dataloader(original_dataloader, original_dataloader_label)
        if original_num_batches is not None:
            self.state.dataloader_len = original_num_batches

    def _use_grad_scaling(self, precision: Union[str, Precision], scaler: Optional[GradScaler]) -> bool:
        """Determines based on precision when to use grad scaling.

        By default, the pytorch GradScaler is a no-op if running on
        unsupported hardware. Here we raise a RuntimeError instead.

        Args:
            precision (Precision): Numerical precision, based on the Precision Enum.
            scaler (GradScaler): Used to make sure that the scaler is enabled when
            using grad scaling.

        Raises:
            RuntimeError:
                Occurs when attempting to use grad scaling without the scaler
                enabled. Likely due to hardware not supporting the provided precision.
        """

        precision = Precision(precision)
        use_grad_scaling = precision == Precision.AMP_FP16

        if use_grad_scaling and (scaler is None or not scaler.is_enabled()):
            raise RuntimeError(f'Attempting to use grad scaling with {precision}, but scaler is not enabled.'
                               f'Potentially your hardware does not support Precision {precision}.')
        return use_grad_scaling

    def _iter_dataloader(self, trainer_mode: TrainerMode):
        """Helper method to iterate over the dataloader.

        This method yields up to :attr:`.State.dataloader_len`` batches from the dataloader. In addition, if the
        profiler is enabled, the dataloader latency recorded via the :class:`.Marker` API.

        Args:
            trainer_mode (TrainerMode): Specifies which mode the trainer is in.
        """
        assert self.state.dataloader is not None, 'the dataloader should be set before calling this method'

        # Initialize dataloader iterator
        if self.state.dataloader_len is None:
            dataloader_iter = iter(self.state.dataloader)
        else:
            dataloader_iter = itertools.islice(self.state.dataloader, int(self.state.dataloader_len))

        batch_idx = 0
        while True:
            try:
                batch = next(dataloader_iter)

                if trainer_mode == TrainerMode.TRAIN:
                    self.state.raw_batch = batch
                    self.engine.run_event(Event.BEFORE_BATCH_SPLIT_DATALOADER)

                # Always calculate sequence stats if local_max_seq_len is provided
                seq_stats = None
                if "local_max_seq_len" in batch:
                    local_max_seq_len = batch.get("local_max_seq_len")
                    seq_stats = self._calculate_sequence_stats(local_max_seq_len)

                # Handle dynamic SP if enabled
                if (
                    self.dynamic_sp is not None
                    and len(self.dynamic_sp) > 1
                    and trainer_mode == TrainerMode.TRAIN
                ):
                    assert seq_stats is not None, (
                        "seq_stats must be provided if dynamic_sp is enabled. "
                        "Maybe you forgot to set `local_max_seq_len` in your dataloader?"
                    )
                    self._handle_dynamic_sp(batch, **seq_stats)

                # Process batch and yield results
                for processed_batch in self._process_batch(batch, batch_idx, trainer_mode):
                    batch_idx += 1
                    if trainer_mode == TrainerMode.TRAIN:
                        self.engine.run_event(Event.BEFORE_DATALOADER)
                    yield processed_batch

            except StopIteration:
                # Handle end of dataloader
                if trainer_mode == TrainerMode.TRAIN:
                    self.engine.run_marker_only_event(Event.AFTER_DATALOADER)
                break

    def _calculate_sequence_stats(self, local_max_seq_len: int) -> Dict[str, float]:
        """Calculate sequence length statistics across all ranks.

        Args:
            local_max_seq_len: The maximum sequence length for the current rank

        Returns:
            Dictionary containing min, average, and global max sequence lengths
        """
        local_sequence_length = torch.tensor([local_max_seq_len], device='cuda')
        gathered_sequence_lengths = dist.all_gather(local_sequence_length)
        sequence_lengths = [sequence_length.item() for sequence_length in gathered_sequence_lengths]
        min_seq_len = min(sequence_lengths)
        avg_seq_len = sum(sequence_lengths) / len(sequence_lengths)
        global_max_seq_len = max(sequence_lengths)

        # Log the metrics
        self.logger.log_metrics({
            'trainer/dataloader/min_max_seq_len': min_seq_len,
            'trainer/dataloader/avg_max_seq_len': avg_seq_len,
            'trainer/dataloader/global_max_seq_len': global_max_seq_len
        })

        return {
            'min_seq_len': min_seq_len,
            'avg_seq_len': avg_seq_len,
            'global_max_seq_len': global_max_seq_len
        }

    def _handle_dynamic_sp(self, batch: Dict[str, Any], **kwargs) -> None:
        """Handle dynamic sequence parallelism configuration.

        Args:
            batch: The current batch from the dataloader
            **kwargs: Optional pre-calculated sequence statistics
        """
        local_max_seq_len = batch.get("local_max_seq_len")
        if local_max_seq_len is None:
            raise Exception("local_max_seq_len is not provided but dynamic_sp is set")

        assert 'global_max_seq_len' in kwargs, (
            "global_max_seq_len must be provided if dynamic_sp is enabled. "
        )

        # Determine appropriate SP size based on sequence length
        new_sp_size = self._get_sp_size_for_seq_len(kwargs['global_max_seq_len'])
        current_sp_size = dist.get_sp_group_size()

        # Switch SP group if needed
        if new_sp_size != current_sp_size:
                    # Log the metrics
            self._switch_sp_group(new_sp_size)

        self.logger.log_metrics({
            'trainer/dynamic_sp/sp_size': dist.get_sp_group_size(),
            'trainer/dynamic_sp/tp_sp_size': dist.get_tp_sp_group_size(),
        })

    def _get_sp_size_for_seq_len(self, seq_len: int) -> int:
        """Determine the appropriate sequence parallelism size for a given sequence length.

        Args:
            seq_len: The sequence length to find SP size for

        Returns:
            The appropriate SP size
        """
        if self.dynamic_sp is None:
            return 1

        for config in self.dynamic_sp:
            if seq_len <= config["max_seq_len"]:
                return config["sp_size"]
        raise Exception(f"Provided sp sizes {self.dynamic_sp} are less then {seq_len=}")

    def _switch_sp_group(self, new_sp_size: int) -> None:
        """Switch to a new sequence parallelism group size.

        Args:
            new_sp_size: The new SP size to switch to
        """
        if self.dynamic_sp is None:
            return

        max_sp_size = max(self.dynamic_sp, key=lambda x: x["max_seq_len"])["sp_size"]

        # Adjust gradient accumulation steps based on SP size ratio
        if not self.state.gigafsdp_enabled:
            model_type = type(self.state.model.model)
            raise Exception(
                f"Dynamic sp is only available for GigaFSDP models. "
                f"Current model type: {model_type} from module {model_type.__module__}. "
            )

        # Adjust gradient accumulation steps based on SP size ratio
        self.state.model.model._gradient_accumulation_steps = (
            self.state.model.model._default_gradient_accumulation_steps //
            (max_sp_size // new_sp_size)
        )

        # Set the new SP group
        dist.set_sp_dynamic_group(new_sp_size)

    def _process_batch(self, batch: Dict[str, Any], batch_idx: int, trainer_mode: TrainerMode) -> List[Dict[str, Any]]:
        """Process a batch, potentially splitting it and handling tensor parallelism.

        Args:
            batch: The batch to process
            batch_idx: Current batch index
            trainer_mode: The trainer mode

        Yields:
            Processed batch parts
        """
        # Split batch if needed

        if (
            self.dynamic_sp is not None
            and len(self.dynamic_sp) > 1
            and trainer_mode == TrainerMode.TRAIN
        ):
            # dataloader returns batch with biggest sequence parallelism size
            # so we need to split it into smaller batches
            # to get an opportunity to switch SP group from gradient accumulation
            sp_group_size = dist.get_sp_group_size() or 1
            max_sp_size = max(self.dynamic_sp, key=lambda x: x["max_seq_len"])["sp_size"]
            batch_splits = recursive_split(batch, max_sp_size // sp_group_size, dim=0, split_batch_fn=self._train_data_spec.split_batch)
        else:
            batch_splits = [batch]

        # Process each split
        for batch_split in batch_splits:
            is_tp_sp_active = dist.get_tp_sp_group() is not None and dist.get_tp_sp_group_size() > 1

            if not is_tp_sp_active:
                # Simple case - no tensor parallelism with sequence parallelism
                yield batch_split
            else:
                # Handle tensor parallelism with sequence parallelism
                yield from self._handle_tp_sp_batch(batch_split, trainer_mode)

    def _handle_tp_sp_batch(self, batch: Dict[str, Any], trainer_mode: TrainerMode) -> List[Dict[str, Any]]:
        """Handle a batch with tensor parallelism and sequence parallelism.

        Args:
            batch: The batch to process
            trainer_mode: The trainer mode

        Yields:
            Processed batches
        """
        # Determine the appropriate gather group based on mode
        if trainer_mode == TrainerMode.TRAIN:
            batch_gather_group = dist.get_tp_sp_group()
            batch_gather_group_size = dist.get_tp_sp_group_size()
            batch_gather_group_rank = dist.get_tp_sp_group_rank()
        else:
            # EVAL and PREDICT modes with tensor parallelism
            batch_gather_group = dist.get_tp_group()
            batch_gather_group_size = dist.get_tp_group_size()
            batch_gather_group_rank = dist.get_tp_group_rank()

        if batch_gather_group_size is not None and batch_gather_group_size > 1:
            if os.getenv("BROADCAST_BATCHES_IN_SP_TP_GROUP_ENABLED"):

                class _batch_gather_group_iter:
                    def __iter__(self):
                        self.broadcast_rank = 0
                        return self

                    def __next__(self):
                        if self.broadcast_rank == batch_gather_group_size:
                            raise StopIteration
                        received_batch_to_process = None

                        broadcast_rank = self.broadcast_rank
                        global_broadcast_rank = torch.distributed.get_global_rank(batch_gather_group, broadcast_rank)
                        send_batch_to_process = batch if batch_gather_group_rank == broadcast_rank else None
                        obj_list = [send_batch_to_process]
                        torch.distributed.broadcast_object_list(obj_list,
                                                                src=global_broadcast_rank,
                                                                group=batch_gather_group
                                                                )
                        received_batch_to_process = (
                            obj_list[0]
                            if batch_gather_group_rank != broadcast_rank
                            else send_batch_to_process
                        )
                        self.broadcast_rank += 1
                        return received_batch_to_process

                for cur_batch in _batch_gather_group_iter():
                    if trainer_mode == TrainerMode.TRAIN:
                        self.engine.run_event(Event.BEFORE_DATALOADER)
                    yield cur_batch

            else:
                # Gather batches across the group
                batches = [create_empty_batch_like(batch) for _ in range(batch_gather_group_size)]
                torch.distributed.all_gather_object(batches, batch, group=batch_gather_group)

                # Yield each gathered batch
                for gathered_batch in batches:
                    yield gathered_batch
        else:
            # No gathering needed
            yield batch

    def _use_closures(self) -> bool:
        """Determines based on precision and optimizers whether to use closures.

        We default to using closures unless AMP is enabled, in which case we only allow closures when using optimizers
        with the _step_supports_amp_closure flag.
        """

        if isinstance(self.state.device, DeviceTPU):
            return False

        if self.state.precision != Precision.AMP_FP16:
            return True

        if self.state.optimizers is None:
            raise RuntimeError('state.optimizers must be set before `_use_closures` can be determined')

        return all(
            getattr(optimizer, '_step_supports_amp_closure', False)
            for optimizer in ensure_tuple(self.state.optimizers))

    def save_checkpoint(
        self,
        name: str = 'ep{epoch}-ba{batch}-rank{rank}',
        *,
        weights_only: bool = False,
    ):
        """Checkpoint the training :class:`~.State`.

        Args:
            name (str, optional): See :func:`.save_checkpoint`.
            weights_only (bool, optional): See :func:`.save_checkpoint`.

        Returns:
            str or None: See :func:`.save_checkpoint`.
        """
        return checkpoint.save_checkpoint(
            state=self.state,
            filename=name,
            weights_only=weights_only,
        )

    def save_checkpoint_to_save_folder(self):
        """Checkpoints the training :class:`~.State` using a CheckpointSaver if it exists.

        Raises:
            ValueError: If ``_checkpoint_saver`` does not exist.

        Returns:
            None
        """
        if self._checkpoint_saver is None:
            raise ValueError(
                'In order to use save_checkpoint_to_save_folder you must pass a save_folder to the Trainer.')
        else:
            self._checkpoint_saver._save_checkpoint(self.state, self.logger)

    def export_for_inference(
        self,
        save_format: Union[str, ExportFormat],
        save_path: str,
        save_object_store: Optional[ObjectStore] = None,
        sample_input: Optional[Any] = None,
        transforms: Optional[Sequence[Transform]] = None,
        input_names: Optional[Sequence[str]] = None,
        output_names: Optional[Sequence[str]] = None,
    ):
        """Export a model for inference.

        Args:
            save_format (Union[str, ExportFormat]):  Format to export to. Either ``"torchscript"`` or ``"onnx"``.
            save_path: (str): The path for storing the exported model. It can be a path to a file on the local disk,
            a URL, or if ``save_object_store`` is set, the object name
                in a cloud bucket. For example, ``my_run/exported_model``.
            save_object_store (ObjectStore, optional): If the ``save_path`` is in an object name in a cloud bucket
                (i.e. AWS S3 or Google Cloud Storage), an instance of
                :class:`~.ObjectStore` which will be used
                to store the exported model. If this is set to ``None``,  will save to ``save_path`` using the trainer's
                logger. (default: ``None``)
            sample_input (Any, optional): Example model inputs used for tracing. This is needed for "onnx" export.
                The ``sample_input`` need not match the batch size you intend to use for inference. However, the model
                should accept the ``sample_input`` as is. (default: ``None``)
            transforms (Sequence[Transform], optional): transformations (usually optimizations) that should
                be applied to the model. Each Transform should be a callable that takes a model and returns a modified model.
            input_names (Sequence[str], optional): names to assign to the input nodes of the graph, in order. If set
                to ``None``, the keys from the `sample_input` will be used. Fallbacks to ``["input"]``.
            output_names (Sequence[str], optional): names to assign to the output nodes of the graph, in order. It set
                to ``None``, it defaults to ``["output"]``.

        Returns:
            None
        """
        export_model = self.state.model
        if not isinstance(export_model, nn.Module):
            raise ValueError(f'Exporting Model requires type torch.nn.Module, got {type(export_model)}')
        if sample_input == None and save_format == 'onnx':
            sample_input = self.state.batch
        export_with_logger(model=export_model,
                           save_format=save_format,
                           save_path=save_path,
                           logger=self.logger,
                           save_object_store=save_object_store,
                           sample_input=(sample_input, {}),
                           transforms=transforms,
                           input_names=input_names,
                           output_names=output_names)
