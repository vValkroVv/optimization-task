# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callback to save checkpoints during training."""

from __future__ import annotations

import re
import concurrent.futures
import functools
import logging
import os
import pathlib
import shutil
import threading
from copy import copy
import tempfile
import textwrap
from pathlib import Path
from typing import Callable, List, Optional, Union

from composer.core import Callback, Event, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, FORMAT_NAME_WITH_DIST_TABLE, PartialFilePath,
                            checkpoint, create_interval_scheduler, create_symlink_file, dist,
                            ensure_folder_has_no_conflicting_files, format_name_with_dist,
                            format_name_with_dist_and_time, reproducibility)
from composer.utils.checkpoint import _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME
from composer.utils.tcp_barrier import get_tcp_barrier

log = logging.getLogger(__name__)

__all__ = ['CheckpointSaver', 'get_current_checkpoint_save']

_TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME = '.metadata'


class CheckpointSaver(Callback):  # noqa: D101
    __doc__ = f"""Callback to save checkpoints.

    .. note::

        If the ``folder`` argument is specified when constructing the :class:`.Trainer`, then the :class:`.CheckpointSaver`
        callback need not be constructed manually. However, for advanced checkpointing use cases
        (such as saving a weights-only checkpoint at one interval and the full training state
        at another interval), instance(s) of this :class:`.CheckpointSaver` callback can be specified in the
        ``callbacks`` argument of the :class:`.Trainer`, as shown in the example below.

    Example

    .. testsetup::

        from composer.callbacks.checkpoint_saver import CheckpointSaver

    .. doctest::

        >>> trainer = Trainer(..., callbacks=[
        ...     CheckpointSaver(
        ...         folder='{{run_name}}/checkpoints',
        ...         filename="ep{{epoch}}-ba{{batch}}-rank{{rank}}",
        ...         latest_filename="latest-rank{{rank}}",
        ...         save_interval="1ep",
        ...         weights_only=False,
        ...     )
        ... ])

    Args:
        folder (str, optional): Format string for the save_folder where checkpoints will be saved.
            Default: ``'{{run_name}}/checkpoints'``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_TABLE, prefix='            ')}

            .. note::

                When training with multiple devices (i.e. GPUs), ensure that ``'{{rank}}'`` appears in the format.
                Otherwise, multiple processes may attempt to write to the same file.

        filename (str, optional): A format string describing how to name checkpoints.
            Default: ``'ep{{epoch}}-ba{{batch}}-rank{{rank}}.pt'``.

            Checkpoints will be saved approximately to ``{{folder}}/{{filename.format(...)}}``.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, prefix='            ')}


            .. note::

                *   By default, only the rank zero process will save a checkpoint file.

                *   To use compression set the file extension
                    to ``'.tar.gz'``, ``'.tgz'``, ``'.tar.bzip'``, or ``'.tar.lzma'`` (depending on the desired
                    compression algorithm).

            .. warning::

                Using compression will block the training loop while checkpoints are being compressed. As such, we
                recommend saving checkpoints without compression.

            Consider the following scenario where:

            *   The :attr:`~.State.run_name` is ``'awesome-training-run'``
            *   The default ``folder='{{run_name}}/checkpoints'`` is used.
            *   The default ``name='ep{{epoch}}-ba{{batch}}-rank{{rank}}'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

        remote_file_name (str, optional): Format string for the checkpoint's remote file name.
            Default: ``"{{run_name}}/checkpoints/ep{{epoch}}-ba{{batch}}-rank{{rank}}"``.

            After the checkpoint is saved, it will be periodically uploaded.
            The remote file name will be determined by this format string.

            .. seealso:: :doc:`Uploading Files</trainer/file_uploading>` for notes for file uploading.

            The same format variables for ``filename`` are available.

            Leading slashes (``'/'``) will be stripped.

            To disable uploading checkpoints, set this parameter to ``None``.
        latest_filename (str, optional): A format string for a symlink which points to the last saved checkpoint.
            Default: ``'latest-rank{{rank}}.pt'``.

            Symlinks will be created approximately at ``{{folder}}/{{latest_filename.format(...)}}``.

            The same format variables as for ``name`` are available.

            To disable symlinks, set this parameter to ``None``.

            Consider the following scenario, where:

            *   The :attr:`~.State.run_name` is 'awesome-training-run'
            *   The default ``folder='{{run_name}}/checkpoints'`` is used.
            *   The default ``name='ep{{epoch}}-ba{{batch}}-rank{{rank}}'`` is used.
            *   The default ``latest_filename='latest-rank{{rank}}'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            Corresponding symlinks will be created at::

                awesome-training-run/checkpoints/latest-rank0.tar -> awesome-training-run/checkpoints/ep1-ba42-rank0.tar
                awesome-training-run/checkpoints/latest-rank1.tar -> awesome-training-run/checkpoints/ep1-ba42-rank1.tar
                awesome-training-run/checkpoints/latest-rank2.tar -> awesome-training-run/checkpoints/ep1-ba42-rank2.tar
                ...
        latest_remote_file_name (str, optional): Format string for the checkpoint's latest symlink remote file name.
            Default: ``'{{run_name}}/checkpoints/latest-rank{{rank}}"``.

            Whenever a new checkpoint is saved, a symlink is created or updated to point to the latest checkpoint's ``remote_file_name``.
            The remote file name will be determined by this format string. This parameter has no effect if ``latest_filename`` or ``remote_file_name`` is ``None``.

            .. seealso:: :doc:`Uploading Files</trainer/file_uploading>` for notes for file uploading.

            The same format variables for ``filename`` are available.

            Leading slashes (``'/'``) will be stripped.

            To disable symlinks in logger, set this parameter to ``None``.

        overwrite (bool, optional): Whether existing checkpoints should be overridden.
            If ``False`` (the default), then the ``folder`` must not exist or must not contain checkpoints which may conflict
            with the current run. Default: ``False``.

        save_interval (Time | str | int | (State, Event) -> bool): A :class:`.Time`, time-string, integer (in epochs),
            or a function that takes (state, event) and returns a boolean whether a checkpoint should be saved.

            If an integer, checkpoints will be saved every n epochs.
            If :class:`.Time` or a time-string, checkpoints will be saved according to this interval.

            .. seealso:: :func:`.checkpoint_periodically`

            If a function, then this function should take two arguments (:class:`.State`, :class:`.Event`).
            The first argument will be the current state of the trainer, and the second argument will be
            be :attr:`.Event.BATCH_CHECKPOINT` or :attr:`.Event.EPOCH_CHECKPOINT` (depending on the current training
            progress). It should return ``True`` if a checkpoint should be saved given the current state and
            event.

        weights_only (bool): If ``True``, save only the model weights instead of the entire training state.
            Default: ``False``.


        num_checkpoints_to_keep (int, optional): The number of checkpoints to keep locally. The oldest checkpoints
            are removed first. Set to ``-1`` to keep all checkpoints locally. Default: ``-1``.

            Checkpoints will be removed after they have been uploaded. For example, when this callback
            is used in conjunction with the :class:`.RemoteUploaderDownloader`, set this
            parameter to ``0`` to immediately delete checkpoints from the local disk after they have been uploaded to
            the object store.

            This parameter only controls how many checkpoints are kept locally; checkpoints are not deleted from
            remote file systems.

    Attributes:
        saved_checkpoints (List[Tuple[Timestamp, List[pathlib.Path]]]): The checkpoint timestamps and filepaths.

            This list contains tuples of the save timestamp and the checkpoint filepaths.
            This list will have at most ``num_checkpoints_to_keep`` entries. The latest checkpoint
            will be at the end.

            .. note::

                Each sub-list will contain only one filepath since only rank zero
                saves checkpoints.
    """

    def __init__(
        self,
        folder: Union[str, pathlib.Path] = '{run_name}/checkpoints',
        filename: Union[str, pathlib.Path] = 'ep{epoch}-ba{batch}-rank{rank}.pt',
        remote_file_name: Optional[Union[str,pathlib.Path]] = '{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}.pt',
        latest_filename: Optional[Union[str, pathlib.Path]] = 'latest-rank{rank}.pt',
        latest_remote_file_name: Optional[Union[str, pathlib.Path]] = '{run_name}/checkpoints/latest-rank{rank}.pt',
        checkpoint_directory_name: Union[str, pathlib.Path] = 'ep{epoch}_ba{batch}',
        save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = '1ep',
        stable_save_interval: Optional[str] = None,
        stable_save_postfix: str = ".stable",
        *,
        overwrite: bool = False,
        num_checkpoints_to_keep: int = -1,
        weights_only: bool = False,
        save_lora_adapter_only: bool = False,
        async_save: bool = False
    ):
        folder = str(folder)
        filename = str(filename)
        remote_file_name = str(remote_file_name) if remote_file_name is not None else None
        latest_filename = str(latest_filename) if latest_filename is not None else None
        latest_remote_file_name = str(latest_remote_file_name) if latest_remote_file_name is not None else None

        if stable_save_interval is not None:
            if callable(stable_save_interval):
                self.stable_save_interval_value = getattr(stable_save_interval, 'batch_steps', None)
            else:
                self._validate_save_intervals(save_interval, stable_save_interval)
                self.stable_save_interval_value = int(stable_save_interval[:-len("ba")])
                stable_save_interval = create_interval_scheduler(stable_save_interval)
            assert filename == 'ep{epoch}-ba{batch}-rank{rank}.pt', "Please change filename template to asserted, others are not supported for stable checkpoints"
            assert latest_filename == 'latest-rank{rank}.pt', "Please change latest filename template to asserted, others are not supported for stable checkpoints"
 
        if not callable(save_interval):
            save_interval = create_interval_scheduler(save_interval)

        self.save_interval = save_interval
        self.stable_save_interval = stable_save_interval
        self.stable_save_postfix = stable_save_postfix
        self.last_checkpoint_batch: Optional[Time] = None
        self.checkpoint_directory_name = checkpoint_directory_name
        
        self.folder = folder

        self.filename = PartialFilePath(filename.lstrip('/'), folder)
        self.latest_filename = PartialFilePath(latest_filename.lstrip('/'), folder) if latest_filename else None
        self.remote_file_name = PartialFilePath(remote_file_name) if remote_file_name else None
        self.latest_remote_file_name = PartialFilePath(latest_remote_file_name) if latest_remote_file_name else None

        self.overwrite = overwrite
        self.saved_checkpoints: List[str] = []

        self.num_checkpoints_to_keep = num_checkpoints_to_keep
        self.weights_only = weights_only
        self.save_lora_adapter_only = save_lora_adapter_only

        self.start_batch = None
        self._thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if async_save else None
        self.async_save = async_save
        self._completed_checkpoint_saves: List[dict] = []
        self._completed_checkpoint_saves_lock = threading.Lock()
        self._current_checkpoint_save: Optional[dict] = None

        if self.async_save:
            self.last_save_job_future = None

        self.shard_save_barrier = None

    def init(self, state: State, logger: Logger) -> None:
        folder = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(folder, exist_ok=True)
        # We are using GigaFSDP
        # and all ranks execute _save_checkpoint
        if state.fsdp_config.get('gigafsdp_enabled', False):
            if state.fsdp_config.get("hybrid_group_size", False):
                num_parties = state.fsdp_config['hybrid_group_size'] * (dist.get_tp_group_size() or 1)
            else:
                num_parties = dist.get_world_size()
        # Saving old-format checkpoint (default FSDP)
        # in this case only tp_size ranks save anything
        else:
            num_parties = state.fsdp_config['tp_size']
        log.debug(f'num_parties={num_parties}')
        self.shard_save_barrier = get_tcp_barrier(num_parties)

    def fit_start(self, state: State, logger: Logger) -> None:
        if not self.overwrite:
            # checks that save_folder contains no files with a timestamp after the current timestamp,
            # which has potential for future conflicts.
            folder = format_name_with_dist(self.folder, state.run_name)

            # Searching for conflicting folders
            ensure_folder_has_no_conflicting_files(folder, self.checkpoint_directory_name, state.timestamp)

        dist.barrier()  # holds all ranks until folder check is done

        self.start_batch = state.timestamp.batch

    def after_forward(self, state: State, logger: Logger):
        if self.async_save and self.last_save_job_future is not None:
            if self.last_save_job_future.done():
                self.last_save_job_future.result()
                log.debug("Checkpoint saved successfully, the thread is joined")
                self.last_save_job_future = None

    def drain_completed_checkpoint_saves(self) -> List[dict]:
        """Return checkpoint saves completed since the last drain."""
        with self._completed_checkpoint_saves_lock:
            completed_checkpoint_saves = list(self._completed_checkpoint_saves)
            self._completed_checkpoint_saves.clear()
        return completed_checkpoint_saves

    def get_current_checkpoint_save(self) -> Optional[dict]:
        """Return the checkpoint save record for the current AFTER_CHECKPOINT_SAVE event."""
        return self._current_checkpoint_save

    def _set_current_checkpoint_save(self, checkpoint_save: Optional[dict]) -> None:
        self._current_checkpoint_save = checkpoint_save

    def wait_for_async_checkpoint(self) -> None:
        """Wait for the in-flight async checkpoint, surfacing save errors."""
        if not self.async_save:
            return

        if self.last_save_job_future is not None:
            self.last_save_job_future.result()
            self.last_save_job_future = None

    def batch_checkpoint(self, state: State, logger: Logger):
        assert callable(self.save_interval)
        if self.save_interval(state, Event.BATCH_CHECKPOINT) and self.last_checkpoint_batch != state.timestamp.batch:
            is_stable = (
                self.stable_save_interval is not None
                and self.stable_save_interval(state, Event.BATCH_CHECKPOINT)
            )
            self._save_checkpoint(
                state,
                logger,
                is_stable_checkpoint=is_stable,
            )

    def epoch_checkpoint(self, state: State, logger: Logger):
        assert callable(self.save_interval)
        if self.save_interval(state, Event.EPOCH_CHECKPOINT) and self.last_checkpoint_batch != state.timestamp.batch:
            is_stable = (
                self.stable_save_interval is not None
                and self.stable_save_interval(state, Event.EPOCH_CHECKPOINT)
            )
            self._save_checkpoint(
                state,
                logger,
                is_stable_checkpoint=is_stable
            )

    def get_state_dict(self, state):
        return {
            'state': state.state_dict(),
            'rng': reproducibility.get_rng_state(),
        }

    def _save_checkpoint(self, state: State, logger: Logger, is_stable_checkpoint: bool):
        self.last_checkpoint_batch = state.timestamp.batch

        # Changing save directory from {save_path}/{filename} to {save_path}/{ep}_{ba}/{filename}
        checkpoint_path = copy(self.filename)
        checkpoint_dirname = format_name_with_dist_and_time(self.checkpoint_directory_name, state.run_name, state.timestamp)
        checkpoint_path.folder = os.path.join(checkpoint_path.folder, checkpoint_dirname)
        
        # save the checkpoint to the filename
        filename_with_placeholders = checkpoint_path.format(state, keep_placeholders=True)

        current_batch_value = state.timestamp.batch.value

        def after_checkpoint_save(
            self: CheckpointSaver,
            saved_path: Union[str, None],
            current_batch_value: int,
            is_stable_checkpoint: bool,
        ) -> None:
            log.debug(f'Checkpoint locally saved to {saved_path}')

            checkpoint_save = None
            if self.stable_save_interval is None or is_stable_checkpoint:
                checkpoint_save = {
                    "saved_path": saved_path,
                    "saved_checkpoint_batch_value": current_batch_value
                }

            if not saved_path:  # not all ranks save
                if checkpoint_save is not None:
                    with self._completed_checkpoint_saves_lock:
                        self._completed_checkpoint_saves.append(checkpoint_save)
                return
                
            metadata_local_file_path = None
            if (dist.get_global_rank() == 0 or dist.get_fsdp_group_rank() == 0) and state.fsdp_elastic_sharded_enabled:
                metadata_local_file_path = format_name_with_dist_and_time(
                    os.path.join(Path(saved_path).parent, _TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME), state.run_name,
                    state.timestamp)

            if self.latest_filename is not None and self.num_checkpoints_to_keep != 0:
                symlink = self.latest_filename.format(state)
                os.makedirs(os.path.dirname(symlink), exist_ok=True)
                try:
                    os.remove(symlink)
                except FileNotFoundError:
                    pass
                # Sharded checkpoints for torch >2.0 use directories not files for load_paths
                if state.fsdp_elastic_sharded_enabled:
                    src_path = str(pathlib.Path(saved_path).parent)
                else:
                    src_path = saved_path
                this_rank_saves_symlinks = (
                    dist.get_global_rank() == 0 or
                    dist.get_fsdp_group_rank() == 0 or
                    dist.get_ep_fsdp_group_rank() == 0 or
                    not state.fsdp_elastic_sharded_enabled
                )
                if this_rank_saves_symlinks:
                    os.symlink(os.path.relpath(src_path, os.path.dirname(symlink)), symlink)

            # if remote file name provided, upload the checkpoint
            if self.remote_file_name is not None:
                if state.fsdp_sharded_state_dict_enabled:
                    remote_file_name = self.remote_file_name.format(
                        state,
                        keep_placeholders=True,
                    ).lstrip('/')
                    assert state.sharded_ckpt_prefix_dir is not None
                    remote_prefix = state.sharded_ckpt_prefix_dir
                    ckpt_filename = _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME()
                    remote_file_name = os.path.join(pathlib.Path(remote_file_name).parent, remote_prefix, ckpt_filename)
                    remote_file_name = format_name_with_dist_and_time(remote_file_name, state.run_name, state.timestamp)
                    # Upload metadata file.
                    # The metadata file contains info related to which shards are saved where.
                    if (dist.get_global_rank() == 0 or dist.get_fsdp_group_rank() == 0) and state.fsdp_elastic_sharded_enabled:
                        metadata_remote_file_name = format_name_with_dist_and_time(
                            os.path.join(Path(remote_file_name).parent, _TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME),
                            state.run_name, state.timestamp)
                        assert metadata_local_file_path is not None
                        logger.upload_file(remote_file_name=metadata_remote_file_name,
                                           file_path=metadata_local_file_path,
                                           overwrite=self.overwrite)
                else:
                    remote_file_name = self.remote_file_name.format(
                        state,
                    ).lstrip('/')

                log.debug(f'Uploading checkpoint to {remote_file_name}')
                try:
                    logger.upload_file(remote_file_name=remote_file_name, file_path=saved_path, overwrite=self.overwrite)
                except FileExistsError as e:
                    raise FileExistsError(
                        f'Uploading checkpoint failed with error: {e}. overwrite was set to {self.overwrite}. To overwrite checkpoints with Trainer, set save_overwrite to True.'
                    ) from e

                # symlinks stay the same with sharded checkpointing
                if self.latest_remote_file_name is not None:
                    symlink_name = self.latest_remote_file_name.format(
                        state,
                    ).lstrip('/') + '.symlink'

                    # create and upload a symlink file
                    with tempfile.TemporaryDirectory() as tmpdir:
                        symlink_filename = os.path.join(tmpdir, 'latest.symlink')
                        # Sharded checkpoints for torch >2.0 use directories not files for load_paths
                        if state.fsdp_elastic_sharded_enabled:
                            src_path = str(pathlib.Path(remote_file_name).parent)
                        else:
                            src_path = remote_file_name
                        log.debug(f'Creating symlink file {symlink_filename} -> {src_path}')
                        this_rank_saves_symlinks = dist.get_global_rank() == 0 or dist.get_fsdp_group_rank() == 0 or not state.fsdp_elastic_sharded_enabled
                        if this_rank_saves_symlinks:
                            create_symlink_file(src_path, symlink_filename)
                            logger.upload_file(
                                remote_file_name=symlink_name,
                                file_path=symlink_filename,
                                overwrite=True,
                            )
            if self.stable_save_interval is not None and self.saved_checkpoints:
                # TODO: move to separate function as in `self._rotate_checkpoints`
                previous_checkpoint = self.saved_checkpoints[-1]
                log.debug(f"Found previous checkpoint {previous_checkpoint}")
                is_previous_stable = _check_is_stable_checkpoint(
                    previous_checkpoint,
                    self.stable_save_interval_value,
                    state.fsdp_elastic_sharded_enabled,
                )
                if not is_previous_stable:
                    log.debug(f"Removing quick checkpoint {previous_checkpoint}")
                    if not state.fsdp_elastic_sharded_enabled:
                        os.remove(previous_checkpoint)
                    else:
                        if dist.get_global_rank() == 0 or dist.get_fsdp_group_rank() == 0:
                            checkpoint_dir = Path(previous_checkpoint).parent
                            if _all_last_checkpoints_saved(str(checkpoint_dir), self.latest_filename.filename):
                                _remove_sharded_checkpoint(str(checkpoint_dir.parent), doesnt_exist_ok=True)

                    # Removing directory if only artifact dir in here
                    if dist.get_global_rank() == 0:
                        save_folder_path = format_name_with_dist(self.folder, state.run_name)
                        save_subfolder_name = previous_checkpoint.replace(save_folder_path, '').strip('/').split('/')[0]
                        save_subfolder_path = os.path.join(save_folder_path, save_subfolder_name)
                        if os.path.exists(save_subfolder_path) and os.listdir(save_subfolder_path) == ['artifacts']:
                            log.debug(f"Removing empty directory of checkpoint {save_subfolder_path}")
                            shutil.rmtree(save_subfolder_path)

                    self.saved_checkpoints.pop(-1)
            self.saved_checkpoints.append(saved_path)
            log.info(f'{len(self.saved_checkpoints)} checkpoints saved to {saved_path}')

            if self.num_checkpoints_to_keep >= 0:
                self._rotate_checkpoints(sharding_enabled=state.fsdp_sharded_state_dict_enabled)

            # At the end of after_checkpoint_save, add:
            if state.fsdp_sharded_state_dict_enabled and dist.get_global_rank() == 0:
                # For sharded checkpoints, saved_path is a directory
                checkpoint_dir = pathlib.Path(saved_path).parent.parent # parent of model/ep_x/ep_x_rank0/__xx_y.distcp
                saved_file = checkpoint_dir / "saved"
                try:
                    # Try because of race condition of nodes
                    open(saved_file, "a").close()  # 'touch' empty file
                    log.info(f'Wrote "checkpoint-fully-saved" marker file to {saved_file}')
                except Exception as e:
                    log.warning(f'Could not save "checkpoint-fully-saved" marker file to {saved_file}, exception: {str(e)}')

            if checkpoint_save is not None:
                with self._completed_checkpoint_saves_lock:
                    self._completed_checkpoint_saves.append(checkpoint_save)

        if not self.async_save:
            log.debug("Saving checkpoint")
            saved_path = checkpoint.save_checkpoint(
                state=state,
                filename=filename_with_placeholders,
                weights_only=self.weights_only,
                save_lora_adapter_only=self.save_lora_adapter_only,
                barrier=self.shard_save_barrier,
            )
            after_checkpoint_save(self, saved_path, current_batch_value, is_stable_checkpoint)
        else:
            log.debug(f'Saving checkpoint asynchronously')
            # Check if the ThreadPool is busy
            if self.last_save_job_future is not None:
                if self.last_save_job_future.done():
                    self.last_save_job_future.result()
                    self.last_save_job_future = None
                else:
                    raise RuntimeError("Previous async checkpoint save is still running. Cannot save checkpoint.")

            assert self._thread_executor is not None
            if self._thread_executor._work_queue.qsize() >= self._thread_executor._max_workers:
                raise RuntimeError("ThreadPool is busy. Cannot save checkpoint.")

            assert self.shard_save_barrier is not None
            self.last_save_job_future = checkpoint.save_checkpoint_async(
                executor=self._thread_executor, # TODO: change to processpool (TODO_2: seems inaffective on tests)
                state=state,
                filename=filename_with_placeholders,
                weights_only=self.weights_only,
                save_lora_adapter_only=self.save_lora_adapter_only,
                after_save_func=functools.partial(
                    after_checkpoint_save,
                    self,
                    current_batch_value=current_batch_value,
                    is_stable_checkpoint=is_stable_checkpoint,
                ),
                barrier=self.shard_save_barrier,
            )
            log.debug("Continuing training...")

    def fit_end(self, state: State, logger: Logger) -> None:
        if self.async_save:
            log.info("Waiting for asynchronous checkpoint save to complete")
            self.wait_for_async_checkpoint()
            assert self._thread_executor is not None
            self._thread_executor.shutdown(wait=True)
            dist.barrier()

    def _rotate_checkpoints(self, sharding_enabled: bool = False):

        while len(self.saved_checkpoints) > self.num_checkpoints_to_keep:
            prefix_dir = None
            checkpoint = self.saved_checkpoints.pop(0)
            prefix_dir = str(Path(checkpoint).parent)
            if not sharding_enabled:
                os.remove(checkpoint)
            else:
                if dist.get_global_rank() == 0 or dist.get_fsdp_group_rank() == 0:
                    shutil.rmtree(prefix_dir)
    @staticmethod
    def _validate_save_intervals(save_interval: str, stable_save_interval: str):
        assert save_interval.endswith("ba"), "only batches supported for stable_save_intervals, save_interval must be of same unit"
        save_interval = Time.from_input(save_interval)
        stable_save_interval = Time.from_input(stable_save_interval)
        assert save_interval.unit == stable_save_interval.unit, "stable_save_interval and save_interval must be of same unit"
        assert stable_save_interval.value % save_interval.value == 0, "stable_save_interval must be divisible by save_interval"


def get_current_checkpoint_save(state: State) -> Optional[dict]:
    """Return the checkpoint save record for the active AFTER_CHECKPOINT_SAVE event."""
    for callback in state.callbacks:
        if isinstance(callback, CheckpointSaver):
            checkpoint_save = callback.get_current_checkpoint_save()
            if checkpoint_save is not None:
                return checkpoint_save
    return None


def _extract_batch_from_checkpoint_name(name: str) -> int:
    name = os.path.basename(name)  # make sure basename is passed
    res = re.findall(r'ep\d+-ba\d+-', name)
    if len(res) == 1:
        return int(re.findall(r'\d+', name)[1])
    raise ValueError("Name poorly formatted")

    
def _check_is_stable_checkpoint(checkpoint: str, stable_save_interval: Optional[Union[int, set]], fsdp_elastic_sharded_enabled: bool) -> bool:
    if stable_save_interval is None:
        return True
    if fsdp_elastic_sharded_enabled:
        checkpoint = os.path.dirname(checkpoint)
    batch_index = _extract_batch_from_checkpoint_name(os.path.basename(checkpoint))
    if isinstance(stable_save_interval, set):
        return batch_index in stable_save_interval
    return batch_index % stable_save_interval == 0


def _remove_sharded_checkpoint(path: os.PathLike, doesnt_exist_ok: bool = False) -> None:
    def _error_handler(func, path, exc_info):
        _, exception, _ = exc_info
        if not isinstance(exception, FileNotFoundError):
            raise exception
    onerror = _error_handler if doesnt_exist_ok else None
    shutil.rmtree(path, onerror=onerror)

    
def _all_last_checkpoints_saved(checkpoint_dir: os.PathLike, latest_rank_symlink_template: str) -> bool:
    """if all last_save symlinks point to checkpoints saved on the same batch
    all last checkpoints are considered to be saved
    """
    latest_batches = set()
    num_ranks = dist.get_tp_group_size() or 1  # handle tp_size = 1
    for rank in range(num_ranks):
        fname = latest_rank_symlink_template.format(rank=rank)
        ckpt_path = os.path.realpath(os.path.join(checkpoint_dir, fname))
        ckpt_dir = os.path.dirname(ckpt_path)
        latest_batches.add(_extract_batch_from_checkpoint_name(ckpt_dir))
    return len(latest_batches) == 1
