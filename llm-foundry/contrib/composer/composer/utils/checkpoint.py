# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for working with training checkpoints."""

from __future__ import annotations

import math
from packaging import version
import time
import contextlib
import tempfile
import fnmatch
import logging
import os
import shutil
import tarfile
import tempfile
import traceback
import textwrap
import warnings
from importlib import import_module
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)
import concurrent.futures
import zlib
import json
from collections import OrderedDict
import re
import uuid

import dill
import torch

from packaging import version

from composer.utils import dist, reproducibility
from composer.utils.iter_helpers import ensure_tuple
from composer.utils.file_helpers import (FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, format_name_with_dist,
                                         format_name_with_dist_and_time, get_file, is_tar)
from composer.utils.misc import get_torch_version
from composer.utils.object_store import ObjectStore
from composer.utils.tcp_barrier import TCPBarrier
from composer.utils.job_dispatcher import JD

if TYPE_CHECKING:
    from composer.core import AlgorithmPass, State
    from composer.loggers import Logger, LoggerDestination
import hashlib
import numpy as np

log = logging.getLogger(__name__)

_jd = JD()

__all__ = ['load_checkpoint', 'save_checkpoint', 'download_checkpoint']

_COMPOSER_STATES_FILENAME = 'composer_states.pt'


def compute_parameter_hash(tensor: torch.Tensor, verbosity=0) -> str:
    """
    Computes CRC32 hash for a single Tensor
    """
    start_time = time.perf_counter()
    tensor_cpu = tensor.detach().cpu()
    if verbosity >= 2:
        log.debug(f"Computing hash for tensor with shape: {tensor.shape}, dtype: {tensor.dtype}")

    if tensor_cpu.is_floating_point():
        tensor_np = tensor_cpu.to(torch.float32).numpy()
    else:
        tensor_np = tensor_cpu.numpy()

    checksum = zlib.crc32(tensor_np.tobytes())
    hash_value = f"{checksum:08x}"

    duration = time.perf_counter() - start_time
    size_mb = tensor_cpu.element_size() * tensor_cpu.nelement() / (1024 * 1024)
    if verbosity >= 2:
        log.debug(f"Hash computed in {duration:.4f}s for {size_mb:.2f}MB tensor. Hash: {hash_value}")

    return hash_value


def compute_state_dict_hash(state_dict: dict, verbosity=2) -> dict:
    """
    Computes hash for entire state dict
    Returns: a dict of hashes for each of Tensors in the state dict
    """
    total_start = time.perf_counter()
    total_size = 0
    hashes = {}

    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            start_time = time.perf_counter()
            hashes[name] = compute_parameter_hash(param, verbosity=verbosity)
            duration = time.perf_counter() - start_time
            size = param.element_size() * param.nelement()
            total_size += size
            if verbosity >= 2:
                log.debug(f"Parameter {name} hashed in {duration:.4f}s (Size: {size/(1024*1024):.2f}MB)")

        elif isinstance(param, dict):
            # HOTFIX: remove model meta from checkpoint validation as meta structure has been changed
            if isinstance(name, str) and name.endswith(".meta"):
                continue
            sub_start = time.perf_counter()
            hashes[name] = compute_state_dict_hash(param, verbosity=0)
            sub_duration = time.perf_counter() - sub_start
            if verbosity >= 2:
                log.debug(f"Sub-dictionary {name} hashed in {sub_duration:.4f}s")

    total_duration = time.perf_counter() - total_start
    if verbosity >= 1:
        log.info(f"State dict hashing completed in {total_duration:.2f}s. Total size: {total_size/(1024*1024):.2f}MB")


    return hashes


def compute_meta_infos_hash(meta_infos : Dict):
    meta = {}
    for key in meta_infos['meta_for_hash']:
        meta[key] = meta_infos[key]
    sha256 = hashlib.sha256()
    sha256.update(json.dumps(meta, sort_keys=True).encode('utf-8'))
    return sha256.hexdigest()


def add_hash_to_state_dict(state_dict: Dict[str, Any]) -> None:
    """
    Computes and adds CRC32 hashes for model parameters and optimizer states to the state dictionary's metadata.
    """
    if 'state' not in state_dict:
        log.warning("Cannot add hashes: 'state' key missing from state_dict.")
        return

    state_content = state_dict['state']

    if 'metadata' not in state_content:
        state_content['metadata'] = {}

    metadata = state_content['metadata']
    metadata['hashing_method'] = 'crc32'

    if 'model' in state_content:
        log.debug("Computing hashes for model parameters...")
        model_state = state_content['model']
        metadata['parameter_hashes'] = compute_state_dict_hash(model_state, verbosity=0)
        log.debug("Model parameter hashes added to metadata.")
    else:
        log.debug("No 'model' key found in state_dict['state'], skipping parameter hashing.")

    if 'optimizers' in state_content:
        optimizers_tuple = ensure_tuple(state_content['optimizers'])
        if isinstance(optimizers_tuple, (tuple, list)):
            log.debug(f"Computing hashes for {len(optimizers_tuple)} optimizer states...")
            all_optimizer_hashes = {}
            for idx, optimizer_state_dict in enumerate(optimizers_tuple):
                 log.debug(f"Computing hash for optimizer {idx}...")
                 all_optimizer_hashes[str(idx)] = compute_state_dict_hash(optimizer_state_dict, verbosity=0)
            metadata['optimizers_hashes'] = all_optimizer_hashes
            log.debug("Optimizer state hashes (indexed by position) added to metadata.")
        else:
            log.warning(f"Expected 'optimizers' to be a tuple/list, but got {type(optimizers_tuple)}. Skipping optimizer hashing.")
    else:
        log.debug("No 'optimizers' key found in state_dict['state'], skipping optimizer hashing.")


def validate_hashes(
    loaded_hashes: Optional[Dict[str, Any]],
    current_state_part: Dict[str, Any],
    strict_check: bool,
    state_part_name: str, # e.g., "Model Parameters" or "Optimizers"
    hashing_method: str = 'crc32'
) -> None:
    """
    Validates precomputed hashes stored in a checkpoint against newly computed hashes.
    """
    if loaded_hashes is None:
        log.debug(f"No precomputed hashes found for {state_part_name}. Skipping validation.")
        return

    log.info(f"Validating {state_part_name} hashes...")

    # we do not support other methods beside CRC32 right now, but may be important in the future
    # loaded_method = state_dict.get('state', {}).get('metadata', {}).get('hashing_method', 'unknown').lower()
    # assert loaded_method == hashing_method, f"Checkpoint hashing method '{loaded_method}' differs from expected '{hashing_method}'. Comparison is impossible."

    current_hashes = compute_state_dict_hash(current_state_part, verbosity=0)
    mismatches_found = False

    for name, loaded_hash in loaded_hashes.items():
        if name in current_hashes:
            if loaded_hash != current_hashes[name]:
                log.warning(f"{state_part_name} hash mismatch for '{name}'. Loaded: {loaded_hash}, Current: {current_hashes[name]}")
                mismatches_found = True
        else:
            log.warning(f"{state_part_name} '{name}' exists in checkpoint hashes but not in current state.")

    for name in current_hashes:
        if name not in loaded_hashes:
            log.warning(f"{state_part_name} '{name}' exists in current state but not in checkpoint hashes.")

    if mismatches_found and strict_check:
        raise RuntimeError(f"Hash mismatch detected for {state_part_name} and strict_check=True.")
    elif mismatches_found:
         log.warning(f"Hash mismatches found for {state_part_name}, but strict_check=False.")
    else:
        log.info(f"{state_part_name} hash validation passed.")


def _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME():
    if dist.get_fsdp_group_rank() is None:
        return f'__{dist.get_global_rank()}_0.distcp'
    return f'__{dist.get_fsdp_group_rank()}_0.distcp'


def _DistWrapper_custom_init(
    self,
    group: Optional[dist.ProcessGroup],
    use_dist: bool,
    coordinator_rank: int,
):
    self.group = group
    self.use_dist = use_dist

    # global rank
    self.coordinator_rank = coordinator_rank
    if self.use_dist:
        # local rank
        self.rank = dist.dist.get_rank(group)
        self.is_coordinator = self.rank == 0
    else:
        self.rank = 0
        self.is_coordinator = True


def _get_checkpoint_validation_function() -> Optional[Callable[[Union[Path, str]], bool]]:
    """Get the validation function by name.

    Args:
        name (str): Qualified name of the checkpoint validation function.
                    It should be in the form '{module_name}.{fn_name}'.

    Returns:
        Callable[[Union[Path, str]], bool] The checkpoint validation function that returns
            True given a valid checkpoint and False otherwise.
    """
    name = os.environ.get('CHECKPOINT_VALIDATION_FUNCTION', None)
    if name is None:
        return None
    splits = name.split('.')
    module_name, fn_name = '.'.join(splits[:-1]), splits[-1]
    module = import_module(module_name)
    fn = getattr(module, fn_name)
    log.debug(f'Checkpoint validation function {name} has been found.')
    return fn


def _ensure_valid_checkpoint(checkpoint_filepath: Union[Path, str]) -> Union[Path, str]:
    """Ensures that the checkpoint at checkpoint_filepath is valid.

    using the function specified by the CHECKPOINT_VALIDATION_FUNCTION environment variable.
    If CHECKPOINT_VALIDATION_FUNCTION is not set, we skip validation.

    Args:
        checkpoint_filepath (Union[Path,str]): The path to the checkpoint file.

    Raises:
        ValueError if checkpoint file is invalid.
    """
    # Get the validation function by name.
    validate = _get_checkpoint_validation_function()

    # No function name has been specified.
    if validate is None:
        # log.debug('No validation function specified. Skipping checkpoint validation.')
        return checkpoint_filepath
    # Validate the checkpoint.
    if not validate(checkpoint_filepath):
        raise ValueError(f'Checkpoint at {checkpoint_filepath} is invalid.')

    log.debug(f'Checkpoint at {checkpoint_filepath} is valid.')
    return checkpoint_filepath


def _torch_load_with_validation(checkpoint_filepath: Union[Path, str], map_location: str) -> Any:
    """Validates and loads a torch checkpoint.

    Args:
        checkpoint_filepath (Union[Path,str]): The path to the checkpoint file.
        map_location (str): The location to load the checkpoint to.
    """
    return torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location=map_location, pickle_module=dill)


def _format_path_with_rank_zero(path: str) -> str:
    """Formats ``path`` with the rank zero values."""
    return path.format(
        rank=0,
        local_rank=0,
        node_rank=0,
    )


def _format_path_with_current_rank(path: str) -> str:
    """Formats ``path`` formatted with the current rank values."""
    return path.format(
        rank=dist.get_global_rank(),
        local_rank=dist.get_local_rank(),
        node_rank=dist.get_node_rank(),
    )


def _get_write_mode(name: str) -> str:
    """Get the write mode to use with :func:`tarfile.open`."""
    if name.endswith('.tar'):
        return 'w'
    if name.endswith('.tar.gz') or name.endswith('.tgz'):
        return 'w:gz'
    if name.endswith('.tar.bz2'):
        return 'w:bz2'
    if name.endswith('.tar.lzma'):
        return 'w:xz'
    raise ValueError(f'{name} does not end with a valid tarfile extension.')


class PartialFilePath:

    def __init__(self, filename: str, folder: Optional[str] = None):
        self.folder = folder
        self.filename = filename

    def format(self, state: State, keep_placeholders: bool = False) -> str:
        if self.folder:
            if keep_placeholders:
                return os.path.join(
                    self.folder,
                    self.filename,
                )
            else:
                return os.path.join(
                    format_name_with_dist(self.folder, state.run_name),
                    format_name_with_dist_and_time(self.filename, state.run_name, state.timestamp),
                )
        else:
            if keep_placeholders:
                return self.filename
            else:
                return format_name_with_dist_and_time(
                    self.filename,
                    state.run_name,
                    state.timestamp,
                )


def is_checkpoint_legacy_sharded(object_store: Optional[ObjectStore], source_path: str):
    metadata_path = str(Path(source_path) / Path('.metadata'))
    if object_store is None:
        return not os.path.exists(metadata_path)
    else:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                metadata_destination = os.path.join(str(temp_dir), '.metadata')
                object_store.download_object(object_name=metadata_path, filename=metadata_destination)
            return False
        except FileNotFoundError:
            return True


def load_checkpoint(
    path: str,
    state: State,
    logger: Logger,
    object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
    load_weights_only: bool = False,
    strict_model_weights: bool = False,
    progress_bar: bool = True,
    ignore_keys: Optional[Union[list[str], Callable[[dict], None]]] = None,
    exclude_algorithms: Optional[list[str]] = None,
    algorithm_passes: Optional[list[AlgorithmPass]] = None,
    teacher_path: Optional[str] = None,
):
    """Load a checkpoint from a local file, URI, or cloud object store into ``state``.

    Args:
        path (str): The path format string to an existing checkpoint file.

            It can be a path to a file on the local disk, a URL, or if ``object_store`` is set, the object name
            for a checkpoint in a cloud bucket.

        state (State): The :class:`~composer.core.State` to load the checkpoint into.
        logger (Logger): The :class:`~composer.logger.Logger` to log any information.
        object_store (Union[ObjectStore, LoggerDestination], optional): If the ``path`` is in an object store
            (i.e. AWS S3 or Google Cloud Storage), an instance of
            :class:`~.ObjectStore` or :class:`~.LoggerDestination` which will be used
            to retrieve the checkpoint. Otherwise, if the checkpoint is a local filepath, set to ``None``.
            (default: ``None``)
        load_weights_only (bool, optional): Whether or not to only restore the model weights from the checkpoint without
            restoring the associated state. (default: ``False``)
        strict_model_weights (bool, optional): Whether or not to force that the checkpointed weights must exactly
            match the model weights. (default: ``False``)
        progress_bar (bool, optional): Whether or not to show a progress bar when downloading checkpoints.
            Ignored if the checkpoint is a local file path. (default: ``True``)
        ignore_keys (list[str] | (dict) -> None, optional): A list of paths for the ``state_dict`` of the checkpoint,
            which, when provided, will be ignored from the state_dict before a checkpoint is loaded. Each path is a list
            of strings specifying the keys to index into ``state_dict`` joined together with `/` as a separator (as PyTorch
            uses `.` in parameter names). If a prefix is provided, all children are also ignored (see Example 2).
            See :mod:`composer.core.state` for the structure of state_dict.

            Example 1: ``ignore_keys = ["state/model/layer1.weights", "state/model/layer1.bias"]`` would ignore
            layer 1 weights and bias.

            Example 2: ``ignore_keys = ["state/model/*"]`` would ignore the entire model, which would have the same
            effect as the previous example if there was only 1 layer.

            Example 3: ``ignore_keys = ["state/model/layer*.weights"]`` would ignore all weights in the model.

            Example 4: ``ignore_keys = ["state/rank_zero_seed", "rng"]`` would reset all randomness when
            loading the checkpoint.

            If a callable, it should take one argument which is the state_dict. The callable is free to arbitrarily modify
            the state_dict before it is loaded.

            (default: ``None``)
        exclude_algorithms (list[str], optional): A list of algorithm names to exclude from loading.
            By default, algorithms with `required_on_load=True` which were enabled when training the loaded
            checkpoint are automatically applied unless they conflict with a user specified algorithm. These
            algorithms often change the model, and not applying them could result in certain layers not having
            weights loaded.

            Example 1: ``exclude_algorithms = ["BlurPool"]`` would exclude BlurPool from loading.

            Example 2: ``exclude_algorithms = ["FusedLayerNorm", "Alibi"]`` would exclude FusedLayerNorm and Alibi from loading.

            (default: ``None``)
        algorithm_passes (list[AlgorithmPass], optional): A list of algorithm passes to apply to autoloaded algorithms
            to sort them into the correct order. (default: ``None``)

    Returns:
        Optional[list[dict[str, Any]]]: The RNG state dicts, indexed by global rank, if
            :attr:`load_weights_only` is not None. Otherwise, None.
    """
    using_legacy_sharded = False
    if state.fsdp_elastic_sharded_enabled:
        assert object_store is None or isinstance(object_store, ObjectStore), (
            'For loading sharded checkpoints load_object_store must be set with the class ObjectStore'
        )
        using_legacy_sharded = is_checkpoint_legacy_sharded(object_store, path)

    def gigafsdp_shard_fn(path : str,
                          meta_infos : List[Tuple[str, Dict]],
                          gigafsdp_state_dict: Dict,
                          gigafsdp_model_normalize_fqns: Optional[Any] = None,) -> str:
        fsdp_group_size = meta_infos[0][1]["fsdp_group_size"]
        tp_size = dist.get_tp_group_size() or 1
        ep_size = dist.get_ep_group_size() or 1
        full_fsdp_group_size = fsdp_group_size * tp_size
        fsdp_group_size = full_fsdp_group_size // max(tp_size, ep_size)

        meta_infos_hash = compute_meta_infos_hash(meta_infos[0][1])

        if not os.path.isfile(path): #not monolith checkpoint
            num_shards = len(os.listdir(path))
            if num_shards != fsdp_group_size:
                raise RuntimeError(
                    f"GigaFSDP checkpoint folder contains {num_shards} shards, but model must be sharded on {fsdp_group_size} parts." +
                    f" Please, reshard checkpoint or set `hybrid_group_size={num_shards}`" +
                    f" (only if {num_shards} < {fsdp_group_size} and {fsdp_group_size} % {num_shards} == 0)"
                )
            return path

        #TP=1 case: sharding processes run on each local 0 rank
        #TP>1 case: sharding processes run on every first local tp group
        shard_condition = dist.get_local_rank() == (dist.get_tp_group_rank() or 0) # 0 for TP=1 case
        shards_count_per_node = dist.get_local_world_size() // (dist.get_tp_group_size() or 1) # 1 for TP=1 case
        assert shards_count_per_node > 0, "TP size > number of GPUs per node. Use smaller tp_size."
        start_idx = dist.get_node_rank() * shards_count_per_node
        end_idx = start_idx + shards_count_per_node

        #for hybrid shard only first hybrid group will shard and write checkpoint
        if start_idx >= fsdp_group_size:
            shard_ids_per_node = None
            shard_condition = False
        else:
            shard_ids_per_node = range(fsdp_group_size)[start_idx : end_idx]

        msg = f"GPU {dist.get_local_rank()} / {dist.get_global_rank()} : shard_condition: {shard_condition},"
        msg += f" shard_ids_per_node: {shard_ids_per_node}."
        msg += f" Info: get_tp_sp_group_rank: {dist.get_tp_sp_group_rank()}, get_tp_group_rank: {dist.get_tp_group_rank()}"
        log.debug(msg)

        if state.fsdp_config['save_gigafsdp_sharded_folder_path']:
            sharded_path = path.strip(".pt").replace("/", "-").strip("-") + f"_{meta_infos_hash}_gigafsdp_{fsdp_group_size}_sharded"
            sharded_path = os.path.join(state.fsdp_config['save_gigafsdp_sharded_folder_path'], sharded_path)
        else:
            sharded_path = path.strip(".pt") + f"_{meta_infos_hash}_gigafsdp_{fsdp_group_size}_sharded"

        if shard_condition and dist.get_node_rank() == 0:
            os.makedirs(sharded_path, exist_ok=True)

        # wait until master rank create a folder
        dist.barrier()

        if os.path.exists(sharded_path) and len(os.listdir(sharded_path)) == fsdp_group_size:
            log.debug(f"Monolith ckpt {path} already sharded at {sharded_path}. Use previuos sharding.")

        elif shard_condition:
            log.debug(f"Sharding monolith ckpt for gigafsdp: {path} -> {sharded_path}")
            add_hash_fn = add_hash_to_state_dict if getattr(state, 'compute_checkpoint_hashes', False) else None
            from giga_fsdp.utils import shard_checkpoint
            shard_checkpoint(
                checkpoint_path=path,
                num_output_rank=fsdp_group_size,
                meta_infos=meta_infos,
                output_path=sharded_path,
                weights_only=load_weights_only,
                ranks_to_write=shard_ids_per_node,
                load_strict=strict_model_weights,
                model_normalize_fqns=gigafsdp_model_normalize_fqns,
                add_hash_to_state_dict_fn=add_hash_fn,
                gigafsdp_state_dict=gigafsdp_state_dict,
                model_config=state.model.config,
            )
            log.debug("Sharding monolith ckpt is complete!")

        dist.barrier()

        #sanity check - all nodes have written their shards
        shard_count = len(os.listdir(sharded_path))
        assert shard_count == fsdp_group_size, \
            f"Not enough shards in folder: need {fsdp_group_size}, but there are {shard_count}"

        return sharded_path

    def check_is_safetensors_format(path):
        p = Path(path)
        return (p.exists() and p.is_dir() and (p / "config.json").exists())

    is_safetensors_checkpoint: bool = check_is_safetensors_format(path)
    if state.gigafsdp_enabled:
        from giga_fsdp import GigaFSDP
        
        safetensors_mono_sd = None
        safetensors_teacher_mono_sd = None

        gigafsdp_model_normalize_fqns = (state.model.gigafsdp_model_normalize_fqns
                                         if hasattr(state.model, 'gigafsdp_model_normalize_fqns') else None)

        if is_safetensors_checkpoint:
            from giga_fsdp.utils_checkpoint import shard_checkpoint_experimental
            meta_infos_model = [(module_name, m._meta_info) for module_name, m in state.model.named_children()
                                if module_name == "model" and isinstance(m, GigaFSDP)]

            assert meta_infos_model[0][0] == "model", "supposed that meta_infos[0] is a meta about main model"
            model_name = meta_infos_model[0][0]
            meta_infos_val = meta_infos_model[0][1]

            local_gigafsdp_model_state_dict = state.model.model.local_state_dict()
            safetensors_mono_sd = shard_checkpoint_experimental(
                checkpoint_path=path,
                meta_infos=meta_infos_val,
                model_name=model_name,
                local_gigafsdp_model_state_dict=local_gigafsdp_model_state_dict,
                load_strict=strict_model_weights,

                gigafsdp_model_normalize_fqns=gigafsdp_model_normalize_fqns,
                model_config=state.model.config)

            safetensors_mono_sd['state']['model']['model.meta'] = meta_infos_val
            safetensors_mono_sd['state']['model']['is_safetensors'] = True
            safetensors_mono_sd = ({"state": {"model": safetensors_mono_sd['state']['model']}}
                                   if load_weights_only else safetensors_mono_sd)
            assert "optimizers" not in safetensors_mono_sd['state'].keys()
        else:
            meta_infos_model = [(module_name, m._meta_info) for module_name, m in state.model.named_children()
                                if module_name == "model" and isinstance(m, GigaFSDP)]
            gigafsdp_state_dict = _get_state(state,
                                             weights_only=False,
                                             state_dict_type="local",
                                             device="cpu")

            path = gigafsdp_shard_fn(path=path,
                                     meta_infos=meta_infos_model,
                                     gigafsdp_state_dict=gigafsdp_state_dict,
                                     gigafsdp_model_normalize_fqns=gigafsdp_model_normalize_fqns)

        if teacher_path:
            is_teacher_safetensors = check_is_safetensors_format(teacher_path)
            if is_teacher_safetensors:
                from giga_fsdp.utils_checkpoint import shard_checkpoint_experimental
                meta_infos_teacher = [(module_name, m._meta_info) for module_name, m in state.model.named_children()
                                      if module_name != "model" and isinstance(m, GigaFSDP)]

                assert meta_infos_teacher[0][0] == "teacher", "supposed that meta_infos[0] is a meta about teacher"
                model_name_teacher = meta_infos_teacher[0][0]
                meta_infos_teacher_val = meta_infos_teacher[0][1]

                local_gigafsdp_model_state_dict_teacher = state.model.teacher.local_state_dict()
                safetensors_teacher_mono_sd = shard_checkpoint_experimental(
                    checkpoint_path=teacher_path,
                    meta_infos=meta_infos_teacher_val,
                    model_name=model_name_teacher,
                    local_gigafsdp_model_state_dict=local_gigafsdp_model_state_dict_teacher,
                    load_strict=strict_model_weights,
                    gigafsdp_model_normalize_fqns=gigafsdp_model_normalize_fqns,
                    model_config=state.model.teacher_config)
                safetensors_teacher_mono_sd['state']['model'][f'{model_name_teacher}.meta'] = meta_infos_teacher_val
                safetensors_teacher_mono_sd['state']['model']['is_safetensors'] = True
                assert "optimizers" not in safetensors_teacher_mono_sd['state'].keys()
                teacher_path = None # Loaded as safetensors, skip FSDP logic
            else:
                meta_infos_teacher_fsdp = [("model", state.model.teacher._meta_info)]
                tmp_techer_sd = state.model.teacher.local_state_dict()
                tmp_techer_sd = {f"model.{k}": v for k, v in tmp_techer_sd.items()}
                gigafsdp_state_dict['state']['model'] = tmp_techer_sd
                teacher_path = gigafsdp_shard_fn(
                    teacher_path,
                    meta_infos_teacher_fsdp,
                    gigafsdp_state_dict=gigafsdp_state_dict,
                    gigafsdp_model_normalize_fqns=gigafsdp_model_normalize_fqns,)

    if state.fsdp_elastic_sharded_enabled and not using_legacy_sharded:
        rng_state_dicts = load_sharded_checkpoint(
            source_path=path,
            state=state,
            logger=logger,
            object_store=object_store,
            load_weights_only=load_weights_only,
            strict_model_weights=strict_model_weights,
            progress_bar=progress_bar,
            ignore_keys=ignore_keys,
            exclude_algorithms=exclude_algorithms,
            algorithm_passes=algorithm_passes,
        )

    else:
        # Download the checkpoint to the node-local folder
        if teacher_path:
            log.debug('Loading checkpoints for student: %s and teacher: %s', path, teacher_path)
        else:
            log.debug('Loading checkpoint at %s', path)

        # Each node gets one unique folder to store checkpoints that is shared amongst all local ranks in that node.
        # If fsdp sharded state_dicts is enabled then EVERY rank gets a unique checkpoint folder.
        needs_unique_checkpoint_folder = state.fsdp_sharded_state_dict_enabled or dist.get_local_rank() == 0
        tempdir_ctx = tempfile.TemporaryDirectory() if needs_unique_checkpoint_folder else contextlib.nullcontext(None)

        with tempdir_ctx as tempdir:
            try:
                if not state.gigafsdp_enabled:
                    def _make_path(path: str) -> str:
                        os.makedirs(path, exist_ok=True)
                        return path

                    # Get the path to the proper checkpoint folder corresponding to the current rank's node.
                    # If fsdp_sharded_state_dict_enabled then just use that rank's unique tempdir.
                    node_checkpoint_folder = (tempdir if state.fsdp_sharded_state_dict_enabled else
                                            _get_local_rank_zero_path(tempdir))
                    assert node_checkpoint_folder is not None
                    composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n = download_checkpoint(
                        path=path,
                        node_checkpoint_folder=node_checkpoint_folder,
                        object_store=object_store,
                        progress_bar=progress_bar,
                        fsdp_sharded_state_dict_enabled=state.fsdp_sharded_state_dict_enabled,
                    )

                    composer_states_filepath = path

                    if teacher_path:
                        log.debug("Download teacher")
                        composer_teacher_states_filepath = teacher_path
                        composer_teacher_states_filepath, _, _ = download_checkpoint(
                            path=teacher_path,
                            node_checkpoint_folder=_make_path(os.path.join(node_checkpoint_folder, "teacher")),
                            object_store=object_store,
                            progress_bar=progress_bar,
                            fsdp_sharded_state_dict_enabled=state.fsdp_sharded_state_dict_enabled,
                        )
                    else:
                        composer_teacher_states_filepath = None

                elif not is_safetensors_checkpoint:
                    def _GIGAFSDP_CHECKPOINTS_FILENAME():
                        num_shards = next(state.model.children())._meta_info['fsdp_group_size']
                        if dist.get_fsdp_group_rank() is None:
                            return f'__{dist.get_global_rank() % num_shards}_0.distcp'
                        return f'__{dist.get_fsdp_group_rank() % num_shards}_0.distcp'

                    composer_states_filepath = path
                    composer_states_filepath = os.path.join(composer_states_filepath,
                                                            _GIGAFSDP_CHECKPOINTS_FILENAME())
                    extracted_checkpoint_folder = None
                    extracted_rank_n = False
                    if teacher_path:
                        composer_teacher_states_filepath = os.path.join(teacher_path, _GIGAFSDP_CHECKPOINTS_FILENAME())
                    else:
                        composer_teacher_states_filepath = None

                else:
                    # state.gigafsdp_enabled and is_safetensors_checkpoint
                    composer_states_filepath = None
                    extracted_rank_n = None
                    extracted_checkpoint_folder = None
                    composer_teacher_states_filepath = None

                rng_state_dicts = _restore_checkpoint(
                    state,
                    logger,
                    composer_states_filepath,
                    extracted_rank_n,
                    extracted_checkpoint_folder,
                    load_weights_only=load_weights_only,
                    strict_model_weights=strict_model_weights,
                    ignore_keys=ignore_keys,
                    exclude_algorithms=exclude_algorithms,
                    algorithm_passes=algorithm_passes,
                    composer_teacher_states_filepath=composer_teacher_states_filepath,
                    safetensors_mono_sd=safetensors_mono_sd,
                    safetensors_teacher_mono_sd=safetensors_teacher_mono_sd,
                )

            finally:
                # Wait for all ranks to finish restoring the checkpoint before releasing the tempdir, since tempdir can
                # be a shared resource between nodes.
                dist.barrier()

        log.info('%s loaded from %s', 'Model weights' if load_weights_only else 'Trainer checkpoint', path)

    step_to_resume_from = state.timestamp.batch.value
    max_step_to_resume_from = state.device.tensor_to_device(torch.tensor(state.timestamp.batch.value,
                                                                         dtype=torch.int64))
    min_step_to_resume_from = state.device.tensor_to_device(torch.tensor(state.timestamp.batch.value,
                                                                         dtype=torch.int64))
    dist.all_reduce(max_step_to_resume_from, reduce_operation='MAX')
    dist.all_reduce(min_step_to_resume_from, reduce_operation='MIN')
    if max_step_to_resume_from.data != min_step_to_resume_from.data:
        raise RuntimeError(
            textwrap.dedent(
                f'Timestamp mismatch error: batch to resume from {step_to_resume_from} is not the same on all ranks. ' +
                'This usually occurs when at least one rank fails to save the last checkpoint ' +
                'while using sharded checkpointing + autoresume. ' +
                'Please manually resume by disabling autoresume and explicitly setting load_path ' +
                'to the most recent checkpoints that all ranks have saved. ' +
                'E.g. for the 10th batch: trainer = Trainer(autoresume=False, load_path="/path/to/checkpoint/ba10-rank{rank}.pt", ...). ' +
                'Remember to keep the {rank} placeholder!'))
    return rng_state_dicts


def load_sharded_checkpoint(
    source_path: str,
    state: State,
    logger: Logger,
    object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
    load_weights_only: bool = False,
    strict_model_weights: bool = False,
    progress_bar: bool = True,
    ignore_keys: Optional[Union[list[str], Callable[[dict], None]]] = None,
    exclude_algorithms: Optional[list[str]] = None,
    algorithm_passes: Optional[list[AlgorithmPass]] = None,
) -> list[dict]:

    using_multinode = dist.get_world_size() != dist.get_local_world_size()

    if state.fsdp_config is None:
        raise ValueError('Loading a sharded checkpoint requires passing an FSDP config to Trainer.')
    load_planner = state.fsdp_config['load_planner']
    _validate_load_planner(load_planner)

    from torch.distributed import checkpoint as dist_cp
    from torch.distributed.checkpoint.metadata import Metadata
    from torch.distributed.checkpoint.optimizer import \
        load_sharded_optimizer_state_dict
    from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner

    # This function is used so we can figure out which ranks need to load saved rngs and which can just make their own.
    def _get_num_ranks_that_saved_rng(metadata: Metadata):
        rng_inds = []
        for field_name, field_value in metadata.planner_data.items():
            if 'rng' in field_name:
                _, rng_rank_index, _ = field_value
                rng_inds.append(rng_rank_index)
        rng_inds = set(rng_inds)
        return len(rng_inds)

    class FileSystemReaderWithValidation(dist_cp.FileSystemReader):
        """FileSystemReader that validates checkpoint files prior to reading."""

        def __init__(self, path: str):
            if _get_checkpoint_validation_function() is None:
                log.info('No checkpoint validation function found when loading sharded checkpoints.')
            super().__init__(path)

        def read_data(self, plan: LoadPlan, planner: LoadPlanner):
            """Reads data file.

            Raises:
                ValueError if the data file is invalid.
            """
            validated_checkpoint_paths = set()
            for read_item in plan.items:
                data_path = self.path / self.storage_data[read_item.storage_index].relative_path
                if data_path in validated_checkpoint_paths:
                    continue
                _ensure_valid_checkpoint(data_path)
                validated_checkpoint_paths.add(data_path)
            return super().read_data(plan, planner)

        def read_metadata(self) -> Metadata:
            """Reads metadata file.

            Raises:
                ValueError if the metadata file is invalid.
            """
            metadata_file_path = self.path / '.metadata'
            _ensure_valid_checkpoint(metadata_file_path)
            return super().read_metadata()

    # A subclass of FileSystemReaderWithValidation that downloads files from the object store before reading them from the local filesystem.
    class DistCPObjectStoreReader(FileSystemReaderWithValidation):

        def __init__(self, source_path: str, destination_path: str, object_store):
            self.source_path = source_path
            self.destination_path = destination_path
            self.object_store = object_store

            # Download metadata file.
            Path(self.destination_path).mkdir(parents=True, exist_ok=True)
            metadata_destination = os.path.join(self.destination_path, '.metadata')
            if dist.get_local_rank() == 0:
                object_store.download_object(object_name=str(Path(source_path) / Path('.metadata')),
                                             filename=metadata_destination)
            dist.barrier()

            # FileSystemReader takes in a root directory in its constructor, which is the dir where
            # the metadata is expected to be stored. Also, this is parent directory for any shard file relative paths
            # specified in the metadata file.
            super().__init__(destination_path)

        def read_data(self, plan: LoadPlan, planner: LoadPlanner):
            # 1. Download to the destination all files that this rank is responsible for.
            for plan_item in plan.items:
                # Each plan item has a storage index which points to the relative path of the shard file at save time.
                relative_file_path = self.storage_data[plan_item.storage_index].relative_path
                # Download the shard file to the relative path it's associated to and save that relative path
                # to the root directory specified to the FileSystem reader constructor.
                file_destination = str(Path(self.destination_path) / Path(relative_file_path))
                # The file could have already been downloaded as diffeent plan items can point to same file.
                if not os.path.exists(file_destination):
                    self.object_store.download_object(object_name=str(
                        Path(self.source_path) / Path(relative_file_path)),
                                                      filename=file_destination)

            # 2. Wait for all ranks to finish.
            dist.barrier()

            # 3. Piggyback off of the FileSystemReader to read all the files now that they are downloaded.
            return super().read_data(plan, planner)

    # Check to make sure source_path is a directory.
    if object_store is None:
        if os.path.islink(source_path):
            source_path = os.path.join(os.path.dirname(source_path), os.readlink(source_path))
        if os.path.exists(source_path):
            if not os.path.isdir(source_path):
                raise ValueError(f'load_path must be a directory when using sharded state dict. Got {source_path}')
        else:
            raise FileNotFoundError(f'{source_path} not found!')

    download_dir_context = tempfile.TemporaryDirectory if object_store is not None else contextlib.nullcontext

    with download_dir_context() as temp_download_dir:
        if object_store is not None:
            # Get the tempfile made on local rank 0.
            local_rank0_index = dist.get_global_rank() - dist.get_local_rank()
            rank0_download_tempdir = str(dist.all_gather_object(temp_download_dir)[local_rank0_index])
            storage_reader = DistCPObjectStoreReader(source_path=source_path,
                                                     destination_path=str(
                                                         Path(rank0_download_tempdir) / Path('checkpoints')),
                                                     object_store=object_store)
        else:
            storage_reader = FileSystemReaderWithValidation(source_path)

        # We need no_grad because we overwrite tensor values with set_() when we do elastic loading and we don't want the set_ op recorded in the computation graph.
        with torch.no_grad():
            # 1. Load model and metadata first
            model_state_dict = None
            if load_weights_only:
                model_state_dict = {'state': {'model': state.get_model_state_dict()}}
            else:
                cur_state_dict = state.state_dict()
                cur_state_dict.pop('optimizers')
                model_state_dict = {'state': cur_state_dict}

            if ignore_keys:
                # Filter provided list of key paths
                if not callable(ignore_keys):
                    ignore_keys = glob_filter(ignore_keys)
                # Call function to modify state_dict
                ignore_keys(model_state_dict)

            from torch.distributed.checkpoint.state_dict_loader import _DistWrapper
            _DistWrapper.__init__ = _DistWrapper_custom_init
            fsdp_group = dist.get_fsdp_group()
            group_global_ranks = dist.get_group_global_ranks(fsdp_group)

            dist_cp.load_state_dict(
                state_dict=model_state_dict,
                storage_reader=storage_reader,
                planner=load_planner,
                process_group=fsdp_group,
                coordinator_rank=group_global_ranks[0]
            )

            should_validate_hashes = False
            if state.gigafsdp_enabled:
                should_validate_hashes = getattr(state, 'compute_checkpoint_hashes', False)

            loaded_meta = None
            if should_validate_hashes:
                 loaded_meta = model_state_dict.get('state', {}).get('metadata', {})
                 log.debug("compute_checkpoint_hashes is True. Will validate hashes after loading state.")

            state.load_state_dict(
                model_state_dict['state'],
                logger,
                strict=strict_model_weights,
                exclude_algorithms=exclude_algorithms,
                algorithm_passes=algorithm_passes,
            )

            if should_validate_hashes and loaded_meta is not None:
                log.info("Validating Optimizer hashes post-load...")
                loaded_all_optim_hashes = loaded_meta.get('optimizers_hashes')

                if isinstance(loaded_all_optim_hashes, dict) and \
                    hasattr(state, 'optimizers') and \
                    isinstance(state.optimizers, (tuple, list, dict)):

                    num_optimizers = len(state.optimizers)
                    log.debug(f"Found {num_optimizers} optimizers in state. Validating against loaded hashes.")

                    for idx, optimizer_obj in enumerate(ensure_tuple(state.optimizers)):
                        optimizer_index_str = str(idx)
                        loaded_hashes_for_this_optim = loaded_all_optim_hashes.get(optimizer_index_str)

                        if loaded_hashes_for_this_optim is None:
                            log.warning(f"No loaded hashes found for optimizer index {idx}. Skipping validation for this optimizer.")
                            continue

                        current_optim_state_after_load = optimizer_obj.state_dict()
                        validate_hashes(
                            loaded_hashes=loaded_hashes_for_this_optim,
                            current_state_part=current_optim_state_after_load,
                            strict_check=strict_model_weights,
                            state_part_name=f"Optimizer {idx}"
                        )
                elif not isinstance(loaded_all_optim_hashes, dict):
                        log.warning(f"Loaded optimizer hashes are not a dictionary (type: {type(loaded_all_optim_hashes)}). Skipping optimizer validation.")
                else:
                        log.warning("Skipping optimizer hash validation as state.optimizers is not a tuple/list or not found after load.")

        for optim_name in current_optim_hashes:
            if optim_name not in loaded_optim_hashes:
                log.warning(f"Optimizer {optim_name} exists in current state but not in checkpoint")

            # 2. Optionally load optimizer
            if not load_weights_only:
                optim_state = load_sharded_optimizer_state_dict(model_state_dict=state.state_dict()['model'],
                                                                optimizer_key='optimizers',
                                                                storage_reader=storage_reader)
                state.load_optim_state(optim_state)

        # 3. Optionally load RNG
        rng_state_dicts = reproducibility.get_rng_state()
        if not load_weights_only:
            # If we are resuming on more ranks than were used at save time we only want to load in rngs for those ranks
            num_ranks_that_saved_rng = _get_num_ranks_that_saved_rng(storage_reader.read_metadata())
            rng_state_dicts_load = {}
            rng_state_dicts_load['rng'] = rng_state_dicts[:num_ranks_that_saved_rng] if len(
                rng_state_dicts) > num_ranks_that_saved_rng else rng_state_dicts
            dist_cp.load_state_dict(
                state_dict=rng_state_dicts_load,
                storage_reader=storage_reader,
                planner=load_planner,
            )
            # We also want to append newly generated rng states for the ranks that don't have an rng state to load in
            # if we are resuming on more ranks than were used at save time.
            if len(rng_state_dicts) > num_ranks_that_saved_rng:
                rng_state_dicts_load['rng'].extend(rng_state_dicts[num_ranks_that_saved_rng:])
            rng_state_dicts = rng_state_dicts_load['rng']

    return rng_state_dicts


def _get_local_rank_zero_path(path: Optional[str]) -> str:
    """Broadcasts the ``path`` from the LOCAL rank zero to all LOCAL ranks."""
    local_rank_zero = dist.get_global_rank() - dist.get_local_rank()
    paths = dist.all_gather_object(path)
    local_rank_zero_path = paths[local_rank_zero]
    assert local_rank_zero_path is not None, 'local rank zero provides the path'
    return local_rank_zero_path


def download_checkpoint(path: str,
                        node_checkpoint_folder: str,
                        object_store: Optional[Union[ObjectStore, LoggerDestination]],
                        progress_bar: bool,
                        fsdp_sharded_state_dict_enabled: bool = False,
                        ) -> tuple[str, Optional[str], bool]:
    """Download the checkpoint stored at ``path``, potentially in ``object_store``, to ``node_checkpoint_folder``.

    Returns a tuple of  (``composer_states_filepath``, ``extracted_checkpoint_folder``, ``extracted_rank_n``).

    *   The ``composer_states_filepath``, is the path to the composer states, which can be passed into
        :meth:`torch.load`.
    *   The ``extracted_checkpoint_folder`` is the path to the checkpoint folder, which can be passed into
        :meth:`deepspeed.DeepSpeedEngine.load_checkpoint`.
    *   The ``extracted_rank_n`` is a boolean flag indicating whether a tarball was extracted on global
        rank greater than 0.
    """
    log.debug("Downloading checkpoint to folder %s", node_checkpoint_folder)
    rank_zero_checkpoint_filepath = os.path.join(node_checkpoint_folder, "rank0_checkpoint")
    rank_n_checkpoint_filepath = os.path.join(node_checkpoint_folder, f"rank{dist.get_global_rank()}_checkpoint")
    rank_tp_checkpoint_filepath = os.path.join(node_checkpoint_folder, f"rank{dist.get_tp_group_rank()}_checkpoint")
    extracted_checkpoint_folder = None
    extracted_rank_n = False

    tp_checkpoint_enabled = dist.get_tp_group() is not None
    if is_tar(path):
        extracted_checkpoint_folder = os.path.join(node_checkpoint_folder, 'checkpoint')
        composer_states_filepath = os.path.join(extracted_checkpoint_folder, _COMPOSER_STATES_FILENAME)
    else:
        # it's not an archive; it's just the composer state dict
        # and only rank zero has this file unless fsdp_sharded_state_dict_enabled then
        # every rank has it's own file.
        extracted_checkpoint_folder = None
        if tp_checkpoint_enabled:
            composer_states_filepath = rank_tp_checkpoint_filepath
        else:
            composer_states_filepath = (
                rank_n_checkpoint_filepath if fsdp_sharded_state_dict_enabled else rank_zero_checkpoint_filepath
            )

    checkpoint_is_sharded = fsdp_sharded_state_dict_enabled
    try:
        if not checkpoint_is_sharded and (
            (not tp_checkpoint_enabled and dist.get_local_rank() == 0) or
            (tp_checkpoint_enabled and dist.get_fsdp_group_rank() == 0)
        ):
            # if the checkpoint is not sharded, then local rank 0 on each node needs to download the
            # global rank 0 checkpoint
            path = _format_path_with_rank_zero(path)
            if not tp_checkpoint_enabled and dist.get_local_rank() == 0:
                destination = rank_zero_checkpoint_filepath
            elif tp_checkpoint_enabled and dist.get_fsdp_group_rank() == 0:
                destination = rank_tp_checkpoint_filepath
            else:
                raise ValueError()
            get_file(destination=destination,
                     path=path,
                     object_store=object_store,
                     progress_bar=progress_bar)
            if extracted_checkpoint_folder is not None:
                try:
                    with tarfile.open(destination) as tarball:
                        tarball.extractall(extracted_checkpoint_folder)
                except FileNotFoundError:
                    # Not re-raising the file-not-found error as that is irrelevant;
                    # the underlying issue is that the checkpoint file does not exist on the disk
                    # or could not be downloaded
                    raise RuntimeError(f"Checkpoint {path} does not exist")
        elif checkpoint_is_sharded:
            # if the checkpoint is sharded, then every rank needs to download its own checkpoint
            try:
                get_file(destination=rank_n_checkpoint_filepath,
                         path=_format_path_with_current_rank(path),
                         object_store=object_store,
                         progress_bar=progress_bar)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    (f'Checkpoint {_format_path_with_current_rank(path)} does not exist, '
                     f'but is required for sharded checkpointing on rank {dist.get_global_rank()}. '
                     'Please ensure that the checkpoint exists and your load_path was specified as a format string'
                     'with the {rank} argument.')) from e

            if extracted_checkpoint_folder is not None:
                try:
                    # it's an archive and needs to be extracted
                    with tarfile.open(rank_n_checkpoint_filepath) as tarball:
                        tarball.extractall(extracted_checkpoint_folder)
                        extracted_rank_n = True
                except FileNotFoundError:
                    # this will happen most of the time (i.e. whenever deepspeed
                    # is not being used) so not logging anything
                    pass

    finally:
        # Use busy wait to avoid timeouts on large downloads for non-sharded checkpoints
        if not checkpoint_is_sharded:
            signal_file_path = os.path.join(node_checkpoint_folder,
                                            f'.node_{dist.get_node_rank()}_local_rank0_completed')
            if dist.get_local_rank() == 0:
                with open(signal_file_path, 'wb') as f:
                    f.write(b'local_rank0_completed')

            # Avoid the collective call until the local rank zero has finished trying to download the
            # checkpoint so that we don't timeout for large downloads. This syncs all processes on the
            # node
            with dist.local_rank_zero_download_and_wait(signal_file_path):
                # Then, wait to ensure every node has finished downloading the checkpoint
                dist.barrier()

            if dist.get_local_rank() == 0:
                os.remove(signal_file_path)
        dist.barrier()

    return composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n


def _flatten_keys(obj: Any, paths: list[str], existing_path: str):
    """Recursively flatten the keys of a dictionary or list into a set of paths."""
    # Store path when we reach end, which is either non-dict or empty dict
    if isinstance(obj, list) and len(obj) > 0:
        for i, elm in enumerate(obj):
            _flatten_keys(elm, paths, f'{existing_path}/{i}')
    elif isinstance(obj, dict) and len(obj) > 0:
        for k, v in obj.items():
            _flatten_keys(v, paths, f'{existing_path}/{k}')
    # Remove leading /
    paths.append(existing_path.lstrip('/'))


def _remove_paths(obj: Union[list, dict[str, Any]], exclude_paths: list[list[str]]):
    # First determine the keys which will be recursed on and which will be removed entirely
    # Group the `exclude_paths` by the key
    keys_to_recurse = {}
    keys_to_remove = []
    for exclude_path_parts in exclude_paths:
        key = exclude_path_parts[0]
        if isinstance(obj, list):
            key = int(key)
        if len(exclude_path_parts) == 1:
            keys_to_remove.append(key)
        else:
            if key not in keys_to_recurse:
                keys_to_recurse[key] = []
            keys_to_recurse[key].append(exclude_path_parts[1:])

    # Recurse first, so in the case of a list, the indexing is consistent
    for key, paths_to_recurse in keys_to_recurse.items():
        _remove_paths(obj[key], paths_to_recurse)

    # Sort the keys in reverse order, so in the case of a list, the indexing is consistent
    keys_to_remove.sort(reverse=True)

    # Remove the keys
    for key in keys_to_remove:
        del obj[key]


def glob_filter(exclude_globs: list[str]) -> Callable[[dict], None]:
    """Provides a function which deletes all subparts of a dictionary based on a list of paths."""

    def filter_func(state_dict: dict) -> None:
        # Flatten dictionary into paths
        paths = []
        _flatten_keys(state_dict, paths, '/')

        filtered_paths = []
        for exclude_glob in exclude_globs:
            filtered_paths_from_glob = fnmatch.filter(paths, exclude_glob)
            if len(filtered_paths_from_glob) == 0:
                warnings.warn(
                    f'No parts from loaded checkpoint state_dict were ignored by load_ignore_key {exclude_glob}')
            filtered_paths.extend(filtered_paths_from_glob)
        filtered_paths = list(set(filtered_paths))
        filtered_paths_str = ', '.join(filtered_paths)
        if filtered_paths:
            log.info(f'Ignoring the following paths from the loaded checkpoint state_dict: {filtered_paths_str}')

        # Loop through all paths to exclude
        paths_to_remove = [path.split('/') for path in filtered_paths]
        _remove_paths(state_dict, paths_to_remove)

    return filter_func


def _validate_save_planner(save_planner: Optional[Any]) -> None:
    """Checks that ``save_planner`` is an instance of a :class:`~torch.distributed.checkpoint.planner.SavePlanner`.

    TODO(GRT-2456): Remove validation once we deprecate torch 1.13 and can use
    type hints.

    Raises:
        ValueError: If ``save_planner`` is not a
            :class:`~torch.distributed.checkpoint.planner.SavePlanner`.
    """
    from torch.distributed.checkpoint.planner import SavePlanner

    if save_planner is not None and not isinstance(save_planner, SavePlanner):
        raise ValueError((f'save_planner {type(save_planner)} is not a '
                          'torch.distributed.checkpoint.planner.SavePlanner'))


def _validate_load_planner(load_planner: Optional[Any]) -> None:
    """Checks that ``load_planner`` is an instance of a :class:`~torch.distributed.checkpoint.planner.LoadPlanner`.

    TODO(GRT-2456): Remove validation once we deprecate torch 1.13 and can use
    type hints.

    Raises:
        ValueError: If ``load_planner`` is not a
            :class:`~torch.distributed.checkpoint.planner.LoadPlanner`.
    """
    from torch.distributed.checkpoint.planner import LoadPlanner

    if load_planner is not None and not isinstance(load_planner, LoadPlanner):
        raise ValueError((f'load_planner {type(load_planner)} is not a '
                          'torch.distributed.checkpoint.planner.LoadPlanner'))


def safe_torch_load(
    composer_states_filepath: Union[Path, str],
    map_location: str = 'cpu',
    load_fsdp_monolith_rank0_only: bool = False,
) -> dict[str, Any]:
    """Load a torch checkpoint, catching errors due to backwards compatibility issues.

    Args:
        composer_states_filepath: The path to the checkpoint file.
        map_location: The location to load the checkpoint to.
        load_fsdp_monolith_rank0_only: Whether the checkpoint is a monolith FSDP checkpoint.
    """
    try:
        if load_fsdp_monolith_rank0_only:
            log.info(
                'Loading monolith FSDP checkpoint. Only rank 0 will load and broadcast non-weight/optimizer state.')
            state_dict_list = [None]
            model = None
            optimizer = None
            tp_size = dist.get_tp_group_size()
            if tp_size is None:
                download_condition = dist.get_global_rank() == 0
            else:
                download_condition = dist.get_fsdp_group_rank() == 0

            if download_condition:
                state_dict_list[0] = _torch_load_with_validation(composer_states_filepath, map_location=map_location)
                # Don't broadcast model/optimizer state if they exist
                if 'model' in state_dict_list[0]['state']:
                    model = state_dict_list[0]['state']['model']
                    state_dict_list[0]['state']['model'] = None
                if 'optimizers' in state_dict_list[0]['state']:
                    optimizer = state_dict_list[0]['state']['optimizers']
                    state_dict_list[0]['state']['optimizers'] = None

            log.debug("Broadcasting state_dict to all ranks in group.")
            if tp_size is None:
                dist.broadcast_object_list(state_dict_list, src=0)
            else:
                fsdp_group = dist.get_fsdp_group()
                # assume that the tp rank is the number of zero rank in dp group
                tp_zero_rank = dist.get_tp_group_rank()
                dist.dist.broadcast_object_list(state_dict_list, src=tp_zero_rank, group=fsdp_group)
            state_dict: Dict[str, Any] = state_dict_list[0]  # type: ignore

            if download_condition:
                if model is not None:
                    state_dict['state']['model'] = model
                if optimizer is not None:
                    state_dict['state']['optimizers'] = optimizer

            return state_dict
        else:
            return _torch_load_with_validation(composer_states_filepath, map_location=map_location)
    except TypeError as e:
        if 'Accuracy.__new__() missing 1 required positional argument' in str(e):
            raise Exception('As of v0.10.0, torchmetrics introduces a new required argument to Accuracy which '
                            'breaks backwards compatibility. Unfortunately, this means that older checkpoints '
                            'cannot be loaded with the metrics. In order to successfully load this model, please '
                            'pass `load_ignore_keys = ["state/train_metrics/*", "state/eval_metrics/*"]`.') from e
        raise e


def _restore_checkpoint(
        state: State,
        logger: Logger,
        composer_states_filepath: str,
        extracted_rank_n: bool,
        extracted_checkpoint_folder: Optional[str],
        load_weights_only: bool,
        strict_model_weights: bool,
        ignore_keys: Optional[Union[list[str], Callable[[dict], None]]],
        exclude_algorithms: Optional[list[str]],
        algorithm_passes: Optional[list[AlgorithmPass]],
        composer_teacher_states_filepath: Optional[str] = None,
        safetensors_mono_sd: Optional[OrderedDict] = None,
        safetensors_teacher_mono_sd: Optional[OrderedDict] = None,
    ) -> Optional[List[Dict[str, Any]]]:
    """Restore a checkpoint into ``state`` and returns the rng state dicts (if ``load_weights_only`` is False)."""

    if safetensors_mono_sd is None:
        # Now, all ranks load the checkpoint that local rank zero downloaded
        state_dict = safe_torch_load(
            composer_states_filepath=composer_states_filepath,
            load_fsdp_monolith_rank0_only=state.load_fsdp_monolith_rank0_only,
        )
    else:
        state_dict = safetensors_mono_sd

    if composer_teacher_states_filepath is not None or safetensors_teacher_mono_sd is not None:
        if composer_teacher_states_filepath:
            teacher_state_dict = safe_torch_load(
                composer_states_filepath=composer_teacher_states_filepath,
                load_fsdp_monolith_rank0_only=state.load_fsdp_monolith_rank0_only,
            )
            if state_dict["state"]["model"] is not None:
                state_dict["state"]["model"] |= {
                    "teacher." + key[len("model."):] : parameter
                    for key, parameter in teacher_state_dict["state"]["model"].items()
                }
        else:
            teacher_state_dict = safetensors_teacher_mono_sd
            if state_dict["state"]["model"] is not None:
                # params here will start with teacher
                state_dict["state"]["model"] |= {
                    key: parameter for key, parameter in teacher_state_dict["state"]["model"].items()
                }

    model_sd = state_dict.get("state", {}).get("model", {}).get("model.meta", {})
    if "checkpoint_version" not in model_sd or float(model_sd["checkpoint_version"]) < 2:
        raise RuntimeError("Only gigafsdp checkpoints version older than 2.0 are supported. " +
                           "Convert your shared checkpoint to monolith and try again.")

    should_validate_hashes = False
    if state.gigafsdp_enabled:
        should_validate_hashes = getattr(state, 'compute_checkpoint_hashes', False)

    loaded_meta = None
    if should_validate_hashes:
        loaded_meta = state_dict.get('state', {}).get('metadata', {})
        log.debug("compute_checkpoint_hashes is set, we will validate hashes after loading state.")

    if ignore_keys:
        # Filter provided list of key paths
        if not callable(ignore_keys):
            ignore_keys = glob_filter(ignore_keys)
        # Call function to modify state_dict
        ignore_keys(state_dict)

    log.debug(f"Loaded checkpoint with keys {state_dict.keys()} and state keys {state_dict['state'].keys()}")

    rng_to_return = None
    if load_weights_only:
        state.load_model_state(
            state_dict['state'],
            logger,
            strict=strict_model_weights,
            exclude_algorithms=exclude_algorithms,
            algorithm_passes=algorithm_passes,
        )
    else:
        state.load_state_dict(
            state_dict['state'],
            logger,
            exclude_algorithms=exclude_algorithms,
            algorithm_passes=algorithm_passes,
        )
        rng_to_return = state_dict.get('rng')

    if should_validate_hashes and loaded_meta is not None and safetensors_mono_sd is None:
        log.info("Validating checkpoint hashes post-load...")

        current_model_state = _get_state(
            state,
            weights_only=True,
            state_dict_type="local",
            device=None)['state']['model']

        validate_hashes(
            loaded_hashes=loaded_meta.get('parameter_hashes'),
            current_state_part=current_model_state,
            strict_check=strict_model_weights,
            state_part_name="Model Parameters"
        )

        if not load_weights_only:
            log.info("Validating Optimizer hashes post-load...")
            loaded_all_optim_hashes = loaded_meta.get('optimizers_hashes')

            if isinstance(loaded_all_optim_hashes, dict) and \
               hasattr(state, 'optimizers') and \
               isinstance(state.optimizers, (tuple, list, dict)):

                num_optimizers = len(state.optimizers)
                log.debug(f"Found {num_optimizers} optimizers in state. Validating against loaded hashes.")

                for idx, optimizer_obj in enumerate(ensure_tuple(state.optimizers)):
                    optimizer_index_str = str(idx)
                    loaded_hashes_for_this_optim = loaded_all_optim_hashes.get(optimizer_index_str)

                    if loaded_hashes_for_this_optim is None:
                        log.warning(f"No loaded hashes found for optimizer index {idx}. Skipping validation for this optimizer.")
                        continue

                    current_optim_state = optimizer_obj.state_dict()
                    validate_hashes(
                        loaded_hashes=loaded_hashes_for_this_optim,
                        current_state_part=current_optim_state,
                        strict_check=strict_model_weights,
                        state_part_name=f"Optimizer {idx}"
                    )
            elif not isinstance(loaded_all_optim_hashes, dict):
                 log.warning(f"Loaded optimizer hashes are not a dictionary (type: {type(loaded_all_optim_hashes)}). Skipping optimizer validation.")
            else:
                 log.warning("Skipping optimizer hash validation as state.optimizers is not a tuple/list or not found after load.")
    return rng_to_return


def _get_state(state: State,
               weights_only: bool,
               save_lora_adapter_only: bool = False,
               state_dict_type: Optional[str] = None,
               device: Optional[Union[str, torch.device]] = None) -> dict[str, Any]:
    log.debug("Getting state_dict...")
    model_state_dict_type = state_dict_type

    if weights_only:
        state_dict = {
            'state': {
                'model': state.get_model_state_dict(state_dict_type=model_state_dict_type),
                'integrations': state._get_integrations_state_dict(),
                'metadata': state._get_state_metadata(),
            },
            'rng': reproducibility.get_rng_state(),
        }
    else:
        state_dict = {
            'state': state.state_dict(state_dict_type=state_dict_type),
            'rng': reproducibility.get_rng_state(),
        }

    if save_lora_adapter_only:
        model_state = state_dict.get("state", {}).get("model")
        if isinstance(model_state, dict):
            trainable_param_names = {
                name for name, parameter in state.model.named_parameters() if parameter.requires_grad
            }
            filtered_model_state = type(model_state)()
            for key, value in model_state.items():
                normalized_key = key.replace("._orig_mod", "")
                keep_by_name = normalized_key in trainable_param_names
                keep_by_lora_pattern = ("lora_" in key.lower()) or ("modules_to_save" in key.lower())
                keep_meta = key == "meta" or key.endswith(".meta")
                if keep_by_name or keep_by_lora_pattern or keep_meta:
                    filtered_model_state[key] = value
            state_dict["state"]["model"] = filtered_model_state
            metadata = state_dict["state"].setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["save_lora_adapter_only"] = True
            log.info(
                "save_lora_adapter_only=True: filtered model state dict from %d to %d tensors.",
                len(model_state),
                len(filtered_model_state),
            )

    should_validate_hashes = False
    if state.gigafsdp_enabled:
        should_validate_hashes = getattr(state, 'compute_checkpoint_hashes', False)

    if should_validate_hashes:
        log.info("compute_checkpoint_hashes is True. Adding hashes to state_dict before saving.")
        add_hash_to_state_dict(state_dict)
    else:
        if 'state' in state_dict and 'metadata' not in state_dict['state']:
            state_dict['state']['metadata'] = {}

    # TO DO: add copy to pinned memory
    if device == torch.device('cpu') or device == 'cpu':
        if 'model' in state_dict['state']:
            state_dict['state']['model'] = state_dict_to_cpu(state_dict['state']['model'])
        if 'optimizers' in state_dict['state']:
            optimizer_state_dict_to_cpu(state_dict['state']['optimizers'])
        log.debug("Getting state_dict and cpu copying is completed.")
    else:
        log.debug(f"Device is not cpu, skipping cpu copying. Device: {device}")

    log.debug('State dict created.')
    return state_dict


def _save_state_dict_on_disk(
    state: State,
    state_dict: dict[str, Any],
    filename: str,
    weights_only: bool,
    is_async: bool,
    barrier: TCPBarrier,
) -> Union[str, None]:

    # Sharded checkpoints get their own little folder.
    if state.fsdp_sharded_state_dict_enabled:
        # To load optimizer states with torch 2.0, the optimizer state must be at the top
        # level of the state dict because the load_sharded_optimizer_state_dict function
        # requires a top level state dict key for the optimizer.
        # See https://github.com/pytorch/pytorch/blob/v2.0.1/torch/distributed/checkpoint/optimizer.py#L271
        # for more info.
        if not state.gigafsdp_enabled:
            if not weights_only:
                state_dict['optimizers'] = state_dict['state'].pop('optimizers')

        # Specify save directory path and save_f
        assert state.sharded_ckpt_prefix_dir is not None
        save_dirpath = Path(Path(filename).parent) / Path(state.sharded_ckpt_prefix_dir)
        save_dirpath = format_name_with_dist_and_time(str(save_dirpath), state.run_name, state.timestamp)

        # New name is now Trainer.save_folder / sharded_ckpt_prefix_dir / __{dist.get_global_rank()}_0.distcp’ if torch > 2
        # else Trainer.save_folder / sharded_ckpt_prefix_dir / ba{batch}_rank{dist.get_global_rank()}.pt’
        # e.g. path/to/my/checkpoints/ep1-ba2/__1_0.distcp if torch >2 else its path/to/my/checkpoints/ep1-ba2/b2-rank1.pt
        ckpt_filename = _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME()
        save_filename = str(Path(save_dirpath) / Path(ckpt_filename))
    else:
        save_filename = PartialFilePath(filename).format(state)

    ep_fsdp_group_rank = dist.get_ep_fsdp_group_rank()

    # Sharded checkpointing for torch >=2.0 uses the torch.distributed.checkpoint module.
    if state.fsdp_elastic_sharded_enabled and not state.gigafsdp_enabled:
        assert not is_async, \
            "Sharded checkpoints cannot be saved asynchronously while using vanilla FSDP. Set option 'save_checkpoint_async: false' in config"
        if state.fsdp_config is None:
            raise ValueError('Saving a sharded checkpoint requires passing an FSDP config to Trainer.')
        save_planner = state.fsdp_config['save_planner']
        _validate_save_planner(save_planner)

        import torch.distributed.checkpoint as dist_cp

        log.debug('Saving sharded checkpoints to %s...', save_filename)

        from torch.distributed.checkpoint.state_dict_loader import _DistWrapper

        _DistWrapper.__init__ = _DistWrapper_custom_init
        fsdp_group = dist.get_fsdp_group()
        group_global_ranks = dist.get_group_global_ranks(fsdp_group)

        with tempfile.TemporaryDirectory(dir=os.path.dirname(dirname), ignore_cleanup_errors=True) as temp_dirname:
            torch_version = get_torch_version()
            if torch_version == version.parse("2.4"):
                dist_cp.save(
                    state_dict=state_dict,
                    storage_writer=dist_cp.FileSystemWriter(temp_dirname),
                    planner=save_planner,
                    process_group=fsdp_group,
                )
            elif torch_version == version.parse("2.1.1"):
                dist_cp.save_state_dict(
                    state_dict=state_dict,
                    storage_writer=dist_cp.FileSystemWriter(temp_dirname),
                    planner=save_planner,
                    process_group=fsdp_group,
                    coordinator_rank=group_global_ranks[0]
                )
            else:
                raise ImportError("Only torch versions 2.1.1 and 2.4 supported")

            os.makedirs(dirname, exist_ok=True)
            for fname in os.listdir(temp_dirname):
                source_path = os.path.join(temp_dirname, fname)
                target_path = os.path.join(dirname, fname)
                os.rename(source_path, target_path)

    # Only rank 0 saves the state_dict unless you are using sharded checkpointing with torch <2.0
    elif dist.get_global_rank() == 0 or state.fsdp_sharded_state_dict_enabled or dist.get_fsdp_group_rank() == 0 or ep_fsdp_group_rank == 0:
        log_msg = (
            f"Saving sharded checkpoints to {save_filename}..."
            if state.fsdp_sharded_state_dict_enabled or dist.get_fsdp_group_rank() == 0
            else f"Saving monolithic checkpoint to {save_filename}"
        )
        if state.gigafsdp_enabled:
            fsdp_rank = dist.get_fsdp_group_rank() \
                if dist.get_fsdp_group() else dist.get_global_rank()
            # Note: fsdp_group_size == hybrid shard group size
            write_shard_condition = fsdp_rank < state_dict['state']['model']['model.meta']['fsdp_group_size']
        else:
            write_shard_condition = True

        if write_shard_condition:
            # if state.gigafsdp_enabled, then save_filename is e.g. run_name/ep0_ba50/ep0_ba50_tp_rank0
            # if not, then save_filename is e.g. run_name/ep0_ba50/ep0_ba50_rank0.pt
            checkpoint_directory = os.path.dirname(save_filename)

            parent_3 = os.path.dirname(os.path.dirname(checkpoint_directory))
            parent_2 = os.path.basename(os.path.dirname(checkpoint_directory))
            parent_1 = os.path.basename(checkpoint_directory)

            if state.gigafsdp_enabled:
                temp_checkpoint_directory = os.path.join(
                    parent_3,
                    "tmp",
                    parent_2,
                    parent_1,
                )
            else:
                temp_checkpoint_directory = os.path.join(
                    parent_3,
                    parent_2,
                    "tmp",
                    parent_1,
                )
            ckpt_basename = os.path.basename(save_filename)
            os.makedirs(temp_checkpoint_directory, exist_ok=True)
            temp_save_path = os.path.join(temp_checkpoint_directory, ckpt_basename)

            BUFF_SIZE = 256 * 1024  # 256K
            with open(temp_save_path, "wb", buffering=BUFF_SIZE) as save_file:
                log.debug(log_msg)
                torch.save(state_dict, save_file, pickle_module=dill, _use_new_zipfile_serialization=True)

            barrier_start = time.perf_counter()
            log.debug("Synchronizing shard writers before moving to final location")
            barrier.wait()
            log.debug(f"Barrier wait took: {time.perf_counter() - barrier_start} seconds")

            # 1. Copy checkpoint files from tmp to final location
            this_rank_saves_symlinks = dist.get_global_rank() == 0 or dist.get_fsdp_group_rank() == 0 or ep_fsdp_group_rank == 0
            if this_rank_saves_symlinks:
                dirname = os.path.dirname(save_filename)

                # create final directory if it does not exist yet
                if dirname:
                    os.makedirs(os.path.dirname(dirname), exist_ok=True)
                    log.debug(f"Rank {dist.get_global_rank()}: Creating directory {os.path.dirname(dirname)}")

                log.debug(f"Rank {dist.get_global_rank()}: Renaming {temp_checkpoint_directory} to {checkpoint_directory}")
                os.rename(temp_checkpoint_directory, checkpoint_directory)

            # 2. Copy metadata folder and write job-dispatcher metadata to artifacts folder
            try:
                if dist.get_global_rank() == 0:
                    if state.gigafsdp_enabled:
                        saved_path_dir = os.path.join(parent_3, parent_2)
                        root_artifact_path = os.path.join(parent_3, 'artifacts')
                    else:
                        saved_path_dir = os.path.join(parent_3, parent_2, parent_1)
                        root_artifact_path = os.path.join(parent_3,parent_2, 'artifacts')
                    log.debug(f'{saved_path_dir=}')
                    model_directory = os.path.join(saved_path_dir, 'artifacts')
                    shutil.copytree(root_artifact_path, model_directory)
                    checkpoint_type = "gigafsdp" if state.gigafsdp_enabled else "fsdp"
                    _jd.write_jd_data(model_directory, checkpoint_type)
                    _jd.submit_send_model(model_directory)
            except Exception as e:
                log.error('Error occured while artifacts saving')
                print(e)

            # wait until all ranks move their temp checkpoint to target destination
            # before cleaning temp directory
            barrier.wait()

            # 3. Remove leftovers from tmp directory
            if dist.get_global_rank() == 0:
                target_dir = os.path.dirname(temp_checkpoint_directory)
                shutil.rmtree(target_dir)

            barrier_start = time.perf_counter()
            log.debug("Synchronizing shard writers after moving to final location (before symlink update)")
            barrier.wait()
            log.debug(f"Barrier wait took: {time.perf_counter() - barrier_start} seconds")

        log.debug(f'Global rank 0 done saving checkpoint to disk at {save_filename}.')

        if is_tar(save_filename):
            _compress_file(save_filename, basename=_COMPOSER_STATES_FILENAME)

    else:
        log.debug(f'Only rank 0 is saving a checkpoint, so rank {dist.get_global_rank()} skips checkpointing.')

    if not is_async:
        dist.barrier()  # ensure all ranks saved their files

    if (
        dist.get_global_rank() == 0
        or state.fsdp_sharded_state_dict_enabled
        or dist.get_fsdp_group_rank() == 0
    ):
        if not state.gigafsdp_enabled:
            assert os.path.exists(save_filename), f"Expected file to have been saved : {save_filename}"
        return save_filename
    else:
        # no file saved
        return None


def save_checkpoint(
    state: State,
    filename: str = 'ep{epoch}-ba{batch}-rank{rank}',
    *,
    weights_only: bool = False,
    save_lora_adapter_only: bool = False,
    barrier: Optional[TCPBarrier] = None,
) -> Union[str, None]:  # noqa: D103
    state_dict = _get_state(
        state,
        weights_only,
        save_lora_adapter_only,
        state_dict_type="local",
        device="cpu"
    )
    return _save_state_dict_on_disk(
        state=state,
        state_dict=state_dict,
        filename=filename,
        weights_only=weights_only,
        is_async=False,
        barrier=barrier,
    )


def state_dict_to_cpu(state_dict: Dict) -> Dict:
    state_dict_cpu = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.distributed.tensor.DTensor):
            state_dict_cpu[k] = v.full_tensor().detach().cpu()
        elif isinstance(v, torch.Tensor):
            if k == "step":
                state_dict_cpu[k] = v.detach().cpu().clone()
            else:
                state_dict_cpu[k] = v.detach().cpu()
        else:
            state_dict_cpu[k] = v
    return state_dict_cpu


def inner_state_dict_to_cpu(state_dict : Dict) -> Dict:
    return {k : state_dict_to_cpu(state_dict[k]) for k in state_dict}


def optimizer_state_dict_to_cpu(state_dict_opt : Dict) -> None:
    for optim_name in state_dict_opt:
        state_dict_opt[optim_name]['state'] = inner_state_dict_to_cpu(state_dict_opt[optim_name]['state'])


def save_checkpoint_async(
    executor: concurrent.futures.Executor,
    state: State,
    filename: str = 'ep{epoch}-ba{batch}-rank{rank}',
    *,
    weights_only: bool = False,
    save_lora_adapter_only: bool = False,
    after_save_func: Callable[[Union[str, None]], None] = None,
    barrier: Optional[TCPBarrier] = None,
) -> concurrent.futures.Future:
    state_dict = _get_state(
        state,
        weights_only,
        save_lora_adapter_only,
        state_dict_type="local",
        device="cpu"
    )

    def save_and_run_after_save():
        try:
            saved_path = _save_state_dict_on_disk(
                state=state,
                state_dict=state_dict,
                filename=filename,
                weights_only=weights_only,
                is_async=True,
                barrier=barrier,
            )
            if after_save_func is not None:
                after_save_func(saved_path)
        except Exception as e:
            log.error(f"Failed to save checkpoint: {e}\n {traceback.format_exc()}")
            raise e

    return executor.submit(save_and_run_after_save)


def _compress_file(filename: str, basename: str):
    """Replace a file with its compressed version.

    The contents will be called ``basename`` inside
    the compressed archive.
    """
    write_mode = _get_write_mode(filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.move(filename, os.path.join(tmpdir, basename))
        with tarfile.open(filename, write_mode) as tarball:
            tarball.add(tmpdir, arcname='')


save_checkpoint.__doc__ = f"""Checkpoint the training ``state``.

Args:
    state (State): The training state.
    logger (Logger): The logger.
    filename (str): A format string describing how to name checkpoints.
        (default: ``'ep{{epoch}}-ba{{batch}}-rank{{rank}}'``)

        The following format variables are available:

        {textwrap.indent(FORMAT_NAME_WITH_DIST_AND_TIME_TABLE, prefix='        ')}

        .. note::

            *   By default, only the rank zero process will save a checkpoint file.


        .. warning::

            Using compression will block the training loop while checkpoints are being compressed. As such, we
            recommend saving checkpoints without compression.

        Consider the following scenario, where:

        *   The default ``name='ep{{epoch}}-ba{{batch}}-rank{{rank}}'`` is used.
        *   The current epoch count is ``1``.
        *   The current batch count is ``42``.

    weights_only (bool, optional): If ``True``, save only the model weights instead of the entire training state.
        (default: ``False``)

    Returns:
        list[pathlib.Path]: The list of checkpoint files saved, indexed by the rank of the process.

        .. note::

            When doing multi-node training, the filepaths are valid only on each process's node;
            Composer does not move checkpoint files between nodes.

            Each list will contain only one filepath,
            since only the rank zero process saves checkpoints.
"""
