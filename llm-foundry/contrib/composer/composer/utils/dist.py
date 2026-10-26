# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helper methods for :mod:`torch.distributed`.

To use :mod:`torch.distributed`, launch your training script with the
:ref:`composer launcher for distributed training <distributed-training>`. For example,
the following command launches an eight-process training run.

.. code-block::

    composer -n 8 path/to/train.py

The composer launcher will automatically configure the following environment variables, which are
required for distributed training:

* ``RANK``: The global rank of the process, which should be on ``[0; WORLD_SIZE - 1]``.
* ``LOCAL_RANK``: The local rank for the process, which should be on ``[0; LOCAL_WORLD_SIZE - 1]``.
* ``NODE_RANK``: The rank of the node.
* ``WORLD_SIZE``: The total number of processes.
* ``LOCAL_WORLD_SIZE``: The number of processes on the current node.
* ``MASTER_ADDR``: The hostname for the rank-zero process.
* ``MASTER_PORT``: The port for the rank-zero process.

If none of these environment variables are set, this module will safely assume a single-rank configuration, where::

    RANK=0
    LOCAL_RANK=0
    NODE_RANK=0
    WORLD_SIZE=1
    LOCAL_WORLD_SIZE=1
"""
from __future__ import annotations

import datetime
import io
import logging
import os
import pickle
import time
import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, TypeVar, Union, cast, Dict
from packaging import version

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch.distributed.distributed_c10d import ProcessGroup

from composer.utils.device import get_device, is_hpu_installed
from composer.utils.misc import get_torch_version
from omegaconf import DictConfig

if TYPE_CHECKING:
    from composer.devices import Device

TObj = TypeVar('TObj')

__all__ = [
    "all_gather",
    "all_gather_object",
    "all_reduce",
    "barrier",
    "broadcast",
    "broadcast_object_list",
    "get_global_rank",
    "get_local_rank",
    "get_local_world_size",
    "get_node_rank",
    "get_node_leader_group",
    "get_sampler",
    "get_world_size",
    "initialize_dist",
    "is_available",
    "is_initialized",
    "get_group_size",
    "get_group_rank",
    "get_fsdp_group",
    "get_fsdp_group_size",
    "get_fsdp_group_rank",
    "get_ep_group",
    "get_ep_group_size",
    "get_ep_group_rank",
    "get_tp_group",
    "get_tp_group_size",
    "get_tp_group_rank",
    "get_sp_group",
    "get_sp_group_size",
    "get_sp_group_rank",
    "get_tp_sp_group",
    "get_tp_sp_group_size",
    "get_tp_sp_group_rank",
    "get_group_global_ranks",
    "is_trivial_process_group",
]

log = logging.getLogger(__name__)

_pickler = pickle.Pickler
_unpickler = pickle.Unpickler

_TENSOR_PARALLEL_GROUP: ProcessGroup = None
_FSDP_PARALLEL_GROUP: ProcessGroup = None
_SEQUENCE_PARALLEL_GROUP: ProcessGroup = None
_TENSOR_SEQUENCE_PARALLEL_GROUP: ProcessGroup = None
_EXPERT_PARALLEL_GROUP: ProcessGroup = None
_EXPERT_FSDP_GROUP: ProcessGroup = None
_NODE_LEADER_GROUP: ProcessGroup = None

_ALL_PROCESS_GROUPS = {}
_CURRENT_TP_SIZE = None
_CURRENT_SP_SIZE = None
_DYNAMIC_SP_ENABLED = False
_SP_GROUPS_SET = False

def _object_to_tensor(obj, device):
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size


def _tensor_to_object(tensor, tensor_size):
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()


def all_gather_object_list_hpu(object_list, obj, group=None):
    """Use this only for habana devices, for other devices use all_gather_object.

    Function is a modified version of
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py.
    Gathers picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the object
    must be picklable in order to be gathered.

    Args:
        object_list (list[Any]): Output list. It should be correctly sized as the
            size of the group for this collective and will contain the output.
        obj (Any): Pickable Python object to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list`` will
        be unmodified.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        :func:`all_gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.all_gather_object(output, gather_objects[dist.get_rank()])
        >>> output
        ['foo', 12, {1: 2}]
    """
    current_device = torch.device('hpu')
    input_tensor, local_size = _object_to_tensor(obj, current_device)
    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = dist.get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size, dtype=torch.long, device=current_device)
    object_size_list = [object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)]
    # Allgather tensor sizes
    handle = dist.all_gather(object_size_list, local_size, group=group, async_op=True)
    handle.wait()
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(max_object_size * group_size, dtype=torch.bfloat16, device=current_device)
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [coalesced_output_tensor[max_object_size * i:max_object_size * (i + 1)] for i in range(group_size)]
    handle = dist.all_gather(output_tensors, input_tensor.to(torch.bfloat16), group=group, async_op=True)
    handle.wait()
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)


class MissingEnvironmentError(Exception):
    pass


def _get_distributed_config_var(
    env_var: str,
    human_name: str,
    default: int,
    fetch_fn_name: Optional[str] = None,
) -> int:
    if not dist.is_available():
        return default

    if dist.is_initialized() and fetch_fn_name is not None:
        dist_value = int(getattr(dist, fetch_fn_name)())
        if env_var in os.environ:
            env_value = int(os.environ[env_var])
            if dist_value != env_value:
                raise RuntimeError('Torch distributed has been initialized with a value of '
                                   f'{dist_value} for {human_name}, but environment variable '
                                   f'{env_var} has value {env_value}.')
        return dist_value

    if env_var in os.environ:
        return int(os.environ[env_var])

    if dist.is_initialized():
        raise MissingEnvironmentError('Torch distributed is initialized but environment variable '
                                      f'{env_var} is not set.')

    return default


def get_world_size() -> int:
    """Returns the world size, which is the number of processes participating in this training run.

    Returns:
        int: The world size.
    """
    return _get_distributed_config_var(env_var='WORLD_SIZE',
                                       human_name='world size',
                                       default=1,
                                       fetch_fn_name='get_world_size')


def get_global_rank() -> int:
    """Returns the global rank of the current process, which is on ``[0; WORLD_SIZE - 1]``.

    Returns:
        int: The global rank.
    """
    return _get_distributed_config_var(env_var='RANK', human_name='global rank', default=0, fetch_fn_name='get_rank')


def get_local_world_size() -> int:
    """Returns the local world size, which is the number of processes for the current node.

    Returns:
        int: The local world size.
    """
    return _get_distributed_config_var(env_var='LOCAL_WORLD_SIZE', default=1, human_name='local world size')


def get_local_rank() -> int:
    """Returns the local rank for the current process, which is on ``[0; LOCAL_WORLD_SIZE - 1]``.

    Returns:
        int: The local rank.
    """
    return _get_distributed_config_var(env_var='LOCAL_RANK', default=0, human_name='local rank')


def get_node_rank() -> int:
    """Returns the node rank.

    For example, if there are 2 nodes, and 2 ranks per node, then global ranks 0-1 will have a
    node rank of 0, and global ranks 2-3 will have a node rank of 1.

    Returns:
        int: The node rank, starting at 0.
    """
    return _get_distributed_config_var(env_var='NODE_RANK', default=0, human_name='node rank')


def barrier() -> None:
    """Synchronizes all processes.

    This function blocks until all processes reach this function.

    .. seealso:: :func:`torch.distributed.barrier`
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


def all_reduce(
    tensor: torch.Tensor,
    reduce_operation: str = 'SUM',
    group: ProcessGroup | None = None,
    async_op: bool = True # Sbr: add arg to use sync operations (coalescing wrapper doesn't work with async ops)
) -> None:
    """Reduce a ``tensor`` by applying the ``reduce_operation``.

    All ranks get the same, bitwise-identical result.

    .. seealso:: :func:`torch.distributed.all_reduce`

    Args:
        tensor (torch.Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): Process group to perform the operation on.
        Defaults to ``None``, which is the default process group.
    Args:
        tensor (torch.Tensor): Tensor to reduce. The function operates in-place.
        reduce_operation (str, optional): The reduction operation (default: ``SUM``).

            Valid options are:
                * ``SUM``
                * ``PRODUCT``
                * ``MIN``
                * ``MAX``
                * ``BAND``
                * ``BOR``
                * ``BXOR``

    Returns:
        None: ``tensor`` is modified in-place.
    """
    if dist.is_available() and dist.is_initialized():
        reduce_op = getattr(dist.ReduceOp, reduce_operation.upper())
        handle = dist.all_reduce(tensor, op=reduce_op, group=group, async_op=async_op)
        if async_op:
            handle.wait()
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


def broadcast(tensor: torch.Tensor, src: int) -> None:
    """Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes participating in the collective.
    See :func:`torch.distributed.broadcast`.

    Args:
        tensor (torch.Tensor): Data to be sent if ``src`` is the rank of current process,
            and tensor to be used to save received data otherwise.
        src (int): Source rank
    """
    if dist.is_available() and dist.is_initialized():
        handle = dist.broadcast(tensor, src, async_op=True)
        handle.wait()
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


def broadcast_object_list(object_list: List[Any], src: int = 0) -> None:
    """Broadcasts picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be broadcasted.

    .. seealso:: :func:`torch.distributed.broadcast`.

    Args:
        object_list (torch.Tensor): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will be broadcast,
            but each rank must provide lists of equal sizes.
        src (int, optional): Source rank (default: ``0``)

    Returns:
        None:  ``object_list`` will be modified in-place and set to values of ``object_list`` from the ``src`` rank.
    """
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(object_list, src)
        # torch.distributed will replace the None's in obj_gather_list with the gathered objects on rank 0
        # or will just be None on non-rank-0
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


def all_gather(tensor: torch.Tensor, group: ProcessGroup | None = None) -> Sequence[torch.Tensor]:
    """Collects a :class:`~torch.Tensor` from each rank.

    .. seealso:: :func:`torch.distributed.all_gather`

    Args:
        tensor (torch.Tensor): Tensor from each rank to be gathered.
        group (ProcessGroup, optional): Process group to perform the gathering for. Defaults to ``None``
        which results into using default process group.

    Returns:
        Sequence[Tensor]: A sequence of tensors indexed by rank.
    """
    if dist.is_available() and dist.is_initialized():
        obj_gather_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size(group))]
        handle = dist.all_gather(obj_gather_list, tensor, group=group, async_op=True)
        handle.wait()
        return obj_gather_list
    world_size = get_world_size()
    if world_size == 1:
        return [tensor]
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


def all_gather_object(obj: TObj) -> List[TObj]:
    """Collect a pickleable object from each rank and return a list of these objects indexed by rank.

    .. seealso:: :func:`torch.distributed.all_gather_object`

    Args:
        obj (TObj): Object to be gathered.

    Returns:
        List[TObj]: A list of objects indexed by rank.
    """
    if dist.is_available() and dist.is_initialized():
        obj_gather_list = [None for _ in range(get_world_size())]
        if is_hpu_installed():
            all_gather_object_list_hpu(obj_gather_list, obj)
        else:
            dist.all_gather_object(obj_gather_list, obj)
        # torch.distributed will replace the None's in obj_gather_list with the gathered objects on rank 0
        # or will just be None on non-rank-0
        return cast(List[TObj], obj_gather_list)
    world_size = get_world_size()
    if world_size == 1:
        return [obj]
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


def is_available():
    """Returns whether PyTorch was built with distributed support.

    .. seealso:: :func:`torch.distributed.is_available`

    Returns:
        bool: Whether PyTorch distributed support is available.
    """
    return dist.is_available()


def is_initialized():
    """Returns whether PyTorch distributed is initialized.

    .. seealso:: :func:`torch.distributed.is_initialized`

    Returns:
        bool: Whether PyTorch distributed is initialized.
    """
    return dist.is_initialized()


def _create_fsdp_tp_groups(tp_size: int, current_rank: int, world_size: int, timeout: float, init_nccl: bool):
    """
    Create global FSDP and Tensor Parallel groups that are identical regardless of sequence parallel size.
    Returns two dictionaries keyed by tuple of ranks.
    """
    fsdp_group = None
    tp_group = None

    if tp_size > 1:
        # Create Tensor Parallel groups
        mesh = np.arange(world_size).reshape((-1, tp_size))
        for row in mesh:
            tp_ranks = row.flatten().tolist()
            tp_group_tmp = (
                _new_high_priority_group(tp_ranks, timeout=timeout, init_nccl=init_nccl)
                if init_nccl
                else tp_ranks
            )
            if current_rank in tp_ranks:
                assert get_group_size(tp_group_tmp) == tp_size, f"TP group size {get_group_size(tp_group_tmp)} != {tp_size}"
                tp_group = tp_group_tmp

        # Create FSDP groups
        for col in range(mesh.shape[1]):
            fsdp_ranks = mesh[:, col].flatten().tolist()
            fsdp_group_tmp = (
                dist.new_group(fsdp_ranks, timeout=datetime.timedelta(seconds=timeout))
                if init_nccl
                else fsdp_ranks
            )
            if current_rank in fsdp_ranks:
                assert get_group_size(fsdp_group_tmp) == (get_world_size() // tp_size), f"FSDP group size {get_group_size(fsdp_group_tmp)} != {get_world_size() // tp_size}"
                fsdp_group = fsdp_group_tmp
    else:
        # When tp_size is 1, the global group is the entire world.
        global_group = (
            dist.new_group(list(range(world_size)), timeout=datetime.timedelta(seconds=timeout))
            if init_nccl
            else None
        )
        fsdp_group = global_group
        tp_group = None

    return fsdp_group, tp_group


def _new_high_priority_group(ranks: list, timeout: float, init_nccl: bool, backend: str="nccl", **kwargs):
    """
    Creates a new process group with high priority options if NCCL is initialized.
    """
    if not init_nccl:
        return None
    pg_options = dist.ProcessGroupNCCL.Options()
    pg_options.is_high_priority_stream = True
    return dist.new_group(
        ranks, timeout=datetime.timedelta(seconds=timeout), backend=backend, pg_options=pg_options, **kwargs
    )


def _create_node_leader_group(node_zero_ranks: list, timeout: float, init_nccl: bool):
    """
    Create process group containing local-rank-zero of each node.
    """
    
    if not init_nccl:
        return None
    
    group = dist.new_group(
        node_zero_ranks,
        timeout=datetime.timedelta(seconds=timeout),
    )

    if get_global_rank() in node_zero_ranks:
        return group
    return None


def get_process_groups(
    tp_size: int,
    sp_sizes: list[int],
    ep_size: int,
    world_size: int,
    process_global_rank: int,
    timeout: float = 300,
    init_nccl: bool = True,
):
    """
    Returns the final process group structure as a dictionary:

    {
      "fsdp": <global fsdp group>,
      "tp": <global tensor parallel group>,
      "sp_groups": {
          sp_size_1: {
              "tp_sp": <combined tensor-sequence parallel group for sp_size_1>,
              "sp": <pure sequence parallel group for sp_size_1>
          },
          sp_size_2: {
              "tp_sp": <combined tensor-sequence parallel group for sp_size_2>,
              "sp": <pure sequence parallel group for sp_size_2>
          },
          ...
      }
    }

    Global FSDP and TP groups are created once and reused.
    For each sp_size, the corresponding sequence parallel groups are computed.
    """
    assert isinstance(sp_sizes, list), "sp_sizes must be a list"

    # Validate tp_size and sp_sizes
    if tp_size % 2 != 0 and tp_size != 1:
        raise ValueError(f"Tensor parallel size should be even or 1, got {tp_size}.")
    if ep_size % 2 != 0 and ep_size != 1:
        raise ValueError(f"Tensor parallel size should be even or 1, got {ep_size}.")
    fsdp_group_size = world_size // tp_size
    for sp_size in sp_sizes:
        if sp_size % 2 != 0 and sp_size != 1:
            raise ValueError(f"Sequence parallel size should be even or 1, got {sp_size}.")
        if fsdp_group_size < sp_size or fsdp_group_size % sp_size == 1:
            raise RuntimeError(
                f"Sequence parallel size should be lower than FSDP group size and split it evenly, "
                f"got FSDP size {fsdp_group_size} and sp_size {sp_size}."
            )
        if tp_size * sp_size > world_size:
            raise RuntimeError(
                f"Incompatible dimensions: tp_size * sp_size = {tp_size * sp_size} > world_size = {world_size}."
            )

    # Create global FSDP and TP groups
    fsdp_group, tp_group = _create_fsdp_tp_groups(tp_size, process_global_rank, world_size, timeout, init_nccl)
    ep_compliment_group, ep_group = _create_fsdp_tp_groups(
        ep_size, process_global_rank, world_size, timeout, init_nccl
    )
    sp_groups_final = {}

    for sp_size in sp_sizes:
        try:
            sp_mesh = np.arange(world_size).reshape((-1, sp_size, tp_size))
        except ValueError as e:
            raise ValueError("world_size must be divisible by (sp_size * tp_size).") from e

        # sequence parallel group (sp)
        sp_group = None
        for fs_dim in range(sp_mesh.shape[0]):
            for tp_dim in range(sp_mesh.shape[2]):
                ranks = sp_mesh[fs_dim, :, tp_dim].flatten().tolist()
                sp_group_tmp = _new_high_priority_group(ranks, timeout, init_nccl, backend="nccl")
                if process_global_rank in ranks:
                    assert get_group_size(sp_group_tmp) == sp_size, f"SP group size {get_group_size(sp_group_tmp)} != {sp_size}"
                    sp_group = sp_group_tmp
        if sp_group is None:
            raise RuntimeError(f"SP group not found for sp_size {sp_size}")

        # tensor-sequence parallel group (tp_sp)
        tp_sp_group = None
        if sp_group is None:
            tp_sp_group = None
        else:
            for fs_dim in range(sp_mesh.shape[0]):
                ranks = sp_mesh[fs_dim, :, :].flatten().tolist()
                tp_sp_group_tmp = _new_high_priority_group(ranks, timeout, init_nccl, backend="nccl")
                if process_global_rank in ranks:
                    assert get_group_size(tp_sp_group_tmp) == tp_size * sp_size, f"TP-SP group size {get_group_size(tp_sp_group_tmp)} != {tp_size * sp_size}"
                    tp_sp_group = tp_sp_group_tmp

            if tp_sp_group is None:
                raise RuntimeError(f"TP-SP group not found for sp_size {sp_size}")

        sp_groups_final[sp_size] = {
            "sp": sp_group,
            "tp_sp": tp_sp_group,
        }

    node_leader_ranks = list(range(0, world_size, get_local_world_size()))
    node_leader_group = _create_node_leader_group(node_leader_ranks, timeout, init_nccl)

    return {
        "fsdp": fsdp_group,
        "tp": tp_group,
        "ep": ep_group,
        "sp_groups": sp_groups_final,
        # TODO: there certainly is a better way of doing this
        "ep_fspd": ep_compliment_group,
        "node_leaders": node_leader_group,
    }


def initialize_dist(
    device: Union[str, Device],
    tp_size: int = 1,
    sp_size: Union[List[int], int] = 1,
    ep_size: int = 1,
    timeout: float = 300.0,
    enable_async_tp: bool = False,
):
    """Initialize the default PyTorch distributed process group.

    This function assumes that the following environment variables are set:

    * ``RANK``: The global rank of the process, which should be on ``[0; WORLD_SIZE - 1]``.
    * ``LOCAL_RANK``: The local rank for the process, which should be on ``[0; LOCAL_WORLD_SIZE - 1]``.
    * ``NODE_RANK``: The rank of the node.
    * ``WORLD_SIZE``: The total number of processes.
    * ``LOCAL_WORLD_SIZE``: The number of processes on the current node.
    * ``MASTER_ADDR``: The hostname for the rank-zero process.
    * ``MASTER_PORT``: The port for the rank-zero process.

    If none of the environment variables are set, this function will assume a single-rank
    configuration and initialize the default process group using a :class:`torch.distributed.HashStore` store.

    .. seealso:: :func:`torch.distributed.init_process_group`

    Args:
        device (str | Device): The device from which the distributed backend is
            interpreted. Either a string corresponding to a device (one of ``'cpu'``,
            ``'gpu'``, ``'mps'``, or ``'tpu'``) or a :class:`.Device`.
        tp_size (int, optional): Defines whether to create or not Tensor Parallel group. Default: ``1``.
        sp_size (int, optional): Defines whether to create or not Sequence Parallel group. Default: ``1``.
        ep_size (int, optional): Defines whether to create or not Expert Parallel group. Default: ``1``.
        timeout (float, optional): The timeout for operations executed against the process
            group, expressed in seconds. (default: ``300.0``).
    """
    # If device is string, get corresponding composer.devices.Device object
    device_obj = get_device(device)
    timeout_timedelta = datetime.timedelta(seconds=timeout)

    if get_world_size() > 1 and not dist.is_available():
        raise RuntimeError('When the world size is > 1, ``torch.distributed`` must be used. However, it is '
                           'not available in your installation of PyTorch. Please install or build PyTorch '
                           'with distributed support.')

    if dist.is_initialized():
        if dist.get_backend() != device_obj.dist_backend.lower():
            raise RuntimeError(f'The requested backend ({device_obj.dist_backend}) differs from the backend '
                               f'of the current process group ({dist.get_backend()}). If you '
                               'wish to change backends, please restart the python process.')
        return

    # If any of these variables are set, and they do not match the single rank defaults,
    # then do not automatically configure distributed. There are no reasonable defaults to infer
    # for the other variables. Instead, let torch.dist error on an incomplete configuration.

    # If none of these variables are set, or some are set but they match the single rank defaults,
    # then fill the rest in.

    dist_env_var_defaults = {
        'NODE_RANK': '0',
        'WORLD_SIZE': '1',
        'LOCAL_WORLD_SIZE': '1',
        'RANK': '0',
        'LOCAL_RANK': '0',
    }

    log.debug(
        'Initializing torch.dist: global_rank=%d, local_rank=%d, world_size=%d, local_world_size=%d, node_rank=%d',
        get_global_rank(),
        get_local_rank(),
        get_world_size(),
        get_local_world_size(),
        get_node_rank(),
    )

    dist_env_vars_match_defaults = all(os.environ.get(k, v) == v for (k, v) in dist_env_var_defaults.items())

    if dist_env_vars_match_defaults:
        # Fill in the remaining single-rank variables
        os.environ.update(dist_env_var_defaults)
        dist.init_process_group(device_obj.dist_backend, store=dist.HashStore(), world_size=1, rank=0)
    else:
        dist.init_process_group(device_obj.dist_backend, timeout=timeout_timedelta)

    # NOTE (Sbr): Get out of this function right away if don't need tensor or sequence parallelism
    # to avoid allocating unneeded resources that persist over all program's work (unused process groups).
    sp_sizes = [sp_size] if isinstance(sp_size, int) else sp_size
    if isinstance(sp_sizes[0], DictConfig):
        sp_sizes = [t['sp_size'] for t in sp_sizes]
    sp_sizes.sort(reverse=True)

    if tp_size == 1 and (len(sp_sizes) == 1 and sp_sizes[0] == 1) and ep_size == 1:
        return

    log.info(f"Initializing dist with tp_size={tp_size}, sp_sizes={sp_sizes}")
    # Make sure that GPU ranks are set correctly
    assert get_global_rank() == torch.cuda.device_count() * get_node_rank() + get_local_rank(), (
        f"Error in setting up devices ranks. Global rank is {get_global_rank()}, "
        f"but {torch.cuda.device_count()} * node rank + local rank is {torch.cuda.device_count() * get_node_rank() + get_local_rank()}."
    )

    global _ALL_PROCESS_GROUPS, _CURRENT_TP_SIZE, _CURRENT_SP_SIZE, _DYNAMIC_SP_ENABLED, _SP_GROUPS_SET

    process_groups = get_process_groups(
        tp_size=tp_size,
        sp_sizes=sp_sizes,
        ep_size=ep_size,
        world_size=get_world_size(),
        process_global_rank=get_global_rank(),
        timeout=timeout,
        init_nccl=True,
    )
    _ALL_PROCESS_GROUPS = process_groups
    _CURRENT_TP_SIZE = tp_size
    _CURRENT_SP_SIZE = sp_sizes[0]
    _DYNAMIC_SP_ENABLED = len(sp_sizes) > 1
    _SP_GROUPS_SET = True

    global _FSDP_PARALLEL_GROUP, _TENSOR_PARALLEL_GROUP, _SEQUENCE_PARALLEL_GROUP, _TENSOR_SEQUENCE_PARALLEL_GROUP, _EXPERT_PARALLEL_GROUP, _EXPERT_FSDP_GROUP, _NODE_LEADER_GROUP
    initial_sp_group = process_groups['sp_groups'][sp_sizes[0]]
    _FSDP_PARALLEL_GROUP = process_groups['fsdp']
    _TENSOR_PARALLEL_GROUP = process_groups['tp']
    _EXPERT_PARALLEL_GROUP = process_groups['ep']
    _SEQUENCE_PARALLEL_GROUP = initial_sp_group['sp']
    _TENSOR_SEQUENCE_PARALLEL_GROUP = initial_sp_group['tp_sp']
    _EXPERT_FSDP_GROUP = process_groups['ep_fspd']
    _NODE_LEADER_GROUP = process_groups['node_leaders']

    # log.info(f"SP_SIZES: {_DYNAMIC_SP_ENABLED} {_SP_GROUPS_SET}")
    os.environ["FSDP_RANK"] = str(get_fsdp_group_rank())
    os.environ["TP_RANK"] = str(get_tp_group_rank())
    # TODO: be careful1
    # os.environ["SP_RANK"] = str(get_sp_group_rank())
    # os.environ["TP_SP_RANK"] = str(get_tp_sp_group_rank())
    assert "SP_RANK" not in os.environ, "SP_RANK should not be set"
    assert "TP_SP_RANK" not in os.environ, "TP_SP_RANK should not be set"

    if len(sp_sizes) > 1:
        log.info(f"DYNAMIC_SP GROUPS SET: {_CURRENT_TP_SIZE=}, {_CURRENT_SP_SIZE=} {sp_sizes=}")
    if enable_async_tp:
        assert tp_size > 1, "tensor parallelism need to be enabled for async tp"
        assert get_torch_version() >= version.parse("2.5.1"), "Symmetric memory (AsyncTP) is only supported for torch>=2.5.1"
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group
        enable_symm_mem_for_group(get_tp_group().group_name)


def get_group_size(group):
    if group is None:
        return None

    return group.size()


def get_group_rank(group):
    if group is None:
        return None

    return group.rank()


def get_node_leader_group():
    """
    Returns process group containing local-rank-zero of each node.
    """
    global _NODE_LEADER_GROUP
    return _NODE_LEADER_GROUP


def get_ep_group():
    global _EXPERT_PARALLEL_GROUP
    return _EXPERT_PARALLEL_GROUP


def get_tp_group():
    global _TENSOR_PARALLEL_GROUP
    return _TENSOR_PARALLEL_GROUP


def get_fsdp_group():
    global _FSDP_PARALLEL_GROUP
    return _FSDP_PARALLEL_GROUP


def get_ep_fsdp_group():
    global _EXPERT_FSDP_GROUP
    return _EXPERT_FSDP_GROUP


def get_sp_group():
    global _SEQUENCE_PARALLEL_GROUP, _DYNAMIC_SP_ENABLED, _SP_GROUPS_SET
    if _DYNAMIC_SP_ENABLED and not _SP_GROUPS_SET:
        raise RuntimeError("SP group not set. Call set_sp_dynamic_group() first.")
    return _SEQUENCE_PARALLEL_GROUP

def get_tp_sp_group():
    global _TENSOR_SEQUENCE_PARALLEL_GROUP, _DYNAMIC_SP_ENABLED, _SP_GROUPS_SET
    if _DYNAMIC_SP_ENABLED and not _SP_GROUPS_SET:
        raise RuntimeError("TP-SP group not set. Call set_sp_dynamic_group() first.")
    return _TENSOR_SEQUENCE_PARALLEL_GROUP

def set_sp_dynamic_group(sp_size: int):
    global _SEQUENCE_PARALLEL_GROUP, _TENSOR_SEQUENCE_PARALLEL_GROUP, _CURRENT_SP_SIZE, _DYNAMIC_SP_ENABLED, _SP_GROUPS_SET, _CURRENT_TP_SIZE

    if not _DYNAMIC_SP_ENABLED:
        raise RuntimeError("Dynamic SP is not enabled. Initialize with multiple sp_sizes to enable.")

    if sp_size not in _ALL_PROCESS_GROUPS['sp_groups']:
        raise ValueError(f"SP size {sp_size} not precomputed in _ALL_PROCESS_GROUPS")

    groups = _ALL_PROCESS_GROUPS['sp_groups'][sp_size]
    _SEQUENCE_PARALLEL_GROUP = groups['sp']
    assert (get_group_size(_SEQUENCE_PARALLEL_GROUP) or 1) == sp_size, \
        f"SP group size {get_group_size(_SEQUENCE_PARALLEL_GROUP)} != {sp_size}"
    _TENSOR_SEQUENCE_PARALLEL_GROUP = groups['tp_sp']
    assert (get_group_size(_TENSOR_SEQUENCE_PARALLEL_GROUP) or 1) == (_CURRENT_TP_SIZE * sp_size), \
        f"TP-SP group size {get_group_size(_TENSOR_SEQUENCE_PARALLEL_GROUP)} != {_CURRENT_TP_SIZE * sp_size}"
    _CURRENT_SP_SIZE = sp_size
    _SP_GROUPS_SET = True

    # with dynamic sp we decided to not set SP_RANK and TP_SP_RANK
    #  this asserts are here to remind us that we should not set them
    assert "SP_RANK" not in os.environ, "SP_RANK should not be set"
    assert "TP_SP_RANK" not in os.environ, "TP_SP_RANK should not be set"


def get_ep_group_rank():
    global _EXPERT_PARALLEL_GROUP
    return get_group_rank(_EXPERT_PARALLEL_GROUP)


def get_ep_group_size():
    global _EXPERT_PARALLEL_GROUP
    return get_group_size(_EXPERT_PARALLEL_GROUP)


def get_tp_group_rank():
    global _TENSOR_PARALLEL_GROUP
    return get_group_rank(_TENSOR_PARALLEL_GROUP)


def get_tp_group_size():
    global _TENSOR_PARALLEL_GROUP
    return get_group_size(_TENSOR_PARALLEL_GROUP)


def get_fsdp_group_rank():
    global _FSDP_PARALLEL_GROUP
    return get_group_rank(_FSDP_PARALLEL_GROUP)


def get_fsdp_group_size():
    global _FSDP_PARALLEL_GROUP
    return get_group_size(_FSDP_PARALLEL_GROUP)


def get_ep_fsdp_group_rank():
    global _EXPERT_FSDP_GROUP
    return get_group_rank(_EXPERT_FSDP_GROUP)


def get_sp_group_rank():
    global _SEQUENCE_PARALLEL_GROUP
    return get_group_rank(_SEQUENCE_PARALLEL_GROUP)


def get_sp_group_size():
    global _SEQUENCE_PARALLEL_GROUP
    return get_group_size(_SEQUENCE_PARALLEL_GROUP)


def get_tp_sp_group_rank():
    global _TENSOR_SEQUENCE_PARALLEL_GROUP
    return get_group_rank(_TENSOR_SEQUENCE_PARALLEL_GROUP)


def get_tp_sp_group_size():
    global _TENSOR_SEQUENCE_PARALLEL_GROUP
    return get_group_size(_TENSOR_SEQUENCE_PARALLEL_GROUP)


def get_ep_fsdp_group_size():
    global _EXPERT_FSDP_GROUP
    return get_group_size(_EXPERT_FSDP_GROUP)


def is_trivial_process_group(
    group: torch.distributed.ProcessGroup | None
) -> bool:
    return group is None or group.size() == 1 or math.isnan(group.size())


def get_group_global_ranks(group: ProcessGroup):
    group_size = dist.get_world_size(group)
    global_rank = get_global_rank()

    global_rank_tensor = torch.tensor([global_rank],
                                      dtype=torch.int32,
                                      device=torch.cuda.current_device())

    gathered_ranks = torch.zeros(group_size,
                                 dtype=torch.int32,
                                 device=torch.cuda.current_device())
    handle = dist.all_gather_into_tensor(gathered_ranks,
                                global_rank_tensor,
                                group=group,
                                async_op=True)
    handle.wait()

    group_global_ranks = gathered_ranks.cpu().tolist()
    return group_global_ranks


def get_sampler(dataset: torch.utils.data.Dataset, *, drop_last: bool = False, shuffle: bool = False):
    """Constructs a :class:`~torch.utils.data.distributed.DistributedSampler` for a dataset.

    The :class:`~torch.utils.data.distributed.DistributedSampler` assumes that each rank has a complete copy of the
    dataset. It ensures that each rank sees a unique shard for each epoch containing
    ``len(dataset) / get_world_size()`` samples.

    .. note::

        If the ``dataset`` is already sharded by rank, use a :class:`~torch.utils.data.SequentialSampler`
        or :class:`~torch.utils.data.RandomSampler`.

    Args:
        dataset (torch.utils.data.Dataset): The dataset.
        drop_last (bool): Whether to trop the last batch.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        torch.utils.data.distributed.DistributedSampler: The sampler.
    """
    return torch.utils.data.DistributedSampler[int](
        dataset,
        drop_last=drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
    )


@contextmanager
def local_rank_zero_download_and_wait(expected_file_path: str):
    """Context manager to wait for a file to exist on all ranks except local rank zero.

    It is expected that the file will be created by local rank zero. This function is useful
    as an alternative to ``run_local_rank_zero_first`` when downloading a file, because it does
    not require dist to be initialized. It only requires that the ``LOCAL_RANK`` environment variable
    is set. If dist is initialized, you should use ``run_local_rank_zero_first`` instead to avoid busy waiting.

    Args:
        expected_file_path (str): The file to wait for existence of
    """
    local_rank = get_local_rank()
    if local_rank != 0:
        while not os.path.exists(expected_file_path):
            time.sleep(0.1)

    yield


@contextmanager
def run_local_rank_zero_first():
    """Context manager to hold all non-zero ranks until rank zero completes.

    The below example will let the local rank zero download
    the dataset, and hold all non-rank zeros until the
    download is complete.

    .. code-block: python

        with run_local_rank_zero_first():
            dataset = CIFAR10(
                ...,
                download=True,
            )

    This prevents race conditions where multiple
    ranks attempt to download the dataset to the
    same location.
    """
    if dist.is_available() and dist.is_initialized():
        # hold non-zero ranks until rank zero done
        if get_local_rank() != 0:
            dist.barrier()
            yield
        else:
            yield
            dist.barrier()
        return
    world_size = get_world_size()
    if world_size == 1:
        yield
        return
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')
