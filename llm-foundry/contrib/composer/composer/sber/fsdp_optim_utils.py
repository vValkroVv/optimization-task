# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import copy
import functools
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Union
from packaging import version

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_pg_default_device
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
from torch.distributed.fsdp._common_utils import _get_param_to_fqns, _FSDPState
from torch.distributed.fsdp._optim_utils import (
    _rekey_sharded_optim_state_dict, _get_fqn_to_fsdp_param_info, _OptimStateKey, _PosDimTensorInfo, FSDPParamInfo,
    _flatten_optim_state
)

from composer.utils.misc import get_torch_version
if get_torch_version() >= version.parse("2.4"):
    from torch.distributed._state_dict_utils import _gather_state_dict
elif get_torch_version() >= version.parse("2.1.1"):
    from torch.distributed.fsdp._shard_utils import _gather_state_dict
else:
    raise ImportError("Versions of torch lower than 2.1.1 not supported")

from torch.utils._pytree import tree_map_only

import composer.utils.dist as cp_dist

def _fix_shard_orig_param_state(
    fsdp_param_info: FSDPParamInfo,
    fqn: str,
    optim_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Shard the optimizer state for the original parameter with the name ``fqn``.
    This API should only be used when ``use_orig_params`` is True.
    """
    if not optim_state:
        return {}
    fsdp_state = fsdp_param_info.state
    flat_param = fsdp_param_info.handle.flat_param
    param_idx = fsdp_param_info.param_indices[fqn]
    shard_param_info = flat_param._shard_param_infos[param_idx]  # type: ignore[attr-defined]
    optim_state = _gather_state_dict(optim_state, pg=fsdp_state.process_group)
    if not shard_param_info.in_shard:
        return {}
    # Flatten and shard the state.
    new_optim_state: Dict[str, Any] = {}
    intra_param_start_idx = shard_param_info.intra_param_start_idx
    intra_param_end_idx = shard_param_info.intra_param_end_idx
    for state_name, value in optim_state.items():
        if (
            torch.is_tensor(value)
            and value.dim() > 0
            and fsdp_state.sharding_strategy != ShardingStrategy.NO_SHARD
        ):
            value = value.flatten()[intra_param_start_idx : intra_param_end_idx + 1].clone()  # type: ignore[operator]
        new_optim_state[state_name] = value
    return new_optim_state

def _broadcast_processed_state_fix(
        fsdp_state: _FSDPState,
        optim_state: Dict[str, Any],
        group: Optional[dist.ProcessGroup],
        src: int = 0
) -> Dict[str, Any]:
    objects: List[Any] = [None]
    if fsdp_state.rank == 0:
        objects[0] = tree_map_only(
            torch.Tensor,
            lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(v.shape, v.dtype),
            optim_state,
        )
    dist.broadcast_object_list(objects, src=src, group=group)
    if fsdp_state.rank == 0:
        return optim_state
    else:
        return objects[0]


def _broadcast_state_fix(
        fsdp_state: _FSDPState, state: Any, group: Optional[dist.ProcessGroup], src: int = 0
) -> Any:
    device = _get_pg_default_device(group)
    if fsdp_state.rank == 0:
        if not isinstance(state, torch.Tensor) or state.dim() == 0:
            return state
        tensor = state.to(device)
    else:
        if isinstance(state, torch.Tensor):
            assert state.dim() == 0, (
                "For non-zero ranks, a tensor state should have zero dimension, "
                "but got the state with shape {state.shape()}."
            )
            return state
        elif not isinstance(state, _PosDimTensorInfo):
            return state
        tensor = torch.zeros(state.shape, dtype=state.dtype, device=device)
    handle = dist.broadcast(tensor, src=src, group=group, async_op=True)
    handle.wait()
    return tensor


def _flatten_optim_state_dict_fix(
        optim_state_dict: Dict[str, Any],
        model: nn.Module,
        use_orig_params: bool = False,
        optim: Optional[torch.optim.Optimizer] = None,
        rank0_only: bool = False,
        group: Optional[dist.ProcessGroup] = None,
        tp_zero_rank: int = 0
) -> Dict[str, Any]:
    """
    Flattens the full optimizer state dict, still keying by unflattened parameter
    names.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP know how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- it is managed by other parallelism and FSDP does not
    know ho to handle/aggregate them.

    Note that ``_flatten_tensor_optim_state`` does not need ``optim`` to
    flatten/shard the state. However, NamedOptimizer and KeyedOptimizer require
    all the states even if the corresponding parameters are empty. To this end,
    ``optim`` will be used to to get the initial state of the empty parameters.
    ``optim`` should only be non-None if the ``optim` is KeyedOptimizer or
    NamedOptimizer.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict.
    """
    unflat_osd = optim_state_dict
    if "state" not in unflat_osd and not rank0_only:
        raise ValueError(
            '`optim_state_dict` must have the keys "state"'
            "to be a valid optimizer state dict"
        )
    param_to_fqns = _get_param_to_fqns(model)
    fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)
    fsdp_state = next(iter(fqn_to_fsdp_param_info.values())).state

    # Broadcast unflat_osd without non-scalar tensor if rank0_only is True.
    if rank0_only:
        unflat_osd = _broadcast_processed_state_fix(fsdp_state, unflat_osd, group=group, src=tp_zero_rank)

    # Construct the "state" part
    flat_osd_state: Dict[Union[_OptimStateKey, str], Any] = {}
    unflat_osd_state = unflat_osd["state"]
    all_state_keys = set(unflat_osd_state.keys())

    for param, fqns in param_to_fqns.items():
        fqn = fqns[0]
        if fqn not in unflat_osd_state:
            continue
        all_state_keys.difference_update(fqns)

        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name in unflat_osd_state[fqn].keys():
                    unflat_osd_state[fqn][state_name] = _broadcast_state_fix(fsdp_state,
                                                                             unflat_osd_state[fqn][state_name],
                                                                             group=group,
                                                                             src=tp_zero_rank)
            fqn = fqns[0]
        if fqn in fqn_to_fsdp_param_info:
            fsdp_param_info = fqn_to_fsdp_param_info[fqn]
            if use_orig_params:
                flat_state = _fix_shard_orig_param_state(fsdp_param_info, fqn, unflat_osd_state[fqn])
            else:
                flat_state = _flatten_optim_state(fsdp_param_info, unflat_osd_state, fqns)
            key = _OptimStateKey(tuple(fqns), True)
            # Only include non-empty states since as expected by
            # `torch.optim.Optimizer` s unless the optimizer is KeyedOptimizer
            # or NamedOptimizer.
            if flat_state:
                flat_osd_state[key] = flat_state
            elif use_orig_params:
                assert (
                        len(fqns) == 1
                ), f"use_orig_params is True but there are multiple FQNs, {fqns}."
                if optim is not None:  # NamedOptimizer or KeyedOptimizer case.
                    state = optim.state.get(param, None)  # type: ignore[call-overload]
                    if state is not None:
                        flat_osd_state[key] = copy.deepcopy(state)
                    else:
                        warnings.warn(
                            f"optim_state[{key}] is not on rank{fsdp_state.rank}."
                        )

            else:
                raise RuntimeError(
                    f"The state of {key} is empty. This should happen when "
                    "use_orig_params=True."
                )
        else:  # do not flatten non-FSDP parameters' states
            assert len(fqns) == 1
            key = _OptimStateKey(tuple(fqns), False)
            flat_osd_state[key] = copy.copy(unflat_osd_state[fqn])

        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name, param_state in list(unflat_osd_state[fqn].items()):
                    if fsdp_state.rank > 0:
                        # Deference the tensor so that PyTorch can collect the memory.
                        del unflat_osd_state[fqn][state_name]
                    else:
                        # Move the tensor in the original osd back to CPU to make the
                        # original osd unaffected.
                        unflat_osd_state[fqn][state_name] = unflat_osd_state[fqn][
                            state_name
                        ].cpu()

    # Handle user-defined state, states that are not associated with parameters.
    for key in all_state_keys:
        user_state = unflat_osd_state[key]
        if isinstance(user_state, torch.Tensor) and rank0_only and use_orig_params:
            user_state = _broadcast_state_fix(fsdp_state, user_state, group=group, src=tp_zero_rank)
        flat_osd_state[key] = copy.copy(user_state)

    # Construct the "param_groups" part -- copy as is since it will be
    # rekeyed later according to the target rank's optimizer
    # Only copy param_groups if it exists in unflat_osd
    if "param_groups" in unflat_osd:
        flat_osd_param_groups = copy.deepcopy(unflat_osd["param_groups"])
        return {"state": flat_osd_state, "param_groups": flat_osd_param_groups}
    else:
        return {"state": flat_osd_state}


def _optim_state_dict_to_load_impl_fixed(
        optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        full_state_dict: bool = True,
        rank0_only: bool = False,
        is_named_optimizer: bool = False,
        group: Optional[dist.ProcessGroup] = None,
        tp_zero_rank: int = 0,
) -> Dict[str, Any]:
    """
    Convert an optimizer state-dict so that it can be loaded into the optimizer associated with the FSDP model.

    This is the internal API that is used by all the load optim_state_dict implementations.
    Given model, optim, and the saved optim_state_dict, this API adds the FSDP
    internal information and internal sharding to the optim_state_dict.
    """
    if full_state_dict:
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input,
            optim,
        )
    else:
        using_optim_input = False
        assert optim_input is None and not rank0_only

    use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[0]._use_orig_params
    assert all(
        use_orig_params == m._use_orig_params
        for m in FullyShardedDataParallel.fsdp_modules(model)
    ), "Not all FSDP modules have the same _use_orig_params value"
    fsdp_rank = cp_dist.get_fsdp_group_rank()
    if (
        rank0_only
        and (
            fsdp_rank is None and dist.get_rank(group) > 0
            or fsdp_rank is not None and fsdp_rank > 0
        )
    ):
        optim_state_dict = {}
    sharded_osd = _flatten_optim_state_dict_fix(optim_state_dict, model=model, use_orig_params=use_orig_params,
                                                optim=(optim if is_named_optimizer else None), rank0_only=rank0_only,
                                                group=group, tp_zero_rank=tp_zero_rank)
    return _rekey_sharded_optim_state_dict(
        sharded_osd,
        model=model,
        optim=optim,
        optim_input=optim_input,
        using_optim_input=using_optim_input,
        is_named_optimizer=is_named_optimizer,
    )
