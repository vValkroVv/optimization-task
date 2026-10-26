from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Iterable, List, Optional, Union, cast, Tuple

import torch
import torch.distributed as dist
import torch.nn
import torch.optim
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel, FlatParameter
from torch.distributed.fsdp import _traversal_utils as traversal_utils
from torch.distributed.fsdp._common_utils import _get_param_to_fqns
from torch.distributed.fsdp._optim_utils import (_gather_orig_param_state, _get_flat_param_to_fqn,
                                                 _get_fqn_to_fsdp_param_info, _get_param_id_to_param_from_optim_input,
                                                 _get_param_key_to_param, _is_named_optimizer, _unflatten_optim_state,
                                                 _unflatten_param_groups, sorted_items, FSDPParamInfo, _OptimStateKey,
                                                 _check_missing_keys_on_rank)
from torch.distributed.fsdp._runtime_utils import (
    _reset_flat_param_grad_info_if_needed,
)

def fix_map_param_key_to_optim_keys(
    optim_state_dict: Dict[str, Any],
    group: Optional[dist.ProcessGroup],
    param_key_to_param: Dict[Union[int, str], nn.Parameter],
    param_to_fqns: Dict[nn.Parameter, List[str]],
    fqn_to_fsdp_param_info: Dict[str, FSDPParamInfo],
    merge_keys: bool = False,
    zero_rank: int = 0,
) -> Tuple[List[_OptimStateKey], Dict[_OptimStateKey, Union[int, str]]]:
    """
    Construct the local mapping between the ``_OptimStateKey`` and parameter keys
    and all the ``_OptimStateKey`` across ranks. If ``merge_keys`` is False, rank0
    must contain all the ``_OptimStateKey``, an exception will be raised otherwise.
    Note that ``merge_keys`` should equal to ``use_orig_params``.
    """
    rank = dist.get_rank(group)
    optim_state_key_to_param_key: Dict[_OptimStateKey, Union[int, str]] = {}  # local
    all_optim_state_keys: List[_OptimStateKey] = []

    for param_key, param in param_key_to_param.items():
        # Do not include parameters without state to avoid empty mappings
        # just like in normal `torch.optim.Optimizer.state_dict()`
        if param_key not in optim_state_dict["state"]:
            continue
        fqns = param_to_fqns[param]
        is_fsdp_managed = isinstance(param, FlatParameter)
        if is_fsdp_managed:
            assert fqns[0] in fqn_to_fsdp_param_info, (
                fqns[0],
                list(fqn_to_fsdp_param_info.keys()),
            )
        is_fsdp_managed = fqns[0] in fqn_to_fsdp_param_info
        optim_state_key = _OptimStateKey(
            unflat_param_names=tuple(fqns),
            is_fsdp_managed=is_fsdp_managed,
        )
        if rank == 0 or merge_keys:
            all_optim_state_keys.append(optim_state_key)
        optim_state_key_to_param_key[optim_state_key] = param_key

    if merge_keys:
        all_keys: List[List[_OptimStateKey]] = [
            [] for _ in range(dist.get_world_size(group))
        ]
        dist.all_gather_object(all_keys, all_optim_state_keys, group=group)
        merge_all_optim_state_keys = [
            key for local_keys in all_keys for key in local_keys
        ]
        all_optim_state_keys = sorted(set(merge_all_optim_state_keys))
    else:
        key_obj_list: List[Optional[List[_OptimStateKey]]] = (
            [all_optim_state_keys] if rank == 0 else [None]
        )
        dist.broadcast_object_list(key_obj_list, src=zero_rank, group=group)
        assert key_obj_list[0] is not None
        all_optim_state_keys = key_obj_list[0]
        _check_missing_keys_on_rank(
            all_optim_state_keys,
            optim_state_key_to_param_key,
            param_key_to_param,
            group,
        )

    return all_optim_state_keys, optim_state_key_to_param_key


def fix_full_optim_state_dict(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: Dict[str, Any],
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[nn.Parameter],
        ]
    ],
    rank0_only: bool,
    shard_state: bool,
    group: Optional[dist.ProcessGroup],
    using_optim_input: bool,
    use_orig_params: bool = False,
) -> Dict[str, Any]:
    """
    Consolidates the optimizer state and returns it as a :class:`dict`
    following the convention of :meth:`torch.optim.Optimizer.state_dict`,
    i.e. with keys ``"state"`` and ``"param_groups"``.
    The flat parameters in ``FSDP`` modules contained in ``model`` are mapped
    back to their unflattened parameters.

    Parameter keys are not well-defined. For a regular optimizer, the optimizer
    state_dict contains a mapping from parameter IDs to parameter states.
    Parameter IDs are the order of parameters in ``optim.param_groups()`` across
    all the groups. This API also allows user to pass ``optim_input`` for the
    mapping between parameters and parameter IDs. Using ``optim_input`` is being
    deprecated.

    If the optimizer is a ``NamedOptimizer``, the optimizer state_dict does not
    contain parameter IDs mapping but a mapping from parameter FQNs to parameter
    states. This API finds the mapping from FQNs to parameters if the optimizer
    is a ``NamedOptimizer``.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP knows how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- those are managed by other parallelisms and FSDP does not
    know how to handle/aggregate them.

    Args:
        model (nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance) whose parameters
            were passed into the optimizer ``optim``.
        optim (torch.optim.Optimizer): Optimizer for ``model`` 's
            parameters.
        rank0_only (bool): If ``True``, saves the populated :class:`dict`
            only on rank 0; if ``False``, saves it on all ranks. (Default:
            ``True``)
        shard_state (bool): If ``True``, shard and distribute all
            non-zero-dimension states.

    Returns:
        Dict[str, Any]: A :class:`dict` containing the optimizer state for
        ``model`` 's original unflattened parameters and including keys
        "state" and "param_groups" following the convention of
        :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=False``,
        then nonzero ranks return an empty :class:`dict`.
    """
    _reset_flat_param_grad_info_if_needed(traversal_utils._get_fsdp_handles(model))
    # SAVE DATA ONLY ON RANK 0 (GLOBAL OR TENSOR PARALLEL)
    to_save = not rank0_only or dist.get_rank(group) == 0 or shard_state

    group_global_ranks = list(range(dist.get_world_size(group)))
    dist.all_gather_object(group_global_ranks, dist.get_rank(), group=group)

    fsdp_osd: Dict[str, Any] = {"state": {}} if to_save else {}
    fsdp_osd_state: Dict[str, Any] = fsdp_osd["state"] if to_save else {}
    param_to_fqns = _get_param_to_fqns(model)
    flat_param_to_fqn = _get_flat_param_to_fqn(model)
    is_named_optimizer = _is_named_optimizer(optim_state_dict)

    param_key_to_param = cast(
        Dict[Union[int, str], nn.Parameter],
        (
            _get_param_id_to_param_from_optim_input(model, optim_input)
            if using_optim_input
            else _get_param_key_to_param(
                optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
            )
        ),
    )
    fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)

    all_optim_state_keys, optim_state_key_to_param_key = fix_map_param_key_to_optim_keys(
        optim_state_dict,
        group,
        param_key_to_param,
        param_to_fqns,
        fqn_to_fsdp_param_info,
        merge_keys=use_orig_params,
        zero_rank=group_global_ranks[0]
    )

    # Iterate in rank 0's flat parameter ID order to ensure aligned all-gathers
    # across ranks
    for optim_state_key in all_optim_state_keys:
        param_key: Union[str, int, None] = optim_state_key_to_param_key.get(
            optim_state_key, None
        )

        if param_key is None:
            assert use_orig_params, (
                "If use_orig_params is False, we must be able to find the "
                f"corresponding param id. {optim_state_key} {param_key}"
            )
            if not optim_state_key.is_fsdp_managed:
                continue

        if optim_state_key.is_fsdp_managed:
            # If there are multiple unflat_param_names (not use_orig_params),
            # they share the same FSDPParamInfo. So the first unflat_param_name
            # is sufficient to fetch the FSDPParamInfo.
            fqn = optim_state_key.unflat_param_names[0]
            fsdp_param_info = fqn_to_fsdp_param_info[fqn]
            if use_orig_params:
                state = (
                    {} if param_key is None else optim_state_dict["state"][param_key]
                )
                unflat_state = [
                    _gather_orig_param_state(
                        fsdp_param_info,
                        fqn,
                        state,
                        shard_state,
                    )
                ]
            else:
                unflat_state = _unflatten_optim_state(
                    fsdp_param_info,
                    optim_state_dict["state"][param_key],
                    to_save,
                    shard_state,
                )
            if to_save:
                assert len(unflat_state) == len(optim_state_key.unflat_param_names)
                for unflat_param_name, unflat_param_state in zip(
                    optim_state_key.unflat_param_names,
                    unflat_state,
                ):
                    fsdp_osd_state[unflat_param_name] = unflat_param_state
        elif to_save:
            assert len(optim_state_key.unflat_param_names) == 1
            unflat_param_name = optim_state_key.unflat_param_names[0]
            fsdp_osd_state[unflat_param_name] = copy.copy(
                optim_state_dict["state"][param_key]
            )
            for state_name, value in sorted_items(fsdp_osd_state[unflat_param_name]):
                if torch.is_tensor(value):
                    fsdp_osd_state[unflat_param_name][state_name] = value.cpu()

    # At this point, communication is complete and ranks can return early if nothing
    # will be saved on that rank.
    if not to_save:
        return fsdp_osd

    flat_param_fqns = set(flat_param_to_fqn.values())
    for key, value in optim_state_dict["state"].items():
        if key in fsdp_osd_state:
            continue
        if key in flat_param_fqns:
            continue
        if key in param_key_to_param:
            continue
        # This key is not recognized by FSDP. It may be a user-defined state
        # or some parameters state that FSDP is unable to map from
        # ``optim.param_groups``.
        warnings.warn(
            f"Found a optim state, {key}, that FSDP cannot process. FSDP "
            "will directly copy everything to the returned state_dict. In "
            "most cases, this is a user-defined state that is not "
            "associated with any particular parameter. Another possible "
            "case is this state is managed by TorchRec. Otherwise, there may "
            " be a mismatched assumption of optim_state_dict of this mode."
        )
        fsdp_osd_state[key] = value

    if "param_groups" in optim_state_dict:
        fsdp_osd["param_groups"] = _unflatten_param_groups(
            optim_state_dict, param_key_to_param, param_to_fqns
        )

    return fsdp_osd

def fix_full_optim_state_dict_impl(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: Dict[str, Any],
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        rank0_only: bool = True,
        full_state_dict: bool = True,
        group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    The internal API that is used by all the optim_state_dict implementations.
    Given model, optim, the original optim_state_dict, this API removes the
    FSDP internal information and internal sharding from the optim_state_dict.
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

    use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[
        0
    ]._use_orig_params
    assert all(
        use_orig_params == m._use_orig_params
        for m in FullyShardedDataParallel.fsdp_modules(model)
    ), "Not all FSDP modules have the same _use_orig_params value"

    return fix_full_optim_state_dict(
        model=model,
        optim=optim,
        optim_state_dict=optim_state_dict,
        optim_input=optim_input,
        rank0_only=rank0_only,
        shard_state=not full_state_dict,
        group=group,
        using_optim_input=using_optim_input,
        use_orig_params=use_orig_params,
    )
