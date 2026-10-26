# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for running distributed data parallel training."""

import re
import collections
from contextlib import contextmanager, ExitStack, ContextDecorator
import functools
import logging
import warnings
from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Generator, ContextManager, Dict, Iterator, List, Optional, Sequence, Tuple, Union, cast

import torch
from packaging import version
from torchmetrics import Metric, MetricCollection
from torch.nn.modules.loss import _Loss

from composer.core import Precision, State
from composer.devices import Device
from composer.trainer.meta_safe_apply import meta_safe_apply
from composer.trainer.mosaic_fsdp import patch_pytorch
from composer.trainer.mosaic_fsdp_utils import BACKWARD_PREFETCH_MAP, SHARDING_MAP, get_cpu_offload, get_mixed_precision
from composer.utils import StringEnum, dist, ensure_tuple, reproducibility
from composer.trainer.sac_utils import apply_ac
from composer.optim.utils import IMixedPrecisionEnabled, IProcessGroupEnabled

__all__ = ['prepare_fsdp_module']

log = logging.getLogger(__name__)

process_group_cache = {}

TORCH_VERSION = re.match(r"^(\d+\.\d+\.\d+(\.[a-z]+)?)", torch.__version__).group()


def set_fsdp_default(fsdp_config: Dict[str, Any]):
    """Modify fsdp_config to set default values for missing keys."""
    fsdp_config.setdefault('activation_checkpointing', False)
    fsdp_config.setdefault('activation_checkpointing_reentrant', True)
    fsdp_config.setdefault('activation_cpu_offload', False)
    fsdp_config.setdefault('backward_prefetch', 'BACKWARD_POST')
    fsdp_config.setdefault('backward_prefetch_limit', 1)
    fsdp_config.setdefault('cpu_offload', False)
    fsdp_config.setdefault('forward_prefetch', False)
    fsdp_config.setdefault('forward_prefetch_limit', 1)
    fsdp_config.setdefault('ignored_modules', None)
    fsdp_config.setdefault('keep_low_precision_grads', False)
    fsdp_config.setdefault('limit_all_gathers', True)
    fsdp_config.setdefault('load_monolith_rank0_only', False)
    fsdp_config.setdefault('load_planner', None)
    fsdp_config.setdefault('mixed_precision', 'DEFAULT')
    fsdp_config.setdefault('save_planner', None)
    fsdp_config.setdefault('sharded_ckpt_prefix_dir', 'ep{epoch}-ba{batch}-tp_rank{tp_rank}-ep_rank{ep_rank}')
    fsdp_config.setdefault('sharding_strategy', 'FULL_SHARD')
    fsdp_config.setdefault('state_dict_type', 'full')
    fsdp_config.setdefault('sync_module_states', False)
    fsdp_config.setdefault('use_orig_params', True)
    fsdp_config.setdefault('verbose', False)
    #gigafsdp fields
    fsdp_config.setdefault('gigafsdp_enabled', False)
    fsdp_config.setdefault('gigafsdp_wrappers', dict())
    fsdp_config.setdefault('activation_checkpointing_auto_wrap_policy', None)
    fsdp_config.setdefault('giga_fsdp_modules_to_wrap_with_names', None)
    fsdp_config.setdefault('giga_fsdp_rogue_layer_norm_modules_with_names', None)
    fsdp_config.setdefault('giga_fsdp_layer_norm_module_cls', None)
    fsdp_config.setdefault('num_layers_to_checkpoint', None)
    fsdp_config.setdefault('hpz_first_layers_num', 0)
    fsdp_config.setdefault('hpz_group_size', None)
    fsdp_config.setdefault('hybrid_group_size', None)
    fsdp_config.setdefault('save_gigafsdp_sharded_folder_path', None)
    fsdp_config.setdefault('all_reduce_grads_across_model_parallel_group', False)
    fsdp_config.setdefault('prefetch_enable', True)
    fsdp_config.setdefault('zero_stage', 3)
    fsdp_config.setdefault('separate_layer_norm_disabled', False)
    fsdp_config.setdefault('sac_strategy', 0)
    fsdp_config.setdefault('num_non_offload_slots', 1)
    fsdp_config.setdefault('offload_inputs_modules', None)
    fsdp_config.setdefault('float8_all_gather_on_experts', False)
    fsdp_config.setdefault('float8_all_gather_skip_patterns', [])
    fsdp_config.setdefault('alignment_value', None)

    return fsdp_config


def _recreate_fsdp_param_groups_from_unwrapped_opt_info(
        fsdp_wrapped_named_params: Iterator[Tuple[str,
                                                  torch.nn.Parameter]],
        non_wrapped_param_names_to_group_num: Dict[str, int],
        group_num_to_optimizer_info: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Helper function to recreate optimizer groups for FSDP wrapped modules.

    Optimizer param groups are formatted as:
    [
        {'params': [p1, p2], 'lr' : lr1}, # group 0
        {'params': [p3], 'lr' : lr2} # group 1
    ]
    ie. there are multiple parameters per group. Here, we track the group number in order to map
    multiple parameters to the same group

    Args:
        fsdp_wrapped_named_params: output of model.named_parameters() of an FSDP wrapped model
        non_wrapped_param_names_to_group_num: a Dict mapping from the original model param names
                                            to the param group number
        group_num_to_optimizer_info: stores info like lr, eps for each group

    Returns a list of param groups, referencing the fsdp parameters
    """

    from torch.distributed.fsdp._common_utils import clean_tensor_name

    # initialize an empty list of parameters for each optimizer group
    for group_num in group_num_to_optimizer_info.keys():
        group_num_to_optimizer_info[group_num]['params'] = []

    for fsdp_name, param in fsdp_wrapped_named_params:

        unwrapped_name = clean_tensor_name(fsdp_name)
        # need to have a 1:1 mapping between a fsdp param name and the non-wrapped vanilla param name
        retrieved_group_num = non_wrapped_param_names_to_group_num[unwrapped_name]
        group_num_to_optimizer_info[retrieved_group_num]['params'].append(param)

    # return sorted optimizer info groups
    return [group_num_to_optimizer_info[num] for num in sorted(group_num_to_optimizer_info.keys())]


def prepare_fsdp_module(
    model: torch.nn.Module,
    optimizers: Optional[Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]]],
    fsdp_config: Dict[str, Any],
    precision: Precision,
    device: Device,
    auto_microbatching: bool,
) -> None:
    """Prepare a module (assumed ComposerModel) and optimizer for use with :class:`torch.distributed.fsdp.FullyShardedDataParallel`.

    Args:
        model (torch.nn.Module): The model to wrap.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional): The optimizer for `model`, assumed to have a single param group := model.parameters().
        fsdp_config (Dict[str, Any]): The FSDP config.
        precision: (Precision): The precision being used by the Trainer, used to fill in defaults for FSDP `mixed_precision` settings.
        device (Device): The device being used by the Trainer.
        auto_microbatching (bool, optional): Whether or not auto microbatching is enabled.
    """

    from torch.distributed.fsdp._init_utils import _get_modules_to_materialize
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (CheckpointImpl,
                                                                             apply_activation_checkpointing,
                                                                             checkpoint_wrapper)
    from torch.distributed.fsdp import FullyShardedDataParallel

    patch_pytorch()

    set_fsdp_default(fsdp_config)

    # Check sync_module_states is True for mixed initialization
    if fsdp_config['sync_module_states'] == False:
        rank_on_meta = 1 if next(model.parameters()).device.type == 'meta' else 0
        all_ranks_meta = device.tensor_to_device(torch.tensor([rank_on_meta], dtype=torch.uint8))
        dist.all_reduce(all_ranks_meta, reduce_operation='MIN')
        any_ranks_meta = device.tensor_to_device(torch.tensor([rank_on_meta], dtype=torch.uint8))
        dist.all_reduce(any_ranks_meta, reduce_operation='MAX')
        if all_ranks_meta.item() == 0 and any_ranks_meta.item() == 1:
            raise ValueError('Detected mixed initialization where some ranks have model on cpu or '
                             'gpu and some ranks are on meta. Either keep all ranks on the same '
                             "device or set fsdp_config['sync_module_states'] = True. Otherwise, "
                             'some weights may be randomly initialized when loading a checkpoint.')

    # Check if other ranks OOMed after forward/backward pass when using auto microbatching. This
    # may happen when close to memory limit or with uneven memory usage across ranks. Since we
    # need to do this before the model weights are gathered for the next FSDP block, we wrap every
    # FSPD block with a hook that checks if any other rank OOMed.
    def sync_hook(*args):
        # Check if any other rank hit an OOM
        found_cuda_oom_tensor = device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
        dist.all_reduce(found_cuda_oom_tensor, reduce_operation='MAX')
        found_cuda_oom = found_cuda_oom_tensor.item()
        # Signal current rank is still in batch
        all_ranks_finished_tensor = device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
        dist.all_reduce(all_ranks_finished_tensor, reduce_operation='MIN')

        if found_cuda_oom == 1:
            raise RuntimeError('CUDA out of memory encountered on a different rank')

    kwargs = {}
    # Support of new parameter `use_orig_params` in PyTorch 2.0 or higher.
    # Setting this to `True` has FSDP use `module`'s original parameters via method
    # `nn.Module.named_parameters` instead of FSDP's internal class `FlatParameter`. However,
    # setting it to `False` exposes FSDP's internal class `FlatParameter` via method
    # `nn.Module.named_parameters`.
    # Setting it to `True` is mandatory when using `torch.compile()`.
    kwargs['use_orig_params'] = fsdp_config['use_orig_params']

    # necessary variables for optimizers with multiple param groups in FSDP
    num_param_groups = None
    param_name_to_group_num = None
    group_num_to_param_group_info = None
    optimizer_specific_info = {}
    if optimizers:
        optimizers_tuple = ensure_tuple(optimizers)
        if len(optimizers_tuple) != 1:
            raise NotImplementedError(f'Only one optimizer is supported; found {len(optimizers_tuple)} optimizers')

        # clearing optimizer param groups and state
        # that will be recreated at the end of prepare_fsdp_module
        optim = optimizers_tuple[0]
        num_param_groups = len(optim.param_groups)
        if num_param_groups > 1:
            if not fsdp_config['gigafsdp_enabled'] and not kwargs['use_orig_params']:
                raise RuntimeError('Multiple optimizer groups with FSDP are only supported on torch 2.0 \
                                   with use_orig_params=True.')
            # optimizer.param_groups do not contain parameter names which are needed
            # to keep track of the different parameters in each group
            # so we use the pointers between model.parameters() and model.named_parameters()
            # to get the names of the parameters within optimizer.param_groups
            param_pointer_to_param_name = {id(p): n for n, p in model.named_parameters()}
            param_name_to_group_num = {}
            group_num_to_param_group_info = {}
            for group_num in range(len(optim.param_groups)):
                # Need to in-line to avoid a reference which causes FSDP to allocate extra GPU memory
                # group = optim.param_groups[group_num]
                for param_num in range(len(optim.param_groups[group_num]['params'])):
                    # Need to in-line to avoid a reference which causes FSDP to allocate extra GPU memory
                    # param = optim.param_groups[group_num]['params'][param_num]
                    param_name_to_group_num[param_pointer_to_param_name[id(
                        optim.param_groups[group_num]['params'][param_num])]] = group_num

                # this includes optimizer-specific values like lr, eps
                # this will be used as the kwargs for the optim param groups later
                optimizer_specific_group_info = {
                    k: v for k, v in optim.param_groups[group_num].items() if k != 'params'
                }
                group_num_to_param_group_info[group_num] = optimizer_specific_group_info
        else:
            optimizer_specific_info = {k: v for k, v in optim.param_groups[0].items() if k != 'params'}

        optim.param_groups.clear()
        optim.state.clear()

    sharding_map_key = fsdp_config['sharding_strategy'].upper()
    sharding_strategy = SHARDING_MAP[sharding_map_key]

    cpu_offload = get_cpu_offload(cpu_offload=fsdp_config['cpu_offload'])

    mixed_precision = fsdp_config['mixed_precision']
    keep_low_precision_grads = fsdp_config['keep_low_precision_grads']
    mixed_precision, param_dtype, _, _ = get_mixed_precision(precision,
                                                             mixed_precision=mixed_precision,
                                                             keep_low_precision_grads=keep_low_precision_grads)

    # Note: FSDP does support the use of torch.float32 with sharding.
    # They just never expected a user to pass in torch.float32 into mixed_precision as a param_dtype.
    # See: https://github.com/pytorch/pytorch/issues/90584
    # The PR fixing this bug is merged into PyTorch, but it hasn't made its way into a release yet.
    # Instead a user needs to pass in `None` as param_dtype to have the parameters as torch.float32.
    # TODO: remove these checks when PyTorch has a release that includes the fix.
    if sharding_map_key != 'NO_SHARD':
        if (precision == Precision.AMP_FP16 and param_dtype not in [torch.float16, None] or
                precision == Precision.AMP_BF16 and param_dtype not in [torch.bfloat16, None]):
            raise ValueError(
                f'FSDP in PyTorch 1.13 does not support precision `{precision}` with sharding strategy `{sharding_strategy}` '
                f'and param_dtype `{param_dtype}.` Consider using one of the predefined mixed_precision strategies '
                "(choose: `'FULL'`, `'DEFAULT'`, `'PURE'`)")

        if param_dtype == torch.float32:
            raise ValueError(
                f'FSDP in PyTorch 1.13 does not support param_dtype `{param_dtype}` with sharding_strategy `{sharding_map_key}` '
                f'Consider using `amp` or `bf16` for precision or setting param_dtype in mixed_precision to `None` '
                f'with sharding strategy `{sharding_map_key}.`')

    backward_prefetch = BACKWARD_PREFETCH_MAP[fsdp_config["backward_prefetch"].upper()]
    activation_checkpointing = fsdp_config["activation_checkpointing"]
    activation_cpu_offload = fsdp_config["activation_cpu_offload"]
    sync_module_states = fsdp_config["sync_module_states"]
    forward_prefetch = fsdp_config["forward_prefetch"]
    limit_all_gathers = fsdp_config["limit_all_gathers"]
    ignored_modules = fsdp_config["ignored_modules"]
    state_dict_type = fsdp_config["state_dict_type"]
    activation_checkpointing_reentrant = fsdp_config["activation_checkpointing_reentrant"]
    sharded_ckpt_prefix_dir = fsdp_config["sharded_ckpt_prefix_dir"]

    # enables process group for Tensor Paralell
    # ---
    # NOTE (Sbr): We get the dp group here, so that we then pass it to FSDP() wrapper call.
    tp_size = dist.get_tp_group_size()
    fsdp_group = dist.get_fsdp_group()
    if tp_size is not None:
        assert fsdp_group is not None
        assert (
            tp_size * fsdp_group.size() == dist.get_world_size()
        ), f"tp_size = {tp_size}, dp_group_size = {fsdp_group.size()}, world_size = {dist.get_world_size()}"
    
    if activation_cpu_offload:
        _activation_cpu_offload_supported_models = frozenset(
            {"ComposerGigaMixCausalLM", "ComposerGigarCausalLM", "ComposerDPOModel", "ComposerGigaOmniForCausalLM"}
        )
        model_cls_name = model.__class__.__name__
        assert model_cls_name in _activation_cpu_offload_supported_models, (
            "activation_cpu_offload is only implemented for "
            f"{sorted(_activation_cpu_offload_supported_models)}; got model class {model_cls_name!r}."
        )
    # We choose to not wrap the ComposerModel directly, but instead wrap any submodules like `ComposerModel.model`
    # This makes it safer to call ComposerModel-specific functions like 'eval_forward' that
    # may make calls to sharded submodules. If we only wrap the submodules, then any call that ComposerModel makes
    # to a FSDP-wrapped submodule's `forward()` function will be safe and all-gather the necessary weights before `forward()`.
    for obj_name, obj in model.named_children():
        if not isinstance(obj, (Metric, MetricCollection, _Loss)):

            # Skip wrapping submodules which are explicitly marked with no wrap
            if hasattr(obj, '_fsdp_wrap') and not bool(obj._fsdp_wrap):
                continue

            if version.parse(TORCH_VERSION) <= version.parse('2.1.0'):
                raise RuntimeError("Torch version must be higher than 2.1.0")

            # A dictionary of all tied parameter pointers to (module, attr) tuples
            tied_pointers = {}

            # Goes through all modules finding which weights have the same pointers
            for mod in obj.modules():
                for attr_name, attr in mod.named_parameters(recurse=False):
                    ptr = id(attr)
                    mod_attr_list = tied_pointers.get(ptr, [])
                    mod_attr_list.append((mod, attr_name))
                    tied_pointers[ptr] = mod_attr_list

            # Dictionary mapping the source module to a list of (target module, source attr, target attr) tuples
            source_mod_to_mod_attr = {}
            for mod_attr_list in tied_pointers.values():
                # If there is only one module for this pointer, then there is no weight tying
                if len(mod_attr_list) == 1:
                    continue

                # Arbitrarily choose the first module as the source module
                first_mod, first_attr = mod_attr_list[0]
                source_mod_to_mod_attr[first_mod] = [
                    (target_mod, first_attr, dest_attr) for target_mod, dest_attr in mod_attr_list[1:]
                ]

            if len(source_mod_to_mod_attr) and fsdp_config['gigafsdp_enabled']:
                raise RuntimeError(f"Weight tying is not supported with GigaFSDP. Tied weights: {source_mod_to_mod_attr}")

            # Clean up no longer needed module references for memory safety
            del tied_pointers

            def _param_init_fn(module: torch.nn.Module) -> None:
                # If we do not have any parameters or buffers on meta device managed by this module directly, we do not need to call the parameter init function.
                # It is assumed that whatever process moved the parameters off of meta device initialized them.
                # We expect this to occur if we have tied weights, as the second module will already have the weights initialized.
                is_meta = any(param.is_meta for param in module.parameters(recurse=False)) or any(
                    buffer.is_meta for buffer in module.buffers(recurse=False))
                has_init_rules = hasattr(module, '_init_rules') and isinstance(module._init_rules, Callable)

                if has_init_rules and not (hasattr(module, "reset_parameters") and isinstance(module.reset_parameters, Callable)):
                    raise ValueError(f"Module {module.__class__.__name__} has _init_rules but no callable reset_parameters method")

                # Note: if module has _init_rules, it handles its own init and we don't skip it
                if not is_meta and not has_init_rules:
                    return

                # Move all parameters and buffers to the current device.
                # if has_init_rules=true then module responsible for children's parameters init -> recurse=True
                module.to_empty(device=f'cuda:{torch.cuda.current_device()}', recurse=has_init_rules)

                # Redo weight tying, which will have been broken by the above line that moves parameters off of meta device
                if module in source_mod_to_mod_attr:
                    for target_mod, first_attr, dest_attr in source_mod_to_mod_attr[module]:
                        setattr(target_mod, dest_attr, getattr(module, first_attr))

                # Run the specified initialization
                if hasattr(obj, 'param_init_fn') and isinstance(obj.param_init_fn, Callable):
                    obj.param_init_fn(module)
                elif hasattr(module, 'reset_parameters') and isinstance(module.reset_parameters, Callable):
                    module.reset_parameters()
                else:
                    raise ValueError(
                        f'Object `{obj_name}` does not have a ``param_init_fn`` or a ``reset_parameters`` function. '
                        'This leaves parameters without initialization. Please add a ``param_init_fn`` or ``reset_parameters`` '
                        f'to module `{obj_name}`.')

            # CustomPolicy is only supported in torch v2.1.0-rc1 or higher
            from torch.distributed.fsdp.wrap import CustomPolicy  # type: ignore

            def lambda_fn(module: torch.nn.Module) -> Union[bool, dict]:
                ret = False
                if hasattr(module, '_fsdp_wrap'):
                    ret = bool(module._fsdp_wrap)
                elif hasattr(obj, 'fsdp_wrap_fn') and isinstance(obj.fsdp_wrap_fn, Callable):
                    ret = obj.fsdp_wrap_fn(module)
                    from composer.trainer.mosaic_fsdp_utils import _set_custom_fsdp_module_kwargs
                    if isinstance(ret, dict):
                        ret = _set_custom_fsdp_module_kwargs(ret, process_group_cache)
                if ret and auto_microbatching:
                    module.register_forward_hook(sync_hook)
                    module.register_full_backward_hook(sync_hook)
                return ret

            _auto_wrap_policy = CustomPolicy(lambda_fn)

            if not fsdp_config['gigafsdp_enabled']:
                fsdp_obj = FullyShardedDataParallel(
                    obj,
                    process_group=fsdp_group,
                    sharding_strategy=sharding_strategy,
                    auto_wrap_policy=_auto_wrap_policy,  # type: ignore FSDP type bug
                    cpu_offload=cpu_offload,
                    mixed_precision=mixed_precision,
                    backward_prefetch=backward_prefetch,
                    ignored_modules=ignored_modules,
                    param_init_fn=_param_init_fn,
                    device_id=torch.cuda.current_device(),
                    sync_module_states=sync_module_states,
                    forward_prefetch=forward_prefetch,
                    limit_all_gathers=limit_all_gathers,
                    **kwargs,
                )
            else:
                from giga_fsdp import GigaFSDP, ParamBufferPrefix, MetaInfoKeys
                wrapper_config: Dict[str, Any] = fsdp_config['gigafsdp_wrappers'][obj_name]
                fsdp_obj = GigaFSDP(
                    obj,
                    zero_stage=fsdp_config['zero_stage'],
                    fsdp_process_group=fsdp_group,
                    model_parallel_process_group=dist.get_tp_group(),
                    param_dtype=mixed_precision.param_dtype,
                    modules_to_wrap_with_names=wrapper_config['giga_fsdp_modules_to_wrap_with_names'],
                    special_process_group_fn=wrapper_config.get('special_process_group_fn', None),
                    gradient_accumulation_steps=fsdp_config['gradient_accumulation_steps'],
                    rogue_layer_norm_modules_with_names=wrapper_config['giga_fsdp_rogue_layer_norm_modules_with_names'],
                    sync_module_states=sync_module_states,
                    all_reduce_grads_across_model_parallel_group=fsdp_config['all_reduce_grads_across_model_parallel_group'],
                    param_init_fn=_param_init_fn,
                    hpz_first_layers_num=fsdp_config['hpz_first_layers_num'],
                    hpz_group_size=fsdp_config['hpz_group_size'],
                    hybrid_group_size=fsdp_config['hybrid_group_size'],
                    layer_norm_module_cls=wrapper_config['giga_fsdp_layer_norm_module_cls'],
                    device_id=torch.cuda.current_device(),
                    prefetch_enable=fsdp_config['prefetch_enable'],
                    float8_all_gather_on_experts=fsdp_config['float8_all_gather_on_experts'],
                    float8_all_gather_skip_patterns=fsdp_config['float8_all_gather_skip_patterns'],
                    alignment_value=fsdp_config['alignment_value'],
                )
                
                optimizer_specific_info.update(
                    {
                        k: v for k, v in fsdp_obj._meta_info.items() 
                        if k.startswith((
                           f"{ParamBufferPrefix.TRAINABLE}_",
                           f"{ParamBufferPrefix.SEPARATE}_",
                        ))
                    }
                )
                optimizer_specific_info["fqn_to_fsdp_group"] = fsdp_obj._meta_info.pop(MetaInfoKeys.PARAM_NAME_TO_FSDP_GROUP)
                optimizer_specific_info["communication_stream"] = fsdp_obj.communication_stream
                optimizer_specific_info["compute_stream"] = fsdp_obj.compute_stream


            if hasattr(fsdp_obj, '_exec_order_data'):
                if hasattr(fsdp_obj._exec_order_data, '_forward_prefetch_limit'):
                    fsdp_obj._exec_order_data._forward_prefetch_limit = fsdp_config['forward_prefetch_limit']
                else:
                    warnings.warn('FSDP._exec_order_data does not have attribute _forward_prefetch_limit '
                                  'which is unexpected and will result in `forward_prefetch_limit` from FSDP '
                                  'config being ignored. Please open an issue to Composer to report this.')
                if hasattr(fsdp_obj._exec_order_data, '_backward_prefetch_limit'):
                    fsdp_obj._exec_order_data._backward_prefetch_limit = fsdp_config['backward_prefetch_limit']
                else:
                    warnings.warn('FSDP._exec_order_data does not have attribute _backward_prefetch_limit '
                                  'which is unexpected and will result in `backward_prefetch_limit` from FSDP '
                                  'config being ignored. Please open an issue to Composer to report this.')
            else:
                warnings.warn('FSDP does not have attribute _exec_order_data which is unexpected and will '
                              'result in `forward_prefetch_limit` and `backward_prefetch_limit` from FSDP '
                              'config being ignored. Please open an issue to Composer to report this.')


            apply_ac(obj, fsdp_obj, obj_name, fsdp_config)
            setattr(model, obj_name, fsdp_obj)

    # Print FSDP wrapped model and FSDP config if `verbose=True`
    if fsdp_config['verbose']:
        print(f'FSDP: Wrapped Model:')
        print(model)
        print(f'FSDP: Using sharding_strategy={sharding_strategy}')
        print(f'FSDP: Using cpu_offload={cpu_offload}')
        print(f'FSDP: Using mixed_precision={mixed_precision}')
        print(f'FSDP: Using backward_prefetch={backward_prefetch}')
        print(f'FSDP: Using activation_checkpointing={activation_checkpointing}')
        print(f'FSDP: Using activation_cpu_offload={activation_cpu_offload}')
        print(f'FSDP: Using sync_module_states={sync_module_states}')
        print(f'FSDP: Using forward_prefetch={forward_prefetch}')
        print(f'FSDP: Using limit_all_gathers={limit_all_gathers}')
        print(f'FSDP: Using state_dict_type={state_dict_type}')
        print(f'FSDP: Using sharded_ckpt_prefix_dir={sharded_ckpt_prefix_dir}')


    # Rebuild optimizer now that parameters are sharded
    if optimizers:
        optimizers_tuple = ensure_tuple(optimizers)
        optim = optimizers_tuple[0]
        optim.param_groups.clear()

        assert num_param_groups is not None
        if num_param_groups > 1:
            optimizer_specific_info.pop("fqn_to_fsdp_group")
            optimizer_specific_info.pop("communication_stream")
            optimizer_specific_info.pop("compute_stream")

            assert param_name_to_group_num is not None
            assert group_num_to_param_group_info is not None

            if not fsdp_config['gigafsdp_enabled']:
                param_groups = _recreate_fsdp_param_groups_from_unwrapped_opt_info(model.named_parameters(),
                                                                                   param_name_to_group_num,
                                                                                   group_num_to_param_group_info)
            else:
                from giga_fsdp.optimizer_utils import recreate_giga_fsdp_param_groups
                param_groups = recreate_giga_fsdp_param_groups(model,
                                                               param_name_to_group_num,
                                                               group_num_to_param_group_info)

            for param_group in param_groups:
                optim.add_param_group(param_group)
        else:
            assert optimizer_specific_info is not None
            optimizer_specific_info.update({'params': list(filter(lambda param: param.requires_grad == True, model.parameters()))})

            if isinstance(optim, IMixedPrecisionEnabled):
                optim.mixed_precision = mixed_precision
            
            fsdp_groups_for_optim = optimizer_specific_info.pop("fqn_to_fsdp_group", None)
            communication_stream = optimizer_specific_info.pop("communication_stream", None)
            compute_stream = optimizer_specific_info.pop("compute_stream", None)
            
            if isinstance(optim, IProcessGroupEnabled) and fsdp_groups_for_optim is not None:
                for _param_fqn, _process_group in fsdp_groups_for_optim.items():
                    optim.set_fsdp_process_group(_param_fqn, _process_group)
                optim.communication_stream = communication_stream
                optim.compute_stream = compute_stream
                
            
            optim.add_param_group(optimizer_specific_info)

        # FSDP rebuilt the param groups with fresh (flattened) tensors, orphaning
        # any per-parameter base LR scales keyed by the pre-shard tensors. Re-resolve
        # the config `lr_scales` spec by parameter *name* against the now-sharded
        # params so the static base scale survives sharding (and every resume).
        if hasattr(optim, 'apply_base_lr_scales'):
            optim.apply_base_lr_scales(model.named_parameters())


@contextmanager
def joined_context_managers(*cms):
    with ExitStack() as stack:
        yield [stack.enter_context(cm) for cm in cms]
