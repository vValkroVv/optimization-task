#!/usr/bin/env python3
import os
import sys
import json
import copy
import re
import torch

from pathlib import Path
from itertools import accumulate
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Tuple, OrderedDict as OD
from concurrent.futures import ThreadPoolExecutor, as_completed

from logger import time_memory_stat, logger

from utils.dist import _get_rank_world, _workplan_from_state_by_bundles
from utils.safetensors_local import _save_chunk_as_safetensors, _write_local_index
from utils.common_utils import copy_dict_excluding_keys
from utils.hf_assembly_manifest import has_hf_assembly_manifest

try:
    import giga_fsdp  # noqa: F401
except ModuleNotFoundError:
    raise ModuleNotFoundError("giga_fsdp is not installed. Add it to your PYTHONPATH.")


BUFFER_PATTERNS: Tuple[str, ...] = ("._gate.expert_bias", "convffn.conv.bn.running_mean", "convffn.conv.bn.running_var", "convffn.conv.bn.num_batches_tracked")
SHARDED_BUFFER_PATTERNS: Tuple[str, ...] = ("convffn.conv.bn.running_mean", "convffn.conv.bn.running_var")


def clean_param_name(param_name : str) -> str:
    return param_name


@time_memory_stat
def process_layer(args,
                  flatten_tensors,
                  param_name_suffix=""
                  ):
    shard_size, param_shapes, param_names, offset = args
    num_ranks = len(flatten_tensors)

    flatten_layer = torch.cat(
        [flatten_tensors[rank].narrow(0, offset, shard_size) for rank in range(num_ranks)]
    )

    layer_state_dict = OrderedDict()
    param_offset = 0
    for param_shape, param_name in zip(param_shapes, param_names):
        param_name = clean_param_name(param_name)
        param_shape = torch.tensor(param_shape)
        param_numel = param_shape.prod()
        unshard_param = flatten_layer[param_offset : param_offset + param_numel].view(*param_shape)

        layer_state_dict[f"model.{param_name}{param_name_suffix}"] = unshard_param
        param_offset += param_numel

    return layer_state_dict


def unshard_flat_tensor(meta,
                        flatten_tensors,
                        param_group='main',
                        param_name_suffix=""):

    shard_sizes = meta[f'{param_group}_parameter_shard_bounaries']
    param_shapes_list = meta[f'{param_group}_parameter_orig_shapes']
    param_names_list = meta[f'{param_group}_shards_fqns']

    offsets = [0] + list(accumulate(shard_sizes))
    offsets = offsets[:-1]

    unshard_state_dict = OrderedDict()
    futures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for task in zip(shard_sizes, param_shapes_list, param_names_list, offsets):
            futures.append(executor.submit(process_layer, task, flatten_tensors, param_name_suffix))

    for future in as_completed(futures):
        unshard_state_dict.update(future.result())

    return unshard_state_dict


def unshard_gigafsdp_model(state_dicts : List[OrderedDict], meta : Dict[str, Any], split_convertation : bool = True, select_names=None):
    try:
        from giga_fsdp import MetaInfoKeys, CHECKPOINT_VERSION
        is_legacy_checkpoint = MetaInfoKeys.CHECKPOINT_VERSION_FIELD not in meta or \
            meta[MetaInfoKeys.CHECKPOINT_VERSION_FIELD] != CHECKPOINT_VERSION
    except ImportError:
        # old version of giga_fsdp
        is_legacy_checkpoint = True
        logger.info("Using legacy unshard_gigafsdp_model")

    if is_legacy_checkpoint:
        return _legacy_unshard_gigafsdp_model(state_dicts, meta)
    else:
        return _unshard_gigafsdp_model(state_dicts, meta, split_convertation, select_names=select_names)


def _normalize_module_name(module_name: str) -> str:
    # only buffers can have _module and root_module substrings, remove them
    module_name = module_name.replace("._module", "")
    module_name = module_name.replace(".root_module", "")
    module_name = _strip_scmoe_gemm(module_name)
    return module_name


def _strip_scmoe_gemm(key: str) -> str:
    """`ScMoEBlock` wraps the experts in an extra `gemm` submodule, so the
    raw fqn looks like `model.X.block_sparse_moe.gemm.experts.weightN`.
    Collapse it back to the canonical mono-state-dict shape used by every
    downstream consumer (`_normalize_single_shard_moe_expert_weights`,
    `_DeepseekPerLayerTransformer`, `mapper_deepseek_moe_to_hf`,
    `giga_fsdp.utils_checkpoint.get_slice`): `block_sparse_moe.experts.*`.
    No-op for keys that do not contain the wrapper.
    """
    return key.replace(".block_sparse_moe.gemm.experts.", ".block_sparse_moe.experts.")


def _normalize_key_prefix(key: str) -> str:
    # remove ComposerModel level prefix
    # model.model.xxx -> model.xxx
    # TO DO: just remove prefix before first dot ?
    key = re.sub(r"^model\.model\.", "model.", key, count=1)
    return key


def check_buffer_params(layer_wise_buffers: List[torch.Tensor], 
                        layer_name: str, 
                        buffer_patterns: Tuple[str, ...] = BUFFER_PATTERNS,
                        sharded_buffer_patterns: Tuple[str, ...] = SHARDED_BUFFER_PATTERNS):
    v0 = layer_wise_buffers[0]
    if not (isinstance(v0, torch.Tensor) and layer_name.startswith("model.")
            and layer_name.endswith(buffer_patterns)):
        return None, None
    for v in layer_wise_buffers:
        if not layer_name.endswith(sharded_buffer_patterns) and not torch.equal(v, v0):
            raise RuntimeError(f"Buffer {layer_name} differs across FSDP ranks.")
    return _normalize_module_name(layer_name), v0


def _unshard_one_layer(
    layer_name: str,
    state_dicts: List[Dict[str, Any]],
    *,
    meta: Dict[str, Any],
    param_suffix_override: Optional[str] = None,
    split_convertation: bool = True
) -> Tuple[str, List[Dict[str, Any]]]:
    from giga_fsdp import EMA_SUFFIX

    layer_weights = [sd[layer_name] for sd in state_dicts]
    final_key, buf_val = check_buffer_params(layer_weights, layer_name)
    if final_key is not None:
        logger.info(f"Found buffer {final_key} in checkpoint. Add it to the state dict.")
        return "buffer", [{final_key: buf_val}]

    if EMA_SUFFIX and layer_name.endswith(EMA_SUFFIX):
        base_name = layer_name[: -len(EMA_SUFFIX)]
        param_suffix = EMA_SUFFIX
    else:
        base_name = layer_name
        param_suffix = ""

    if param_suffix_override is not None:
        param_suffix = param_suffix_override

    layer_sd_list = layerwise_unshard(
        layer_weights,
        meta,
        layer_name=base_name,
        param_suffix=param_suffix,
        # process all tp/ep/tp+ep ranks together, not separately
        split_convertation=split_convertation
    )
    layer_sd_list = [
        {_strip_scmoe_gemm(k): v for k, v in sd.items()}
        for sd in layer_sd_list
    ]
    return "params", layer_sd_list


@time_memory_stat
def _unshard_gigafsdp_model(
    state_dicts : List[OrderedDict], 
    meta : Dict[str, Any], 
    split_convertation : bool = True, 
    select_names=None
    ):
    # tp_size -> list(state_dict)
    unshard_state_dict = OrderedDict()
    layer_names = state_dicts[0].keys()

    if select_names is not None:
        layer_names = [n for n in layer_names if n in set(select_names)]

    for layer_name in layer_names: # can be parallelized by layer names
        if layer_name == "model.meta":
            continue

        kind, items = _unshard_one_layer(
            layer_name,
            state_dicts,
            meta=meta,
            split_convertation=split_convertation
        )

        # buffer in checkpoint (not weights)
        if kind == "buffer":
            unshard_state_dict.setdefault(1, [{}])
            if not isinstance(unshard_state_dict[1], list):
                unshard_state_dict[1] = [{}]
            elif not unshard_state_dict[1]:
                unshard_state_dict[1].append({})
            # FIX IT!!!
            unshard_state_dict[1][0].update(items[0])
            continue

        # kind == "params"
        layer_sd_list = items
        current_tp_size = len(layer_sd_list)
        if current_tp_size not in unshard_state_dict:
            unshard_state_dict[current_tp_size] = [dict() for _ in range(current_tp_size)]

        for tp_rank, layer_sd in enumerate(layer_sd_list):
            unshard_state_dict[current_tp_size][tp_rank].update(layer_sd)

    return unshard_state_dict


@time_memory_stat
def _legacy_unshard_gigafsdp_model(
    flatten_state_dicts : List[OD],
    meta : Dict[str, Any],
    param_groups : Optional[List[str]] = None
    ):
    logger.warning("Using _legacy_unshard_gigafsdp_model. This has limited support for vision buffers copy.")

    from giga_fsdp import EMA_SUFFIX

    if param_groups is None:
        param_groups = ['main', 'layernorm', 'freezed', "layernormfreezed"]

    unshard_state_dict = OrderedDict()

    for param_group in param_groups:
        if f'model._{param_group}_parameter' not in flatten_state_dicts[0]:
            continue
        flatten_tensors = [sd[f'model._{param_group}_parameter'] for sd in flatten_state_dicts]
        unshard_state_dict.update(unshard_flat_tensor(meta, flatten_tensors, param_group=param_group))
        if f'model._{param_group}_parameter{EMA_SUFFIX}' in flatten_state_dicts[0]:
            flatten_tensors_ema = [sd[f'model._{param_group}_parameter{EMA_SUFFIX}'] for sd in flatten_state_dicts]
            unshard_state_dict.update(unshard_flat_tensor(meta, flatten_tensors_ema, param_group=param_group, param_name_suffix=EMA_SUFFIX))

    first_shard = flatten_state_dicts[0]

    processed_keys = set()
    processed_keys.add("model.meta")

    for param_group in param_groups:
        processed_keys.add(f"model._{param_group}_parameter")
        processed_keys.add(f"model._{param_group}_parameter{EMA_SUFFIX}")

    for key, value in first_shard.items():
        if key.endswith(SHARDED_BUFFER_PATTERNS):
            raise RuntimeError(f"Sharded buffers are not supported in _legacy_unshard_gigafsdp_model. Found key: {key}")

        if key in processed_keys:
            continue

        if isinstance(value, torch.Tensor):
            unshard_state_dict[key] = value.cpu().clone()

    # tp=1 -> the only one state_dict
    return {1 : [unshard_state_dict]}


def legacy_unshard_gigafsdp_optimizer(flatten_opt_state_dicts : List[OD],
                                      meta : Dict[str, Any]):
    from giga_fsdp import MetaInfoKeys

    optimizer_name = list(flatten_opt_state_dicts[0].keys())[0]
    group_idx=0 #supports only 1 param_group
    unshard_opt_state_dict = copy.deepcopy(flatten_opt_state_dicts[0])
    unshard_opt_state_dict[optimizer_name]['param_groups'][group_idx]['params'] = []

    flatten_states = [sd[optimizer_name]['state'] for sd in flatten_opt_state_dicts]

    for i, param_group in meta[MetaInfoKeys.PARAMS_MAPPING]:
        flat_opt_moment = unshard_opt_state_dict[optimizer_name]['state'].pop(i)
        for moment_name in ['exp_avg', 'exp_avg_sq']:
            flatten_tensors = [state[i][moment_name] for state in flatten_states]
            unflat_opt_moment = unshard_flat_tensor(meta, flatten_tensors, param_group=param_group)
            for param_name in unflat_opt_moment:
                if param_name not in unshard_opt_state_dict[optimizer_name]['state']:
                    unshard_opt_state_dict[optimizer_name]['state'][param_name] = {'step' : flat_opt_moment['step']}
                unshard_opt_state_dict[optimizer_name]['state'][param_name][moment_name] = unflat_opt_moment[param_name]

        for names in meta[f'{param_group}_shards_fqns']:
            unshard_opt_state_dict[optimizer_name]['param_groups'][group_idx]['params'].extend(["model." + name for name in names])

    return unshard_opt_state_dict

@time_memory_stat
def layerwise_unshard(flatten_state_dicts : List[torch.Tensor],
                      meta : Dict[str, Any],
                      state_idx : Optional[int] = None,
                      layer_name : Optional[str] = None,
                      param_suffix : Optional[str] = "",
                      split_convertation : bool = True):
    from giga_fsdp import MetaInfoKeys
    from giga_fsdp.optimizer_utils import get_layer_meta_info_idx

    assert state_idx is None or layer_name is None, "Only one of state_idx or layer_name must be provided"

    if layer_name is None:
        layer_name = meta[MetaInfoKeys.OPTIMIZER_STATE_IDX_TO_LAYER_NAME][state_idx]
    layer_name = layer_name.lstrip("model.")

    meta_info_idx, param_group = get_layer_meta_info_idx(layer_name, meta)

    if f'{param_group}_parameter_pg_size' in meta and not split_convertation:
        _, tp_size = meta[f'{param_group}_parameter_pg_size'][meta_info_idx]
    else:
        _, tp_size = meta['fsdp_group_size'], 1

    unshard_state_dict = [dict() for _ in range(tp_size)]
    for tp_rank in range(tp_size):
        flatten_layer = torch.cat(flatten_state_dicts[tp_rank::tp_size])

        param_offset = 0
        for param_shape, param_numel, param_name in zip(meta[f'{param_group}_parameter_orig_shapes'][meta_info_idx],
                                                        meta[f'{param_group}_shards_numels'][meta_info_idx],
                                                        meta[f'{param_group}_shards_fqns'][meta_info_idx],
                                                        ):
            unshard_param = flatten_layer[param_offset : param_offset + param_numel].view(*param_shape)

            unshard_state_dict[tp_rank][f"model.{param_name}{param_suffix}"] = unshard_param
            param_offset += param_numel

    return unshard_state_dict


def _unshard_gigafsdp_optimizer(flatten_opt_state_dicts : List[OD],
                                meta : Dict[str, Any]):
    optimizer_name = list(flatten_opt_state_dicts[0].keys())[0]
    unshard_opt_state_dict = copy.deepcopy(flatten_opt_state_dicts[0])
    unshard_state = unshard_opt_state_dict[optimizer_name]['state']
    unshard_param_groups = unshard_opt_state_dict[optimizer_name]['param_groups']

    flatten_states = [sd[optimizer_name]['state'] for sd in flatten_opt_state_dicts]

    groups_count = len(unshard_param_groups)
    for group_idx in range(groups_count):
        state_idx_list = unshard_param_groups[group_idx]['params']
        unshard_param_groups[group_idx]['params'] = []
        for state_idx in state_idx_list:
            flat_opt_moment = unshard_state.pop(state_idx)
            for moment_name in ['exp_avg', 'exp_avg_sq']:
                moment_shards = [flatten_states[i][state_idx][moment_name] for i in range(len(flatten_states))]
                unflat_opt_moment = layerwise_unshard(moment_shards, meta, state_idx=state_idx)
                assert len(unflat_opt_moment) == 1, "Mixed tp optimizer convertation is not supported yet"
                unflat_opt_moment = unflat_opt_moment[0]

                for param_name in unflat_opt_moment:
                    if param_name not in unshard_state:
                        unshard_state[param_name] = {'step' : flat_opt_moment['step']}
                    unshard_state[param_name][moment_name] = unflat_opt_moment[param_name]
            unshard_param_groups[group_idx]['params'].extend(list(unflat_opt_moment.keys()))

    return unshard_opt_state_dict


@time_memory_stat
def unshard_gigafsdp_optimizer(flatten_opt_state_dicts : List[OD],
                               meta : Dict[str, Any]):
    from giga_fsdp import MetaInfoKeys

    if MetaInfoKeys.PARAMS_MAPPING in meta:
        return legacy_unshard_gigafsdp_optimizer(flatten_opt_state_dicts, meta)
    else:
        return _unshard_gigafsdp_optimizer(flatten_opt_state_dicts, meta)


@time_memory_stat
def _stream_core(
    *,
    state_dicts: Dict[str, Any],
    output_dir: str,
    chunk_layers: int = 10,
    args=None,
    mapper=None
):
    assert mapper is not None
    os.makedirs(output_dir, exist_ok=True)
    rank, _ = _get_rank_world(args)
    out_rank_dir = os.path.join(output_dir, f"rank{rank:02d}")
    os.makedirs(out_rank_dir, exist_ok=True)

    config = state_dicts['state']['integrations']['huggingface']['model']['config']['content']
    mapper.ensure_config(config, args)

    if rank == 0:
        # save composer state
        composer_state = copy_dict_excluding_keys(state_dicts, keys_to_ignore=['state.model', 'state.optimizers'])
        torch.save(composer_state, os.path.join(output_dir, "composer_state.pt"))

        from base_mappers import DeepseekV3Mapper

        if isinstance(mapper, DeepseekV3Mapper):
            try:
                mapper.write_export_manifest(config, args, output_dir)
                if has_hf_assembly_manifest(output_dir):
                    logger.info("[rank 0] Saved HF assembly manifest for post-merge checkpoint assembly")
            except Exception as e:
                logger.warning(f"[rank 0] Failed to save HF assembly manifest: {e}")
        else:
            try:
                mapper.save_meta(config, args, output_dir)
                logger.info("[rank 0] Saved config/tokenizer (and runtime code if applicable)")
            except Exception as e:
                logger.warning(f"[rank 0] Failed to save meta: {e}")

    work_plan = _workplan_from_state_by_bundles(state_dicts["state"]["model"], chunk_layers)
    if not work_plan:
        _write_local_index(out_rank_dir)
        return

    for chunk_idx, chunk_keys in work_plan:
        chunk_block = {_normalize_key_prefix(k): state_dicts['state']['model'][k] for k in chunk_keys}

        mapped = mapper.map(chunk_block)
        mapped.update(mapper.flush())

        shard_file = f"shard-r{rank:02d}-c{chunk_idx:05d}.safetensors"
        shard_path = os.path.join(out_rank_dir, shard_file)
        shard_size = _save_chunk_as_safetensors(mapped, shard_path)

        weight_map = {k: shard_file for k in mapped.keys()}
        with open(os.path.join(out_rank_dir, f"shard-r{rank:02d}-c{chunk_idx:05d}.weight_map.json"), "w") as fh:
            json.dump({"weight_map": weight_map, "total_size": shard_size}, fh)

        mapped.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_local_index(out_rank_dir)
