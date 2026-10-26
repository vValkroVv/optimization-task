import torch
import copy
import os
import gc
import dill
import logging

from collections import OrderedDict
from typing import Dict, Any, Optional, List, Tuple, Callable, Set, Union

from .giga_fsdp import GigaFSDP, ParamBufferPrefix, MetaInfoKeys, EMA_SUFFIX
from .optimizer_utils import get_layer_meta_info_idx

log = logging.getLogger(__name__)

PARAM_BUFFER_NAMES = [p.value for p in ParamBufferPrefix]
OPTIMIZERS_FIELD = "optimizers"
DUMMY_VALUE = -100
REQUIRED_BUFFER_PATTERNS: Tuple[str, ...] = (
    "._gate.expert_bias",
)
OPTIONAL_BUFFER_PATTERNS: Tuple[str, ...] = (
    "convffn.conv.bn.running_mean",
    "convffn.conv.bn.running_var",
    "convffn.conv.bn.num_batches_tracked"
)

def clean_param_name(param_name):
    param_name = param_name.replace("._orig_mod", "")
    return param_name


# Parameter FQN aliases between interchangeable module implementations that keep
# identical weights under different attribute paths. Used only to match a checkpoint
# flat buffer against the live model layout; live FQNs are never rewritten.
# ``ScMoEBlock`` holds experts as ``block_sparse_moe.gemm.experts.weight{1,2}``,
# ``DeepseekGMMMoeBlock`` as ``block_sparse_moe.experts.weight{1,2}``.
FQN_ALIASES: Tuple[Tuple[str, str], ...] = (
    (".gemm.experts.", ".experts."),
)


def canonical_fqn(fqn: str) -> str:
    """Map an FQN onto its canonical spelling shared by aliased implementations."""
    for alias, canonical in FQN_ALIASES:
        fqn = fqn.replace(alias, canonical)
    return fqn


def _canonical_fqns(fqns: List[str]) -> List[str]:
    canonical = [canonical_fqn(fqn) for fqn in fqns]
    if len(set(canonical)) != len(canonical):
        # Aliasing must never collapse two distinct parameters onto one name.
        return list(fqns)
    return canonical

# note: copied from gigachat-utils/convertation/mapper_gigafsdp_to_monolith.py
def _normalize_module_name(module_name: str) -> str:
    # only buffers can have _module and root_module substrings, remove them
    module_name = module_name.replace("._module", "")
    module_name = module_name.replace(".root_module", "")
    return module_name


def add_prefix_in_state_dict(
    state_dict: Dict[str, Any], prefix: str
) -> None:
    r"""Add prefix in state_dict in place.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        newkey = prefix + key
        state_dict[newkey] = state_dict.pop(key)


def compare_state_dicts(dict1, dict2, path=""):
    """
    Recursively compare two state_dicts and print the differences.
    
    Args:
    - dict1: The first state_dict.
    - dict2: The second state_dict.
    - path: The current path in the state_dicts (used for recursion).
    """
    for key in dict1:
        if key not in dict2:
            print(f"Key {path + key} not found in the second state_dict.")
        else:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                compare_state_dicts(dict1[key], dict2[key], path + str(key) + ".")
            elif isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor):
                if not torch.equal(dict1[key], dict2[key]):
                    diff_norm = torch.norm(dict1[key] - dict2[key])
                    print(f"Difference found in {path + key}: norm of difference = {diff_norm}")
            else:
                try:
                    if dict1[key] != dict2[key]:
                        print(f"Difference found in {path + key}: {dict1[key]} != {dict2[key]}")
                except Exception as e:
                    print(f"Cant compare {key}: {e}")
    
    for key in dict2:
        if key not in dict1:
            print(f"Key {path + key} not found in the first state_dict.")


def copy_state_dict(state_dict_object: Any) -> Any:
    if isinstance(state_dict_object, torch.Tensor):
        tensor_ = torch.zeros_like(state_dict_object).copy_(state_dict_object)
        return tensor_

    elif isinstance(state_dict_object, (dict, OrderedDict)):
        result = type(state_dict_object)()
        for key, value in state_dict_object.items():
            result[key] = copy_state_dict(value)
        return result

    elif isinstance(state_dict_object, list):
        return [copy_state_dict(item) for item in state_dict_object]

    elif isinstance(state_dict_object, tuple):
        return tuple(copy_state_dict(item) for item in state_dict_object)

    elif isinstance(state_dict_object, set):
        return {copy_state_dict(item) for item in state_dict_object}

    elif isinstance(state_dict_object, (str, int, float, bool, type(None))):
        return state_dict_object

    elif hasattr(state_dict_object, '__array__'):
        import numpy as np
        if isinstance(state_dict_object, np.ndarray):
            return copy.deepcopy(state_dict_object)
        return state_dict_object

    else:
        return state_dict_object


def get_mixed_model_state_dict(
    module: torch.nn.Module, 
    state_dict_type: str = 'full'
) -> Dict[str, Any]:
    state_dict = OrderedDict()
    for module_name, module in module.named_children():
        if isinstance(module, GigaFSDP):
            if module._meta_info['is_inference_mode']:
                continue
            module_sd = gigafsdp_get_model_state_dict(module, state_dict_type)    
        else:
            module_sd = module.state_dict()
        add_prefix_in_state_dict(module_sd, f"{module_name}.")
        state_dict.update(module_sd)

    return state_dict


def gigafsdp_get_model_state_dict(
    module: torch.nn.Module, 
    state_dict_type: str = 'full'
) -> Dict[str, Any]:
    if not isinstance(module, GigaFSDP):
        raise RuntimeError("'gigafsdp_get_model_state_dict' methdo only works with GigaFSDP modules!")

    if state_dict_type == 'full':
        return module.state_dict() #monolith, rank 0
    elif state_dict_type == 'local':
        return module.local_state_dict() #flatten, all ranks
    elif state_dict_type == 'sharded': #unflatten, allranks
        raise NotImplementedError("sharded checkpoint is not implemented with gigafsdp")


def _flat_buffer_layout(
    meta: Dict[str, Any], layer_name: str
) -> Tuple[List[str], List[int], List[Tuple[int, ...]], int]:
    if layer_name.endswith(EMA_SUFFIX):
        layer_name = layer_name[:-len(EMA_SUFFIX)]
    if not _is_flat_buffer_name(layer_name):
        raise KeyError(f"Not a GigaFSDP flat buffer: {layer_name}")
    meta_idx, prefix = get_layer_meta_info_idx(layer_name, meta)
    prefix = prefix.value
    return (
        list(meta[f"{prefix}_shards_fqns"][meta_idx]),
        list(meta[f"{prefix}_shards_numels"][meta_idx]),
        [tuple(shape) for shape in meta[f"{prefix}_parameter_orig_shapes"][meta_idx]],
        int(meta[f"{prefix}_parameter_padded_numels"][meta_idx]),
    )


def _is_flat_buffer_name(layer_name: str) -> bool:
    if layer_name.endswith(EMA_SUFFIX):
        layer_name = layer_name[:-len(EMA_SUFFIX)]
    return layer_name.endswith(
        (
            ".flatten",
            ".flatten_freezed",
            "._layernorm_parameter",
            "._layernormfreezed_parameter",
        )
    )


def _flat_buffer_layouts_match(
    source_meta: Dict[str, Any],
    source_layer_name: str,
    target_meta: Dict[str, Any],
    target_layer_name: str,
) -> bool:
    try:
        source = _flat_buffer_layout(source_meta, source_layer_name)
        target = _flat_buffer_layout(target_meta, target_layer_name)
    except (KeyError, IndexError):
        return False
    if source == target:
        return True
    source_fqns, *source_rest = source
    target_fqns, *target_rest = target
    if source_rest != target_rest:
        return False
    if _canonical_fqns(source_fqns) != _canonical_fqns(target_fqns):
        return False
    log.warning(
        "Load gigafsdp model state dict: %s matches %s through FQN aliases %s",
        source_layer_name,
        target_layer_name,
        FQN_ALIASES,
    )
    return True


def _flat_buffer_process_group(
    module: GigaFSDP, layer_name: str
) -> torch.distributed.ProcessGroup:
    if layer_name.endswith(EMA_SUFFIX):
        layer_name = layer_name[:-len(EMA_SUFFIX)]
    for suffix in (".flatten_freezed", ".flatten"):
        if not layer_name.endswith(suffix):
            continue
        module_path = layer_name[:-len(suffix)]
        try:
            wrapped_module = module.get_submodule(module_path)
            return wrapped_module.pg_kit.fsdp_process_group
        except (AttributeError, KeyError) as error:
            raise RuntimeError(
                f"Cannot resolve GigaFSDP process group for {layer_name}"
            ) from error
    return module.pg_manager.default_pg_kit.fsdp_process_group


def _parameter_offsets(fqns: List[str], numels: List[int]) -> Dict[str, Tuple[int, int]]:
    offsets = {}
    cursor = 0
    for fqn, numel in zip(fqns, numels):
        offsets[fqn] = (cursor, numel)
        cursor += numel
    return offsets


def _mapped_pieces(
    mappings: List[Tuple[int, int, int]],
    source_rank: int,
    destination_rank: int,
    source_shard_size: int,
    target_shard_size: int,
) -> List[Tuple[int, int, int]]:
    """Return (source-local offset, length, target-global offset) pieces."""
    source_rank_start = source_rank * source_shard_size
    source_rank_end = source_rank_start + source_shard_size
    target_rank_start = destination_rank * target_shard_size
    target_rank_end = target_rank_start + target_shard_size
    pieces = []

    for source_start, target_start, length in mappings:
        source_begin = max(source_start, source_rank_start)
        source_end = min(source_start + length, source_rank_end)
        if source_begin >= source_end:
            continue

        mapped_begin = target_start + source_begin - source_start
        mapped_end = mapped_begin + source_end - source_begin
        target_begin = max(mapped_begin, target_rank_start)
        target_end = min(mapped_end, target_rank_end)
        if target_begin >= target_end:
            continue

        source_global = source_begin + target_begin - mapped_begin
        pieces.append(
            (
                source_global - source_rank_start,
                target_end - target_begin,
                target_begin,
            )
        )
    return pieces


@torch.no_grad()
def _copy_with_dummy_mask(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    source = source.to(device=target.device, dtype=target.dtype)
    dummy_mask = source == DUMMY_VALUE
    if dummy_mask.any():
        target.copy_(torch.where(dummy_mask, target, source))
    else:
        target.copy_(source)
    return dummy_mask


@torch.no_grad()
def _redistribute_flat_segments(
    source: torch.Tensor,
    target: torch.Tensor,
    mappings: List[Tuple[int, int, int]],
    source_padded_numel: int,
    target_padded_numel: int,
    process_group: torch.distributed.ProcessGroup,
) -> None:
    group_size = torch.distributed.get_world_size(process_group)
    group_rank = torch.distributed.get_rank(process_group)
    if source_padded_numel % group_size or target_padded_numel % group_size:
        raise RuntimeError("GigaFSDP transition buffers are not divisible by the FSDP group size")

    source_shard_size = source_padded_numel // group_size
    target_shard_size = target_padded_numel // group_size
    if source.numel() != source_shard_size or target.numel() != target_shard_size:
        raise RuntimeError(
            "GigaFSDP transition shard size mismatch: "
            f"source={source.numel()}/{source_shard_size}, "
            f"target={target.numel()}/{target_shard_size}"
        )

    source = source.to(device=target.device)
    send_pieces = [
        _mapped_pieces(
            mappings,
            group_rank,
            destination,
            source_shard_size,
            target_shard_size,
        )
        for destination in range(group_size)
    ]
    send_splits = [sum(length for _, length, _ in pieces) for pieces in send_pieces]
    send_chunks = [
        source.narrow(0, source_offset, length)
        for pieces in send_pieces
        for source_offset, length, _ in pieces
    ]
    send = torch.cat(send_chunks) if send_chunks else source.new_empty(0)

    recv_pieces = [
        _mapped_pieces(
            mappings,
            source_rank,
            group_rank,
            source_shard_size,
            target_shard_size,
        )
        for source_rank in range(group_size)
    ]
    recv_splits = [sum(length for _, length, _ in pieces) for pieces in recv_pieces]
    recv = source.new_empty(sum(recv_splits))
    if group_size == 1:
        recv.copy_(send)
    else:
        torch.distributed.all_to_all_single(
            recv,
            send,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=process_group,
        )

    recv_offset = 0
    target_rank_start = group_rank * target_shard_size
    for pieces in recv_pieces:
        for _, length, target_global_offset in pieces:
            target_offset = target_global_offset - target_rank_start
            _copy_with_dummy_mask(
                target.narrow(0, target_offset, length),
                recv.narrow(0, recv_offset, length),
            )
            recv_offset += length


def _remap_flat_buffer(
    module_name: str,
    module: GigaFSDP,
    target_layer_name: str,
    target: torch.Tensor,
    model_state_dict: OrderedDict,
) -> Optional[List[str]]:
    source_meta = model_state_dict.get(f"{module_name}.meta")
    target_meta = module._meta_info
    if not isinstance(source_meta, dict):
        return None

    try:
        target_fqns, target_numels, _, target_padded_numel = _flat_buffer_layout(
            target_meta, target_layer_name
        )
    except (KeyError, IndexError):
        return None

    source_param_layers = source_meta.get(MetaInfoKeys.PARAM_NAME_TO_LAYER_NAME, {})
    # Target FQN -> source FQN, exact name first, then through FQN_ALIASES. The
    # source spelling addresses source offsets, the target spelling the target buffer.
    source_by_canonical: Dict[str, Optional[str]] = {}
    for source_fqn in source_param_layers:
        key = canonical_fqn(source_fqn)
        source_by_canonical[key] = (
            None if key in source_by_canonical else source_fqn
        )
    source_fqn_of: Dict[str, str] = {}
    for fqn in target_fqns:
        if fqn in source_param_layers:
            source_fqn_of[fqn] = fqn
            continue
        aliased = source_by_canonical.get(canonical_fqn(fqn))
        if aliased is not None:
            source_fqn_of[fqn] = aliased
    shared_target_fqns = [fqn for fqn in target_fqns if fqn in source_fqn_of]
    if not shared_target_fqns:
        return None

    def source_layer_of(fqn: str) -> str:
        return source_param_layers[source_fqn_of[fqn]]

    target_offsets = _parameter_offsets(target_fqns, target_numels)
    source_layers = sorted(
        {source_layer_of(fqn) for fqn in shared_target_fqns},
        key=lambda layer: min(
            target_offsets[fqn][0]
            for fqn in shared_target_fqns
            if source_layer_of(fqn) == layer
        ),
    )
    source_layouts = {}
    source_fqn_set = set()
    for source_layer in source_layers:
        try:
            source_layout = _flat_buffer_layout(source_meta, source_layer)
        except (KeyError, IndexError):
            return None
        source_fqns, source_numels, _, _ = source_layout
        source_fqn_set.update(source_fqns)
        source_offsets = _parameter_offsets(source_fqns, source_numels)
        for fqn in shared_target_fqns:
            if source_layer_of(fqn) != source_layer:
                continue
            if source_offsets[source_fqn_of[fqn]][1] != target_offsets[fqn][1]:
                return None
        source_layouts[source_layer] = source_layout

    # An empty diff is a legal layout transition: the stage-1 -> stage-2 merge
    # rebuilds one trainable buffer from the frozen-base and trainable-indexer
    # source buffers with the same combined FQN set. Only a diff containing
    # non-indexer parameters is unsafe (some base weight would silently keep
    # its initialization).
    changed_fqns = {canonical_fqn(fqn) for fqn in source_fqn_set}.symmetric_difference(
        canonical_fqn(fqn) for fqn in target_fqns
    )
    if any(".indexer." not in fqn for fqn in changed_fqns):
        return None

    process_group = _flat_buffer_process_group(module, target_layer_name)

    consumed = []
    for source_layer in source_layers:
        source_process_group = _flat_buffer_process_group(module, source_layer)
        if source_process_group is not process_group:
            raise RuntimeError(
                "GigaFSDP transition source and target buffers use different process groups: "
                f"source={source_layer}, target={target_layer_name}"
            )
        source_key = f"{module_name}.{source_layer}"
        source = model_state_dict.get(source_key)
        if not isinstance(source, torch.Tensor):
            return None
        source_fqns, source_numels, _, source_padded_numel = source_layouts[
            source_layer
        ]
        source_offsets = _parameter_offsets(source_fqns, source_numels)
        mappings = [
            (
                source_offsets[source_fqn_of[fqn]][0],
                target_offsets[fqn][0],
                target_offsets[fqn][1],
            )
            for fqn in shared_target_fqns
            if source_layer_of(fqn) == source_layer
        ]
        _redistribute_flat_segments(
            source,
            target,
            mappings,
            source_padded_numel,
            target_padded_numel,
            process_group,
        )
        consumed.append(source_key)

    log.warning(
        "Load gigafsdp model state dict: rebuilt %s from %s",
        f"{module_name}.{target_layer_name}",
        consumed,
    )
    return consumed


def _resolve_compatible_buffer_name(
    module_name: str,
    module: GigaFSDP,
    target_name: str,
    target: torch.Tensor,
    model_state_dict: OrderedDict,
) -> Optional[str]:
    """Resolve an identical flat-buffer layout across a trainable/frozen transition."""
    source_meta = model_state_dict.get(f"{module_name}.meta")
    target_meta = module._meta_info
    prefix = f"{module_name}."
    if not target_name.startswith(prefix):
        return None
    target_layer_name = target_name[len(prefix):]

    def is_compatible(source_name: str) -> bool:
        if not source_name.startswith(prefix):
            return False
        source = model_state_dict.get(source_name)
        if not isinstance(source, torch.Tensor) or source.shape != target.shape:
            return False
        source_layer_name = source_name[len(prefix):]
        is_flat = _is_flat_buffer_name(source_layer_name) or _is_flat_buffer_name(
            target_layer_name
        )
        if not is_flat:
            return True
        return isinstance(source_meta, dict) and _flat_buffer_layouts_match(
            source_meta,
            source_layer_name,
            target_meta,
            target_layer_name,
        )

    if is_compatible(target_name):
        return target_name

    suffix = EMA_SUFFIX if target_name.endswith(EMA_SUFFIX) else ""
    base_name = target_name[:-len(suffix)] if suffix else target_name
    if base_name.endswith(".flatten_freezed"):
        candidate = base_name[:-len("_freezed")] + suffix
    elif base_name.endswith(".flatten"):
        candidate = base_name + "_freezed" + suffix
    elif base_name.endswith("._layernormfreezed_parameter"):
        candidate = base_name.replace(
            "._layernormfreezed_parameter", "._layernorm_parameter"
        ) + suffix
    elif base_name.endswith("._layernorm_parameter"):
        candidate = base_name.replace(
            "._layernorm_parameter", "._layernormfreezed_parameter"
        ) + suffix
    else:
        return None

    if not is_compatible(candidate):
        return None
    log.warning(
        "Load gigafsdp model state dict: remap compatible buffer %s -> %s",
        candidate,
        target_name,
    )
    return candidate


@torch.no_grad() # type: ignore
def load_gigafsdp_model_state_dict(model: torch.nn.Module, model_state_dict: OrderedDict) -> Tuple[List[str], List[str]]:
    """ Upload gigafsdp sharded checkpoint. """
    model_keys = []
    missing_keys = []

    for module_name, module in model.named_children():
        if not isinstance(module, GigaFSDP):
            continue
        module_missing_start = len(missing_keys)
        module_layout_transition = False

        local_state_dict = module.local_state_dict()
        for param_name, param in local_state_dict.items():
            if "meta" in param_name:
                continue

            module_param_name = f"{module_name}.{param_name}"
            source_param_name = _resolve_compatible_buffer_name(
                module_name,
                module,
                module_param_name,
                param,
                model_state_dict,
            )
            if source_param_name is None and not module_param_name.endswith(EMA_SUFFIX):
                consumed = _remap_flat_buffer(
                    module_name,
                    module,
                    param_name,
                    param,
                    model_state_dict,
                )
                if consumed is not None:
                    model_keys.extend(consumed)
                    module_layout_transition = True
                    continue
            if source_param_name is None:
                log.warning(f"Load gigafsdp model state dict: {module_param_name} not in state_dict!")
                missing_keys.append(module_param_name)
                if not module_param_name.endswith(EMA_SUFFIX):
                    module_layout_transition = True
                continue

            model_keys.append(source_param_name)
            if source_param_name != module_param_name and not module_param_name.endswith(EMA_SUFFIX):
                module_layout_transition = True
            loaded_val = model_state_dict[source_param_name]
            if loaded_val is None:
                log.error(f"Load gigafsdp model state dict: {module_param_name} is None in state_dict, skipping!")
                missing_keys.append(module_param_name)
                continue
            param_mask = _copy_with_dummy_mask(param, loaded_val)
            if param_mask.any():
                log.info(
                    f"Load gigafsdp model state dict: {param_name} has non-trivial mask for missing weights, "
                    f"keep {param_mask.sum().item()} out of {param_mask.numel()} "
                    f"original weights from model"
                )

        source_has_ema = any(key.endswith(EMA_SUFFIX) for key in model_state_dict.keys())
        if module_layout_transition:
            # A stage transition changes the trainable EMA flat layout as well, while frozen
            # parameters have no EMA shard. Ignore partial source EMA buffers and let the EMA
            # callback initialize a complete target EMA from the remapped raw weights after load.
            missing_keys[module_missing_start:] = [
                key
                for key in missing_keys[module_missing_start:]
                if not key.endswith(EMA_SUFFIX)
            ]
            model_keys.extend(
                key
                for key, value in model_state_dict.items()
                if isinstance(value, torch.Tensor)
                and key.startswith(f"{module_name}.")
                and key.endswith(EMA_SUFFIX)
            )
        module.has_ema_flag = source_has_ema and not module_layout_transition
        module.copy_master_weight_to_bit16_buffer()

    gc.collect()
    torch.cuda.empty_cache()

    model_state_dict_filtered = set([key for key, value in model_state_dict.items()
                                     if isinstance(value, torch.Tensor)])
    unexpected_keys = list(set(model_state_dict_filtered) - set(model_keys))
    return missing_keys, unexpected_keys


def update_gigafsdp_dict(param_shapes: List[Tuple[int, ...]],
                         param_names: List[str],
                         model_normalize_fqns: Optional[Callable],
                         module_name: str,
                         mono_sd: OrderedDict,
                         load_strict: bool,
                         current_model_keys: Set[str],
                         padded_numels: int,
                         ranks_to_write: List[int],
                         shard_size: int,
                         gigafsdp_state_dicts: List[OrderedDict],
                         param_name_to_layer_name: Dict[str, str],
                         param_name_suffix: str = "",) -> None:
    params = []
    numels_count = 0
    missed_weights_count, matched_weights_count = 0, 0
    for param_shape, param_name in zip(param_shapes, param_names):
        # global normalize function for all models fqns (etc. remove compile artifacts in fqns)
        param_name = clean_param_name(param_name)
        model_param_name = param_name

        # model-based normalize function for particular Composer Model (etc. remove substring for LoraModel)
        if model_normalize_fqns is not None:
            param_name = model_normalize_fqns(param_name)

        param_name = f"{module_name}." + param_name + param_name_suffix
        model_param_name = f"{module_name}." + model_param_name + param_name_suffix

        if param_name not in mono_sd:
            log.info(f'{param_name=} not in monolith checkpoint, fill with dummy value {DUMMY_VALUE}')
            if load_strict:  # with missing keys we get an error
                raise RuntimeError(f"GigaFSDP sharding: Parameter `{param_name}` is not in monolith checkpoint," + 
                                   " but it's in current model. Disable load_strict logic if you still want to load this model.")
            # fill missed weights with dummy value -100
            current_param = torch.full(param_shape, fill_value=DUMMY_VALUE, dtype=torch.float)
            missed_weights_count += 1

        elif list(mono_sd[param_name].shape) != param_shape:
            raise RuntimeError(f"GigaFSDP sharding: Parameter `{param_name}` has different shape in monolith checkpoint" +
                               f" and current model: monolith checkpoint - {mono_sd[param_name].shape}, current model - {param_shape}")

        else:
            log.info(f'{param_name=} in monolith checkpoint, load weights')
            current_param = mono_sd[param_name]
            matched_weights_count += 1

        params.append(current_param.reshape(-1))
        numels_count += torch.tensor(param_shape).prod()
        current_model_keys.add(param_name)

    # can take last processed param_name to detect it's shard name
    model_param_name_wo_ema = model_param_name[len(module_name)+1:].replace(param_name_suffix, "")
    shard_name = f"{module_name}.{param_name_to_layer_name[model_param_name_wo_ema]}{param_name_suffix}"

    if matched_weights_count == 0:
        log.warning(f"GigaFSDP sharding: No weights matched for {shard_name=}."
                    "Don't save this layer to sharded checkpoint.")
        # delete unnecessary 0-shard weights from all ranks
        for idx, output_rank in enumerate(ranks_to_write):
            if shard_name in gigafsdp_state_dicts[idx]['state']['model']:
                del gigafsdp_state_dicts[idx]['state']['model'][shard_name]
        return None
    else:
        log.info(f"GigaFSDP sharding: {shard_name=} are saved to sharded checkpoint.")

    flat_padded_params = torch.cat(params + [torch.zeros(padded_numels - numels_count)])
    for idx, output_rank in enumerate(ranks_to_write):
        shard = flat_padded_params.narrow_copy(0, shard_size * output_rank, shard_size)
        gigafsdp_state_dicts[idx]['state']['model'][shard_name] = shard.data.clone()
        assert shard_name in gigafsdp_state_dicts[idx]['state']['model'], (
            f"{shard_name=} not in {gigafsdp_state_dicts[idx]['state']['model'].keys()} !"
        )


def shard_unflat_state_dict(mono_sd: OrderedDict,
                            gigafsdp_state_dicts: List[OrderedDict],
                            ranks_to_write: List[int],
                            meta: Dict[str, Any],
                            param_groups: List[str] = PARAM_BUFFER_NAMES,
                            module_name: str = "model",
                            load_strict: bool = False,
                            model_normalize_fqns: Optional[Callable] = None, ) -> List[OrderedDict]:
    current_model_keys: Set = set()

    param_name_to_layer_name = meta["param_name_to_layer_name"]
    for param_group in param_groups:
        layers_per_shard = len(meta[f'{param_group}_parameter_shard_bounaries'])

        for layer_idx in range(layers_per_shard):
            (
                shard_size,
                param_shapes,
                param_names,
                padded_numels,
            ) = (
                meta[f'{param_group}_parameter_shard_bounaries'][layer_idx],
                meta[f'{param_group}_parameter_orig_shapes'][layer_idx],
                meta[f'{param_group}_shards_fqns'][layer_idx],
                meta[f'{param_group}_parameter_padded_numels'][layer_idx],
            )
            log.debug(f"{shard_size}, {param_shapes}, {param_names}, {padded_numels}")

            args = (param_shapes,
                    param_names,
                    model_normalize_fqns,
                    module_name,
                    mono_sd,
                    load_strict,
                    current_model_keys,
                    padded_numels,
                    ranks_to_write,
                    shard_size,
                    gigafsdp_state_dicts,
                    param_name_to_layer_name)

            update_gigafsdp_dict(*args)

            has_ema = any(key.endswith(EMA_SUFFIX) for key in gigafsdp_state_dicts[0]['state']['model'].keys())
            if has_ema:
                args = args + (EMA_SUFFIX,)
                update_gigafsdp_dict(*args)

    def gigafsdp_dicts_with_buffers(mono_sd: OrderedDict,
                                    gigafsdp_state_dicts: List[OrderedDict],
                                    current_model_keys: Set,) -> None:
        model_state_dict_keys = gigafsdp_state_dicts[0]['state']['model'].keys()

        bufs_dict = {}
        for is_required, patterns in [(True, REQUIRED_BUFFER_PATTERNS), (False, OPTIONAL_BUFFER_PATTERNS)]:
            buffer_names = [n for n in model_state_dict_keys if n.endswith(patterns)]
            for buf_name in buffer_names:
                buf_name_wo_module = _normalize_module_name(buf_name)
                try:
                    bufs_dict[buf_name] = mono_sd[buf_name_wo_module]
                    current_model_keys.add(buf_name_wo_module)
                except KeyError as e:
                    if is_required:
                        raise KeyError(f"Required buffer {buf_name} (normalized: {buf_name_wo_module}) not found in monolith checkpoint.") from e
                    else:
                        log.warning(f"Optional buffer {buf_name} not found in monolith checkpoint. This is expected when loading LLM checkpoint for omni pretraining. Error: {e}")
                        continue

        for sd in gigafsdp_state_dicts:
            sd['state']['model'].update(bufs_dict)

    gigafsdp_dicts_with_buffers(mono_sd, gigafsdp_state_dicts, current_model_keys)

    # check unexpected keys
    checkpoint_keys = set(key for key in mono_sd if key.startswith(f"{module_name}."))
    unexpected_keys = checkpoint_keys - current_model_keys
    missing_keys = current_model_keys - checkpoint_keys

    msg = f"GigaFSDP sharding: unexpected keys for module `{module_name}` from monolith checkpoint: {unexpected_keys}."
    msg += f" missing keys for module `{module_name}` in monolith checkpoint: {missing_keys}"

    if load_strict and unexpected_keys:
        raise RuntimeError(msg)

    log.info(msg)
    return gigafsdp_state_dicts


def legacy_shard_unflat_opt_state_dict(
    mono_opt_sd: OrderedDict,
    ranks_to_write: List[int],
    meta: Dict[str, Any],
    module_name: str = "model",
    model_normalize_fqns: Optional[Callable] = None
    ) -> List[OrderedDict]:

    optimizer_name = list(mono_opt_sd.keys())[0]
    group_idx=0 #supports only 1 param_group

    shard_param_groups = mono_opt_sd[optimizer_name]['param_groups'][group_idx]
    shard_param_groups['params'] = [pair[0] for pair in meta[MetaInfoKeys.PARAMS_MAPPING]]
    param_names = [pair[1] for pair in meta[MetaInfoKeys.PARAMS_MAPPING]]

    shard_opt_state_dicts = [ OrderedDict({optimizer_name : {'state' : {},
                                                            'param_groups' : [copy.copy(shard_param_groups)]}
                                        }) for _ in ranks_to_write ]
    
    for moment_name in ['exp_avg', 'exp_avg_sq']:
        moment_mono_sd = OrderedDict()
        flatten_opt_state_dicts = []

        for key in mono_opt_sd[optimizer_name]['state']:
            step = mono_opt_sd[optimizer_name]['state'][key]['step']
            moment_mono_sd[key] = mono_opt_sd[optimizer_name]['state'][key][moment_name]
        
        for idx in range(len(ranks_to_write)):
            shard_states = dict()
            for i, param_group in meta[MetaInfoKeys.PARAMS_MAPPING]:
                param_name = f'{module_name}._{param_group}_parameter'
                if param_name not in shard_opt_state_dicts[idx][optimizer_name]['state']:
                    shard_opt_state_dicts[idx][optimizer_name]['state'][param_name] = {'step' : step}

                shard_opt_state_dicts[idx][optimizer_name]['state'][param_name][moment_name] = torch.zeros(sum(meta[f'{param_group}_parameter_shard_bounaries']))
                shard_states[param_name] = shard_opt_state_dicts[idx][optimizer_name]['state'][param_name][moment_name]
            
                print(param_name, len(shard_states[param_name]))
            flatten_opt_state_dicts.append(shard_states)

        
        flatten_opt_state_dicts = shard_unflat_state_dict(moment_mono_sd,
                                                          flatten_opt_state_dicts,
                                                          ranks_to_write,
                                                          meta,
                                                          param_groups=param_names,
                                                          model_normalize_fqns=model_normalize_fqns
                                                         )

    for idx in range(len(ranks_to_write)):
        for i, param_group in meta[MetaInfoKeys.PARAMS_MAPPING]:
            param_name = f'{module_name}._{param_group}_parameter'
            shard_opt_state_dicts[idx][optimizer_name]['state'][i] = shard_opt_state_dicts[idx][optimizer_name]['state'].pop(param_name)

    return shard_opt_state_dicts


def shard_layerwise_param_state(
    param_tensors : List[torch.Tensor], 
    meta : Dict[str, Any], 
    layer_name : str, 
    ranks_to_write : List[int]
) -> List[torch.Tensor]:

    shards = []
    
    meta_info_idx, param_group = get_layer_meta_info_idx(layer_name, meta)
    padded_numel = meta[f'{param_group}_parameter_padded_numels'][meta_info_idx]
    shard_size = meta[f'{param_group}_parameter_shard_bounaries'][meta_info_idx]
    total_numels = sum(meta[f'{param_group}_shards_numels'][meta_info_idx])

    flat_padded_params = torch.cat(param_tensors + [torch.zeros(padded_numel - total_numels)])
    assert flat_padded_params.shape[0] == padded_numel, \
        f"Params count at layer {layer_name} != expected layer total numels." \
        f" {flat_padded_params.shape[0]=} != {padded_numel=}, expected {total_numels=}"

    for output_rank in ranks_to_write:
        shard = flat_padded_params.narrow(0, shard_size * output_rank, shard_size)
        shards.append(shard)
    
    return shards


def _shard_unflat_opt_state_dict(
    mono_opt_sd: OrderedDict,
    ranks_to_write: List[int],
    meta: Dict[str, Any],
    module_name: str = "model",
    model_normalize_fqns: Optional[Callable] = None
    ) -> List[OrderedDict]:

    optimizer_name = list(mono_opt_sd.keys())[0]
    layer_name_to_optimizer_state_idx = {v : k for k, v in meta[MetaInfoKeys.OPTIMIZER_STATE_IDX_TO_LAYER_NAME].items()}
    
    if model_normalize_fqns is None:
        def normalizer_fn(x):
            return clean_param_name(x)
    else:
        def normalizer_fn(x):
            return model_normalize_fqns(clean_param_name(x))
    param_name_to_layer_name = {
        normalizer_fn(k): v for k, v in meta[MetaInfoKeys.PARAM_NAME_TO_LAYER_NAME].items()
        if not v.endswith(f"_{ParamBufferPrefix.SEPARATE_FREEZED}_parameter") and 
        not v.endswith("flatten_freezed")
    }

    mono_opt_param_groups = copy.deepcopy(mono_opt_sd[optimizer_name]['param_groups'])
    for group in mono_opt_param_groups:
        group['params'] = []

    shard_opt_state_dicts = [
        OrderedDict({
            optimizer_name: {
                'state': {},
                'param_groups': copy.deepcopy(mono_opt_param_groups)
            }
        }) for _ in ranks_to_write
    ]

    visited_layer_names = set()

    def check_visited(layer_name):
        # can't hold layer params at different param groups
        if layer_name in visited_layer_names:
            raise RuntimeError(f"Layer {layer_name} is splited on different param groups")
        visited_layer_names.add(layer_name)

    def create_state(layer_name : str, 
                     param_names : List[str], 
                     group_idx : int):
        check_visited(layer_name)

        state_idx = layer_name_to_optimizer_state_idx[layer_name]
        for shard_opt_state_dict in shard_opt_state_dicts:
            shard_opt_state_dict[optimizer_name]['state'][state_idx] = {'step' : mono_opt_sd[optimizer_name]['state'][param_names[0]]['step']}
            shard_opt_state_dict[optimizer_name]['param_groups'][group_idx]['params'].append(state_idx)

        for moment_name in ['exp_avg', 'exp_avg_sq']:
            param_tensors = []
            for param_name in param_names:
                param_tensors.append(mono_opt_sd[optimizer_name]['state'][param_name][moment_name].flatten())

            shards = shard_layerwise_param_state(param_tensors, 
                                                 meta, 
                                                 layer_name,
                                                 ranks_to_write)

            for idx in range(len(ranks_to_write)):
                shard_opt_state_dicts[idx][optimizer_name]['state'][state_idx][moment_name] = shards[idx]

    for group_idx in range(len(mono_opt_sd[optimizer_name]['param_groups'])):
        group_params = [ param for param in mono_opt_sd[optimizer_name]['param_groups'][group_idx]['params'] if param.startswith(module_name) ]
        layer_names = []
        for param in group_params:
            param_name_wo_module = param[len(module_name) + 1 :]
            if param_name_wo_module not in param_name_to_layer_name:
                raise KeyError(
                    f'Param {param_name_wo_module} not found in '
                    f'param_name_to_layer_name for module {module_name} in '
                    f'current model. It looks like the model is not '
                    f'compatible with the checkpoint. Can\'t shard '
                    f'optimizer state.'
                )
            layer_names.append(param_name_to_layer_name[param_name_wo_module])

        current_layer_name, current_param_names = layer_names[0], [group_params[0]]
        for layer_name, param_name in zip(layer_names[1:], group_params[1:]):
            if layer_name != current_layer_name:
                create_state(current_layer_name, current_param_names, group_idx)
                current_layer_name, current_param_names = layer_name, [param_name]                
            else:
                current_param_names.append(param_name)

        create_state(current_layer_name, current_param_names, group_idx)

    return shard_opt_state_dicts


def shard_unflat_opt_state_dict(
    mono_opt_sd: OrderedDict,
    ranks_to_write: List[int],
    meta: Dict[str, Any],
    module_name: str = "model",
    model_normalize_fqns: Optional[Callable] = None
    ) -> List[OrderedDict]:
    
    if MetaInfoKeys.PARAMS_MAPPING in meta:
        return legacy_shard_unflat_opt_state_dict(mono_opt_sd, 
                                                  ranks_to_write, 
                                                  meta,
                                                  module_name,
                                                  model_normalize_fqns)
    else:
        return _shard_unflat_opt_state_dict(mono_opt_sd, 
                                            ranks_to_write, 
                                            meta,
                                            module_name,
                                            model_normalize_fqns)


def _model_config_get(config: Any, key: str, default: Any = False) -> Any:
    """Read a field from model config: plain dict or HuggingFace PretrainedConfig (e.g. GigaMixConfig)."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def shard_checkpoint(*,
    checkpoint_path : str, 
    num_output_rank : int,
    meta_infos : List[Tuple],
    gigafsdp_state_dict: OrderedDict,
    model_config: Union[Dict[str, Any], Any],
    output_path : Optional[str] = None,
    weights_only: bool = False,
    ranks_to_write: Optional[List] = None,
    load_strict: bool = False,
    model_normalize_fqns: Optional[Callable] = None,
    add_hash_to_state_dict_fn: Optional[Callable] = None
):
    """Convert monolith checkpoint to sharded one.

    Args:
        weights_only (bool, optional): Whether to shard only model weight or optimizer
            state as well. Default: ``False``.
    """
    mono_sd = torch.load(checkpoint_path, 
                         map_location='cpu', 
                         mmap=True,
                         weights_only=False,)
    mono_model_sd = mono_sd['state'].pop('model')

    new_expert_layout_flag_from_model = _model_config_get(
        model_config, "use_new_expert_weight_layout", False)
    new_expert_layout_flag_from_ckpt = mono_sd['state']['integrations']['huggingface']['model']['config']['content'].get(
        "use_new_expert_weight_layout", False)

    assert new_expert_layout_flag_from_model == new_expert_layout_flag_from_ckpt, (
        "you are trying to load a checkpoint with different expert weight layout than the model config. " +
        f"model_config: {new_expert_layout_flag_from_model}, ckpt: {new_expert_layout_flag_from_ckpt}"
    )

    start_from_sf_checkpoint_in_old_expert_weight_layout = _model_config_get(
        model_config, "start_from_sf_checkpoint_in_old_expert_weight_layout", False)

    if start_from_sf_checkpoint_in_old_expert_weight_layout and new_expert_layout_flag_from_model:
        raise RuntimeError("While loading checkpoint from shards, it's not allowed " +
            "to load checkpoint for new layout model from old layout!")

    has_optimizer = OPTIMIZERS_FIELD in mono_sd["state"]
    if has_optimizer:
        mono_opt_sd = mono_sd['state'].pop(OPTIMIZERS_FIELD)

    if not has_optimizer and OPTIMIZERS_FIELD in gigafsdp_state_dict['state']:
        gigafsdp_state_dict['state'].pop(OPTIMIZERS_FIELD)

    do_load_optimizer_state = has_optimizer and not weights_only

    if output_path and ranks_to_write is None: #single-process setup
        os.makedirs(output_path, exist_ok=True)

    ranks_to_write = ranks_to_write or list(range(num_output_rank))
    gigafsdp_state_dicts = [copy_state_dict(gigafsdp_state_dict) for _ in ranks_to_write]

    for module_name, meta in meta_infos:
        gigafsdp_state_dicts = shard_unflat_state_dict(mono_model_sd, 
                                                       gigafsdp_state_dicts,
                                                       ranks_to_write,
                                                       meta,
                                                       module_name=module_name,
                                                       load_strict=load_strict,
                                                       model_normalize_fqns=model_normalize_fqns)

        if do_load_optimizer_state and not meta['is_inference_mode']:
            # TO DO: support multiple trainable modules optimizer sharding
            assert module_name == "model", (
                "Gigafsdp now support only one trainable nn.Module - 'model' optimizer sharding"
            )
            shard_opts = shard_unflat_opt_state_dict(mono_opt_sd, 
                                                     ranks_to_write, 
                                                     meta, 
                                                     module_name=module_name,
                                                     model_normalize_fqns=model_normalize_fqns)

    sharded_state_dicts = [copy.copy(mono_sd) for _ in ranks_to_write]

    for idx, rank in enumerate(ranks_to_write):
        sharded_state_dicts[idx]['state']['model'] = gigafsdp_state_dicts[idx]['state']['model']
        if do_load_optimizer_state:
            sharded_state_dicts[idx]['state'][OPTIMIZERS_FIELD] = shard_opts[idx]
        
        for module_name, meta in meta_infos:
            sharded_state_dicts[idx]['state']['model'][f'{module_name}.meta'] = meta

        if output_path:
            if add_hash_to_state_dict_fn:
                log.info('Adding hashes to state dicts before saving them...')
                add_hash_to_state_dict_fn(sharded_state_dicts[idx])
            torch.save(sharded_state_dicts[idx], 
                       os.path.join(output_path, f"__{rank}_0.distcp"), 
                       pickle_module=dill)

    return sharded_state_dicts
