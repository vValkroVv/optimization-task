import torch
import re
from typing import Callable, Dict, Any, List, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict

from .common import ParamBufferPrefix, MetaInfoKeys
from .logger import logger

if TYPE_CHECKING:
    from .giga_fsdp import GigaFSDP


def validate_checkpoint_version(
        meta_info_loaded : Dict[str, Any], 
        meta_info_current : Dict[str, Any]
    ) -> None:
    """ 
    Validate checkpoint version.
    """
    if MetaInfoKeys.PARAMS_MAPPING in meta_info_loaded:
        raise ValueError(
            "Your checkpoint has outdated format of optimizer state. "
            "It can't be loaded with current version of GigaFSDP. "
            "Please convert your checkpoint to monolith format and try load again. "
            "Or you can disable loading optimizer by setting load_weights_only=True."
        )


def fill_optimizer_meta_info(
        model : "GigaFSDP", 
        special_process_group_fn: Optional[Callable] = None,
    ) -> None:
    """ Add to model._meta_info mappings for convertation optimizer to/from monolith checkpoint.
    Mappings:
        - str: layer_name -> int: it's meta info idx (only for trainable layers)
        - str: param_name -> str: layer_name
        - int: optimizer_state_idx -> str: layer_name

    Args:
        model (GigaFSDP): The model to fill the optimizer meta info.
    """
    from .giga_fsdp import SuperTensorModule

    # mapping from layer_name to it's meta info idx (only for trainable layers)
    model._meta_info[MetaInfoKeys.LAYER_NAME_TO_META_INFO_IDX] = {}
    for idx, (n, _) in enumerate(np for np in model._module.named_parameters() if np[1].requires_grad):
        model._meta_info[MetaInfoKeys.LAYER_NAME_TO_META_INFO_IDX][f"_module.{n}"] = idx

    model._meta_info[MetaInfoKeys.FREEZED_LAYER_NAME_TO_META_INFO_IDX] = {}
    for idx, (n, _) in enumerate(np for np in model._module.named_parameters() if not np[1].requires_grad):
        model._meta_info[MetaInfoKeys.FREEZED_LAYER_NAME_TO_META_INFO_IDX][f"_module.{n}"] = idx

    # mapping from param_name to layer_name
    model._meta_info[MetaInfoKeys.PARAM_NAME_TO_LAYER_NAME] = {}
    model._meta_info[MetaInfoKeys.PARAM_NAME_TO_FSDP_GROUP] = {}
    for n, module in model.named_modules():
        if isinstance(module, SuperTensorModule):
            for param in module.super_tensor.params:
                model._meta_info[MetaInfoKeys.PARAM_NAME_TO_LAYER_NAME][param.fqn] = f"{n}.flatten"
                model._meta_info[MetaInfoKeys.PARAM_NAME_TO_FSDP_GROUP][param.fqn] = model._get_pg_kit_for_module(
                    module.root_module, model.pg_manager, special_process_group_fn
                ).fsdp_process_group
            for param in module.super_tensor.params_freezed:
                model._meta_info[MetaInfoKeys.PARAM_NAME_TO_LAYER_NAME][param.fqn] = f"{n}.flatten_freezed"

    # add params from separate buffer (layernorms)
    separate_module_names = []
    if model._separate_layer_norm_enabled:
        for param in model._layernorm_supertensor.params:
            model._meta_info[MetaInfoKeys.PARAM_NAME_TO_LAYER_NAME][param.fqn] = \
                f"_{ParamBufferPrefix.SEPARATE}_parameter"
        for param in model._layernorm_supertensor.params_freezed:
            # all layernorms are freezed
            if len(model._layernorm_supertensor.params) == 0:
                buffer_name = f"_{ParamBufferPrefix.SEPARATE}_parameter"
            else:
                buffer_name = f"_{ParamBufferPrefix.SEPARATE_FREEZED}_parameter"
            model._meta_info[MetaInfoKeys.PARAM_NAME_TO_LAYER_NAME][param.fqn] = buffer_name

        # add names of separate modules (layernorms)
        for fqn, submodule in model._module.named_modules():
            if isinstance(submodule, model._layer_norm_module_cls):
                separate_module_names.append(fqn)
    model._meta_info[MetaInfoKeys.SEPARATE_MODULE_NAMES] = separate_module_names

    # mapping from optimizer_state_idx to layer_name (default 1 param group case)
    set_optimizer_state_idx_to_layer_name(model)


def set_optimizer_state_idx_to_layer_name(
        model : "GigaFSDP", 
        updated_mapping : Optional[dict[int, str]] = None
    ):
    """ 
    Set mapping from optimizer_state_idx (key in optimizer.state_dict()['state']) to layer_name. 
    If updated_mapping is provided, it will be used instead of creating default mapping.
    default mapping correspond to 1 param group in optimizer.
    """
    if updated_mapping is not None:
        model._meta_info[MetaInfoKeys.OPTIMIZER_STATE_IDX_TO_LAYER_NAME] = updated_mapping
        return

    model._meta_info[MetaInfoKeys.OPTIMIZER_STATE_IDX_TO_LAYER_NAME] = dict()
    # filter only trainable layers
    for idx, (name, _) in enumerate(np for np in model.named_parameters() if np[1].requires_grad):
        model._meta_info[MetaInfoKeys.OPTIMIZER_STATE_IDX_TO_LAYER_NAME][idx] = name


def get_layer_meta_info_idx(layer_name : str, meta : Dict[str, Any]) -> Tuple[int, str]:
    if layer_name.endswith(f"flatten"):
        meta_info_idx = meta[MetaInfoKeys.LAYER_NAME_TO_META_INFO_IDX][layer_name]
        param_group = ParamBufferPrefix.TRAINABLE
    elif layer_name.endswith(f"flatten_freezed"):
        meta_info_idx = meta[MetaInfoKeys.FREEZED_LAYER_NAME_TO_META_INFO_IDX][layer_name]
        param_group = ParamBufferPrefix.FREEZED
    elif layer_name.endswith(f"{ParamBufferPrefix.SEPARATE_FREEZED}_parameter"):
        meta_info_idx = 0
        param_group = ParamBufferPrefix.SEPARATE_FREEZED
    else:
        meta_info_idx = 0
        param_group = ParamBufferPrefix.SEPARATE

    return meta_info_idx, param_group


def recreate_giga_fsdp_param_groups(
        model: torch.nn.Module,
        non_wrapped_param_names_to_group_num: Dict[str, int],
        group_num_to_optimizer_info: Dict[int, Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
    """Helper function to recreate optimizer groups for GigaFSDP wrapped modules."""
    from .giga_fsdp import GigaFSDP

    def get_group_num_for_prefix(prefix: str, param_group_mapping: Dict[str, int]) -> int:
        """Find the group number for all parameters under a prefix."""
        prefix = f"{prefix}."
        matching_keys = [key for key in param_group_mapping.keys() if key.startswith(prefix)]
        
        if not matching_keys:
            raise ValueError(f"No parameters found matching prefix: {prefix}")
        
        logger.info(f"layer prefix: {prefix}, matching_keys: {matching_keys}")
        # Remove matched keys and get their values
        values = [param_group_mapping.pop(key) for key in matching_keys]
        
        # Verify all parameters belong to same group
        if len(set(values)) != 1:
            raise ValueError(f"Parameters under {prefix} belong to different groups")
            
        return values[0]
    
    def normalize_param_name(name: str) -> str:
        """Normalize parameter names by removing common prefixes/suffixes."""
        name = re.sub(r"\._module|\.root_module", "", name)
        name = re.sub(r"\.flatten(_freezed)?$", "", name)
        return name

    def process_separate_parameters(
        module: GigaFSDP,
        model_name: str,
        param_group_mapping: Dict[str, int]
    ) -> Tuple[int, List[torch.nn.Parameter], List[str]]:
        """Process separate parameters (e.g. layer norms) for a GigaFSDP module."""
        sep_module_names = [
            f"{model_name}.{normalize_param_name(name)}"
            for name in module._meta_info[MetaInfoKeys.SEPARATE_MODULE_NAMES]
        ]
        
        group_nums = [
            get_group_num_for_prefix(name, param_group_mapping)
            for name in sep_module_names
        ]
        
        if len(set(group_nums)) != 1:
            raise ValueError("All separate parameters must belong to the same group")
            
        return group_nums[0]

    def create_state_idx_to_layer_name_mapping(group_num_to_layer_name: Dict[int, List[str]]) -> Dict[int, str]:
        """Create mapping from optimizer state index to layer name"""
        state_idx_to_layer_name = {}
        current_idx = 0
        for group_idx in range(len(group_num_to_layer_name)):
            for layer_name in group_num_to_layer_name[group_idx]:
                state_idx_to_layer_name[current_idx] = layer_name
                current_idx += 1
        return state_idx_to_layer_name

    def create_filtered_mapping(model_name: str, state_idx_to_layer_name: Dict[int, str]) -> Dict[int, str]:
        """Creates filtered mapping for a specific model by removing the model name prefix.
        
        Args:
            model_name: Name of the model component to filter for
            state_idx_to_layer_name: Full mapping of state indices to layer names
            
        Returns:
            Filtered mapping containing only entries for the specified model with shortened names
        """
        prefix = f"{model_name}."
        return {
            state_idx: layer_name[len(prefix):]
            for state_idx, layer_name in state_idx_to_layer_name.items()
            if layer_name.startswith(prefix)
        }
    
    group_num_to_layer_name = defaultdict(list)
    for group_num in group_num_to_optimizer_info.keys():
        group_num_to_optimizer_info[group_num]['params'] = []
    
    for model_name, module in model.named_children():
        if not isinstance(module, GigaFSDP):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if param_name.endswith(f"{ParamBufferPrefix.SEPARATE}_parameter"):
                group_num = process_separate_parameters(
                    module, model_name, non_wrapped_param_names_to_group_num
                )
                if param.requires_grad:
                    group_num_to_optimizer_info[group_num]['params'].append(param)
                    group_num_to_layer_name[group_num].append(f"{model_name}.{param_name}")

    visited_layer_names = dict()
    # iterating over layerwise buffer params
    for layer_name, param in model.named_parameters():
        # Skip already processed separate parameters
        if any(layer_name.endswith(f"{suffix}_parameter") 
               for suffix in (ParamBufferPrefix.SEPARATE, ParamBufferPrefix.SEPARATE_FREEZED)):
            continue

        normalized_name = normalize_param_name(layer_name)
        # find the group of params in this layer
        if normalized_name in visited_layer_names:
            group_num = visited_layer_names[normalized_name]
        else:
            group_num = get_group_num_for_prefix(normalized_name, non_wrapped_param_names_to_group_num)
            visited_layer_names[normalized_name] = group_num

        # add layerwise buffer to the param group
        if param.requires_grad: 
            group_num_to_optimizer_info[group_num]['params'].append(param)
            group_num_to_layer_name[group_num].append(layer_name)

    # Create mapping from layer name to optimizer state index
    state_idx_to_layer_name = create_state_idx_to_layer_name_mapping(group_num_to_layer_name)

    # Update GigaFSDP modules with mapping
    for model_name, module in model.named_children():
        if not isinstance(module, GigaFSDP):
            continue
        filtered_mapping = create_filtered_mapping(model_name, state_idx_to_layer_name)
        set_optimizer_state_idx_to_layer_name(module, filtered_mapping)

    # check that all params are in some group after our deleting process
    assert len(non_wrapped_param_names_to_group_num) == 0, "param_name_to_group_num should be empty"

    return group_num_to_optimizer_info.values()
