import argparse
import json
import sys
import os
import shutil
import warnings
import copy
import gc
import decorator
import dill
import torch
import re
import types
import transformers

from pathlib import Path
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Tuple, Optional, Union, Iterator, Any
from itertools import chain
from packaging import version
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import warnings
from logger import logger, time_memory_stat

from base_mappers import LayerMapper, RawMapper, DeepseekV3Mapper, Gigachat35DenseMapper, GigachatSLMMapper, OmniRawMapper, OmniDeepseekV3Mapper
from mapper_gfsdp2_dcp_to_hf import (
    convert_dcp_to_safetensors,
    finalize_hf_checkpoint,
    read_titan_param_shapes,
    TitanToHF,
    write_hf_metadata,
)

try:
    from utils.dist import (
        _get_rank_world,
        _select_keys_for_rank_by_bundles,
        clear_barrier_state,
        _file_barrier,
    )
except BaseException as _:
    # NOTE (fedorovgv): want to use it in jupyter notebooks
    current_file = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_file / "utils"))
    from dist import (
        _get_rank_world,
        _select_keys_for_rank_by_bundles,
        clear_barrier_state,
        _file_barrier,
    )

if version.parse(transformers.__version__) > version.parse('4.36.2'):
    from mapper_deepseek_moe_to_hf import convert_deepseek_to_hf
else:
    warnings.warn("transformers version <= 4.36.2 -> cant import convert_deepseek_to_hf function." +
                  "Use transformers > 4.36.2.")
    convert_deepseek_to_hf = None

from mapper_hf_to_deepseek_moe import stream_hf_to_local_safetensors
from mapper_local_safetensor_to_hf import stream_local_safetensor_to_hf
from mapper_vision_utils import (
    add_tokenizer as legacy_add_tokenizer,
    load_yaml_config as load_yaml_config_legacy,
    extract_llm_state_dict_and_config as extract_llm_state_dict_and_config_legacy
)

from mapper_omni_utils import (
    load_yaml_config,
    extract_llm_state_dict_and_config,
    build_gigaomni_for_causal_lm,
    add_processors,
    _merge_hf_llm_safetensors,
)

from mapper_tts_utils import (
    extract_audio_llm_state_dict_and_config,
    build_composer_gigatts_model,
    add_modelling_tts,
)

#TODO: understand the cause of the bug, now it's hotfix
def _drop_teacher_keys(state_dicts):                                                                                                                                                                                                
    for sd in state_dicts:                                                                                                                                                                                                          
        model_sd = sd["state"]["model"]                                                                                                                                                                                             
        for k in [k for k in model_sd if k.startswith("teacher.")]:                                                                                                                                                                 
            del model_sd[k]                                                                                                                                                                                                         
    return state_dicts

def part_and_gc(mapper, args):
    result = partial(mapper, args=args)
    gc.collect()
    return result


def _normalize_gigar_sliding_window_config_for_hf(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Qwen3 sliding-window fields before HF consumes the config."""
    if config.get("model_type") != "qwen3":
        return config

    training_layer_types = config.pop("sliding_window_layer_types", None)
    layer_types = config.get("layer_types")

    if layer_types is not None and training_layer_types is not None:
        if list(layer_types) != list(training_layer_types):
            raise ValueError(
                "Qwen3 config contains both layer_types and sliding_window_layer_types, "
                "but they differ."
            )
    elif layer_types is None and training_layer_types is not None:
        layer_types = list(training_layer_types)
        config["layer_types"] = layer_types

    num_hidden_layers = config.get("num_hidden_layers")
    if layer_types is not None:
        if not isinstance(layer_types, (list, tuple)):
            raise ValueError(
                "layer_types must be a list or tuple for Qwen3 conversion, "
                f"got {type(layer_types).__name__}"
            )
        layer_types = list(layer_types)
        config["layer_types"] = layer_types

        if num_hidden_layers is not None and len(layer_types) != num_hidden_layers:
            raise ValueError(
                "layer_types length must match num_hidden_layers for Qwen3 conversion: "
                f"{len(layer_types)} != {num_hidden_layers}"
            )

        valid_layer_types = {"full_attention", "sliding_attention"}
        unknown_layer_types = sorted(set(layer_types) - valid_layer_types)
        if unknown_layer_types:
            raise ValueError(
                "Unsupported Qwen3 layer_types values for conversion: "
                f"{unknown_layer_types}. Expected only {sorted(valid_layer_types)}."
            )

    has_sliding_layers = bool(layer_types) and any(
        layer_type == "sliding_attention" for layer_type in layer_types
    )
    sliding_window = config.get("sliding_window")
    has_sliding_window = sliding_window is not None
    use_sliding_window = (
        has_sliding_window
        and (config.get("use_sliding_window") is True or has_sliding_layers)
    )

    if has_sliding_layers and not has_sliding_window:
        raise ValueError(
            "Qwen3 config contains sliding_attention layers but does not define sliding_window."
        )

    if has_sliding_window:
        if isinstance(sliding_window, bool) or not isinstance(sliding_window, int):
            raise ValueError(
                "sliding_window must be a positive integer for Qwen3 conversion, "
                f"got {sliding_window!r}"
            )
        if sliding_window <= 0:
            raise ValueError(
                "sliding_window must be a positive integer for Qwen3 conversion, "
                f"got {sliding_window}"
            )

    if not use_sliding_window:
        config["use_sliding_window"] = False
        config["sliding_window"] = None
        return config

    config["use_sliding_window"] = True
    if has_sliding_window:
        config["sliding_window"] = sliding_window
    return config


@decorator.decorator
def sequence_apply(
    mapper,
    state_dict: Union[List[Dict], Dict],
    *some_args,
) -> Union[List[Dict], Dict]:
    args = some_args[0]

    if isinstance(state_dict, list):
        return list(map(part_and_gc(mapper, args=args), state_dict))
    else:
        return mapper(state_dict, args=args)


def optimized_chunk(input, chunks: int, dim: int = 0):
    tensors = [tensor.clone() for tensor in torch.chunk(input, chunks=chunks, dim=dim)]
    return tensors


def get_language_model_config_from_fsdp_state_dict(fsdp_state_dict: Dict) -> Dict:
    config = fsdp_state_dict["state"]["integrations"]["huggingface"]["model"]["config"]["content"]

    if config["model_type"] == "giga_tts":
        config = config["llm_config"]["config"]
    elif config["model_type"] == "gigaomni":
        config = config["language_model"]

    return config


def get_hf_config_from_lora_state(model_state: Union[List[Dict], Dict], tp_size: int) -> Dict:
    """Extract model config from either a single FSDP state or TP-sharded states."""
    if tp_size > 1 and "state" not in model_state:
        return model_state[0]["state"]["integrations"]["huggingface"]["model"]["config"]["content"]
    return model_state["state"]["integrations"]["huggingface"]["model"]["config"]["content"]


def _is_new_layout_expert_tensor(weight: torch.Tensor) -> bool:
    # Legacy fused MoE expert tensors are always saved 2D; the new layout
    # stores them stacked per expert in 3D. Some checkpoints trained in the
    # new layout do not carry `use_new_expert_weight_layout` in the HF config
    # content, so a 3D tensor is treated as new-layout evidence on its own.
    return weight.ndim == 3


def _reshape_moe_expert_weight1(
    weight: torch.Tensor,
    model_config: Dict[str, Any],
    tp_size: int,
) -> torch.Tensor:
    use_new_expert_weight_layout = (
        model_config.get("use_new_expert_weight_layout", False)
        or _is_new_layout_expert_tensor(weight)
    )
    num_local_experts = model_config["num_routed_experts"] // tp_size
    hidden_size = model_config["hidden_size"]

    if use_new_expert_weight_layout:
        return weight.view(num_local_experts, -1, hidden_size).mT.contiguous()
    return weight.view(num_local_experts, hidden_size, -1)


def _reshape_moe_expert_weight2(
    weight: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    # (E, H, D) -> (E, D, H) -> (E*D, H)
    return weight.mT.contiguous().reshape(-1, hidden_size)


def _normalize_single_shard_moe_expert_weights(
    model_state: OrderedDict,
    model_config: Dict[str, Any],
) -> None:
    hidden_size = model_config["hidden_size"]
    use_new_expert_weight_layout = model_config.get("use_new_expert_weight_layout", False)
    weight1_suffixes = (
        "block_sparse_moe.experts.weight1",
        "block_sparse_moe.experts.base_layer.weight1",
    )
    weight2_suffixes = (
        "block_sparse_moe.experts.weight2",
        "block_sparse_moe.experts.base_layer.weight2",
    )

    for key in list(model_state.keys()):
        if any(suffix in key for suffix in weight1_suffixes):
            weight = model_state[key]
            if use_new_expert_weight_layout and weight.ndim == 2 and weight.shape == (hidden_size, hidden_size):
                raise ValueError(
                    f"Ambiguous single-shard MoE layout for {key}: square weight1 shape {tuple(weight.shape)} "
                    "matches both legacy and normalized layouts"
                )
            if weight.ndim == 2 and weight.shape[0] == hidden_size:
                continue
            model_state[key] = _reshape_moe_expert_weight1(
                weight,
                model_config,
                tp_size=1,
            ).reshape(hidden_size, -1)
        elif any(suffix in key for suffix in weight2_suffixes) and (
            use_new_expert_weight_layout
            or _is_new_layout_expert_tensor(model_state[key])
        ):
            weight = model_state[key]
            if not use_new_expert_weight_layout:
                logger.warning(
                    "Checkpoint config lacks use_new_expert_weight_layout, but %s has 3D shape %s; "
                    "treating it as the new expert weight layout",
                    key,
                    tuple(weight.shape),
                )
            if weight.ndim == 2 and weight.shape == (hidden_size, hidden_size):
                raise ValueError(
                    f"Ambiguous single-shard MoE layout for {key}: square weight2 shape {tuple(weight.shape)} "
                    "matches both legacy and normalized layouts"
                )
            if weight.ndim == 2 and weight.shape[1] == hidden_size:
                continue
            model_state[key] = _reshape_moe_expert_weight2(
                weight,
                hidden_size,
            )


def _normalize_single_shard_moe_expert_optimizer_states(
    optimizer_state: Dict[str, Dict[str, Any]],
    model_config: Dict[str, Any],
) -> None:
    hidden_size = model_config["hidden_size"]
    use_new_expert_weight_layout = model_config.get("use_new_expert_weight_layout", False)
    weight1_suffixes = (
        "block_sparse_moe.experts.weight1",
        "block_sparse_moe.experts.base_layer.weight1",
    )
    weight2_suffixes = (
        "block_sparse_moe.experts.weight2",
        "block_sparse_moe.experts.base_layer.weight2",
    )

    for key, param_state in optimizer_state.items():
        if any(suffix in key for suffix in weight1_suffixes):
            for state_name, state_value in param_state.items():
                if not isinstance(state_value, torch.Tensor):
                    continue
                if (
                    use_new_expert_weight_layout
                    and state_value.ndim == 2
                    and state_value.shape == (hidden_size, hidden_size)
                ):
                    raise ValueError(
                        f"Ambiguous single-shard MoE optimizer layout for {key}.{state_name}: "
                        f"square weight1 shape {tuple(state_value.shape)} matches both legacy and normalized layouts"
                    )
                if state_value.ndim == 2 and state_value.shape[0] == hidden_size:
                    continue
                param_state[state_name] = _reshape_moe_expert_weight1(
                    state_value,
                    model_config,
                    tp_size=1,
                ).reshape(hidden_size, -1)
        elif any(suffix in key for suffix in weight2_suffixes):
            for state_name, state_value in param_state.items():
                if not isinstance(state_value, torch.Tensor):
                    continue
                if not (
                    use_new_expert_weight_layout
                    or _is_new_layout_expert_tensor(state_value)
                ):
                    continue
                if state_value.ndim == 2 and state_value.shape == (hidden_size, hidden_size):
                    raise ValueError(
                        f"Ambiguous single-shard MoE optimizer layout for {key}.{state_name}: "
                        f"square weight2 shape {tuple(state_value.shape)} matches both legacy and normalized layouts"
                    )
                if state_value.ndim == 2 and state_value.shape[1] == hidden_size:
                    continue
                param_state[state_name] = _reshape_moe_expert_weight2(
                    state_value,
                    hidden_size,
                )


@sequence_apply
@time_memory_stat
def load_checkpoint(checkpoint_path : str, args = None) -> Dict:
    if checkpoint_path.startswith('s3://'):
        from s3torchconnector import S3Checkpoint
        checkpoint = S3Checkpoint(
            region=os.environ['S3_REGION'],
            endpoint=os.environ['S3_ENDPOINT']
        )
        logger.info(f'Loading {checkpoint_path} from S3 storage, can take quite a long time')
        with checkpoint.reader(checkpoint_path) as reader:
            state_dict = torch.load(reader, weights_only=False)
        return state_dict
    else:
        if os.path.isdir(checkpoint_path): # sharded checkpoint
            return load_gigafsdp_checkpoint(checkpoint_path)
        else: # monolith checkpoint
            return torch.load(checkpoint_path,
                        map_location="cpu",
                        mmap=True,
                        pickle_module=dill,
                        weights_only=False
                        )

@time_memory_stat
def load_hf_model(checkpoint_path: str, args=None) -> nn.Module:
    # TODO: add custom parameters for load model (like dtype)
    return AutoModelForCausalLM.from_pretrained(checkpoint_path)


@time_memory_stat
def load_gigafsdp_checkpoint(checkpoint_path: str) -> Tuple[Dict]:
    all_files = os.listdir(checkpoint_path)
    shard_files_count = len(list(filter(lambda file: file.endswith("distcp"), all_files)))

    if shard_files_count != len(all_files):
        wrong_files = list(filter(lambda file: not file.endswith("distcp"), all_files))
        warnings.warn(f"Gigafsdp sharded folder contains some unexpected files. Unexpected files: {wrong_files}.")

    state_dicts = []
    for rank in tqdm(range(shard_files_count)):
        state_dicts.append(torch.load(os.path.join(checkpoint_path, f"__{rank}_0.distcp"),
                                      map_location='cpu',
                                      mmap=True,
                                      weights_only=False))
    state_dicts = _drop_teacher_keys(tuple(state_dicts))
    return state_dicts


@time_memory_stat
def load_mixed_gigafsdp_checkpoint(checkpoint_path: str, args: Optional[argparse.Namespace]) -> List[Dict]:
    _NUM_RE = re.compile(r"\d+")
    _RANK_DIR_RE = re.compile(r"tp_rank(\d+).*?ep_rank(\d+)", re.IGNORECASE)

    def _extract_ranks_from_dirname(dirname: str) -> Optional[Tuple[int, int]]:
        m = _RANK_DIR_RE.search(dirname)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def _nums_from_name(p: Path) -> Tuple[int, ...]:
        nums = tuple(int(x) for x in _NUM_RE.findall(p.name))
        # second digit it .distcp file name is always 0 and not related with tp_rank or ep_rank
        # so we need to extract tp_rank and ep_rank from the parent directory name
        tp_rank, ep_rank = _extract_ranks_from_dirname(p.parent.name)
        return (nums[0], tp_rank, ep_rank) if nums else (10**12,)

    def _find_distcp_files(base_path: Path) -> List[Path]:
        if not base_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {base_path}")
        return sorted(base_path.rglob("*.distcp"), key=_nums_from_name)

    def _load_distcp_in_order(base_path: str, map_location: str = "cpu") -> Tuple[List[Dict], int, int]:
        path = Path(base_path).resolve()
        files = _find_distcp_files(path)
        if not files:
            raise ValueError(f"No .distcp files found under: {checkpoint_path}")
        tp_size = ep_size = 0
        loaded: List[Dict] = []
        logger.info(f"Loading files: {files}")
        for p in files:
            ranks = _extract_ranks_from_dirname(p.parent.name)
            if ranks is None:
                raise ValueError(
                    f"Failed to extract tp/ep ranks from directory name: '{p.parent.name}'. "
                    f"Expected pattern 'tp_rank_<N>_ep_rank_<M>' in path: {p}"
                )
            tp_rank, ep_rank = ranks
            tp_size = max(tp_size, tp_rank + 1)
            ep_size = max(ep_size, ep_rank + 1)
            loaded.append(torch.load(p, map_location=map_location, mmap=True, weights_only=False))
        return loaded, tp_size, ep_size

    loaded, tp_size, ep_size = _load_distcp_in_order(checkpoint_path)
    if args is not None:
        args.input_tp_size = tp_size
        args.input_ep_size = ep_size
        logger.info(f"Loaded mixed checkpoint with tp_size: {args.input_tp_size}, ep_size: {args.input_ep_size}")
    return _drop_teacher_keys(loaded)


@time_memory_stat
def save_checkpoint(
    fsdp_state_dict: Union[List[List[Dict]], List[Dict], Dict],
    checkpoint_path: Union[List[str], str],
    args=None
) -> None:
    if args.output_tp_size > 1:
        for state_dict, path in zip(fsdp_state_dict, checkpoint_path):
            if isinstance(state_dict, list): #gigafsdp sharded checkpoint
                _save_gigafsdp_checkpoint(state_dict, path, args=args)
            else:  #fsdp monolith checkpoint
                torch.save(state_dict, path, pickle_module=dill)
    else:
        if isinstance(fsdp_state_dict, list): #gigafsdp sharded checkpoint
            _save_gigafsdp_checkpoint(fsdp_state_dict,
                                      checkpoint_path,
                                      args=args)
        else:
            torch.save(fsdp_state_dict,
                       checkpoint_path,
                       pickle_module=dill)


def _save_gigafsdp_checkpoint(
    sharded_state_dicts: List[Dict],
    checkpoint_path: str,
    args=None
) -> None:
    os.makedirs(checkpoint_path, exist_ok=True)
    for rank in range(len(sharded_state_dicts)):
        torch.save(sharded_state_dicts[rank],
                   os.path.join(checkpoint_path, f"__{rank}_0.distcp"),
                   pickle_module=dill)


def replace_checkpoint(old_checkpoint_path: str, new_checkpoint_path: str) -> Dict:
    shutil.move(old_checkpoint_path, new_checkpoint_path)


def get_matched_weight(state_dict: OrderedDict[str, Tensor], param_name: str):
    return [key for key in state_dict.keys() if param_name in key]


def get_optimizer_name(fsdp_state_dict):
    return list(fsdp_state_dict["state"]["optimizers"].keys())[0]


@sequence_apply
@time_memory_stat
def remove_fsdp_optimizer(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None
) -> Dict:
    del fsdp_state_dict["state"]["optimizers"]
    return fsdp_state_dict


@sequence_apply
@time_memory_stat
def remove_gigafsdp_optimizer(
    state_dicts: Tuple[Dict],
    args: Optional[argparse.Namespace] = None
) -> Union[List[Dict], Dict]:

    for rank in range(len(state_dicts)):
        del state_dicts[rank]["state"]["optimizers"]

    return state_dicts


# ============================================================================
# Weight conversion: sonic MoE kernel <-> g_gemm MoE kernel
#
# g_gemm (GroupedLlamaMLP) — 2D packed weights:
#   weight1: (H, 2*I*E) — view(E, H, 2*I), last dim chunked [gate(I) | up(I)]
#   weight2: (I*E, H)   — view(E, I, H)
#
# sonic (GroupedSonicMLP) — 2D weights matching GroupedLlamaMLP shapes:
#   weight1: (H, 2*I*E) — forward: .view(E, 2*I, H), interleaved [g0, u0, g1, u1, ...]
#   weight2: (I*E, H)   — forward: .view(E, H, I)
# ============================================================================


def _convert_w1_ggemm_to_sonic(w1: torch.Tensor, E: int, H: int, I: int) -> torch.Tensor:
    """w1: g_gemm (H, 2*I*E) -> sonic (H, 2*I*E).

    g_gemm view(E, H, 2*I): chunked [gate; up]
    sonic  view(E, 2*I, H): interleaved [g0, u0, g1, u1, ...]
    Interleave gate/up + per-expert transpose, then flatten back to 2D.
    """
    w = w1.view(E, H, 2 * I)
    gate = w[:, :, :I]
    up = w[:, :, I:]
    w_interleaved = torch.stack([gate, up], dim=-1).reshape(E, H, 2 * I)
    return w_interleaved.permute(0, 2, 1).contiguous().view(H, 2 * I * E)


def _convert_w2_ggemm_to_sonic(w2: torch.Tensor, E: int, H: int, I: int) -> torch.Tensor:
    """w2: g_gemm (I*E, H) -> sonic (I*E, H).

    g_gemm view(E, I, H) -> per-expert transpose -> sonic view(E, H, I) -> flatten to 2D.
    """
    return w2.view(E, I, H).permute(0, 2, 1).contiguous().view(I * E, H)


def _convert_w1_sonic_to_ggemm(w1: torch.Tensor, E: int, H: int, I: int) -> torch.Tensor:
    """w1: sonic (H, 2*I*E) -> g_gemm (H, 2*I*E).

    sonic  view(E, 2*I, H): interleaved [g0, u0, ...]
    g_gemm view(E, H, 2*I): chunked [gate; up]
    Per-expert transpose + de-interleave, then flatten back to 2D.
    """
    w = w1.view(E, 2 * I, H).permute(0, 2, 1).contiguous()  # (E, H, 2*I) interleaved
    gate = w[:, :, 0::2]
    up = w[:, :, 1::2]
    w_chunked = torch.cat([gate, up], dim=-1)  # (E, H, 2*I) chunked
    return w_chunked.contiguous().view(H, 2 * I * E)


def _convert_w2_sonic_to_ggemm(w2: torch.Tensor, E: int, H: int, I: int) -> torch.Tensor:
    """w2: sonic (I*E, H) -> g_gemm (I*E, H).

    sonic view(E, H, I) -> per-expert transpose -> g_gemm view(E, I, H) -> flatten to 2D.
    """
    return w2.view(E, H, I).permute(0, 2, 1).contiguous().view(I * E, H)


def _apply_moe_weight_conversion(fsdp_state_dict: Dict, direction: str) -> Dict:
    """Apply sonic<->g_gemm weight conversion to all MoE expert weights in the checkpoint.

    Both formats now use 2D shapes: weight1 (H, 2*I*E), weight2 (I*E, H).
    The conversion reorders elements for different 3D view conventions used in forward.

    Args:
        fsdp_state_dict: FSDP checkpoint state dict.
        direction: 'ggemm_to_sonic' or 'sonic_to_ggemm'.
    """
    model_sd = fsdp_state_dict["state"]["model"]
    config = fsdp_state_dict["state"]["integrations"]["huggingface"]["model"]["config"]["content"]
    E = config["num_routed_experts"]
    H = config["hidden_size"]

    for key in list(model_sd.keys()):
        if "block_sparse_moe.experts.weight1" in key:
            w1 = model_sd[key]
            # Both formats: (H, 2*I*E)
            I = w1.shape[1] // (2 * E)
            if direction == "ggemm_to_sonic":
                model_sd[key] = _convert_w1_ggemm_to_sonic(w1, E, H, I)
            else:
                model_sd[key] = _convert_w1_sonic_to_ggemm(w1, E, H, I)
            logger.info(f"Converted {key}: {w1.shape} -> {model_sd[key].shape}")

        elif "block_sparse_moe.experts.weight2" in key:
            w2 = model_sd[key]
            # Both formats: (I*E, H)
            I = w2.shape[0] // E
            if direction == "ggemm_to_sonic":
                model_sd[key] = _convert_w2_ggemm_to_sonic(w2, E, H, I)
            else:
                model_sd[key] = _convert_w2_sonic_to_ggemm(w2, E, H, I)
            logger.info(f"Converted {key}: {w2.shape} -> {model_sd[key].shape}")

    return fsdp_state_dict


@sequence_apply
@time_memory_stat
def map_ggemm_to_sonic(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None,
) -> Dict:
    """Convert MoE expert weights from g_gemm kernel format to sonic kernel format."""
    return _apply_moe_weight_conversion(fsdp_state_dict, direction="ggemm_to_sonic")


@sequence_apply
@time_memory_stat
def map_sonic_to_ggemm(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None,
) -> Dict:
    """Convert MoE expert weights from sonic kernel format to g_gemm kernel format."""
    return _apply_moe_weight_conversion(fsdp_state_dict, direction="sonic_to_ggemm")


# ============================================================================
# Safetensor-to-safetensor MoE weight conversion
# ============================================================================

def _apply_moe_safetensor_conversion(input_path: str, output_path: str, direction: str, config_path: Optional[str] = None):
    """Convert MoE expert weights between sonic and g_gemm formats in safetensor files.

    Reads safetensors from input_path, applies conversion on MoE expert weight keys,
    writes converted safetensors to output_path. Non-weight files (config, tokenizer, etc.)
    are copied as-is.

    Args:
        input_path: Path to input safetensor checkpoint directory.
        output_path: Path to output directory.
        direction: 'ggemm_to_sonic' or 'sonic_to_ggemm'.
        config_path: Optional path to config.json. If None, looks for config.json in input_path
                     or composer_state.pt.
    """
    from safetensors.torch import load_file as load_safetensors
    try:
        from utils.safetensors_local import _save_chunk_as_safetensors
    except ImportError:
        from safetensors_local import _save_chunk_as_safetensors

    input_path = Path(input_path)
    output_path = Path(output_path)
    os.makedirs(output_path, exist_ok=True)

    # --- Resolve config to get E, H ---
    E, H = None, None
    # Try config.json in input dir
    cfg_file = Path(config_path) if config_path else input_path / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)
        E = cfg.get("num_routed_experts") or cfg.get("n_routed_experts")
        H = cfg.get("hidden_size")
    # Try composer_state.pt
    if (E is None or H is None) and (input_path / "composer_state.pt").exists():
        composer = torch.load(input_path / "composer_state.pt", weights_only=False)
        cfg = composer["state"]["integrations"]["huggingface"]["model"]["config"]["content"]
        E = cfg.get("num_routed_experts") or cfg.get("n_routed_experts")
        H = cfg.get("hidden_size")

    if E is None or H is None:
        raise ValueError(
            f"Could not determine num_routed_experts and hidden_size from config. "
            f"Provide config_path or ensure config.json / composer_state.pt exists in {input_path}"
        )

    # --- Find and process safetensor files ---
    # Collect safetensor files (support: flat dir, index, rank dirs)
    safetensor_files = sorted(input_path.rglob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {input_path}")

    for sf in safetensor_files:
        rel = sf.relative_to(input_path)
        out_sf = output_path / rel
        os.makedirs(out_sf.parent, exist_ok=True)

        tensors = load_safetensors(str(sf))
        converted = {}
        for key, tensor in tensors.items():
            if "block_sparse_moe.experts.weight1" in key:
                # Both formats: (H, 2*I*E) — derive I
                I = tensor.shape[1] // (2 * E)
                if direction == "ggemm_to_sonic":
                    tensor = _convert_w1_ggemm_to_sonic(tensor, E, H, I)
                else:
                    tensor = _convert_w1_sonic_to_ggemm(tensor, E, H, I)
                logger.info(f"Converted {key}: {tensors[key].shape} -> {tensor.shape}")
            elif "block_sparse_moe.experts.weight2" in key:
                # Both formats: (I*E, H) — derive I
                I = tensor.shape[0] // E
                if direction == "ggemm_to_sonic":
                    tensor = _convert_w2_ggemm_to_sonic(tensor, E, H, I)
                else:
                    tensor = _convert_w2_sonic_to_ggemm(tensor, E, H, I)
                logger.info(f"Converted {key}: {tensors[key].shape} -> {tensor.shape}")
            converted[key] = tensor

        _save_chunk_as_safetensors(converted, str(out_sf))
        logger.info(f"Saved {rel}")
        del tensors, converted

    # --- Copy non-safetensor files (config, index, tokenizer, etc.) ---
    for f in input_path.rglob("*"):
        if f.is_file() and not f.suffix == ".safetensors":
            rel = f.relative_to(input_path)
            dst = output_path / rel
            os.makedirs(dst.parent, exist_ok=True)
            if not dst.exists():
                shutil.copy2(str(f), str(dst))

    logger.info(f"MoE weight conversion ({direction}) complete: {input_path} -> {output_path}")


@time_memory_stat
def map_safetensor_ggemm_to_sonic(checkpoint_path, args=None):
    """Convert MoE expert weights from g_gemm to sonic format in safetensor files."""
    if not isinstance(checkpoint_path, (str, Path)):
        checkpoint_path = args.input_checkpoint_path
    output_path = args.output_checkpoint_path if args else str(Path(checkpoint_path).parent / "sonic_output")
    config_path = getattr(args, "config_path", None) if args else None
    _apply_moe_safetensor_conversion(checkpoint_path, output_path, "ggemm_to_sonic", config_path)


@time_memory_stat
def map_safetensor_sonic_to_ggemm(checkpoint_path, args=None):
    """Convert MoE expert weights from sonic to g_gemm format in safetensor files."""
    if not isinstance(checkpoint_path, (str, Path)):
        checkpoint_path = args.input_checkpoint_path
    output_path = args.output_checkpoint_path if args else str(Path(checkpoint_path).parent / "ggemm_output")
    config_path = getattr(args, "config_path", None) if args else None
    _apply_moe_safetensor_conversion(checkpoint_path, output_path, "sonic_to_ggemm", config_path)


@sequence_apply
@time_memory_stat
def map_base_to_packed_attn(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None
) -> Dict:
    model_sd = fsdp_state_dict["state"]["model"]
    if not args.skip_optimizer:
        optimizer_sd = fsdp_state_dict["state"]["optimizers"][get_optimizer_name(fsdp_state_dict)][
            "state"
        ]
        optimizer_pgs = fsdp_state_dict["state"]["optimizers"][get_optimizer_name(fsdp_state_dict)][
            "param_groups"
        ]

    q_weights = sorted(get_matched_weight(model_sd, ".q_proj.weight"))
    k_weights = sorted(get_matched_weight(model_sd, ".k_proj.weight"))
    v_weights = sorted(get_matched_weight(model_sd, ".v_proj.weight"))

    q_bias = sorted(get_matched_weight(model_sd, ".q_proj.bias"))
    k_bias = sorted(get_matched_weight(model_sd, ".k_proj.bias"))
    v_bias = sorted(get_matched_weight(model_sd, ".v_proj.bias"))

    if len(q_weights) == 0:
        warnings.warn("No keys matched with 'q_proj' pattern. Nothing to change.")

    if len(q_bias) == 0:
        warnings.warn("This model has no attention bias.")
    else:
        q_weights = chain(q_weights, q_bias)
        k_weights = chain(k_weights, k_bias)
        v_weights = chain(v_weights, v_bias)

    for q_proj, k_proj, v_proj in zip(q_weights, k_weights, v_weights):
        # map weight
        qkv_proj = q_proj.replace("q_proj", "qkv_proj")
        model_sd[qkv_proj] = torch.cat(
            [model_sd[q_proj], model_sd[k_proj], model_sd[v_proj]], dim=0
        )
        del model_sd[q_proj], model_sd[k_proj], model_sd[v_proj]

        if not args.skip_optimizer and q_proj in optimizer_sd:
            optimizer_sd[qkv_proj] = dict()

            # map optimizer
            for moment_name in ["exp_avg", "exp_avg_sq"]:
                optimizer_sd[qkv_proj][moment_name] = torch.cat(
                    [
                        optimizer_sd[q_proj][moment_name],
                        optimizer_sd[k_proj][moment_name],
                        optimizer_sd[v_proj][moment_name],
                    ],
                    dim=0,
                )
            optimizer_sd[qkv_proj]["step"] = optimizer_sd[q_proj]["step"]

            for optimizer_pg in optimizer_pgs:
                if q_proj in optimizer_pg["params"]:
                    idx = optimizer_pg["params"].index(q_proj)
                    optimizer_pg["params"].remove(q_proj)
                    optimizer_pg["params"].remove(k_proj)
                    optimizer_pg["params"].remove(v_proj)
                    optimizer_pg["params"].insert(idx, qkv_proj)

            del optimizer_sd[q_proj]
            del optimizer_sd[k_proj]
            del optimizer_sd[v_proj]

    return fsdp_state_dict


@sequence_apply
@time_memory_stat
def map_packed_to_base_attn(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None
) -> Dict:
    model_sd = fsdp_state_dict["state"]["model"]
    if not args.skip_optimizer:
        optimizer_sd = fsdp_state_dict["state"]["optimizers"][get_optimizer_name(fsdp_state_dict)][
            "state"
        ]
        optimizer_pgs = fsdp_state_dict["state"]["optimizers"][get_optimizer_name(fsdp_state_dict)][
            "param_groups"
        ]

    config = get_language_model_config_from_fsdp_state_dict(fsdp_state_dict)

    if config.get("attention_hidden_size", None):
        kv_hidden_size = config['attention_hidden_size'] // config['num_attention_heads'] * config['num_key_value_heads']
        attention_size = config['attention_hidden_size']
    else:
        key_value_heads = config.get('num_key_value_heads', None) # in tests there are model without such key in config, assumed to be equal to num_attention_heads
        key_value_heads = config['num_attention_heads'] if key_value_heads is None else key_value_heads
        kv_hidden_size = config['hidden_size'] // config['num_attention_heads'] *  key_value_heads
        attention_size = config['hidden_size']

    # Only standard-attention qkv_proj weights should be split here. Linear
    # attention (Qwen3NextGatedDeltaNet with linear_use_legacy_qkvz_layout=False)
    # also has a `qkv_proj`, but it is nested under `._attn.` (or
    # `.gated_delta_net.` for MTP) and packs [q, k, v] with different
    # dimensions (2*key_dim + value_dim); leave those keys alone.
    qkv_weights = [
        k for k in get_matched_weight(model_sd, "qkv_proj")
        if k.endswith(".self_attn.qkv_proj.weight")
    ]
    if len(qkv_weights) == 0:
        warnings.warn("No keys matched with 'qkv_proj' pattern. Nothing to change.")

    for qkv_proj in qkv_weights:
        # map weight
        q_proj = qkv_proj.replace("qkv_proj", "q_proj")
        k_proj = qkv_proj.replace("qkv_proj", "k_proj")
        v_proj = qkv_proj.replace("qkv_proj", "v_proj")

        (model_sd[q_proj], model_sd[k_proj], model_sd[v_proj]) = torch.split(
            model_sd[qkv_proj],
            (
                attention_size // config["tp_size"],
                kv_hidden_size // config["tp_size"],
                kv_hidden_size // config["tp_size"],
            ),
            dim=0,
        )

        del model_sd[qkv_proj]

        if not args.skip_optimizer and qkv_proj in optimizer_sd:

            optimizer_sd[q_proj] = dict()
            optimizer_sd[k_proj] = dict()
            optimizer_sd[v_proj] = dict()

            # map optimizer
            for moment_name in ["exp_avg", "exp_avg_sq"]:
                (
                    optimizer_sd[q_proj][moment_name],
                    optimizer_sd[k_proj][moment_name],
                    optimizer_sd[v_proj][moment_name],
                ) = torch.split(
                    optimizer_sd[qkv_proj][moment_name],
                    (
                        attention_size // config["tp_size"],
                        kv_hidden_size // config["tp_size"],
                        kv_hidden_size // config["tp_size"],
                    ),
                    dim=0,
                )

            optimizer_sd[q_proj]["step"] = optimizer_sd[qkv_proj]["step"]

            for optimizer_pg in optimizer_pgs:
                if qkv_proj in optimizer_pg["params"]:
                    idx = optimizer_pg["params"].index(qkv_proj)
                    optimizer_pg["params"].insert(idx, v_proj)
                    optimizer_pg["params"].insert(idx, k_proj)
                    optimizer_pg["params"].insert(idx, q_proj)
                    optimizer_pg["params"].remove(qkv_proj)

            del optimizer_sd[qkv_proj]

    return fsdp_state_dict


def _get_ema_activation_info(state_dict: Dict) -> tuple[Optional[int], Optional[int]]:
    """Return (checkpoint_batch, ema_start_batch) from the loaded state dict, or (None, None)."""
    try:
        checkpoint_batch = state_dict["state"]["timestamp"]["Timestamp"]["batch"]
        start_batch = state_dict["state"]["callbacks"]["EMA"]["start_batch"]
        return checkpoint_batch, start_batch
    except (KeyError, TypeError):
        return None, None


def put_ema_to_weights_or_delele_ema(fsdp_state_dict: Dict, args: Optional[argparse.Namespace] = None) -> None:
    """ For state dicts with both ema / non-ema weights we can replace orig weights with ema weights,
        or save orig weights and delete ema weights.
    """
    from giga_fsdp import EMA_SUFFIX
    delete_ema = (args is not None) and (not getattr(args, "use_ema", False))
    for key in list(fsdp_state_dict["state"]["model"].keys()):
        if key.endswith(EMA_SUFFIX):
            orig_key = key[:-len(EMA_SUFFIX)]
            if not getattr(args, "prepare_ema_weight", False):
                assert orig_key in fsdp_state_dict["state"]["model"], f"State dict has no '{orig_key}' param but only it's EMA analogue"
            model = fsdp_state_dict["state"]["model"]
            if not delete_ema:
                model[orig_key] = model[key]
            del model[key]


@time_memory_stat
def map_monolith_fsdp_to_hf(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None
) -> None:
    config = fsdp_state_dict["state"]["integrations"]["huggingface"]["model"]["config"]["content"]
    put_ema_to_weights_or_delele_ema(fsdp_state_dict)

    if config['model_type'] == "giga_mix" or config['model_type'] == "papaya_mix":
        _map_monolith_deepseek_to_hf(fsdp_state_dict, args=args)
    elif config['model_type'] == "gigar" or config['model_type'] == "papayar":
        _map_monolith_gigar_to_hf(fsdp_state_dict, args=args)
    elif config['model_type'] == "giga_tts":
        _map_monolith_gigatts_to_hf(fsdp_state_dict, args=args)
    elif config['model_type'] == "gigaomni":
        _map_monolith_gigaomni_to_hf(fsdp_state_dict, args=args)
    else:
        raise ValueError(f"model_type={config['model_type']} is not supported!")


def _map_monolith_deepseek_to_hf(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None
) -> None:
    convert_deepseek_to_hf(fsdp_state_dict, args)


def _map_monolith_gigar_to_hf(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None
) -> None:
    HF_CHECKPOINT_DIR = args.output_checkpoint_path

    os.makedirs(HF_CHECKPOINT_DIR, exist_ok=True)

    config = fsdp_state_dict["state"]["integrations"]["huggingface"]["model"]["config"]["content"]

    # Map `model_type` to HF class and update HF class architecture accordigly.
    if args.model_type is not None:
        config["model_type"] = args.model_type
    else:
        if config.get("attention_bias", None):
            config["model_type"] = "qwen2"
        elif config.get("apply_qk_norm", False):
            config["model_type"] = "qwen3"
        else:
            config["model_type"] = MAP_ARCH_NAMES[config["model_type"]]

    _normalize_gigar_sliding_window_config_for_hf(config)

    if config["architectures"] is None:
        config["architectures"] = [MAP_MODEL_TYPE_TO_ARCH[config["model_type"]]]
    else:
        config["architectures"][0] = MAP_MODEL_TYPE_TO_ARCH[config["model_type"]]

    if (
        config.get("attention_hidden_size", None)
        and config["attention_hidden_size"] != config["hidden_size"]
    ):
        assert version.parse(transformers.__version__) >= version.parse(
            "4.45.0"
        ), (
            "Convertation model with attention_hidden_size!=hidden_size requires transformers>=4.45.0"
        )

        logger.info("Saving Gigar checkpoint with extended attention architecture.")
        config["head_dim"] = int(
            config["attention_hidden_size"] // config["num_attention_heads"]
        )
        del config["attention_hidden_size"]
    elif "head_dim" not in config or config["head_dim"] is None:
        config["head_dim"] = int(
            config["hidden_size"] // config["num_attention_heads"]
        )

    if config["head_dim"] != 128:
        warnings.warn(
            f"head_dim={config['head_dim']} in your model. Make sure that this value is correct."
        )

    assert config["architectures"][0] in SUPPORTED_ARCHS, (
        f"only {SUPPORTED_ARCHS} models are supported at the moment, not {config['architectures'][0]}"
    )

    with open(f"{HF_CHECKPOINT_DIR}/config.json", "w") as file:
        file.write(json.dumps(config))

    if args.tokenizer_path:
        TOKENIZER_PATH = args.tokenizer_path
    else:
        TOKENIZER_PATH = config["_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(f"{HF_CHECKPOINT_DIR}/config.json", trust_remote_code=True)
    cfg.bos_token_id = tokenizer.bos_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # fix some attr
    if hasattr(cfg, "tie_word_embeddings"):
        cfg.tie_word_embeddings = False
    if hasattr(cfg, "return_dict"):
        del cfg.return_dict

    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    ckpt = fsdp_state_dict
    model.load_state_dict(
        {k[len("model.") :]: v for k, v in ckpt["state"]["model"].items()}, strict=True
    )

    model.save_pretrained(HF_CHECKPOINT_DIR)
    tokenizer.save_pretrained(HF_CHECKPOINT_DIR)


@time_memory_stat
def _map_monolith_gigaomni_to_hf(
    fsdp_state_dict: Dict,
    args: argparse.Namespace = None
) -> None:
    save_path = Path(args.output_checkpoint_path)
    save_path.mkdir(exist_ok=True)

    save_llm_path = save_path / "language_model"

    yaml_config = load_yaml_config(args.input_checkpoint_path)
    llm_config, llm_state_dict = extract_llm_state_dict_and_config(fsdp_state_dict)

    if save_llm_path.exists():
        shutil.rmtree(save_llm_path, ignore_errors=True)

    args.output_checkpoint_path = str(save_llm_path)
    # Calling map_monolith_fsdp_to_hf again with llm checkpoint
    map_monolith_fsdp_to_hf(llm_state_dict, args=args)
    args.output_checkpoint_path = str(save_path)

    language_model = AutoModelForCausalLM.from_pretrained(save_llm_path, trust_remote_code=True)
    model = build_gigaomni_for_causal_lm(yaml_config, llm_config, str(save_llm_path))

    missing_keys, unexpected_keys = model.load_state_dict({
        name[len("model."):]: param for name, param in fsdp_state_dict["state"]["model"].items() if "language_model" not in name
    }, strict=False)

    missing_keys = [key for key in missing_keys if "language_model" not in key]
    unexpected_keys = [key for key in unexpected_keys if "language_model" not in key]

    assert len(missing_keys) + len(unexpected_keys) == 0, \
        f"Got weights mismatch during loading components weights:\nmissing_keys: {missing_keys}\nunexpected_keys:{unexpected_keys}"

    model.language_model.load_state_dict(language_model.state_dict(), strict=True)
    model.to(getattr(torch, args.torch_dtype))
    model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(yaml_config.tokenizer.name, **yaml_config.tokenizer.kwargs)
    tokenizer.save_pretrained(save_path)

    add_processors(save_path, yaml_config, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(save_path, trust_remote_code=True)

    for path in save_llm_path.iterdir():
        if path.is_file() and path.stat().st_size / (1024 ** 3) > 1:
            path.unlink()


def _map_monolith_gigatts_to_hf(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None
) -> None:
    save_path = Path(args.output_checkpoint_path)
    save_path.mkdir(exist_ok=True)

    yaml_config = load_yaml_config_legacy(args.input_checkpoint_path)

    # Add 'architecture' in Model Config
    fsdp_state_dict["state"]["integrations"]["huggingface"]["model"]["config"]["content"]['llm_config']['config']['architectures'] = ['LlamaModel', 'LlamaForCausalLM']
    fsdp_state_dict["state"]["integrations"]["huggingface"]["model"]["config"]["content"]['audio_adapter_config']['params']['model_params']['architectures'] = ['LlamaModel', 'LlamaForCausalLM']

    # LLM
    save_llm_tmp_path = save_path / "tmp_llm_dir"
    llm_config, llm_state_dict = extract_llm_state_dict_and_config_legacy(fsdp_state_dict)

    # map packed attention (llm)
    llm_state_dict = map_packed_to_base_attn(llm_state_dict, args=args)
    llm_config.pop('attention_type', None)

    args.output_checkpoint_path = str(save_llm_tmp_path)
    # Calling map_monolith_fsdp_to_hf again with llm checkpoint
    map_monolith_fsdp_to_hf(llm_state_dict, args=args)
    args.output_checkpoint_path = str(save_path)
    llm = AutoModelForCausalLM.from_pretrained(save_llm_tmp_path, trust_remote_code=True)

    # Audio LLM
    save_audio_llm_tmp_path = save_path / "tmp_audio_llm_dir"
    audio_llm_config, audio_llm_state_dict = extract_audio_llm_state_dict_and_config(fsdp_state_dict)

    # map packed attention (audio llm)
    audio_llm_state_dict = map_packed_to_base_attn(audio_llm_state_dict, args=args)
    audio_llm_config.pop('attention_type', None)

    args.output_checkpoint_path = str(save_audio_llm_tmp_path)
    # Calling map_monolith_fsdp_to_hf again with audio llm checkpoint
    map_monolith_fsdp_to_hf(audio_llm_state_dict, args=args)
    args.output_checkpoint_path = str(save_path)
    audio_llm = AutoModelForCausalLM.from_pretrained(save_audio_llm_tmp_path, trust_remote_code=True)

    # Composer
    model = build_composer_gigatts_model(
        yaml_config,
        updated_llm_config=llm_config,
        updated_audio_llm_config=audio_llm_config,
        tmp_llm_path=str(save_llm_tmp_path),
        tmp_audio_llm_path=str(save_audio_llm_tmp_path)
    )

    missing_key, unexpected_key = model.load_state_dict({
        name: param for name, param in fsdp_state_dict["state"]["model"].items() if "_language_model" not in name and '_mm_audio_adapter' not in name
    }, strict=False)

    assert len([key for key in missing_key if '_language_model' not in key and '_mm_audio_adapter' not in key]) == 0
    assert len(unexpected_key) == 0

    # Load LLM
    model.model._language_model.load_state_dict(llm.state_dict())

    # Load Audio Adapter
    audio_adapter_state_dict = audio_llm.state_dict()
    audio_adapter_state_dict.pop('model.embed_tokens.weight')
    model.model._mm_audio_adapter.load_state_dict({
        name.replace("model.", 'audio_adapter.audio_llm.').replace('lm_head.weight', 'audio_adapter.lm_head.weight'): param for name, param in audio_adapter_state_dict.items()
    })

    model.model.to(getattr(torch, args.torch_dtype))
    model.model.save_pretrained(args.output_checkpoint_path)

    add_modelling_tts(save_path)
    legacy_add_tokenizer(save_path, yaml_config)

    logger.info("Loading Model to check correctness")
    model = AutoModelForCausalLM.from_pretrained(save_path, trust_remote_code=True)
    logger.info("Loading Model to check correctness: Done")

    shutil.rmtree(str(save_llm_tmp_path), ignore_errors=True)
    shutil.rmtree(str(save_audio_llm_tmp_path), ignore_errors=True)


@time_memory_stat
def map_hf_to_monolith_fsdp(
    hf_model: nn.Module,
    args: Optional[argparse.Namespace] = None
) -> Dict:
    model_state_dict = hf_model.state_dict()
    model_config = hf_model.config

    target_state_dict = OrderedDict()
    for layer_name, layer_weight in model_state_dict.items():
        target_state_dict[f"model.{layer_name}"] = layer_weight

    target_fsdp_ckpt = {
        "state": {
            "model": target_state_dict
        }
    }

    if not hasattr(model_config, "head_dim"):
        model_config.head_dim = int(model_config.hidden_size // model_config.num_attention_heads)

    if model_config.head_dim * model_config.num_attention_heads != model_config.hidden_size:
        model_config.attention_hidden_size = model_config.head_dim * model_config.num_attention_heads

    model_config.tp_size = 1

    target_fsdp_ckpt["state"]["integrations"] = {
        "huggingface": {
            "model": {
                "config": {
                    "file_extension": "json",
                    "content": model_config.to_dict()
                }
            }
        }
    }

    return target_fsdp_ckpt


def is_substr_of_strings(layer_name, keys):
    for k in keys:
        if k in layer_name:
            return True
    return False


def _infer_parallel_embedding_type(
    embedding_weight_shard: torch.Tensor,
    model_config: Dict,
    input_tp_size: int,

) -> str:
    sharded_embedding_size = embedding_weight_shard.size(1)
    expected_hidden_size = model_config["hidden_size"]
    if expected_hidden_size == sharded_embedding_size * input_tp_size:
        return "EmbeddingParallelEmbedding"
    return "VocabParallelEmbedding"

COLUMN_PARALLEL_KEYS = [
    "embed_tokens",
    # base
    ".q_proj",
    ".k_proj",
    ".v_proj",
    # MLA
    ".uq_proj",
    ".uv_proj",
    ".uk_proj",
    ".kr_proj",
    # packed
    "_gate",
    "gate_proj",
    "up_proj",
    "lm_head",
    "block_sparse_moe.w2",
    "block_sparse_moe.shared_w1",
    "block_sparse_moe.shared_w3",
    "block_sparse_moe.experts.weight1",
    "block_sparse_moe.experts.weight2",
]

ROW_PARALLEL_KEYS = [
    "o_proj",
    "down_proj",
    # MLA
    ".dq_proj",
    ".dkv_proj",
    "block_sparse_moe.w1",
    "block_sparse_moe.w3",
    "block_sparse_moe.shared_w2",
    "block_sparse_moe.gate",
]

EXPERT_LORA_KEYS = [
    "block_sparse_moe.experts.lora_A1",
    "block_sparse_moe.experts.lora_B1",
    "block_sparse_moe.experts.lora_A2",
    "block_sparse_moe.experts.lora_B2",
]

@time_memory_stat
def map_tp_to_monolith(
    tp_ckpts: List[Dict],
    args: Optional[argparse.Namespace] = None
) -> Dict:
    TP_SIZE = len(tp_ckpts)

    monolith_checkpoint = dict()

    assert not is_key_contained_in_model(tp_ckpts[0], 'qkv_proj'), "The mapper only works with the base attention"

    if not args.skip_optimizer:
        monolith_checkpoint["rng"] = deepcopy(tp_ckpts[0]["rng"])

    monolith_checkpoint["state"] = dict()
    for k, v in tp_ckpts[0]["state"].items():
        if k in ["model", "optimizers"]:
            continue
        monolith_checkpoint["state"][k] = deepcopy(v)

    model_config = get_language_model_config_from_fsdp_state_dict(monolith_checkpoint)

    def concatenate_optim_tensors(tensors, axis, concatenate: bool=True):
        res = defaultdict(list)
        assert len(tensors) == TP_SIZE
        for tensor in tensors:
            for k, v in tensor.items():
                res[k].append(v)
        for k in res.keys():
            if k == "step" or not concatenate:
                res[k] = res[k][0]
            else:
                res[k] = torch.concatenate(res[k], axis)
        return dict(res)

    def concatenate_states(ckpt_states, mode):
        assert mode in ["model", "optimizer"]
        model_states = defaultdict(list)
        for ckpt in ckpt_states:
            for k, v in tqdm(ckpt.items()):
                model_states[k].append(v)
        #### Column Parallel
        for k in tqdm(model_states.keys()):
            if (
                is_substr_of_strings(k, EXPERT_LORA_KEYS)
                or (
                    is_substr_of_strings(k, COLUMN_PARALLEL_KEYS)
                    and "lora_A" not in k
                )
            ):
                # Expert LoRA tensors are sharded by experts across EP ranks
                # and also need concatenation along axis 0.
                if "embed_tokens" in k and args.input_tp_embedding_type is None:
                    args.input_tp_embedding_type = _infer_parallel_embedding_type(
                        model_states[k][0], model_config, args.input_tp_size
                    )

                if "embed_tokens" in k and args.input_tp_embedding_type == "EmbeddingParallelEmbedding":
                    concat_axis = 1
                else:
                    concat_axis = 0

                if "embed_tokens" in k and not mode == "optimizer":
                    expected_hidden_size = model_config["hidden_size"]
                    result_hidden_size = model_states[k][0].size(1)
                    if concat_axis == 1:
                        result_hidden_size *= args.input_tp_size

                    assert (
                        result_hidden_size == expected_hidden_size
                    ), f"Expected unshared embedding hidden size to be {expected_hidden_size}, got {result_hidden_size}. You might have confused VocabParallel and EmbeddingParallel. See `--tp-embedding-type` param for more info"

                # TMP: hotfix
                use_new_expert_weight_layout = model_config.get("use_new_expert_weight_layout", False)
                if "block_sparse_moe.experts.weight1" in k:
                    concat_axis = 0
                    model_states[k] = [
                        _reshape_moe_expert_weight1(v, model_config, TP_SIZE)
                        for v in model_states[k]
                    ]

                if mode == "model":
                    model_states[k] = torch.concatenate(model_states[k], concat_axis)
                else:
                    model_states[k] = concatenate_optim_tensors(model_states[k], concat_axis)

                if (
                    mode == "model"
                    and "block_sparse_moe.experts.weight2" in k
                    and (
                        use_new_expert_weight_layout
                        or _is_new_layout_expert_tensor(model_states[k])
                    )
                ):
                    hidden_size = model_config["hidden_size"]
                    model_states[k] = _reshape_moe_expert_weight2(
                        model_states[k],
                        hidden_size,
                    )

                # TMP: hotfix
                if "block_sparse_moe.experts.weight1" in k and mode == "model":
                    hidden_size = model_config["hidden_size"]
                    model_states[k] = model_states[k].reshape(hidden_size, -1)

            #### Row Parallel
            elif is_substr_of_strings(k, ROW_PARALLEL_KEYS) and "lora_B" not in k:
                if mode == "model":
                    model_states[k] = torch.concatenate(model_states[k], 1)
                else:
                    model_states[k] = concatenate_optim_tensors(model_states[k], 1)
            else:
                if mode == "model":
                    model_states[k] = model_states[k][0]
                else:
                    model_states[k] = concatenate_optim_tensors(model_states[k], 0, False)
        return model_states

    ckpt_states = []
    for ckpt in tp_ckpts:
        ckpt_states.append(ckpt["state"]["model"])

    monolith_checkpoint["state"]["model"] = OrderedDict(
        concatenate_states(ckpt_states, mode="model")
    )

    config = get_language_model_config_from_fsdp_state_dict(monolith_checkpoint)
    config["tp_size"] = 1

    if not args.skip_optimizer:
        monolith_checkpoint["state"]["optimizers"] = dict()
        for optim_key in tp_ckpts[0]["state"]["optimizers"].keys():
            monolith_checkpoint["state"]["optimizers"][optim_key] = dict()
            tp_ckpt_dict = tp_ckpts[0]["state"]["optimizers"][optim_key]
            assert len(tp_ckpt_dict) == 2 and "param_groups" in tp_ckpt_dict and "state" in tp_ckpt_dict
            monolith_checkpoint["state"]["optimizers"][optim_key]["param_groups"] = deepcopy(
                tp_ckpt_dict["param_groups"]
            )

            ckpt_states = []
            for ckpt in tp_ckpts:
                ckpt_states.append(ckpt["state"]["optimizers"][optim_key]["state"])
            monolith_checkpoint["state"]["optimizers"][optim_key]["state"] = concatenate_states(
                ckpt_states, mode="optimizer"
            )

    return monolith_checkpoint


def is_key_contained_in_model(state_dict, key):
    for name in state_dict['state']['model']:
        if key in name:
            return True
    return False


@time_memory_stat
def map_monolith_to_tp(
    ckpt: Dict,
    args: Optional[argparse.Namespace] = None
) -> List[Dict]:
    TP_SIZE = args.output_tp_size

    assert not is_key_contained_in_model(ckpt, 'qkv_proj'), "The mapper only works with the base attention"

    tp_ckpts = []
    for _ in range(TP_SIZE):
        tmp = dict()
        if "rng" in ckpt:
            tmp["rng"] = deepcopy(ckpt["rng"])
        tmp["state"] = dict()
        for k, v in ckpt["state"].items():
            if k in ["model", "optimizers"]:
                continue
            tmp["state"][k] = deepcopy(v)
        tp_ckpts.append(tmp)

    def prepare_tensors(k, v, res_list):
        ####
        def chunk_broadcast_tensor(v, axis, res_list):
            tensors = optimized_chunk(v, TP_SIZE, axis)
            for i in range(TP_SIZE):
                res_list[i].append((k, tensors[i]))

        ####
        def chunk_broadcast_tensor_dict(v, axis, res_list):
            tensor_dicts = []
            for i in range(TP_SIZE):
                tensor_dicts.append(dict())
            for optim_param_name, optim_param in v.items():
                if optim_param_name == "step":
                    for i in range(TP_SIZE):
                        tensor_dicts[i][optim_param_name] = deepcopy(optim_param)
                else:
                    tensors = optimized_chunk(optim_param, TP_SIZE, axis)
                    for i in range(TP_SIZE):
                        tensor_dicts[i][optim_param_name] = tensors[i]
            for i in range(TP_SIZE):
                res_list[i].append((k, tensor_dicts[i]))

        #### Column Parallel
        if is_substr_of_strings(k, COLUMN_PARALLEL_KEYS):
            if "embed_tokens" in k and args.output_tp_embedding_type == "EmbeddingParallelEmbedding":
                broadcast_index = 1
            else:
                broadcast_index = 0

            if isinstance(v, torch.Tensor):
                chunk_broadcast_tensor(v, broadcast_index, res_list)
            elif isinstance(v, dict):
                chunk_broadcast_tensor_dict(v, broadcast_index, res_list)
            else:
                raise ValueError(type(v))
        #### Row Parallel
        elif is_substr_of_strings(k, ROW_PARALLEL_KEYS):
            if isinstance(v, torch.Tensor):
                chunk_broadcast_tensor(v, 1, res_list)
            elif isinstance(v, dict):
                chunk_broadcast_tensor_dict(v, 1, res_list)
            else:
                raise ValueError(type(v))
        else:
            for i in range(TP_SIZE):
                res_list[i].append((k, deepcopy(v)))

    tp_state_models = []

    ### Chunk model states ###
    for _ in range(TP_SIZE):
        tp_state_models.append(list())

    for k, v in tqdm(ckpt["state"]["model"].items()):
        prepare_tensors(k, v, tp_state_models)

    for i in range(TP_SIZE):
        tp_ckpts[i]["state"]["model"] = OrderedDict(tp_state_models[i])
        config = get_language_model_config_from_fsdp_state_dict(tp_ckpts[i])
        config["tp_size"] = TP_SIZE

    if not args.skip_optimizer:
        ### Chunk and prepare optim states ###
        for i in range(TP_SIZE):
            tp_ckpts[i]["state"]["optimizers"] = dict()

        for optim_key, optim_states in ckpt["state"]["optimizers"].items():
            for i in range(TP_SIZE):
                tp_ckpts[i]["state"]["optimizers"][optim_key] = dict()
                tp_ckpts[i]["state"]["optimizers"][optim_key]["param_groups"] = deepcopy(
                    optim_states["param_groups"]
                )

            tp_state_optim = []
            for _ in range(TP_SIZE):
                tp_state_optim.append(list())
            for k, v in tqdm(optim_states["state"].items()):
                prepare_tensors(k, v, tp_state_optim)
            for i in range(TP_SIZE):
                tp_ckpts[i]["state"]["optimizers"][optim_key]["state"] = dict(tp_state_optim[i])

    return tp_ckpts


# GIGAFSDP MAPPERS


@sequence_apply
@time_memory_stat
def map_gigafsdp_to_fsdp(
    checkpoint_path: str,
    args: Optional[argparse.Namespace] = None
) -> Union[List[Dict], Dict]:
    from mapper_gigafsdp_to_monolith import unshard_gigafsdp_model,  unshard_gigafsdp_optimizer

    state_dicts = load_gigafsdp_checkpoint(checkpoint_path)
    

    meta = state_dicts[0]["state"]["model"]["model.meta"]
    mono_state_dict = copy.deepcopy(state_dicts[0])

    if 'monolith' in mono_state_dict['state']:
        mono_state_dict['state'].pop('monolith')

    unshard_state_dict = unshard_gigafsdp_model([sd['state']['model'] for sd in state_dicts], meta, split_convertation=True)
    assert len(unshard_state_dict) == 1, "Use map_mixed_gigafsdp_to_fsdp for mixed tp convertation"

    mono_state_dict['state']['model'] = unshard_state_dict[1][0] # tp_size=1 and it's the only one state_dict
    _normalize_single_shard_moe_expert_weights(
        mono_state_dict['state']['model'],
        get_language_model_config_from_fsdp_state_dict(mono_state_dict),
    )

    merge_lora = getattr(args, 'merge_lora', False)
    skip_optimizer = getattr(args, 'skip_optimizer', False)

    if merge_lora:
        mono_state_dict = map_merge_lora(mono_state_dict, args=args)
        
    if not skip_optimizer:
        mono_state_dict['state']['optimizers'] = unshard_gigafsdp_optimizer([sd['state']['optimizers']
                                                                             for sd in state_dicts], meta)
        model_config = get_language_model_config_from_fsdp_state_dict(mono_state_dict)
        for optimizer_sd in mono_state_dict['state']['optimizers'].values():
            _normalize_single_shard_moe_expert_optimizer_states(
                optimizer_sd['state'],
                model_config,
            )

    return mono_state_dict


def build_monolith_from_mixed_gigafsdp_state(
    state_dicts: List[Dict[str, Any]],
    args,
    *,
    select_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from mapper_gigafsdp_to_monolith import unshard_gigafsdp_model

    base0 = state_dicts[0]
    meta = base0['state']['model']['model.meta']

    cfg_content = base0["state"]["integrations"]["huggingface"]["model"]["config"]["content"]

    mono_state_dict = {"state": {}}
    for k, v in base0["state"].items():
        if k in ("model", "optimizers", "monolith"):
            continue
        mono_state_dict["state"][k] = copy.deepcopy(v)

    mono_state_dict["state"]["model"] = OrderedDict()

    mixed_tp_state_dicts = unshard_gigafsdp_model(
        [sd['state']['model'] for sd in state_dicts],
        meta,
        split_convertation=False,
        select_names=select_names,
    )

    local_args = copy.copy(args)
    local_args.skip_optimizer = True

    for tp_size, tp_models in mixed_tp_state_dicts.items():
        tmp_state_dicts = [
            {
                "state": {
                    "model": tp_model_sd,
                    "integrations": {
                        "huggingface": {
                            "model": {
                                "config": {"content": cfg_content}
                            }
                        }
                    },
                }
            }
            for tp_model_sd in tp_models
        ]

       
        lm_cfg = get_language_model_config_from_fsdp_state_dict(tmp_state_dicts[0]) or {}
        has_lora = 'lora_config' in cfg_content or 'lora' in lm_cfg
        if has_lora and getattr(args, 'merge_lora', False):
            logger.info('Start merge lora in to base model')
            # map_merge_lora expects a single state dict, apply to each TP shard
            tmp_state_dicts = [map_merge_lora(sd, args=args) for sd in tmp_state_dicts]

        tp_state_dicts = map_packed_to_base_attn(tmp_state_dicts, args=local_args)
        if tp_size == 1:
            _normalize_single_shard_moe_expert_weights(
                tp_state_dicts[0]['state']['model'],
                get_language_model_config_from_fsdp_state_dict(tp_state_dicts[0]),
            )
            mono_state_dict['state']['model'].update(
                tp_state_dicts[0]['state']['model']
            )
        else:
            merged = map_tp_to_monolith(tp_state_dicts, args=args)
            mono_state_dict['state']['model'].update(merged['state']['model'])
            mono_state_dict['state']['integrations'] = merged['state']['integrations']
        config = get_language_model_config_from_fsdp_state_dict(mono_state_dict)
        if config["model_type"] == "gigar" and "Packed" not in config["attention_type"]:
            logger.info("Skip to packed attention transformation for GigaR model")
        else:
            tp_state_dicts = map_base_to_packed_attn(mono_state_dict, args=local_args)

        del tmp_state_dicts, tp_state_dicts
        gc.collect()

    if not getattr(args, "skip_optimizer", True):
        raise NotImplementedError(
            "Optimizer is not supported for mixed-TP conversion yet."
        )

    # build_monolith always emits MoE weight1/weight2 in the 2D `(H, E*2I)` /
    # `(E*I, H)` "old layout" on-disk shape (via _normalize_single_shard_moe_expert_weights
    # for tp==1 or map_tp_to_monolith for tp>1). Declare the output as old-layout so
    # downstream loaders relying on `start_from_sf_checkpoint_in_old_expert_weight_layout`
    # don't trip the ckpt-vs-flag assert (gigafsdp/papayafsdp utils_checkpoint.py:463-467).
    if cfg_content.get("use_new_expert_weight_layout", False):
        mono_cfg = get_language_model_config_from_fsdp_state_dict(mono_state_dict)
        mono_cfg["use_new_expert_weight_layout"] = False

    return mono_state_dict


@time_memory_stat
def map_mixed_gigafsdp_to_fsdp(checkpoint_path: str, args=None):
    state_dicts = load_mixed_gigafsdp_checkpoint(checkpoint_path, args)
    state_dicts = map_ema_to_weights(state_dicts, args=args)
    return build_monolith_from_mixed_gigafsdp_state(state_dicts, args)

# FUSED MAPPERS

@time_memory_stat
def map_tp_to_tp(
    state_dicts: Union[List[Dict], Dict],
    args: Optional[argparse.Namespace] = None
) -> Union[List[Dict], Dict]:
    if args.input_tp_size == args.output_tp_size:
        return state_dicts
    state_dicts = map_packed_to_base_attn(state_dicts, args=args)
    if args.input_tp_size > 1:
        state_dicts = map_tp_to_monolith(state_dicts, args=args)
    if args.output_tp_size > 1:
        state_dicts = map_monolith_to_tp(state_dicts, args=args)
    state_dicts = map_base_to_packed_attn(state_dicts, args=args)

    return state_dicts


@time_memory_stat
def map_fsdp_to_hf(
    state_dicts: Union[List[Dict], Dict],
    args: Optional[argparse.Namespace] = None
) -> Union[List[Dict], Dict]:

    config = get_language_model_config_from_fsdp_state_dict(
        state_dicts[0] if isinstance(state_dicts, List) else state_dicts
    )

    if config['model_type'] != "giga_tts":
        # GigaTTS has 2 GigaR with different parameters (map_monolith has correct unpacking)
        state_dicts = map_packed_to_base_attn(state_dicts, args=args)
    if config["tp_size"] > 1:
        state_dicts = map_tp_to_monolith(state_dicts, args=args)
    map_monolith_fsdp_to_hf(state_dicts, args=args)


@time_memory_stat
def map_hf_to_fsdp(
    hf_model: nn.Module,
    args: Optional[argparse.Namespace] = None
) -> Union[List[Dict], Dict]:

    state_dicts = map_hf_to_monolith_fsdp(hf_model, args=args)
    if args.output_tp_size > 1:
        state_dicts = map_monolith_to_tp(state_dicts, args=args)
    state_dicts = map_base_to_packed_attn(state_dicts, args=args)
    return state_dicts


@time_memory_stat
def map_gigafsdp_to_hf(
    state_dicts: Union[List[Tuple[Dict]], Tuple[Dict]],
    args: Optional[argparse.Namespace] = None
) -> None:

    state_dicts = map_gigafsdp_to_fsdp(state_dicts, args=args)
    map_fsdp_to_hf(state_dicts, args=args)


# SAFETENSORS MAPPERS

@time_memory_stat
def stream_gigafsdp_to_safetensors(checkpoint_path: str, args, *, mapper: LayerMapper):
    from mapper_gigafsdp_to_monolith import _stream_core

    state_dicts = load_mixed_gigafsdp_checkpoint(checkpoint_path, args)

    state_dicts = map_ema_to_weights(state_dicts, args=args)

    if isinstance(state_dicts, Dict):
        all_keys = [n for n in state_dicts["state"]["model"].keys() if n != "model.meta"]
    else:
        all_keys = [n for n in state_dicts[0]["state"]["model"].keys() if n != "model.meta"]

    rank, world = _get_rank_world(args)
    chunk_layers = getattr(args, "chunk_layers", 2) or 2

    selected_for_rank = _select_keys_for_rank_by_bundles(all_keys, chunk_layers, rank, world)

    mono = build_monolith_from_mixed_gigafsdp_state(
        state_dicts, args, select_names=selected_for_rank
    )

    if hasattr(args, "remove_normalize_lora_keys_for_debug") and args.remove_normalize_lora_keys_for_debug:
        assert args.input_tp_size == 1

        # NOTE (fedorovgv): only for debug
        def _normalize_single_state_dict(sd):
            new_sd = {}
            for k, v in sd.items():
                if "lora" in k:
                    continue
                new_sd[k.replace(".base_layer", "")] = v
            _ = new_sd.keys()
            return new_sd

        mono['state']['model'] = _normalize_single_state_dict(mono['state']['model'])
        logger.info("TEST: normalize lora keys")

    return _stream_core(
        state_dicts=mono,
        output_dir=args.output_checkpoint_path,
        chunk_layers=chunk_layers,
        args=args,
        mapper=mapper,
    )


@time_memory_stat
def map_gigafsdp_to_fsdp_safetensor(checkpoint_path: str, args=None):
    return stream_gigafsdp_to_safetensors(checkpoint_path, args, mapper=RawMapper())


@time_memory_stat
def map_gigafsdp_to_omni_safetensor(checkpoint_path: str, args=None):
    return stream_gigafsdp_to_safetensors(checkpoint_path, args, mapper=OmniRawMapper())


@time_memory_stat
def map_gigafsdp_to_hf_safetensor(checkpoint_path: str, args=None):
    return stream_gigafsdp_to_safetensors(checkpoint_path, args, mapper=DeepseekV3Mapper())


@time_memory_stat
def map_gigafsdp_to_hf_omni_safetensor_stream(checkpoint_path: str, args=None):
    return stream_gigafsdp_to_safetensors(checkpoint_path, args, mapper=OmniDeepseekV3Mapper())


@time_memory_stat
def map_gigafsdp_dense_gqa_to_hf_safetensor(checkpoint_path: str, args=None):
    return stream_gigafsdp_to_safetensors(checkpoint_path, args, mapper=Gigachat35DenseMapper())


@time_memory_stat
def map_gigafsdp_slm_to_hf_safetensor(checkpoint_path: str, args=None):
    return stream_gigafsdp_to_safetensors(checkpoint_path, args, mapper=GigachatSLMMapper())


@time_memory_stat
def map_gfsdp2_dcp_to_hf(checkpoint_path: str, args=None):
    """torchtitan gfsdp2 DCP -> HF safetensors."""
    rank, world = _get_rank_world(args)
    chunk_layers = getattr(args, "chunk_layers", 2) or 2
    torch_dtype = getattr(args, "torch_dtype", None)
    export_dtype = getattr(torch, torch_dtype) if torch_dtype else None

    hf_config_path = getattr(args, "hf_config_path", None)
    if not hf_config_path or not os.path.isfile(hf_config_path):
        raise ValueError(
            "map_gfsdp2_dcp_to_hf needs 'hf_config_path' pointing to the "
            "model's HF config.json (raw gfsdp2 DCP checkpoints carry no "
            f"embedded config); got: {hf_config_path!r}"
        )
    with open(hf_config_path) as fh:
        hf_config = json.load(fh)

    tokenizer_path = getattr(args, "tokenizer_path", None)
    use_ema = bool(getattr(args, "use_ema", False))
    output_path = args.output_checkpoint_path

    if getattr(args, "finalize_only", False):
        finalize_hf_checkpoint(output_path, world)
        write_hf_metadata(output_path, hf_config_path, tokenizer_path, hf_config)
        return

    shapes = read_titan_param_shapes(checkpoint_path)

    convert_dcp_to_safetensors(
        checkpoint_path,
        output_path,
        transform=TitanToHF(hf_config),
        shapes=shapes,
        use_ema=use_ema,
        export_dtype=export_dtype,
        global_rank=rank,
        world_size=world,
        chunk_layers=chunk_layers,
    )
    if world == 1:
        finalize_hf_checkpoint(output_path, 1)
        write_hf_metadata(output_path, hf_config_path, tokenizer_path, hf_config)


@time_memory_stat
def map_gigafsdp_to_hf_omni_safetensor(checkpoint_path: str, args=None):
    rank, world = _get_rank_world(args)
    save_path = Path(args.output_checkpoint_path)
    save_llm_path = save_path / "language_model"
    sync_dir = save_path / ".convert_sync"
    if rank == 0:
        save_path.mkdir(parents=True, exist_ok=True)
    clear_barrier_state(sync_dir, rank, save_llm_path=save_llm_path)

    _file_barrier(sync_dir, rank, world, tag="llm_dir_ready")

    map_gigafsdp_to_hf_omni_safetensor_stream(
        checkpoint_path,
        args,
    )

    # Free streamed LLM tensors before barrier / Phase B.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _file_barrier(sync_dir, rank, world, tag="llm_stream")
    if rank != 0:
        return None

    logger.info("[rank 0] Merging language_model safetensor ranks")
    os.environ["WORLD_SIZE"] = "1"
    _merge_hf_llm_safetensors(
        str(args.output_checkpoint_path),
        args=args,
        mapper=OmniDeepseekV3Mapper(),
    )


@time_memory_stat
def map_hf_safetensor_to_local(checkpoint_path: str, args=None):
    """
    Reverse conversion: HF safetensors (FP8/bf16) → local training format (bf16).
    Supports DeepSeek-V3 / GigaChat 3.0 and GigaChat 3.5. Family and per-layer attention
    type are auto-detected from the HF config.json.
    Multi-node compatible. Each rank processes a subset of layers.
    After all ranks complete, run merge_safetensors_ranks.py on master.
    """
    rank, world = _get_rank_world(args)
    chunk_layers = getattr(args, "chunk_layers", 2) or 2
    output_path = args.output_checkpoint_path

    input_dir = checkpoint_path
    if isinstance(checkpoint_path, (list, tuple)):
        input_dir = checkpoint_path[0]
    input_dir = str(input_dir)

    with open(os.path.join(input_dir, "config.json")) as f:
        hf_config = json.load(f)

    block_size = tuple(
        hf_config.get("quantization_config", {}).get("weight_block_size", [128, 128])
    )

    if hf_config.get("moe_kernel") == "sonic" or hf_config.get("moe_impl") == "SonicGMMMoeBlock":
        moe_layout = "sonicmoe"
    else:
        moe_layout = getattr(args, "moe_layout", "ggemm") or "ggemm"

    stream_hf_to_local_safetensors(
        input_dir=input_dir,
        output_dir=output_path,
        rank=rank,
        world_size=world,
        chunk_layers=chunk_layers,
        moe_layout=moe_layout,
        block_size=block_size,
        hf_config=hf_config,
    )
    return None  # streaming save, no state_dict to return


@time_memory_stat
def map_local_safetensor_to_hf(checkpoint_path: str, args=None):
    """
    Convert local safetensors checkpoint (local naming, bf16) → HF DeepSeek-V3 safetensors.
    Multi-node compatible. Reads llm_foundry_config.yaml or config.json for model config.
    After all ranks complete, run merge_safetensors_ranks.py on master.
    """
    rank, world = _get_rank_world(args)
    chunk_layers = getattr(args, "chunk_layers", 1) or 1
    output_path = args.output_checkpoint_path

    logger.info(f"map_local_safetensor_to_hf: checkpoint_path type={type(checkpoint_path).__name__}, value={checkpoint_path!r}")
    logger.info(f"map_local_safetensor_to_hf: args.input_checkpoint_path={getattr(args, 'input_checkpoint_path', 'N/A')!r}")

    input_dir = checkpoint_path
    if isinstance(checkpoint_path, (list, tuple)):
        if len(checkpoint_path) == 0:
            # checkpoint_path is empty — fall back to args.input_checkpoint_path
            input_dir = str(getattr(args, "input_checkpoint_path", checkpoint_path))
        else:
            input_dir = checkpoint_path[0]
    input_dir = str(input_dir)

    stream_local_safetensor_to_hf(
        input_dir=input_dir,
        output_dir=output_path,
        rank=rank,
        world_size=world,
        chunk_layers=chunk_layers,
        args=args,
    )
    return None  # streaming save, no state_dict to return


@sequence_apply
@time_memory_stat
def map_merge_lora(fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None
) -> Dict:
    from peft import LoraConfig
    import math

    state_dict = fsdp_state_dict['state']['model']
    model_config = get_language_model_config_from_fsdp_state_dict(fsdp_state_dict)
    lora_config = LoraConfig(**model_config['lora']['args'])

    scaling = lora_config.lora_alpha / math.sqrt(lora_config.r) if lora_config.use_rslora else lora_config.lora_alpha / lora_config.r

    def matmul_lora_adapters_with_cast(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return (B.float() @ A.float()).bfloat16()

    # First pass: merge expert LoRA (GroupedLlamaMLP)
    # Collect expert base_layer keys to process them
    expert_base_keys = [k for k in state_dict if 'block_sparse_moe.experts.base_layer.weight1' in k]
    processed_expert_prefixes = set()
    for module in expert_base_keys:
        # Extract prefix like "model.model.layers.{i}.block_sparse_moe.experts."
        prefix = module.replace('base_layer.weight1', '')
        if prefix in processed_expert_prefixes:
            continue
        processed_expert_prefixes.add(prefix)

        base_w1_key = prefix + 'base_layer.weight1'
        base_w2_key = prefix + 'base_layer.weight2'
        lora_A1_key = prefix + 'lora_A1'
        lora_B1_key = prefix + 'lora_B1'
        lora_A2_key = prefix + 'lora_A2'
        lora_B2_key = prefix + 'lora_B2'
        merged_w1_key = prefix + 'weight1'
        merged_w2_key = prefix + 'weight2'

        if all(k in state_dict for k in [base_w1_key, base_w2_key, lora_A1_key, lora_B1_key, lora_A2_key, lora_B2_key]):
            base_w1 = state_dict[base_w1_key]  # [H, 2*D*E]
            base_w2 = state_dict[base_w2_key]  # [D*E, H]
            lora_A1 = state_dict[lora_A1_key]  # stored NT: [E, r, H]
            lora_B1 = state_dict[lora_B1_key]  # stored NT: [E, 2*D, r]
            lora_A2 = state_dict[lora_A2_key]  # stored NT: [E, r, D]
            lora_B2 = state_dict[lora_B2_key]  # stored NT: [E, H, r]

            E = lora_A1.shape[0]
            H = lora_A1.shape[2]

            # Merge weight1: view as [E, H, 2*D], add LoRA delta, reshape back
            # Transpose from stored NT layout [E, r, H] / [E, 2*D, r] to logical [E, H, r] / [E, r, 2*D]
            w1_viewed = base_w1.view(E, H, -1).float()
            lora_delta1 = torch.bmm(lora_A1.float().transpose(1, 2), lora_B1.float().transpose(1, 2))  # [E, H, 2*D]
            w1_merged = (w1_viewed + scaling * lora_delta1).bfloat16()
            state_dict[merged_w1_key] = w1_merged.reshape(base_w1.shape)

            # Merge weight2: view as [E, D, H], add LoRA delta, reshape back
            # Transpose from stored NT layout [E, r, D] / [E, H, r] to logical [E, D, r] / [E, r, H]
            D_per_expert = lora_A2.shape[2]
            w2_viewed = base_w2.view(E, D_per_expert, H).float()
            lora_delta2 = torch.bmm(lora_A2.float().transpose(1, 2), lora_B2.float().transpose(1, 2))  # [E, D, H]
            w2_merged = (w2_viewed + scaling * lora_delta2).bfloat16()
            state_dict[merged_w2_key] = w2_merged.reshape(base_w2.shape)

            # Clean up LoRA and base_layer keys
            del state_dict[base_w1_key], state_dict[base_w2_key]
            del state_dict[lora_A1_key], state_dict[lora_B1_key]
            del state_dict[lora_A2_key], state_dict[lora_B2_key]

            logger.info(f"Merged expert LoRA for {prefix}")

    # Second pass: merge regular LoRA (LoraLinear, LoraQKV)
    module_names = list(state_dict)
    for module in module_names:
        if 'base_layer' in module and 'block_sparse_moe.experts' in module:
            # Already handled in expert merge pass above
            continue
        elif 'base_layer' in module and "qkv_proj" not in module:
            A_key = module.replace('base_layer', 'lora_A.default')
            B_key = module.replace('base_layer', 'lora_B.default')

            merged_layer_key = module.replace('base_layer.', '')

            state_dict[merged_layer_key] = state_dict[module] + scaling * matmul_lora_adapters_with_cast(state_dict[A_key], state_dict[B_key])

            del state_dict[A_key], state_dict[B_key], state_dict[module]

        elif 'base_layer' in module and "qkv_proj" in module:
            q_A_key = module.replace('base_layer', 'q_proj.lora_A.default')
            q_B_key = module.replace('base_layer', 'q_proj.lora_B.default')

            k_A_key = module.replace('base_layer', 'k_proj.lora_A.default')
            k_B_key = module.replace('base_layer', 'k_proj.lora_B.default')

            v_A_key = module.replace('base_layer', 'v_proj.lora_A.default')
            v_B_key = module.replace('base_layer', 'v_proj.lora_B.default')

            merged_layer_key = module.replace('base_layer.', '')

            q_weight = matmul_lora_adapters_with_cast(state_dict[q_A_key], state_dict[q_B_key])
            k_weight = matmul_lora_adapters_with_cast(state_dict[k_A_key], state_dict[k_B_key])
            v_weight = matmul_lora_adapters_with_cast(state_dict[v_A_key], state_dict[v_B_key])

            concat_adapter = torch.cat([q_weight, k_weight, v_weight], dim=0)
            state_dict[merged_layer_key] = state_dict[module] + scaling * concat_adapter

            del state_dict[q_B_key], state_dict[q_A_key]
            del state_dict[k_B_key], state_dict[k_A_key]
            del state_dict[v_B_key], state_dict[v_A_key]
            del state_dict[module]

    return fsdp_state_dict


@time_memory_stat
def map_fsdp_lora_adapter_to_hf(fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None
) -> None:

    def _get_model_state(config, args, model_state):
        model_type = config.get("model_type")
        final_config = config

        if config.get("attention_bias"):
            final_config = dict(config)
            final_config["model_type"] = "qwen2"
            return final_config, model_state

        if model_type == "giga_mix":
            from transformers_utils.deepseek import _LoraDeepseekPerLayerTransformer
            transformer = _LoraDeepseekPerLayerTransformer(config)
            mapper = DeepseekV3Mapper(config, tr=transformer)
            logger.info(f'prepeare model state for DeepSeek V3')
            model_state["state"]["model"] = mapper.map(model_state["state"]["model"])
            return final_config, model_state

        final_config = dict(config)
        final_config["model_type"] = MAP_ARCH_NAMES.get(model_type, model_type)
        return final_config, model_state

    def _update_lora_target_modules(config: dict) -> None:
        """Updates LoRA target modules, splitting qkv into separate projections."""
        target_modules = config['lora_config']['target_modules']
        
        if isinstance(target_modules, list):
            final_targets = []
            for module in target_modules:
                if module == "qkv_proj":
                    final_targets.extend(["q_proj", "k_proj", "v_proj"])
                else:
                    final_targets.append(module)
            config['lora_config']['target_modules'] = final_targets
        else:
            if "qkv_proj" in target_modules:
                logger.info(f'target_modules: {target_modules}')
                config['lora_config']['target_modules'] = target_modules.replace("qkv_proj", "(q|k|v)_proj")
                logger.info(f'Updated lora_config target_modules: {config["lora_config"]["target_modules"]}')
            else:
                config['lora_config']['target_modules'] = target_modules
           
    
    def _create_lora_config(config: dict):

        config['lora_config'].pop('init_lora_weights', None)
        # Module mapping for concat for MoE
        if config.get("model_type") == 'giga_mix':
            # Check if expert LoRA was used
            target_modules = config['lora_config'].get('target_modules', [])
            has_expert_lora = isinstance(target_modules, list) and 'experts' in target_modules

            mapping = {
                'dkv_proj': 'kv_a_proj_with_mqa',
                'kr_proj': 'kv_a_proj_with_mqa',
                'uk_proj': 'kv_b_proj',
                'uv_proj': 'kv_b_proj',
            }

            # Remove "experts" from target_modules (not a valid HF PEFT target)
            # Expert LoRA is converted to per-expert gate/up/down_proj format
            filtered_targets = [
                item for item in target_modules if item != 'experts'
            ] if isinstance(target_modules, list) else target_modules

            # Apply mapping and get unique values
            config['lora_config']['target_modules'] = list(
                set(mapping.get(item, item) for item in filtered_targets)
            )
            logger.info(f'Updated lora_config target_modules: {config["lora_config"]["target_modules"]}')

            # Only exclude expert modules if expert LoRA was NOT used
            if not has_expert_lora:
                config['lora_config']['exclude_modules'] = r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
            else:
                logger.info('Expert LoRA detected - not excluding expert modules from PEFT config')
        
        else:
            _update_lora_target_modules(config)
        
        lora_config = LoraConfig(**config['lora_config'])
        return  lora_config


    def _calculate_attention_split_sizes(config: dict) -> Tuple[int, int, int]:
        """Calculates split sizes for QKV attention projection."""
        if config.get("attention_hidden_size", None):
            kv_hidden_size = config['attention_hidden_size'] // config['num_attention_heads'] * config['num_key_value_heads']
            attention_size = config['attention_hidden_size']
        else:    
            kv_hidden_size = config['hidden_size'] // config['num_attention_heads'] * config['num_key_value_heads']
            attention_size = config['hidden_size']
        
        return (
            attention_size // config["tp_size"],
            kv_hidden_size // config["tp_size"],
            kv_hidden_size // config["tp_size"],
        )
    

    def _process_qkv_weights_for_lora(state_dict_part: Dict, split_sizes: Tuple[int, int, int]) -> None:
        """
        Processes QKV weights by splitting packed qkv_proj into separate q, k, v projections.
        
        Handles both base layer weights and LoRA adapter weights.
        """
        patterns = {
            'base': 'qkv_proj.base_layer',
            'lora': 'self_attn.qkv_proj'
        }
        
        # Process base weights
        for key in get_matched_weight(state_dict_part, patterns['base']):
            prefixes = ['q_proj', 'k_proj', 'v_proj']
            new_keys = [key.replace("qkv_proj", p) for p in prefixes]
            
            tensors = torch.split(state_dict_part[key], split_sizes, dim=0)
            for new_key, tensor in zip(new_keys, tensors):
                state_dict_part[new_key] = tensor
            del state_dict_part[key]
        
        # Process LoRA A and B weights
        for qkv_proj_key in get_matched_weight(state_dict_part, patterns['lora']):
            for proj in ["q_proj", "k_proj", "v_proj"]:
                new_key = qkv_proj_key.replace(f"qkv_proj.{proj}", proj)
                state_dict_part[new_key] = state_dict_part[qkv_proj_key]
            
            del state_dict_part[qkv_proj_key]


    model_state = fsdp_state_dict
    tp_size = args.input_tp_size if args.input_tp_size else 1
    
    # Get and process model configuration
    config = get_hf_config_from_lora_state(model_state, tp_size)
    final_config,model_state = _get_model_state(config, args, model_state)

    # Create LoRA model
    lora_config = _create_lora_config(final_config)
   
    # Calculate split sizes for QKV attention
    split_sizes = _calculate_attention_split_sizes(final_config)

    # Process weights for each TP partition
    if final_config.get("model_type") != 'giga_mix':
        if tp_size > 1:
            for i in range(tp_size):
                _process_qkv_weights_for_lora(model_state[i]["state"]["model"], split_sizes)
        else:
            _process_qkv_weights_for_lora(model_state["state"]["model"], split_sizes)
    
    # Merge TP partitions if needed (skip if already monolith, e.g. from map_mixed_gigafsdp_to_fsdp)
    if tp_size > 1 and "state" not in model_state:
        model_state = map_tp_to_monolith(model_state, args=args)

    from safetensors.torch import save_file
    def extract_lora_state_dict(sd):
        lora_adapters = {}
        for k, v in sd.items():
            if ".lora_A" not in k and ".lora_B" not in k:
                continue

            nk = f'base_model.{k}'
            nk = nk.replace(".lora_A.default.weight", ".lora_A.weight")
            nk = nk.replace(".lora_B.default.weight", ".lora_B.weight")

            lora_adapters[nk] = v
            logger.info(f"lora_adapter: {nk}")
        return lora_adapters
           
    lora_sd = extract_lora_state_dict(model_state["state"]["model"])
    # Оставляем в адаптере только слои, входящие в базовую модель (0 .. num_hidden_layers-1),
    # иначе при загрузке в SGLang будет "index N is out of range"
    config_num_layers = final_config.get("num_hidden_layers") or final_config.get("n_layer")
    if config_num_layers is not None:
        filtered_lora_sd = {}
        for key in lora_sd:
            match = re.search(r"\.layers\.(\d+)\.", key)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx >= config_num_layers:
                    logger.warning(
                        "Dropping adapter keys for layer %s (>= num_hidden_layers=%s) to match base model",
                        layer_idx,
                        config_num_layers,
                    )
                    continue
            filtered_lora_sd[key] = lora_sd[key]
        lora_sd = filtered_lora_sd
    logger.info(f"saving adapter to {os.path.join(args.output_checkpoint_path, 'adapter_model.safetensors')}")
    os.makedirs(args.output_checkpoint_path, exist_ok=True)
    save_file(lora_sd, os.path.join(args.output_checkpoint_path, "adapter_model.safetensors"))
    lora_config.save_pretrained(args.output_checkpoint_path)
    


@sequence_apply
@time_memory_stat
def map_fsdp_to_omni_fsdp(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None,
) -> Union[List[Dict], Dict]:
    model_sd = fsdp_state_dict["state"]["model"]
    for name, param in list(fsdp_state_dict["state"]["model"].items()):
        omni_name = name.replace("model.model", "model.language_model.model") \
            .replace("model.lm_head", "model.language_model.lm_head")
        model_sd[omni_name] = param
        del model_sd[name]

    fsdp_state_dict["state"]["model"] = model_sd
    return fsdp_state_dict


@sequence_apply
@time_memory_stat
def map_vision_fsdp_to_omni_fsdp(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None,
) -> Union[List[Dict], Dict]:
    model_sd = fsdp_state_dict["state"]["model"]
    for name, param in list(fsdp_state_dict["state"]["model"].items()):
        omni_name = name.replace("model._language_model.model", "model.language_model.model") \
            .replace("model._language_model.lm_head", "model.language_model.lm_head") \
            .replace("model._mm_vision_tower.vision_tower.vision_tower", "model.encoders.InternViTForVisualLanguageModeling") \
            .replace("model._mm_projector.projector", "model.adapters.VisionPixelShuffleAdapter")
        model_sd[omni_name] = param
        del model_sd[name]

    fsdp_state_dict["state"]["model"] = model_sd
    return fsdp_state_dict


@time_memory_stat
def map_expand_embeddings_and_lm_head_fsdp(
    fsdp_state_dict: Dict,
    args: Optional[argparse.Namespace] = None,
) -> Union[List[Dict], Dict]:
    config = get_language_model_config_from_fsdp_state_dict(fsdp_state_dict)
    assert config["tp_size"] == 1, \
        "Can't expand emebddings and lm_head for checkpoint with tp_size > 1"

    def expand_weights(weights: torch.Tensor, expand_size: int) -> torch.Tensor:
        vocab_size, hidden_size = weights.shape
        new_weights = torch.zeros((vocab_size + expand_size, hidden_size), dtype=weights.dtype)
        new_weights.data.normal_(mean=0.0, std=0.02)
        new_weights[:vocab_size] = weights
        return new_weights

    for name, param in fsdp_state_dict["state"]["model"].items():
        if is_substr_of_strings(name, ["embed_tokens.weight", "lm_head.weight"]):
            fsdp_state_dict["state"]["model"][name] = expand_weights(
                param, args.embeddings_expand_size
            )
    return fsdp_state_dict


@sequence_apply
@time_memory_stat
def map_ema_to_weights(state_dict_object: Union[Iterator[Dict], Dict],
                       args: Optional[argparse.Namespace]) -> Union[Iterator[Dict], Dict]:
    if getattr(args, "skip_ema_mapper", False):
        return state_dict_object

    # Detect whether EMA was actually live at this checkpoint.
    # If checkpoint_batch < start_batch the EMA buffers were never updated and must not be
    # applied even when --use-ema=true is passed.
    first = state_dict_object if isinstance(state_dict_object, Dict) else next(iter(state_dict_object), None)
    if first is not None:
        checkpoint_batch, start_batch = _get_ema_activation_info(first)
        if (
            checkpoint_batch is not None
            and start_batch is not None
            and checkpoint_batch < start_batch
        ):
            if getattr(args, "use_ema", False):
                logger.warning(
                    f"--use-ema=true is set but EMA was not yet active at this checkpoint "
                    f"(checkpoint batch {checkpoint_batch} < start_batch {start_batch}). "
                    f"EMA weights will be deleted, not applied."
                )
            args = copy.copy(args)
            args.use_ema = False

    if isinstance(state_dict_object, Dict):
        put_ema_to_weights_or_delele_ema(state_dict_object, args)
    else:
        for state_dict in state_dict_object:
            put_ema_to_weights_or_delele_ema(state_dict, args)
    return state_dict_object


def _patch_mtp_heads(model_sd: dict, num_new_heads: int) -> None:
    head0_layer_keys = [k for k in model_sd if "mtp_block.mtp_layers.0." in k]
    head0_norm_keys  = [k for k in model_sd if "mtp_block.mtp_norms.0." in k]
    if not head0_layer_keys and not head0_norm_keys:
        return  # head0 not in this rank's slice
    sample = head0_layer_keys[0] if head0_layer_keys else head0_norm_keys[0]
    prefix = sample[:sample.index("mtp_block")]
    logger.info(f"_patch_mtp_heads: prefix='{prefix}', new heads={num_new_heads}")
    for new_idx in range(1, num_new_heads + 1):
        for k in head0_layer_keys:
            model_sd[k.replace("mtp_layers.0.", f"mtp_layers.{new_idx}.")] = copy.deepcopy(model_sd[k])
        for k in head0_norm_keys:
            model_sd[k.replace("mtp_norms.0.", f"mtp_norms.{new_idx}.")] = copy.deepcopy(model_sd[k])
        logger.info(f"  Replicated head 0 → head {new_idx}")


@time_memory_stat
def map_mixed_gigafsdp_to_mtp_safetensors(checkpoint_path: str, args=None):
    """Load gigafsdp distcp checkpoint, patch MTP heads, save as safetensors."""
    from utils.safetensors_local import _save_chunk_as_safetensors, _write_local_index

    state_dicts = load_mixed_gigafsdp_checkpoint(checkpoint_path, args)
    state_dicts = map_ema_to_weights(state_dicts, args=args)

    first = state_dicts if isinstance(state_dicts, dict) else state_dicts[0]
    all_keys = [k for k in first["state"]["model"].keys() if k != "model.meta"]

    rank, world = _get_rank_world(args)
    chunk_layers = getattr(args, "chunk_layers", 2) or 2
    num_new = getattr(args, "num_new_mtp_heads", 2)

    selected_for_rank = _select_keys_for_rank_by_bundles(all_keys, chunk_layers, rank, world)

    mono = build_monolith_from_mixed_gigafsdp_state(state_dicts, args, select_names=selected_for_rank)
    del state_dicts; gc.collect()

    model_sd  = mono["state"]["model"]
    hf_config = mono["state"]["integrations"]["huggingface"]["model"]["config"]["content"]
    logger.info(f"[rank {rank}/{world}] Monolith slice: {len(model_sd)} keys")

    if num_new > 0:
        _patch_mtp_heads(model_sd, num_new)  # no-op if head0 not in this rank's slice

    out_dir = Path(args.output_checkpoint_path)
    out_rank_dir = out_dir / f"rank{rank:02d}"
    out_rank_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        _write_hf_config_and_composer_state(out_dir, hf_config)

    shard_gb = getattr(args, "shard_gb", 4.0)
    shard_bytes = int(shard_gb * 1024 ** 3)
    cur, cur_size, chunk_idx = {}, 0, 0

    def _flush(cur, chunk_idx):
        if not cur:
            return
        fname = f"shard-r{rank:02d}-c{chunk_idx:05d}.safetensors"
        shard_size = _save_chunk_as_safetensors(cur, str(out_rank_dir / fname))
        with open(str(out_rank_dir / f"shard-r{rank:02d}-c{chunk_idx:05d}.weight_map.json"), "w") as fh:
            json.dump({"weight_map": {k: fname for k in cur}, "total_size": shard_size}, fh)
        logger.info(f"[rank {rank}] Wrote {fname} ({shard_size/1e9:.2f} GB, {len(cur)} tensors)")

    for k, v in model_sd.items():
        nb = v.nbytes
        if cur and cur_size + nb > shard_bytes:
            _flush(cur, chunk_idx); cur = {}; cur_size = 0; chunk_idx += 1
        cur[k] = v; cur_size += nb
    _flush(cur, chunk_idx)

    _write_local_index(str(out_rank_dir))
    logger.info(f"[rank {rank}] Done")
    return None


def _load_hf_weight_map(input_dir: str) -> Dict[str, str]:
    """Read HF safetensors weight_map (key -> shard filename) from index.json or by scanning shards."""
    import safetensors

    idx_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            return json.load(f)["weight_map"]
    wm: Dict[str, str] = {}
    for sf in sorted(Path(input_dir).glob("*.safetensors")):
        with safetensors.safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                wm[k] = sf.name
    return wm


def _write_hf_config_and_composer_state(out_dir: Path, hf_config: dict) -> None:
    """Write config.json + composer_state.pt alongside a rank-sharded MTP HF safetensors output."""
    (out_dir / "config.json").write_text(json.dumps(hf_config, indent=2))
    torch.save(
        {"state": {"integrations": {"huggingface": {"model": {"config": {"content": hf_config}}}}}},
        str(out_dir / "composer_state.pt"),
        pickle_module=dill,
    )


def _stream_safetensors_groups_to_shards(
    groups: Dict[Tuple[str, str], List[Tuple[str, str]]],
    out_rank_dir: Path,
    rank: int,
    shard_gb: float,
) -> None:
    """Stream tensors from grouped source shards into rank-local sharded safetensors output.

    groups: {(src_dir, src_shard_filename): [(src_key, dst_key), ...]}.
    On rename (src_key != dst_key) tensors are cloned so the output never sees mmap-shared storage.
    """
    import safetensors

    from utils.safetensors_local import _save_chunk_as_safetensors

    shard_bytes = int(shard_gb * 1024 ** 3)
    cur: Dict[str, torch.Tensor] = {}
    cur_size = 0
    chunk_idx = 0

    def _flush(cur_local, chunk_idx_local):
        if not cur_local:
            return
        fname = f"shard-r{rank:02d}-c{chunk_idx_local:05d}.safetensors"
        shard_size = _save_chunk_as_safetensors(cur_local, str(out_rank_dir / fname))
        with open(str(out_rank_dir / f"shard-r{rank:02d}-c{chunk_idx_local:05d}.weight_map.json"), "w") as fh:
            json.dump({"weight_map": {k: fname for k in cur_local}, "total_size": shard_size}, fh)
        logger.info(f"[rank {rank}] Wrote {fname} ({shard_size/1e9:.2f} GB, {len(cur_local)} tensors)")

    for (src_dir, src_shard), pairs in groups.items():
        shard_path = os.path.join(src_dir, src_shard)
        with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
            for src_k, dst_k in pairs:
                t = f.get_tensor(src_k)
                if src_k != dst_k:
                    t = t.clone()
                nb = t.numel() * t.element_size()
                if cur and cur_size + nb > shard_bytes:
                    _flush(cur, chunk_idx)
                    cur = {}
                    cur_size = 0
                    chunk_idx += 1
                cur[dst_k] = t
                cur_size += nb
    _flush(cur, chunk_idx)


@time_memory_stat
def map_safetensors_to_mtp_safetensors(checkpoint_path: str, args=None):
    """Read an HF safetensors checkpoint, replicate MTP head 0 into extra heads, save as safetensors.

    HF DeepSeek-style convention: MTP head i lives at `model.layers.{num_hidden_layers + i}.*`.
    Input must already carry head 0 (config.json `num_nextn_predict_layers >= 1`). Naming stays in
    HF format end-to-end; downstream `map_hf_safetensor_to_local` can convert to local-monolith.
    """
    from utils.safetensors_local import _write_local_index

    input_dir = checkpoint_path
    if isinstance(checkpoint_path, (list, tuple)):
        input_dir = checkpoint_path[0]
    input_dir = str(input_dir)

    rank, world = _get_rank_world(args)
    chunk_layers = getattr(args, "chunk_layers", 2) or 2
    num_new = getattr(args, "num_new_mtp_heads", 2)

    with open(os.path.join(input_dir, "config.json")) as f:
        hf_config = json.load(f)

    num_hidden_layers = int(hf_config["num_hidden_layers"])
    base_mtp = int(hf_config.get("num_nextn_predict_layers", 0))
    if base_mtp < 1:
        raise ValueError(
            f"num_nextn_predict_layers={base_mtp} in {input_dir}/config.json; nothing to replicate. "
            "Expected an MTP-enabled HF DeepSeek-style checkpoint."
        )

    weight_map = _load_hf_weight_map(input_dir)
    src_all_keys = sorted(weight_map.keys())

    head0_prefix = f"model.layers.{num_hidden_layers}."
    head0_keys = [k for k in src_all_keys if k.startswith(head0_prefix)]
    if not head0_keys:
        raise ValueError(
            f"No MTP head 0 keys found under prefix '{head0_prefix}' in {input_dir}."
        )
    logger.info(
        f"map_safetensors_to_mtp_safetensors: num_hidden_layers={num_hidden_layers}, "
        f"base_mtp={base_mtp}, num_new={num_new}, head0_keys={len(head0_keys)}"
    )

    new_keys: List[str] = []
    src_for_new: Dict[str, str] = {}
    for j in range(num_new):
        dst_prefix = f"model.layers.{num_hidden_layers + base_mtp + j}."
        for k in head0_keys:
            nk = dst_prefix + k[len(head0_prefix):]
            new_keys.append(nk)
            src_for_new[nk] = k

    target_all_keys = src_all_keys + new_keys
    selected = _select_keys_for_rank_by_bundles(target_all_keys, chunk_layers, rank, world)
    n_orig = sum(1 for k in selected if k in weight_map)
    n_new = len(selected) - n_orig
    logger.info(
        f"[rank {rank}/{world}] selected {len(selected)} keys ({n_orig} original, {n_new} replicated)"
    )

    out_dir = Path(args.output_checkpoint_path)
    out_rank_dir = out_dir / f"rank{rank:02d}"
    out_rank_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        new_config = copy.deepcopy(hf_config)
        new_config["num_nextn_predict_layers"] = base_mtp + num_new
        _write_hf_config_and_composer_state(out_dir, new_config)
        logger.info(
            f"[rank 0] Wrote config.json + composer_state.pt "
            f"(num_nextn_predict_layers={new_config['num_nextn_predict_layers']})"
        )

    groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for k in selected:
        src_k = k if k in weight_map else src_for_new[k]
        groups.setdefault((input_dir, weight_map[src_k]), []).append((src_k, k))

    shard_gb = getattr(args, "shard_gb", 4.0)
    _stream_safetensors_groups_to_shards(groups, out_rank_dir, rank, shard_gb)

    _write_local_index(str(out_rank_dir))
    logger.info(f"[rank {rank}] Done")
    return None


@time_memory_stat
def map_safetensors_graft_mtp_safetensors(checkpoint_path: str, args=None):
    """Graft MTP heads from a donor HF safetensors checkpoint onto a base HF safetensors checkpoint.

    HF DeepSeek-style convention: MTP head i lives at `model.layers.{num_hidden_layers + i}.*`.
    The base keeps its regular layers and its own MTP heads; selected donor heads are appended at
    consecutive new layer indices, and config.json `num_nextn_predict_layers` is bumped to match.
    Naming stays HF end-to-end. The base config (model_type / architecture) is preserved, except
    `nextn_is_linear` / `nextn_is_sparse`, which follow the grafted donor heads; base heads that
    stay must already run in the donor's mode, since one config describes every head.

    args:
      mtp_donor_path:     donor checkpoint dir (the source of the extra heads). Required.
      donor_mtp_heads:    optional list of 0-indexed donor head indices to graft (head 0 = first MTP
                          head at layer num_hidden_layers). Default: every donor head whose index is
                          >= the base's existing head count (i.e. all heads the base doesn't have).
      replace_base_heads: if True, donor head d overwrites base head d in-place (target layer
                          stays at n_hidden + d, base's tensors for that slot are dropped, and
                          num_nextn_predict_layers is NOT bumped). Each donor index must satisfy
                          d < base_M. Default False (append after base heads, bump config).
    """
    from utils.safetensors_local import _write_local_index
    from utils.hf_assembly_manifest import write_hf_assembly_manifest
    from utils.llmfoundry_config_utils import is_gigachat_model
    from utils.mtp_attention import resolve_grafted_mtp_fields

    def _as_dir(p):
        if isinstance(p, (list, tuple)):
            p = p[0]
        return str(p)

    def _mtp_layer_ids(weight_map, n_hidden):
        ids = set()
        for k in weight_map:
            m = re.match(r"model\.layers\.(\d+)\.", k)
            if m and int(m.group(1)) >= n_hidden:
                ids.add(int(m.group(1)))
        return sorted(ids)

    base_dir = _as_dir(checkpoint_path)
    donor_dir = getattr(args, "mtp_donor_path", None)
    if not donor_dir:
        raise ValueError("map_safetensors_graft_mtp_safetensors requires args.mtp_donor_path")
    donor_dir = _as_dir(donor_dir)

    rank, world = _get_rank_world(args)
    chunk_layers = getattr(args, "chunk_layers", 2) or 2

    with open(os.path.join(base_dir, "config.json")) as f:
        base_config = json.load(f)
    with open(os.path.join(donor_dir, "config.json")) as f:
        donor_config = json.load(f)

    n_hidden = int(base_config["num_hidden_layers"])
    if int(donor_config["num_hidden_layers"]) != n_hidden:
        raise ValueError(
            f"num_hidden_layers mismatch: base={n_hidden}, donor={donor_config['num_hidden_layers']}"
        )

    base_wm = _load_hf_weight_map(base_dir)
    donor_wm = _load_hf_weight_map(donor_dir)

    # Detect actual MTP heads from layer indices (config.num_nextn_predict_layers can be stale).
    base_layer_ids = _mtp_layer_ids(base_wm, n_hidden)
    donor_layer_ids = _mtp_layer_ids(donor_wm, n_hidden)

    for label, ids in (("base", base_layer_ids), ("donor", donor_layer_ids)):
        expected = list(range(n_hidden, n_hidden + len(ids)))
        if ids != expected:
            raise ValueError(
                f"{label} MTP layers must be contiguous starting at layer {n_hidden} "
                f"(this mapper assumes head index == layer offset from n_hidden). "
                f"Got {ids}, expected {expected}."
            )

    base_M = len(base_layer_ids)
    donor_M = len(donor_layer_ids)

    donor_heads = getattr(args, "donor_mtp_heads", None)
    if donor_heads is None:
        donor_heads = list(range(base_M, donor_M))  # everything the base doesn't already have
    donor_heads = [int(h) for h in donor_heads]
    for h in donor_heads:
        if not (0 <= h < donor_M):
            raise ValueError(f"donor head index {h} out of range [0,{donor_M})")
    if not donor_heads:
        raise ValueError(
            f"No donor heads to graft (base_M={base_M}, donor_M={donor_M}). "
            "Set args.donor_mtp_heads explicitly."
        )

    replace_base_heads = bool(getattr(args, "replace_base_heads", False))
    if replace_base_heads:
        for h in donor_heads:
            if h >= base_M:
                raise ValueError(
                    f"replace_base_heads=True but donor head {h} has no matching base slot "
                    f"(base_M={base_M}); cannot replace a slot the base doesn't have."
                )
        tgt_layers = [n_hidden + d for d in donor_heads]
    else:
        tgt_layers = [n_hidden + base_M + j for j in range(len(donor_heads))]
    logger.info(
        f"graft_mtp: n_hidden={n_hidden}, base_M={base_M}, donor_M={donor_M}, "
        f"replace_base_heads={replace_base_heads}; "
        f"grafting donor heads {donor_heads} (layers {[n_hidden + d for d in donor_heads]}) "
        f"-> target layers {tgt_layers}"
    )

    keep_base_heads = base_M > 0 and (
        not replace_base_heads or set(donor_heads) != set(range(base_M))
    )
    mtp_fields = resolve_grafted_mtp_fields(base_config, donor_config, keep_base_heads=keep_base_heads)

    if replace_base_heads:
        replaced_prefixes = tuple(f"model.layers.{lid}." for lid in tgt_layers)
        base_all_keys = sorted(k for k in base_wm if not k.startswith(replaced_prefixes))
    else:
        base_all_keys = sorted(base_wm.keys())
    grafted_keys: List[str] = []
    src_for_graft: Dict[str, str] = {}
    for j, d in enumerate(donor_heads):
        dp = f"model.layers.{n_hidden + d}."
        donor_keys_d = [k for k in donor_wm if k.startswith(dp)]
        if not donor_keys_d:
            raise ValueError(f"donor head {d} has no keys under prefix '{dp}' in {donor_dir}")
        tgt_prefix = f"model.layers.{tgt_layers[j]}."
        for dk in donor_keys_d:
            tk = tgt_prefix + dk[len(dp):]
            grafted_keys.append(tk)
            src_for_graft[tk] = dk

    grafted_set = set(grafted_keys)
    target_all_keys = base_all_keys + grafted_keys
    selected = _select_keys_for_rank_by_bundles(target_all_keys, chunk_layers, rank, world)
    n_graft_sel = sum(1 for k in selected if k in grafted_set)
    n_base_sel = len(selected) - n_graft_sel
    logger.info(
        f"[rank {rank}/{world}] selected {len(selected)} keys ({n_base_sel} base, {n_graft_sel} grafted)"
    )

    out_dir = Path(args.output_checkpoint_path)
    out_rank_dir = out_dir / f"rank{rank:02d}"
    out_rank_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        new_config = copy.deepcopy(base_config)
        new_config["num_nextn_predict_layers"] = base_M if replace_base_heads else base_M + len(donor_heads)
        new_config.update(mtp_fields)
        _write_hf_config_and_composer_state(out_dir, new_config)
        logger.info(
            f"[rank 0] Wrote config.json + composer_state.pt "
            f"(num_nextn_predict_layers={new_config['num_nextn_predict_layers']}, {mtp_fields})"
        )

        manifest_ns = types.SimpleNamespace(
            tokenizer_path=getattr(args, "tokenizer_path", None) or os.path.abspath(base_dir),
            torch_dtype=getattr(args, "torch_dtype", None) or new_config.get("torch_dtype"),
            input_checkpoint_path=getattr(args, "input_checkpoint_path", None) or os.path.abspath(base_dir),
        )
        model_family = "gigachat3_5" if is_gigachat_model(new_config) else "deepseek_v3"
        write_hf_assembly_manifest(str(out_dir), manifest_ns, model_family=model_family)
        logger.info(
            f"[rank 0] Wrote hf_assembly_plan.json "
            f"(model_family={model_family}, tokenizer_path={manifest_ns.tokenizer_path})"
        )

    # Group selected keys by their physical source (dir, shard) for one open per shard.
    groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for k in selected:
        if k in grafted_set:
            dk = src_for_graft[k]
            groups.setdefault((donor_dir, donor_wm[dk]), []).append((dk, k))
        else:
            groups.setdefault((base_dir, base_wm[k]), []).append((k, k))

    shard_gb = getattr(args, "shard_gb", 4.0)
    _stream_safetensors_groups_to_shards(groups, out_rank_dir, rank, shard_gb)

    _write_local_index(str(out_rank_dir))
    logger.info(f"[rank {rank}] Done")
    return None


MAP_ARCH_NAMES = {
    "gigar": "llama",
    "papayar": "llama",
}


SUPPORTED_ARCHS = {
    "LlamaForCausalLM",
    "LlamaModel",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
}


MAP_MODEL_TYPE_TO_ARCH = {
    "llama": "LlamaForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
}


MAP_CHECKPOINT_FUNCTIONS = {
    "load_checkpoint": load_checkpoint,
    "load_mixed_gigafsdp_checkpoint": load_mixed_gigafsdp_checkpoint,
    "load_hf_model": load_hf_model,
    "remove_fsdp_optimizer": remove_fsdp_optimizer,
    "remove_gigafsdp_optimizer": remove_gigafsdp_optimizer,
    "map_base_to_packed_attn": map_base_to_packed_attn,
    "map_packed_to_base_attn": map_packed_to_base_attn,
    "map_tp_to_monolith": map_tp_to_monolith,
    "map_monolith_to_tp": map_monolith_to_tp,
    "map_tp_to_tp": map_tp_to_tp,
    "map_gigafsdp_to_fsdp": map_gigafsdp_to_fsdp,
    "map_mixed_gigafsdp_to_fsdp": map_mixed_gigafsdp_to_fsdp,
    "map_mixed_gigafsdp_to_mtp_safetensors": map_mixed_gigafsdp_to_mtp_safetensors,
    "map_safetensors_to_mtp_safetensors": map_safetensors_to_mtp_safetensors,
    "map_safetensors_graft_mtp_safetensors": map_safetensors_graft_mtp_safetensors,
    "map_monolith_fsdp_to_hf": map_monolith_fsdp_to_hf,
    "map_hf_to_monolith_fsdp": map_hf_to_monolith_fsdp,
    "map_fsdp_to_hf": map_fsdp_to_hf,
    "map_hf_to_fsdp": map_hf_to_fsdp,
    "map_gigafsdp_to_hf": map_gigafsdp_to_hf,
    "map_merge_lora": map_merge_lora,
    "map_fsdp_lora_adapter_to_hf": map_fsdp_lora_adapter_to_hf,
    "map_fsdp_to_omni_fsdp": map_fsdp_to_omni_fsdp,
    "map_vision_fsdp_to_omni_fsdp": map_vision_fsdp_to_omni_fsdp,
    "map_expand_embeddings_and_lm_head_fsdp": map_expand_embeddings_and_lm_head_fsdp,
    "map_ema_to_weights":  map_ema_to_weights,
    "map_gigafsdp_to_fsdp_safetensor": map_gigafsdp_to_fsdp_safetensor,
    "map_gigafsdp_to_omni_safetensor": map_gigafsdp_to_omni_safetensor,
    "map_gigafsdp_to_hf_safetensor": map_gigafsdp_to_hf_safetensor,
    "map_gigafsdp_dense_gqa_to_hf_safetensor": map_gigafsdp_dense_gqa_to_hf_safetensor,
    "map_gigafsdp_slm_to_hf_safetensor": map_gigafsdp_slm_to_hf_safetensor,
    "map_gigafsdp_to_hf_omni_safetensor": map_gigafsdp_to_hf_omni_safetensor,
    "map_gfsdp2_dcp_to_hf": map_gfsdp2_dcp_to_hf,
    "map_ggemm_to_sonic": map_ggemm_to_sonic,
    "map_sonic_to_ggemm": map_sonic_to_ggemm,
    "map_safetensor_ggemm_to_sonic": map_safetensor_ggemm_to_sonic,
    "map_safetensor_sonic_to_ggemm": map_safetensor_sonic_to_ggemm,
    "map_hf_safetensor_to_local": map_hf_safetensor_to_local,
    "map_local_safetensor_to_hf": map_local_safetensor_to_hf,
}
