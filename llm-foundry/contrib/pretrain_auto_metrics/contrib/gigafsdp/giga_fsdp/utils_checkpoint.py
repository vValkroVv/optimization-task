import gc
import pickle
import hashlib
import logging
import typing as tp
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, OrderedDict


import torch
import numpy as np

from composer.utils import dist

from .common import EMA_SUFFIX
from .utils_safetensors import SafeMonoSD
from .utils import _normalize_module_name, _model_config_get
from .meta_param import WeightTPSplitType

logger = logging.getLogger(__name__)


@dataclass
class Shard:
    name: str
    weight_names: tp.List[str]
    weight_orig_shapes: tp.List[tp.Tuple[int, ...]]
    local_size: int
    global_size: int
    pg_size: tp.Tuple[int, int]
    parallel_type: tp.List[str]
    """ Shard meta dataclass.
    Args:
        local_size (tp.Optional[int], optional): shard size after fsdp partitioning.
        global_size (tp.Optional[int], optional): shard size before fsdp partitioning.
        pg_size (tp.Optional[tp.Tuple[int, int]], optional): model and data parallel group sizes.
        parallel_type (tp.Optional[tp.List[str]], optional): parallel type for each param in the shard.
    """


def sha256_hash(data: tp.List[str]) -> str:
    data_bytes = pickle.dumps(tuple(data))
    hash_ = hashlib.sha256(data_bytes).digest()
    return hash_.hex()


def prepeare_weights_to_shard_name_map(weight_name_to_shard_name: tp.Dict[str, str],) -> tp.Dict[str, str]:
    """ Build a map from shard weights to shard name, e.g.
        ["model.self_attn.up_proj", "model.self_attn.down_proj", ..] to "model.self_attn.flatten".
        but replace list with names to hash of the list.
    """
    shard_name_to_weights_names: tp.Dict[str, tp.List[str]] = defaultdict(list)
    for key, value in weight_name_to_shard_name.items():
        shard_name_to_weights_names[value].append(key)
    shard_name_to_weights_names = dict(shard_name_to_weights_names)
    weights_to_shard_name = {sha256_hash(v): k for k, v in shard_name_to_weights_names.items()}
    return weights_to_shard_name


def prepeare_shards_meta_info(meta_infos: tp.Dict) -> tp.List[Shard]:
    """ Prepeare information about each shard as Shard dataclass object.
    """
    weight_name_to_shard_name = meta_infos['param_name_to_layer_name']
    weights_to_shard_name = prepeare_weights_to_shard_name_map(weight_name_to_shard_name)

    # prepeare shards info
    param_groups = ["main", "freezed", "layernorm", "layernormfreezed"]

    shards_dc = []
    for param_group in param_groups:
        (
            weights_name,
            local_shard_sizes,
            total_shard_sizes,
            weights_orig_shape,
            pd_sizes,
            parallel_types,
        ) = (
            meta_infos[f"{param_group}_shards_fqns"],
            meta_infos[f"{param_group}_parameter_shard_bounaries"],
            meta_infos[f"{param_group}_shards_numels"],
            meta_infos[f"{param_group}_parameter_orig_shapes"],
            meta_infos[f"{param_group}_parameter_pg_size"],
            meta_infos[f"{param_group}_parameter_parallel_type"],
        )

        shard_names = []
        for weight_names_ in weights_name:
            shard_name = weights_to_shard_name[sha256_hash(weight_names_)]
            shard_names.append(shard_name)

        for (shard_name, weights_names, weights_orig_shapes, local_shard_size, total_shard_size, pd_size, parallel_type) in zip(
            shard_names, weights_name, weights_orig_shape, local_shard_sizes, total_shard_sizes, pd_sizes, parallel_types
        ):
            shard = Shard(shard_name, weights_names, weights_orig_shapes, local_shard_size, total_shard_size, pd_size, parallel_type)
            shards_dc.append(shard)
    return shards_dc


def split_tp_weight(weight: torch.Tensor,
                    tp_rank: tp.Optional[int],
                    tp_size: tp.Optional[int],
                    parallel_type: tp.Optional[str],
                    ) -> torch.Tensor:
    if parallel_type == WeightTPSplitType.COLUMN_PARALLEL.value:
        return torch.chunk(weight, chunks=tp_size, dim=0)[tp_rank].clone()

    elif parallel_type == WeightTPSplitType.ROW_PARALLEL.value:
        return torch.chunk(weight, chunks=tp_size, dim=1)[tp_rank].clone()
    else:
        return weight


def get_slice(weight: torch.Tensor,
              weight_key: str,
              ep_rank: tp.Optional[int],
              ep_size: tp.Optional[int],
              tp_rank: tp.Optional[int],
              tp_size: tp.Optional[int],
              parallel_type: tp.Optional[str],
              *,
              safe_mono_sd: SafeMonoSD,
              use_new_expert_weight_layout: bool = False,
              ) -> torch.Tensor:
    """ Dynamic split tp/ep layers and get nessesary part.
    """
    ep_rank_eff = 0 if ep_rank is None else ep_rank
    ep_size_eff = 1 if ep_size is None else ep_size

    if "block_sparse_moe.experts.weight1" in weight_key:
        mono_sd_meta = safe_mono_sd.load_mono_sd_meta()
        model_config = mono_sd_meta['state']['integrations']['huggingface']['model']['config']
        content = model_config["content"]
        if "num_routed_experts" in content:
            num_routed_experts = int(content["num_routed_experts"])
        else:
            num_routed_experts = int(content["language_model"]["num_routed_experts"])

        hidden_size = weight.shape[0]
        # hidden_size x (2 * intermediate_size * num_local_experts)
        #      -> num_routed_experts x hidden_size x (2 * intermediate_size)
        weight = weight.reshape(num_routed_experts, hidden_size, -1)
        # num_routed_experts x hidden_size x (2 * intermediate_size)
        #      -> num_local_experts x hidden_size x (2 * intermediate_size)
        weight = torch.chunk(weight, ep_size_eff, dim=0)[ep_rank_eff]

        if use_new_expert_weight_layout:
            # num_local_experts x hidden_size x (2 * intermediate_size)
            #     -> num_local_experts x (2 * intermediate_size) x hidden_size
            weight = weight.mT.contiguous()
        else:
            weight = weight.reshape(hidden_size, -1)

    elif "block_sparse_moe.experts.weight2" in weight_key:
        if use_new_expert_weight_layout:
            mono_sd_meta = safe_mono_sd.load_mono_sd_meta()
            model_config = mono_sd_meta['state']['integrations']['huggingface']['model']['config']
            content = model_config["content"]
            if "num_routed_experts" in content:
                num_routed_experts = int(content["num_routed_experts"])
            else:
                num_routed_experts = int(content["language_model"]["num_routed_experts"])

            hidden_size = weight.shape[-1]

            # weight: (num_routed_experts * intermediate_size, hidden_size)
            assert weight.shape[0] % num_routed_experts == 0, (
                f"weight2 bad shape: rows={weight.shape[0]} not divisible by "
                f"num_routed_experts={num_routed_experts}"
            )
            intermediate_size = weight.shape[0] // num_routed_experts
            weight = weight.reshape(num_routed_experts, intermediate_size, hidden_size)

            assert 0 <= ep_rank_eff < ep_size_eff, f"ep_rank={ep_rank_eff} is out of range [0, {ep_size_eff})"
            weight = torch.chunk(weight, ep_size_eff, dim=0)[ep_rank_eff]
            weight = weight.mT.contiguous()
        else:
            weight = torch.chunk(weight, ep_size_eff, dim=0)[ep_rank_eff]

    if tp_rank is not None:
        mono_sd_meta = safe_mono_sd.load_mono_sd_meta()
        config = mono_sd_meta["state"]["integrations"]["huggingface"]["model"]["config"]["content"]
        if config["model_type"] == "giga_tts":
            config = config["llm_config"]["config"]
        elif config["model_type"] == "gigaomni":
            config = config["language_model"]

        if config.get("attention_hidden_size", None):
            kv_hidden_size = config['attention_hidden_size'] // config['num_attention_heads'] * config['num_key_value_heads']
            attention_size = config['attention_hidden_size']
        else:
            num_key_value_heads = config['num_key_value_heads']
            if num_key_value_heads is None:
                num_key_value_heads = config['num_attention_heads']
            kv_hidden_size = config['hidden_size'] // config['num_attention_heads'] * num_key_value_heads
            attention_size = config['hidden_size']
        if "qkv_proj" in weight_key:
            q, k, v = torch.split(weight, (attention_size, kv_hidden_size, kv_hidden_size), dim=0)
            q = split_tp_weight(q, tp_rank, tp_size, parallel_type)
            k = split_tp_weight(k, tp_rank, tp_size, parallel_type)
            v = split_tp_weight(v, tp_rank, tp_size, parallel_type)
            weight = torch.cat((q, k, v), dim=0)
        else:
            weight = split_tp_weight(weight, tp_rank, tp_size, parallel_type)

    weight_per_ep_rank = weight.flatten()
    return weight_per_ep_rank


def make_shard_from_safetensors_mono_model_sd(*,
                                              shard: Shard,
                                              model_name: str = "model",
                                              is_ema_shard: bool = False,
                                              load_strict: bool = False,
                                              safe_mono_sd: tp.Optional[SafeMonoSD] = None,
                                              local_gigafsdp_model_state_dict: tp.Optional[tp.Dict] = None,
                                              shard_name_in_model: tp.Optional[str] = None,
                                              gigafsdp_model_normalize_fqns: tp.Optional[tp.Callable] = None,
                                              hybrid_group_size: tp.Optional[int] = 1,
                                              use_new_expert_weight_layout: bool = False) -> OrderedDict:
    """ Process single shard. We iterate through the shard weights, reading only the portion that
        belongs to this shard.
    """
    assert (
        shard.weight_names is not None
        and shard.weight_orig_shapes is not None
        and len(shard.weight_names) == len(shard.weight_orig_shapes)
    ), "uncorrect shard"

    # assume that groups are ordered in such way:
    #
    # (fsdp,tp/ep)  | (0, 0) |(0, 1) | ..., |(0, N)  |(1, 0)  |(1, 1)  | ..., | (K-1, 0) | ..., | (K-1, N)
    # global rank   | 0      | 1     | ..., | N      | N+1    | N+2    |...,  |   N*(K-1)| ..., | N*K- 1
    # so can compute fsdp, tp/ep ranks from global rank using formulas
    # TODO: fix this logic when you change group placement

    ep_rank = dist.get_global_rank() % shard.pg_size[1] if dist.get_ep_group_rank() is not None else None
    ep_size = shard.pg_size[1] if dist.get_ep_group_size() is not None else None
    tp_rank = dist.get_global_rank() % shard.pg_size[1] if dist.get_tp_group_rank() is not None else None
    tp_size = shard.pg_size[1] if dist.get_tp_group_size() is not None else None
    assert ep_rank is None or tp_rank is None, "ep and tp can't be enabled at the same time"

    if hybrid_group_size is not None:
        if dist.get_ep_group_size() is not None:
            assert shard.pg_size[0]*shard.pg_size[1] == hybrid_group_size, f"{shard.name} pg_size doesn't match hybrid_group_size: {shard.pg_size[0]}*{shard.pg_size[1]} != {hybrid_group_size}"

    fsdp_group_rank = dist.get_global_rank() // shard.pg_size[1] % shard.pg_size[0]

    offset_in_weighs_start = shard.local_size * fsdp_group_rank
    offset_in_weighs_end = shard.local_size * (fsdp_group_rank + 1)

    logger.info(f"GigaFSDP safetensors: currently trying to assemble a shard with a name {shard.name} " +
                "from safetensors checkpoint.")

    flat_shard = []
    cur_total_param = 0

    for weight_name, weight_orig_shape, parallel_type in zip(shard.weight_names, shard.weight_orig_shapes, shard.parallel_type):
        weight_name = f"{model_name}.{weight_name}"
        weight_name = weight_name + (f"{EMA_SUFFIX}" if is_ema_shard else "")
        flat_weight_size = np.prod(weight_orig_shape)

        if cur_total_param + flat_weight_size >= offset_in_weighs_start:
            weight_start = max(offset_in_weighs_start - cur_total_param, 0)
            weight_end = min(offset_in_weighs_end - cur_total_param, flat_weight_size)

            weight_slice = safe_mono_sd.get_tensor(weight_name=weight_name)

            if weight_slice is None and gigafsdp_model_normalize_fqns:
                # we have weight_name from meta_info from initialized gigafsdp module which already wrapped with lora,
                # thus, weight_name will contain .base_layer.weight, but checkpoint may contain weight without this prefix
                # try one more time without .base_layer in weight name
                non_normalize_name = weight_name
                weight_name = gigafsdp_model_normalize_fqns(weight_name)
                weight_slice = safe_mono_sd.get_tensor(weight_name=weight_name)
                if weight_slice is None:
                    logger.info(f"\tCan't find both {non_normalize_name} and {weight_name} in safetensors checkpoint!")
                else:
                    logger.info(f"\tCan't find {non_normalize_name}, but found {weight_name} in safetensors checkpoint!")

            if weight_slice is None and load_strict:
                raise RuntimeError(f"GigaFSDP sharding: Parameter `{weight_name}` is not in safetensors checkpoint," +
                    f" but it's in current model. Disable load_strict {load_strict=} logic if you still want to load this model.")
            else:
                # we don't find weight in safetensors checkpoint and strict=False, so use initialized weight from
                # current gigafsdp checkpoint
                try:
                    shard_name_in_model = shard_name_in_model.removeprefix("model.").removeprefix("teacher.")
                    shard_tesnor: torch.Tensor = local_gigafsdp_model_state_dict[shard_name_in_model]
                except BaseException as e:
                    raise RuntimeError(f"can't find {shard_name_in_model} in " +
                                       f"{local_gigafsdp_model_state_dict.keys()} and error: \n{e}\n")

            # NOTE (fedorovgv): If weight exists in safetensors, extract the portion corresponding to this shard
            #                   Else, obtain the missing weight from the shard of the initialized GigaFSDP model
            if weight_slice is not None:
                weight_slice = get_slice(weight=weight_slice,
                                        weight_key=weight_name,
                                        ep_rank=ep_rank,
                                        ep_size=ep_size,
                                        tp_rank=tp_rank,
                                        tp_size=tp_size,
                                        parallel_type=parallel_type,
                                        safe_mono_sd=safe_mono_sd,
                                        use_new_expert_weight_layout=use_new_expert_weight_layout)
                weight_slice = weight_slice[weight_start: weight_end]

            else:
                assert offset_in_weighs_end - offset_in_weighs_start == shard_tesnor.shape[0], (
                    f" {offset_in_weighs_end - offset_in_weighs_start} != {shard_tesnor.shape[0]} "
                )
                # need to get part of the shard which is equal to weight
                weight_start_in_shard = max(cur_total_param - offset_in_weighs_start, 0)
                if cur_total_param - offset_in_weighs_start >= 0:
                    weight_end_in_shard = min(cur_total_param - offset_in_weighs_start + flat_weight_size, shard.local_size)
                else:
                    weight_end_in_shard = min(flat_weight_size - (offset_in_weighs_start - cur_total_param), shard.local_size)
                # TODO (fedorovgv) : avoid this
                weight_slice = shard_tesnor[weight_start_in_shard: weight_end_in_shard].cpu()

            flat_shard.append(weight_slice)

        cur_total_param += flat_weight_size
        if cur_total_param > offset_in_weighs_end:
            break

    if len(flat_shard) == 0:
        # This rank's slice lies entirely in alignment-padding past the end of real
        # weights (happens for small "flatten_freezed" shards when world is large).
        # Fall back to the local model's shard tensor (it's just padding anyway).
        shard_name_in_model_clean = shard_name_in_model.removeprefix("model.").removeprefix("teacher.")
        local_tensor = local_gigafsdp_model_state_dict[shard_name_in_model_clean]
        
        assert local_tensor.shape[0] == shard.local_size, (
            f"padding fallback: {local_tensor.shape[0]} != {shard.local_size} for {shard.name}"
        )
        
        flat_shard = [local_tensor.detach().cpu()]
        
    flat_shard = torch.cat(flat_shard)

    # The last FSDP rank's slice overlaps the alignment-padding region of the flat
    # buffer (_init_shard appends zeros so padded_numel % (world_size * alignment_value) == 0).
    # Those padding bytes are not stored in the checkpoint, so we restore them here.
    if flat_shard.shape[0] < shard.local_size:
        total_meta = int(sum(int(np.prod(s)) for s in shard.weight_orig_shapes))
        is_boundary_rank = offset_in_weighs_start < total_meta <= offset_in_weighs_end
        
        if not is_boundary_rank:
            raise AssertionError(
                f"shard_name={shard.name}: unexpected size mismatch "
                f"(rank={fsdp_group_rank}, fsdp_size={shard.pg_size[0]}): "
                f"{flat_shard.shape[0]} != {shard.local_size} "
                f"(offset_start={offset_in_weighs_start}, offset_end={offset_in_weighs_end}, "
                f"total_meta={total_meta})"
            )
        
        pad_size = shard.local_size - flat_shard.shape[0]
        logger.warning(
            f"shard_name={shard.name}: appending {pad_size} zero-padding elements "
            f"(boundary FSDP rank {fsdp_group_rank}/{shard.pg_size[0]}, "
            f"real data ends at offset {total_meta} inside slice "
            f"[{offset_in_weighs_start}, {offset_in_weighs_end}])"
        )
        
        flat_shard = torch.cat([flat_shard, flat_shard.new_zeros(pad_size)])

    assert flat_shard.shape[0] == shard.local_size, (
        f"shard_name={shard.name}, {flat_shard.shape[0]} != {shard.local_size}"
    )
    return flat_shard


def add_buffers_from_safetensors_mono_model_sd(local_flat_gigafsdp_sd: OrderedDict,
                                               model_name: str = "model",
                                               safe_mono_sd: tp.Optional[SafeMonoSD] = None,
                                               local_gigafsdp_model_state_dict: tp.Optional[OrderedDict] = None,
                                               load_strict: bool = False) -> OrderedDict:
    # this buffer names may contains fsdp artifacts like _module and root_module
    buffer_keys = [k for k in local_gigafsdp_model_state_dict.keys() if "expert_bias" in k]

    # find normalized name in safetensor checkpoint
    for key_from_fsdp in buffer_keys:
        key_from_fsdp_with_module = f"{model_name}.{key_from_fsdp}"
        norm_key_from_safetensors = _normalize_module_name(key_from_fsdp_with_module)
        tensor = safe_mono_sd.get_tensor(norm_key_from_safetensors)
        if tensor is None:
            msg = (f"can't find model key '{key_from_fsdp_with_module}' in safetensors checkpoint " +
                   f"by name {norm_key_from_safetensors} in keys {safe_mono_sd.map_weight_name_to_file.keys()} !")
            if load_strict:
                raise ValueError(msg)
            else:
                logger.info(msg)
        local_flat_gigafsdp_sd[key_from_fsdp_with_module] = tensor
    return local_flat_gigafsdp_sd


def make_local_flat_gigafsdp_sd(safe_mono_sd: SafeMonoSD,
                                meta_infos: tp.Dict,
                                model_name: str,
                                local_gigafsdp_model_state_dict: tp.Dict,
                                load_strict: bool = False,
                                gigafsdp_model_normalize_fqns: tp.Optional[tp.Callable] = None,
                                use_new_expert_weight_layout: bool = False) -> OrderedDict:
    """ Make local fsdp shards from mono safetensors checkpoint.
    """
    local_flat_gigafsdp_sd = OrderedDict()
    shards: tp.List[Shard] = prepeare_shards_meta_info(meta_infos)

    has_ema_weights = any(key.endswith(EMA_SUFFIX) for key in safe_mono_sd.map_weight_name_to_file.keys())

    hybrid_group_size = meta_infos["hybrid_group_size"]

    for shard in shards:
        shard_name_in_model = f"{model_name}.{shard.name}"
        flat_shard = make_shard_from_safetensors_mono_model_sd(shard=shard,
                                                               # model_name=model_name,
                                                               model_name="model",
                                                               safe_mono_sd=safe_mono_sd,
                                                               local_gigafsdp_model_state_dict=local_gigafsdp_model_state_dict,
                                                               shard_name_in_model=shard_name_in_model,
                                                               is_ema_shard=False,
                                                               load_strict=load_strict,
                                                               gigafsdp_model_normalize_fqns=gigafsdp_model_normalize_fqns,
                                                               hybrid_group_size=hybrid_group_size,
                                                               use_new_expert_weight_layout=use_new_expert_weight_layout)
        local_flat_gigafsdp_sd[shard_name_in_model] = flat_shard
        gc.collect()

        if has_ema_weights:
            shard_name_in_model = f"{model_name}.{shard.name}{EMA_SUFFIX}"
            flat_shard = make_shard_from_safetensors_mono_model_sd(shard=shard,
                                                                   # model_name=model_name,
                                                                   model_name="model",
                                                                   safe_mono_sd=safe_mono_sd,
                                                                   local_gigafsdp_model_state_dict=local_gigafsdp_model_state_dict,
                                                                   shard_name_in_model=shard_name_in_model,
                                                                   is_ema_shard=True,
                                                                   load_strict=load_strict,
                                                                   gigafsdp_model_normalize_fqns=gigafsdp_model_normalize_fqns,
                                                                   hybrid_group_size=hybrid_group_size,
                                                                   use_new_expert_weight_layout=use_new_expert_weight_layout)
            local_flat_gigafsdp_sd[shard_name_in_model] = flat_shard
            gc.collect()

    local_flat_gigafsdp_sd = add_buffers_from_safetensors_mono_model_sd(
        local_flat_gigafsdp_sd,
        model_name,
        safe_mono_sd=safe_mono_sd,
        local_gigafsdp_model_state_dict=local_gigafsdp_model_state_dict,
        load_strict=load_strict)

    return local_flat_gigafsdp_sd


def shard_checkpoint_experimental(*,
                                  checkpoint_path: str,
                                  meta_infos: tp.Dict,
                                  model_name: str,
                                  local_gigafsdp_model_state_dict: tp.Dict,
                                  load_strict: bool,
                                  model_config: tp.Any,
                                  gigafsdp_model_normalize_fqns: tp.Optional[tp.Callable] = None) -> OrderedDict:
    checkpoint_path: Path = Path(checkpoint_path)
    safe_mono_sd = SafeMonoSD(safe_mono_sd_dirpath=checkpoint_path)

    if hasattr(model_config, "language_model"):
        model_config = model_config.language_model

    # Invariant: map_tp_to_monolith always writes experts in the OLD layout,
    # regardless of the training layout. get_slice therefore transposes
    # old -> new exactly when the target gigafsdp model uses the new layout.
    use_new_expert_weight_layout = _model_config_get(
        model_config, "use_new_expert_weight_layout", False)

    local_flat_gigafsdp_sd = make_local_flat_gigafsdp_sd(
        safe_mono_sd,
        meta_infos,
        model_name,
        local_gigafsdp_model_state_dict,
        load_strict=load_strict,
        gigafsdp_model_normalize_fqns=gigafsdp_model_normalize_fqns,
        use_new_expert_weight_layout=use_new_expert_weight_layout)

    mono_sd = safe_mono_sd.load_mono_sd_meta()
    mono_sd['state']["model"] = local_flat_gigafsdp_sd
    return mono_sd
