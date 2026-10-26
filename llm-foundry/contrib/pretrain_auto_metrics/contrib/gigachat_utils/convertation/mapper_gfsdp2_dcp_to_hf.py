import glob
import json
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from logger import logger, time_memory_stat
from transformers_utils.deepseek import _DeepseekPerLayerTransformer
from utils.file_utils import vendor_gigachat35_runtime_code
from utils.dist import _make_bundles_simple
from utils.safetensors_local import _save_chunk_as_safetensors, _write_local_index

EMA_NAMESPACE = "ema_model."
NON_MODEL_NAMESPACES = (
    "optimizer.",
    "lr_scheduler.",
    "dataloader.",
    "train_state.",
    EMA_NAMESPACE,
)


def read_dcp_tensor_metadata(checkpoint_path: str) -> Dict[str, TensorStorageMetadata]:
    metadata = FileSystemReader(checkpoint_path).read_metadata()
    return {
        key: value
        for key, value in metadata.state_dict_metadata.items()
        if isinstance(value, TensorStorageMetadata)
    }


def select_namespace_keys(keys: Sequence[str], namespace: str) -> List[str]:
    if namespace:
        return [k for k in keys if k.startswith(namespace)]
    return [k for k in keys if not k.startswith(NON_MODEL_NAMESPACES)]


def build_ema_source_keys(
    tensor_metadata: Dict[str, TensorStorageMetadata],
) -> Dict[str, str]:
    """Output key -> source key for a use_ema export."""
    keys = list(tensor_metadata.keys())
    ema_keys = set(select_namespace_keys(keys, EMA_NAMESPACE))
    if not ema_keys:
        raise ValueError(
            "use_ema requested, but the checkpoint has no ema_model.* tensors"
        )
    source_keys = {}
    from_ema = 0
    for key in select_namespace_keys(keys, ""):
        ema_key = EMA_NAMESPACE + key
        if ema_key in ema_keys:
            source_keys[key] = ema_key
            from_ema += 1
        else:
            source_keys[key] = key
    logger.info(
        f"use_ema: {from_ema}/{len(source_keys)} tensors sourced from EMA, "
        f"{len(source_keys) - from_ema} fall back to main weights"
    )
    return source_keys


def load_dcp_tensors(
    checkpoint_path: str,
    keys: Sequence[str],
    tensor_metadata: Dict[str, TensorStorageMetadata],
) -> Dict[str, torch.Tensor]:
    state_dict = {
        key: torch.empty(
            tuple(tensor_metadata[key].size),
            dtype=tensor_metadata[key].properties.dtype,
        )
        for key in keys
    }
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    return state_dict


def reshape_flat_tensors(
    state_dict: Dict[str, torch.Tensor],
    shapes: Optional[Dict[str, Sequence[int]]],
) -> Dict[str, torch.Tensor]:
    if not shapes:
        return state_dict
    reshaped = {}
    for key, tensor in state_dict.items():
        target_shape = shapes.get(key)
        if target_shape is None or tuple(tensor.shape) == tuple(target_shape):
            reshaped[key] = tensor
            continue
        target_numel = 1
        for dim_size in target_shape:
            target_numel *= dim_size
        if tensor.numel() != target_numel:
            raise ValueError(
                f"Cannot reshape '{key}': checkpoint numel {tensor.numel()} "
                f"!= target shape {tuple(target_shape)}"
            )
        reshaped[key] = tensor.view(tuple(target_shape))
    return reshaped


@time_memory_stat
def convert_dcp_to_safetensors(
    checkpoint_path: str,
    output_path: str,
    *,
    transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    shapes: Optional[Dict[str, Sequence[int]]] = None,
    namespace: str = "",
    use_ema: bool = False,
    export_dtype: Optional[torch.dtype] = None,
    global_rank: int = 0,
    world_size: int = 1,
    chunk_layers: int = 2,
) -> None:
    tensor_metadata = read_dcp_tensor_metadata(checkpoint_path)
    if use_ema:
        if namespace:
            raise ValueError("'use_ema' and 'namespace' are mutually exclusive")
        source_keys = build_ema_source_keys(tensor_metadata)
    else:
        selected = select_namespace_keys(list(tensor_metadata.keys()), namespace)
        source_keys = {key[len(namespace):]: key for key in selected}
    if not source_keys:
        raise ValueError(
            f"No tensors under namespace '{namespace or '<main>'}' in {checkpoint_path}"
        )
    keys = list(source_keys)

    bundles = _make_bundles_simple(keys)
    bundle_chunks = [bundles[i:i + chunk_layers] for i in range(0, len(bundles), chunk_layers)]

    out_rank_dir = os.path.join(output_path, f"rank{global_rank:02d}")
    os.makedirs(out_rank_dir, exist_ok=True)

    mode = "ema-merged" if use_ema else (namespace or "<main>")
    logger.info(
        f"[rank {global_rank}] converting '{mode}': "
        f"{len(keys)} tensors, {len(bundle_chunks)} chunks total, "
        f"mine: {len(range(global_rank, len(bundle_chunks), world_size))}"
    )

    for chunk_idx, chunk_bundles in enumerate(bundle_chunks):
        if chunk_idx % world_size != global_rank:
            continue
        chunk_keys = [key for bundle in chunk_bundles for key in bundle]

        block = load_dcp_tensors(
            checkpoint_path,
            [source_keys[key] for key in chunk_keys],
            tensor_metadata,
        )
        block = {key: block[source_keys[key]] for key in chunk_keys}
        block = reshape_flat_tensors(block, shapes)

        mapped = transform(block) if transform is not None else block
        if export_dtype is not None:
            mapped = {
                key: tensor.to(export_dtype) if tensor.is_floating_point() else tensor
                for key, tensor in mapped.items()
            }

        shard_file = f"shard-r{global_rank:02d}-c{chunk_idx:05d}.safetensors"
        shard_size = _save_chunk_as_safetensors(mapped, os.path.join(out_rank_dir, shard_file))

        weight_map = {key: shard_file for key in mapped.keys()}
        weight_map_file = f"shard-r{global_rank:02d}-c{chunk_idx:05d}.weight_map.json"
        with open(os.path.join(out_rank_dir, weight_map_file), "w") as fh:
            json.dump({"weight_map": weight_map, "total_size": shard_size}, fh)

    _write_local_index(out_rank_dir)


def finalize_hf_checkpoint(output_path: str, world_size: int = 1) -> None:
    """Merge per-rank shard dirs into a flat HF checkpoint with a single index.

    Run once after every rank of a multi-rank conversion has finished.
    """
    rank_dirs = [
        os.path.join(output_path, f"rank{rank:02d}") for rank in range(world_size)
    ]
    index_path = os.path.join(output_path, "model.safetensors.index.json")
    if not any(os.path.isdir(d) for d in rank_dirs) and os.path.isfile(index_path):
        logger.info(f"{output_path} is already finalized, skipping")
        return
    missing = [
        d for d in rank_dirs if not os.path.isfile(os.path.join(d, "rank-index.json"))
    ]
    if missing:
        raise ValueError(
            f"Cannot finalize {output_path}: rank outputs incomplete "
            f"(no rank-index.json in: {', '.join(missing)}). "
            "Run the conversion for every rank first."
        )

    weight_map: Dict[str, str] = {}
    total_size = 0
    for rank_dir in rank_dirs:
        with open(os.path.join(rank_dir, "rank-index.json")) as fh:
            rank_index = json.load(fh)
        weight_map.update(rank_index["weight_map"])
        total_size += int(rank_index["metadata"]["total_size"])
        for shard in sorted(glob.glob(os.path.join(rank_dir, "shard-*.safetensors"))):
            os.rename(shard, os.path.join(output_path, os.path.basename(shard)))
        shutil.rmtree(rank_dir)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
        fh.write("\n")
    logger.info(
        f"Finalized HF checkpoint: {output_path} ({len(weight_map)} tensors)"
    )


# torchtitan writes ``model_type`` itself: gigachat3_5 for anything DeepSeek-V3
# cannot express (GDN layers, gated attention, gated norms), deepseek_v3 otherwise.
GIGACHAT_MODEL_TYPE = "gigachat3_5"


def write_hf_metadata(
    output_path: str,
    hf_config_path: str,
    tokenizer_path: Optional[str],
    hf_config: Dict[str, Any],
) -> None:
    """Everything beside the weights that an export needs to be loadable.

    A gigachat3_5 ``config.json`` names its implementation through ``auto_map``,
    so the export is unloadable unless those modules sit next to it. The
    llm-foundry mappers vendor them for the same reason (``save_meta`` in
    base_mappers.py); ``regenerate_modeling=False`` copies the checked-in
    ``modeling_gigachat3_5.py`` instead of re-running codegen inside the job.
    """
    target_config = os.path.join(output_path, "config.json")
    if not os.path.exists(target_config):
        shutil.copyfile(hf_config_path, target_config)
    if tokenizer_path and os.path.isdir(tokenizer_path):
        for item in os.listdir(tokenizer_path):
            source = os.path.join(tokenizer_path, item)
            target = os.path.join(output_path, item)
            if os.path.isfile(source) and not os.path.exists(target):
                shutil.copyfile(source, target)
    if hf_config.get("model_type") == GIGACHAT_MODEL_TYPE:
        vendor_gigachat35_runtime_code(output_path, regenerate_modeling=False)


PARAM_SHAPES = "_titan_param_shapes"


def read_titan_param_shapes(checkpoint_path: str) -> Dict[str, List[int]]:
    """The real parameter shapes, read out of the checkpoint itself.

    gfsdp2 writes every managed parameter as a flat ``(numel,)`` slice, so DCP
    metadata alone cannot describe the model. torchtitan therefore stores a JSON
    map of the true shapes under ``PARAM_SHAPES``; it is a string, not a nested
    mapping, because DCP flattens mappings into one entry per key.
    """
    metadata = FileSystemReader(checkpoint_path).read_metadata().state_dict_metadata
    # One entry per writing rank, plus the un-suffixed key older checkpoints
    # use. Under pipeline parallelism a rank holds only its own stage, so the
    # map has to be the union: any single entry describes part of the model.
    keys = [
        key
        for key in metadata
        if key == PARAM_SHAPES or key.startswith(PARAM_SHAPES + ".")
    ]
    if not keys:
        raise ValueError(
            f"{checkpoint_path} carries no '{PARAM_SHAPES}' entry, so its flat "
            "tensors cannot be reshaped. It was written by a torchtitan that "
            "predates shape recording; re-save the checkpoint."
        )
    holder: Dict[str, Any] = {key: None for key in keys}
    dcp.load(holder, checkpoint_id=checkpoint_path)

    shapes: Dict[str, List[int]] = {}
    for key in sorted(keys):
        for fqn, shape in json.loads(holder[key]).items():
            recorded = shapes.setdefault(fqn, shape)
            if recorded != shape:
                raise ValueError(
                    f"{checkpoint_path} disagrees with itself about '{fqn}': "
                    f"{recorded} and {shape}. Two ranks recorded different "
                    "shapes for one parameter."
                )
    logger.info(f"read {len(shapes)} parameter shapes from {len(keys)} entries")
    return shapes


# ---------------------------------------------------------------------------
# torchtitan -> HF names
#
# Suffix tables, applied under a layer prefix. Everything below is a rename or
# a slice; the only real tensor surgery is the GDN attention, delegated to
# `_DeepseekPerLayerTransformer`.
# ---------------------------------------------------------------------------

# Sublayer norms. Every one of these is a *gated* RMSNorm in GIFT 3.5.
_LAYER_NORMS = {
    "attention_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    # Sandwich norms GIFT 3.5 adds that DeepSeek-V3 has no slot for.
    "post_attn_norm": "post_self_attn_layernorm",
    "post_ffn_norm": "post_feedforward_layernorm",
}

# The MTP head's own norms, under the head's HF layer index.
_MTP_NORMS = {
    "token_norm": "enorm",
    "hidden_norm": "hnorm",
    "final_norm": "shared_head.norm",
}

# A gated RMSNorm's parameters: the scale plus the two gate projections.
_NORM_PARAMS = {
    "weight": "weight",
    "gate_up_weight": "gate_up_projection.weight",
    "gate_down_weight": "gate_down_projection.weight",
}

# MLA, 1:1 under `attention.` -> `self_attn.`. `wq`/`wq_a`+`wq_b` are
# alternatives: which pair exists depends on `q_lora_rank`.
_MLA_DIRECT = {
    "wq.weight": "q_proj.weight",
    "wq_a.weight": "q_a_proj.weight",
    "wq_b.weight": "q_b_proj.weight",
    "wkv_a.weight": "kv_a_proj_with_mqa.weight",
    "wkv_b.weight": "kv_b_proj.weight",
    "wo.weight": "o_proj.weight",
    "gate.gate_proj.weight": "gate_proj.weight",
}

_MLA_NORMS = {"q_norm": "q_a_layernorm", "kv_norm": "kv_a_layernorm"}

# GDN parameters that need no surgery. The o_norm's gate is `z_proj`, not a
# pair of gate projections, so it maps as a plain weight.
_GDN_DIRECT = {
    "ba_proj.weight": "in_proj_ba.weight",
    "o_norm.weight": "norm.weight",
    "o_proj.weight": "out_proj.weight",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
}

_FFN = {
    "w1.weight": "gate_proj.weight",
    "w3.weight": "up_proj.weight",
    "w2.weight": "down_proj.weight",
}

# GroupedExperts stacks. Each slice already carries HF's orientation, so the
# split is a plain index -- no transpose. `w1_EFD`/`w3_EFD` are the layout of
# checkpoints written before torchtitan fused them into `w13_E2FD`.
_EXPERT_STACKS = {"w1_EFD": "gate_proj", "w3_EFD": "up_proj", "w2_EDF": "down_proj"}

# Fused gate/up stack, `(E, 2, F, D)`: slot 0 is the gate, slot 1 the up.
_FUSED_EXPERT_STACKS = {"w13_E2FD": ("gate_proj", "up_proj")}


class TitanToHF:
    """Rename one block of a torchtitan state dict into HF names.

    Built from the model's ``config.json`` alone: the GDN geometry, the layer
    count and the MTP head count all live there, so the conversion needs
    nothing from torchtitan itself.

    GDN attention is the one place that rearranges values rather than renaming
    them, and it is delegated to ``_DeepseekPerLayerTransformer`` -- the
    converter PAM already runs -- instead of being reimplemented here. Only its
    two rearrangement helpers are used, with prefixes passed in explicitly: its
    own key dispatch infers a layer's attention type from
    ``full_attention_layers``, which does not cover MTP head indices, so an MLA
    head would be taken for a GDN one. Dispatching on tensor names instead
    cannot make that mistake.
    """

    def __init__(self, hf_config: Dict[str, Any]) -> None:
        self.n_main_layers = int(hf_config["num_hidden_layers"])
        self.num_mtp = int(hf_config.get("num_nextn_predict_layers") or 0)
        self._gdn = _DeepseekPerLayerTransformer(hf_config)

    def __call__(self, block: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, tensor in block.items():
            self._emit(name, tensor, out)
        if self._gdn.qkvz_buf:
            raise ValueError(
                "GDN qkv_proj and z_proj did not pair up within one block "
                f"({sorted(self._gdn.qkvz_buf)}); the bundler must keep a "
                "layer's tensors together."
            )
        return out

    # ---- dispatch ---------------------------------------------------------

    def _emit(self, name: str, tensor: torch.Tensor, out: Dict[str, torch.Tensor]) -> None:
        if name == "tok_embeddings.weight":
            out["model.embed_tokens.weight"] = tensor
            # HF keeps a copy of the embedding and of the output head inside
            # every nextn layer; torchtitan stores one of each. Real copies,
            # not aliases: safetensors refuses to serialize two entries backed
            # by the same storage.
            for hf_layer in self._mtp_layer_indices():
                out[f"model.layers.{hf_layer}.embed_tokens.weight"] = (
                    tensor.contiguous().clone()
                )
            return
        if name == "lm_head.weight":
            out["lm_head.weight"] = tensor
            for hf_layer in self._mtp_layer_indices():
                out[f"model.layers.{hf_layer}.shared_head.head.weight"] = (
                    tensor.contiguous().clone()
                )
            return
        if name.startswith("norm."):
            out[f"model.norm.{self._norm_param(name, name[len('norm.'):])}"] = tensor
            return
        if name.startswith("mtp.layers."):
            self._emit_mtp(name, name[len("mtp.layers."):], tensor, out)
            return
        if name.startswith("layers."):
            layer, suffix = self._split_index(name[len("layers."):])
            self._emit_layer(name, layer, suffix, tensor, out)
            return
        raise ValueError(f"unknown titan tensor '{name}'")

    def _emit_mtp(
        self, name: str, tail: str, tensor: torch.Tensor, out: Dict[str, torch.Tensor]
    ) -> None:
        head, suffix = self._split_index(tail)
        hf_layer = self.n_main_layers + head
        if suffix.startswith("block."):
            # A head's decoder block is an ordinary layer at index N + k.
            self._emit_layer(name, hf_layer, suffix[len("block.") :], tensor, out)
            return
        if suffix == "input_proj.weight":
            out[f"model.layers.{hf_layer}.eh_proj.weight"] = self._swap_eh_proj_halves(
                tensor
            )
            return
        prefix, _, param = suffix.rpartition(".")
        if prefix in _MTP_NORMS:
            hf_norm = _MTP_NORMS[prefix]
            out[
                f"model.layers.{hf_layer}.{hf_norm}.{self._norm_param(name, param)}"
            ] = tensor
            return
        raise ValueError(f"unknown titan MTP tensor '{name}'")

    def _emit_layer(
        self,
        name: str,
        hf_layer: int,
        suffix: str,
        tensor: torch.Tensor,
        out: Dict[str, torch.Tensor],
    ) -> None:
        prefix = f"model.layers.{hf_layer}."
        if suffix.startswith("attention."):
            self._emit_attention(name, prefix, suffix[len("attention.") :], tensor, out)
            return
        if suffix.startswith("feed_forward."):
            leaf = suffix[len("feed_forward.") :]
            if leaf not in _FFN:
                raise ValueError(f"unknown titan dense-FFN tensor '{name}'")
            out[f"{prefix}mlp.{_FFN[leaf]}"] = tensor
            return
        if suffix.startswith("moe."):
            self._emit_moe(name, prefix, suffix[len("moe.") :], tensor, out)
            return
        norm, _, param = suffix.rpartition(".")
        if norm in _LAYER_NORMS:
            out[
                f"{prefix}{_LAYER_NORMS[norm]}.{self._norm_param(name, param)}"
            ] = tensor
            return
        raise ValueError(f"unknown titan tensor '{name}'")

    def _emit_attention(
        self,
        name: str,
        prefix: str,
        suffix: str,
        tensor: torch.Tensor,
        out: Dict[str, torch.Tensor],
    ) -> None:
        attn = f"{prefix}self_attn."
        # `_maybe_accumulate_and_return_qkvz` returns {} until it holds both
        # halves, then the fused tensor; both arrive in the same block.
        if suffix == "qkv.qkv_proj.weight":
            out.update(
                self._gdn._maybe_accumulate_and_return_qkvz(
                    prefix.rstrip("."), "self_attn", "qkv", tensor
                )
            )
            return
        if suffix == "z_proj.weight":
            out.update(
                self._gdn._maybe_accumulate_and_return_qkvz(
                    prefix.rstrip("."), "self_attn", "z", tensor
                )
            )
            return
        if suffix == "qkv.qkv_conv1d.weight":
            out[f"{attn}conv1d.weight"] = self._gdn._interleaved_qkv_to_conv_order(
                tensor
            )
            return
        if suffix in _GDN_DIRECT:
            out[f"{attn}{_GDN_DIRECT[suffix]}"] = tensor
            return
        if suffix in _MLA_DIRECT:
            out[f"{attn}{_MLA_DIRECT[suffix]}"] = tensor
            return
        norm, _, param = suffix.rpartition(".")
        if norm in _MLA_NORMS:
            out[f"{attn}{_MLA_NORMS[norm]}.{self._norm_param(name, param)}"] = tensor
            return
        raise ValueError(f"unknown titan attention tensor '{name}'")

    def _emit_moe(
        self,
        name: str,
        prefix: str,
        suffix: str,
        tensor: torch.Tensor,
        out: Dict[str, torch.Tensor],
    ) -> None:
        if suffix == "router.gate.weight":
            out[f"{prefix}mlp.gate.weight"] = tensor
            return
        if suffix == "expert_bias_E":
            out[f"{prefix}mlp.gate.e_score_correction_bias"] = tensor
            return
        if suffix.startswith("shared_experts."):
            leaf = suffix[len("shared_experts.") :]
            if leaf not in _FFN:
                raise ValueError(f"unknown titan shared-expert tensor '{name}'")
            out[f"{prefix}mlp.shared_experts.{_FFN[leaf]}"] = tensor
            return
        if suffix.startswith("experts."):
            stack = suffix[len("experts.") :]
            if stack in _FUSED_EXPERT_STACKS:
                if tensor.ndim != 4 or tensor.shape[1] != len(_FUSED_EXPERT_STACKS[stack]):
                    raise ValueError(
                        f"'{name}' must be (num_experts, 2, hidden_dim, dim), "
                        f"got {tuple(tensor.shape)}"
                    )
                for slot, projection in enumerate(_FUSED_EXPERT_STACKS[stack]):
                    for expert in range(tensor.shape[0]):
                        out[f"{prefix}mlp.experts.{expert}.{projection}.weight"] = (
                            tensor[expert, slot]
                        )
                return
            if stack not in _EXPERT_STACKS:
                raise ValueError(f"unknown titan expert stack '{name}'")
            projection = _EXPERT_STACKS[stack]
            for expert in range(tensor.shape[0]):
                out[f"{prefix}mlp.experts.{expert}.{projection}.weight"] = tensor[expert]
            return
        raise ValueError(f"unknown titan MoE tensor '{name}'")

    # ---- helpers ----------------------------------------------------------

    def _mtp_layer_indices(self) -> range:
        return range(self.n_main_layers, self.n_main_layers + self.num_mtp)

    @staticmethod
    def _split_index(tail: str) -> Tuple[int, str]:
        index, _, suffix = tail.partition(".")
        return int(index), suffix

    @staticmethod
    def _norm_param(name: str, param: str) -> str:
        if param not in _NORM_PARAMS:
            raise ValueError(f"unknown gated-norm parameter in '{name}'")
        return _NORM_PARAMS[param]

    @staticmethod
    def _swap_eh_proj_halves(weight: torch.Tensor) -> torch.Tensor:
        """torchtitan concatenates ``[hidden, embedding]`` for the MTP input
        projection; HF ``eh_proj`` expects ``[embedding, hidden]``. The weight is
        ``[dim, 2*dim]``, so swapping the column halves converts either way."""
        half = weight.shape[1] // 2
        return torch.cat([weight[:, half:], weight[:, :half]], dim=1).contiguous()
