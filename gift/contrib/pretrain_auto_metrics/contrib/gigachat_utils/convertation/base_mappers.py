from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Protocol, Any, Optional, Tuple
from copy import deepcopy

import torch
from transformers import PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from logger import logger
from models.gigachat3_5.configuration_gigachat3_5 import Gigachat35Config
from models.gigachat3_5_dense.configuration_gigachat3_5_dense import Gigachat35DenseConfig
from models.gigachat_slm.configuration_gigachat_slm import GigachatSLMConfig, normalize_sconv_sites

AutoConfig.register("gigachat3_5", Gigachat35Config)
AutoConfig.register("gigachat3_5_dense", Gigachat35DenseConfig)
AutoConfig.register("gigachat_slm", GigachatSLMConfig)

_GIGACHAT35_MODEL_REGISTERED = False


def _ensure_gigachat35_model_registered() -> None:
    global _GIGACHAT35_MODEL_REGISTERED
    if _GIGACHAT35_MODEL_REGISTERED:
        return

    from models.gigachat3_5.modeling_gigachat3_5 import Gigachat35ForCausalLM

    AutoModelForCausalLM.register(Gigachat35Config, Gigachat35ForCausalLM)
    _GIGACHAT35_MODEL_REGISTERED = True

try:
    from utils.file_utils import (
        _atomic_write_json,
        vendor_deepseek_runtime_code,
        vendor_gigachat35_runtime_code,
        vendor_runtime_code,
    )
    from utils.hf_assembly_manifest import write_hf_assembly_manifest
    from utils.hf_model_specs import DEEPSEEK_V3_SPEC, GIGACHAT3_5_DENSE_SPEC, GIGACHAT3_5_SPEC, GIGACHAT_SLM_SPEC, HFModelSpec
    from utils.llmfoundry_config_utils import is_gigachat_model
    from utils.mtp_attention import resolve_mtp_uses_linear_attention
except BaseException as _:
    # NOTE (fedorovgv): want to use it in jupyter notebooks
    current_file = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_file / "utils"))
    from file_utils import (
        _atomic_write_json,
        vendor_deepseek_runtime_code,
        vendor_gigachat35_runtime_code,
        vendor_runtime_code,
    )
    from hf_assembly_manifest import write_hf_assembly_manifest
    from hf_model_specs import DEEPSEEK_V3_SPEC, GIGACHAT3_5_DENSE_SPEC, GIGACHAT3_5_SPEC, GIGACHAT_SLM_SPEC, HFModelSpec
    from llmfoundry_config_utils import is_gigachat_model
    from mtp_attention import resolve_mtp_uses_linear_attention

from mapper_deepseek_moe_to_hf import init_empty_weights

_DSA_CONFIG_RENAMES = {
    "indexer_n_heads": "index_n_heads",
    "indexer_head_dim": "index_head_dim",
    "indexer_topk": "index_topk",
    "indexer_rotate_activation_enabled": "indexer_rotate_activation_enabled",
}
_DSA_TRAINING_ONLY_FIELDS = (
    "aux_kl_loss_koeff",
    "topk_chunk_size",
    "dsa_stage",
    "indexer_tile",
)

def _normalize_dsa_config_fields(cfg_dict: Dict[str, Any]) -> None:
    for llmf_name, hf_name in _DSA_CONFIG_RENAMES.items():
        value = cfg_dict.pop(llmf_name, None)
        if value is not None and cfg_dict.get(hf_name) is None:
            cfg_dict[hf_name] = value
    for field in _DSA_TRAINING_ONLY_FIELDS:
        cfg_dict.pop(field, None)


def _coerce_template_float_fields(
    cfg_dict: Dict[str, Any], template: Dict[str, Any]
) -> None:
    for key, ref in template.items():
        if type(ref) is float and type(cfg_dict.get(key)) is int:
            cfg_dict[key] = float(cfg_dict[key])


class LayerMapper(Protocol):
    def map(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: ...
    def flush(self) -> Dict[str, torch.Tensor]: ...
    def ensure_config(self, ckpt_cfg: Dict[str, Any], args) -> None: ...  # <-- добавить
    def build_config(self, ckpt_cfg: Dict[str, Any], args) -> PretrainedConfig | Dict[str, Any]: ...
    def tokenizer_source(self, ckpt_cfg: Dict[str, Any], args) -> str: ...
    def save_meta(self, cfg_src: Dict[str, Any], args, output_dir: str) -> None: ...
    def write_export_manifest(self, cfg_src: Dict[str, Any], args, output_dir: str) -> None: ...


def _to_bf16(block: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (v if v.dtype == torch.bfloat16 else v.to(torch.bfloat16))
            for k, v in block.items()}


def _save_config_and_tokenizer(
    mapper: "LayerMapper",
    cfg_src: Dict[str, Any],
    args,
    output_dir: str,
    *,
    with_tmp_model: bool = False
) -> Tuple[Any, Optional[Dict[str, Any]], Optional[Any]]:
    os.makedirs(output_dir, exist_ok=True)

    cfg_obj = mapper.build_config(cfg_src, args)
    cfg_dict = cfg_obj.to_dict() if isinstance(cfg_obj, PretrainedConfig) else cfg_obj
    for _nextn_field in ("nextn_is_sparse", "nextn_is_linear"):
        if cfg_dict.get(_nextn_field) is None:
            cfg_dict.pop(_nextn_field, None)
    auto_map_for_save = getattr(cfg_obj, "_auto_map_for_save", None)
    if auto_map_for_save is not None:
        cfg_dict["auto_map"] = auto_map_for_save
        del cfg_dict["_auto_map_for_save"]
    _atomic_write_json(os.path.join(output_dir, "config.json"), cfg_dict)

    tok_src = mapper.tokenizer_source(cfg_src, args)
    if not tok_src:
        raise RuntimeError("Tokenizer path is undefined")

    tok = AutoTokenizer.from_pretrained(tok_src)
    tok.save_pretrained(output_dir)

    tmp_model = None
    if with_tmp_model:
        try:
            _ensure_gigachat35_model_registered()
            with init_empty_weights():
                tmp_model = AutoModelForCausalLM.from_config(
                    cfg_obj if isinstance(cfg_obj, PretrainedConfig)
                    else AutoConfig.from_dict(cfg_dict, trust_remote_code=True),
                    # trust_remote_code=True,
                )
        except Exception as e:
            print(f"Exception while loading model: {e}")
            tmp_model = None

    return cfg_obj, cfg_dict, tmp_model

class RawMapper:
    def map(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return _to_bf16(tensors)

    def ensure_config(self, ckpt_cfg: Dict[str, Any], args) -> None:
        return

    def flush(self) -> Dict[str, torch.Tensor]:
        return {}

    def build_config(self, ckpt_cfg: Dict[str, Any], args) -> Dict[str, Any]:
        cfg = dict(ckpt_cfg)
        for k in ("architectures", "auto_map", "quantization_config"):
            cfg.pop(k, None)
        cfg.setdefault("bos_token_id", 1)
        cfg.setdefault("eos_token_id", 2)
        cfg.setdefault("pad_token_id", 2)
        normalized_dtype = getattr(args, "torch_dtype", "bfloat16")
        cfg["torch_dtype"] = normalized_dtype
        cfg["dtype"] = normalized_dtype
        return cfg

    def tokenizer_source(self, ckpt_cfg: Dict[str, Any], args) -> str:
        return getattr(args, "tokenizer_path", None) or ckpt_cfg.get("_name_or_path") or ""

    def save_meta(self, cfg_src: Dict[str, Any], args, output_dir: str) -> None:
        _save_config_and_tokenizer(self, cfg_src, args, output_dir, with_tmp_model=False)

    def write_export_manifest(self, cfg_src: Dict[str, Any], args, output_dir: str) -> None:
        return


class DeepseekV3Mapper:
    def __init__(self, cfg: Optional[Any] = None,  tr: Optional[Any] = None):
        self.cfg = cfg
        self.tr = tr
        if cfg is not None and tr is None:
            self.tr = self._build_transformer(cfg)

    def _build_transformer(self, cfg: Any):
        from transformers_utils.deepseek import _DeepseekPerLayerTransformer
        return _DeepseekPerLayerTransformer(cfg)

    def _model_spec(self, cfg_dict: Dict[str, Any]) -> HFModelSpec:
        return GIGACHAT3_5_SPEC if is_gigachat_model(cfg_dict) else DEEPSEEK_V3_SPEC

    @staticmethod
    def _has_linear_attention(cfg: Optional[Any]) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, dict):
            return cfg.get("linear_attention_type") is not None
        return getattr(cfg, "linear_attention_type", None) is not None

    def ensure_config(self, ckpt_cfg: Dict[str, Any], args) -> None:
        if self.cfg is None:
            self.cfg = self.build_config(ckpt_cfg, args)
        if self.tr is None:
            self.tr = self._build_transformer(self.cfg)

    def map(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._has_linear_attention(self.cfg):
            tensors = self.pack_conv_weights(tensors)
        out = self.tr.rename_and_split(tensors)
        return _to_bf16(out)

    def flush(self) -> Dict[str, torch.Tensor]:
        return self.tr.flush_pending()

    @staticmethod
    def _get_nextn_is_sparse(cfg_dict: Dict[str, Any]) -> Optional[bool]:
        mtp_block_type = cfg_dict.get("mtp_block_type")
        if mtp_block_type is None:
            return None

        mtp_block_type = str(mtp_block_type).strip().lower()
        if mtp_block_type == "dense":
            return False
        if mtp_block_type == "moe":
            return True
        raise ValueError(
            f"Unsupported mtp_block_type={mtp_block_type!r}. Expected 'dense' or 'moe'."
        )

    # --- build_config steps shared with subclasses ---

    @staticmethod
    def _strip_export_fields(ckpt_cfg: Dict[str, Any]) -> Dict[str, Any]:
        ckpt_cfg = dict(ckpt_cfg)
        for k in ("architectures", "auto_map", "quantization_config"):
            ckpt_cfg.pop(k, None)
        return ckpt_cfg

    @staticmethod
    def _load_template(spec: HFModelSpec) -> Dict[str, Any]:
        with open(spec.template_dir / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _apply_logit_softcapping(cfg_dict: Dict[str, Any], ckpt_cfg: Dict[str, Any]) -> None:
        if ckpt_cfg.get("lm_head_logit_softcapping") is not None:
            cfg_dict["final_logit_softcapping"] = ckpt_cfg["lm_head_logit_softcapping"]

    @staticmethod
    def _apply_token_id_defaults(cfg_dict: Dict[str, Any]) -> None:
        cfg_dict.setdefault("bos_token_id", 1)
        cfg_dict.setdefault("eos_token_id", 2)
        cfg_dict.setdefault("pad_token_id", 2)

    @staticmethod
    def _apply_torch_dtype(cfg_dict: Dict[str, Any], args) -> str:
        torch_dtype = getattr(args, "torch_dtype", "bfloat16")
        assert torch_dtype in ("float16", "bfloat16"), f"Unsupported torch_dtype={torch_dtype}"
        cfg_dict["torch_dtype"] = torch_dtype
        cfg_dict["dtype"] = torch_dtype
        return torch_dtype

    def _apply_mtp_fields(self, cfg_dict: Dict[str, Any], ckpt_cfg: Dict[str, Any]) -> None:
        if cfg_dict.get("use_mtp"):
            cfg_dict["num_nextn_predict_layers"] = (
                1 if cfg_dict.get("mtp_recurrent") else cfg_dict["mtp_predictor_num"]
            )
            nextn_is_sparse = self._get_nextn_is_sparse(cfg_dict)
            if nextn_is_sparse is not None:
                cfg_dict["nextn_is_sparse"] = nextn_is_sparse
            else:
                cfg_dict.pop("nextn_is_sparse", None)

            cfg_dict["nextn_is_linear"] = resolve_mtp_uses_linear_attention(ckpt_cfg)
        elif "num_nextn_predict_layers" not in ckpt_cfg:
            cfg_dict["num_nextn_predict_layers"] = 0
            cfg_dict.pop("nextn_is_sparse", None)
            cfg_dict.pop("nextn_is_linear", None)
        cfg_dict.pop("use_linear_attention_in_mtp", None)

    @staticmethod
    def _normalize_rope_scaling(scaling: Any) -> Optional[Dict[str, Any]]:
        allowed = {"linear", "dynamic", "yarn"}
        if isinstance(scaling, dict):
            scaling["type"] = scaling.get("type") or scaling.pop("rope_type", None)
            if scaling["type"] not in allowed:
                raise ValueError(f"Unknown RoPE scaling type {scaling['type']!r}")
        elif isinstance(scaling, str):
            if scaling not in allowed:
                raise ValueError(f"Unknown RoPE scaling type {scaling!r}")
            scaling = {"type": scaling}
        elif scaling is not None:
            raise TypeError("rope_scaling must be dict, str or None")
        return scaling

    def build_config(self, ckpt_cfg: Dict[str, Any], args) -> AutoConfig:
        ckpt_cfg = self._strip_export_fields(ckpt_cfg)

        is_gigachat = is_gigachat_model(ckpt_cfg)
        if not is_gigachat and ckpt_cfg.get("model_type") == "giga_mix":
            ckpt_cfg["model_type"] = "deepseek_v3"

        spec = self._model_spec(ckpt_cfg)
        template = self._load_template(spec)
        cfg_dict = {**template, **ckpt_cfg}
        _normalize_dsa_config_fields(cfg_dict)

        # In giga_mix config:
        #   intermediate_size       = per-expert FFN size  → HF moe_intermediate_size
        #   dense_intermediate_size = dense-layer FFN size → HF intermediate_size
        # When dense_intermediate_size is absent, intermediate_size is used for both.
        has_dense = "dense_intermediate_size" in ckpt_cfg and ckpt_cfg["dense_intermediate_size"] is not None
        has_expert = "intermediate_size" in ckpt_cfg and ckpt_cfg["intermediate_size"] is not None

        if has_dense:
            cfg_dict["intermediate_size"] = ckpt_cfg["dense_intermediate_size"]

        if "moe_intermediate_size" in ckpt_cfg and ckpt_cfg["moe_intermediate_size"] is not None:
            cfg_dict["moe_intermediate_size"] = ckpt_cfg["moe_intermediate_size"]
        elif has_expert:
            # training intermediate_size is always the per-expert size
            cfg_dict["moe_intermediate_size"] = ckpt_cfg["intermediate_size"]
        elif has_dense:
            # no per-expert size → same as dense
            cfg_dict["moe_intermediate_size"] = ckpt_cfg["dense_intermediate_size"]

        if "num_routed_experts" in ckpt_cfg and ckpt_cfg["num_routed_experts"] is not None:
            cfg_dict["n_routed_experts"] = cfg_dict["num_routed_experts"]

        if "router_aux_loss_coef" in ckpt_cfg and ckpt_cfg["router_aux_loss_coef"] is not None:
            cfg_dict["aux_loss_alpha"] = ckpt_cfg["router_aux_loss_coef"]

        if (
            "moe_router_routed_scaling_factor" in ckpt_cfg
            and ckpt_cfg["moe_router_routed_scaling_factor"] is not None
        ):
            cfg_dict["routed_scaling_factor"] = cfg_dict["moe_router_routed_scaling_factor"]

        if is_gigachat:
            self._apply_logit_softcapping(cfg_dict, ckpt_cfg)

        if cfg_dict.get("q_lora_rank", None) == 0:
            cfg_dict["q_lora_rank"] = None

        cfg_dict.pop("linear_o_norm_zero_centered", None)

        self._apply_token_id_defaults(cfg_dict)
        torch_dtype = self._apply_torch_dtype(cfg_dict, args)

        cfg_dict.setdefault("model_type", "deepseek_v3")

        attn_type = cfg_dict.get("attention_type")
        assert attn_type in ("LlamaLatentAttention", "LlamaLatentSparseAttention"), \
            f"Only attention_type in ('LlamaLatentAttention', 'LlamaLatentSparseAttention') is supported, got: {attn_type}"

        num_krope_heads = cfg_dict.get("num_krope_heads", None)
        assert num_krope_heads == 1, "Only num_krope_heads = 1 is supported"

        aux_free = cfg_dict.get("aux_loss_free", True)
        if not aux_free:
            warnings.warn(
                "DeepseekV3Mapper was tested with aux_loss_free=True. "
                "It should work with aux_loss_free=False, but correctness is not guaranteed.",
                RuntimeWarning,
            )

        scoring = cfg_dict.get("scoring_func", "sigmoid")
        assert scoring == "sigmoid", (
            "This mapper expects scoring_func='sigmoid'. "
            "If you need softmax gating, patch the modeling to handle 'softmax'."
        )

        g_topk = cfg_dict.get("moe_router_group_topk")
        n_groups = cfg_dict.get("moe_router_num_groups")

        if (g_topk is None) ^ (n_groups is None):
            raise ValueError("Specify both moe_router_group_topk and moe_router_num_groups, or neither.")

        if g_topk is None and n_groups is None:
            cfg_dict["n_group"] = 1
            cfg_dict["topk_group"] = 1
        elif (g_topk, n_groups) == (4, 8):
            cfg_dict["n_group"] = n_groups
            cfg_dict["topk_group"] = g_topk
        else:
            warnings.warn("This (group_topk, num_groups) configuration is not tested yet.", RuntimeWarning)
            cfg_dict["n_group"] = n_groups if n_groups is not None else 1
            cfg_dict["topk_group"] = g_topk if g_topk is not None else 1

        n_experts = cfg_dict.get("n_routed_experts")
        if n_experts is not None:
            assert n_experts % cfg_dict["n_group"] == 0, \
                f"n_routed_experts ({n_experts}) must be divisible by n_group ({cfg_dict['n_group']})"

        self._apply_mtp_fields(cfg_dict, ckpt_cfg)
        cfg_dict["rope_scaling"] = self._normalize_rope_scaling(cfg_dict.get("rope_scaling"))
        _coerce_template_float_fields(cfg_dict, template)

        torch.set_default_dtype(getattr(torch, torch_dtype))
        hf_cfg = AutoConfig.from_pretrained(
            spec.template_dir,
            trust_remote_code=True,
            **cfg_dict
        )

        hf_cfg.__dict__.pop("quantization_config", None)

        hf_cfg.bos_token_id = 1
        hf_cfg.eos_token_id = 2
        hf_cfg.pad_token_id = 2

        # Checkpoint `content` may carry moe_kernel / moe_impl for training; DeepseekV3Config does not
        # declare them as first-class fields, so they may not be readable from hf_cfg after
        # from_pretrained. Downstream _DeepseekPerLayerTransformer must see them for sonic layout.
        for _k in ("moe_kernel", "moe_impl"):
            if _k in cfg_dict and cfg_dict[_k] is not None:
                try:
                    setattr(hf_cfg, _k, cfg_dict[_k])
                except Exception:
                    hf_cfg.__dict__[_k] = cfg_dict[_k]
        logger.info(
            "DeepseekV3Mapper MoE layout flags from checkpoint: moe_kernel=%r moe_impl=%r "
            "(use_sonic will be moe_kernel=='sonic' or moe_impl=='SonicGMMMoeBlock')",
            getattr(hf_cfg, "moe_kernel", None),
            getattr(hf_cfg, "moe_impl", None),
        )

        if not is_gigachat:
            return hf_cfg

        hf_cfg.model_type = spec.model_type
        hf_cfg.architectures = [spec.architecture]

        hybrid_cfg_dict = deepcopy(hf_cfg.to_dict())
        hybrid_hf_cfg = Gigachat35Config.from_dict(hybrid_cfg_dict)
        hybrid_hf_cfg._auto_map_for_save = spec.auto_map

        if hybrid_hf_cfg.head_dim is None:
            hybrid_hf_cfg.head_dim = hybrid_hf_cfg.qk_rope_head_dim

        for attr in ("_name_or_path", "name_or_path"):
            if hasattr(hybrid_hf_cfg, attr):
                setattr(hybrid_hf_cfg, attr, None)
        return hybrid_hf_cfg

    def pack_conv_weights(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        linear_attention_type = (
            self.cfg.get("linear_attention_type")
            if isinstance(self.cfg, dict)
            else getattr(self.cfg, "linear_attention_type", None)
        )
        if linear_attention_type == "KimiDeltaAttention":
            return tensors

        new_tensors = {}
        short_conv_buffers = {}

        for k, v in tensors.items():
            print(f"Layer: {k}")

            if ".self_attn." in k and any(f".{c}_conv1d." in k for c in ("q", "k", "v")):
                parts = k.split(".")

                for c in ("q", "k", "v"):
                    name = f"{c}_conv1d"
                    if name in parts:
                        idx = parts.index(name)
                        prefix = ".".join(parts[:idx])
                        param = parts[-1]
                        break

                short_conv_buffers.setdefault(prefix, {})
                short_conv_buffers[prefix][(c, param)] = v
                continue

            new_tensors[k] = v

        for prefix, buf in short_conv_buffers.items():
            try:
                q_w = buf[("q", "weight")]
                k_w = buf[("k", "weight")]
                v_w = buf[("v", "weight")]
            except KeyError as e:
                raise RuntimeError(f"Missing short-conv weight for layer {prefix}: {e}")

            new_tensors[f"{prefix}.conv1d.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

            if ("q", "bias") in buf:
                new_tensors[f"{prefix}.conv1d.bias"] = torch.cat(
                    [buf[("q", "bias")], buf[("k", "bias")], buf[("v", "bias")]],
                    dim=0,
                )

        return new_tensors
        
    def tokenizer_source(self, ckpt_cfg: Dict[str, Any], args) -> str:
        return getattr(args, "tokenizer_path", None) or ckpt_cfg.get("_name_or_path") or ""

    def save_meta(self, cfg_src: Dict[str, Any], args, output_dir: str) -> None:
        cfg_obj, cfg_dict, tmp_model = _save_config_and_tokenizer(
            self, cfg_src, args, output_dir, with_tmp_model=True
        )
        try:
            vendor_deepseek_runtime_code(
                output_dir,
                cfg_obj=getattr(self, "cfg", None) or cfg_obj,
                model_obj=tmp_model
            )
        except Exception as e:
            logger.warning(f"Failed to vendor DeepSeek runtime code: {e}")
        finally:
            del tmp_model

        # For GigaChat3.5 exports, also vendor the model-specific runtime files.
        effective_cfg = getattr(self, "cfg", None) or cfg_obj
        effective_cfg_dict = (
            effective_cfg.to_dict()
            if hasattr(effective_cfg, "to_dict")
            else (cfg_dict or {})
        )
        if is_gigachat_model(effective_cfg_dict):
            try:
                regenerate = getattr(args, "regenerate_modeling", True)
                vendor_gigachat35_runtime_code(
                    output_dir,
                    regenerate_modeling=regenerate,
                )
                logger.info(
                    "[rank 0] Vendored gigachat3_5 modeling + configuration files."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to vendor GigaChat3.5 runtime code: {e}"
                )

    def write_export_manifest(self, cfg_src: Dict[str, Any], args, output_dir: str) -> None:
        # Prefer self.cfg (built by build_config) over the raw training cfg_src so that detection uses HF-side fields
        effective = self.cfg if self.cfg is not None else cfg_src
        if hasattr(effective, "to_dict"):
            effective = effective.to_dict()
        write_hf_assembly_manifest(output_dir, args, model_family=self._model_spec(effective).model_type)


class Gigachat35DenseMapper(DeepseekV3Mapper):
    """GigaChat3.5 with dense MLP and GQA full attention, optionally hybrid with linear attention.

    Shares MTP, norm and linear-attention handling with the parent; MLA/MoE steps are not applied.
    """

    def _build_transformer(self, cfg: Any):
        from transformers_utils.gigachat3_5_dense import _Gigachat35DensePerLayerTransformer
        return _Gigachat35DensePerLayerTransformer(cfg)

    def _model_spec(self, cfg_dict: Dict[str, Any]) -> HFModelSpec:
        return GIGACHAT3_5_DENSE_SPEC

    @staticmethod
    def _head_dim(ckpt_cfg: Dict[str, Any]) -> int:
        # Same precedence as the training LlamaAttention
        attention_hidden_size = ckpt_cfg.get("attention_hidden_size") or ckpt_cfg["hidden_size"]
        return ckpt_cfg.get("head_dim") or attention_hidden_size // ckpt_cfg["num_attention_heads"]

    def _validate_dense_gqa(self, ckpt_cfg: Dict[str, Any]) -> None:
        attn_type = ckpt_cfg.get("attention_type")
        if attn_type != "LlamaAttention":
            raise ValueError(
                f"Gigachat35DenseMapper supports attention_type='LlamaAttention', got {attn_type!r}; "
                "MLA checkpoints are handled by DeepseekV3Mapper"
            )
        if ckpt_cfg.get("apply_qk_norm"):
            raise ValueError("apply_qk_norm=True is not supported by the dense GQA modeling")
        if ckpt_cfg.get("attention_bias"):
            raise ValueError("attention_bias=True is not supported by the dense mapper")

        # Training rotates `head_dim or hidden_size // heads` dims; the HF modeling rotates the whole head
        head_dim = self._head_dim(ckpt_cfg)
        rotary_dim = ckpt_cfg.get("head_dim") or ckpt_cfg["hidden_size"] // ckpt_cfg["num_attention_heads"]
        if rotary_dim != head_dim:
            raise ValueError(
                f"attention_hidden_size={ckpt_cfg.get('attention_hidden_size')} implies partial rotary "
                f"({rotary_dim} of {head_dim} dims), which the dense modeling does not support"
            )

        num_experts = ckpt_cfg.get("num_routed_experts") or ckpt_cfg.get("n_routed_experts") or 0
        first_moe_layer = ckpt_cfg.get("first_k_dense_replace")
        if num_experts > 0 and (first_moe_layer is None or first_moe_layer < ckpt_cfg["num_hidden_layers"]):
            raise ValueError(
                f"num_routed_experts={num_experts}, first_k_dense_replace={first_moe_layer}: "
                "MoE layers present, use DeepseekV3Mapper"
            )
        # The training default for mtp_block_type is 'moe', so it has to be stated explicitly
        if ckpt_cfg.get("use_mtp") and self._get_nextn_is_sparse(ckpt_cfg) is not False:
            raise ValueError("Dense MTP requires mtp_block_type='dense' in the checkpoint config")
        if ckpt_cfg.get("use_mtp") and resolve_mtp_uses_linear_attention(ckpt_cfg):
            raise ValueError(
                "Dense MTP requires full attention in the MTP block: "
                "set use_linear_attention_in_mtp=False (nextn_is_linear=False)"
            )

    def build_config(self, ckpt_cfg: Dict[str, Any], args) -> Gigachat35DenseConfig:
        ckpt_cfg = self._strip_export_fields(ckpt_cfg)
        self._validate_dense_gqa(ckpt_cfg)

        spec = self._model_spec(ckpt_cfg)
        template = self._load_template(spec)
        cfg_dict = {**template, **ckpt_cfg}

        if ckpt_cfg.get("dense_intermediate_size") is not None:
            cfg_dict["intermediate_size"] = ckpt_cfg["dense_intermediate_size"]
        cfg_dict["head_dim"] = self._head_dim(ckpt_cfg)
        cfg_dict["num_key_value_heads"] = ckpt_cfg.get("num_key_value_heads") or cfg_dict["num_attention_heads"]

        self._apply_logit_softcapping(cfg_dict, ckpt_cfg)
        self._apply_token_id_defaults(cfg_dict)
        self._apply_torch_dtype(cfg_dict, args)
        self._apply_mtp_fields(cfg_dict, ckpt_cfg)
        cfg_dict["rope_scaling"] = self._normalize_rope_scaling(cfg_dict.get("rope_scaling"))

        # The template lists the exported fields; training-only keys are dropped here
        exported = {
            k: cfg_dict[k] for k in template
            if k in cfg_dict and k not in ("model_type", "architectures", "auto_map")
        }
        hf_cfg = Gigachat35DenseConfig(**exported)
        self._apply_sglang_mla_moe_fields(hf_cfg)
        hf_cfg.architectures = [spec.architecture]
        hf_cfg._auto_map_for_save = spec.auto_map
        return hf_cfg

    @staticmethod
    def _apply_sglang_mla_moe_fields(hf_cfg: Gigachat35DenseConfig) -> None:
        # SGLang builds a DeepseekV2 layer before swapping in GQA and dense MLP; HF modeling never reads these
        hf_cfg.first_k_dense_replace = hf_cfg.num_hidden_layers
        hf_cfg.n_routed_experts = 0
        hf_cfg.moe_layer_freq = 1
        hf_cfg.nextn_is_sparse = False
        hf_cfg.nextn_is_linear = False
        hf_cfg.q_lora_rank = None
        hf_cfg.kv_lora_rank = 1  # only sized by SGLang, never used; must be > 0
        hf_cfg.qk_nope_head_dim = hf_cfg.qk_rope_head_dim = hf_cfg.v_head_dim = hf_cfg.head_dim

    def save_meta(self, cfg_src: Dict[str, Any], args, output_dir: str) -> None:
        _save_config_and_tokenizer(self, cfg_src, args, output_dir)
        spec = self._model_spec(cfg_src)
        vendor_runtime_code(output_dir, spec)
        logger.info("[rank 0] Vendored %s modeling + configuration files.", spec.model_type)


class GigachatSLMMapper(DeepseekV3Mapper):
    """Dense SLM with `attention_type: LlamaAttention`: GQA, qk-norm, SWA, dense MTP, gated attention, XSA, short convs."""

    def _build_transformer(self, cfg: Any):
        from transformers_utils.gigachat_slm import _GigachatSLMPerLayerTransformer
        return _GigachatSLMPerLayerTransformer(cfg)

    def _model_spec(self, cfg_dict: Dict[str, Any]) -> HFModelSpec:
        return GIGACHAT_SLM_SPEC

    def map(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return _to_bf16(self.tr.rename_and_split(tensors))

    @staticmethod
    def _head_dim(ckpt_cfg: Dict[str, Any]) -> int:
        # Same precedence as the training LlamaAttention
        attention_hidden_size = ckpt_cfg.get("attention_hidden_size") or ckpt_cfg["hidden_size"]
        return ckpt_cfg.get("head_dim") or attention_hidden_size // ckpt_cfg["num_attention_heads"]

    def _validate(self, ckpt_cfg: Dict[str, Any]) -> None:
        attn_type = ckpt_cfg.get("attention_type") or "LlamaAttention"
        if attn_type != "LlamaAttention":
            raise ValueError(
                f"GigachatSLMMapper supports attention_type='LlamaAttention', got {attn_type!r}"
            )
        if ckpt_cfg.get("apply_qk_norm") is False:
            raise ValueError("apply_qk_norm=False is not supported: the gigachat_slm modeling always applies q/k RMSNorm")
        if ckpt_cfg.get("use_qk_gain"):
            raise ValueError("use_qk_gain=True is not supported by the gigachat_slm modeling yet")
        if (ckpt_cfg.get("swiglu_limit") or 0.0) > 0:
            raise ValueError(f"swiglu_limit={ckpt_cfg['swiglu_limit']} is not supported by the gigachat_slm modeling")
        if ckpt_cfg.get("embed_scale", 1.0) not in (None, 1.0):
            raise ValueError(f"embed_scale={ckpt_cfg['embed_scale']} is not supported by the gigachat_slm modeling")
        if ckpt_cfg.get("attention_bias"):
            raise ValueError("attention_bias=True is not supported by GigachatSLMMapper")
        layernorm_type = ckpt_cfg.get("layernorm_type") or "pre"
        if layernorm_type != "pre":
            raise ValueError(
                f"layernorm_type={layernorm_type!r} is not supported: the gigachat_slm modeling is pre-norm only"
            )
        norm_type = ckpt_cfg.get("norm_type") or "LlamaRMSNorm"
        if norm_type not in ("LlamaRMSNorm", "rmsnorm"):
            raise ValueError(f"norm_type={norm_type!r} is not supported: the gigachat_slm modeling uses RMSNorm")
        if ckpt_cfg.get("linear_attention_type") is not None:
            raise ValueError(
                f"linear_attention_type={ckpt_cfg['linear_attention_type']!r}: hybrid linear attention is not "
                "supported by GigachatSLMMapper, use Gigachat35DenseMapper"
            )
        softcapping = ckpt_cfg.get("lm_head_logit_softcapping") or ckpt_cfg.get("final_logit_softcapping")
        if softcapping:
            raise ValueError(f"final_logit_softcapping={softcapping} is not supported by the gigachat_slm modeling")

        head_dim = self._head_dim(ckpt_cfg)
        rotary_dim = ckpt_cfg.get("head_dim") or ckpt_cfg["hidden_size"] // ckpt_cfg["num_attention_heads"]
        if rotary_dim != head_dim:
            raise ValueError(
                f"attention_hidden_size={ckpt_cfg.get('attention_hidden_size')} implies partial rotary "
                f"({rotary_dim} of {head_dim} dims), which the gigachat_slm modeling does not support"
            )

        num_experts = ckpt_cfg.get("num_routed_experts") or ckpt_cfg.get("n_routed_experts") or 0
        first_moe_layer = ckpt_cfg.get("first_k_dense_replace")
        if num_experts > 0 and (first_moe_layer is None or first_moe_layer < ckpt_cfg["num_hidden_layers"]):
            raise ValueError(
                f"num_routed_experts={num_experts}, first_k_dense_replace={first_moe_layer}: "
                "MoE layers present, use DeepseekV3Mapper"
            )
        if ckpt_cfg.get("use_mtp") and self._get_nextn_is_sparse(ckpt_cfg) is not False:
            raise ValueError("Dense MTP requires mtp_block_type='dense' in the checkpoint config")

    @staticmethod
    def _apply_attn_features(cfg_dict: Dict[str, Any]) -> None:
        sconv = normalize_sconv_sites(cfg_dict.get("sconv"))
        cfg_dict["sconv"] = sconv
        if sconv:
            if cfg_dict.get("sconv_kernel_size") is None:
                raise ValueError(
                    f"sconv sites {sorted(sconv)} are enabled but sconv_kernel_size is missing from the checkpoint "
                    "config; set it explicitly"
                )
        else:
            cfg_dict["sconv_kernel_size"] = None
        cfg_dict["gated_attention"] = bool(cfg_dict.get("gated_attention", False))
        cfg_dict["use_xsa"] = bool(cfg_dict.get("use_xsa", False))

    @staticmethod
    def _apply_sliding_window(cfg_dict: Dict[str, Any]) -> None:
        training_layer_types = cfg_dict.pop("sliding_window_layer_types", None)
        if training_layer_types is None:
            return
        layer_types = cfg_dict.get("layer_types")
        if layer_types is not None and list(layer_types) != list(training_layer_types):
            raise ValueError("Config contains both layer_types and sliding_window_layer_types, but they differ.")
        cfg_dict["layer_types"] = list(training_layer_types)

    @staticmethod
    def _tie_word_embeddings(ckpt_cfg: Dict[str, Any]) -> bool:
        embedding_group = {"model.embed_tokens", "lm_head"}
        tied_groups = ckpt_cfg.get("tied_modules_groups") or []
        return bool(ckpt_cfg.get("tie_word_embeddings", False)) or any(
            embedding_group.issubset(set(group)) for group in tied_groups
        )

    def build_config(self, ckpt_cfg: Dict[str, Any], args) -> GigachatSLMConfig:
        ckpt_cfg = self._strip_export_fields(ckpt_cfg)
        self._validate(ckpt_cfg)

        spec = self._model_spec(ckpt_cfg)
        template = self._load_template(spec)
        cfg_dict = {**template, **ckpt_cfg}

        if ckpt_cfg.get("dense_intermediate_size") is not None:
            cfg_dict["intermediate_size"] = ckpt_cfg["dense_intermediate_size"]
        cfg_dict["head_dim"] = self._head_dim(ckpt_cfg)
        cfg_dict["num_key_value_heads"] = ckpt_cfg.get("num_key_value_heads") or cfg_dict["num_attention_heads"]
        cfg_dict["use_cache"] = True
        cfg_dict["tie_word_embeddings"] = self._tie_word_embeddings(ckpt_cfg)

        self._apply_attn_features(cfg_dict)
        self._apply_sliding_window(cfg_dict)
        self._apply_token_id_defaults(cfg_dict)
        self._apply_torch_dtype(cfg_dict, args)
        self._apply_mtp_fields(cfg_dict, ckpt_cfg)
        cfg_dict["rope_scaling"] = self._normalize_rope_scaling(cfg_dict.get("rope_scaling"))

        exported = {
            k: cfg_dict[k] for k in template
            if k in cfg_dict and k not in ("model_type", "architectures", "auto_map")
        }
        hf_cfg = GigachatSLMConfig(**exported)
        hf_cfg.architectures = [spec.architecture]
        hf_cfg._auto_map_for_save = spec.auto_map
        return hf_cfg

    def save_meta(self, cfg_src: Dict[str, Any], args, output_dir: str) -> None:
        _save_config_and_tokenizer(self, cfg_src, args, output_dir)
        spec = self._model_spec(cfg_src)
        vendor_runtime_code(output_dir, spec)
        logger.info("[rank 0] Vendored %s modeling + configuration files.", spec.model_type)


class OmniDeepseekV3Mapper(DeepseekV3Mapper):
    def _remove_model_prefix(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k[6:] if k.startswith("model.") else k : v 
            for k, v in tensors.items()
            }

    @staticmethod
    def _add_language_model_prefix(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {f"language_model.{name}": weight for name, weight in tensors.items()}

    def map(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tensors = self._remove_model_prefix(tensors)
        lm_weights = {}
        other_weights = {}
        for name, weight in tensors.items():
            if name.startswith("language_model."):
                lm_weights[name[len("language_model."):]] = weight
            else:
                other_weights[name] = weight

        out = _to_bf16(other_weights)
        if lm_weights:
            out.update(self._add_language_model_prefix(super().map(lm_weights)))
        return out

    def flush(self) -> Dict[str, torch.Tensor]:
        return self._add_language_model_prefix(super().flush())

    def ensure_config(self, ckpt_cfg: Dict[str, Any], args) -> None:
        if ckpt_cfg["model_type"] == "gigaomni":
            ckpt_cfg = ckpt_cfg["language_model"]
        super().ensure_config(ckpt_cfg, args)
    
    def save_meta(self, cfg_src: Dict[str, Any], args, output_dir: str) -> None:
        if cfg_src["model_type"] == "gigaomni":
            cfg_src = cfg_src["language_model"]
        super().save_meta(cfg_src, args, output_dir)
    
    def build_config(self, ckpt_cfg: Dict[str, Any], args) -> AutoConfig:
        if ckpt_cfg["model_type"] == "gigaomni":
            ckpt_cfg = ckpt_cfg["language_model"]
        if "pretrained_model_name_or_path" in ckpt_cfg:
            del ckpt_cfg["pretrained_model_name_or_path"]
        return super().build_config(ckpt_cfg, args)
        

class OmniRawMapper(RawMapper):
    def ensure_config(self, ckpt_cfg: Dict[str, Any], args) -> None:
        self.model_type = ckpt_cfg.get("model_type")
        super().ensure_config(ckpt_cfg, args)

    def map(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.model_type == "gigaomni":
            return _to_bf16(tensors)
        return self._rename_layers(_to_bf16(tensors))

    def _rename_layers(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        renamed_tensors = {}
        for name, param in tensors.items():
            renamed_tensors[self._rename_param_name(name)] = param
        return renamed_tensors

    def _rename_param_name(self, name: str) -> str:
        if ".lm_head" in name:
            return name.replace("model.lm_head", "model.language_model.lm_head")
        return name.replace("model", "model.language_model.model")
