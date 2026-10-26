from __future__ import annotations

import json
import os
import inspect
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
from transformers import AutoTokenizer, GenerationConfig, PretrainedConfig
from tokenizers.processors import TemplateProcessing

try:
    from convertation.base_mappers import DeepseekV3Mapper, Gigachat35Config, Gigachat35DenseMapper, GigachatSLMMapper, LayerMapper
except BaseException:
    try:
        from base_mappers import DeepseekV3Mapper, Gigachat35Config, Gigachat35DenseMapper, GigachatSLMMapper, LayerMapper
    except BaseException:
        current_file = Path(__file__).resolve().parent
        convertation_dir = current_file.parent
        if str(convertation_dir) not in sys.path:
            sys.path.insert(0, str(convertation_dir))
        from base_mappers import DeepseekV3Mapper, Gigachat35Config, Gigachat35DenseMapper, GigachatSLMMapper, LayerMapper

try:
    from utils.file_utils import _atomic_write_json, _copy_path, vendor_runtime_code
    from utils.hf_assembly_manifest import HF_ASSEMBLY_MANIFEST, load_hf_assembly_manifest
    from utils.hf_model_specs import (
        DEEPSEEK_V3_SPEC,
        GIGACHAT3_5_DENSE_SPEC,
        GIGACHAT3_5_SPEC,
        GIGACHAT_SLM_SPEC,
        MODEL_SPECS,
        HFModelSpec,
    )
    from utils.llmfoundry_config_utils import (
        _find_llmfoundry_yaml,
        enrich_hf_config_from_llmfoundry_yaml,
        is_gigachat_model,
        load_llmfoundry_yaml,
    )
except BaseException:
    current_file = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_file))
    from file_utils import _atomic_write_json, _copy_path, vendor_runtime_code
    from hf_assembly_manifest import HF_ASSEMBLY_MANIFEST, load_hf_assembly_manifest
    from hf_model_specs import (
        DEEPSEEK_V3_SPEC,
        GIGACHAT3_5_DENSE_SPEC,
        GIGACHAT3_5_SPEC,
        GIGACHAT_SLM_SPEC,
        MODEL_SPECS,
        HFModelSpec,
    )
    from llmfoundry_config_utils import (
        _find_llmfoundry_yaml,
        enrich_hf_config_from_llmfoundry_yaml,
        is_gigachat_model,
        load_llmfoundry_yaml,
    )


GIGACHAT3_5 = GIGACHAT3_5_SPEC.auto_map

# Families whose config cannot be built by the default DeepseekV3Mapper
_MAPPER_BY_FAMILY: dict[str, type] = {
    GIGACHAT3_5_DENSE_SPEC.model_type: Gigachat35DenseMapper,
    GIGACHAT_SLM_SPEC.model_type: GigachatSLMMapper,
}


def _mapper_for_spec(spec: HFModelSpec) -> LayerMapper:
    return _MAPPER_BY_FAMILY.get(spec.model_type, DeepseekV3Mapper)()

_TOKENIZER_ARTIFACT_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_tokenizer_path_from_yaml(yaml_path: Optional[str]) -> Optional[str]:
    if yaml_path is None:
        return None

    yaml_cfg = load_llmfoundry_yaml(yaml_path)
    tokenizer_path = yaml_cfg.get("tokenizer_path")
    if tokenizer_path:
        return str(tokenizer_path)

    tokenizer_cfg = yaml_cfg.get("tokenizer", {})
    tokenizer_name = tokenizer_cfg.get("name") if tokenizer_cfg else None
    if tokenizer_name:
        return str(tokenizer_name)

    return None


def _load_embedded_hf_config(checkpoint_dir: str) -> dict[str, Any]:
    composer_state_path = Path(checkpoint_dir) / "composer_state.pt" if Path(checkpoint_dir).name != "language_model" else Path(checkpoint_dir).parent / "composer_state.pt"
    if not composer_state_path.is_file():
        return {}

    composer_state = _torch_load(composer_state_path)
    try:
        return dict(
            composer_state["state"]["integrations"]["huggingface"]["model"]["config"]["content"]
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Could not extract embedded HF config from {composer_state_path}"
        ) from exc


def _load_existing_config(checkpoint_dir: str) -> dict[str, Any]:
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.is_file():
        return {}
    return _load_json(config_path)


def _resolve_llmfoundry_yaml(checkpoint_dir: str, explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        return str(Path(explicit_path).resolve())

    found = _find_llmfoundry_yaml(checkpoint_dir)
    return str(found) if found is not None else None


def resolve_tokenizer_source(
    checkpoint_dir: str,
    *,
    explicit_tokenizer_path: Optional[str] = None,
    llmfoundry_config: Optional[str] = None,
    embedded_cfg: Optional[dict[str, Any]] = None,
    existing_cfg: Optional[dict[str, Any]] = None,
) -> str:
    embedded_cfg = embedded_cfg or {}
    existing_cfg = existing_cfg or {}
    manifest = load_hf_assembly_manifest(checkpoint_dir)

    high_priority_candidates = [
        explicit_tokenizer_path,
        manifest.get("tokenizer_path"),
    ]

    for candidate in high_priority_candidates:
        if candidate:
            return str(candidate)

    try:
        yaml_tokenizer_path = _extract_tokenizer_path_from_yaml(llmfoundry_config)
    except Exception as exc:
        warnings.warn(
            "Failed to extract tokenizer path from llm_foundry_config.yaml. "
            f"Continuing with lower-priority fallbacks. Reason: {exc}",
            RuntimeWarning,
        )
        yaml_tokenizer_path = None

    if yaml_tokenizer_path:
        return str(yaml_tokenizer_path)

    fallback_candidates = [
        embedded_cfg.get("_name_or_path"),
        existing_cfg.get("_name_or_path"),
    ]

    for candidate in fallback_candidates:
        if candidate:
            return str(candidate)

    raise RuntimeError(
        "Tokenizer path is undefined. Checked: CLI --tokenizer-path, "
        "hf_assembly_plan.json, llm_foundry_config.yaml tokenizer_path/tokenizer.name, "
        "embedded config _name_or_path, existing config _name_or_path."
    )


def _select_model_spec(cfg_dict: dict[str, Any]) -> HFModelSpec:
    if cfg_dict.get("model_type") == GIGACHAT_SLM_SPEC.model_type:
        return GIGACHAT_SLM_SPEC
    if (
        cfg_dict.get("model_type") == GIGACHAT3_5_DENSE_SPEC.model_type
        or cfg_dict.get("attention_type") == "LlamaAttention"
    ):
        return GIGACHAT3_5_DENSE_SPEC
    if is_gigachat_model(cfg_dict):
        return GIGACHAT3_5_SPEC
    return DEEPSEEK_V3_SPEC


def _force_hybrid_config_dict(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    forced_cfg = dict(cfg_dict)
    # `layer_types` must be derived from `linear_attention_type` + `full_attention_layers`
    # by `Gigachat35Config.__init__`. If we keep a stale value from a template or an earlier
    # DeepSeek/GigaChat config round-trip, `PretrainedConfig.from_dict` will restore it via
    # `**kwargs` after init and override the freshly computed layout.
    forced_cfg.pop("layer_types", None)
    forced_cfg["model_type"] = GIGACHAT3_5_SPEC.model_type
    forced_cfg["architectures"] = ["Gigachat35ForCausalLM"]
    forced_cfg["auto_map"] = dict(GIGACHAT3_5)

    num_layers = forced_cfg.get("num_hidden_layers")
    full_attention_layers = forced_cfg.get("full_attention_layers")
    if num_layers is not None and (
        full_attention_layers is None or (full_attention_layers == [] and forced_cfg.get("linear_attention_type") is None)
    ):
        forced_cfg["full_attention_layers"] = list(range(int(num_layers)))

    forced_cfg.setdefault("linear_attention_type", None)
    if forced_cfg.get("q_lora_rank") == 0:
        forced_cfg["q_lora_rank"] = None
    return forced_cfg


def _remap_giga_mix_intermediate_sizes(cfg_dict: dict[str, Any], source_cfg: dict[str, Any]) -> None:
    """Re-apply giga_mix → HF intermediate size mapping using the original source_cfg.

    In giga_mix: intermediate_size = per-expert FFN size → HF moe_intermediate_size
                 dense_intermediate_size = dense-layer FFN size → HF intermediate_size
    _preserve_force_hybrid_fields may have restored the raw training value of intermediate_size,
    so we re-apply the mapping here from source_cfg which still holds the original field names.
    """
    dense_sz = source_cfg.get("dense_intermediate_size")
    expert_sz = source_cfg.get("intermediate_size")
    if dense_sz is not None:
        cfg_dict["intermediate_size"] = dense_sz
        if expert_sz is not None and not source_cfg.get("moe_intermediate_size"):
            cfg_dict["moe_intermediate_size"] = expert_sz


def _build_forced_hybrid_config(cfg_dict: dict[str, Any]) -> Gigachat35Config:
    forced_cfg_dict = _force_hybrid_config_dict(cfg_dict)
    hybrid_cfg = Gigachat35Config.from_dict(forced_cfg_dict)
    hybrid_cfg._auto_map_for_save = dict(GIGACHAT3_5)
    if hybrid_cfg.head_dim is None:
        hybrid_cfg.head_dim = hybrid_cfg.qk_rope_head_dim
    for attr in ("_name_or_path", "name_or_path"):
        if hasattr(hybrid_cfg, attr):
            setattr(hybrid_cfg, attr, None)
    return hybrid_cfg


def _gigachat35_config_param_names() -> set[str]:
    return {
        name
        for name, param in inspect.signature(Gigachat35Config.__init__).parameters.items()
        if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
    }


def _preserve_force_hybrid_fields(
    cfg_dict: dict[str, Any],
    source_cfg: dict[str, Any],
) -> dict[str, Any]:
    preserved = dict(cfg_dict)
    for key in _gigachat35_config_param_names():
        if key in source_cfg and source_cfg[key] is not None:
            preserved[key] = source_cfg[key]
    return preserved


def _apply_cfg_obj_auto_map(cfg_obj: Optional[PretrainedConfig], cfg_dict: dict[str, Any]) -> dict[str, Any]:
    if cfg_obj is None:
        return cfg_dict
    auto_map_for_save = getattr(cfg_obj, "_auto_map_for_save", None)
    if auto_map_for_save is not None:
        cfg_dict["auto_map"] = auto_map_for_save
    # to_dict() dumps the private attribute too; it must not end up in config.json
    cfg_dict.pop("_auto_map_for_save", None)
    return cfg_dict


def _normalize_torch_dtype(torch_dtype: Optional[str]) -> str:
    if torch_dtype in ("float16", "bfloat16"):
        return torch_dtype
    return "bfloat16"


def _write_config_json(config_path: Path, cfg: dict[str, Any], overwrite_config: bool) -> None:
    _atomic_write_json(str(config_path), cfg, overwrite=overwrite_config)


def _copy_local_tokenizer_files(tokenizer_path: str, output_dir: str) -> bool:
    src_dir = Path(tokenizer_path)
    if not src_dir.is_dir():
        return False

    copied_any = False
    for filename in _TOKENIZER_ARTIFACT_FILENAMES:
        copied_any = _copy_path(str(src_dir / filename), output_dir, dst_name=filename) or copied_any

    return copied_any


def _repair_tokenizer_post_processor(output_dir: str) -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
    except Exception as exc:
        warnings.warn(
            f"Could not reload tokenizer from {output_dir} to repair post-processor: {exc}",
            RuntimeWarning,
        )
        return

    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if backend_tokenizer is None or bos_token_id is None:
        return

    backend_tokenizer.post_processor = TemplateProcessing(
        single="<s> $A",
        special_tokens=[("<s>", bos_token_id)],
    )
    tokenizer.save_pretrained(output_dir)


def _cleanup_rank_metadata_dirs(output_dir: str) -> None:
    for rank_dir in sorted(output_dir.glob("rank*")):
        if not rank_dir.is_dir():
            continue

        for path in rank_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.name == "rank-index.json" or path.name.endswith(".weight_map.json"):
                path.unlink()

        if any(rank_dir.rglob("*.safetensors")):
            warnings.warn(
                f"Skipping cleanup of {rank_dir} because it still contains .safetensors shards.",
                RuntimeWarning,
            )
            continue

        for path in sorted(rank_dir.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

        try:
            rank_dir.rmdir()
        except OSError:
            warnings.warn(
                f"Could not remove rank metadata directory {rank_dir}; it still contains files.",
                RuntimeWarning,
            )


def _cleanup_intermediate_artifacts(output_dir: str) -> None:
    for filename in ("composer_state.pt", HF_ASSEMBLY_MANIFEST):
        path = output_dir / filename
        if path.is_file():
            path.unlink()

    _cleanup_rank_metadata_dirs(output_dir)


def _copy_runtime_for_spec(output_dir: str, spec: HFModelSpec) -> None:
    """Install the runtime files of `spec` and drop those left by other families."""
    vendor_runtime_code(output_dir, spec)
    for other in MODEL_SPECS.values():
        for filename in other.runtime_filenames:
            stale_file = Path(output_dir) / filename
            if filename not in spec.runtime_filenames and stale_file.is_file():
                stale_file.unlink()


def _save_generation_config(output_dir: str, cfg_obj: Optional[PretrainedConfig]) -> None:
    if cfg_obj is None:
        return
    try:
        generation_config = GenerationConfig.from_model_config(cfg_obj)
        generation_config.save_pretrained(output_dir)
    except Exception:
        pass


def build_hf_config(
    checkpoint_dir: str,
    *,
    llmfoundry_config: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    model_spec: Optional[HFModelSpec] = None,
    mapper: Optional[LayerMapper] = None,
) -> tuple[Optional[PretrainedConfig], dict[str, Any], str]:
    checkpoint_dir = os.path.realpath(checkpoint_dir)
    resolved_yaml = _resolve_llmfoundry_yaml(checkpoint_dir, llmfoundry_config)
    embedded_cfg = _load_embedded_hf_config(checkpoint_dir)
    existing_cfg = _load_existing_config(checkpoint_dir)
    source_cfg = embedded_cfg or existing_cfg
    if not source_cfg:
        raise RuntimeError(
            f"Could not build HF config for {checkpoint_dir}: neither composer_state.pt nor config.json is available."
        )

    source_cfg = enrich_hf_config_from_llmfoundry_yaml(
        dict(source_cfg),
        checkpoint_path=checkpoint_dir,
        yaml_path=resolved_yaml,
    )

    args = SimpleNamespace(
        input_checkpoint_path=checkpoint_dir,
        torch_dtype=_normalize_torch_dtype(torch_dtype or source_cfg.get("torch_dtype")),
    )

    # Explicit spec, then the family recorded by the streaming step, then detection from the config
    effective_spec = (
        model_spec
        or MODEL_SPECS.get(load_hf_assembly_manifest(checkpoint_dir).get("model_family"))
        or _select_model_spec(source_cfg)
    )
    mapper = mapper or _mapper_for_spec(effective_spec)
    cfg_obj = mapper.build_config(dict(source_cfg), args)
    cfg_dict = cfg_obj.to_dict() if isinstance(cfg_obj, PretrainedConfig) else dict(cfg_obj)
    cfg_dict = _apply_cfg_obj_auto_map(cfg_obj, cfg_dict)

    if effective_spec is GIGACHAT3_5_SPEC and not is_gigachat_model(cfg_dict):
        cfg_dict = _preserve_force_hybrid_fields(cfg_dict, source_cfg)
        _remap_giga_mix_intermediate_sizes(cfg_dict, source_cfg)
        cfg_obj = _build_forced_hybrid_config(cfg_dict)
        cfg_dict = cfg_obj.to_dict()
        cfg_dict = _apply_cfg_obj_auto_map(cfg_obj, cfg_dict)

    cfg_dict["model_type"] = effective_spec.model_type
    cfg_dict["architectures"] = [effective_spec.architecture]
    cfg_dict.pop("_name_or_path", None)
    cfg_dict.pop("name_or_path", None)

    resolved_tokenizer_path = resolve_tokenizer_source(
        checkpoint_dir,
        explicit_tokenizer_path=tokenizer_path,
        llmfoundry_config=resolved_yaml,
        embedded_cfg=embedded_cfg,
        existing_cfg=existing_cfg,
    )
    tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_path)
    if tokenizer.bos_token_id is not None:
        cfg_dict["bos_token_id"] = tokenizer.bos_token_id
        if cfg_obj is not None:
            cfg_obj.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        cfg_dict["eos_token_id"] = tokenizer.eos_token_id
        if cfg_obj is not None:
            cfg_obj.eos_token_id = tokenizer.eos_token_id
    cfg_dict["pad_token_id"] = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    if cfg_obj is not None:
        cfg_obj.pad_token_id = cfg_dict["pad_token_id"]

    return cfg_obj if isinstance(cfg_obj, PretrainedConfig) else None, cfg_dict, resolved_tokenizer_path


def assemble_hf_checkpoint(
    checkpoint_dir: str,
    *,
    llmfoundry_config: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    overwrite_config: bool = False,
    torch_dtype: Optional[str] = None,
    model_spec: Optional[HFModelSpec] = None,
    cleanup_intermediate_artifacts: bool = True,
    force: Optional[bool] = None,
    mapper: Optional[LayerMapper] = None,
) -> dict[str, Any]:
    checkpoint_dir = os.path.realpath(checkpoint_dir)
    output_dir = Path(checkpoint_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    if force is not None:
        overwrite_config = overwrite_config or force

    cfg_obj, cfg_dict, resolved_tokenizer_path = build_hf_config(
        checkpoint_dir,
        llmfoundry_config=llmfoundry_config,
        tokenizer_path=tokenizer_path,
        torch_dtype=torch_dtype,
        model_spec=model_spec,
        mapper=mapper,
    )

    _write_config_json(
        output_dir / "config.json",
        cfg_dict,
        overwrite_config=overwrite_config,
    )

    if not _copy_local_tokenizer_files(resolved_tokenizer_path, checkpoint_dir):
        tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_path)
        tokenizer.save_pretrained(checkpoint_dir)
    _repair_tokenizer_post_processor(checkpoint_dir)
    _save_generation_config(checkpoint_dir, cfg_obj)

    spec = _select_model_spec(cfg_dict)
    _copy_runtime_for_spec(checkpoint_dir, spec)

    if cleanup_intermediate_artifacts:
        _cleanup_intermediate_artifacts(output_dir)

    return {
        "config": cfg_dict,
        "tokenizer_path": resolved_tokenizer_path,
        "model_spec": spec.model_type,
    }
