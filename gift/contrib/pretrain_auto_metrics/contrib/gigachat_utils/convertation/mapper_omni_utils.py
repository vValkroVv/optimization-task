import copy
import json
import shutil
import sys
import tempfile
import types

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

from omegaconf import DictConfig, OmegaConf as om
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from base_mappers import LayerMapper

from logger import logger
from utils.merge_ranks import flatten_move_and_reindex
from utils.hf_assembly_manifest import has_hf_assembly_manifest, load_hf_assembly_manifest
from utils.hf_model_specs import MODEL_SPECS
from utils.hf_checkpoint_assembly import (
    _write_config_json,
    assemble_hf_checkpoint,
)

_DUMMY_PROCESSOR_CONFIG_DIR = Path(__file__).absolute().parent / "sglang_resources" / "dummy_config"


def _find_llmfoundry_root() -> Path:
    for parent in Path(__file__).absolute().parents:
        if (parent / "llmfoundry").is_dir():
            return parent
    raise RuntimeError(f"Couldn't find llm-foundry above {__file__}")


def _register_namespace_package(package_name: str, directory: str):
    '''
    Register fake package into sys.modules pointing at directory, so its real __init__.py never runs.
    Submodules are still importable, e.g. after _register_namespace_package("llmfoundry.data", ".../llmfoundry/data")
    `import llmfoundry.data.processors` works, but llmfoundry/data/__init__.py does not run.
    '''
    if package_name in sys.modules:
        return
    package = types.ModuleType(package_name)
    package.__path__ = [directory]
    package.__package__ = package_name
    sys.modules[package_name] = package
    parent, _, child = package_name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, package)


def add_llmfoundry_to_path():
    root = _find_llmfoundry_root()
    sys.path = [
        str(root),
        str(root / "contrib" / "composer"),
        str(root / "contrib" / "streaming"),
        str(root / "contrib" / "gigafsdp"),
    ] + sys.path

    llmf_path = root / "llmfoundry"
    _register_namespace_package("llmfoundry", str(llmf_path))
    _register_namespace_package("llmfoundry.models", str(llmf_path / "models"))
    _register_namespace_package("llmfoundry.models.gigaomni", str(llmf_path / "models" / "gigaomni"))
    _register_namespace_package("llmfoundry.data", str(llmf_path / "data"))


def load_yaml_config(checkpoint_path: Union[List[str], str]) -> DictConfig:
    if isinstance(checkpoint_path, list):
        checkpoint_path = checkpoint_path[0]

    yaml_config = om.load(Path(checkpoint_path).parent / "artifacts" / "llm_foundry_config.yaml")
    return yaml_config


def extract_llm_state_dict_and_config(fsdp_state_dict: Dict):
    from mappers import get_language_model_config_from_fsdp_state_dict

    llm_config = get_language_model_config_from_fsdp_state_dict(fsdp_state_dict)

    if "architectures" not in llm_config:
        llm_config["architectures"] = None
    if "_name_or_path" not in llm_config:
        llm_config["_name_or_path"] = None
    if "trust_remote_code" in llm_config:
        del llm_config["trust_remote_code"]
    if "enable_async_tp" in llm_config:
        llm_config["enable_async_tp"] = False
    if "pretrained_model_name_or_path" in llm_config:
        del llm_config["pretrained_model_name_or_path"]

    if llm_config["name"] == "giga_mix_causal_lm":
        llm_config["name"] = "deepseek_causal_lm"
    elif llm_config["name"] == "gigar_causal_lm":
        llm_config["name"] = "llama_causal_lm"

    llm_state_dict = {
        "state": {
            "model": {},
            "integrations": {"huggingface": {"model": {"config": {"content": llm_config}}}}
        }
    }
    for name, param in fsdp_state_dict["state"]["model"].items():
        if "language_model" in  name:
            llm_state_dict["state"]["model"][name.replace("language_model.", "")] = param

    return llm_config, llm_state_dict


def build_gigaomni_for_causal_lm(yaml_config: DictConfig, llm_config: Dict, llm_tmp_path: str):
    add_llmfoundry_to_path()
    from llmfoundry.models.gigaomni.configuration_gigaomni import GigaOmniConfig
    from llmfoundry.models.gigaomni.modelling_gigaomni import GigaOmniForCausalLM

    yaml_config.model.language_model.update(llm_config)

    # NOTE: We don't need to load any weights from checkpoints as model could be trained on other cluster.
    # So set pretrained to False and pretrained_model_name_or_path to None for all components.
    yaml_config.model.language_model.pretrained = False
    yaml_config.model.language_model.pretrained_model_name_or_path = llm_tmp_path
    for group in ["encoders", "adapters", "heads"]:
        for component in yaml_config.model.get(group, []):
            if "pretrained_model_name_or_path" in component:
                component.pretrained_model_name_or_path = None

    gigaomni_config = GigaOmniConfig(
        yaml_config.model.language_model,
        yaml_config.model.get("encoders", []),
        yaml_config.model.get("adapters", []),
        yaml_config.model.get("heads", []),
        yaml_config.model.get("modalities", []),
    )

    model = GigaOmniForCausalLM(gigaomni_config)
    return model


def _copy_directory_contents(source_path: Path, target_path: Path) -> None:
    target_path.mkdir(exist_ok=True, parents=True)
    for item in source_path.iterdir():
        destination = target_path / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)

def add_processors(save_path: str, yaml_config: DictConfig, tokenizer: PreTrainedTokenizerBase):
    add_llmfoundry_to_path()
    from llmfoundry.data import processors

    for processor_config in yaml_config.train_loader.dataset.processors:
        processor_name = processor_config.pop("name")
        (Path(save_path) / "processors" / processor_name).mkdir(exist_ok=True, parents=True)

        processor = getattr(processors, processor_name)(**om.to_container(processor_config, resolve=True))
        if processor.tokenizer is None:
            processor.tokenizer = tokenizer

        processor.save_pretrained(Path(save_path) / "processors" / processor_name)

    # Temporary support for old processor format
    if (Path(save_path) / "processors" / "GigaOmniVisionProcessor").exists():
        target_path = Path(save_path) / "processors" / "GigaOmniVisionProcessor"
        _copy_directory_contents(_DUMMY_PROCESSOR_CONFIG_DIR, target_path)


def _normalize_llm_name_for_gigaomni(llm_config: Dict[str, Any], yaml_config: DictConfig) -> Dict[str, Any]:
    llm_config = dict(llm_config)
    lm_name = yaml_config.model.language_model.get("name")
    if lm_name == "giga_mix_causal_lm":
        lm_name = "deepseek_causal_lm"
    elif lm_name == "gigar_causal_lm":
        lm_name = "llama_causal_lm"
    if lm_name is not None:
        llm_config["name"] = lm_name
    llm_config.setdefault("pretrained", False)
    return llm_config


def _build_root_gigaomni_config(
    yaml_config: DictConfig,
    llm_config: Dict[str, Any],
    language_model_dir: Path,
) -> Dict[str, Any]:
    """Build root GigaOmni config.json without importing llmfoundry GigaOmni classes."""
    lm_cfg = _normalize_llm_name_for_gigaomni(llm_config, yaml_config)
    lm_cfg["pretrained"] = False
    lm_cfg["return_dict"] = True
    lm_cfg["pretrained_model_name_or_path"] = str(language_model_dir.resolve())

    def _components(group: str) -> List[Dict[str, Any]]:
        group_config = yaml_config.model.get(group)
        if group_config is None:
            return []
        items = om.to_container(group_config, resolve=True)
        for item in items:
            if isinstance(item, dict) and "pretrained_model_name_or_path" in item:
                item["pretrained_model_name_or_path"] = None
        return items

    modalities = yaml_config.model.get("modalities")
    cfg_dict = {
        "model_type": "gigaomni",
        "architectures": ["GigaOmniForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_gigaomni.GigaOmniConfig",
            "AutoModelForCausalLM": "modelling_gigaomni.GigaOmniForCausalLM",
        },
        "language_model": lm_cfg,
        "encoders": _components("encoders"),
        "adapters": _components("adapters"),
        "heads": _components("heads"),
        "modalities": om.to_container(modalities, resolve=True) if modalities is not None else [],
        "return_dict": True,
    }
    return cfg_dict


def _copy_dummy_gigaomni_assets(
    save_path: Path,
    yaml_config: DictConfig,
    llm_config: Dict[str, Any],
    language_model_dir: Path,
) -> None:
    """Use a minimal GigaOmni model only to generate runtime files and modalities."""
    dummy_yaml_config = copy.deepcopy(yaml_config)
    dummy_yaml_config.model.encoders = []
    dummy_yaml_config.model.adapters = []
    dummy_yaml_config.model.heads = []
    dummy_yaml_config.model.modalities = []

    dummy_llm_config = dict(llm_config)
    dummy_llm_config.update(
        {
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 64,
            "dense_intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "n_routed_experts": 4,
            "num_routed_experts": 4,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1,
            "first_k_dense_replace": 1,
            "full_attention_layers": [0],
            "q_lora_rank": None,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 16,
            "v_head_dim": 16,
            "qk_head_dim": 32,
            "head_dim": 16,
            "vocab_size": 128,
        }
    )
    dummy_llm_config.pop("layer_types", None)

    with tempfile.TemporaryDirectory(prefix="gigaomni_") as temp_dir:
        model = build_gigaomni_for_causal_lm(
            dummy_yaml_config,
            dummy_llm_config,
            str(language_model_dir),
        )
        model.save_pretrained(temp_dir, state_dict={})
        for name in (
            "configuration_gigaomni.py",
            "modelling_gigaomni.py",
            "utils_gigaomni.py",
        ):
            shutil.copy2(Path(temp_dir) / name, save_path / name)
        shutil.copytree(
            Path(temp_dir) / "modalities",
            save_path / "modalities",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )


def _assemble_gigaomni_hf_shell(
    save_path: Path,
    *,
    args=None,
    mapper: Optional[LayerMapper] = None,
) -> None:
    """Assemble nested language_model/ + root GigaOmni shell around streamed weight shards."""
    input_checkpoint_path = getattr(args, "input_checkpoint_path", None)
    if input_checkpoint_path is None:
        raise RuntimeError(
            "Cannot assemble GigaOmni HF shell: args.input_checkpoint_path is undefined."
        )

    language_model_dir = save_path / "language_model"
    language_model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[rank 0] Assembling nested language_model at {language_model_dir}")
    # The streaming manifest belongs to the root Omni checkpoint. Preserve its
    # language-model family when assembling metadata in the nested directory.
    model_family = load_hf_assembly_manifest(str(save_path)).get("model_family")
    assemble_result = assemble_hf_checkpoint(
        str(language_model_dir),
        model_spec=MODEL_SPECS.get(model_family),
        tokenizer_path=getattr(args, "tokenizer_path", None) if args is not None else None,
        torch_dtype=getattr(args, "torch_dtype", None) if args is not None else None,
        mapper=mapper,
        cleanup_intermediate_artifacts=False,
        overwrite_config=True,
    )
    llm_config = dict(assemble_result["config"])

    yaml_config = load_yaml_config(input_checkpoint_path)
    llm_config = _normalize_llm_name_for_gigaomni(llm_config, yaml_config)

    _copy_dummy_gigaomni_assets(
        save_path,
        yaml_config,
        llm_config,
        language_model_dir,
    )

    root_cfg = _build_root_gigaomni_config(yaml_config, llm_config, language_model_dir)
    _write_config_json(save_path / "config.json", root_cfg, overwrite_config=True)

    tokenizer = AutoTokenizer.from_pretrained(
        yaml_config.tokenizer.name, **yaml_config.tokenizer.kwargs
    )
    tokenizer.save_pretrained(save_path)
    add_processors(str(save_path), yaml_config, tokenizer)

    logger.info(
        f"[rank 0] GigaOmni HF shell ready: language_model/, modalities/, processors/ under {save_path}"
    )


def _merge_hf_llm_safetensors(
    llm_dir: str,
    args=None,
    mapper: Optional[LayerMapper] = None,
) -> None:
    """Flatten rank shards and assemble the full GigaOmni HF shell.

    Keeps streamed root weight shards/index untouched and builds:
      - ``language_model/`` LLM metadata + modeling files
      - root GigaOmni ``config.json`` + runtime Python
      - ``modalities/`` and ``processors/``
    """
    flatten_move_and_reindex(llm_dir)
    if not has_hf_assembly_manifest(llm_dir):
        return

    _assemble_gigaomni_hf_shell(Path(llm_dir), args=args, mapper=mapper)
