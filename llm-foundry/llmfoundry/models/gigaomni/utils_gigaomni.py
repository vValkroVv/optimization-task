import sys
import copy
import dill
import logging
import importlib
import torch
import torch.nn as nn

from accelerate import load_checkpoint_in_model
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Union
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s |  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def import_if_available(
    module_name: str, class_name: Optional[str] = None, package: Optional[str] = None
):
    try:
        module = importlib.import_module(module_name, package=package)
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError):
        return None


# TODO: rename this function and do it better (if possible)
def import_module_if_available_from_local_path(
    module_name: str, class_name: Optional[str] = None
):
    """
    Function hides imports from automatic processing of modeling files by transformers and also
    tries to do imports from local files (used for huggingface checkpoint).
    """
    module = import_if_available(module_name, class_name)
    if module:
        return module

    module_name = module_name.replace("llmfoundry.models.gigaomni", "").strip(".")

    pythonpath_addition = str(Path(__file__).absolute().parent)
    sys.path.append(pythonpath_addition)
    module = import_if_available(module_name, class_name)
    sys.path.remove(pythonpath_addition)

    return module


modalities = import_module_if_available_from_local_path(
    "llmfoundry.models.gigaomni.modalities"
)
BaseModalityConfig = import_module_if_available_from_local_path(
    "llmfoundry.models.gigaomni.modalities", "BaseModalityConfig"
)
get_sp_split_type = import_if_available(
    "llmfoundry.models.utils.configuration_utils", "get_sp_split_type"
)
get_tp_sp_group_size = import_if_available(
    "composer.utils.dist", "get_tp_sp_group_size"
) or (lambda: None)
seed_context = import_if_available(
    "composer.utils.reproducibility", "seed_context"
) or (lambda _: nullcontext())


GIGAOMNI_DEFAULT_LANGUAGE_MODEL_NAME = "default"


GIGAOMNI_CONFIG_REGISTRY = {
    "default": AutoConfig,
    "llama_causal_lm": AutoConfig,
    "deepseek_causal_lm": AutoConfig,
    "gigar_causal_lm": import_if_available(
        "llmfoundry.models.gigar.configuration_gigar", "GigarConfig"
    ),
    "giga_mix_causal_lm": import_if_available(
        "llmfoundry.models.giga_mix.configuration_giga_mix", "GigaMixConfig"
    ),
}


GIGAOMNI_MODEL_REGISTRY = {
    "default": AutoModelForCausalLM,
    "llama_causal_lm": AutoModelForCausalLM,
    "deepseek_causal_lm": AutoModelForCausalLM,
    "gigar_causal_lm": import_if_available(
        "llmfoundry.models.gigar.modelling_gigar", "LlamaForCausalLM"
    ),
    "giga_mix_causal_lm": import_if_available(
        "llmfoundry.models.giga_mix.modelling_giga_mix", "GigaMixForCausalLM"
    ),
}


GIGAOMNI_AUTO_CLASS_REQUIRED = {
    name
    for name, cls in GIGAOMNI_MODEL_REGISTRY.items()
    if hasattr(cls, "__name__") and cls.__name__ == "AutoModelForCausalLM"
}


def build_language_model_config(
    lm_config: Optional[Dict[str, Any]], lm_name: Optional[str] = None
) -> PretrainedConfig:
    if lm_config is None:
        return GIGAOMNI_CONFIG_REGISTRY[GIGAOMNI_DEFAULT_LANGUAGE_MODEL_NAME]()

    assert lm_name in GIGAOMNI_CONFIG_REGISTRY, (
        f"Unknown language model model type: {lm_name}"
    )

    LLMConfigCls = GIGAOMNI_CONFIG_REGISTRY[lm_name]
    if lm_config.get("pretrained_model_name_or_path", None):
        return LLMConfigCls.from_pretrained(
            lm_config.pop("pretrained_model_name_or_path"),
            trust_remote_code=lm_config.pop("trust_remote_code", True),
            **lm_config,
        )

    config = LLMConfigCls(**lm_config)
    if hasattr(config, "sp_split_type"):
        config.sp_split_type = get_sp_split_type(  # type: ignore
            split_type=lm_config.get("sp_split_type"),
            attention_type=lm_config.get("attention_type"),
            varlen_input=lm_config.get("varlen_input", False),
        )

    return config


def build_language_model(lm_dict_config: Optional[Dict[str, Any]]) -> PreTrainedModel:
    if lm_dict_config is None:
        return GIGAOMNI_MODEL_REGISTRY[GIGAOMNI_DEFAULT_LANGUAGE_MODEL_NAME]()

    lm_name = lm_dict_config["name"]
    pretrained = lm_dict_config["pretrained"]
    pretrained_model_name_or_path = lm_dict_config["pretrained_model_name_or_path"]

    lm_config = build_language_model_config(copy.deepcopy(lm_dict_config), lm_name)
    lm_dict_config.update(lm_config.to_diff_dict())

    if pretrained_model_name_or_path is not None and pretrained:
        return GIGAOMNI_MODEL_REGISTRY[lm_name].from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            config=lm_config,
        )

    if lm_name in GIGAOMNI_AUTO_CLASS_REQUIRED:
        return AutoModelForCausalLM.from_config(lm_config, trust_remote_code=True)

    return GIGAOMNI_MODEL_REGISTRY[lm_name](lm_config)


def build_modality_config(modality_config: Union[Dict[str, Any]]) -> BaseModalityConfig:
    modality_config = copy.deepcopy(modality_config)
    if isinstance(modality_config, BaseModalityConfig):
        return modality_config

    assert "name" in modality_config, "`name` is required in the modality_config"
    return getattr(modalities, modality_config.pop("name")).config_class(
        **modality_config
    )


def build_modality_component(
    component_dict_config: Dict[str, Any], module: ModuleType
) -> Dict[str, Union[nn.Module, PreTrainedModel]]:
    component_dict_config_for_init = copy.deepcopy(component_dict_config)

    name = component_dict_config_for_init.pop("name")
    alias = component_dict_config_for_init.pop("alias", name)
    init_device = component_dict_config_for_init.pop("init_device", "cpu")
    init_seed = component_dict_config_for_init.pop("init_seed", 42)
    pretrained_model_name_or_path = component_dict_config_for_init.pop(
        "pretrained_model_name_or_path", None
    )

    ComponentCls = getattr(module, name)

    if not hasattr(ComponentCls, "config_class"):
        with seed_context(init_seed):
            component: nn.Module = ComponentCls(**component_dict_config_for_init)

        if pretrained_model_name_or_path is not None:
            checkpoint = torch.load(
                pretrained_model_name_or_path,
                map_location=init_device,
                pickle_module=dill,
            )

            missing_keys, unexpected_keys = component.load_state_dict(
                checkpoint,
                strict=False,
                assign=bool(init_device == "meta"),
            )

            if missing_keys or unexpected_keys:
                logger.warning(
                    f"Found missing keys: {missing_keys} and unexpected keys: {unexpected_keys} "
                    f"when loading {name} from {pretrained_model_name_or_path}!"
                )

        return {alias: component}

    if pretrained_model_name_or_path is not None:
        config: PretrainedConfig = ComponentCls.config_class.from_pretrained(
            pretrained_model_name_or_path,
            **component_dict_config_for_init,
        )

        component_dict_config.update(config.to_diff_dict())

        if init_device == "meta":
            component = ComponentCls(config).to_empty(device="cpu", recurse=True)
            load_checkpoint_in_model(
                component, pretrained_model_name_or_path, strict=True
            )
        else:
            component = ComponentCls.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
            )

        return {alias: component}

    config = ComponentCls.config_class(**component_dict_config_for_init)
    component_dict_config.update(config.to_diff_dict())
    with seed_context(init_seed):
        return {alias: ComponentCls(config)}
