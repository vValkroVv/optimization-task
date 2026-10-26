from llmfoundry.models.gigar import ComposerGigarCausalLM, LlamaForCausalLM, GigarConfig
from llmfoundry.models.gigaclf import ComposerGigarClassifier, LlamaForClassification
from llmfoundry.models.giga_mix import (
    ComposerGigaMixCausalLM,
    GigaMixForCausalLM,
    GigaMixConfig,
)
from llmfoundry.models.gigaomni import ComposerGigaOmniForCausalLM, GigaOmniForCausalLM, GigaOmniConfig


def get_composer_model_class(model_type):
    HF_TO_COMPOSER_REGISTRY = {
        GigaMixForCausalLM: ComposerGigaMixCausalLM,
        LlamaForCausalLM: ComposerGigarCausalLM,
        LlamaForClassification: ComposerGigarClassifier,
        GigaOmniForCausalLM: ComposerGigaOmniForCausalLM,
    }
    if model_type not in HF_TO_COMPOSER_REGISTRY:
        raise ValueError(f"Not sure how to build model with name={model_type}")

    return HF_TO_COMPOSER_REGISTRY[model_type]


def get_inner_model(model_name: str):
    MODEL_REGISTRY = {
        "gigar_causal_lm": LlamaForCausalLM,
        "giga_mix_causal_lm": GigaMixForCausalLM,
        "gigaomni_causal_lm": GigaOmniForCausalLM,
        "gigar_classifier": LlamaForClassification,
    }
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Not sure how to build model with name={model_name}")
    return MODEL_REGISTRY[model_name]


def get_model_config(config_name: str):
    CONFIG_REGISTRY = {
        "gigar_causal_lm": GigarConfig,
        "giga_mix_causal_lm": GigaMixConfig,
        "gigaomni_causal_lm": GigaOmniConfig,
        "gigar_classifier": GigarConfig,
    }

    if config_name is None:
        raise ValueError("Please specify the model name in the yaml config")
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"Not sure how to build model with name={config_name}")
    return CONFIG_REGISTRY[config_name]
