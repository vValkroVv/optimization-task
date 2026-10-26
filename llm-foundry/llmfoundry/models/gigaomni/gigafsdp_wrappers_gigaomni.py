import torch
import functools

from functools import partial
from typing import Any, Dict, List, Tuple
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from llmfoundry.models.layers import NORM_CLASS_REGISTRY
from llmfoundry.models.layers.norm import resolve_norm_class
from llmfoundry.models.giga_mix.modelling_giga_mix import ComposerGigaMixCausalLM

from .modelling_gigaomni import GigaOmniForCausalLM


def _activation_checkpointing_auto_wrap_policy_gigar(
    fsdp_config: Dict[str, Any], model: GigaOmniForCausalLM
):
    component_checkpointed_modules = set(
        _get_components_modules_for_activation_checkpointing(model)
    )

    def _checkpoint_policy_fn(m: torch.nn.Module) -> bool:
        num_layers_to_checkpoint = fsdp_config["num_layers_to_checkpoint"]
        if num_layers_to_checkpoint is None:
            num_layers_to_checkpoint = len(model.language_model.model.layers)

        checkpointed_modules = set(
            model.language_model.model.layers[:num_layers_to_checkpoint]
        )
        if model.config.language_model.get("use_mtp", False):
            checkpointed_modules |= {model.language_model.model.mtp_block}

        checkpointed_modules = {next(layer.children()) for layer in checkpointed_modules}
        checkpointed_modules |= component_checkpointed_modules

        return m in checkpointed_modules

    return functools.partial(
        lambda_auto_wrap_policy,
        lambda_fn=_checkpoint_policy_fn,
    )


def _giga_fsdp_modules_to_wrap_with_names_gigar(
    model: GigaOmniForCausalLM, separate_layer_norm_disabled: bool
):
    modules_with_names = [
        (model.language_model.model.embed_tokens, "language_model.model.embed_tokens")
    ]

    for layer_idx, module in enumerate(model.language_model.model.layers):
        modules_with_names.append((module, f"language_model.model.layers.{layer_idx}"))

    if separate_layer_norm_disabled:
        modules_with_names.append(
            (model.language_model.model.norm, "language_model.model.norm")
        )

    modules_with_names.append((model.language_model.lm_head, "language_model.lm_head"))

    if model.config.language_model.get("use_mtp", False):
        modules_with_names.append(
            (model.language_model.model.mtp_block, "language_model.model.mtp_block")
        )

    components_modules = _get_components_modules_to_wrap_with_names(model)
    modules_with_names.extend(components_modules)

    return modules_with_names, [module for module, _ in components_modules]


def _giga_fsdp_rogue_layer_norm_modules_with_names_gigar(model: GigaOmniForCausalLM):
    """
    LayerNorm is rouge only in case it is not a part of STM (SuperTensorModule)
    STM wrapping is declared in giga_fsdp_modules_to_wrap_with_names
    """
    rouge_layer_norms = {model.language_model.model.norm: "language_model.model.norm"}
    if model.config.language_model.get("use_mtp", False):
        rouge_layer_norms |= {
            m: f"language_model.model.mtp_block.mtp_norms.{i}"
            for i, m in enumerate(model.language_model.model.mtp_block.mtp_norms)
        }
    return rouge_layer_norms


def _activation_checkpointing_auto_wrap_policy_giga_mix(
    model: GigaOmniForCausalLM, fsdp_config: Dict[str, Any]
):
    llm_policy_fn = ComposerGigaMixCausalLM._activation_checkpointing_auto_wrap_policy(model.language_model, fsdp_config)
    component_checkpointed_modules = set(
        _get_components_modules_for_activation_checkpointing(model)
    )

    def _components_policy_fn(m: torch.nn.Module) -> bool:
        is_component_act_layer = m in component_checkpointed_modules
        return is_component_act_layer or llm_policy_fn.keywords["lambda_fn"](m)

    return functools.partial(
        lambda_auto_wrap_policy,
        lambda_fn=_components_policy_fn,
    )


def _giga_fsdp_modules_to_wrap_with_names_giga_mix(model: GigaOmniForCausalLM, fsdp_config: Dict[str, Any]):
    modules_with_names = ComposerGigaMixCausalLM._giga_fsdp_modules_to_wrap_with_names(model.language_model, fsdp_config)

    updated_modules_with_names = []
    for idx in range(len(modules_with_names)):
        updated_modules_with_names.append((
            modules_with_names[idx][0],
            modules_with_names[idx][1].replace("model.", "language_model.model.").replace("lm_head", "language_model.lm_head")
        ))

    components_modules = _get_components_modules_to_wrap_with_names(model)
    updated_modules_with_names.extend(components_modules)
    return updated_modules_with_names, [module for module, _ in components_modules]


def _giga_fsdp_rogue_layer_norm_modules_with_names_giga_mix(model: GigaOmniForCausalLM):
    rouge_layer_norms_with_names = ComposerGigaMixCausalLM._giga_fsdp_rogue_layer_norm_modules_with_names(model.language_model)

    for module in rouge_layer_norms_with_names.keys():
        rouge_layer_norms_with_names[module] = rouge_layer_norms_with_names[module].replace("model.", "language_model.model.")

    return rouge_layer_norms_with_names


def _special_process_group_fn_giga_mix(
    module: torch.nn.Module, components_modules: List[torch.nn.Module]
):
    if module in components_modules:
        return (torch.distributed.distributed_c10d._get_default_group(), None)
    return ComposerGigaMixCausalLM._special_process_group_fn(module)


def _special_process_group_fn_gigar(
    module: torch.nn.Module, components_modules: List[torch.nn.Module]
):
    if module in components_modules:
        return (torch.distributed.distributed_c10d._get_default_group(), None)

    return None


def _get_components_modules_to_wrap_with_names(
    model: GigaOmniForCausalLM,
) -> List[Tuple[torch.nn.Module, str]]:
    modules_with_names = []

    components = []
    for name, component in model.encoders.items():
        components.append((f"encoders.{name}", component))

    for name, component in model.adapters.items():
        components.append((f"adapters.{name}", component))

    for name, component in model.heads.items():
        components.append((f"heads.{name}", component))

    for name, component in components:
        if hasattr(component, "get_giga_fsdp_modules_to_wrap_with_names"):
            modules_with_names.extend(
                component.get_giga_fsdp_modules_to_wrap_with_names(prefix=name)
            )
        else:
            modules_with_names.append((component, name))
    return modules_with_names


def _get_components_modules_for_activation_checkpointing(
    model: GigaOmniForCausalLM,
) -> List[torch.nn.Module]:
    modules = []
    for component in (
        list(model.encoders.values())
        + list(model.adapters.values())
        + list(model.heads.values())
    ):
        if hasattr(component, "get_giga_fsdp_modules_for_activation_checkpointing"):
            modules.extend(
                component.get_giga_fsdp_modules_for_activation_checkpointing()
            )
        else:
            modules.append(component)
    return modules


def _build_gigafsdp_wrappers_config(
    model: GigaOmniForCausalLM, lm_name: str, fsdp_config: Dict[str, Any]
) -> None:
    separate_layer_norm_disabled = fsdp_config.get("separate_layer_norm_disabled", True)
    if model.config.language_model.get("enable_async_tp", False):
        separate_layer_norm_disabled = False

    fsdp_config["num_layers_to_checkpoint"] = model.config.language_model.get(
        "activation_checkpoint_layers_num", None
    )

    assert lm_name in ["gigar_causal_lm", "giga_mix_causal_lm", "default"], (
        f"GigaFSDP wrapping is not implemented for: {lm_name}"
    )

    if lm_name == "gigar_causal_lm" or lm_name == "default":
        modules_to_wrap_with_names, components_modules = (
            _giga_fsdp_modules_to_wrap_with_names_gigar(
                model, separate_layer_norm_disabled
            )
        )
        fsdp_config["gigafsdp_wrappers"]["model"] = {
            "activation_checkpointing_auto_wrap_policy": _activation_checkpointing_auto_wrap_policy_gigar(
                fsdp_config, model
            ),
            "giga_fsdp_modules_to_wrap_with_names": modules_to_wrap_with_names,
            "giga_fsdp_rogue_layer_norm_modules_with_names": None
            if separate_layer_norm_disabled
            else _giga_fsdp_rogue_layer_norm_modules_with_names_gigar(model),
            "giga_fsdp_layer_norm_module_cls": None
            if separate_layer_norm_disabled
            else NORM_CLASS_REGISTRY[model.config.language_model["norm_type"]],
            "special_process_group_fn": partial(
                _special_process_group_fn_gigar, components_modules=components_modules
            ),
        }
    elif lm_name == "giga_mix_causal_lm":
        modules_to_wrap_with_names, components_modules = (
            _giga_fsdp_modules_to_wrap_with_names_giga_mix(model, fsdp_config)
        )
        fsdp_config["gigafsdp_wrappers"]["model"] = {
            "activation_checkpointing_auto_wrap_policy": _activation_checkpointing_auto_wrap_policy_giga_mix(
                model, fsdp_config
            ),
            "giga_fsdp_modules_to_wrap_with_names": modules_to_wrap_with_names,
            "giga_fsdp_rogue_layer_norm_modules_with_names": _giga_fsdp_rogue_layer_norm_modules_with_names_giga_mix(
                model
            ),
            "giga_fsdp_layer_norm_module_cls": resolve_norm_class(model.config.language_model["norm_type"]),
            "special_process_group_fn": partial(
                _special_process_group_fn_giga_mix,
                components_modules=components_modules,
            ),
        }
