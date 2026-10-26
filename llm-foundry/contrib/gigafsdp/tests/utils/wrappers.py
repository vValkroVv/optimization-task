import enum
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import torch


class WrapStrategy(enum.Enum):
    CLASSIC = "classic"
    NO_SEPARATE_LN = "no_separate_ln"
    MULTIMODAL = "multimodal"
    MULTIMODAL_NO_SEPARATE_LN = "multimodal_no_separate_ln"

def wrap_model(model, wrap_strategy):
    return WRAP_STRATEGY_FN[wrap_strategy](model)

def _wrap_model_base(model, 
                     modules_to_wrap_with_names_fn, 
                     rogue_layer_norm_modules_with_names_fn,
                     rmsnorm_cls=None):
    wrappers = dict()

    wrappers['giga_fsdp_modules_to_wrap_with_names'] = modules_to_wrap_with_names_fn(model)
    if rogue_layer_norm_modules_with_names_fn is not None:
        wrappers['giga_fsdp_rogue_layer_norm_modules_with_names'] = rogue_layer_norm_modules_with_names_fn(model)
    else:
        wrappers['giga_fsdp_rogue_layer_norm_modules_with_names'] = None
    wrappers['giga_fsdp_layer_norm_module_cls'] = rmsnorm_cls
    return wrappers
     
def _wrap_model_classic(model):
    giga_fsdp_modules_to_wrap_with_names=lambda model: (
        [(model.model.embed_tokens, "model.embed_tokens")]
        + [
            (
                m,
                f"model.layers.{i}",
            )
            for i, m in enumerate(model.model.layers)
        ]
        + [(model.lm_head, "lm_head")]
    )
    
    giga_fsdp_rogue_layer_norm_modules_with_names=lambda model: {model.model.norm: "model.norm"}

    return _wrap_model_base(model, 
                            giga_fsdp_modules_to_wrap_with_names, 
                            giga_fsdp_rogue_layer_norm_modules_with_names,
                            LlamaRMSNorm)

def _wrap_model_no_separate_ln(model):
    giga_fsdp_modules_to_wrap_with_names=lambda model: (
        [(model.model.embed_tokens, "model.embed_tokens")]
        + [
            (
                m,
                f"model.layers.{i}",
            )
            for i, m in enumerate(model.model.layers)
        ]
        + [(model.model.norm, "model.norm")]
        + [(model.lm_head, "lm_head")]
    )
    
    giga_fsdp_rogue_layer_norm_modules_with_names=None

    return _wrap_model_base(model, 
                            giga_fsdp_modules_to_wrap_with_names, 
                            giga_fsdp_rogue_layer_norm_modules_with_names,
                            None)

def _wrap_multimodal_model(model, giga_fsdp_rogue_layer_norm_modules_with_names = None, rmsnorm_cls=None):
    named_children = [name for name, _ in model.named_children() if name not in ['llm', 'lm_head']]

    giga_fsdp_modules_to_wrap_with_names=lambda model: (
        [(model.embed_tokens, "embed_tokens")] 
        + [(module, f'{name}.{module_name}') 
           for name in named_children 
           for module_name, module in getattr(model, name).named_modules() 
           if isinstance(module, torch.nn.Linear)]
        + [(m, f'llm.{i}') for i, m in enumerate(model.llm)]
        + [(model.lm_head, "lm_head")]
    )

    return _wrap_model_base(model, 
                            giga_fsdp_modules_to_wrap_with_names, 
                            giga_fsdp_rogue_layer_norm_modules_with_names,
                            rmsnorm_cls=rmsnorm_cls)

WRAP_STRATEGY_FN = {
    WrapStrategy.CLASSIC : _wrap_model_classic,
    WrapStrategy.NO_SEPARATE_LN : _wrap_model_no_separate_ln,
    WrapStrategy.MULTIMODAL_NO_SEPARATE_LN: _wrap_multimodal_model,
    WrapStrategy.MULTIMODAL: lambda model: _wrap_multimodal_model(model, rmsnorm_cls=LlamaRMSNorm)
}
