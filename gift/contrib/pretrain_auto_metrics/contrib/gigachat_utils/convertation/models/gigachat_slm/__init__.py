from .configuration_gigachat_slm import GigachatSLMConfig

__all__ = [
    "GigachatSLMConfig",
    "GigachatSLMForCausalLM",
]


def __getattr__(name):
    if name == "GigachatSLMForCausalLM":
        from .modeling_gigachat_slm import GigachatSLMForCausalLM

        return GigachatSLMForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
