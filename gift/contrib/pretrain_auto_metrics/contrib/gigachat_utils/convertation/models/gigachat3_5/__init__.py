from .configuration_gigachat3_5 import Gigachat35Config

__all__ = [
    "Gigachat35Config",
    "Gigachat35ForCausalLM",
]


def __getattr__(name):
    if name == "Gigachat35ForCausalLM":
        from .modeling_gigachat3_5 import Gigachat35ForCausalLM

        return Gigachat35ForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
