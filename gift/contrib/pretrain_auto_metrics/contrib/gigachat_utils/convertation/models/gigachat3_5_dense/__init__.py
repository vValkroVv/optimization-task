from .configuration_gigachat3_5_dense import Gigachat35DenseConfig

__all__ = [
    "Gigachat35DenseConfig",
    "Gigachat35DenseForCausalLM",
]


def __getattr__(name):
    if name == "Gigachat35DenseForCausalLM":
        from .modeling_gigachat3_5_dense import Gigachat35DenseForCausalLM

        return Gigachat35DenseForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
