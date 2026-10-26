"""Every explicitly declared runtime binding resolves to a compatible callable."""

import inspect
import sys
import types
from typing import get_args, get_origin, Union

import pytest

from torchtitan.config.configurable import Configurable, import_spec
from torchtitan.config.registry import import_schema
from torchtitan.config.schema.protocols.module import ModuleConfig

_OPTIONAL_DEPENDENCIES = {"torchdata", "streaming", "torchft", "vllm", "ray"}
_INTERFACE_BINDINGS = {
    # Runtime dataset/rank adapter; concrete YAML loaders own config construction.
    "torchtitan.components.dataloader:ParallelAwareDataloader",
    "torchtitan.protocols.model:ModelConfigConverter",
    "torchtitan.models.common.decoder_blocks.transformer_block:TransformerBlock",
}


def _bindings():
    import_schema()
    bindings = {}
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("torchtitan.config.schema."):
            continue
        for cls in vars(module).values():
            if (
                isinstance(cls, type)
                and cls.__module__ == name
                and issubclass(cls, Configurable.Config)
                and cls.__dict__.get("_owner_spec")
            ):
                bindings[f"{name}.{cls.__qualname__}"] = cls
    return sorted(bindings.items())


@pytest.mark.parametrize(
    "name,schema",
    _bindings(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_runtime_binding_imports_and_accepts_declared_schema(name, schema):
    loaded_modules = set(sys.modules)
    try:
        owner = import_spec(schema._owner_spec)
    except ModuleNotFoundError as error:
        if error.name and error.name.partition(".")[0] in _OPTIONAL_DEPENDENCIES:
            # A failed package import can leave child modules cached without
            # their parent. Restore only newly imported repository modules.
            for module in set(sys.modules) - loaded_modules:
                if module.startswith("torchtitan."):
                    del sys.modules[module]
            pytest.skip(f"Optional runtime dependency is unavailable: {error.name}")
        raise
    assert callable(owner), name
    declared = getattr(owner, "Config", None)
    # Abstract family bases are importable bindings, but cannot be constructed.
    # Schemas with their own build adapter need not pass config to the owner.
    if (
        schema._owner_spec in _INTERFACE_BINDINGS
        or inspect.isabstract(owner)
        or schema.build not in (Configurable.Config.build, ModuleConfig.build)
    ):
        if declared is not None:
            assert issubclass(schema, declared), name
        return
    signature = inspect.signature(owner)
    parameter = signature.parameters.get("config")
    assert parameter is not None, f"{name}: runtime callable must accept config"
    assert parameter.kind is not inspect.Parameter.POSITIONAL_ONLY, name
    annotation = parameter.annotation
    if annotation == "Config":
        annotation = declared
    elif isinstance(annotation, str):
        namespace = vars(sys.modules[owner.__module__])
        annotation = eval(
            annotation, namespace, vars(owner) if isinstance(owner, type) else {}
        )
    assert annotation is not inspect.Parameter.empty, f"{name}: config must be typed"
    candidates = (
        get_args(annotation)
        if get_origin(annotation) in (types.UnionType, Union)
        else (annotation,)
    )
    assert any(
        isinstance(candidate, type) and issubclass(schema, candidate)
        for candidate in candidates
    ), name


def test_missing_repository_runtime_is_not_an_optional_dependency():
    class MissingRuntime(Configurable.Config):
        _owner_spec = "torchtitan.nonexistent_runtime:Constructor"

    with pytest.raises(ModuleNotFoundError, match="torchtitan.nonexistent_runtime"):
        test_runtime_binding_imports_and_accepts_declared_schema(
            "missing", MissingRuntime
        )


def test_binding_rejects_a_constructor_for_an_unrelated_schema(monkeypatch):
    class Incoming(Configurable.Config):
        _owner_spec = "incorrect_binding"

    class Other(Configurable.Config):
        pass

    def constructor(*, config: Other):
        raise AssertionError("The inventory must never execute constructors")

    monkeypatch.setattr(sys.modules[__name__], "import_spec", lambda _: constructor)
    with pytest.raises(AssertionError, match="incorrect"):
        test_runtime_binding_imports_and_accepts_declared_schema("incorrect", Incoming)
