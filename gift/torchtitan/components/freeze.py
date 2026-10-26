"""Freeze selected modules without introducing runtime wrappers or new FQNs."""

from typing import TypeAlias

from torchtitan.config.configurable import Configurable
from torchtitan.config.schema.components.freeze import (
    FreezeConverterConfig,
    resolve_freeze_selectors,
)
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.protocols.model import ModelConfigConverter


class FreezeConverter(ModelConfigConverter):
    # Config aliases specialize the schema attached to a runtime owner.
    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FreezeConverterConfig

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config: Configurable.Config) -> None:
        selected = resolve_freeze_selectors(model_config, self.config.selectors)
        # Wrapping a parent freezes every descendant; avoid redundant wrappers.
        covered: list[str] = []
        for fqn, child, parent, key in selected:
            if any(fqn == path or fqn.startswith(path + ".") for path in covered):
                continue
            if isinstance(parent, FrozenConfig) and key == "inner":
                covered.append(fqn)
                continue
            wrapper = FrozenConfig(inner=child)
            if isinstance(parent, list) and isinstance(key, int):
                parent[key] = wrapper
            elif isinstance(key, str):
                setattr(parent, key, wrapper)
            else:
                raise ValueError(f"unsupported config parent for module {fqn!r}")
            covered.append(fqn)
