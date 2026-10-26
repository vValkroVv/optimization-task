"""Freeze existing modules selected by their runtime parameter namespace."""

from fnmatch import fnmatchcase
from typing import ClassVar, TypeAlias

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.schema.protocols.model import ModelConfigConverterConfig
from torchtitan.config.schema.protocols.module import ModuleConfig

ModuleLocation: TypeAlias = tuple[str, ModuleConfig, object, str | int]


def resolve_freeze_selectors(
    model_config: Configurable.Config, selectors: list[str]
) -> list[ModuleLocation]:
    """Resolve every selector before changing the tree, preserving atomic failure."""

    def collect(config: Configurable.Config, prefix: str) -> list[ModuleLocation]:
        result: list[ModuleLocation] = []
        # traverse stops at a match, so descend each module separately.
        for location in config.traverse(ModuleConfig, _prefix=prefix):
            result.append(location)
            result.extend(collect(location[1], location[0]))
        return result

    candidates = collect(model_config, "")
    selected: list[ModuleLocation] = []
    for selector in selectors:
        matches = [item for item in candidates if fnmatchcase(item[0], selector)]
        if not matches:
            raise ValueError(f"freeze selector {selector!r} matches no module FQN")
        selected.extend(matches)
    return sorted(selected, key=lambda item: item[0].count("."))


@config(type="freeze")
class FreezeConverterConfig(ModelConfigConverterConfig):
    _owner_spec: ClassVar[str] = "torchtitan.components.freeze:FreezeConverter"

    selectors: list[str]
    """Module FQN globs, for example ``layers.0`` or ``layers.*.attention``."""

    def __post_init__(self) -> None:
        if not self.selectors or any(not value.strip() for value in self.selectors):
            raise ValueError("freeze.selectors must contain nonempty module FQN globs")

    def validate_pipeline(self, model_config, previous, trainer_config=None) -> None:
        resolve_freeze_selectors(model_config, self.selectors)
