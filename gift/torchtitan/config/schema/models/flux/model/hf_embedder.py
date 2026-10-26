from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.value_types import ConfigValue
from torchtitan.config.schema.protocols.module import ModuleConfig


@config
class FluxEmbedderConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.flux.model.hf_embedder:FluxEmbedder"

    version: str
    random_init: bool = False
    hf_kwargs: dict[str, ConfigValue] = field(default_factory=dict)
