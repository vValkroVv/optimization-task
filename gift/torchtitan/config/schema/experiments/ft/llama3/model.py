from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.llama3.model import Llama3ModelConfig


@config(type="ft/llama3")
class FaultTolerantLlama3ModelConfig(Llama3ModelConfig):
    """llama3 in a ``FaultTolerantModelSpec`` with the DiLoCo ``fragment_fn``;
    pairs with ``trainer: {type: ft}``."""

    _owner_spec: ClassVar[str] = "torchtitan.models.llama3.model:Llama3Model"
    family: ClassVar[str] = "ft/llama3"
    trainer_type: ClassVar[str] = "ft"
    _runtime_spec: ClassVar[str] = "torchtitan.experiments.ft.llama3.versions:BASE"
