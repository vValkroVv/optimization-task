"""qwen3 Config for the graph trainer; pairs with ``trainer: {type: graph_trainer}``."""

from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.qwen3.model import Qwen3ModelConfig


@config(type="graph_trainer/qwen3")
class GraphTrainerQwen3ModelConfig(Qwen3ModelConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.graph_trainer.qwen3.model:GraphTrainerQwen3Model"
    )
    family: ClassVar[str] = "graph_trainer/qwen3"
    trainer_type: ClassVar[str] = "graph_trainer"
    _runtime_spec: ClassVar[str] = (
        "torchtitan.experiments.graph_trainer.qwen3.versions:BASE"
    )
