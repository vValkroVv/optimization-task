from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.llama3.model import Llama3ModelConfig


@config(type="graph_trainer/llama3")
class GraphTrainerLlama3ModelConfig(Llama3ModelConfig):
    """llama3 with the graph parallelize function; pairs with
    ``trainer: {type: graph_trainer}``."""

    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.graph_trainer.llama3.model:GraphTrainerLlama3Model"
    )
    family: ClassVar[str] = "graph_trainer/llama3"
    trainer_type: ClassVar[str] = "graph_trainer"
    _runtime_spec: ClassVar[str] = (
        "torchtitan.experiments.graph_trainer.llama3.versions:BASE"
    )
