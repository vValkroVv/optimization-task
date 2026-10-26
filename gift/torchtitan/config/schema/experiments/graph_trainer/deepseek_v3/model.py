from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.deepseek_v3.model import DeepSeekV3ModelConfig


@config(type="graph_trainer/deepseek_v3")
class GraphTrainerDeepSeekV3ModelConfig(DeepSeekV3ModelConfig):
    """deepseek_v3 with the graph parallelize function; pairs with
    ``trainer: {type: graph_trainer}``."""

    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.graph_trainer.deepseek_v3.model:GraphTrainerDeepSeekV3Model"
    )
    family: ClassVar[str] = "graph_trainer/deepseek_v3"
    trainer_type: ClassVar[str] = "graph_trainer"
    _runtime_spec: ClassVar[str] = (
        "torchtitan.experiments.graph_trainer.deepseek_v3.versions:BASE"
    )
