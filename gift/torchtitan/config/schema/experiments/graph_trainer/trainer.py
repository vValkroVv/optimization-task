from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
)
from torchtitan.config.schema.trainer import TrainerConfig


@config
class GraphTrainerConfig(TrainerConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.graph_trainer.trainer:GraphTrainer"
    )

    # pyrefly: ignore [bad-override]
    compile: GraphTrainerCompileConfig = field(
        default_factory=GraphTrainerCompileConfig
    )
