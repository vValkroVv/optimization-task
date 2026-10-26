from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.experiments.ft.checkpoint import FTCheckpointManagerConfig
from torchtitan.config.schema.experiments.ft.config.job_config import FaultTolerance
from torchtitan.config.schema.experiments.ft.optimizer import (
    FTOptimizersContainerConfig,
)
from torchtitan.config.schema.trainer import TrainerConfig


@config
class FaultTolerantTrainerConfig(TrainerConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.ft.trainer:FaultTolerantTrainer"
    )

    fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)

    # pyrefly: ignore [bad-override]
    optimizer: FTOptimizersContainerConfig = field(
        default_factory=FTOptimizersContainerConfig
    )
    # pyrefly: ignore [bad-override]
    checkpoint: FTCheckpointManagerConfig = field(
        default_factory=FTCheckpointManagerConfig
    )
