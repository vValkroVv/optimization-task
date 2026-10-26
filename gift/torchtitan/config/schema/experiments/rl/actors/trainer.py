from dataclasses import field
from typing import ClassVar

from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.config.configurable import config, Configurable
from torchtitan.config.schema.components.checkpoint import CheckpointManagerConfig
from torchtitan.config.schema.components.lr_scheduler import LRSchedulersContainerConfig
from torchtitan.config.schema.components.optimizer import OptimizersContainerConfig
from torchtitan.config.schema.experiments.rl.loss import RLLossConfig


@config
class PolicyTrainerConfig(Configurable.Config):
    """PolicyTrainer configuration for optimizer, training, and parallelism."""

    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.rl.actors.trainer:PolicyTrainer"
    )

    optimizer: OptimizersContainerConfig = field(
        default_factory=OptimizersContainerConfig
    )
    lr_scheduler: LRSchedulersContainerConfig = field(
        default_factory=LRSchedulersContainerConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    loss: RLLossConfig
    ac_config: ActivationCheckpointConfig = field(
        default_factory=lambda: ActivationCheckpointConfig(mode="none")
    )
    checkpoint: CheckpointManagerConfig = field(default_factory=CheckpointManagerConfig)
    dump_folder: str = ""
    """Folder for AC debug dumps when using memory_budget mode."""
