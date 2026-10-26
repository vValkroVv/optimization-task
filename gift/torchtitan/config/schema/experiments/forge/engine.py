from dataclasses import field
from typing import ClassVar, TYPE_CHECKING

from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.config.configurable import config, Configurable
from torchtitan.config.schema.components.checkpoint import CheckpointManagerConfig
from torchtitan.config.schema.components.loss import (
    BaseLossConfig,
    CrossEntropyLossConfig,
)
from torchtitan.config.schema.components.lr_scheduler import LRSchedulersContainerConfig
from torchtitan.config.schema.components.optimizer import OptimizersContainerConfig

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec


@config
class ForgeEngineConfig(Configurable.Config):
    _owner_spec: ClassVar[str] = "torchtitan.experiments.forge.engine:ForgeEngine"

    hf_assets_path: str = "./tests/assets/tokenizer"
    dump_folder: str = "./outputs"
    model_spec: "ModelSpec | None" = field(init=False, default=None)
    """Set programmatically (``init=False``): never a YAML key."""
    optimizer: OptimizersContainerConfig = field(
        default_factory=OptimizersContainerConfig
    )
    lr_scheduler: LRSchedulersContainerConfig = field(
        default_factory=LRSchedulersContainerConfig
    )
    loss: BaseLossConfig = field(default_factory=CrossEntropyLossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    checkpoint: CheckpointManagerConfig = field(default_factory=CheckpointManagerConfig)
    activation_checkpoint: ActivationCheckpointConfig = field(
        default_factory=ActivationCheckpointConfig
    )
    compile: CompileConfig = field(default_factory=CompileConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
