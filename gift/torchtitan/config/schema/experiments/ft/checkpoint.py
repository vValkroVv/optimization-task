from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.components.checkpoint import CheckpointManagerConfig


# Its own family, not a member of the core `checkpoint` family: only the ft
# trainer takes it, and the ft trainer takes nothing else.
@config(family="ft checkpoint", default="ft", type="ft")
class FTCheckpointManagerConfig(CheckpointManagerConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.ft.checkpoint:FTCheckpointManager"
    )

    enable_ft_dataloader_checkpoints: bool = True
    """Checkpoint the dataloader index for torchft. When disabled, fault
    tolerant replicas can train over the same data multiple times."""
