from dataclasses import dataclass, field
from typing import ClassVar

from torchtitan.config.configurable import config, Configurable


@dataclass(kw_only=True, slots=True)
class BatchConfig:
    """Batch shape parameters for the RL batcher.

    TODO: Refactor the pre-training trainer to use an owned batch config
    instead of keeping batch shape fields directly on TrainingConfig.
    """

    local_batch_size: int = 8
    """Per-DP-rank batch size (rows per forward pass)."""

    global_batch_size: int = -1
    """Target number of rows per optimizer step. Defaults to
    ``local_batch_size * data-parallel degree`` when set to -1."""

    seq_len: int = 2048
    """Tokens per row (packed sequence length)."""


@config
class BatcherConfig(Configurable.Config):
    _owner_spec: ClassVar[str] = "torchtitan.experiments.rl.batcher:Batcher"

    batch: BatchConfig = field(default_factory=BatchConfig)
