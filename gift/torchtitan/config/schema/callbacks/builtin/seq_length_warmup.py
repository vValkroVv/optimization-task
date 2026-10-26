from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="seq_length_warmup")
class SeqLengthWarmupConfig(CallbackConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.builtin.seq_length_warmup:SeqLengthWarmup"
    )

    min_seq_len: int = 256
    max_seq_len: int | None = None
    warmup_steps: int = 1000
    emit_metrics: bool = False
    log_interval: int = 1

    def __post_init__(self) -> None:
        if self.min_seq_len <= 0:
            raise ValueError("SeqLengthWarmup min_seq_len must be positive")
        if self.max_seq_len is not None and self.max_seq_len < self.min_seq_len:
            raise ValueError("SeqLengthWarmup max_seq_len must be >= min_seq_len")
        if self.warmup_steps < 0:
            raise ValueError("SeqLengthWarmup warmup_steps must be non-negative")
        if self.log_interval <= 0:
            raise ValueError("SeqLengthWarmup log_interval must be positive")
