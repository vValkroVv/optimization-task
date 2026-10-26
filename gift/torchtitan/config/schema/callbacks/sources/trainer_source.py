from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.sources.base import MonitorSourceConfig


@config(type="trainer")
class TrainerSourceConfig(MonitorSourceConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.sources.trainer_source:TrainerSource"
    )

    # Empty list disables token accuracy. The vocab top-k runs only on
    # logging steps but is still a full-vocab topk per token.
    topk_accuracy: list[int] = field(default_factory=list)
    logits_module_pattern: str = "lm_head$"
    log_lr: bool = True
    include_parameter_lr_tags: bool = False
    log_throughput: bool = False
    throughput_window_size: int = 100
    throughput_log_interval: int = 1
    log_remaining_estimate: bool = False
    remaining_estimate_skip_steps: int = 1

    def __post_init__(self) -> None:
        if any(k <= 0 for k in self.topk_accuracy):
            raise ValueError("TrainerSource topk_accuracy entries must be positive")
        if self.throughput_window_size <= 0:
            raise ValueError("TrainerSource throughput_window_size must be positive")
        if self.throughput_log_interval <= 0:
            raise ValueError("TrainerSource throughput_log_interval must be positive")
        if self.remaining_estimate_skip_steps < 0:
            raise ValueError(
                "TrainerSource remaining_estimate_skip_steps must be non-negative"
            )
