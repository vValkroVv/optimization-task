from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config, Configurable


@config
class MetricsProcessorConfig(Configurable.Config):
    """Metrics processor configuration."""

    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.rl.observability.metrics.processor:MetricsProcessor"
    )

    console_log_keys_train: list[str] = field(
        default_factory=lambda: [
            "loss/mean",
            "loss/ratio/clipped_frac",
            "reward/_mean",
            "reward/_max",
            "reward/zero_std_frac",
            "rollout/response_length/max",
            "train/grad_norm/mean",
            "train/lr",
            "perf/tokens_per_second",
            "timing/step",
        ]
    )
    """Regex search patterns selecting console keys for train log lines.
    [".*"] prints every key; [] prints nothing. Backends always
    receive every reduced metric regardless of this filter."""

    console_log_keys_validation: list[str] = field(
        default_factory=lambda: [
            "validation/reward/_mean",
            "validation/reward/_max",
            "validation/response_length/mean",
            "timing/validate",
        ]
    )
    """Same as console_log_keys_train but used when
    MetricsProcessor.log is called with is_validation=True."""

    enable_wandb: bool = False

    enable_tensorboard: bool = False
    """Log metrics to TensorBoard. Writes under log_dir."""

    wandb_project: str = "titan_rl"
    """W&B project name. Written to WANDB_PROJECT env var via
    setdefault, so a user-set env var takes precedence."""
