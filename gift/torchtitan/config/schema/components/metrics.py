from typing import ClassVar

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.errors import ConfigFieldError

# Default size limit (MiB) past which the TensorBoard event file is rotated.
TB_ROTATE_MIB_DEFAULT = 64


@config
class MetricsProcessorConfig(Configurable.Config):
    _owner_spec: ClassVar[str] = "torchtitan.components.metrics:MetricsProcessor"

    def __post_init__(self) -> None:
        if type(self.log_freq) is not int or self.log_freq <= 0:
            raise ConfigFieldError("log_freq", "must be a positive integer")

    log_freq: int = 10
    """In iterations."""

    enable_tensorboard: bool = False

    disable_color_printing: bool = False

    save_tb_folder: str = "tb"

    tb_rotate_mib: int = TB_ROTATE_MIB_DEFAULT
    """Start a new event file once the current one exceeds this many MiB."""

    tb_folder_per_launch: bool = False
    """Give each launch its own timestamped subfolder of ``save_tb_folder``.
    Off by default so a resumed run stays one curve."""

    save_for_all_ranks: bool = False
    """
    Whether to save TensorBoard/Wandb metrics only for rank 0 or for all ranks.
    When this option is False and pipeline_parallel_degree is > 1, the metrics
    component uses the 0th rank of the last stage pipeline group, which is the
    only stage that computes loss metrics.
    """

    enable_wandb: bool = False
