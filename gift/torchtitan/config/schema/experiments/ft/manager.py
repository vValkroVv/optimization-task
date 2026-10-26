from typing import ClassVar

from torchtitan.config.configurable import config, Configurable


@config
class FTManagerConfig(Configurable.Config):
    _owner_spec: ClassVar[str] = "torchtitan.experiments.ft.manager:FTManager"

    enable: bool = False
    """
    Enable TorchFT integration. When TorchFT is enabled, HSDP will be used.
    And --fault_tolerance.data_parallel_replicate_degree should be 1 and
    --fault_tolerance.group_size will be used to control the maximum
    replicate group size as the replicate group size is dynamic.
    Note that this is still an experimental feature.
    """

    process_group: str = "gloo"
    """
    The process group to use for fault tolerance. Currently, only "gloo" and "nccl" are supported.
    """

    process_group_timeout_ms: int = 10000
    """
    The process group will abort if operations don't succeed within this duration.
    Note: This currently only works with gloo process group.
    """

    replica_id: int = 0
    """The TorchFT replica ID of this run."""

    group_size: int = 0
    """
    The number of TorchFT replicate groups. This number will be used for
    dataloader to split the dataset across the replicate groups and FSDP
    dimension
    """

    min_replica_size: int = 1
    """The minimum number of FT replica for each step."""

    semi_sync_method: str | None = None
    """
    The algorithm to use for semi-sync training. Currently, only "local_sgd" and "diloco" from
    torchft are supported
    (https://github.com/pytorch/torchft/blob/360c5c534bdeac959507e9d238ba9f3902d3fda9/torchft/local_sgd.py#L41)
    """
