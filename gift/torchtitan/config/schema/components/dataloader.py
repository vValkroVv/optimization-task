from typing import ClassVar

from torchtitan.config.configurable import config, Configurable


@config(family="dataloader")
class BaseDataLoaderConfig(Configurable.Config):
    """
    ``use_data_world``: built with ``DataWorld`` coordinates instead of
    plain ``dp_rank``/``dp_world_size``. ``replicate_batch_over``: mesh
    axes whose ranks need identical batches. Any subset of
    ``("pp", "cp", "tp")``; degree-1 axes are skipped.
    """

    use_data_world: ClassVar[bool] = False
    replicate_batch_over: ClassVar[tuple[str, ...]] = ()
    supports_reiteration: ClassVar[bool] = False
    dataset: str = ""
    dataset_path: str | None = None

    def validate_for_validation_loop(self) -> None:
        """Raise ValueError when the config makes re-iteration non-deterministic.

        Called only when ``supports_reiteration`` is set."""


@config
class ParallelAwareDataloaderConfig(BaseDataLoaderConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.components.dataloader:ParallelAwareDataloader"
    )

    num_workers: int = 0

    persistent_workers: bool = False
    """Keep workers alive between epochs. Only valid when num_workers > 0."""

    pin_memory: bool = False
    """Copy tensors to CUDA pinned memory before returning them."""

    prefetch_factor: int | None = None
    """
    Number of batches loaded in advance by each worker. Only valid when num_workers > 0.
    Default is 2 when num_workers > 0, otherwise None.
    """
