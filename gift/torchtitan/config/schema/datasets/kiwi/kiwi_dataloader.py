from collections.abc import Sequence
from dataclasses import field
from typing import ClassVar, Literal

from torchtitan.config.configurable import config
from torchtitan.config.schema.components.dataloader import BaseDataLoaderConfig
from torchtitan.datasets.kiwi.configs import (
    DatasetConfig,
    PackingConfig,
    PreprocessorConfig,
    S3ClientConfig,
)
from torchtitan.datasets.kiwi.constants import DIFFUSION_GENERATION, DataModality

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


_PREPROCESSOR_MODALITIES = (
    "text",
    "semantic_image",
    "semantic_video",
    "pixel_image",
    "pixel_video",
    "gen_image",
    "gen_video",
)


def _preprocessor_modality_keys(cfg: PreprocessorConfig) -> set[str]:
    return {"text"} | {m for m in _PREPROCESSOR_MODALITIES if m in cfg.template}


def _count_paddings(supported_modalities: Sequence[DataModality]) -> int:
    num_paddings = 0
    for modality in supported_modalities:
        if modality == "text":
            continue
        num_paddings += 1
        if f"<{modality}>" in DIFFUSION_GENERATION:
            num_paddings += 1
    return num_paddings


@config
class KiwiDataLoaderConfig(BaseDataLoaderConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.datasets.kiwi.kiwi_dataloader:KiwiDataLoader"
    )
    use_data_world: ClassVar[bool] = False
    replicate_batch_over: ClassVar[tuple[str, ...]] = ("pp", "cp", "tp")
    supports_reiteration: ClassVar[bool] = False
    datasets: list[DatasetConfig] = field(default_factory=list)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    s3_client: S3ClientConfig | None = None

    token_budget: int = 8192
    """Tokens per packed batch."""

    packing: PackingConfig = field(default_factory=PackingConfig)
    """K-bounded best-fit bank cap and leftover scorer weights."""

    supported_modalities: list[DataModality] = field(default_factory=list)
    """Preprocessor modalities that must appear in every batch (dummy-padded if missing)."""

    num_packing_workers: int = 1
    """Packing worker processes, on top of one IO process."""

    batch_queue_size: int = 1024
    """Capacity of the finished-batch queue."""

    worker_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    seed: int = 42
    """Base seed; each (rank, worker) derives a distinct stream from it."""

    startup_timeout: float = 600.0
    """Seconds to wait for one queue get before giving up."""

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("kiwi dataloader requires at least one dataset")
        names = [dataset.name for dataset in self.datasets]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate dataset names: {sorted(duplicates)}")
        available = _preprocessor_modality_keys(self.preprocessor)
        unknown = set(self.supported_modalities) - available
        if unknown:
            raise ValueError(
                "supported_modalities refers to modalities not built by "
                f"the preprocessor: {sorted(unknown)}"
            )
        reserve = _count_paddings(self.supported_modalities)
        if self.token_budget <= reserve:
            raise ValueError(
                "token_budget must be greater than reserved modality "
                f"padding ({reserve}), got {self.token_budget}"
            )
        if self.token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {self.token_budget}")
        if self.num_packing_workers < 1:
            raise ValueError(
                f"num_packing_workers must be >= 1, got {self.num_packing_workers}"
            )
        if self.batch_queue_size < 1:
            raise ValueError(
                f"batch_queue_size must be >= 1, got {self.batch_queue_size}"
            )
        if self.worker_log_level not in _LOG_LEVELS:
            raise ValueError(
                f"worker_log_level must be one of {_LOG_LEVELS}, "
                f"got {self.worker_log_level!r}"
            )
        if not self.preprocessor.tokenizer_path:
            raise ValueError("preprocessor.tokenizer_path must be set")
