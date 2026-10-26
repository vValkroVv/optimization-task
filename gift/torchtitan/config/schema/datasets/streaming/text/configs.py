from dataclasses import dataclass, field

from torchtitan.config.schema.datasets.streaming.core.configs import (
    StreamConfig,
    StreamingDatasetConfig,
)
from torchtitan.config.schema.datasets.streaming.text.transforms import (
    validate_fim_rates,
)


@dataclass(kw_only=True, slots=True)
class TextStreamConfig(StreamConfig):
    """StreamConfig plus text-loader fields not passed to ``streaming.Stream``."""

    fim_rates: dict[str, float] | None = None

    def __post_init__(self) -> None:
        validate_fim_rates(self.fim_rates)


@dataclass(kw_only=True, slots=True)
class StreamingTextDatasetConfig(StreamingDatasetConfig):
    # pyrefly: ignore [bad-override]
    streams: list[TextStreamConfig] = field(default_factory=list)
    fim_rates: dict[str, float] | None = None
    enable_syntax_aware_fim: bool = False

    def __post_init__(self) -> None:
        validate_fim_rates(self.fim_rates)
