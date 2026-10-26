from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.datasets.streaming.core.dataloader import (
    BaseStreamingDataLoaderConfig,
)
from torchtitan.config.schema.datasets.streaming.text.configs import (
    StreamingTextDatasetConfig,
)
from torchtitan.config.schema.datasets.streaming.text.transforms import (
    DEFAULT_FIM_RATES,
)


def _stream_fim_rates(
    config: StreamingTextDatasetConfig,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Resolve fim rates per stream, indexed like ``config.streams``."""
    global_rates = config.fim_rates or DEFAULT_FIM_RATES
    stream_rates = [
        stream_config.fim_rates or global_rates for stream_config in config.streams
    ]
    if not config.enable_syntax_aware_fim:
        global_rates = {**global_rates, "syntax_aware_fim": 0.0}
        stream_rates = [{**rates, "syntax_aware_fim": 0.0} for rates in stream_rates]
    return stream_rates, global_rates


def _has_enabled_fim(
    stream_rates: list[dict[str, float]],
    default_rates: dict[str, float],
) -> bool:
    return sum(default_rates.values()) > 0 or any(
        sum(rates.values()) > 0 for rates in stream_rates
    )


@config(type="streaming_text")
class StreamingTextDataLoaderConfig(BaseStreamingDataLoaderConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.datasets.streaming.text.dataloader:StreamingTextDataLoader"
    )

    dataset: StreamingTextDatasetConfig = field(  # pyrefly: ignore [bad-override]
        default_factory=StreamingTextDatasetConfig
    )
    token_field: str = "tokens"
    transforms_epoch: int | None = None
    """Fixed epoch for sample transforms; None follows the streaming epoch."""

    def _validate_extra_for_validation_loop(self) -> None:
        stream_rates, default_rates = _stream_fim_rates(self.dataset)
        if _has_enabled_fim(stream_rates, default_rates) and (
            self.transforms_epoch is None
        ):
            raise ValueError(
                "streaming validation dataloader with FIM enabled must pin "
                "transforms_epoch (e.g. 0), otherwise FIM cuts change on "
                "every pass"
            )
