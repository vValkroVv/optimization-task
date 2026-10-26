from dataclasses import dataclass, field
from typing import ClassVar, Literal

from torchtitan.config.configurable import config
from torchtitan.config.value_types import ConfigValue
from torchtitan.config.schema.components.dataloader import ParallelAwareDataloaderConfig
from torchtitan.config.schema.datasets.hf.chat_processors import (
    ChatSampleProcessorConfig,
)


@config(type="hf_text")
class HuggingFaceTextDataLoaderConfig(ParallelAwareDataloaderConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.datasets.hf.text_datasets:HuggingFaceTextDataLoader"
    )

    dataset: str = "c4_test"

    infinite: bool = True


@dataclass(kw_only=True)
class HFDataSource:
    """One dataset source and its sampling weight."""

    dataset: str = "c4_test"
    dataset_path: str | None = None
    infinite: bool = True
    weight: float = 1


@config(type="hf_text_interleaved")
class InterleavedHuggingFaceTextDataLoaderConfig(ParallelAwareDataloaderConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.datasets.hf.text_datasets:InterleavedHuggingFaceTextDataLoader"
    )

    sources: list[HFDataSource] = field(default_factory=lambda: [HFDataSource()])

    seed: int = 42

    stopping_strategy: Literal["on_first_exhausted", "all_exhausted"] = (
        "on_first_exhausted"
    )

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("At least one source should be defined.")
        infinite_values = [source.infinite for source in self.sources]
        if len(set(infinite_values)) > 1:
            raise ValueError(
                f"All data sources must have the same 'infinite' setting, "
                f"got: {[(s.dataset, s.infinite) for s in self.sources]}"
            )


@config(type="hf_chat")
class ChatDataLoaderConfig(ParallelAwareDataloaderConfig):
    _owner_spec: ClassVar[str] = "torchtitan.datasets.hf.text_datasets:ChatDataLoader"

    dataset_path: str | None = None
    """HuggingFace dataset path (e.g. 'openai/gsm8k') or local path. Required."""

    load_dataset_kwargs: dict[str, ConfigValue] = field(default_factory=dict)
    """Extra kwargs passed to datasets.load_dataset()."""

    sample_processor: ChatSampleProcessorConfig
    """Turns one dataset sample into ``[user, assistant]`` messages."""

    infinite: bool = True
    """Whether to loop the dataset infinitely. Might hang on multi-GPU."""

    def __post_init__(self) -> None:
        if not self.dataset_path:
            raise ValueError(
                "Config requires dataset_path to be set "
                "(e.g., 'openai/gsm8k' or 'json')."
            )


@dataclass(kw_only=True)
class ChatDataSource:
    """One chat dataset source and its sampling weight."""

    dataset_path: str | None = None
    load_dataset_kwargs: dict[str, ConfigValue] = field(default_factory=dict)
    sample_processor: ChatSampleProcessorConfig
    infinite: bool = True
    weight: float = 1

    def __post_init__(self) -> None:
        if not self.dataset_path:
            raise ValueError(
                "Config requires dataset_path to be set "
                "(e.g., 'openai/gsm8k' or 'json')."
            )


@config(type="hf_chat_interleaved")
class InterleavedChatDataLoaderConfig(ParallelAwareDataloaderConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.datasets.hf.text_datasets:InterleavedChatDataLoader"
    )

    sources: list[ChatDataSource] = field(default_factory=list)

    seed: int = 42

    stopping_strategy: Literal["on_first_exhausted", "all_exhausted"] = (
        "on_first_exhausted"
    )

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("At least one source should be defined.")
        infinite_values = [source.infinite for source in self.sources]
        if len(set(infinite_values)) > 1:
            raise ValueError(
                f"All data sources must have the same 'infinite' setting, "
                f"got: {[(s.dataset_path, s.infinite) for s in self.sources]}"
            )
