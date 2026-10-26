"""Configuration dataclasses for the kiwi (multimodal) token dataloader.

Schema only: this module must stay importable without torch, boto3 or any
runtime builder (see the architecture rules in AGENTS.md).

``KiwiDataLoader.Config`` is declared in
``torchtitan.config.schema.datasets.kiwi.kiwi_dataloader``; its nested settings
live here.
"""

from dataclasses import dataclass, field
from typing import Literal

from .constants import FLASH_ATTN_BLOCK_SIZE, DataModality


@dataclass(kw_only=True, slots=True)
class PackingConfig:
    """Parameters for :class:`~torchtitan.datasets.kiwi.packing.BankRouter`.

    ``max_banks`` is the K-Bounded Best Fit open-bin cap. ``delta``, ``w_age``,
    and ``w_fit`` are kept for the age/fit scorer if it is wired back in.
    """

    max_banks: int = 8
    delta: int = 4096
    w_age: float = 1.0
    w_fit: float = 1.0

    def __post_init__(self) -> None:
        if self.max_banks < 1:
            raise ValueError(f"max_banks must be >= 1, got {self.max_banks}")
        if self.delta < 0:
            raise ValueError(f"delta must be >= 0, got {self.delta}")


@dataclass(kw_only=True, slots=True)
class S3ClientConfig:
    """Credentials for the S3 endpoint serving visual tar shards.

    Leaving the key fields at ``None`` makes boto3 fall back to its usual
    environment/instance-profile resolution, which is the preferred way to
    pass secrets: do not commit them into config registry functions.
    """

    endpoint_url: str = ""
    max_pool_connections: int = 10
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    def __post_init__(self) -> None:
        if not self.endpoint_url:
            raise ValueError("s3_client.endpoint_url must be set")
        if (self.aws_access_key_id is None) != (self.aws_secret_access_key is None):
            raise ValueError(
                "s3_client: set both aws_access_key_id and aws_secret_access_key, "
                "or neither (to resolve credentials from the environment)"
            )


@dataclass(kw_only=True, slots=True)
class BaseDataStreamConfig:
    """One shard list feeding samples of a single kind into a dataset.

    Fields defaulting to ``None`` are *inheritable*: ``DatasetConfig`` copies
    its own value into them when the stream leaves them unset. They are
    resolved to concrete values before the stream is built, so runtime code
    can read them directly.
    """

    name: str = ""
    proportion: float = 1.0
    data_file: str = ""
    """Path to a newline-separated list of parquet shards."""

    num_reader_threads: int | None = None
    """ThreadPool workers that load parquet shards inside the stream."""

    prefetch_size: int | None = None
    """Max samples in in-flight prefetch futures (sum of pending shard sizes)."""
    filter: str = ""
    """Python expression evaluated per markup row; column names are locals.

    Passed to ``eval`` as written. Example:
    ``(width * height > 0.43 * 1024) and (quality_score > 10)``.
    """

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("data stream requires a non-empty name")
        if not self.data_file:
            raise ValueError(f"data stream {self.name!r} requires data_file")
        if self.proportion <= 0:
            raise ValueError(
                f"data stream {self.name!r} needs proportion > 0, got {self.proportion}"
            )


@dataclass(kw_only=True, slots=True)
class TextDataStreamConfig(BaseDataStreamConfig):
    text_field_name: str = "tokens"
    """Parquet column holding the pre-tokenized document."""

    fim_rates: dict[str, float] | None = None
    """Fill-in-the-middle transform rates; ``None`` inherits from the dataset."""


@dataclass(kw_only=True, slots=True)
class VisionDataStreamConfig(BaseDataStreamConfig):
    visual_field_name: str = ""
    """Parquet column holding the tar shard name the sample lives in."""

    visual_type: str = ""
    """Key the loaded tensor is passed to the preprocessor under, e.g. ``gen_image``."""

    text_probs: dict[str, float] = field(default_factory=dict)
    """Relative probability of picking each caption column."""

    tar_path_replace: str | None = None
    """``old->new`` rewrite applied to tar paths, e.g. to switch S3 for NFS."""

    uncond_prob: float | None = None
    text_loss: bool | None = None

    def __post_init__(self) -> None:
        # slots dataclasses rewrite the class object, so super() in
        # ``__post_init__`` can see a stale type. Call the parent explicitly.
        BaseDataStreamConfig.__post_init__(self)
        if not self.visual_field_name:
            raise ValueError(f"vision stream {self.name!r} requires visual_field_name")
        if not self.visual_type:
            raise ValueError(f"vision stream {self.name!r} requires visual_type")


@dataclass(kw_only=True, slots=True)
class DatasetConfig:
    """A group of streams sharing one content type and one set of defaults.

    Text and vision streams are separate lists rather than one heterogeneous
    list so that the config stays a plain dataclass tree that tyro can drive
    from the CLI.
    """

    name: str = ""
    type_of_content: Literal["image", "video", "image_instruct", "text"] = "text"
    proportion: float = 1.0

    text_streams: list[TextDataStreamConfig] = field(default_factory=list)
    vision_streams: list[VisionDataStreamConfig] = field(default_factory=list)

    # defaults inherited by streams that leave the matching field at None
    num_reader_threads: int = 1
    prefetch_size: int = 1024
    uncond_prob: float = 0.0
    text_loss: bool = False
    fim_rates: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset requires a non-empty name")
        if self.proportion <= 0:
            raise ValueError(
                f"dataset {self.name!r} needs proportion > 0, got {self.proportion}"
            )
        if not self.streams:
            raise ValueError(f"dataset {self.name!r} has no streams")

        names = [stream.name for stream in self.streams]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"dataset {self.name!r} has duplicate stream names: {sorted(duplicates)}"
            )

        self._inherit_defaults()

    @property
    def streams(self) -> list[BaseDataStreamConfig]:
        """Every stream of the dataset, text first."""
        return [*self.text_streams, *self.vision_streams]

    def _inherit_defaults(self) -> None:
        """Push dataset-level defaults into streams that left them unset.

        Idempotent on purpose: ``Configurable.Config.build`` copies configs
        with ``dataclasses.replace``, which is shallow, so the very same
        stream objects can be validated more than once.
        """
        shared = ("num_reader_threads", "prefetch_size")
        for stream in self.streams:
            for name in shared:
                if getattr(stream, name) is None:
                    setattr(stream, name, getattr(self, name))
        for stream in self.text_streams:
            if stream.fim_rates is None:
                stream.fim_rates = dict(self.fim_rates)
        for stream in self.vision_streams:
            if stream.uncond_prob is None:
                stream.uncond_prob = self.uncond_prob
            if stream.text_loss is None:
                stream.text_loss = self.text_loss


@dataclass(kw_only=True, slots=True)
class PreprocessorConfig:
    """Tokenization and patchification settings shared by every stream."""

    tokenizer_path: str = ""
    """HuggingFace tokenizer directory."""

    template: list[DataModality] = field(default_factory=lambda: ["text"])
    """Modalities to build preprocessors for, e.g. ``["text", "gen_image"]``."""

    patch_size: int | None = 16
    patch_size_latent: int | None = None
    max_len: int | None = None
    max_len_latent: int | None = None
    max_seq_len: int | None = 4096
    max_size: int = 256
    flash_block_size: int = FLASH_ATTN_BLOCK_SIZE
    enable_syntax_aware_fim: bool = False
    """When False, ``syntax_aware_fim`` rates are forced to zero before FIM."""

    def __post_init__(self) -> None:
        if not self.template:
            raise ValueError("preprocessor.template must list at least one modality")
