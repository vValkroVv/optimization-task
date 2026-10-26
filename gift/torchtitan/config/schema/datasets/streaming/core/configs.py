from dataclasses import dataclass, field


@dataclass(kw_only=True, slots=True)
class StreamConfig:
    """Configuration for one streaming Stream.

    ``name`` is Titan-side metadata, not a ``streaming.Stream`` constructor arg.
    """

    name: str = ""
    remote: str | None = None
    local: str | None = None
    split: str | None = None

    proportion: float | None = None
    repeat: float | None = None
    choose: int | None = None

    download_retry: int | None = None
    download_timeout: float | None = None
    validate_hash: str | None = None
    keep_zip: bool | None = None


@dataclass(kw_only=True, slots=True)
class StreamingDatasetConfig:
    streams: list[StreamConfig] = field(default_factory=list)

    epoch_size: int | str | None = None
    predownload: int | None = None
    cache_limit: int | str | None = None
    partition_algo: str = "orig"
    num_canonical_nodes: int | None = None
    ncn_multiplier: int | None = None

    shuffle: bool = True
    shuffle_algo: str = "py1br"
    shuffle_seed: int = 9176
    shuffle_block_size: int = 1 << 18

    sampling_method: str = "balanced"
    batching_method: str = "random"
    legacy: bool = True
