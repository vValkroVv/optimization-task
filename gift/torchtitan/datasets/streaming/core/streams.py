from streaming.base import Stream

from torchtitan.config.schema.datasets.streaming.core.configs import StreamConfig


def build_stream(config: StreamConfig) -> Stream:
    if config.local is None and config.remote is None:
        stream_name = f" {config.name!r}" if config.name else ""
        raise ValueError(f"Stream{stream_name} must define local and/or remote")

    return Stream(
        remote=config.remote,
        local=config.local,
        split=config.split,
        proportion=config.proportion,
        repeat=config.repeat,
        choose=config.choose,
        download_retry=config.download_retry,
        download_timeout=config.download_timeout,
        validate_hash=config.validate_hash,
        keep_zip=config.keep_zip,
    )


def build_streams(configs: list[StreamConfig]) -> list[Stream]:
    if not configs:
        raise ValueError("At least one stream must be configured")
    return [build_stream(config) for config in configs]
