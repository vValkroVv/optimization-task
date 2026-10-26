"""Unit tests for StreamConfig -> streaming.Stream conversion"""

import pytest

pytest.importorskip("streaming")

from torchtitan.datasets.streaming.core import streams as streams_module
from torchtitan.config.schema.datasets.streaming.core.configs import StreamConfig
from torchtitan.datasets.streaming.core.streams import build_stream, build_streams
from torchtitan.config.schema.datasets.streaming.text.configs import TextStreamConfig


class _RecordingStream:
    """Captures the exact kwargs build_stream passes to streaming.Stream"""

    __slots__ = ("kwargs",)

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_requires_local_or_remote():
    with pytest.raises(
        ValueError, match="Stream 'web' must define local and/or remote"
    ):
        build_stream(StreamConfig(name="web"))


def test_forwards_stream_fields_and_drops_titan_metadata(monkeypatch):
    monkeypatch.setattr(streams_module, "Stream", _RecordingStream)
    config = TextStreamConfig(
        name="code",
        local="/data/code",
        split="train",
        proportion=0.25,
        download_retry=3,
        fim_rates={"vanilla_fim": 1.0},
    )

    stream = build_stream(config)

    # name and fim_rates are Titan-side metadata and must not reach streaming.Stream
    assert stream.kwargs == {
        "remote": None,
        "local": "/data/code",
        "split": "train",
        "proportion": 0.25,
        "repeat": None,
        "choose": None,
        "download_retry": 3,
        "download_timeout": None,
        "validate_hash": None,
        "keep_zip": None,
    }


def test_real_stream_accepts_forwarded_kwargs(tmp_path):
    stream = build_stream(
        StreamConfig(local=str(tmp_path), split="train", proportion=0.5)
    )
    assert isinstance(stream, streams_module.Stream)
    assert stream.split == "train"


def test_build_streams_preserves_config_order(monkeypatch):
    monkeypatch.setattr(streams_module, "Stream", _RecordingStream)
    configs = [StreamConfig(local=f"/data/{name}") for name in ("a", "b", "c")]

    built = build_streams(configs)

    assert [stream.kwargs["local"] for stream in built] == [
        "/data/a",
        "/data/b",
        "/data/c",
    ]


def test_build_streams_rejects_empty_list():
    with pytest.raises(ValueError, match="At least one stream"):
        build_streams([])
