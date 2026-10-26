import pytest

from job_submitter.infrastructure.s3.config import (
    S3StorageConfig,
    build_db_url,
    load_config,
)


pytestmark = pytest.mark.unit


def test_load_config_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://obs.example.local")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("S3_BUCKET", "my-bucket")
    monkeypatch.setenv("S3_KEY", "data.sqlite")

    config = load_config()

    assert config.endpoint_url == "https://obs.example.local"
    assert config.region == "eu-west-1"
    assert config.bucket == "my-bucket"
    assert config.key == "data.sqlite"


def test_load_config_raises_on_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("S3_ENDPOINT_URL", "AWS_REGION", "S3_BUCKET", "S3_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(Exception):
        load_config()


def test_load_config_with_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://obs.example.local")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("S3_BUCKET", "original")
    monkeypatch.setenv("S3_KEY", "data.sqlite")

    from omegaconf import OmegaConf

    config = load_config(overrides=OmegaConf.create({"bucket": "overridden"}))

    assert config.bucket == "overridden"
    assert config.region == "eu-west-1"


def test_build_db_url_constructs_virtual_host_url() -> None:
    config = S3StorageConfig(
        endpoint_url="https://obs.example.local",
        region="r1",
        bucket="bucket-x",
        key="folder/db.sqlite",
    )

    assert build_db_url(config) == "https://bucket-x.obs.example.local/folder/db.sqlite"


def test_build_db_url_rejects_invalid_endpoint() -> None:
    config = S3StorageConfig(
        endpoint_url="not-a-url",
        region="r1",
        bucket="b1",
        key="k1",
    )

    with pytest.raises(ValueError, match="Invalid endpoint URL"):
        build_db_url(config)
