from __future__ import annotations

import pytest

from sqlite_s3_query import S3DatabaseConfig


def test_from_env_defaults_are_neutral_and_secure():
    config = S3DatabaseConfig.from_env(env={})

    assert config.endpoint_url == ""
    assert config.region == "us-east-1"
    assert config.bucket == ""
    assert config.key_prefix == "tracker/"
    assert config.verify_ssl is True
    assert config.force_path_style is False


def test_from_env_silences_insecure_request_warnings_when_ssl_disabled():
    import urllib3

    calls: list = []
    original = urllib3.disable_warnings
    urllib3.disable_warnings = lambda *a, **k: calls.append((a, k))
    try:
        S3DatabaseConfig.from_env(env={"S3_VERIFY_SSL": "false"})
        assert len(calls) == 1
        S3DatabaseConfig.from_env(env={"S3_VERIFY_SSL": "true"})
        assert len(calls) == 1
    finally:
        urllib3.disable_warnings = original


def test_from_env_reads_on_prem_gateway_flags():
    config = S3DatabaseConfig.from_env(
        env={
            "S3_ENDPOINT_URL": "https://clone.static.space",
            "AWS_REGION": "us-east-1",
            "S3_BUCKET": "store",
            "S3_VERIFY_SSL": "false",
            "S3_FORCE_PATH_STYLE": "true",
        }
    )

    assert config.endpoint_url == "https://clone.static.space"
    assert config.bucket == "store"
    assert config.verify_ssl is False
    assert config.force_path_style is True


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", False),
        ("false", False),
        ("No", False),
        ("OFF", False),
        ("", False),
        ("1", True),
        ("true", True),
        ("yes", True),
    ],
)
def test_verify_ssl_flag_parsing(raw, expected):
    config = S3DatabaseConfig.from_env(env={"S3_VERIFY_SSL": raw})
    assert config.verify_ssl is expected


def test_object_url_virtual_hosted_by_default():
    config = S3DatabaseConfig(
        endpoint_url="https://s3.example.com",
        region="us-east-1",
        bucket="store",
        key_prefix="tracker/",
    )
    assert (
        config.object_url("inbox.sqlite")
        == "https://store.s3.example.com/tracker/inbox.sqlite"
    )


def test_object_url_path_style_for_on_prem_gateway():
    config = S3DatabaseConfig(
        endpoint_url="https://clone.static.space",
        region="us-east-1",
        bucket="store",
        key_prefix="tracker/",
        force_path_style=True,
    )
    assert (
        config.object_url("daemons/inbox.sqlite")
        == "https://clone.static.space/store/tracker/daemons/inbox.sqlite"
    )


def test_object_url_rejects_invalid_endpoint():
    config = S3DatabaseConfig(
        endpoint_url="not-a-url",
        region="us-east-1",
        bucket="store",
        key_prefix="",
    )
    with pytest.raises(ValueError):
        config.object_url("inbox.sqlite")
