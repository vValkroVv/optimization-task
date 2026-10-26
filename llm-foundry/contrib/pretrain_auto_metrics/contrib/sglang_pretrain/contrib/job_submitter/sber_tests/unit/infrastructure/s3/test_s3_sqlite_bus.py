from __future__ import annotations

import builtins
import importlib.util
import sys
import types
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from job_submitter.infrastructure.s3.exceptions import (
    ConcurrentWriteError,
    DatabaseSyncError,
)
from job_submitter.infrastructure.s3.bus import (
    S3SQLiteBus,
    _extract_error_code,
    _states_equal,
    build_s3_client,
)
from job_submitter.infrastructure.s3.config import S3StorageConfig
from job_submitter.infrastructure.s3.sync_state import RemoteObjectState


pytestmark = pytest.mark.unit


class _S3Error(Exception):
    def __init__(self, code: str | None) -> None:
        if code is None:
            self.response = None
        else:
            self.response = {"Error": {"Code": code}}
        super().__init__(code)


class _DummyClient:
    def __init__(self) -> None:
        self.exists = True
        self.etag = "etag-1"
        self.version_id = "v1"
        self.remote_payload: bytes | None = None

        self.head_calls: list[tuple[str, str]] = []
        self.download_calls: list[tuple[str, str, str]] = []
        self.upload_calls: list[tuple[str, str, str]] = []

        self.head_exc: Exception | None = None
        self.download_exc: Exception | None = None
        self.upload_exc: Exception | None = None

    def head_object(self, *, Bucket: str, Key: str):
        self.head_calls.append((Bucket, Key))
        if self.head_exc is not None:
            raise self.head_exc
        if not self.exists:
            raise _S3Error("404")
        return {"ETag": f'"{self.etag}"', "VersionId": self.version_id}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.download_calls.append((bucket, key, filename))
        if self.download_exc is not None:
            raise self.download_exc
        if not self.exists:
            raise _S3Error("404")
        path = Path(filename)
        if self.remote_payload is not None:
            path.write_bytes(self.remote_payload)
        else:
            _write_sqlite_file(path)

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.upload_calls.append((filename, bucket, key))
        if self.upload_exc is not None:
            raise self.upload_exc
        self.exists = True
        self.etag = f"etag-{len(self.upload_calls) + 1}"
        self.version_id = f"v{len(self.upload_calls) + 1}"


@pytest.fixture
def _config() -> S3StorageConfig:
    return S3StorageConfig(
        endpoint_url="https://obs.example.local",
        region="ru-x",
        bucket="bucket-a",
        key="metrics.sqlite",
    )


def _write_sqlite_file(path: Path) -> bytes:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    return path.read_bytes()


def test_extract_error_code_variants() -> None:
    assert _extract_error_code(_S3Error("NoSuchKey")) == "NoSuchKey"
    assert _extract_error_code(_S3Error(None)) is None
    assert _extract_error_code(Exception("x")) is None

    broken = _S3Error("NoSuchKey")
    broken.response = {"Error": "not-dict"}
    assert _extract_error_code(broken) is None


def test_states_equal_covers_version_and_etag() -> None:
    s1 = RemoteObjectState(etag="a", version_id="1")
    s2 = RemoteObjectState(etag="b", version_id="1")
    s3 = RemoteObjectState(etag="a", version_id="2")
    s4 = RemoteObjectState(etag="a", version_id=None)
    s5 = RemoteObjectState(etag="a", version_id=None)

    assert _states_equal(None, None)
    assert not _states_equal(None, s1)
    assert _states_equal(s1, s2)
    assert not _states_equal(s1, s3)
    assert _states_equal(s4, s5)


def test_bus_download_success_returns_state(
    _config: S3StorageConfig, tmp_path: Path
) -> None:
    client = _DummyClient()
    bus = S3SQLiteBus(s3_client=client, config=_config)
    local_file = tmp_path / "db.sqlite"

    state = bus.download_or_initialize(local_file)

    assert isinstance(state, RemoteObjectState)
    assert state is not None
    assert state.etag == "etag-1"
    assert local_file.exists()
    assert local_file.read_bytes().startswith(b"SQLite format 3\x00")


def test_bus_download_repairs_chunked_sqlite_payload(
    _config: S3StorageConfig, tmp_path: Path
) -> None:
    seed = tmp_path / "seed.sqlite"
    payload = _write_sqlite_file(seed)
    client = _DummyClient()
    client.remote_payload = b"%x\r\n" % len(payload) + payload + b"\r\n0\r\n\r\n"
    bus = S3SQLiteBus(s3_client=client, config=_config)
    local_file = tmp_path / "db.sqlite"

    state = bus.download_or_initialize(local_file)

    assert state is not None
    assert local_file.read_bytes() == payload


def test_bus_download_rejects_non_sqlite_payload_and_removes_local(
    _config: S3StorageConfig, tmp_path: Path
) -> None:
    client = _DummyClient()
    client.remote_payload = b"not a sqlite file"
    bus = S3SQLiteBus(s3_client=client, config=_config)
    local_file = tmp_path / "db.sqlite"

    with pytest.raises(DatabaseSyncError, match="non-SQLite"):
        bus.download_or_initialize(local_file)

    assert not local_file.exists()


def test_bus_download_missing_returns_none(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    client.exists = False
    bus = S3SQLiteBus(s3_client=client, config=_config)

    local_file = tmp_path / "db.sqlite"
    local_file.write_text("stale", encoding="utf-8")

    state = bus.download_or_initialize(local_file)

    assert state is None
    assert not local_file.exists()


def test_bus_download_head_error_raises_sync_error(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    client.head_exc = _S3Error("AccessDenied")
    bus = S3SQLiteBus(s3_client=client, config=_config)

    with pytest.raises(DatabaseSyncError, match="metadata"):
        bus.download_or_initialize(tmp_path / "db.sqlite")


def test_bus_upload_requires_existing_file(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    bus = S3SQLiteBus(s3_client=client, config=_config)

    with pytest.raises(DatabaseSyncError, match="does not exist"):
        bus.upload(tmp_path / "missing.sqlite")


def test_bus_upload_detects_concurrent_write(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    bus = S3SQLiteBus(s3_client=client, config=_config)

    local_file = tmp_path / "db.sqlite"
    _write_sqlite_file(local_file)

    expected = RemoteObjectState(etag="old", version_id="old")
    with pytest.raises(ConcurrentWriteError):
        bus.upload(local_file, expected_state=expected)


def test_bus_upload_wraps_client_error(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    client.upload_exc = RuntimeError("boom")
    bus = S3SQLiteBus(s3_client=client, config=_config)

    local_file = tmp_path / "db.sqlite"
    _write_sqlite_file(local_file)

    expected = RemoteObjectState(etag="etag-1", version_id="v1")
    with pytest.raises(DatabaseSyncError, match="Failed to upload"):
        bus.upload(local_file, expected_state=expected)


def test_bus_upload_success_returns_new_state(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    bus = S3SQLiteBus(s3_client=client, config=_config)

    local_file = tmp_path / "db.sqlite"
    _write_sqlite_file(local_file)

    expected = RemoteObjectState(etag="etag-1", version_id="v1")
    state = bus.upload(local_file, expected_state=expected)

    assert isinstance(state, RemoteObjectState)
    assert state is not None
    assert state.version_id == "v2"
    assert client.upload_calls == [(str(local_file), "bucket-a", "metrics.sqlite")]


def test_bus_upload_rejects_non_sqlite_local_file(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    bus = S3SQLiteBus(s3_client=client, config=_config)

    local_file = tmp_path / "db.sqlite"
    local_file.write_text("not sqlite", encoding="utf-8")

    with pytest.raises(DatabaseSyncError, match="Refusing to upload non-SQLite"):
        bus.upload(
            local_file,
            expected_state=RemoteObjectState(etag="etag-1", version_id="v1"),
        )

    assert client.upload_calls == []


def _fake_boto(monkeypatch, calls):
    fake_boto3 = types.SimpleNamespace(
        client=lambda **kwargs: calls.append(kwargs) or kwargs
    )

    class _FakeConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_botocore_config = types.ModuleType("botocore.config")
    fake_botocore_config.Config = _FakeConfig

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_botocore_config)
    return _FakeConfig


def test_upload_sends_sqlite_without_trailing_checksum(tmp_path: Path) -> None:
    local_file = tmp_path / "inbox.sqlite"
    _write_sqlite_file(local_file)
    config = S3StorageConfig(
        endpoint_url="https://s3.example.test",
        region="us-east-1",
        bucket="tracker",
        key="inbox.sqlite",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        aws_session_token="",
    )
    captured = {}

    def capture_request(request, **kwargs):
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["body"] = request.body.read()
        raise RuntimeError("request captured before network access")

    with closing(build_s3_client(config)) as client:
        client.meta.events.register("before-send.s3.PutObject", capture_request)
        with pytest.raises(RuntimeError, match="request captured"):
            client.upload_file(str(local_file), config.bucket, config.key)

    assert captured["body"] == local_file.read_bytes()
    assert captured["headers"]["x-amz-content-sha256"] == b"UNSIGNED-PAYLOAD"
    assert "content-encoding" not in captured["headers"]
    assert "x-amz-trailer" not in captured["headers"]
    assert int(captured["headers"]["content-length"]) == local_file.stat().st_size


def test_build_s3_client_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    fake_cfg_cls = _fake_boto(monkeypatch, calls)

    config = S3StorageConfig(
        endpoint_url="https://obs.example.local",
        region="ru-moscow-1",
        bucket="b",
        key="k",
        aws_access_key_id="",
        aws_secret_access_key="",
        aws_session_token="",
    )

    result = build_s3_client(config)

    assert isinstance(result["config"], fake_cfg_cls)
    assert result["region_name"] == "ru-moscow-1"
    assert "aws_access_key_id" not in result
    assert len(calls) == 1


def test_build_s3_client_with_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    _fake_boto(monkeypatch, calls)

    config = S3StorageConfig(
        endpoint_url="https://obs.example.local",
        region="ru-moscow-1",
        bucket="b",
        key="k",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        aws_session_token="token",
    )

    result = build_s3_client(config)

    assert result["aws_access_key_id"] == "key"
    assert result["aws_secret_access_key"] == "secret"
    assert result["aws_session_token"] == "token"
    assert len(calls) == 1


def test_build_s3_client_with_credentials_without_session_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _fake_boto(monkeypatch, calls)

    config = S3StorageConfig(
        endpoint_url="https://obs.example.local",
        region="ru-moscow-1",
        bucket="b",
        key="k",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        aws_session_token="",
    )

    result = build_s3_client(config)

    assert result["aws_access_key_id"] == "key"
    assert result["aws_secret_access_key"] == "secret"
    assert "aws_session_token" not in result
    assert len(calls) == 1


def test_build_s3_client_defaults_verify_and_virtual_addressing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _fake_boto(monkeypatch, calls)

    config = S3StorageConfig(
        endpoint_url="https://obs.example.local",
        region="us-east-1",
        bucket="b",
        key="k",
        aws_access_key_id="",
        aws_secret_access_key="",
        aws_session_token="",
    )

    result = build_s3_client(config)

    assert result["verify"] is True
    assert result["config"].kwargs["s3"]["addressing_style"] == "virtual"


def test_build_s3_client_supports_on_prem_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _fake_boto(monkeypatch, calls)

    config = S3StorageConfig(
        endpoint_url="https://proxy.example.local",
        region="us-east-1",
        bucket="store",
        key="k",
        aws_access_key_id="",
        aws_secret_access_key="",
        aws_session_token="",
        verify_ssl=False,
        addressing_style="path",
    )

    result = build_s3_client(config)

    assert result["verify"] is False
    assert result["config"].kwargs["s3"]["addressing_style"] == "path"


@pytest.mark.parametrize(
    ("botocore_version", "expected_checksum_options"),
    [
        ("1.34.46", {}),
        (
            "1.36.0",
            {
                "request_checksum_calculation": "when_required",
                "response_checksum_validation": "when_required",
            },
        ),
        (
            "1.43.107",
            {
                "request_checksum_calculation": "when_required",
                "response_checksum_validation": "when_required",
            },
        ),
    ],
)
def test_build_s3_client_passes_checksum_options_only_to_botocore_that_knows_them(
    monkeypatch: pytest.MonkeyPatch,
    botocore_version: str,
    expected_checksum_options: dict[str, str],
) -> None:
    calls: list[dict[str, object]] = []
    _fake_boto(monkeypatch, calls)
    monkeypatch.setattr("botocore.__version__", botocore_version)

    config = S3StorageConfig(
        endpoint_url="https://proxy.example.local",
        region="us-east-1",
        bucket="store",
        key="k",
        aws_access_key_id="",
        aws_secret_access_key="",
        aws_session_token="",
    )

    result = build_s3_client(config)

    checksum_options = {
        key: value
        for key, value in result["config"].kwargs.items()
        if key in {"request_checksum_calculation", "response_checksum_validation"}
    }
    assert checksum_options == expected_checksum_options


def test_bus_download_missing_without_local_file(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    client.exists = False
    bus = S3SQLiteBus(s3_client=client, config=_config)

    local_file = tmp_path / "db.sqlite"
    state = bus.download_or_initialize(local_file)

    assert state is None
    assert not local_file.exists()


def test_bus_download_non_404_error_during_download_raises_sync_error(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    client.download_exc = _S3Error("AccessDenied")
    bus = S3SQLiteBus(s3_client=client, config=_config)

    with pytest.raises(DatabaseSyncError, match="Failed to download"):
        bus.download_or_initialize(tmp_path / "db.sqlite")


def test_bus_upload_metadata_error_is_wrapped(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    client.head_exc = _S3Error("AccessDenied")
    bus = S3SQLiteBus(s3_client=client, config=_config)

    local_file = tmp_path / "db.sqlite"
    _write_sqlite_file(local_file)

    with pytest.raises(DatabaseSyncError, match="metadata"):
        bus.upload(local_file, expected_state=None)


def test_bus_download_missing_during_download_returns_none_and_removes_local(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    client.download_exc = _S3Error("404")
    bus = S3SQLiteBus(s3_client=client, config=_config)

    local_file = tmp_path / "db.sqlite"
    local_file.write_text("stale", encoding="utf-8")

    state = bus.download_or_initialize(local_file)

    assert state is None
    assert not local_file.exists()


def test_bus_upload_when_remote_missing_returns_none_state(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    client.head_exc = _S3Error("404")
    bus = S3SQLiteBus(s3_client=client, config=_config)

    local_file = tmp_path / "db.sqlite"
    _write_sqlite_file(local_file)

    state = bus.upload(local_file, expected_state=None)

    assert state is None
    assert client.upload_calls == [(str(local_file), "bucket-a", "metrics.sqlite")]


def test_bus_download_missing_during_download_without_local_file(
    _config: S3StorageConfig,
    tmp_path: Path,
) -> None:
    client = _DummyClient()
    client.download_exc = _S3Error("NoSuchKey")
    bus = S3SQLiteBus(s3_client=client, config=_config)

    local_file = tmp_path / "db.sqlite"
    state = bus.download_or_initialize(local_file)

    assert state is None
    assert not local_file.exists()


def test_bus_import_requires_sqlite_s3_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus_path = (
        Path(__file__).resolve().parents[4]
        / "job_submitter"
        / "infrastructure"
        / "s3"
        / "bus.py"
    )
    real_import = builtins.__import__
    module_name = "job_submitter.infrastructure.s3.bus_import_error_test"

    for cached_name in list(sys.modules):
        if cached_name == "sqlite_s3_query" or cached_name.startswith(
            "sqlite_s3_query."
        ):
            monkeypatch.delitem(sys.modules, cached_name, raising=False)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sqlite_s3_query" or name.startswith("sqlite_s3_query."):
            raise ModuleNotFoundError("sqlite_s3_query")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location(module_name, bus_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader

    with pytest.raises(ModuleNotFoundError) as exc_info:
        spec.loader.exec_module(module)
    assert "sqlite_s3_query is not available" in str(exc_info.value)
    assert "contrib/job_submitter/contrib/sqlite_s3_query" in str(exc_info.value)
