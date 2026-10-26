from __future__ import annotations

from pathlib import Path

import pytest

from metric_sampler.contract.raw_layout import AttemptDir, Dump
from metric_sampler.island.s3_target import S3Credentials, S3Target

pytestmark = pytest.mark.unit


def _write_rclone_conf(path: Path) -> None:
    path.write_text(
        "[metadata-s3]\n"
        "access_key_id = test-key\n"
        "secret_access_key = test-secret\n"
        "endpoint = https://s3.example.invalid\n"
    )


def test_credentials_from_rclone_conf(tmp_path: Path) -> None:
    conf_path = tmp_path / "rclone.conf"
    _write_rclone_conf(conf_path)

    credentials = S3Credentials.from_rclone_conf(conf_path, "metadata-s3")

    assert credentials == S3Credentials(access_key="test-key", secret_key="test-secret")


def test_credentials_from_rclone_conf_missing_section(tmp_path: Path) -> None:
    conf_path = tmp_path / "rclone.conf"
    _write_rclone_conf(conf_path)

    with pytest.raises(KeyError):
        S3Credentials.from_rclone_conf(conf_path, "does-not-exist")


@pytest.fixture
def attempt() -> AttemptDir:
    return AttemptDir(path=Path("/raw/expA/run1/attempt=3"), exp="expA", run="run1", attempt="3")


def test_object_url(target: S3Target) -> None:
    assert target.object_url("metric-sampler/v3/foo.parquet") == (
        "https://s3.example.invalid/metrics-bucket/metric-sampler/v3/foo.parquet"
    )


@pytest.mark.parametrize(
    ("structure", "expected_tail"),
    [(None, "'Parquet')"), ("step UInt32", "'Parquet', 'step UInt32')")],
)
def test_table_function_names_the_object_with_its_credentials_inline(
    target: S3Target, structure: str | None, expected_tail: str
) -> None:
    function = target.table_function("metric-sampler/v3/foo.parquet", "Parquet", structure)

    assert function == (
        "s3('https://s3.example.invalid/metrics-bucket/metric-sampler/v3/foo.parquet', 'test-key', 'test-secret', "
        + expected_tail
    )


def test_monitor_expert_ownership_key(target: S3Target, attempt: AttemptDir) -> None:
    assert target.monitor_expert_ownership_key(attempt) == (
        "metric-sampler/v3/metrics/exp=expA/run=run1/attempt=3/monitor_expert_ownership.json"
    )


def test_data_key(target: S3Target, attempt: AttemptDir) -> None:
    key = target.data_key("scalars", attempt, first_step=100, last_step=149, batch_hash="abc123def456")

    assert key == (
        "metric-sampler/v3/family=scalars/exp=expA/run=run1/attempt=3/part-00000100-00000149-abc123def456.parquet"
    )


def test_data_key_for_logged_family(target: S3Target, attempt: AttemptDir) -> None:
    key = target.data_key("logged", attempt, first_step=100, last_step=149, batch_hash="abc123def456")

    assert key == (
        "metric-sampler/v3/family=logged/exp=expA/run=run1/attempt=3/part-00000100-00000149-abc123def456.parquet"
    )


def test_node_manifests_key(target: S3Target, attempt: AttemptDir) -> None:
    key = target.node_manifests_key(attempt, first_step=0, last_step=99, batch_hash="abc123def456")

    assert key == (
        "metric-sampler/v3/node_manifests/exp=expA/run=run1/attempt=3/part-00000000-00000099-abc123def456.json"
    )


def test_batch_manifest_key(target: S3Target, attempt: AttemptDir) -> None:
    key = target.batch_manifest_key(attempt, first_step=0, last_step=99, batch_hash="abc123def456")

    assert key == (
        "metric-sampler/v3/batches/exp=expA/run=run1/attempt=3/part-00000000-00000099-abc123def456.json"
    )


def test_metrics_key(target: S3Target, attempt: AttemptDir) -> None:
    assert target.metrics_key(attempt) == "metric-sampler/v3/metrics/exp=expA/run=run1/attempt=3/metrics.json"


def test_streams_key(target: S3Target, attempt: AttemptDir) -> None:
    assert target.streams_key(attempt) == "metric-sampler/v3/metrics/exp=expA/run=run1/attempt=3/streams.json"


def test_monitor_parameters_key(target: S3Target, attempt: AttemptDir) -> None:
    assert target.monitor_parameters_key(attempt) == (
        "metric-sampler/v3/metrics/exp=expA/run=run1/attempt=3/monitor_parameters.json"
    )


def test_dump_key(dump_target: S3Target, attempt: AttemptDir) -> None:
    dump = Dump(path=Path("/irrelevant"), step=1200, rank=17)

    assert dump_target.dump_key(attempt, dump) == (
        "metric-sampler-dumps/exp=expA/run=run1/attempt=3/step=00001200/rank=00017.parquet"
    )


def test_dump_index_key(target: S3Target, attempt: AttemptDir) -> None:
    dump = Dump(path=Path("/irrelevant"), step=1200, rank=17)

    assert target.dump_index_key(attempt, dump) == (
        "metric-sampler/v3/dumps/exp=expA/run=run1/attempt=3/step=00001200/rank=00017.json"
    )
