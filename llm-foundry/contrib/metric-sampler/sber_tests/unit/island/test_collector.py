from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
import zlib
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metric_sampler.contract.metrics import Family
from metric_sampler.contract.raw_layout import AttemptDir, Dump, RawRoot
from metric_sampler.island import collector as collector_module
from metric_sampler.island.collector import Collector, Receipt, Shard
from metric_sampler.island.s3_target import S3Target

pytestmark = pytest.mark.unit

_UPLOAD_SETTINGS = (
    "s3_min_upload_part_size=8388608, s3_max_single_part_upload_size=8388608, "
    "s3_max_inflight_parts_for_one_file=128"
)
_ENCODE_SETTINGS = (
    "output_format_parquet_compression_method='zstd', output_format_parquet_write_bloom_filter=0, "
    "output_format_parquet_max_dictionary_size=0"
)
_DUMP_STATS = {"rows": 2, "points": ["resid"], "layers": [0], "tokens": 2}


def _attempt_path(root: Path) -> Path:
    return root / "expA" / "run1" / "attempt=0"


def _collected_attempt_path(root: Path) -> Path:
    return root / "_collected" / "expA" / "run1" / "attempt=0"


def _attempt_dir(root: Path) -> AttemptDir:
    return AttemptDir(_attempt_path(root), "expA", "run1", "0")


def _expected_batch_hash(relative_dirs: list[str]) -> str:
    return hashlib.sha1("\n".join(sorted(relative_dirs)).encode("utf-8")).hexdigest()[:12]


def _write_categories(session) -> list[str]:
    categories = []
    for sql, _ in session.queries:
        if sql.startswith("SELECT item, rows FROM ("):
            categories.append("verify")
        if not sql.startswith("INSERT INTO FUNCTION s3("):
            continue
        if ") SELECT * FROM file(" in sql:
            for family in Family:
                if f"family={family.value}/" in sql:
                    categories.append(f"data:{family.value}")
                    break
        elif "node_manifests/" in sql:
            categories.append("node_manifest")
        elif "/batches/" in sql:
            categories.append("batch_manifest")
        elif "/metrics/" in sql:
            categories.append("metrics")
    return categories


def _literal_content(sql: str) -> str:
    return sql.split("SELECT '", 1)[1].rsplit("' SETTINGS", 1)[0]


@pytest.mark.parametrize("family_name", ["scalars", "logged"])
def test_single_window_preserves_parquet_bytes(
    family_name, raw_root, target, fake_session_factory,
    write_window, write_logged_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    if family_name == "logged":
        window = write_logged_window(raw_root.path, "expA", "run1", "0", 10, 19)
    else:
        window = write_window(
            raw_root.path, "expA", "run1", "0", "host1", 10, 19,
            families=(Family(family_name),),
        )
    original = (window / f"{family_name}.parquet").read_bytes()
    objects = {}

    class StorageSession(fake_session_factory):
        def query(self, sql, fmt=None):
            upload = re.fullmatch(
                r"INSERT INTO FUNCTION s3\('([^']+)', 'test-key', 'test-secret', 'RawBLOB'\) "
                r"SELECT \* FROM file\('([^']+)', 'RawBLOB'\) SETTINGS .+",
                sql,
            )
            if upload is not None:
                objects[upload[1]] = Path(upload[2]).read_bytes()
            elif sql.startswith("SELECT count() FROM s3("):
                url = sql.split("'", 2)[1]
                self.uploaded_rows[family_name] = pq.read_metadata(BytesIO(objects[url])).num_rows
            return super().query(sql, fmt)

    session = StorageSession(stats={family_name: (10, 10, 19)})
    receipts = Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root)

    attempt = _attempt_dir(raw_root.path)
    relative_window = str(window.relative_to(attempt.path)).replace("\\", "/")
    key = target.data_key(family_name, attempt, 10, 19, _expected_batch_hash([relative_window]))
    assert receipts == [Receipt(family_name, key, 10)]
    assert objects == {target.object_url(key): original}
    collected = raw_root.path / "_collected" / window.relative_to(raw_root.path)
    assert (collected / f"{family_name}.parquet").read_bytes() == original
    assert (collected / "manifest.json").is_file()
    assert not window.exists()
    assert not (attempt.path / "_collector_batch.json").exists()


def test_collect_cycle_builds_one_batch_with_manifests_in_order(
    raw_root: RawRoot, target: S3Target, fake_session_factory, write_window, write_metrics_registry,
    scratch_dir: Path,
) -> None:
    root = raw_root.path
    metrics_content = '[{"metric_id": 1}]'
    write_metrics_registry(root, "expA", "run1", "0", content=metrics_content)
    host1_dir = write_window(
        root, "expA", "run1", "0", "host1", 0, 49,
        families=(Family.SCALARS, Family.EXPERT_COUNTS, Family.ROUTER_VECTORS),
    )
    host2_dir = write_window(root, "expA", "run1", "0", "host2", 0, 49, families=(Family.SCALARS,))
    host1_manifest = (host1_dir / "manifest.json").read_text()
    host2_manifest = (host2_dir / "manifest.json").read_text()

    session = fake_session_factory(stats={"scalars": (12, 0, 49)})
    collector = Collector(session, target, collected_retention_s=10**9)

    receipts = collector.collect_cycle(raw_root)

    attempt = _attempt_dir(root)
    relative_dirs = ["node=host1/w00000000-00000049", "node=host2/w00000000-00000049"]
    batch_hash = _expected_batch_hash(relative_dirs)

    scalars_key = target.data_key("scalars", attempt, 0, 49, batch_hash)
    expert_counts_key = target.data_key("expert_counts", attempt, 0, 49, batch_hash)
    router_vectors_key = target.data_key("router_vectors", attempt, 0, 49, batch_hash)

    assert receipts == [
        Receipt(family="scalars", key=scalars_key, rows=12),
        Receipt(family="expert_counts", key=expert_counts_key, rows=50),
        Receipt(family="router_vectors", key=router_vectors_key, rows=50),
    ]

    scalars_source = (
        f"file('{attempt.path}/{{node=host1/w00000000-00000049,node=host2/w00000000-00000049}}"
        "/scalars.parquet', 'Parquet')"
    )
    scalars_url = target.object_url(scalars_key)
    scalars_insert = (
        f"INSERT INTO FUNCTION s3('{scalars_url}', 'test-key', 'test-secret', 'Parquet') "
        f"SELECT * FROM {scalars_source} "
        f"SETTINGS s3_truncate_on_insert=1, {_ENCODE_SETTINGS}, use_hive_partitioning=0, {_UPLOAD_SETTINGS}"
    )
    assert scalars_insert in [sql for sql, fmt in session.queries if fmt is None]
    assert (
        f"SELECT count(), min(step), max(step) FROM {scalars_source} SETTINGS use_hive_partitioning=0",
        "CSV",
    ) in session.queries
    expert_counts_url = target.object_url(expert_counts_key)
    router_vectors_url = target.object_url(router_vectors_key)
    verified = [
        f"SELECT {index} AS item, rows FROM (SELECT count() AS rows FROM "
        f"s3('{url}', 'test-key', 'test-secret', 'Parquet', 'step UInt32'))"
        for index, url in enumerate([scalars_url, expert_counts_url, router_vectors_url])
    ]
    assert (f"SELECT item, rows FROM ({' UNION ALL '.join(verified)}) SETTINGS use_hive_partitioning=0", "CSV") in (
        session.queries
    )
    assert [sql for sql, fmt in session.queries if fmt == "CSV" and sql.startswith("SELECT item")] == [
        f"SELECT item, rows FROM ({' UNION ALL '.join(verified)}) SETTINGS use_hive_partitioning=0"
    ]

    for family_name, key in [("expert_counts", expert_counts_key), ("router_vectors", router_vectors_key)]:
        merged = scratch_dir / f"metric-sampler-collector.0.{family_name}.parquet"
        assert (
            f"INSERT INTO FUNCTION s3('{target.object_url(key)}', 'test-key', 'test-secret', 'RawBLOB') "
            f"SELECT * FROM file('{merged}', 'RawBLOB') "
            f"SETTINGS s3_truncate_on_insert=1, {_ENCODE_SETTINGS}, use_hive_partitioning=0, {_UPLOAD_SETTINGS}"
        ) in [sql for sql, fmt in session.queries if fmt is None]
        assert not merged.exists()

    manifests_url = target.object_url(target.node_manifests_key(attempt, 0, 49, batch_hash))
    assert (
        f"INSERT INTO FUNCTION s3('{manifests_url}', 'test-key', 'test-secret', 'RawBLOB') "
        f"SELECT '{host1_manifest}\n{host2_manifest}' SETTINGS s3_truncate_on_insert=1, use_hive_partitioning=0"
    ) in [sql for sql, fmt in session.queries]

    metrics_url = target.object_url(target.metrics_key(attempt))
    metrics_queries = [
        sql for sql, fmt in session.queries if sql.startswith(f"INSERT INTO FUNCTION s3('{metrics_url}'")
    ]
    assert metrics_queries == [
        f"INSERT INTO FUNCTION s3('{metrics_url}', 'test-key', 'test-secret', 'RawBLOB') "
        f"SELECT '{metrics_content}' SETTINGS s3_truncate_on_insert=1, use_hive_partitioning=0"
    ]

    batch_manifest_key = target.batch_manifest_key(attempt, 0, 49, batch_hash)
    batch_manifest_url = target.object_url(batch_manifest_key)
    batch_manifest_sql = next(
        sql for sql, fmt in session.queries if sql.startswith(f"INSERT INTO FUNCTION s3('{batch_manifest_url}'")
    )
    batch_manifest = json.loads(_literal_content(batch_manifest_sql))
    collected_at = batch_manifest.pop("collected_at")
    assert batch_manifest == {
        "schema_version": 1,
        "exp": "expA",
        "run": "run1",
        "attempt": "0",
        "batch": batch_hash,
        "first_step": 0,
        "last_step": 49,
        "source_windows": sorted(relative_dirs),
        "nodes": ["host1", "host2"],
        "rows_per_family": {"scalars": 12, "expert_counts": 50, "router_vectors": 50},
        "logged_dropped_rows": 0,
    }
    datetime.fromisoformat(collected_at)

    assert _write_categories(session) == [
        "metrics",
        "data:scalars",
        "data:expert_counts",
        "data:router_vectors",
        "verify",
        "node_manifest",
        "batch_manifest",
    ]

    assert not host1_dir.exists()
    assert not host2_dir.exists()
    collected_host1 = _collected_attempt_path(root) / "node=host1" / "w00000000-00000049"
    collected_host2 = _collected_attempt_path(root) / "node=host2" / "w00000000-00000049"
    assert (collected_host1 / "manifest.json").exists()
    assert (collected_host1 / "scalars.parquet").exists()
    assert (collected_host2 / "manifest.json").exists()


@pytest.mark.parametrize("mismatched_family", ["scalars", "expert_counts", "router_vectors"])
def test_a_row_count_mismatch_of_any_family_stops_the_batch_before_its_manifests(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry, caplog, mismatched_family,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    windows = [
        write_window(raw_root.path, "expA", "run1", "0", host, 0, 9, families=tuple(Family))
        for host in ["host1", "host2"]
    ]
    session = fake_session_factory(stats={"scalars": (12, 0, 9)}, uploaded_rows={mismatched_family: 2})

    with caplog.at_level(logging.ERROR, logger="metric_sampler.island.collector"):
        receipts = Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root)

    assert receipts == []
    assert all(window.exists() for window in windows)
    assert _write_categories(session) == [
        "metrics", "data:scalars", "data:expert_counts", "data:router_vectors", "verify",
    ]
    (record,) = caplog.records
    source_rows = {"scalars": 12, "expert_counts": 20, "router_vectors": 20}[mismatched_family]
    assert re.search(
        rf"row count mismatch for \S+/family={mismatched_family}/\S+: uploaded 2, source {source_rows}",
        record.getMessage(),
    )
    journal = _attempt_path(raw_root.path) / "_collector_batch.json"
    assert json.loads(journal.read_bytes())["receipts"] is None


def test_a_batch_without_family_files_uploads_its_manifests_without_verifying_rows(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    window = write_window(raw_root.path, "expA", "run1", "0", "host1", 0, 9, families=())
    session = fake_session_factory()

    assert Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root) == []

    assert _write_categories(session) == ["metrics", "node_manifest", "batch_manifest"]
    assert not window.exists()


def test_a_batch_uploads_the_manifests_of_all_its_node_windows_as_one_object(
    raw_root: RawRoot, target: S3Target, fake_session_factory, write_window, write_metrics_registry
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    manifests = []
    for host, first_step in [("host1", 0), ("host2", 0), ("host3", 10)]:
        window_dir = write_window(
            root, "expA", "run1", "0", host, first_step, first_step + 9, families=(Family.SCALARS,)
        )
        manifests.append(json.loads((window_dir / "manifest.json").read_bytes()))
    session = fake_session_factory(stats={"scalars": (3, 0, 19)})

    Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root)

    uploads = [_literal_content(sql) for sql, _ in session.queries if "node_manifests/" in sql]
    assert len(uploads) == 1
    decoder = json.JSONDecoder()
    documents = []
    position = 0
    while position < len(uploads[0]):
        document, position = decoder.raw_decode(uploads[0], position)
        documents.append(document)
        position += 1
    assert documents == manifests


def test_collect_cycle_skips_windows_without_manifest(
    raw_root: RawRoot, target: S3Target, fake_session_factory, write_window, write_metrics_registry
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    window_dir = write_window(root, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    (window_dir / "manifest.json").unlink()

    session = fake_session_factory()
    collector = Collector(session, target, collected_retention_s=10**9)

    receipts = collector.collect_cycle(raw_root)

    assert receipts == []
    assert session.queries == []
    assert window_dir.exists()


def test_collect_cycle_with_empty_root(raw_root: RawRoot, target: S3Target, fake_session_factory) -> None:
    raw_root.path.mkdir(parents=True)
    collector = Collector(fake_session_factory(), target, collected_retention_s=10**9)

    assert collector.collect_cycle(raw_root) == []


def test_collect_cycle_leaves_a_failed_attempt_in_place_and_collects_the_others(
    raw_root: RawRoot, target: S3Target, fake_session_factory, write_window, write_metrics_registry, caplog
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    write_metrics_registry(root, "expB", "run1", "0")
    failed_dir = write_window(root, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    collected_dir = write_window(root, "expB", "run1", "0", "host1", 0, 49, families=(Family.EXPERT_COUNTS,))

    session = fake_session_factory(
        stats={"scalars": (10, 0, 49), "expert_counts": (1, 0, 49)}, uploaded_rows={"scalars": 3}
    )
    collector = Collector(session, target, collected_retention_s=10**9)

    with caplog.at_level(logging.ERROR, logger="metric_sampler.island.collector"):
        receipts = collector.collect_cycle(raw_root)

    assert [receipt.family for receipt in receipts] == ["expert_counts"]
    assert failed_dir.exists()
    assert not collected_dir.exists()
    assert [sql for sql, _ in session.queries if "node_manifests/" in sql and "exp=expA" in sql] == []
    (record,) = caplog.records
    assert record.exc_info is None
    message = record.getMessage()
    assert message.startswith(
        f"collector: attempt {failed_dir.parents[1]} failed: CollectError: row count mismatch for "
    ) and message.endswith("uploaded 3, source 10")


def test_family_missing_from_entire_batch_is_skipped(
    raw_root: RawRoot, target: S3Target, fake_session_factory, write_window, write_metrics_registry
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    write_window(
        root, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS, Family.EXPERT_COUNTS)
    )

    session = fake_session_factory(stats={"scalars": (1, 0, 49), "expert_counts": (1, 0, 49)})
    collector = Collector(session, target, collected_retention_s=10**9)

    receipts = collector.collect_cycle(raw_root)

    assert {receipt.family for receipt in receipts} == {"scalars", "expert_counts"}
    assert not any("router_vectors" in sql for sql, _ in session.queries)


def test_metrics_registry_uploaded_once_and_reuploaded_on_change(
    raw_root: RawRoot, target: S3Target, fake_session_factory, write_window, write_metrics_registry
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0", content="[1]")
    write_window(root, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    session = fake_session_factory(stats={"scalars": (1, 0, 49)})
    collector = Collector(session, target, collected_retention_s=10**9)

    def metrics_uploads() -> list[str]:
        return [sql for sql, _ in session.queries if "/metrics/" in sql]

    collector.collect_cycle(raw_root)
    assert len(metrics_uploads()) == 1

    write_window(root, "expA", "run1", "0", "host2", 50, 99, families=(Family.SCALARS,))
    collector.collect_cycle(raw_root)
    assert len(metrics_uploads()) == 1

    write_metrics_registry(root, "expA", "run1", "0", content="[1, 2]")
    write_window(root, "expA", "run1", "0", "host3", 100, 149, families=(Family.SCALARS,))
    collector.collect_cycle(raw_root)
    assert len(metrics_uploads()) == 2


def test_batch_hash_depends_only_on_the_window_set(
    raw_root: RawRoot, target: S3Target, fake_session_factory, write_window, write_metrics_registry
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    write_metrics_registry(root, "expB", "run1", "0")
    write_window(root, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    write_window(root, "expB", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    write_window(root, "expB", "run1", "0", "host2", 0, 49, families=(Family.SCALARS,))

    session = fake_session_factory(stats={"scalars": (1, 0, 49)})
    collector = Collector(session, target, collected_retention_s=10**9)

    receipts = collector.collect_cycle(raw_root)

    def batch_hash_of(exp: str) -> str:
        (receipt,) = [r for r in receipts if f"exp={exp}/" in r.key]
        return receipt.key.rsplit("-", 1)[1].removesuffix(".parquet")

    hash_a = batch_hash_of("expA")
    hash_b = batch_hash_of("expB")
    assert hash_a == _expected_batch_hash(["node=host1/w00000000-00000049"])
    assert hash_b == _expected_batch_hash(["node=host1/w00000000-00000049", "node=host2/w00000000-00000049"])
    assert hash_a != hash_b


@pytest.mark.parametrize("dropped_rows", [None, 0, 17])
def test_collect_cycle_collects_logged_windows_into_the_same_batch(
    raw_root: RawRoot,
    target: S3Target,
    fake_session_factory,
    write_window,
    write_logged_window,
    write_metrics_registry,
    write_streams_registry,
    dropped_rows,
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    write_streams_registry(root, "expA", "run1", "0", content='[{"stream_id": 0, "name": "ar"}]')
    write_window(root, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    logged_dir = write_logged_window(root, "expA", "run1", "0", 0, 49)
    if dropped_rows is not None:
        manifest_path = logged_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["dropped_rows"] = dropped_rows
        manifest_path.write_text(json.dumps(manifest))

    session = fake_session_factory(stats={"scalars": (1, 0, 49), "logged": (2, 0, 49)})
    collector = Collector(session, target, collected_retention_s=10**9)

    receipts = collector.collect_cycle(raw_root)

    attempt = _attempt_dir(root)
    relative_dirs = ["logged/w00000000-00000049", "node=host1/w00000000-00000049"]
    batch_hash = _expected_batch_hash(relative_dirs)
    logged_key = target.data_key("logged", attempt, 0, 49, batch_hash)

    assert receipts == [
        Receipt(family="scalars", key=target.data_key("scalars", attempt, 0, 49, batch_hash), rows=1),
        Receipt(family="logged", key=logged_key, rows=2),
    ]
    assert _write_categories(session).count("node_manifest") == 1

    batch_manifest_sql = next(
        sql for sql, _ in session.queries
        if sql.startswith(
            f"INSERT INTO FUNCTION s3('{target.object_url(target.batch_manifest_key(attempt, 0, 49, batch_hash))}'"
        )
    )
    batch_manifest = json.loads(_literal_content(batch_manifest_sql))
    assert batch_manifest["nodes"] == ["host1"]
    assert batch_manifest["source_windows"] == sorted(relative_dirs)
    assert batch_manifest["rows_per_family"] == {"scalars": 1, "logged": 2}
    assert batch_manifest["logged_dropped_rows"] == (dropped_rows or 0)

    streams_url = target.object_url(target.streams_key(attempt))
    assert any(sql.startswith(f"INSERT INTO FUNCTION s3('{streams_url}'") for sql, _ in session.queries)

    assert not (_attempt_path(root) / "logged" / "w00000000-00000049").exists()
    collected_logged = _collected_attempt_path(root) / "logged" / "w00000000-00000049"
    assert (collected_logged / "logged.parquet").exists()


def test_collect_cycle_sums_dropped_rows_across_logged_windows(
    raw_root, target, fake_session_factory, write_logged_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    for first_step, dropped in [(0, 3), (50, 7)]:
        directory = write_logged_window(raw_root.path, "expA", "run1", "0", first_step, first_step + 49)
        path = directory / "manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["dropped_rows"] = dropped
        path.write_text(json.dumps(manifest))
    session = fake_session_factory(stats={"logged": (4, 0, 99)})
    Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root)
    manifest_sql = next(sql for sql, _ in session.queries if "/batches/" in sql)
    manifest = json.loads(_literal_content(manifest_sql))
    assert manifest["logged_dropped_rows"] == 10
    assert manifest["source_windows"] == ["logged/w00000000-00000049", "logged/w00000050-00000099"]


def test_collect_cycle_with_only_logged_windows(
    raw_root: RawRoot,
    target: S3Target,
    fake_session_factory,
    write_logged_window,
    write_metrics_registry,
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    write_logged_window(root, "expA", "run1", "0", 0, 49)

    session = fake_session_factory(stats={"logged": (2, 0, 49)})
    collector = Collector(session, target, collected_retention_s=10**9)

    receipts = collector.collect_cycle(raw_root)

    assert [r.family for r in receipts] == ["logged"]
    assert not any("node_manifests/" in sql for sql, _ in session.queries)


def test_streams_registry_uploaded_once_and_reuploaded_on_change_and_skipped_if_absent(
    raw_root: RawRoot,
    target: S3Target,
    fake_session_factory,
    write_window,
    write_metrics_registry,
    write_streams_registry,
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    write_window(root, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    session = fake_session_factory(stats={"scalars": (1, 0, 49)})
    collector = Collector(session, target, collected_retention_s=10**9)

    def streams_uploads() -> list[str]:
        return [sql for sql, _ in session.queries if "streams.json" in sql]

    collector.collect_cycle(raw_root)
    assert streams_uploads() == []

    write_streams_registry(root, "expA", "run1", "0", content='[{"stream_id": 0, "name": "ar"}]')
    write_window(root, "expA", "run1", "0", "host2", 50, 99, families=(Family.SCALARS,))
    collector.collect_cycle(raw_root)
    assert len(streams_uploads()) == 1

    write_window(root, "expA", "run1", "0", "host3", 100, 149, families=(Family.SCALARS,))
    collector.collect_cycle(raw_root)
    assert len(streams_uploads()) == 1

    write_streams_registry(
        root, "expA", "run1", "0", content='[{"stream_id": 0, "name": "ar"}, {"stream_id": 1, "name": "code"}]'
    )
    write_window(root, "expA", "run1", "0", "host4", 150, 199, families=(Family.SCALARS,))
    collector.collect_cycle(raw_root)
    assert len(streams_uploads()) == 2


@pytest.mark.parametrize("filename,records", [
    ("monitor_parameters.json", [dict(kind="parameter", index=0, name="model.layers.0.weight", rank=-1),
                                 dict(kind="optimizer_group", index=0, name="all", rank=-1)]),
    ("monitor_expert_ownership.json", [dict(rank=0, layer=2, global_expert_ids=[0, 1]),
                                       dict(rank=1, layer=2, global_expert_ids=[2, 3])]),
])
def test_monitor_catalog_preserves_names_and_rejects_changed_identity(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry, caplog,
    filename, records,
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    session = fake_session_factory(stats={"scalars": (1, 0, 49)})
    collector = Collector(session, target, collected_retention_s=10**9)
    attempt = _attempt_dir(root)
    content = json.dumps(records)
    for index in range(3):
        if index == 1:
            (attempt.path / filename).write_text(content)
        write_window(root, "expA", "run1", "0", "host1", index * 50, index * 50 + 49, families=(Family.SCALARS,))
        collector.collect_cycle(raw_root)
        uploads = [sql for sql, _ in session.queries if filename in sql]
        assert len(uploads) == (0 if index == 0 else 1)
    assert json.loads(_literal_content(uploads[0])) == records
    key = f"{target.prefix}/metrics/exp=expA/run=run1/attempt=0/{filename}"
    assert uploads[0].startswith("INSERT INTO FUNCTION s3('" + target.object_url(key) + "'")
    (attempt.path / filename).write_text(content + " ")
    write_window(root, "expA", "run1", "0", "host1", 150, 199, families=(Family.SCALARS,))
    assert collector.collect_cycle(raw_root) == []
    assert filename + " changed within an attempt" in caplog.text
    assert (attempt.path / "node=host1/w00000150-00000199/scalars.parquet").is_file()


def test_retention_deletes_expired_logged_windows_too(
    raw_root: RawRoot,
    target: S3Target,
    fake_session_factory,
    write_logged_window,
    write_metrics_registry,
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    write_logged_window(root, "expA", "run1", "0", 0, 49)

    session = fake_session_factory(stats={"logged": (1, 0, 49)})
    collector = Collector(session, target, collected_retention_s=3600)

    collector.collect_cycle(raw_root)

    collected_dir = _collected_attempt_path(root) / "logged" / "w00000000-00000049"
    assert collected_dir.exists()

    old_mtime = time.time() - 7200
    os.utime(collected_dir, (old_mtime, old_mtime))
    write_logged_window(root, "expA", "run1", "0", 50, 99)
    collector.collect_cycle(raw_root)

    assert not collected_dir.exists()


def test_move_collected_refreshes_the_window_mtime_so_a_backlog_is_not_retired_immediately(
    raw_root: RawRoot, target: S3Target, fake_session_factory, write_window, write_metrics_registry
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    window_dir = write_window(root, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))

    # Simulate a backlog: os.renames alone would carry the window's stale mtime into _collected,
    # so retention would delete it right away instead of after collected_retention_s.
    old_mtime = time.time() - 7200
    os.utime(window_dir, (old_mtime, old_mtime))

    session = fake_session_factory(stats={"scalars": (1, 0, 49)})
    collector = Collector(session, target, collected_retention_s=3600)

    collector.collect_cycle(raw_root)

    collected_dir = _collected_attempt_path(root) / "node=host1" / "w00000000-00000049"
    assert collected_dir.exists()
    assert collected_dir.stat().st_mtime > old_mtime + 3600


def test_retention_deletes_only_expired_collected_windows(
    raw_root: RawRoot, target: S3Target, fake_session_factory, write_window, write_metrics_registry
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    write_window(root, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    write_window(root, "expA", "run1", "0", "host2", 50, 99, families=(Family.SCALARS,))

    session = fake_session_factory(stats={"scalars": (1, 0, 49)})
    collector = Collector(session, target, collected_retention_s=3600)

    collector.collect_cycle(raw_root)

    old_dir = _collected_attempt_path(root) / "node=host1" / "w00000000-00000049"
    fresh_dir = _collected_attempt_path(root) / "node=host2" / "w00000050-00000099"
    assert old_dir.exists() and fresh_dir.exists()

    old_mtime = time.time() - 7200
    os.utime(old_dir, (old_mtime, old_mtime))

    collector.collect_cycle(raw_root)

    assert not old_dir.exists()
    assert fresh_dir.exists()


def test_dumps_are_collected_after_metrics_and_produce_an_index_and_a_collected_file(
    raw_root: RawRoot,
    target: S3Target,
    dump_target: S3Target,
    fake_session_factory,
    write_window,
    write_metrics_registry,
    write_dump,
) -> None:
    root = raw_root.path
    write_metrics_registry(root, "expA", "run1", "0")
    write_window(root, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    dump_path = write_dump(root, "expA", "run1", "0", step=100, rank=7, tokens=2)
    dump_size = dump_path.stat().st_size

    session = fake_session_factory(
        stats={"scalars": (1, 0, 49)},
        dump_stats={(100, 7): _DUMP_STATS},
    )
    collector = Collector(session, target, collected_retention_s=10**9, dump_target=dump_target)

    collector.collect_cycle(raw_root)

    # Metrics (and the window's data/manifests) are all queried before the dump stage starts.
    all_sql = [sql for sql, _ in session.queries]
    non_dump_indices = [i for i, sql in enumerate(all_sql) if not session.is_dump_query(sql)]
    dump_indices = [i for i, sql in enumerate(all_sql) if session.is_dump_query(sql)]
    assert non_dump_indices and any("scalars" in all_sql[i] for i in non_dump_indices)
    assert dump_indices
    assert max(non_dump_indices) < min(dump_indices)

    attempt = _attempt_dir(root)
    dump = Dump(path=dump_path, step=100, rank=7)
    dump_key = dump_target.dump_key(attempt, dump)
    dump_url = dump_target.object_url(dump_key)

    upload_sql = (
        f"INSERT INTO FUNCTION s3('{dump_url}', 'dump-key', 'dump-secret', 'RawBLOB') "
        f"SELECT * FROM file('{dump_path}', 'RawBLOB') "
        f"SETTINGS s3_truncate_on_insert=1, use_hive_partitioning=0, {_UPLOAD_SETTINGS}"
    )
    assert upload_sql in all_sql
    assert (
        f"SELECT count() FROM s3('{dump_url}', 'dump-key', 'dump-secret', 'Parquet', 'step Int64') "
        "SETTINGS use_hive_partitioning=0"
    ) in all_sql

    index_url = target.object_url(target.dump_index_key(attempt, dump))
    index_sql = next(sql for sql in all_sql if sql.startswith(f"INSERT INTO FUNCTION s3('{index_url}'"))
    index_manifest = json.loads(_literal_content(index_sql))
    assert index_manifest == {
        "exp": "expA",
        "run": "run1",
        "attempt": "0",
        "step": 100,
        "rank": 7,
        "rows": 2,
        "bytes": dump_size,
        "points": ["resid"],
        "layers": [0],
        "tokens": 2,
        "s3_key": f"{dump_target.bucket}/{dump_key}",
    }

    assert not dump_path.exists()
    collected_dump = _collected_attempt_path(root) / "dumps" / "step=00000100" / "rank=00007.parquet"
    assert collected_dump.exists()


def test_dump_upload_is_skipped_and_logged_once_when_no_dump_target_is_configured(
    raw_root: RawRoot, target: S3Target, fake_session_factory, write_dump, caplog
) -> None:
    root = raw_root.path
    dump_path = write_dump(root, "expA", "run1", "0", step=100, rank=7)
    session = fake_session_factory()
    collector = Collector(session, target, collected_retention_s=10**9)

    with caplog.at_level(logging.INFO, logger="metric_sampler.island.collector"):
        collector.collect_cycle(raw_root)
        collector.collect_cycle(raw_root)

    assert dump_path.exists()
    assert not session.dump_queries
    warnings = [r for r in caplog.records if "activation dumps stay on NFS" in r.message]
    assert len(warnings) == 1


def test_dump_row_count_mismatch_leaves_the_file_in_place_and_does_not_block_other_dumps(
    raw_root: RawRoot,
    target: S3Target,
    dump_target: S3Target,
    fake_session_factory,
    write_dump,
    caplog,
) -> None:
    root = raw_root.path
    bad_dump = write_dump(root, "expA", "run1", "0", step=100, rank=1, tokens=2)
    good_dump = write_dump(root, "expA", "run1", "0", step=200, rank=2, tokens=2)

    session = fake_session_factory(
        dump_stats={
            (100, 1): _DUMP_STATS,
            (200, 2): _DUMP_STATS,
        },
        dump_uploaded_rows={(100, 1): 1},
    )
    collector = Collector(session, target, collected_retention_s=10**9, dump_target=dump_target)

    with caplog.at_level(logging.ERROR, logger="metric_sampler.island.collector"):
        collector.collect_cycle(raw_root)

    assert bad_dump.exists()
    assert not good_dump.exists()
    (record,) = caplog.records
    assert record.exc_info is None
    assert record.getMessage().startswith(f"collector: dump {bad_dump} failed: CollectError: ")


def test_a_failed_dump_is_one_log_line_without_the_dump_targets_credentials(
    raw_root: RawRoot,
    target: S3Target,
    dump_target: S3Target,
    fake_session_factory,
    write_dump,
    caplog,
) -> None:
    dump = write_dump(raw_root.path, "expA", "run1", "0", step=100, rank=1)
    session = fake_session_factory(dump_stats={(100, 1): _DUMP_STATS})

    class StatementQuotingSession:
        def query(self, sql, fmt=None):
            if session.is_dump_query(sql) and sql.startswith("INSERT"):
                raise RuntimeError(f"S3 error in {sql}")
            return session.query(sql, fmt)

    collector = Collector(StatementQuotingSession(), target, collected_retention_s=10**9, dump_target=dump_target)
    with caplog.at_level(logging.ERROR, logger="metric_sampler.island.collector"):
        collector.collect_cycle(raw_root)

    (record,) = caplog.records
    message = record.getMessage()
    assert message.startswith(f"collector: dump {dump} failed: RuntimeError: S3 error in INSERT")
    assert record.exc_info is None
    assert "***" in message
    assert dump_target.credentials.access_key not in message and dump_target.credentials.secret_key not in message


def test_dumps_are_collected_oldest_step_first_within_the_per_cycle_byte_budget(
    raw_root: RawRoot,
    target: S3Target,
    dump_target: S3Target,
    fake_session_factory,
    write_dump,
    write_window,
    write_metrics_registry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = raw_root.path
    older_dump = write_dump(root, "expA", "run1", "0", step=100, rank=1)
    newer_dump = write_dump(root, "expA", "run1", "0", step=200, rank=2)
    monkeypatch.setattr(collector_module, "_DUMP_BYTES_PER_CYCLE", older_dump.stat().st_size)
    now = 1000.0
    monkeypatch.setattr(collector_module.time, "monotonic", lambda: now)

    session = fake_session_factory(
        stats={"scalars": (10, 0, 9)},
        dump_stats={
            (100, 1): _DUMP_STATS,
            (200, 2): _DUMP_STATS,
        }
    )
    collector = Collector(session, target, collected_retention_s=10**9, dump_target=dump_target)

    collector.collect_cycle(raw_root)
    assert not older_dump.exists()
    assert newer_dump.exists()

    now += 59.999
    write_metrics_registry(root, "expA", "run1", "0")
    window = write_window(root, "expA", "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    receipts = collector.collect_cycle(raw_root)
    assert [(receipt.family, receipt.rows) for receipt in receipts] == [("scalars", 10)]
    assert not window.exists()
    assert newer_dump.exists()

    now = 1060.0
    collector.collect_cycle(raw_root)
    assert not newer_dump.exists()


def test_a_single_dump_bigger_than_the_budget_is_still_uploaded_when_first_in_the_cycle(
    raw_root: RawRoot,
    target: S3Target,
    dump_target: S3Target,
    fake_session_factory,
    write_dump,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = raw_root.path
    dump_path = write_dump(root, "expA", "run1", "0", step=100, rank=1)
    monkeypatch.setattr(collector_module, "_DUMP_BYTES_PER_CYCLE", 1)

    session = fake_session_factory(
        dump_stats={(100, 1): _DUMP_STATS}
    )
    collector = Collector(session, target, collected_retention_s=10**9, dump_target=dump_target)

    collector.collect_cycle(raw_root)

    assert not dump_path.exists()


def test_unreadable_dumps_are_skipped_with_one_line_per_cycle_and_collected_once_readable(
    raw_root: RawRoot,
    target: S3Target,
    dump_target: S3Target,
    fake_session_factory,
    write_dump,
    monkeypatch: pytest.MonkeyPatch,
    caplog,
) -> None:
    root = raw_root.path
    first_unreadable = write_dump(root, "expA", "run1", "0", step=100, rank=1)
    second_unreadable = write_dump(root, "expA", "run1", "0", step=100, rank=2)
    readable = write_dump(root, "expA", "run1", "0", step=200, rank=3)
    unreadable_paths = {first_unreadable, second_unreadable}
    real_access = os.access
    monkeypatch.setattr(
        collector_module.os, "access", lambda path, mode: Path(path) not in unreadable_paths and real_access(path, mode)
    )
    # Room for one dump per cycle: the unreadable ones must not use it up.
    monkeypatch.setattr(collector_module, "_DUMP_BYTES_PER_CYCLE", readable.stat().st_size)
    now = 1000.0
    monkeypatch.setattr(collector_module.time, "monotonic", lambda: now)
    session = fake_session_factory(dump_stats=dict.fromkeys([(100, 1), (100, 2), (200, 3)], _DUMP_STATS))
    collector = Collector(session, target, collected_retention_s=3600, dump_target=dump_target)

    with caplog.at_level(logging.WARNING, logger="metric_sampler.island.collector"):
        collector.collect_cycle(raw_root)

    assert not readable.exists()
    assert first_unreadable.exists() and second_unreadable.exists()
    assert [record.getMessage() for record in caplog.records] == [
        f"collector: skipping 2 unreadable dumps, first {first_unreadable}"
    ]
    assert not any("rank=00001" in sql or "rank=00002" in sql for sql in session.dump_queries)
    collected = _collected_attempt_path(root) / "dumps"
    assert [path.name for path in collected.glob("*/*")] == ["rank=00003.parquet"]

    unreadable_paths.clear()
    now += 60
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="metric_sampler.island.collector"):
        collector.collect_cycle(raw_root)

    assert not first_unreadable.exists() and second_unreadable.exists()
    assert caplog.records == []
    now += 60
    collector.collect_cycle(raw_root)
    assert not second_unreadable.exists()


def test_collect_cycle_with_dump_target_configured_but_no_dumps_present(
    raw_root: RawRoot, target: S3Target, dump_target: S3Target, fake_session_factory
) -> None:
    raw_root.path.mkdir(parents=True)
    collector = Collector(fake_session_factory(), target, collected_retention_s=10**9, dump_target=dump_target)

    assert collector.collect_cycle(raw_root) == []


def test_dump_retention_deletes_expired_collected_dumps_but_keeps_fresh_ones(
    raw_root: RawRoot,
    target: S3Target,
    dump_target: S3Target,
    fake_session_factory,
    write_dump,
) -> None:
    root = raw_root.path
    write_dump(root, "expA", "run1", "0", step=100, rank=1)
    write_dump(root, "expA", "run1", "0", step=200, rank=2)

    session = fake_session_factory(
        dump_stats={
            (100, 1): _DUMP_STATS,
            (200, 2): _DUMP_STATS,
        }
    )
    collector = Collector(session, target, collected_retention_s=3600, dump_target=dump_target)
    collector.collect_cycle(raw_root)

    old_dump = _collected_attempt_path(root) / "dumps" / "step=00000100" / "rank=00001.parquet"
    fresh_dump = _collected_attempt_path(root) / "dumps" / "step=00000200" / "rank=00002.parquet"
    assert old_dump.exists() and fresh_dump.exists()

    old_mtime = time.time() - 7200
    os.utime(old_dump.parent, (old_mtime, old_mtime))

    collector.collect_cycle(raw_root)

    assert not old_dump.exists()
    assert not old_dump.parent.exists()
    assert fresh_dump.exists()


def test_collecting_a_dump_preserves_file_mtime_and_refreshes_directory_retention(
    raw_root: RawRoot, target: S3Target, dump_target: S3Target, fake_session_factory, write_dump,
) -> None:
    source = write_dump(raw_root.path, "expA", "run1", "0", step=100, rank=1)
    old_mtime = time.time() - 7200
    os.utime(source, (old_mtime, old_mtime))
    original_mtime_ns = source.stat().st_mtime_ns
    original_content = source.read_bytes()
    session = fake_session_factory(
        dump_stats={(100, 1): _DUMP_STATS}
    )
    collector = Collector(session, target, collected_retention_s=3600, dump_target=dump_target)

    collector.collect_cycle(raw_root)

    destination = _collected_attempt_path(raw_root.path) / "dumps" / "step=00000100" / "rank=00001.parquet"
    assert not source.exists()
    assert destination.read_bytes() == original_content
    assert destination.stat().st_mtime_ns == original_mtime_ns
    assert destination.parent.stat().st_mtime > old_mtime + 3600


def test_dump_retention_keeps_all_ranks_until_the_last_collected_rank_expires(
    raw_root: RawRoot, target: S3Target, dump_target: S3Target, fake_session_factory, write_dump,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = raw_root.path
    now = 1000.0
    monkeypatch.setattr(collector_module.time, "monotonic", lambda: now)
    first_source = write_dump(root, "expA", "run1", "0", step=100, rank=1)
    first_content = first_source.read_bytes()
    session = fake_session_factory(
        dump_stats={
            (100, 1): _DUMP_STATS,
            (100, 2): _DUMP_STATS,
        }
    )
    collector = Collector(session, target, collected_retention_s=3600, dump_target=dump_target)
    collector.collect_cycle(raw_root)
    step_dir = _collected_attempt_path(root) / "dumps" / "step=00000100"
    old_mtime = time.time() - 7200
    os.utime(step_dir, (old_mtime, old_mtime))
    second_source = write_dump(root, "expA", "run1", "0", step=100, rank=2)
    second_content = second_source.read_bytes()

    now += 60
    collector.collect_cycle(raw_root)

    assert not first_source.exists() and not second_source.exists()
    assert {path.name: path.read_bytes() for path in step_dir.iterdir()} == {
        "rank=00001.parquet": first_content,
        "rank=00002.parquet": second_content,
    }
    assert step_dir.stat().st_mtime > old_mtime + 3600
    os.utime(step_dir, (old_mtime, old_mtime))

    collector.collect_cycle(raw_root)

    assert not step_dir.exists()


def test_batches_limit_combined_family_bytes_and_leave_the_next_window_for_the_next_cycle(
    raw_root, target, fake_session_factory, write_window, write_logged_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    first = write_window(
        raw_root.path, "expA", "run1", "0", "host2", 0, 9,
        families=(Family.SCALARS, Family.EXPERT_COUNTS),
    )
    second = write_window(raw_root.path, "expA", "run1", "0", "host1", 10, 19, families=(Family.SCALARS,))
    third = write_logged_window(raw_root.path, "expA", "run1", "0", 20, 29)
    with (first / "scalars.parquet").open("r+b") as stream:
        stream.truncate(collector_module._BATCH_BYTES // 4)
    first_size = sum(path.stat().st_size for path in first.glob("*.parquet"))
    with (second / "scalars.parquet").open("r+b") as stream:
        stream.truncate(collector_module._BATCH_BYTES - first_size)
    session = fake_session_factory(stats={"scalars": (20, 0, 19), "logged": (10, 20, 29)})
    collector = Collector(session, target, collected_retention_s=10**9)

    receipts = collector.collect_cycle(raw_root)

    assert [(receipt.family, receipt.rows) for receipt in receipts] == [("scalars", 20), ("expert_counts", 10)]
    assert not first.exists() and not second.exists()
    assert third.exists()
    manifests = [json.loads(_literal_content(sql)) for sql, _ in session.queries if "/batches/" in sql]
    assert [manifest["source_windows"] for manifest in manifests] == [[
        "node=host1/w00000010-00000019", "node=host2/w00000000-00000009",
    ]]

    receipts = collector.collect_cycle(raw_root)

    assert [(receipt.family, receipt.rows) for receipt in receipts] == [("logged", 10)]
    assert not third.exists()


def test_one_oversized_window_is_not_split_or_blocked_by_the_byte_limit(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    first = write_window(raw_root.path, "expA", "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    second = write_window(raw_root.path, "expA", "run1", "0", "host1", 10, 19, families=(Family.SCALARS,))
    with (first / "scalars.parquet").open("r+b") as stream:
        stream.truncate(collector_module._BATCH_BYTES + 1)
    session = fake_session_factory(stats={"scalars": (10, 0, 9)})

    receipts = Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root)

    assert len(receipts) == 1 and receipts[0].rows == 10
    assert not first.exists() and second.exists()
    manifest = json.loads(_literal_content(next(sql for sql, _ in session.queries if "/batches/" in sql)))
    assert manifest["source_windows"] == ["node=host1/w00000000-00000009"]


def test_window_limit_selects_oldest_steps_and_gives_other_attempts_a_turn(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry,
) -> None:
    for exp in ["expA", "expB"]:
        write_metrics_registry(raw_root.path, exp, "run1", "0")
    windows = [
        write_window(
            raw_root.path, "expA", "run1", "0", f"host{129 - index:03d}",
            index * 10, index * 10 + 9, families=(Family.SCALARS,),
        )
        for index in range(130)
    ]
    other = write_window(raw_root.path, "expB", "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    session = fake_session_factory(stats={"scalars": (10, 0, 9)})

    receipts = Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root)

    assert len(receipts) == 2
    assert all(not path.exists() for path in windows[:128])
    assert all(path.exists() for path in windows[128:])
    assert not other.exists()
    manifests = [json.loads(_literal_content(sql)) for sql, _ in session.queries if "/batches/" in sql]
    assert [manifest["exp"] for manifest in manifests] == ["expA", "expB"]
    assert manifests[0]["source_windows"] == sorted(
        f"node=host{129 - index:03d}/w{index * 10:08d}-{index * 10 + 9:08d}" for index in range(128)
    )


@pytest.mark.parametrize("failure_stage", ["family=scalars/", "node_manifests/", "/batches/"])
def test_restart_retries_the_same_batch_when_newer_and_late_windows_arrive(
    raw_root, target, fake_session_factory, write_window, write_logged_window, write_metrics_registry,
    failure_stage,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    original = write_window(raw_root.path, "expA", "run1", "0", "host1", 10, 19, families=(Family.SCALARS,))
    logged = write_logged_window(raw_root.path, "expA", "run1", "0", 10, 19)
    failed = fake_session_factory(stats={"scalars": (10, 10, 19), "logged": (10, 10, 19)})

    class InterruptedSession:
        def query(self, sql, fmt=None):
            result = failed.query(sql, fmt)
            if sql.startswith("INSERT INTO FUNCTION s3(") and failure_stage in sql:
                raise OSError("upload response lost")
            return result

    assert Collector(InterruptedSession(), target, collected_retention_s=10**9).collect_cycle(raw_root) == []
    assert original.exists() and logged.exists()
    journal = _attempt_path(raw_root.path) / "_collector_batch.json"
    sources = ["logged/w00000010-00000019", "node=host1/w00000010-00000019"]
    assert json.loads(journal.read_bytes()) == {"source_windows": sources, "receipts": None}
    older = write_window(raw_root.path, "expA", "run1", "0", "host0", 0, 9, families=(Family.SCALARS,))
    newer = write_window(raw_root.path, "expA", "run1", "0", "host1", 20, 29, families=(Family.SCALARS,))
    resumed = fake_session_factory(stats={"scalars": (10, 10, 19), "logged": (10, 10, 19)})

    receipts = Collector(resumed, target, collected_retention_s=10**9).collect_cycle(raw_root)

    attempt = _attempt_dir(raw_root.path)
    batch_hash = _expected_batch_hash(sources)
    assert receipts == [
        Receipt("scalars", target.data_key("scalars", attempt, 10, 19, batch_hash), 10),
        Receipt("logged", target.data_key("logged", attempt, 10, 19, batch_hash), 10),
    ]
    assert not original.exists() and not logged.exists() and not journal.exists()
    assert older.exists() and newer.exists()
    manifest = json.loads(_literal_content(next(sql for sql, _ in resumed.queries if "/batches/" in sql)))
    assert manifest["source_windows"] == sources
    scalar_upload = next(sql for sql, _ in failed.queries if sql.startswith("INSERT") and "family=scalars/" in sql)
    assert scalar_upload in [sql for sql, _ in resumed.queries]


@pytest.mark.parametrize("finish_moves_before_restart", [False, True])
def test_restart_finishes_committed_batch_moves_without_reupload_and_retention_preserves_inflight(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry, finish_moves_before_restart,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    first = write_window(raw_root.path, "expA", "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    second = write_window(raw_root.path, "expA", "run1", "0", "host2", 0, 9, families=(Family.SCALARS,))
    collected = _collected_attempt_path(raw_root.path)
    collected.mkdir(parents=True)
    obstruction = collected / "node=host2"
    obstruction.write_bytes(b"prevents the second move")
    session = fake_session_factory(stats={"scalars": (20, 0, 9)})

    assert Collector(session, target, collected_retention_s=0).collect_cycle(raw_root) == []

    journal = _attempt_path(raw_root.path) / "_collector_batch.json"
    saved = json.loads(journal.read_bytes())
    assert saved["receipts"] == [{"family": "scalars", "key": next(
        sql.split("'", 2)[1].split(f"/{target.bucket}/", 1)[1]
        for sql, _ in session.queries if sql.startswith("INSERT") and "family=scalars/" in sql
    ), "rows": 20}]
    assert not first.exists() and second.exists()
    assert (collected / "node=host1" / first.name / "scalars.parquet").exists()
    old_mtime = time.time() - 7200
    os.utime(collected / "node=host1" / first.name, (old_mtime, old_mtime))
    obstruction.unlink()
    if finish_moves_before_restart:
        destination = collected / "node=host2" / second.name
        destination.parent.mkdir()
        shutil.move(second, destination)
    resumed = fake_session_factory()

    receipts = Collector(resumed, target, collected_retention_s=3600).collect_cycle(raw_root)

    assert [(receipt.family, receipt.rows) for receipt in receipts] == [("scalars", 20)]
    assert resumed.queries == []
    assert not journal.exists() and not second.exists()
    assert (collected / "node=host1" / first.name / "scalars.parquet").exists()
    assert (collected / "node=host2" / second.name / "scalars.parquet").exists()
    assert (collected / "node=host1" / first.name).stat().st_mtime > old_mtime + 3600


@pytest.mark.parametrize("source", ["../node=host1/w00000000-00000009", "node=../../outside/w00000000-00000009"])
def test_invalid_journal_path_is_reported_without_reading_or_uploading_it(
    raw_root, target, fake_session_factory, write_metrics_registry, caplog, source,
) -> None:
    registry = write_metrics_registry(raw_root.path, "expA", "run1", "0")
    journal = registry.parent / "_collector_batch.json"
    journal.write_text(json.dumps({"source_windows": [source], "receipts": None}))
    session = fake_session_factory()

    with caplog.at_level(logging.ERROR, logger="metric_sampler.island.collector"):
        receipts = Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root)

    assert receipts == [] and session.queries == []
    assert journal.exists()
    assert [record.getMessage() for record in caplog.records] == [
        f"collector: attempt {journal.parent} failed: CollectError: invalid source window in {journal}: {source}"
    ]


def test_retention_keeps_only_the_window_waiting_for_the_producer_publication_ack(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    for first in [0, 10]:
        write_window(raw_root.path, "expA", "run1", "0", "host1", first, first + 9, families=(Family.SCALARS,))
    session = fake_session_factory(stats={"scalars": (20, 0, 19)})
    collector = Collector(session, target, collected_retention_s=3600)
    collector.collect_cycle(raw_root)
    collected_node = _collected_attempt_path(raw_root.path) / "node=host1"
    waiting = collected_node / "w00000000-00000009"
    acknowledged = collected_node / "w00000010-00000019"
    source_node = _attempt_path(raw_root.path) / "node=host1"
    intent = source_node / "_spool" / "publish.json"
    intent.parent.mkdir(parents=True)
    intent.write_bytes((waiting / "manifest.json").read_bytes())
    original_files = {path.name: path.read_bytes() for path in waiting.iterdir()}
    old = time.time() - 7200
    for directory in [waiting, acknowledged]:
        os.utime(directory, (old, old))
    queries = len(session.queries)

    assert collector.collect_cycle(raw_root) == []

    assert {path.name: path.read_bytes() for path in waiting.iterdir()} == original_files
    assert not acknowledged.exists()
    assert intent.exists()
    assert len(session.queries) == queries
    intent.unlink()

    assert collector.collect_cycle(raw_root) == []

    assert not waiting.exists()
    assert len(session.queries) == queries


def test_shards_split_node_windows_by_node_and_only_shard_zero_collects_logged_windows(
    raw_root, target, fake_session_factory, write_window, write_logged_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    nodes = ["host1", "host2", "host3", "host4"]
    owner = {node: zlib.crc32(node.encode("utf-8")) % 2 for node in nodes}
    assert set(owner.values()) == {0, 1}
    windows = {
        node: write_window(raw_root.path, "expA", "run1", "0", node, 0, 9, families=(Family.SCALARS,))
        for node in nodes
    }
    logged = write_logged_window(raw_root.path, "expA", "run1", "0", 0, 9)
    sessions = []

    for index in (1, 0):
        session = fake_session_factory(stats={"scalars": (10, 0, 9), "logged": (10, 0, 9)})
        sessions.append(session)
        Collector(session, target, collected_retention_s=10**9, shard=Shard(index, 2)).collect_cycle(raw_root)

        assert {node for node, window in windows.items() if window.exists()} == {
            node for node in nodes if owner[node] < index
        }
        assert logged.exists() == (index == 1)

    manifests = [
        [json.loads(_literal_content(sql))["source_windows"] for sql, _ in session.queries if "/batches/" in sql]
        for session in sessions
    ]
    assert manifests == [
        [sorted(f"node={node}/w00000000-00000009" for node in nodes if owner[node] == 1)],
        [sorted(["logged/w00000000-00000009",
                 *(f"node={node}/w00000000-00000009" for node in nodes if owner[node] == 0)])],
    ]


def test_a_pending_batch_of_any_shard_keeps_its_attempt_from_expiring(
    raw_root, target, fake_session_factory, write_metrics_registry,
) -> None:
    registry = write_metrics_registry(raw_root.path, "expA", "run1", "0")
    collected = _collected_attempt_path(raw_root.path) / "node=host1" / "w00000000-00000009"
    collected.mkdir(parents=True)
    old = time.time() - 7200
    os.utime(collected, (old, old))
    journal = registry.parent / "_collector_batch.1.json"
    journal.write_text(json.dumps({"source_windows": [], "receipts": []}))
    primary = Collector(fake_session_factory(), target, collected_retention_s=3600)

    primary.collect_cycle(raw_root)

    assert journal.exists() and collected.exists()

    Collector(fake_session_factory(), target, collected_retention_s=3600, shard=Shard(1, 2)).collect_cycle(raw_root)

    assert not journal.exists() and collected.exists()

    primary.collect_cycle(raw_root)

    assert not collected.exists()


def test_only_shard_zero_collects_dumps_and_expires_collected_windows(
    raw_root, target, dump_target, fake_session_factory, write_dump,
) -> None:
    root = raw_root.path
    dump = write_dump(root, "expA", "run1", "0", step=100, rank=1)
    collected = _collected_attempt_path(root) / "node=host1" / "w00000000-00000009"
    collected.mkdir(parents=True)
    old = time.time() - 7200
    os.utime(collected, (old, old))
    session = fake_session_factory(dump_stats={(100, 1): _DUMP_STATS})

    Collector(
        session, target, collected_retention_s=3600, dump_target=dump_target, shard=Shard(1, 2)
    ).collect_cycle(raw_root)

    assert session.queries == [] and dump.exists() and collected.exists()

    Collector(session, target, collected_retention_s=3600, dump_target=dump_target).collect_cycle(raw_root)

    assert not dump.exists() and not collected.exists()


def _merged_families_batch(
    raw_root, write_window, write_metrics_registry, families=(Family.EXPERT_COUNTS, Family.ROUTER_VECTORS)
) -> dict[str, Path]:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    return {
        "node=host2/w00000000-00000009": write_window(
            raw_root.path, "expA", "run1", "0", "host2", 0, 9, families=families, row_group_size=4
        ),
        "node=host1/w00000000-00000009": write_window(
            raw_root.path, "expA", "run1", "0", "host1", 0, 9, families=families
        ),
        "node=host3/w00000010-00000019": write_window(
            raw_root.path, "expA", "run1", "0", "host3", 10, 19, families=families, row_group_size=5
        ),
    }


_ROUTER_ORDER = ("metric_id", "source_metric_id", "layer", "microbatch", "rank", "step")


def _metric_major(table: pa.Table) -> pa.Table:
    """The rows by metric, source metric, layer, microbatch, rank, step; equal keys in their given order."""
    rows = sorted(table.to_pylist(), key=lambda row: tuple(row[name] for name in _ROUTER_ORDER))
    return pa.Table.from_pylist(rows, schema=table.schema)


def _row_groups(path: Path) -> list[pa.Table]:
    window = pq.ParquetFile(path)
    return [window.read_row_group(index) for index in range(window.num_row_groups)]


@pytest.mark.parametrize("family", [Family.EXPERT_COUNTS, Family.ROUTER_VECTORS])
def test_merged_families_upload_the_windows_row_group_by_row_group_as_the_nodes_values_in_byte_streams(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry, scratch_dir, family,
) -> None:
    windows = _merged_families_batch(raw_root, write_window, write_metrics_registry, families=(family,))
    node_order = pa.concat_tables(
        group
        for relative_dir in sorted(windows)
        for group in _row_groups(windows[relative_dir] / f"{family.value}.parquet")
    )
    expected = pa.concat_tables(
        _metric_major(group) if family is Family.ROUTER_VECTORS else group
        for relative_dir in sorted(windows)
        for group in _row_groups(windows[relative_dir] / f"{family.value}.parquet")
    )
    assert expected.equals(node_order) is (family is Family.EXPERT_COUNTS)
    session = fake_session_factory()

    receipts = Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root)

    attempt = _attempt_dir(raw_root.path)
    key = target.data_key(family.value, attempt, 0, 19, _expected_batch_hash(list(windows)))
    assert receipts == [Receipt(family.value, key, 30)]
    uploaded = session.objects[target.object_url(key)]
    assert pq.read_table(BytesIO(uploaded)).equals(expected)
    assert pq.read_schema(BytesIO(uploaded)).equals(expected.schema)
    metadata = pq.read_metadata(BytesIO(uploaded))
    assert [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)] == [10, 4, 4, 2, 5, 5]
    for index in range(metadata.num_row_groups):
        columns = {
            metadata.row_group(index).column(position).path_in_schema: metadata.row_group(index).column(position)
            for position in range(metadata.num_columns)
        }
        assert {column.compression for column in columns.values()} == {"ZSTD"}
        assert not any(column.has_dictionary_page for column in columns.values())
        assert {name for name, column in columns.items() if "BYTE_STREAM_SPLIT" in column.encodings} == {
            "values.list.element"
        }
    assert not list(scratch_dir.iterdir())
    assert not any(window.exists() for window in windows.values())


def test_the_key_of_a_merged_family_names_the_steps_of_its_data_and_not_of_its_window_names(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    write_window(
        raw_root.path, "expA", "run1", "0", "host1", 0, 49, families=(Family.ROUTER_VECTORS,), data_steps=(5, 30)
    )
    write_window(
        raw_root.path, "expA", "run1", "0", "host2", 0, 49, families=(Family.ROUTER_VECTORS,), data_steps=(12, 40)
    )

    (receipt,) = Collector(fake_session_factory(), target, collected_retention_s=10**9).collect_cycle(raw_root)

    assert receipt.key.rsplit("/", 1)[1].startswith("part-00000005-00000040-")
    assert receipt.rows == 26 + 29


def test_the_merge_goes_to_the_scratch_directory_given_under_the_shards_own_name(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry, tmp_path, scratch_dir,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    node = next(node for node in (f"host{index}" for index in range(50)) if zlib.crc32(node.encode()) % 2 == 1)
    write_window(raw_root.path, "expA", "run1", "0", node, 0, 9, families=(Family.EXPERT_COUNTS,))
    session = fake_session_factory()
    collector = Collector(
        session, target, collected_retention_s=10**9, shard=Shard(1, 2), scratch_dir=tmp_path / "local" / "disk"
    )

    (receipt,) = collector.collect_cycle(raw_root)

    merged = tmp_path / "local" / "disk" / "metric-sampler-collector.1.expert_counts.parquet"
    assert f"SELECT * FROM file('{merged}', 'RawBLOB')" in next(
        sql for sql, _ in session.queries if "family=expert_counts" in sql
    )
    assert receipt.rows == 10
    assert list(merged.parent.iterdir()) == []
    assert not scratch_dir.exists()


@pytest.mark.parametrize("family", [Family.EXPERT_COUNTS, Family.ROUTER_VECTORS])
def test_a_failed_upload_leaves_no_merged_file_and_the_restart_uploads_the_same_object(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry, scratch_dir, family,
) -> None:
    windows = _merged_families_batch(raw_root, write_window, write_metrics_registry, families=(family,))
    failed = fake_session_factory()

    class InterruptedSession:
        def query(self, sql, fmt=None):
            result = failed.query(sql, fmt)
            if sql.startswith("INSERT INTO FUNCTION s3(") and f"family={family.value}/" in sql:
                raise OSError("upload response lost")
            return result

    assert Collector(InterruptedSession(), target, collected_retention_s=10**9).collect_cycle(raw_root) == []

    assert not list(scratch_dir.iterdir())
    assert all(window.exists() for window in windows.values())
    journal = _attempt_path(raw_root.path) / "_collector_batch.json"
    assert json.loads(journal.read_bytes())["receipts"] is None
    assert len(failed.objects) == 1
    resumed = fake_session_factory()

    receipts = Collector(resumed, target, collected_retention_s=10**9).collect_cycle(raw_root)

    assert [(receipt.family, receipt.rows) for receipt in receipts] == [(family.value, 30)]
    assert resumed.objects == failed.objects
    assert not list(scratch_dir.iterdir()) and not journal.exists()
    assert not any(window.exists() for window in windows.values())


def test_windows_that_do_not_share_a_schema_fail_the_attempt_without_an_upload_or_a_merged_file(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry, scratch_dir, caplog,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    write_window(raw_root.path, "expA", "run1", "0", "host1", 0, 9, families=(Family.EXPERT_COUNTS,))
    other = write_window(raw_root.path, "expA", "run1", "0", "host2", 0, 9, families=(Family.EXPERT_COUNTS,))
    pq.write_table(
        pa.table({"step": pa.array([0], type=pa.uint32()), "values": [[0.5]]}), other / "expert_counts.parquet"
    )
    session = fake_session_factory()

    with caplog.at_level(logging.ERROR, logger="metric_sampler.island.collector"):
        assert Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root) == []

    (record,) = caplog.records
    assert record.getMessage().startswith(f"collector: attempt {other.parents[1]} failed: ValueError: ")
    assert not list(scratch_dir.iterdir())
    assert not any("family=expert_counts" in sql for sql, _ in session.queries)
    assert other.exists()


def test_an_attempt_without_its_registry_is_logged_in_one_line_and_leaves_the_other_attempts_collected(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry, caplog,
) -> None:
    root = raw_root.path
    broken = write_window(root, "expA", "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    write_metrics_registry(root, "expB", "run1", "0")
    healthy = write_window(root, "expB", "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    session = fake_session_factory(stats={"scalars": (10, 0, 9)})
    collector = Collector(session, target, collected_retention_s=10**9)

    for _ in range(2):
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="metric_sampler.island.collector"):
            collector.collect_cycle(raw_root)

        (record,) = caplog.records
        assert record.exc_info is None
        assert record.getMessage().startswith(
            f"collector: attempt {broken.parents[1]} failed: FileNotFoundError: "
        ) and "metrics.json" in record.getMessage()
        assert not any("expA" in sql for sql, _ in session.queries if sql.startswith("INSERT INTO FUNCTION s3("))
        assert broken.exists()

    assert not healthy.exists()
    journal = broken.parents[1] / "_collector_batch.json"
    assert json.loads(journal.read_bytes()) == {"source_windows": ["node=host1/w00000000-00000009"], "receipts": None}


def test_an_attempt_whose_windows_cannot_be_listed_is_logged_and_skipped(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry, caplog, monkeypatch,
) -> None:
    root = raw_root.path
    for exp in ["expA", "expB"]:
        write_metrics_registry(root, exp, "run1", "0")
        write_window(root, exp, "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    real_listing = AttemptDir.complete_windows

    def complete_windows(attempt):
        if attempt.exp == "expA":
            raise PermissionError(13, "Permission denied", str(attempt.path / "node=host1"))
        return real_listing(attempt)

    monkeypatch.setattr(AttemptDir, "complete_windows", complete_windows)
    session = fake_session_factory(stats={"scalars": (10, 0, 9)})

    with caplog.at_level(logging.ERROR, logger="metric_sampler.island.collector"):
        receipts = Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root)

    assert [receipt.key.split("exp=")[1].split("/")[0] for receipt in receipts] == ["expB"]
    (record,) = caplog.records
    assert record.getMessage().startswith(
        f"collector: attempt {_attempt_path(root)} failed: PermissionError: "
    )


def test_the_failure_line_is_short_and_carries_no_credentials(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry, caplog,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    window = write_window(raw_root.path, "expA", "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    session = fake_session_factory(stats={"scalars": (10, 0, 9)})

    class RefusingSession:
        def query(self, sql, fmt=None):
            if sql.startswith("INSERT INTO FUNCTION s3(") and "family=scalars/" in sql:
                raise RuntimeError(
                    f"Code: 499. DB::Exception: failed\nquery: {sql}\n" + "padding " * 100
                )
            return session.query(sql, fmt)

    with caplog.at_level(logging.ERROR, logger="metric_sampler.island.collector"):
        assert Collector(RefusingSession(), target, collected_retention_s=10**9).collect_cycle(raw_root) == []

    (record,) = caplog.records
    prefix = (
        f"collector: attempt {window.parents[1]} failed: RuntimeError: Code: 499. DB::Exception: failed query: "
        "INSERT INTO FUNCTION s3("
    )
    message = record.getMessage()
    assert message.startswith(prefix)
    assert "\n" not in message and len(message) == len(f"collector: attempt {window.parents[1]} failed: ") + 300
    assert "test-key" not in message and "test-secret" not in message


def test_credentials_are_redacted_whole_even_when_the_line_is_cut(target, fake_session_factory) -> None:
    credentials = target.credentials
    collector = Collector(fake_session_factory(), target, collected_retention_s=10**9)

    summary = collector._summary(
        OSError(f"{credentials.access_key} and {credentials.secret_key}\nand\t more " + "x" * 400)
    )

    assert summary.startswith("OSError: *** and *** and more xxx") and len(summary) == 300


def test_router_vectors_rows_are_written_metric_major_within_each_row_group_and_ties_keep_the_nodes_order(
    raw_root, target, fake_session_factory, write_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    windows = [
        write_window(raw_root.path, "expA", "run1", "0", host, 0, 9, families=(Family.ROUTER_VECTORS,))
        for host in ["host1", "host2"]
    ]
    schema = pq.read_schema(windows[0] / "router_vectors.parquet")

    def row(step, metric_id, layer, rank, value, microbatch=0, source_metric_id=0):
        return {
            "node": "n", "step": step, "rank": rank, "microbatch": microbatch, "layer": layer, "metric_id": metric_id,
            "source_metric_id": source_metric_id, "contributors": 1, "expected_contributors": 1, "identical": False,
            "verbose": False, "values": [value],
        }

    # The node's order is step first; these rows repeat their keys step after step, with a tie (same key twice).
    node_rows = {
        "host1": [
            row(0, 2, 1, 0, 1.0), row(0, 1, 1, -1, 2.0), row(0, 1, 0, 0, 3.0, microbatch=1), row(0, 1, 0, 0, 4.0),
            row(1, 2, 1, 0, 5.0), row(1, 1, 1, -1, 6.0), row(1, 1, 0, 0, 7.0, microbatch=1), row(1, 1, 0, 0, 8.0),
            row(1, 1, 0, 0, 9.0), row(1, 1, 0, 0, 10.0, source_metric_id=3),
        ],
        "host2": [row(0, 3, 0, 5, 11.0), row(0, 1, 2, 5, 12.0), row(1, 3, 0, 4, 13.0), row(1, 1, 2, 5, 14.0)],
    }
    for window, host in zip(windows, node_rows):
        pq.write_table(pa.Table.from_pylist(node_rows[host], schema=schema), window / "router_vectors.parquet")
    session = fake_session_factory()

    (receipt,) = Collector(session, target, collected_retention_s=10**9).collect_cycle(raw_root)

    uploaded = pq.read_table(BytesIO(session.objects[target.object_url(receipt.key)]))
    assert [row["values"][0] for row in uploaded.to_pylist()] == [
        # host1: metric 1 first (source 0, layer 0: microbatch 0 then 1, the tie in node order), then layer 1
        # (rank -1 before the others), metric 1 with source metric 3, then metric 2.
        4.0, 8.0, 9.0, 3.0, 7.0, 2.0, 6.0, 10.0, 1.0, 5.0,
        # host2: metric 1, then metric 3 by rank and step.
        12.0, 14.0, 13.0, 11.0,
    ]
    assert receipt.rows == 14 and uploaded.schema.equals(schema)
