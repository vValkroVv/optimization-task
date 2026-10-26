from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Protocol

import pyarrow.compute as pc
import pyarrow.parquet as pq

from metric_sampler.contract.metrics import Family
from metric_sampler.contract.raw_layout import (
    COLLECTED_DIR_NAME,
    LOGGED_DATA_NAME,
    PUBLISH_INTENT_NAME,
    SPOOL_DIR_NAME,
    AttemptDir,
    Dump,
    LoggedWindow,
    RawRoot,
    Window,
    attempt_dir_name,
    family_file_name,
    window_name,
    write_atomically,
)
from metric_sampler.island.s3_target import S3Target

_BATCH_MANIFEST_SCHEMA_VERSION = 1
_BATCH_HASH_LENGTH = 12
_LOGGED_FAMILY = "logged"
_FAILURE_LINE_LENGTH = 300

# Large batches keep the per-object round trips (row count checks, manifests) small next to the bytes they move.
# A merged expert_counts or router_vectors object is read back whole to be uploaded (chdb's RawBLOB holds the file in
# memory, a few times its size), which this bound keeps well inside the session's memory limit.
_BATCH_BYTES = 2 * 1024**3
_BATCH_WINDOWS = 128
_BATCH_JOURNAL = "_collector_batch.json"
_BATCH_JOURNAL_GLOB = "_collector_batch*.json"

# One connection to a high-latency endpoint moves only a few MiB/s; a large object uploads fast only with many parts
# in flight at once.
_S3_PART_BYTES = 8 * 1024**2
_S3_PARTS_IN_FLIGHT = 128
_S3_UPLOAD_SETTINGS = (
    f"s3_min_upload_part_size={_S3_PART_BYTES}, s3_max_single_part_upload_size={_S3_PART_BYTES}, "
    f"s3_max_inflight_parts_for_one_file={_S3_PARTS_IN_FLIGHT}"
)
# Re-encoded windows carry no bloom filters or dictionaries, which only add bytes for value arrays.
_PARQUET_ENCODE_SETTINGS = (
    "output_format_parquet_compression_method='zstd', output_format_parquet_write_bloom_filter=0, "
    "output_format_parquet_max_dictionary_size=0"
)

# The families that pyarrow merges, because chdb would encode their value arrays without byte-stream-split, which
# makes them larger than the node files; pyarrow writes the values' leaf column split into byte streams, with no
# dictionary, as the node does.
# Nodes write rows step first. The rows of a router_vectors window are written metric first and step last, so the
# arrays of one series (metric, layer, microbatch, rank) follow one another over the window's steps and compress
# better; rows with equal keys keep the node's order (the sort is stable). expert_counts compress worse reordered and
# stay in the node's order.
_MERGE_ORDERS = {
    Family.EXPERT_COUNTS: [],
    Family.ROUTER_VECTORS: [
        ("metric_id", "ascending"), ("source_metric_id", "ascending"), ("layer", "ascending"),
        ("microbatch", "ascending"), ("rank", "ascending"), ("step", "ascending"),
    ],
}
_VALUES_COLUMN = "values.list.element"
_MERGE_ZSTD_LEVEL = 3

# Dumps go out at a bounded rate, which leaves the shared S3 endpoint's bandwidth to the metric batches.
_DUMP_BYTES_PER_CYCLE = 512 * 1024**2
_DUMP_INTERVAL_S = 60

_SOURCE_WINDOW = re.compile(r"(?:node=(?P<node>[^/\\]+)|logged)/w(?P<first_step>\d{8})-(?P<last_step>\d{8})")

log = logging.getLogger(__name__)


class QueryResult(Protocol):
    def data(self) -> str: ...


class Session(Protocol):
    """The part of a chdb session that the collector uses."""

    def query(self, sql: str, fmt: str = "CSV") -> QueryResult: ...


class CollectError(Exception):
    """Raised when a batch fails a consistency check: a row count, a journal entry or an immutable attempt file."""


@dataclass(frozen=True)
class Receipt:
    """A data object uploaded for a batch: its family, its S3 key and the number of source rows it holds."""

    family: str
    key: str
    rows: int


@dataclass(frozen=True)
class Shard:
    """This collector's part of a raw root that `count` collectors share.

    A node's windows belong to shard crc32(node) % count, so they always go to the same collector.
    Shard 0 also collects the logged windows and dumps and expires collected windows. Change `count`
    only while no shard has a batch journal pending: a journaled window could otherwise go to two shards.
    """

    index: int = 0
    count: int = 1

    def owns(self, window: Window) -> bool:
        return zlib.crc32(window.node.encode("utf-8")) % self.count == self.index

    @property
    def primary(self) -> bool:
        return self.index == 0

    @property
    def journal_name(self) -> str:
        return _BATCH_JOURNAL if self.primary else f"_collector_batch.{self.index}.json"

    @property
    def lock_name(self) -> str:
        return ".collector.lock" if self.primary else f".collector.{self.index}.lock"


# The only collector of its raw root.
_SOLE_SHARD = Shard()


@dataclass(frozen=True)
class _DumpStats:
    rows: int
    points: list[str]
    layers: list[int]
    tokens: int


@dataclass(frozen=True)
class _Batch:
    # Either list may be empty, but not both (the caller only builds a batch when at least one
    # kind of window exists).
    node_windows: list[Window]
    logged_windows: list[LoggedWindow]

    @cached_property
    def all_windows(self) -> list[Window | LoggedWindow]:
        return [*self.node_windows, *self.logged_windows]

    @cached_property
    def source_windows(self) -> list[str]:
        return sorted(window.relative_dir for window in self.all_windows)

    @cached_property
    def batch_hash(self) -> str:
        digest = hashlib.sha1("\n".join(self.source_windows).encode("utf-8"))
        return digest.hexdigest()[:_BATCH_HASH_LENGTH]

    @cached_property
    def nodes(self) -> list[str]:
        return sorted({window.node for window in self.node_windows})

    @cached_property
    def first_step(self) -> int:
        return min(window.first_step for window in self.all_windows)

    @cached_property
    def last_step(self) -> int:
        return max(window.last_step for window in self.all_windows)

    def windows_with_family(self, family: Family) -> list[Window]:
        return [window for window in self.node_windows if window.family_path(family.value).exists()]


class Collector:
    """Batches complete node aggregator windows into S3 as Parquet, with manifests."""

    def __init__(
        self,
        session: Session,
        target: S3Target,
        collected_retention_s: float,
        dump_target: S3Target | None = None,
        shard: Shard = _SOLE_SHARD,
        scratch_dir: Path | None = None,
    ) -> None:
        self._session = session
        self._target = target
        self._collected_retention_s = collected_retention_s
        self._dump_target = dump_target
        self._shard = shard
        self._scratch_dir = Path(tempfile.gettempdir()) if scratch_dir is None else scratch_dir
        self._dump_target_warned = False
        self._next_dump_at = 0.0
        self._uploaded_file_sha1: dict[tuple[str, str, str, str], str] = {}

    def collect_cycle(self, raw_root: RawRoot) -> list[Receipt]:
        receipts = []
        for attempt in raw_root.attempt_dirs():
            # A broken attempt (a missing or unreadable registry, a corrupt window, a failed batch) is retried
            # next cycle with its journal and windows as they are, and must not block the other attempts.
            try:
                node_windows = [window for window in attempt.complete_windows() if self._shard.owns(window)]
                logged_windows = attempt.complete_logged_windows() if self._shard.primary else []
                if not node_windows and not logged_windows and not (attempt.path / self._shard.journal_name).exists():
                    continue
                receipts.extend(self._collect_attempt(raw_root, attempt, node_windows, logged_windows))
            except Exception as error:
                log.error("collector: attempt %s failed: %s", attempt.path, self._summary(error))
        if self._shard.primary:
            self._collect_dumps(raw_root)
            self._delete_expired(raw_root)
        return receipts

    def _summary(self, error: Exception) -> str:
        """One short line for the log: an S3 or chdb error can quote the statement, and so the credentials of
        either target."""
        text = " ".join(f"{type(error).__name__}: {error}".split())
        for target in filter(None, (self._target, self._dump_target)):
            for secret in filter(None, (target.credentials.access_key, target.credentials.secret_key)):
                text = text.replace(secret, "***")
        return text[:_FAILURE_LINE_LENGTH]

    def _collect_dumps(self, raw_root: RawRoot) -> None:
        dump_target = self._dump_target
        if dump_target is None:
            if not self._dump_target_warned:
                log.info("collector: no dump target configured, activation dumps stay on NFS")
                self._dump_target_warned = True
            return

        if time.monotonic() < self._next_dump_at:
            return

        uploaded_bytes = 0
        for attempt, dump in self._readable_dumps(raw_root):
            if uploaded_bytes >= _DUMP_BYTES_PER_CYCLE:
                break
            try:
                uploaded_bytes += self._collect_dump(raw_root, dump_target, attempt, dump)
            except Exception as error:
                log.error("collector: dump %s failed: %s", dump.path, self._summary(error))
        self._next_dump_at = time.monotonic() + _DUMP_INTERVAL_S

    @staticmethod
    def _readable_dumps(raw_root: RawRoot) -> list[tuple[AttemptDir, Dump]]:
        """The dumps the collector can read, the oldest step first."""
        readable = []
        unreadable = []
        for attempt in raw_root.attempt_dirs():
            for dump in attempt.complete_dumps():
                if os.access(dump.path, os.R_OK):
                    readable.append((attempt, dump))
                else:
                    unreadable.append(dump.path)
        # Another user's dump can stay unreadable for days; chdb would fail on it with a traceback every
        # cycle. It stays in place, and the next cycle checks it again.
        if unreadable:
            log.warning("collector: skipping %d unreadable dumps, first %s", len(unreadable), unreadable[0])

        def oldest_step_first(candidate: tuple[AttemptDir, Dump]) -> tuple[int, str, str, str, int]:
            attempt, dump = candidate
            return dump.step, attempt.exp, attempt.run, attempt.attempt, dump.rank

        readable.sort(key=oldest_step_first)
        return readable

    def _collect_dump(self, raw_root: RawRoot, dump_target: S3Target, attempt: AttemptDir, dump: Dump) -> int:
        stats = self._dump_stats(dump.path)
        size = dump.path.stat().st_size

        key = dump_target.dump_key(attempt, dump)
        self._session.query(
            f"INSERT INTO FUNCTION {dump_target.table_function(key, 'RawBLOB')} "
            f"SELECT * FROM file('{dump.path}', 'RawBLOB') "
            f"SETTINGS s3_truncate_on_insert=1, use_hive_partitioning=0, {_S3_UPLOAD_SETTINGS}"
        )

        uploaded_rows = self._count_rows(
            f"SELECT count() FROM {dump_target.table_function(key, 'Parquet', 'step Int64')} "
            "SETTINGS use_hive_partitioning=0"
        )
        if uploaded_rows != stats.rows:
            raise CollectError(f"row count mismatch for dump {key}: uploaded {uploaded_rows}, source {stats.rows}")

        self._upload_dump_index(dump_target, attempt, dump, stats, key, size)
        destination = attempt.collected_dump_path(raw_root, dump)
        os.renames(dump.path, destination)
        # The shared directory is writable even when another user's dump is read-only.
        os.utime(destination.parent, None)
        return size

    def _dump_stats(self, path: Path) -> _DumpStats:
        answer = self._session.query(
            "SELECT count() AS rows, arraySort(groupUniqArray(point)) AS points, "
            "arraySort(groupUniqArray(layer)) AS layers, uniqExact(token_index) AS tokens "
            f"FROM file('{path}', 'Parquet') SETTINGS use_hive_partitioning=0",
            "JSONEachRow",
        ).data()
        parsed = json.loads(answer)
        return _DumpStats(
            rows=int(parsed["rows"]),
            points=list(parsed["points"]),
            layers=[int(layer) for layer in parsed["layers"]],
            tokens=int(parsed["tokens"]),
        )

    def _upload_dump_index(
        self, dump_target: S3Target, attempt: AttemptDir, dump: Dump, stats: _DumpStats, key: str, size: int
    ) -> None:
        manifest = {
            "exp": attempt.exp,
            "run": attempt.run,
            "attempt": attempt.attempt,
            "step": dump.step,
            "rank": dump.rank,
            "rows": stats.rows,
            "bytes": size,
            "points": stats.points,
            "layers": stats.layers,
            "tokens": stats.tokens,
            "s3_key": f"{dump_target.bucket}/{key}",
        }
        self._upload_bytes(self._target.dump_index_key(attempt, dump), json.dumps(manifest, indent=2).encode("utf-8"))

    def _collect_attempt(
        self,
        raw_root: RawRoot,
        attempt: AttemptDir,
        node_windows: list[Window],
        logged_windows: list[LoggedWindow],
    ) -> list[Receipt]:
        journal = attempt.path / self._shard.journal_name
        if journal.exists():
            batch, receipts = self._read_batch(attempt, journal)
        else:
            batch = self._select_batch(node_windows, logged_windows)
            receipts = None
            self._write_batch(journal, batch, receipts)

        if receipts is None:
            receipts = self._upload_batch(attempt, batch)
            self._write_batch(journal, batch, receipts)
        self._move_collected(raw_root, attempt, batch.all_windows)
        journal.unlink()
        return receipts

    def _upload_batch(self, attempt: AttemptDir, batch: _Batch) -> list[Receipt]:
        self._upload_metrics_registry(attempt)
        self._upload_streams_registry(attempt)
        self._upload_monitor_parameters(attempt)
        self._upload_monitor_expert_ownership(attempt)

        receipts = []
        for family in Family:
            family_windows = batch.windows_with_family(family)
            if not family_windows:
                continue
            file_name = family_file_name(family.value)
            if family in _MERGE_ORDERS:
                receipts.append(
                    self._upload_merged_parquet(
                        attempt, batch, family.value, file_name, family_windows, _MERGE_ORDERS[family]
                    )
                )
            else:
                receipts.append(self._upload_parquet(attempt, batch, family.value, file_name, family_windows))
        if batch.logged_windows:
            receipts.append(
                self._upload_parquet(attempt, batch, _LOGGED_FAMILY, LOGGED_DATA_NAME, batch.logged_windows)
            )
        self._verify_uploaded_rows(receipts)

        self._upload_node_manifests(attempt, batch)
        self._upload_batch_manifest(attempt, batch, receipts)
        return receipts

    def _select_batch(self, node_windows: list[Window], logged_windows: list[LoggedWindow]) -> _Batch:
        windows: list[Window | LoggedWindow] = [*node_windows, *logged_windows]
        selected: list[Window | LoggedWindow] = []
        selected_bytes = 0
        for window in sorted(
            windows,
            key=lambda candidate: (candidate.first_step, candidate.last_step, candidate.relative_dir),
        ):
            if len(selected) >= _BATCH_WINDOWS:
                break
            window_bytes = sum(path.stat().st_size for path in window.path.glob("*.parquet"))
            # A complete window is indivisible; an oversized first window must still progress.
            if selected and selected_bytes + window_bytes > _BATCH_BYTES:
                break
            selected.append(window)
            selected_bytes += window_bytes
        return _Batch(
            node_windows=[window for window in selected if isinstance(window, Window)],
            logged_windows=[window for window in selected if isinstance(window, LoggedWindow)],
        )

    def _read_batch(self, attempt: AttemptDir, journal: Path) -> tuple[_Batch, list[Receipt] | None]:
        saved = json.loads(journal.read_bytes())
        node_windows = []
        logged_windows = []
        for source in saved["source_windows"]:
            match = _SOURCE_WINDOW.fullmatch(source)
            path = attempt.path / source
            if match is None or not path.resolve().is_relative_to(attempt.path.resolve()):
                raise CollectError(f"invalid source window in {journal}: {source}")
            first_step, last_step = int(match["first_step"]), int(match["last_step"])
            if match["node"] is None:
                logged_windows.append(LoggedWindow(path, first_step, last_step))
            else:
                node_windows.append(Window(path, match["node"], first_step, last_step))
        saved_receipts = saved["receipts"]
        receipts = None if saved_receipts is None else [
            Receipt(family=item["family"], key=item["key"], rows=item["rows"]) for item in saved_receipts
        ]
        return _Batch(node_windows, logged_windows), receipts

    def _write_batch(self, journal: Path, batch: _Batch, receipts: list[Receipt] | None) -> None:
        saved = {
            "source_windows": batch.source_windows,
            "receipts": None if receipts is None else [
                {"family": receipt.family, "key": receipt.key, "rows": receipt.rows} for receipt in receipts
            ],
        }
        write_atomically(journal, json.dumps(saved).encode("utf-8"))

    def _upload_parquet(
        self,
        attempt: AttemptDir,
        batch: _Batch,
        family_name: str,
        file_name: str,
        windows: list[Window] | list[LoggedWindow],
    ) -> Receipt:
        """Uploads the family's windows as one object; the receipt carries the source row count."""
        source = self._file_source(attempt.path, file_name, windows)
        source_rows, first_step, last_step = self._row_stats(source)

        key = self._target.data_key(family_name, attempt, first_step, last_step, batch.batch_hash)
        upload_source, upload_format = source, "Parquet"
        if len(windows) == 1:
            # A singleton is already valid Parquet; preserve its bytes instead of encoding it again.
            upload_source = f"file('{windows[0].path / file_name}', 'RawBLOB')"
            upload_format = "RawBLOB"
        self._insert_data_object(key, upload_format, upload_source)
        return Receipt(family=family_name, key=key, rows=source_rows)

    def _upload_merged_parquet(
        self,
        attempt: AttemptDir,
        batch: _Batch,
        family_name: str,
        file_name: str,
        windows: list[Window],
        order: list[tuple[str, str]],
    ) -> Receipt:
        """Merges the windows into a local file with pyarrow and uploads its bytes as they are."""
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        # One name per shard and family: a file a killed process left behind is overwritten by the next
        # merge of that family instead of piling up.
        merged = self._scratch_dir / f"metric-sampler-collector.{self._shard.index}.{family_name}.parquet"
        try:
            source_rows, first_step, last_step = self._merge_windows(windows, file_name, merged, order)
            key = self._target.data_key(family_name, attempt, first_step, last_step, batch.batch_hash)
            self._insert_data_object(key, "RawBLOB", f"file('{merged}', 'RawBLOB')")
        finally:
            merged.unlink(missing_ok=True)
        return Receipt(family=family_name, key=key, rows=source_rows)

    @staticmethod
    def _merge_windows(
        windows: list[Window], file_name: str, destination: Path, order: list[tuple[str, str]]
    ) -> tuple[int, int, int]:
        """Writes the windows' row groups one after another, in the windows' sorted order and with the node's column
        types, one row group in memory at a time (twice over while it is sorted). Each row group's rows are sorted
        by `order`, or stay as the node wrote them if it is empty. Returns the row count and the first and last
        step."""
        paths = [window.path / file_name for window in sorted(windows, key=lambda window: window.relative_dir)]
        rows = 0
        first_steps = []
        last_steps = []
        with pq.ParquetWriter(
            destination, pq.read_schema(paths[0]), compression="zstd", compression_level=_MERGE_ZSTD_LEVEL,
            use_dictionary=False, use_byte_stream_split=[_VALUES_COLUMN],
        ) as writer:
            for path in paths:
                with pq.ParquetFile(path) as parquet_file:
                    for row_group_index in range(parquet_file.num_row_groups):
                        table = parquet_file.read_row_group(row_group_index)
                        if order:
                            table = table.sort_by(order)
                        writer.write_table(table)
                        rows += table.num_rows
                        steps = pc.min_max(table["step"])
                        first_steps.append(steps["min"].as_py())
                        last_steps.append(steps["max"].as_py())
        return rows, min(first_steps), max(last_steps)

    def _insert_data_object(self, key: str, upload_format: str, upload_source: str) -> None:
        self._session.query(
            f"INSERT INTO FUNCTION {self._target.table_function(key, upload_format)} "
            f"SELECT * FROM {upload_source} "
            f"SETTINGS s3_truncate_on_insert=1, {_PARQUET_ENCODE_SETTINGS}, "
            f"use_hive_partitioning=0, {_S3_UPLOAD_SETTINGS}"
        )

    def _verify_uploaded_rows(self, receipts: list[Receipt]) -> None:
        if not receipts:
            return
        # One statement for the batch: ClickHouse reads the footers of its UNION ALL branches at the same time,
        # where one count per object would wait out the round trips one after another. An explicit structure skips
        # schema inference, which reads the footer a second time; count() still comes from the row counts in the
        # uploaded object's footer. Each count stays a query of its own, without the item column beside it, which
        # is the shape ClickHouse answers from the footer.
        branches = " UNION ALL ".join(
            f"SELECT {index} AS item, rows FROM (SELECT count() AS rows FROM "
            f"{self._target.table_function(receipt.key, 'Parquet', 'step UInt32')})"
            for index, receipt in enumerate(receipts)
        )
        answer = self._session.query(
            f"SELECT item, rows FROM ({branches}) SETTINGS use_hive_partitioning=0", "CSV"
        ).data()
        uploaded_rows = {}
        for line in answer.splitlines():
            item, rows = line.split(",")
            uploaded_rows[int(item)] = int(rows)
        for index, receipt in enumerate(receipts):
            if uploaded_rows[index] != receipt.rows:
                raise CollectError(
                    f"row count mismatch for {receipt.key}: uploaded {uploaded_rows[index]}, source {receipt.rows}"
                )

    def _upload_node_manifests(self, attempt: AttemptDir, batch: _Batch) -> None:
        if not batch.node_windows:
            return
        # One object per batch, because every upload waits at least one round trip. The ingest reads each JSON
        # document as a row.
        manifests = b"\n".join(window.manifest_path.read_bytes() for window in batch.node_windows)
        key = self._target.node_manifests_key(attempt, batch.first_step, batch.last_step, batch.batch_hash)
        self._upload_bytes(key, manifests)

    def _upload_batch_manifest(self, attempt: AttemptDir, batch: _Batch, receipts: list[Receipt]) -> None:
        manifest = {
            "schema_version": _BATCH_MANIFEST_SCHEMA_VERSION,
            "exp": attempt.exp,
            "run": attempt.run,
            "attempt": attempt.attempt,
            "batch": batch.batch_hash,
            "first_step": batch.first_step,
            "last_step": batch.last_step,
            "source_windows": batch.source_windows,
            "nodes": batch.nodes,
            "rows_per_family": {receipt.family: receipt.rows for receipt in receipts},
            "logged_dropped_rows": sum(
                json.loads(window.manifest_path.read_bytes()).get("dropped_rows", 0)
                for window in batch.logged_windows
            ),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
        key = self._target.batch_manifest_key(attempt, batch.first_step, batch.last_step, batch.batch_hash)
        self._upload_bytes(key, json.dumps(manifest, indent=2).encode("utf-8"))

    def _upload_metrics_registry(self, attempt: AttemptDir) -> None:
        self._upload_attempt_file(attempt, attempt.metrics_path, self._target.metrics_key(attempt))

    def _upload_streams_registry(self, attempt: AttemptDir) -> None:
        if not attempt.streams_path.exists():
            return
        self._upload_attempt_file(attempt, attempt.streams_path, self._target.streams_key(attempt))

    def _upload_monitor_parameters(self, attempt: AttemptDir) -> None:
        if not attempt.monitor_parameters_path.exists():
            return
        self._upload_attempt_file(
            attempt, attempt.monitor_parameters_path, self._target.monitor_parameters_key(attempt), immutable=True
        )

    def _upload_monitor_expert_ownership(self, attempt: AttemptDir) -> None:
        if not attempt.monitor_expert_ownership_path.exists():
            return
        self._upload_attempt_file(
            attempt, attempt.monitor_expert_ownership_path, self._target.monitor_expert_ownership_key(attempt),
            immutable=True,
        )

    def _upload_attempt_file(self, attempt: AttemptDir, path: Path, key: str, *, immutable: bool = False) -> None:
        """Uploads the file unless its last upload had the same content; an immutable file must not change."""
        content = path.read_bytes()
        digest = hashlib.sha1(content).hexdigest()
        uploaded_key = (attempt.exp, attempt.run, attempt.attempt, path.name)
        if self._uploaded_file_sha1.get(uploaded_key) == digest:
            return
        if immutable and uploaded_key in self._uploaded_file_sha1:
            raise CollectError(f"{path.name} changed within an attempt")
        self._upload_bytes(key, content)
        self._uploaded_file_sha1[uploaded_key] = digest

    def _move_collected(self, raw_root: RawRoot, attempt: AttemptDir, windows: list[Window | LoggedWindow]) -> None:
        for window in windows:
            destination = attempt.collected_path(raw_root, window)
            if window.path.exists() or not destination.is_dir():
                os.renames(window.path, destination)
            # os.renames preserves the original mtime, so retention would count from when the
            # window was first written rather than when it was moved into _collected.
            os.utime(destination, None)

    def _delete_expired(self, raw_root: RawRoot) -> None:
        collected_root = raw_root.path / COLLECTED_DIR_NAME
        if not collected_root.is_dir():
            return
        cutoff = time.time() - self._collected_retention_s
        self._delete_expired_windows(raw_root, collected_root, cutoff)
        for dump_dir in collected_root.glob("*/*/attempt=*/dumps/step=*"):
            if dump_dir.is_dir() and dump_dir.stat().st_mtime < cutoff:
                shutil.rmtree(dump_dir)

    def _delete_expired_windows(self, raw_root: RawRoot, collected_root: Path, cutoff: float) -> None:
        pending_attempts = {
            collected_root / attempt.exp / attempt.run / attempt_dir_name(attempt.attempt)
            for attempt in raw_root.attempt_dirs()
            if any(attempt.path.glob(_BATCH_JOURNAL_GLOB))
        }
        for window_dir in collected_root.glob("*/*/attempt=*/*/w*"):
            if window_dir.parent.parent in pending_attempts:
                continue
            if not window_dir.is_dir() or window_dir.stat().st_mtime >= cutoff:
                continue
            source_node = raw_root.path / window_dir.parent.relative_to(collected_root)
            if not self._awaits_publication_ack(window_dir, source_node):
                shutil.rmtree(window_dir)

    @staticmethod
    def _awaits_publication_ack(window_dir: Path, source_node: Path) -> bool:
        """Whether the node still has this window as the one it is publishing, so its collected copy must stay."""
        try:
            intent = json.loads((source_node / SPOOL_DIR_NAME / PUBLISH_INTENT_NAME).read_bytes())
        except FileNotFoundError:
            return False
        return window_dir.name == window_name(intent["first_step"], intent["last_step"])

    def _upload_bytes(self, key: str, content: bytes) -> None:
        escaped = content.decode("utf-8").replace("\\", "\\\\").replace("'", "\\'")
        self._session.query(
            f"INSERT INTO FUNCTION {self._target.table_function(key, 'RawBLOB')} SELECT '{escaped}' "
            "SETTINGS s3_truncate_on_insert=1, use_hive_partitioning=0"
        )

    @staticmethod
    def _file_source(attempt_path: Path, file_name: str, windows: list[Window] | list[LoggedWindow]) -> str:
        relative_dirs = sorted(window.relative_dir for window in windows)
        if len(relative_dirs) == 1:
            pattern = f"{attempt_path}/{relative_dirs[0]}/{file_name}"
        else:
            pattern = f"{attempt_path}/{{{','.join(relative_dirs)}}}/{file_name}"
        return f"file('{pattern}', 'Parquet')"

    def _row_stats(self, source: str) -> tuple[int, int, int]:
        answer = self._session.query(
            f"SELECT count(), min(step), max(step) FROM {source} SETTINGS use_hive_partitioning=0", "CSV"
        ).data()
        count, first_step, last_step = answer.strip().split(",")
        return int(count), int(first_step), int(last_step)

    def _count_rows(self, sql: str) -> int:
        return int(self._session.query(sql, "CSV").data().strip())
