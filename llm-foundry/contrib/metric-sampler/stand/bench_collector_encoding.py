from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TypeVar

import pyarrow as pa
import pyarrow.parquet as pq

from metric_sampler.contract.metrics import Family
from metric_sampler.contract.raw_layout import AttemptDir, LoggedWindow, RawRoot, Window
from metric_sampler.island.cli import SESSION_SETTINGS, _build_session
from metric_sampler.island.collector import (
    _MERGE_ORDERS,
    _PARQUET_ENCODE_SETTINGS,
    _S3_UPLOAD_SETTINGS,
    Collector,
    Receipt,
    Session,
)
from metric_sampler.island.s3_target import S3Credentials, S3Target

# The ingest's column structure per family (metric_sampler/cloud/ingest.sql): the types chdb has to
# decode every candidate to.
_BASE_STRUCTURE = (
    "node String, step UInt32, rank Int16, microbatch Int16, layer Int16, metric_id UInt16, "
    "source_metric_id UInt16, contributors UInt8, expected_contributors UInt8, identical Bool, verbose Bool"
)
_STRUCTURES = {
    "scalars": f"{_BASE_STRUCTURE}, values Array(Float32)",
    "expert_counts": f"{_BASE_STRUCTURE}, alias_metric_id UInt16, values Array(UInt32)",
    "router_vectors": f"{_BASE_STRUCTURE}, values Array(Float32)",
    "logged": "step UInt32, name String, value Float64, source_step Nullable(UInt32)",
}
_NODE_OUTPUT_MIB_S = 15
_BYTES_PER_MIB = 2**20
_ROW_GROUP_BYTES = 256 * _BYTES_PER_MIB
_PAGE_BYTES = 8 * _BYTES_PER_MIB
_ARROW_ROW_GROUP_ROWS = 64 * 1024
_RUN_ID_LENGTH = 8
_ERROR_TEXT_LIMIT = 160
_REFERENCE_NAME = "node files (reference)"
_PRODUCTION_NAME = "chdb current"
_COLLECTOR_NAME = "collector merge (production)"
_MERGE_ORDERS_BY_FAMILY = {family.value: order for family, order in _MERGE_ORDERS.items()}
# The sort of the "metric-major rows" candidate is applied to all sampled windows together, the collector's
# only to each row group of a window, so the production row shows what streaming window by window keeps of it.
_METRIC_MAJOR = _MERGE_ORDERS[Family.ROUTER_VECTORS]

T = TypeVar("T")


@dataclass(frozen=True)
class ChdbCandidate:
    name: str
    settings: str


@dataclass(frozen=True)
class ArrowCandidate:
    name: str
    compression: str = "zstd"
    level: int | None = 3
    dictionary: bool = False
    split_floats: bool = True
    split_integers: bool = False
    statistics: bool = True
    metric_major: bool = False
    row_group_rows: int | None = None


# The settings of the first candidate are the collector's own; the others change one thing each.
_CHDB_CANDIDATES = (
    ChdbCandidate(_PRODUCTION_NAME, _PARQUET_ENCODE_SETTINGS),
    ChdbCandidate("chdb zstd level 1", f"{_PARQUET_ENCODE_SETTINGS}, output_format_compression_level=1"),
    ChdbCandidate("chdb zstd level 9", f"{_PARQUET_ENCODE_SETTINGS}, output_format_compression_level=9"),
    ChdbCandidate("chdb zstd level 19", f"{_PARQUET_ENCODE_SETTINGS}, output_format_compression_level=19"),
    ChdbCandidate("chdb defaults (dictionary, bloom filter)", "output_format_parquet_compression_method='zstd'"),
    ChdbCandidate(
        "chdb lz4",
        "output_format_parquet_compression_method='lz4', output_format_parquet_write_bloom_filter=0, "
        "output_format_parquet_max_dictionary_size=0",
    ),
    ChdbCandidate(
        "chdb 256 MiB row groups",
        f"{_PARQUET_ENCODE_SETTINGS}, output_format_parquet_row_group_size_bytes={_ROW_GROUP_BYTES}",
    ),
    ChdbCandidate(
        "chdb 8 MiB pages", f"{_PARQUET_ENCODE_SETTINGS}, output_format_parquet_data_page_size={_PAGE_BYTES}"
    ),
    ChdbCandidate("chdb arrow writer", f"{_PARQUET_ENCODE_SETTINGS}, output_format_parquet_use_custom_encoder=0"),
    ChdbCandidate(
        "chdb format version 2.latest", f"{_PARQUET_ENCODE_SETTINGS}, output_format_parquet_version='2.latest'"
    ),
)
# The first one writes what the node writes; "z3 split all" is what the collector writes for
# expert_counts and router_vectors, the others are the collector's chdb families' alternatives.
_ARROW_CANDIDATES = (
    ArrowCandidate("arrow node-style z3"),
    ArrowCandidate("arrow z1 split all", level=1, split_integers=True),
    ArrowCandidate("arrow z3 split all", split_integers=True),
    ArrowCandidate("arrow z6 split all", level=6, split_integers=True),
    ArrowCandidate("arrow z9 split all", level=9, split_integers=True),
    ArrowCandidate("arrow z15 split all", level=15, split_integers=True),
    ArrowCandidate("arrow z19 split all", level=19, split_integers=True),
    ArrowCandidate("arrow z6 split floats only", level=6),
    ArrowCandidate("arrow z6 split all, no statistics", level=6, split_integers=True, statistics=False),
    ArrowCandidate("arrow z6 dictionary", level=6, dictionary=True, split_floats=False),
    ArrowCandidate("arrow brotli 5 split all", compression="brotli", level=5, split_integers=True),
    ArrowCandidate("arrow z6 split all, metric-major rows", level=6, split_integers=True, metric_major=True),
    ArrowCandidate(
        "arrow z6 split all, 64Ki-row groups", level=6, split_integers=True, row_group_rows=_ARROW_ROW_GROUP_ROWS
    ),
)


@dataclass(frozen=True)
class Sample:
    """The first complete windows of one family of an attempt."""

    family: str
    attempt: AttemptDir
    windows: list[Window] | list[LoggedWindow]

    @property
    def file_name(self) -> str:
        return f"{self.family}.parquet"

    @property
    def structure(self) -> str:
        return _STRUCTURES[self.family]

    @cached_property
    def paths(self) -> list[Path]:
        return [window.path / self.file_name for window in self.windows]

    @cached_property
    def source_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.paths)

    @cached_property
    def pattern(self) -> str:
        """The glob the collector reads these windows with."""
        relative_dirs = sorted(window.relative_dir for window in self.windows)
        if len(relative_dirs) == 1:
            return f"{self.attempt.path}/{relative_dirs[0]}/{self.file_name}"
        return f"{self.attempt.path}/{{{','.join(relative_dirs)}}}/{self.file_name}"


@dataclass(frozen=True)
class Result:
    name: str
    size: int
    cpu_s: float
    wall_s: float
    decode_cpu_s: float
    decode_wall_s: float
    check: str
    footer: str


def _timed(call: Callable[[], T]) -> tuple[T, float, float]:
    """The result of `call` with the CPU seconds of the whole process (chdb runs in it) and the wall seconds."""
    cpu, wall = time.process_time(), time.perf_counter()
    result = call()
    return result, time.process_time() - cpu, time.perf_counter() - wall


def _decoded(session: Session, pattern: str, structure: str) -> tuple[int, int]:
    answer = session.query(
        f"SELECT count(), sum(cityHash64(*)) FROM file('{pattern}', 'Parquet', '{structure}') "
        "SETTINGS use_hive_partitioning=0, input_format_parquet_allow_missing_columns=1",
        "CSV",
    ).data()
    rows, checksum = answer.strip().split(",")
    return int(rows), int(checksum)


def _footer_verdict(session: Session, path: Path, rows: int) -> str:
    try:
        answer = session.query(f"SELECT sum(num_rows) FROM file('{path}', 'ParquetMetadata')", "CSV").data()
    except Exception as error:  # the format may be missing in an older chdb; say so instead of failing the row
        return _error(error)
    return "OK" if int(answer.strip()) == rows else f"rows={answer.strip()}"


def _error(error: Exception) -> str:
    return f"ERROR {type(error).__name__}: {str(error)[:_ERROR_TEXT_LIMIT]}"


def _measure(
    session: Session,
    sample: Sample,
    name: str,
    path: Path,
    encode: Callable[[], None],
    expected: tuple[int, int],
    read_cpu_s: float = 0.0,
    read_wall_s: float = 0.0,
) -> Result:
    try:
        _, cpu_s, wall_s = _timed(encode)
        decoded, decode_cpu_s, decode_wall_s = _timed(lambda: _decoded(session, str(path), sample.structure))
    except Exception as error:  # an unsupported setting or codec must not hide the other candidates
        return Result(name, 0, 0.0, 0.0, 0.0, 0.0, _error(error), "-")
    check = "OK" if decoded == expected else f"MISMATCH rows={decoded[0]} checksum={decoded[1]}"
    return Result(
        name, path.stat().st_size, cpu_s + read_cpu_s, wall_s + read_wall_s, decode_cpu_s, decode_wall_s, check,
        _footer_verdict(session, path, expected[0]),
    )


def _encode_chdb(session: Session, sample: Sample, candidate: ChdbCandidate, path: Path) -> None:
    session.query(
        f"INSERT INTO FUNCTION file('{path}', 'Parquet') SELECT * FROM file('{sample.pattern}', 'Parquet') "
        f"SETTINGS engine_file_truncate_on_insert=1, use_hive_partitioning=0, {candidate.settings}"
    )


def _byte_stream_split_columns(schema: pa.Schema, candidate: ArrowCandidate) -> list[str]:
    if "values" in schema.names:
        column, value_type = "values.list.element", schema.field("values").type.value_type
    else:
        column, value_type = "value", schema.field("value").type
    split = candidate.split_floats if pa.types.is_floating(value_type) else candidate.split_integers
    return [column] if split else []


def _encode_arrow(table: pa.Table, candidate: ArrowCandidate, path: Path) -> None:
    if candidate.metric_major:
        table = table.sort_by(_METRIC_MAJOR)
    pq.write_table(
        table, path, compression=candidate.compression, compression_level=candidate.level,
        use_dictionary=candidate.dictionary,
        use_byte_stream_split=_byte_stream_split_columns(table.schema, candidate) or False,
        write_statistics=candidate.statistics, row_group_size=candidate.row_group_rows,
    )


def _selected(name: str, only: list[str]) -> bool:
    return (
        name in (_PRODUCTION_NAME, _COLLECTOR_NAME) or not only or any(token.lower() in name.lower() for token in only)
    )


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _benchmark_family(
    session: Session, sample: Sample, directory: Path, only: list[str]
) -> tuple[tuple[int, int], list[Result]]:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"{sample.family}: decoding {len(sample.windows)} node windows", file=sys.stderr)
    expected, decode_cpu_s, decode_wall_s = _timed(lambda: _decoded(session, sample.pattern, sample.structure))
    results = [Result(_REFERENCE_NAME, sample.source_bytes, 0.0, 0.0, decode_cpu_s, decode_wall_s, "reference", "-")]
    for chdb_candidate in _CHDB_CANDIDATES:
        if _selected(chdb_candidate.name, only):
            print(f"{sample.family}: {chdb_candidate.name}", file=sys.stderr)
            path = directory / f"{_slug(chdb_candidate.name)}.parquet"
            results.append(_measure(
                session, sample, chdb_candidate.name, path,
                lambda: _encode_chdb(session, sample, chdb_candidate, path), expected,
            ))
    arrow_candidates = [candidate for candidate in _ARROW_CANDIDATES if _selected(candidate.name, only)]
    if arrow_candidates:
        table, read_cpu_s, read_wall_s = _timed(
            lambda: pa.concat_tables([pq.read_table(path, partitioning=None) for path in sample.paths])
        )
        for arrow_candidate in arrow_candidates:
            if arrow_candidate.metric_major and "metric_id" not in table.schema.names:
                continue
            print(f"{sample.family}: {arrow_candidate.name}", file=sys.stderr)
            path = directory / f"{_slug(arrow_candidate.name)}.parquet"
            results.append(_measure(
                session, sample, arrow_candidate.name, path,
                lambda: _encode_arrow(table, arrow_candidate, path), expected, read_cpu_s, read_wall_s,
            ))
    if sample.family in _MERGE_ORDERS_BY_FAMILY:
        print(f"{sample.family}: {_COLLECTOR_NAME}", file=sys.stderr)
        path = directory / f"{_slug(_COLLECTOR_NAME)}.parquet"
        results.append(_measure(
            session, sample, _COLLECTOR_NAME, path,
            lambda: Collector._merge_windows(
                sample.windows, sample.file_name, path, _MERGE_ORDERS_BY_FAMILY[sample.family]
            ),
            expected,
        ))
    return expected, results


def _print_family(sample: Sample, rows: int, results: list[Result]) -> None:
    source_mib = sample.source_bytes / _BYTES_PER_MIB
    print(f"\n== {sample.family}: {len(sample.windows)} windows, {rows} rows, node files {source_mib:.1f} MiB")
    production = next(result.size for result in results if result.name == _PRODUCTION_NAME)
    print(
        f"{'candidate':<42} {'MiB':>9} {'vs node':>8} {'vs chdb':>8} {'cpu_s':>8} {'wall_s':>8} "
        f"{'cores@15':>8} {'dec_cpu_s':>9} {'dec_wall_s':>10}  check / footer"
    )
    for result in results:
        size = f"{result.size / _BYTES_PER_MIB:>9.1f}" if result.size else f"{'-':>9}"
        versus_node = f"{result.size / sample.source_bytes - 1:>+8.1%}" if result.size else f"{'-':>8}"
        versus_production = f"{result.size / production - 1:>+8.1%}" if result.size and production else f"{'-':>8}"
        cores = result.cpu_s / source_mib * _NODE_OUTPUT_MIB_S
        print(
            f"{result.name:<42} {size} {versus_node} {versus_production} {result.cpu_s:>8.2f} {result.wall_s:>8.2f} "
            f"{cores:>8.2f} {result.decode_cpu_s:>9.2f} {result.decode_wall_s:>10.2f}  {result.check} / {result.footer}"
        )


def _upload_streaming(session: Session, target: S3Target, key: str, sample: Sample) -> None:
    """The collector's statement for several windows of scalars or logged: decode, encode and upload in one stream."""
    credentials = target.credentials
    session.query(
        f"INSERT INTO FUNCTION s3('{target.object_url(key)}', '{credentials.access_key}', "
        f"'{credentials.secret_key}', 'Parquet') SELECT * FROM file('{sample.pattern}', 'Parquet') "
        f"SETTINGS s3_truncate_on_insert=1, {_PARQUET_ENCODE_SETTINGS}, use_hive_partitioning=0, {_S3_UPLOAD_SETTINGS}"
    )


def _upload_file(session: Session, target: S3Target, key: str, path: Path) -> None:
    """The collector's statement for a singleton window: the bytes of a local Parquet file as they are."""
    credentials = target.credentials
    session.query(
        f"INSERT INTO FUNCTION s3('{target.object_url(key)}', '{credentials.access_key}', "
        f"'{credentials.secret_key}', 'RawBLOB') SELECT * FROM file('{path}', 'RawBLOB') "
        f"SETTINGS s3_truncate_on_insert=1, use_hive_partitioning=0, {_S3_UPLOAD_SETTINGS}"
    )


def _verify_per_object(session: Session, target: S3Target, receipts: list[Receipt]) -> None:
    """Verification with one count after each object."""
    credentials = target.credentials
    for receipt in receipts:
        answer = session.query(
            f"SELECT count() FROM s3('{target.object_url(receipt.key)}', '{credentials.access_key}', "
            f"'{credentials.secret_key}', 'Parquet', 'step UInt32') SETTINGS use_hive_partitioning=0",
            "CSV",
        ).data()
        _require_rows(receipt, int(answer.strip()))


def _verify_footers(session: Session, target: S3Target, receipts: list[Receipt]) -> None:
    credentials = target.credentials
    branches = " UNION ALL ".join(
        f"SELECT {index} AS item, rows FROM (SELECT num_rows AS rows FROM s3('{target.object_url(receipt.key)}', "
        f"'{credentials.access_key}', '{credentials.secret_key}', 'ParquetMetadata'))"
        for index, receipt in enumerate(receipts)
    )
    answer = session.query(f"SELECT item, rows FROM ({branches}) SETTINGS use_hive_partitioning=0", "CSV").data()
    for line in answer.splitlines():
        item, rows = line.split(",")
        _require_rows(receipts[int(item)], int(rows))


def _require_rows(receipt: Receipt, uploaded_rows: int) -> None:
    if uploaded_rows != receipt.rows:
        raise ValueError(f"uploaded {uploaded_rows} rows of {receipt.key}, source {receipt.rows}")


def _s3_phase(
    session: Session, target: S3Target, samples: list[Sample], expected: dict[str, tuple[int, int]],
    results: dict[str, list[Result]], directory: Path,
) -> None:
    run_id = uuid.uuid4().hex[:_RUN_ID_LENGTH]
    base = f"{target.prefix}/bench-{run_id}"
    print(f"\n== S3 phase: objects under {target.object_url(base)}/")
    print(f"{'upload':<52} {'MiB':>9} {'wall_s':>8} {'MiB/s':>8}")
    for sample in samples:
        production_path = directory / sample.family / f"{_slug(_PRODUCTION_NAME)}.parquet"
        verified = [result for result in results[sample.family] if result.check == "OK"]
        smallest = min(verified, key=lambda result: result.size)
        smallest_path = directory / sample.family / f"{_slug(smallest.name)}.parquet"
        uploads = (
            (f"{sample.family}: chdb streaming, as the collector", production_path,
             lambda: _upload_streaming(session, target, f"{base}/{sample.family}/streaming.parquet", sample)),
            (f"{sample.family}: RawBLOB of {smallest.name}", smallest_path,
             lambda: _upload_file(session, target, f"{base}/{sample.family}/smallest.parquet", smallest_path)),
        )
        for label, path, upload in uploads:
            try:
                _, _, wall_s = _timed(upload)
            except Exception as error:  # the proxy or an S3 setting may refuse one upload; report it and go on
                print(f"{label:<52} {_error(error)}")
                continue
            size = path.stat().st_size / _BYTES_PER_MIB
            print(f"{label:<52} {size:>9.1f} {wall_s:>8.2f} {size / wall_s:>8.1f}")

    print(f"\n{'verification of every family at once':<52} {'objects':>9} {'wall_s':>8}")
    collector = Collector(session, target, collected_retention_s=0)
    variants = (
        ("one count per object (before)", "per-object", lambda receipts: _verify_per_object(session, target, receipts)),
        ("one UNION ALL of counts (collector)", "counts", collector._verify_uploaded_rows),
        ("one UNION ALL of ParquetMetadata", "footers", lambda receipts: _verify_footers(session, target, receipts)),
    )
    for label, slug, verify in variants:
        # Every variant reads objects nothing has read before, so no cache answers for it.
        receipts = []
        for sample in samples:
            key = f"{base}/{slug}/{sample.family}.parquet"
            _upload_file(session, target, key, directory / sample.family / f"{_slug(_PRODUCTION_NAME)}.parquet")
            receipts.append(Receipt(sample.family, key, expected[sample.family][0]))
        try:
            _, _, wall_s = _timed(lambda: verify(receipts))
        except Exception as error:  # ParquetMetadata may be unsupported; the other variants still run
            print(f"{label:<52} {_error(error)}")
            continue
        print(f"{label:<52} {len(receipts):>9} {wall_s:>8.2f}")


def _choose_attempt(raw_root: RawRoot, exp: str | None, run: str | None, attempt: str | None) -> AttemptDir:
    attempts = [
        candidate for candidate in raw_root.attempt_dirs()
        if exp in (None, candidate.exp) and run in (None, candidate.run) and attempt in (None, candidate.attempt)
    ]
    if not attempts:
        sys.exit(f"no attempt under {raw_root.path}")
    chosen = max(attempts, key=lambda candidate: len(candidate.complete_windows()))
    if not chosen.complete_windows():
        sys.exit(f"no complete window in {chosen.path}")
    return chosen


def _samples(attempt: AttemptDir, families: list[str], count: int) -> list[Sample]:
    node_windows = sorted(attempt.complete_windows(), key=lambda window: (window.first_step, window.node))
    samples = []
    for family in families:
        if family == "logged":
            windows = attempt.complete_logged_windows()[:count]
        else:
            windows = [window for window in node_windows if window.family_path(family).exists()][:count]
        if windows:
            samples.append(Sample(family, attempt, windows))
    return samples


def run(arguments: argparse.Namespace, session: Session, directory: Path) -> None:
    attempt = _choose_attempt(RawRoot(arguments.raw_root), arguments.exp, arguments.run, arguments.attempt)
    samples = _samples(attempt, arguments.families, arguments.windows)
    version = session.query("SELECT version()", "CSV").data().strip()
    print(f"attempt {attempt.path}; ClickHouse {version}; pyarrow {pa.__version__}; {os.cpu_count()} CPUs")
    print(f"session: SET {SESSION_SETTINGS}")
    expected = {}
    results = {}
    for sample in samples:
        expected[sample.family], results[sample.family] = _benchmark_family(
            session, sample, directory / sample.family, arguments.only
        )
        _print_family(sample, expected[sample.family][0], results[sample.family])
    if arguments.s3_endpoint:
        credentials = S3Credentials.from_rclone_conf(arguments.rclone_conf, arguments.s3_remote_section)
        target = S3Target(arguments.s3_endpoint, arguments.s3_bucket, arguments.s3_prefix, credentials)
        _s3_phase(session, target, samples, expected, results, directory)


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Parquet encodings for the collector on real node windows."
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--exp")
    parser.add_argument("--run")
    parser.add_argument("--attempt")
    parser.add_argument("--windows", type=int, default=4, help="windows per family, oldest steps first")
    parser.add_argument("--families", nargs="+", choices=list(_STRUCTURES), default=list(_STRUCTURES))
    parser.add_argument("--only", nargs="+", default=[], help="run only candidates whose name contains one of these")
    parser.add_argument(
        "--work-dir", type=Path, help="keep the candidate files here instead of a deleted temporary directory"
    )
    parser.add_argument("--chdb-config", type=Path)
    parser.add_argument("--s3-endpoint")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-prefix")
    parser.add_argument("--s3-remote-section")
    parser.add_argument("--rclone-conf", type=Path, default=Path("~/.config/rclone/rclone.conf").expanduser())
    arguments = parser.parse_args(argv)
    if arguments.s3_endpoint and not (arguments.s3_bucket and arguments.s3_prefix and arguments.s3_remote_section):
        parser.error("--s3-endpoint needs --s3-bucket, --s3-prefix and --s3-remote-section")
    return arguments


def main(argv: list[str] | None = None) -> None:
    arguments = _parse_arguments(argv)
    session = _build_session(arguments.chdb_config)
    session.query(f"SET {SESSION_SETTINGS}")
    directory = arguments.work_dir or Path(tempfile.mkdtemp(prefix="collector-encoding-"))
    try:
        run(arguments, session, directory)
    finally:
        if arguments.work_dir is None:
            shutil.rmtree(directory)


if __name__ == "__main__":
    main()
