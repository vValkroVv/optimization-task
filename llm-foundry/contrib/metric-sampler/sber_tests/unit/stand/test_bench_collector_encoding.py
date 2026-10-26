from __future__ import annotations

import json
import re
import runpy
import sys
import zlib
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metric_sampler.contract.metrics import Family
from metric_sampler.contract.raw_layout import AttemptDir, RawRoot
from metric_sampler.island import cli
from metric_sampler.island.collector import _MERGE_ORDERS, Collector, Receipt
from metric_sampler.island.s3_target import S3Credentials, S3Target
from stand import bench_collector_encoding as bench

pytestmark = pytest.mark.unit

_INGEST_SQL = Path(__file__).parents[3] / "metric_sampler" / "cloud" / "ingest.sql"
_VECTOR_WIDTH = 16
_BASE_FIELDS = [
    ("node", pa.string()), ("step", pa.uint32()), ("rank", pa.int16()), ("microbatch", pa.int16()),
    ("layer", pa.int16()), ("metric_id", pa.uint16()), ("source_metric_id", pa.uint16()),
    ("contributors", pa.uint8()), ("expected_contributors", pa.uint8()), ("identical", pa.bool_()),
    ("verbose", pa.bool_()),
]


def _family_table(family: str, node: str, first_step: int, last_step: int) -> pa.Table:
    generator = np.random.default_rng(first_step)
    steps = list(range(first_step, last_step + 1))
    rows = [(step, layer, rank) for step in steps for layer in range(3) for rank in range(2)]
    counts = family == "expert_counts"
    value_type = pa.uint32() if counts else pa.float32()
    columns = {
        "node": [node] * len(rows),
        "step": pa.array([step for step, _, _ in rows], type=pa.uint32()),
        "rank": pa.array([rank for _, _, rank in rows], type=pa.int16()),
        "microbatch": pa.array([0] * len(rows), type=pa.int16()),
        "layer": pa.array([layer for _, layer, _ in rows], type=pa.int16()),
        "metric_id": pa.array([7 + layer for _, layer, _ in rows], type=pa.uint16()),
        "source_metric_id": pa.array([0] * len(rows), type=pa.uint16()),
        "contributors": pa.array([1] * len(rows), type=pa.uint8()),
        "expected_contributors": pa.array([2] * len(rows), type=pa.uint8()),
        "identical": [False] * len(rows),
        "verbose": [False] * len(rows),
    }
    fields = list(_BASE_FIELDS)
    if counts:
        columns["alias_metric_id"] = pa.array([0] * len(rows), type=pa.uint16())
        fields.append(("alias_metric_id", pa.uint16()))
        vectors = generator.integers(0, 5000, size=(len(rows), _VECTOR_WIDTH)).astype(np.uint32)
    else:
        vectors = generator.normal(0.0, 0.01, size=(len(rows), _VECTOR_WIDTH)).astype(np.float32)
    columns["values"] = pa.array(vectors.tolist(), type=pa.list_(value_type))
    fields.append(("values", pa.list_(value_type)))
    return pa.table(columns, schema=pa.schema(fields))


def _write_node_window(attempt: Path, node: str, first_step: int, last_step: int, families: list[str]) -> Path:
    window = attempt / f"node={node}" / f"w{first_step:08d}-{last_step:08d}"
    window.mkdir(parents=True)
    for family in families:
        pq.write_table(
            _family_table(family, node, first_step, last_step), window / f"{family}.parquet",
            compression="zstd", use_dictionary=False,
        )
    (window / "manifest.json").write_text(json.dumps({"first_step": first_step, "last_step": last_step}))
    return window


def _write_logged_window(attempt: Path, first_step: int, last_step: int) -> Path:
    window = attempt / "logged" / f"w{first_step:08d}-{last_step:08d}"
    window.mkdir(parents=True)
    steps = list(range(first_step, last_step + 1))
    table = pa.table(
        {
            "step": pa.array(steps, type=pa.uint32()),
            "name": [f"lr{step % 2}" for step in steps],
            "value": pa.array([0.5 * step for step in steps], type=pa.float64()),
            "source_step": pa.array([None if step % 2 else step for step in steps], type=pa.uint32()),
        }
    )
    pq.write_table(table, window / "logged.parquet", compression="zstd")
    (window / "manifest.json").write_text("{}")
    return window


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    attempt = tmp_path / "raw" / "expA" / "run1" / "attempt=0"
    every_family = ["scalars", "expert_counts", "router_vectors"]
    _write_node_window(attempt, "host2", 0, 3, every_family)
    _write_node_window(attempt, "host1", 0, 3, every_family)
    _write_node_window(attempt, "host1", 4, 7, ["scalars"])
    _write_logged_window(attempt, 0, 3)
    return tmp_path / "raw"


@dataclass
class _Answer:
    payload: str

    def data(self) -> str:
        return self.payload


@dataclass
class BenchSession:
    """The chdb statements of the benchmark, answered by decoding with pyarrow."""

    refused_setting: str = ""
    refused_object: str = ""
    footers_unsupported: bool = False
    queries: list[str] = field(default_factory=list)
    objects: dict[str, bytes] = field(default_factory=dict)

    def query(self, sql: str, fmt: str | None = None) -> _Answer:
        self.queries.append(sql)
        if sql == "SELECT version()":
            return _Answer("26.1.1.1\n")
        if sql.startswith("SET "):
            return _Answer("")
        encode = re.fullmatch(
            r"INSERT INTO FUNCTION file\('(.+)', 'Parquet'\) SELECT \* FROM file\('(.+)', 'Parquet'\) "
            r"SETTINGS (.+)", sql
        )
        if encode is not None:
            if self.refused_setting and self.refused_setting in encode[3]:
                raise RuntimeError(f"Unknown setting {self.refused_setting}")
            Path(encode[1]).write_bytes(self._reencoded(encode[2]))
            return _Answer("")
        decode = re.fullmatch(
            r"SELECT count\(\), sum\(cityHash64\(\*\)\) FROM file\('(.+)', 'Parquet', '(.+)'\) SETTINGS .+", sql
        )
        if decode is not None:
            rows = [row for path in _expand(decode[1]) for row in pq.read_table(path, partitioning=None).to_pylist()]
            checksum = sum(zlib.crc32(repr(sorted(row.items())).encode()) for row in rows)
            return _Answer(f"{len(rows)},{checksum}\n")
        footer = re.fullmatch(r"SELECT sum\(num_rows\) FROM file\('(.+)', 'ParquetMetadata'\)", sql)
        if footer is not None:
            self._require_footers()
            return _Answer(f"{pq.read_metadata(footer[1]).num_rows}\n")
        return self._answer_s3(sql)

    def _answer_s3(self, sql: str) -> _Answer:
        upload = re.match(
            r"INSERT INTO FUNCTION s3\('([^']+)', '[^']*', '[^']*', '(Parquet|RawBLOB)'\) "
            r"SELECT \* FROM file\('(.+?)', ", sql
        )
        if upload is not None:
            if self.refused_object and self.refused_object in upload[1]:
                raise OSError("proxy refused the upload")
            self.objects[upload[1]] = (
                Path(upload[3]).read_bytes() if upload[2] == "RawBLOB" else self._reencoded(upload[3])
            )
            return _Answer("")
        per_object = re.fullmatch(
            r"SELECT count\(\) FROM s3\('([^']+)', '[^']*', '[^']*', 'Parquet', 'step UInt32'\) SETTINGS .+", sql
        )
        if per_object is not None:
            return _Answer(f"{self._object_rows(per_object[1])}\n")
        branches = re.findall(
            r"SELECT (\d+) AS item, rows FROM \(SELECT (?:count\(\)|num_rows) AS rows FROM s3\('([^']+)'", sql
        )
        if branches:
            if "'ParquetMetadata'" in sql:
                self._require_footers()
            return _Answer("\n".join(f"{item},{self._object_rows(url)}" for item, url in branches))
        raise AssertionError(f"unexpected statement: {sql}")

    def _require_footers(self) -> None:
        if self.footers_unsupported:
            raise RuntimeError("Unknown input format ParquetMetadata")

    def _object_rows(self, url: str) -> int:
        return pq.read_metadata(BytesIO(self.objects[url])).num_rows

    @staticmethod
    def _reencoded(pattern: str) -> bytes:
        table = pa.concat_tables([pq.read_table(path, partitioning=None) for path in _expand(pattern)])
        sink = BytesIO()
        pq.write_table(table, sink, compression="zstd", use_dictionary=False)
        return sink.getvalue()


def _expand(pattern: str) -> list[Path]:
    braces = re.fullmatch(r"(.*)\{(.*)\}(.*)", pattern)
    if braces is None:
        return [Path(pattern)]
    return [Path(braces[1] + alternative + braces[3]) for alternative in braces[2].split(",")]


def _samples(raw_root: Path, families: list[str] | None = None, windows: int = 2) -> list[bench.Sample]:
    attempt = bench._choose_attempt(RawRoot(raw_root), None, None, None)
    return bench._samples(attempt, families or list(bench._STRUCTURES), windows)


def test_a_sample_takes_the_oldest_windows_of_every_node_that_have_the_family(raw_root: Path) -> None:
    by_family = {sample.family: sample for sample in _samples(raw_root, windows=2)}

    assert list(by_family) == ["scalars", "expert_counts", "router_vectors", "logged"]
    assert [window.relative_dir for window in by_family["scalars"].windows] == [
        "node=host1/w00000000-00000003", "node=host2/w00000000-00000003",
    ]
    assert [window.relative_dir for window in _samples(raw_root, ["scalars"], windows=3)[0].windows] == [
        "node=host1/w00000000-00000003", "node=host2/w00000000-00000003", "node=host1/w00000004-00000007",
    ]
    assert [window.relative_dir for window in by_family["router_vectors"].windows] == [
        "node=host1/w00000000-00000003", "node=host2/w00000000-00000003",
    ]
    assert [window.relative_dir for window in by_family["logged"].windows] == ["logged/w00000000-00000003"]
    assert by_family["scalars"].source_bytes == sum(path.stat().st_size for path in by_family["scalars"].paths)


@pytest.mark.parametrize("window_count", [1, 2])
def test_a_sample_reads_its_windows_with_the_glob_the_collector_uses(raw_root: Path, window_count: int) -> None:
    sample = _samples(raw_root, ["scalars"], windows=window_count)[0]
    attempt = AttemptDir(sample.attempt.path, "expA", "run1", "0")

    assert f"file('{sample.pattern}', 'Parquet')" == Collector._file_source(
        attempt.path, "scalars.parquet", sample.windows
    )


def test_structures_are_the_ones_the_ingest_reads() -> None:
    ingest = _INGEST_SQL.read_text()
    structures = dict(
        re.findall(r"filename = 'family=(\w+)/[^']*', format = 'Parquet',\s*structure = '([^']+)'", ingest)
    )

    assert structures == bench._STRUCTURES


@pytest.mark.parametrize(
    ("exp", "run", "attempt", "expected_windows"),
    [
        (None, None, None, 3), ("expA", None, None, 2), ("expB", None, None, 1), (None, "run2", None, 1),
        (None, None, "1", 3), ("expA", "run1", "0", 2),
    ],
)
def test_the_attempt_with_most_windows_is_chosen_among_the_filtered(
    tmp_path, exp, run, attempt, expected_windows,
) -> None:
    root = tmp_path / "raw"
    _write_node_window(root / "expA" / "run1" / "attempt=0", "host1", 0, 3, ["scalars"])
    _write_node_window(root / "expA" / "run1" / "attempt=0", "host2", 0, 3, ["scalars"])
    _write_node_window(root / "expB" / "run1" / "attempt=1", "host1", 0, 3, ["scalars"])
    _write_node_window(root / "expA" / "run2" / "attempt=2", "host1", 0, 3, ["scalars"])
    for node in ["host1", "host2", "host3"]:
        _write_node_window(root / "expC" / "run1" / "attempt=1", node, 0, 3, ["scalars"])

    chosen = bench._choose_attempt(RawRoot(root), exp, run, attempt)

    assert len(chosen.complete_windows()) == expected_windows


def test_without_a_matching_attempt_or_a_complete_window_the_benchmark_stops(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    (root / "expA" / "run1" / "attempt=0" / "node=host1").mkdir(parents=True)

    with pytest.raises(SystemExit, match="no attempt under"):
        bench._choose_attempt(RawRoot(root), "expB", None, None)
    with pytest.raises(SystemExit, match="no complete window in"):
        bench._choose_attempt(RawRoot(root), None, None, None)


def test_every_candidate_decodes_to_the_node_rows_in_every_family(raw_root: Path, tmp_path: Path) -> None:
    session = BenchSession()
    for sample in _samples(raw_root):
        expected, results = bench._benchmark_family(session, sample, tmp_path / sample.family, [])

        arrow_names = [candidate.name for candidate in bench._ARROW_CANDIDATES]
        if sample.family == "logged":
            arrow_names = [name for name in arrow_names if "metric-major" not in name]
        merged = [bench._COLLECTOR_NAME] if sample.family in ("expert_counts", "router_vectors") else []
        assert [result.name for result in results] == [
            bench._REFERENCE_NAME, *(candidate.name for candidate in bench._CHDB_CANDIDATES), *arrow_names, *merged,
        ]
        assert expected[0] == sum(pq.read_metadata(path).num_rows for path in sample.paths)
        reference, *candidates = results
        assert (reference.size, reference.check) == (sample.source_bytes, "reference")
        assert [(result.check, result.footer) for result in candidates] == [("OK", "OK")] * len(candidates)
        assert all(result.size == (tmp_path / sample.family / f"{bench._slug(result.name)}.parquet").stat().st_size
                   for result in candidates)


def _values_chunk(path: Path):
    row_group = pq.read_metadata(path).row_group(0)
    return next(
        row_group.column(index) for index in range(row_group.num_columns)
        if row_group.column(index).path_in_schema == "values.list.element"
    )


@pytest.mark.parametrize(
    ("family", "name", "byte_stream_split"),
    [
        ("router_vectors", "arrow node-style z3", True),
        ("router_vectors", "arrow z6 split floats only", True),
        ("router_vectors", "arrow z6 split all", True),
        ("expert_counts", "arrow node-style z3", False),
        ("expert_counts", "arrow z6 split floats only", False),
        ("expert_counts", "arrow z6 split all", True),
        ("expert_counts", "arrow z1 split all", True),
    ],
)
def test_arrow_candidates_split_the_byte_streams_of_the_columns_their_names_say(
    raw_root: Path, tmp_path: Path, family: str, name: str, byte_stream_split: bool
) -> None:
    sample = _samples(raw_root, [family])[0]
    bench._benchmark_family(BenchSession(), sample, tmp_path, [name])

    chunk = _values_chunk(tmp_path / f"{bench._slug(name)}.parquet")

    assert ("BYTE_STREAM_SPLIT" in chunk.encodings) is byte_stream_split
    assert not chunk.has_dictionary_page


def test_arrow_candidates_choose_dictionaries_and_statistics_as_their_names_say(raw_root: Path, tmp_path: Path) -> None:
    sample = _samples(raw_root, ["expert_counts"])[0]
    bench._benchmark_family(
        BenchSession(), sample, tmp_path, ["z6 split all, no statistics", "arrow z6 dictionary", "z6 split all"]
    )

    dictionary = _values_chunk(tmp_path / f"{bench._slug('arrow z6 dictionary')}.parquet")
    without_statistics = _values_chunk(tmp_path / f"{bench._slug('arrow z6 split all, no statistics')}.parquet")
    with_statistics = _values_chunk(tmp_path / f"{bench._slug('arrow z6 split all')}.parquet")
    assert dictionary.has_dictionary_page and "BYTE_STREAM_SPLIT" not in dictionary.encodings
    assert not without_statistics.is_stats_set and with_statistics.is_stats_set


def test_arrow_candidates_split_logged_floats_like_the_values_of_other_families(tmp_path: Path, raw_root: Path) -> None:
    sample = _samples(raw_root, ["logged"])[0]
    bench._benchmark_family(BenchSession(), sample, tmp_path, ["node-style"])

    row_group = pq.read_metadata(tmp_path / f"{bench._slug('arrow node-style z3')}.parquet").row_group(0)
    encodings = {row_group.column(index).path_in_schema: row_group.column(index).encodings
                 for index in range(row_group.num_columns)}
    assert "BYTE_STREAM_SPLIT" in encodings["value"]
    assert "BYTE_STREAM_SPLIT" not in encodings["step"]


def test_the_metric_major_candidate_reorders_rows_and_the_row_group_candidate_splits_them(
    raw_root: Path, tmp_path: Path
) -> None:
    session = BenchSession()
    sample = _samples(raw_root, ["scalars"], windows=1)[0]
    bench._benchmark_family(session, sample, tmp_path, ["metric-major", "64Ki"])
    bench._benchmark_family(session, sample, tmp_path / "small", [])
    table = pa.concat_tables([pq.read_table(path, partitioning=None) for path in sample.paths])

    reordered = pq.read_table(tmp_path / f"{bench._slug('arrow z6 split all, metric-major rows')}.parquet")

    assert reordered["metric_id"].to_pylist() == sorted(table["metric_id"].to_pylist())
    assert reordered.sort_by([("step", "ascending"), ("metric_id", "ascending"), ("rank", "ascending")]).equals(
        table.sort_by([("step", "ascending"), ("metric_id", "ascending"), ("rank", "ascending")])
    )
    assert (
        pq.read_metadata(tmp_path / f"{bench._slug('arrow z6 split all, 64Ki-row groups')}.parquet").num_row_groups == 1
    )


def test_a_setting_chdb_does_not_know_is_reported_in_its_row_and_the_others_still_run(
    raw_root: Path, tmp_path: Path
) -> None:
    sample = _samples(raw_root, ["scalars"])[0]

    _, results = bench._benchmark_family(
        BenchSession(refused_setting="output_format_compression_level=19"), sample, tmp_path, ["level", "z3 split all"]
    )

    verdicts = {result.name: (result.size, result.check) for result in results}
    assert verdicts["chdb zstd level 19"] == (
        0, "ERROR RuntimeError: Unknown setting output_format_compression_level=19"
    )
    assert verdicts["chdb zstd level 1"][1] == verdicts["chdb current"][1] == verdicts["arrow z3 split all"][1] == "OK"


def test_a_candidate_decoding_to_other_rows_is_a_mismatch(
    raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = _samples(raw_root, ["scalars"])[0]
    real_encode = bench._encode_arrow
    monkeypatch.setattr(
        bench, "_encode_arrow", lambda table, candidate, path: real_encode(table.slice(1), candidate, path)
    )

    _, results = bench._benchmark_family(BenchSession(), sample, tmp_path, ["arrow z1 split all"])

    (mismatch,) = [result for result in results if result.name == "arrow z1 split all"]
    assert re.fullmatch(r"MISMATCH rows=\d+ checksum=\d+", mismatch.check)
    assert [result.check for result in results if result.name == "chdb current"] == ["OK"]


def test_a_chdb_without_the_footer_format_still_checks_the_decoded_rows(raw_root: Path, tmp_path: Path) -> None:
    sample = _samples(raw_root, ["scalars"])[0]

    _, results = bench._benchmark_family(BenchSession(footers_unsupported=True), sample, tmp_path, ["chdb current"])

    assert [(result.check, result.footer) for result in results[1:]] == [
        ("OK", "ERROR RuntimeError: Unknown input format ParquetMetadata")
    ]


def test_a_footer_with_other_rows_is_reported_with_its_count(raw_root: Path, tmp_path: Path, monkeypatch) -> None:
    session = BenchSession()
    sample = _samples(raw_root, ["scalars"])[0]
    real_query = session.query

    def query(sql, fmt=None):
        return _Answer("1\n") if "ParquetMetadata" in sql else real_query(sql, fmt)

    monkeypatch.setattr(session, "query", query)

    _, results = bench._benchmark_family(session, sample, tmp_path, ["chdb current"])

    assert results[1].footer == "rows=1"


def test_the_family_table_reports_sizes_ratios_cores_and_checks(raw_root: Path, capsys) -> None:
    sample = _samples(raw_root, ["scalars"])[0]
    mib = 2**20
    results = [
        bench.Result(bench._REFERENCE_NAME, sample.source_bytes, 0.0, 0.0, 0.5, 0.25, "reference", "-"),
        bench.Result(bench._PRODUCTION_NAME, 10 * mib, 3.0, 1.5, 1.0, 0.5, "OK", "OK"),
        bench.Result("arrow z6", 8 * mib, 6.0, 6.5, 0.75, 0.5, "OK", "ERROR RuntimeError: x"),
        bench.Result("chdb broken", 0, 0.0, 0.0, 0.0, 0.0, "ERROR ValueError: y", "-"),
    ]

    bench._print_family(sample, 24, results)

    lines = capsys.readouterr().out.splitlines()
    source_mib = sample.source_bytes / mib
    assert lines[1] == f"== scalars: 2 windows, 24 rows, node files {source_mib:.1f} MiB"
    assert lines[2].split() == [
        "candidate", "MiB", "vs", "node", "vs", "chdb", "cpu_s", "wall_s", "cores@15", "dec_cpu_s", "dec_wall_s",
        "check", "/", "footer",
    ]
    assert lines[3].split()[:4] == ["node", "files", "(reference)", f"{sample.source_bytes / mib:.1f}"]
    assert lines[4].split() == [
        "chdb", "current", "10.0", f"{10 / source_mib - 1:+.1%}", "+0.0%", "3.00", "1.50",
        f"{3.0 / source_mib * 15:.2f}", "1.00", "0.50", "OK", "/", "OK",
    ]
    assert lines[5].split()[:4] == ["arrow", "z6", "8.0", f"{8 / source_mib - 1:+.1%}"]
    assert "-20.0%" in lines[5]
    assert lines[5].endswith("OK / ERROR RuntimeError: x")
    assert lines[6].split()[:5] == ["chdb", "broken", "-", "-", "-"]
    assert lines[6].endswith("ERROR ValueError: y / -")


def _args(raw_root: Path, *extra: str):
    return bench._parse_arguments(["--raw-root", str(raw_root), "--windows", "2", *extra])


@pytest.fixture
def rclone_conf(tmp_path: Path) -> Path:
    path = tmp_path / "rclone.conf"
    path.write_text("[metadata-s3]\naccess_key_id = test-key\nsecret_access_key = test-secret\n")
    return path


def _s3_args(raw_root: Path, rclone_conf: Path, *extra: str):
    return _args(
        raw_root, "--s3-endpoint", "https://s3.example.invalid", "--s3-bucket", "store", "--s3-prefix", "scratch/bench",
        "--s3-remote-section", "metadata-s3", "--rclone-conf", str(rclone_conf), *extra,
    )


def test_nothing_is_uploaded_without_an_s3_endpoint(raw_root: Path, tmp_path: Path, capsys) -> None:
    session = BenchSession()

    bench.run(_args(raw_root, "--only", "z1 "), session, tmp_path / "work")

    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        f"attempt {raw_root / 'expA' / 'run1' / 'attempt=0'}; ClickHouse 26.1.1.1; pyarrow {pa.__version__}; "
        f"{bench.os.cpu_count()} CPUs"
    )
    assert out.splitlines()[1] == f"session: SET {cli.SESSION_SETTINGS}"
    assert [line.split()[0] for line in out.splitlines() if line.startswith("==")] == ["==", "==", "==", "=="]
    assert "S3 phase" not in out
    assert session.objects == {} and not any("s3(" in sql for sql in session.queries)


def test_the_s3_phase_times_uploads_and_three_verifications_of_objects_nothing_has_read(
    raw_root: Path, rclone_conf: Path, tmp_path: Path, capsys
) -> None:
    session = BenchSession()

    bench.run(_s3_args(raw_root, rclone_conf, "--only", "z1 "), session, tmp_path / "work")

    captured = capsys.readouterr()
    out = captured.out
    assert "test-secret" not in out + captured.err and "test-key" not in out + captured.err
    (base_line,) = [line for line in out.splitlines() if line.startswith("== S3 phase")]
    base = base_line.split("objects under ")[1].rstrip("/")
    assert re.fullmatch(r"https://s3\.example\.invalid/store/scratch/bench/bench-[0-9a-f]{8}", base)
    families = ["scalars", "expert_counts", "router_vectors", "logged"]
    keys = {url.removeprefix(base + "/") for url in session.objects}
    assert keys == {
        f"{prefix}/{family}.parquet" if prefix in ("per-object", "counts", "footers") else f"{family}/{prefix}.parquet"
        for family in families for prefix in ("streaming", "smallest", "per-object", "counts", "footers")
    }
    assert "ERROR" not in out
    for label in [
        "one count per object (before)", "one UNION ALL of counts (collector)", "one UNION ALL of ParquetMetadata"
    ]:
        (line,) = [line for line in out.splitlines() if line.startswith(label)]
        assert line.split()[-2] == "4"
    uploads = [line for line in out.splitlines() if line.startswith("scalars: ")]
    assert [
        line.split(":")[1].split(",")[0].split(" of ")[0].strip() for line in uploads
    ] == ["chdb streaming", "RawBLOB"]
    smallest = min(
        (path for path in (tmp_path / "work" / "scalars").glob("*.parquet")
         if path.name != f"{bench._slug(bench._REFERENCE_NAME)}.parquet"),
        key=lambda path: path.stat().st_size,
    )
    assert session.objects[f"{base}/scalars/smallest.parquet"] == smallest.read_bytes()


def test_a_refused_upload_and_unsupported_footers_are_reported_while_the_rest_of_the_phase_runs(
    raw_root: Path, rclone_conf: Path, tmp_path: Path, capsys
) -> None:
    session = BenchSession(refused_object="smallest", footers_unsupported=True)

    bench.run(_s3_args(raw_root, rclone_conf, "--families", "scalars", "--only", "z1 "), session, tmp_path / "work")

    out = capsys.readouterr().out
    assert "scalars: RawBLOB of" in out
    assert any(line.endswith("ERROR OSError: proxy refused the upload") for line in out.splitlines())
    assert any(
        line.startswith("one UNION ALL of ParquetMetadata") and "ERROR RuntimeError: Unknown input format" in line
        for line in out.splitlines()
    )
    assert any(
        line.startswith("one UNION ALL of counts (collector)") and "ERROR" not in line for line in out.splitlines()
    )
    assert any(line.startswith("scalars: chdb streaming") and "ERROR" not in line for line in out.splitlines())


def test_the_batch_verification_it_times_is_the_collectors_own(
    raw_root: Path, rclone_conf: Path, tmp_path: Path,
) -> None:
    session = BenchSession()

    bench.run(_s3_args(raw_root, rclone_conf, "--families", "scalars", "--only", "z1 "), session, tmp_path / "work")

    target = S3Target("https://s3.example.invalid", "store", "scratch/bench", S3Credentials("test-key", "test-secret"))
    verifications = [sql for sql in session.queries if sql.startswith("SELECT item, rows FROM (") and "count()" in sql]
    (sql,) = verifications
    url = re.search(r"s3\('([^']+)'", sql)[1]
    recorder = BenchSession()
    recorder.objects[url] = b"x"
    recorded = []
    recorder.query = lambda statement, fmt=None: recorded.append(statement) or _Answer("0,5")
    Collector(recorder, target, collected_retention_s=0)._verify_uploaded_rows(
        [Receipt("scalars", url.split("/store/")[1], 5)]
    )
    assert recorded == [sql]


def test_a_row_count_other_than_the_source_fails_a_verification() -> None:
    receipt = Receipt("scalars", "scratch/scalars.parquet", 5)

    bench._require_rows(receipt, 5)
    with pytest.raises(ValueError, match=r"uploaded 4 rows of scratch/scalars\.parquet, source 5"):
        bench._require_rows(receipt, 4)


def test_the_s3_target_needs_all_of_its_parts(raw_root: Path) -> None:
    with pytest.raises(SystemExit):
        _args(raw_root, "--s3-endpoint", "https://s3.example.invalid", "--s3-bucket", "store")


def test_main_runs_in_a_temporary_directory_it_removes_or_in_the_one_given(
    raw_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    session = BenchSession()
    monkeypatch.setattr(bench, "_build_session", lambda chdb_config: session)
    base = ["--raw-root", str(raw_root), "--windows", "1", "--families", "scalars", "--only", "z1 "]
    used = []
    real_run = bench.run
    monkeypatch.setattr(
        bench, "run", lambda args, session, directory: used.append(directory) or real_run(args, session, directory)
    )

    bench.main(base)
    kept = tmp_path / "kept"
    bench.main([*base, "--work-dir", str(kept)])

    assert session.queries[0] == f"SET {cli.SESSION_SETTINGS}"
    assert not used[0].exists()
    assert used[1] == kept and (kept / "scalars" / "chdb-current.parquet").is_file()
    assert capsys.readouterr().out.count("== scalars") == 2


def test_families_the_attempt_has_no_window_of_are_not_sampled(tmp_path: Path) -> None:
    attempt = tmp_path / "raw" / "expA" / "run1" / "attempt=0"
    _write_node_window(attempt, "host1", 0, 3, ["scalars"])

    assert [sample.family for sample in _samples(tmp_path / "raw")] == ["scalars"]


def test_the_script_runs_as_a_program(raw_root: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    session = BenchSession()
    monkeypatch.setattr(cli, "_build_session", lambda chdb_config: session)
    monkeypatch.setattr(
        sys, "argv",
        ["bench_collector_encoding.py", "--raw-root", str(raw_root), "--families", "scalars", "--only", "z1 "],
    )

    runpy.run_path(str(Path(bench.__file__)), run_name="__main__")

    assert "== scalars: 3 windows" in capsys.readouterr().out


@pytest.mark.parametrize("family", ["expert_counts", "router_vectors"])
def test_the_collector_merge_row_is_the_collectors_own_merge_of_the_sample(
    raw_root: Path, tmp_path: Path, family,
) -> None:
    sample = _samples(raw_root, [family], windows=3)[0]

    _, results = bench._benchmark_family(BenchSession(), sample, tmp_path, ["z1 "])

    (merge,) = [result for result in results if result.name == bench._COLLECTOR_NAME]
    assert (merge.check, merge.footer) == ("OK", "OK")
    path = tmp_path / f"{bench._slug(bench._COLLECTOR_NAME)}.parquet"
    assert merge.size == path.stat().st_size
    windows = sorted(sample.windows, key=lambda window: window.relative_dir)
    node_rows = [pq.ParquetFile(window.path / sample.file_name).read_row_group(0) for window in windows]
    uploaded = pq.ParquetFile(path)
    assert [uploaded.read_row_group(index).num_rows for index in range(uploaded.num_row_groups)] == [
        group.num_rows for group in node_rows
    ]
    for index, group in enumerate(node_rows):
        expected_order = _MERGE_ORDERS[Family(family)]
        assert uploaded.read_row_group(index).equals(group.sort_by(expected_order) if expected_order else group)
    metric_ids = uploaded.read_row_group(0)["metric_id"].to_pylist()
    assert (metric_ids == sorted(metric_ids)) is (family == "router_vectors")


def test_the_scalars_and_logged_samples_have_no_collector_merge_row(raw_root: Path, tmp_path: Path) -> None:
    for sample in _samples(raw_root, ["scalars", "logged"]):
        _, results = bench._benchmark_family(BenchSession(), sample, tmp_path / sample.family, ["z1 "])

        assert bench._COLLECTOR_NAME not in [result.name for result in results]
