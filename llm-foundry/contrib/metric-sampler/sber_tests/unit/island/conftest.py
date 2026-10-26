from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metric_sampler.contract.metrics import Family
from metric_sampler.contract.raw_layout import AttemptDir, RawRoot
from metric_sampler.island.s3_target import S3Credentials, S3Target

# A dump query's SQL always names its file or S3 key with `step=.../rank=....parquet`, whether it
# is the local path (file()) or the uploaded key (s3()); nothing else in the collector's SQL uses
# that shape, so it doubles as a cheap way to tell a dump query apart from a family/manifest one.
# The local path is a pathlib.Path formatted with the platform separator, hence `[/\\]`.
_DUMP_QUERY_RE = re.compile(r"step=(\d+)[/\\]rank=(\d+)")
_VERIFIED_OBJECT_RE = re.compile(r"SELECT (\d+) AS item, rows FROM \(SELECT count\(\) AS rows FROM s3\('([^']+)'")
_FILE_UPLOAD_RE = re.compile(
    r"INSERT INTO FUNCTION s3\('([^']+)', '[^']*', '[^']*', 'RawBLOB'\) SELECT \* FROM file\('([^']+)'"
)
_MERGED_FAMILIES = ("expert_counts", "router_vectors")

_DUMP_SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("rank", pa.int32()),
        ("microbatch", pa.int16()),
        ("point", pa.string()),
        ("layer", pa.int32()),
        ("token_index", pa.int32()),
        ("token_id", pa.int32()),
        ("loss", pa.float32()),
        ("stream_id", pa.int32()),
        ("dtype", pa.string()),
        ("values", pa.list_(pa.float32(), 4)),
    ]
)

_VALUE_TYPE = {
    Family.SCALARS: pa.float32(),
    Family.EXPERT_COUNTS: pa.uint32(),
    Family.ROUTER_VECTORS: pa.float32(),
}

_BASE_FIELDS = [
    ("node", pa.string()),
    ("step", pa.uint32()),
    ("rank", pa.int16()),
    ("microbatch", pa.int16()),
    ("layer", pa.int16()),
    ("metric_id", pa.uint16()),
    ("source_metric_id", pa.uint16()),
    ("contributors", pa.uint8()),
    ("expected_contributors", pa.uint8()),
    ("identical", pa.bool_()),
    ("verbose", pa.bool_()),
]


def _write_family_parquet(
    path: Path, node: str, first_step: int, last_step: int, row_group_size: int | None = None
) -> None:
    steps = list(range(first_step, last_step + 1))
    count = len(steps)
    value_type = _VALUE_TYPE[Family(path.stem)]
    schema = pa.schema([*_BASE_FIELDS, ("values", pa.list_(value_type))])
    table = pa.table(
        {
            "node": [node] * count,
            "step": pa.array(steps, type=pa.uint32()),
            "rank": [0] * count,
            "microbatch": [0] * count,
            "layer": pa.array([step % 3 for step in steps], type=pa.int16()),
            "metric_id": pa.array([1 + step % 2 for step in steps], type=pa.uint16()),
            "source_metric_id": pa.array([0] * count, type=pa.uint16()),
            "contributors": [1] * count,
            "expected_contributors": [1] * count,
            "identical": [False] * count,
            "verbose": [False] * count,
            "values": [[step, step * 7 % 1000] for step in steps],
        },
        schema=schema,
    )
    pq.write_table(table, path, row_group_size=row_group_size)


@dataclass
class FakeQueryResult:
    payload: str

    def data(self) -> str:
        return self.payload


@dataclass
class FakeSession:
    """Answers row-stats and count queries from configured per-family stats and records every SQL sent.

    The families the collector merges with pyarrow are not read by chdb, so their uploaded rows are the
    footer of the bytes the session was asked to upload, unless `uploaded_rows` says otherwise."""

    stats: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    uploaded_rows: dict[str, int] = field(default_factory=dict)
    objects: dict[str, bytes] = field(default_factory=dict)
    queries: list[tuple[str, str | None]] = field(default_factory=list)
    # Keyed by (step, rank). dump_stats mirrors the JSONEachRow answer to the collector's one
    # points/layers/tokens query; dump_uploaded_rows defaults to that same row count, so a test
    # only needs to set it explicitly to simulate a mismatch.
    dump_stats: dict[tuple[int, int], dict] = field(default_factory=dict)
    dump_uploaded_rows: dict[tuple[int, int], int] = field(default_factory=dict)

    @staticmethod
    def is_dump_query(sql: str) -> bool:
        return _DUMP_QUERY_RE.search(sql) is not None

    @property
    def dump_queries(self) -> list[str]:
        return [sql for sql, _ in self.queries if self.is_dump_query(sql)]

    def query(self, sql: str, fmt: str | None = None) -> FakeQueryResult:
        self.queries.append((sql, fmt))
        dump_match = _DUMP_QUERY_RE.search(sql)
        if dump_match is not None:
            dump_key = (int(dump_match.group(1)), int(dump_match.group(2)))
            if fmt == "JSONEachRow":
                return FakeQueryResult(json.dumps(self.dump_stats[dump_key]))
            if fmt == "CSV":
                uploaded = self.dump_uploaded_rows.get(dump_key, self.dump_stats[dump_key]["rows"])
                return FakeQueryResult(str(uploaded))
            return FakeQueryResult("")
        upload = _FILE_UPLOAD_RE.match(sql)
        if upload is not None:
            self.objects[upload[1]] = Path(upload[2]).read_bytes()
        if fmt != "CSV":
            return FakeQueryResult("")
        verified = _VERIFIED_OBJECT_RE.findall(sql)
        if verified:
            return FakeQueryResult("\n".join(f"{item},{self._uploaded_rows_of(url)}" for item, url in verified))
        if "min(step)" in sql:
            count, first_step, last_step = self.stats[self._family_of(sql)]
            return FakeQueryResult(f"{count},{first_step},{last_step}")
        raise AssertionError(f"unexpected CSV query: {sql}")

    def _uploaded_rows_of(self, url: str) -> int:
        family = self._family_of(url)
        if family in self.uploaded_rows:
            return self.uploaded_rows[family]
        if family in _MERGED_FAMILIES:
            return pq.read_metadata(BytesIO(self.objects[url])).num_rows
        return self.stats[family][0]

    @staticmethod
    def _family_of(sql: str) -> str:
        for family_name in [*(family.value for family in Family), "logged"]:
            if f"{family_name}.parquet" in sql or f"family={family_name}/" in sql:
                return family_name
        raise AssertionError(f"cannot infer family from query: {sql}")


@pytest.fixture
def fake_session_factory():
    return FakeSession


@pytest.fixture(autouse=True)
def scratch_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The system temporary directory of every test: where the collector merges its files by default."""
    directory = tmp_path / "scratch"
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


@pytest.fixture
def dump_target() -> S3Target:
    return S3Target(
        endpoint="https://s3.example.invalid",
        bucket="dumps-bucket",
        prefix="metric-sampler-dumps",
        credentials=S3Credentials(access_key="dump-key", secret_key="dump-secret"),
    )


@pytest.fixture
def raw_root(tmp_path: Path) -> RawRoot:
    return RawRoot(tmp_path / "raw")


@pytest.fixture
def write_window():
    def _write_window(
        root: Path,
        exp: str,
        run: str,
        attempt: str,
        node: str,
        first_step: int,
        last_step: int,
        families: tuple[Family, ...] = tuple(Family),
        data_steps: tuple[int, int] | None = None,
        row_group_size: int | None = None,
    ) -> Path:
        window_dir = (
            root / exp / run / f"attempt={attempt}" / f"node={node}" / f"w{first_step:08d}-{last_step:08d}"
        )
        window_dir.mkdir(parents=True)
        first_data_step, last_data_step = data_steps or (first_step, last_step)
        for family in families:
            _write_family_parquet(
                window_dir / f"{family.value}.parquet", node, first_data_step, last_data_step, row_group_size
            )
        manifest = {
            "schema_version": 1,
            "attempt": attempt,
            "node": node,
            "first_step": first_step,
            "last_step": last_step,
        }
        (window_dir / "manifest.json").write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))
        return window_dir

    return _write_window


@pytest.fixture
def write_metrics_registry():
    def _write_metrics_registry(root: Path, exp: str, run: str, attempt: str, content: str = "[]") -> Path:
        path = root / exp / run / f"attempt={attempt}" / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
        return path

    return _write_metrics_registry


@pytest.fixture
def write_streams_registry():
    def _write_streams_registry(root: Path, exp: str, run: str, attempt: str, content: str = "[]") -> Path:
        path = root / exp / run / f"attempt={attempt}" / "streams.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
        return path

    return _write_streams_registry


@pytest.fixture
def write_logged_window():
    def _write_logged_window(root: Path, exp: str, run: str, attempt: str, first_step: int, last_step: int) -> Path:
        window_dir = root / exp / run / f"attempt={attempt}" / "logged" / f"w{first_step:08d}-{last_step:08d}"
        window_dir.mkdir(parents=True)
        steps = list(range(first_step, last_step + 1))
        table = pa.table(
            {
                "step": pa.array(steps, type=pa.uint32()),
                "name": ["lr"] * len(steps),
                "value": [0.1] * len(steps),
            },
            schema=pa.schema([("step", pa.uint32()), ("name", pa.string()), ("value", pa.float64())]),
        )
        pq.write_table(table, window_dir / "logged.parquet")
        manifest = {
            "schema_version": 1,
            "first_step": first_step,
            "last_step": last_step,
            "rows": len(steps),
            "dropped_rows": 0,
        }
        (window_dir / "manifest.json").write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))
        return window_dir

    return _write_logged_window


@pytest.fixture
def write_dump():
    def _write_dump(
        root: Path, exp: str, run: str, attempt: str, step: int, rank: int, tokens: int = 2
    ) -> Path:
        attempt_dir = AttemptDir(path=root / exp / run / f"attempt={attempt}", exp=exp, run=run, attempt=attempt)
        path = attempt_dir.dump_path(step, rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "step": [step] * tokens,
                "rank": [rank] * tokens,
                "microbatch": [0] * tokens,
                "point": ["resid"] * tokens,
                "layer": [0] * tokens,
                "token_index": list(range(tokens)),
                "token_id": [7] * tokens,
                "loss": [0.5] * tokens,
                "stream_id": [0] * tokens,
                "dtype": ["bf16"] * tokens,
                "values": [[1.0, 2.0, 3.0, 4.0] for _ in range(tokens)],
            },
            schema=_DUMP_SCHEMA,
        )
        pq.write_table(table, path, compression="zstd", use_byte_stream_split=["values"])
        return path

    return _write_dump
