from __future__ import annotations

import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from stand import check_byte_stream_split_read as check

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    directory = tmp_path_factory.mktemp("bss") / "out"
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        check.main(["--out", str(directory)])
    return directory, output.getvalue()


@pytest.mark.parametrize("kind", ["uint32", "float32"])
def test_the_split_file_holds_the_rows_of_the_plain_file_in_byte_streams(written, kind: str) -> None:
    directory, _ = written
    plain, split = (
        pq.read_table(directory / f"plain_{kind}.parquet"), pq.read_table(directory / f"split_{kind}.parquet")
    )

    def encodings(name: str) -> set[str]:
        row_group = pq.read_metadata(directory / name).row_group(0)
        return set(row_group.column(2).encodings)

    assert plain.num_rows == 20000 and split.num_rows == plain.num_rows
    assert split.schema.equals(plain.schema)
    assert split["step"].equals(plain["step"]) and split["rank"].equals(plain["rank"])
    assert split["values"].combine_chunks().offsets.equals(plain["values"].combine_chunks().offsets)
    # The float rows hold NaN, which no value comparison treats as equal to itself: compare the bits.
    assert (
        split["values"].combine_chunks().values.to_numpy().view(np.uint32)
        == plain["values"].combine_chunks().values.to_numpy().view(np.uint32)
    ).all()
    assert "BYTE_STREAM_SPLIT" in encodings(f"split_{kind}.parquet")
    assert "BYTE_STREAM_SPLIT" not in encodings(f"plain_{kind}.parquet")
    assert pq.read_metadata(directory / f"split_{kind}.parquet").num_row_groups == 3


def test_the_rows_cover_the_edges_of_the_types_and_the_empty_arrays(written) -> None:
    directory, _ = written
    counts = pq.read_table(directory / "split_uint32.parquet")["values"].to_pylist()
    floats = pq.read_table(directory / "split_float32.parquet")["values"].to_pylist()

    assert counts[1][:9] == [0, 1, 255, 256, 65535, 65536, 2**31 - 1, 2**31, 2**32 - 1]
    assert floats[1][2:8] == [
        1.401298464324817e-45, 3.4028234663852886e38, -3.4028234663852886e38, np.inf, -np.inf, floats[1][7]
    ]
    assert np.isnan(floats[1][7]) and str(floats[1][1]) == "-0.0"
    assert [index for index, row in enumerate(counts) if not row] == list(range(0, 20000, 17))
    assert len({len(row) for row in counts if row}) == 1


def test_the_expected_numbers_are_the_exact_sum_of_the_uint32_values_and_the_count_of_the_float_values(written) -> None:
    directory, out = written
    counts = pq.read_table(directory / "plain_uint32.parquet")["values"].to_pylist()
    floats = pq.read_table(directory / "plain_float32.parquet")["values"].to_pylist()

    assert f"uint32: 20000 rows, expected sum of values {sum(sum(row) for row in counts)}" in out
    assert f"float32: 20000 rows, expected number of values {sum(len(row) for row in floats)}" in out
    assert f"wrote 4 Parquet files and {directory / 'check.sql'}" in out


def test_check_sql_compares_both_encodings_under_both_readers_of_each_type(written) -> None:
    directory, _ = written

    *statements, end = (directory / "check.sql").read_text().split(";\n")

    assert len(statements) == 4 and end == ""
    for statement, (kind, setting) in zip(
        statements,
        [("uint32", "= 0"), ("uint32", "= 1"), ("float32", "= 0"), ("float32", "= 1")],
        strict=True,
    ):
        assert statement.endswith(f"SETTINGS input_format_parquet_use_native_reader_v3 {setting} FORMAT Vertical")
        assert (
            f"file('{directory}/plain_{kind}.parquet', 'Parquet', 'step UInt32, rank Int16, values Array(" in statement
        )
        assert (
            f"file('{directory}/split_{kind}.parquet', 'Parquet', 'step UInt32, rank Int16, values Array(" in statement
        )
        assert statement.count("sum(cityHash64(*))") == 2 and "plain = split AS identical" in statement
    assert statements[0].count("toUInt64(x)") == 2 and statements[2].count("sum(length(values))") == 2


def test_the_server_path_replaces_the_local_one_in_the_sql(tmp_path: Path) -> None:
    check.main(["--out", str(tmp_path), "--clickhouse-dir", "/var/lib/clickhouse/user_files/bss_check"])

    sql = (tmp_path / "check.sql").read_text()

    assert f"{Path('/var/lib/clickhouse/user_files/bss_check')}/plain_uint32.parquet" in sql
    assert str(tmp_path) not in sql


def test_the_script_runs_as_a_program(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["check_byte_stream_split_read.py", "--out", str(tmp_path / "program")])

    runpy.run_path(str(Path(check.__file__)), run_name="__main__")

    assert (tmp_path / "program" / "check.sql").is_file()
