from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metric_sampler.capture.dump_buffer import DumpEntry
from metric_sampler.contract.raw_layout import AttemptDir
from metric_sampler.node.dump_writer import DumpWriter
from sber_tests.unit.node.conftest import fill_dump_buffer

pytestmark = pytest.mark.unit

_DUMP_TOKENS = 4
_DUMP_HIDDEN = 3
_DUMP_ENTRIES = (DumpEntry("residual", 0), DumpEntry("final_norm", -1))


@pytest.fixture
def attempt_dir(tmp_path: Path) -> AttemptDir:
    return AttemptDir(path=tmp_path / "exp" / "run" / "attempt=a1", exp="exp", run="run", attempt="a1")


def test_write_ready_dumps_writes_a_parquet_file_matching_the_contract(dump_buffers, attempt_dir) -> None:
    buffer = dump_buffers(_DUMP_ENTRIES, _DUMP_TOKENS, _DUMP_HIDDEN)
    residual = np.arange(_DUMP_TOKENS * _DUMP_HIDDEN, dtype=np.float32).reshape(_DUMP_TOKENS, _DUMP_HIDDEN)
    final_norm = (residual * 10.0) + 1.0
    fill_dump_buffer(
        buffer,
        step=5,
        token_ids=[10, 11, 12, 13],
        loss=[0.5, float("nan"), 1.5, 2.5],
        stream_ids=[0, 0, 1, 1],
        entries_values=[(0, residual), (2, final_norm)],
    )

    DumpWriter(attempt_dir, {7: buffer}).write_ready_dumps()

    assert buffer.state == 0  # released back to FREE

    dump_path = attempt_dir.path / "dumps" / "step=00000005" / "rank=00007.parquet"
    assert dump_path.exists()
    table = pq.read_table(dump_path, partitioning=None)
    assert table.schema.field("step").type == pa.int64()
    assert table.schema.field("rank").type == pa.int32()
    assert table.schema.field("microbatch").type == pa.int16()
    assert table.schema.field("point").type == pa.string()
    assert table.schema.field("layer").type == pa.int32()
    assert table.schema.field("token_index").type == pa.int32()
    assert table.schema.field("token_id").type == pa.int32()
    assert table.schema.field("loss").type == pa.float32()
    assert table.schema.field("stream_id").type == pa.int32()
    assert table.schema.field("dtype").type == pa.string()
    assert table.schema.field("values").type == pa.list_(pa.float32(), _DUMP_HIDDEN)
    metadata = pq.ParquetFile(dump_path).metadata
    assert "BYTE_STREAM_SPLIT" in next(
        metadata.row_group(0).column(i).encodings for i in range(metadata.num_columns)
        if metadata.row_group(0).column(i).path_in_schema.startswith("values.")
    )

    rows = table.to_pylist()
    assert len(rows) == 2 * _DUMP_TOKENS
    residual_rows = [row for row in rows if row["point"] == "residual"]
    final_norm_rows = [row for row in rows if row["point"] == "final_norm"]
    assert len(residual_rows) == _DUMP_TOKENS
    assert len(final_norm_rows) == _DUMP_TOKENS

    for row in rows:
        assert row["step"] == 5
        assert row["rank"] == 7
        assert row["microbatch"] == 0

    by_index = sorted(residual_rows, key=lambda row: row["token_index"])
    assert [row["token_index"] for row in by_index] == [0, 1, 2, 3]
    assert [row["token_id"] for row in by_index] == [10, 11, 12, 13]
    assert [row["stream_id"] for row in by_index] == [0, 0, 1, 1]
    assert by_index[0]["loss"] == pytest.approx(0.5)
    assert np.isnan(by_index[1]["loss"])
    assert by_index[0]["layer"] == 0
    assert by_index[0]["dtype"] == "bfloat16"
    np.testing.assert_allclose(by_index[0]["values"], residual[0].tolist(), atol=1e-2)
    np.testing.assert_allclose(by_index[3]["values"], residual[3].tolist(), atol=1e-2)

    final_norm_row = next(row for row in final_norm_rows if row["token_index"] == 0)
    assert final_norm_row["layer"] == -1
    assert final_norm_row["dtype"] == "float32"
    np.testing.assert_allclose(final_norm_row["values"], final_norm[0].tolist())


def test_write_ready_dumps_skips_a_buffer_that_is_not_ready(dump_buffers, attempt_dir) -> None:
    buffer = dump_buffers(_DUMP_ENTRIES, _DUMP_TOKENS, _DUMP_HIDDEN)

    DumpWriter(attempt_dir, {0: buffer}).write_ready_dumps()

    assert not (attempt_dir.path / "dumps").exists()


def test_write_ready_dumps_omits_entries_that_were_never_captured(dump_buffers, attempt_dir) -> None:
    buffer = dump_buffers(_DUMP_ENTRIES, _DUMP_TOKENS, _DUMP_HIDDEN)
    residual = np.zeros((_DUMP_TOKENS, _DUMP_HIDDEN), dtype=np.float32)
    fill_dump_buffer(
        buffer, step=1, token_ids=[0, 1, 2, 3], loss=[0.0] * 4, stream_ids=[0] * 4,
        entries_values=[(2, residual), (-1, residual)],
    )

    DumpWriter(attempt_dir, {0: buffer}).write_ready_dumps()

    table = pq.read_table(attempt_dir.path / "dumps" / "step=00000001" / "rank=00000.parquet", partitioning=None)
    assert set(table.column("point").to_pylist()) == {"residual"}


def test_write_ready_dumps_writes_nothing_when_no_entry_was_captured(dump_buffers, attempt_dir) -> None:
    buffer = dump_buffers(_DUMP_ENTRIES, _DUMP_TOKENS, _DUMP_HIDDEN)
    zeros = np.zeros((_DUMP_TOKENS, _DUMP_HIDDEN), dtype=np.float32)
    fill_dump_buffer(
        buffer, step=1, token_ids=[0, 1, 2, 3], loss=[0.0] * 4, stream_ids=[0] * 4,
        entries_values=[(-1, zeros), (-1, zeros)],
    )

    DumpWriter(attempt_dir, {0: buffer}).write_ready_dumps()

    assert not (attempt_dir.path / "dumps").exists()
    assert buffer.state == 0


def test_a_ready_buffer_is_kept_until_a_failed_write_is_retried(dump_buffers, attempt_dir) -> None:
    buffer = dump_buffers(_DUMP_ENTRIES, _DUMP_TOKENS, _DUMP_HIDDEN)
    zeros = np.zeros((_DUMP_TOKENS, _DUMP_HIDDEN), dtype=np.float32)
    fill_dump_buffer(
        buffer, step=1, token_ids=[0, 1, 2, 3], loss=[0.0] * 4, stream_ids=[0] * 4,
        entries_values=[(2, zeros), (2, zeros)],
    )
    writer = DumpWriter(attempt_dir, {0: buffer})
    destination = attempt_dir.path / "dumps" / "step=00000001" / "rank=00000.parquet"
    destination.mkdir(parents=True)

    with pytest.raises(OSError):
        writer.write_ready_dumps()

    assert buffer.is_ready
    destination.rmdir()
    writer.write_ready_dumps()

    assert not buffer.is_ready
    rows = pq.ParquetFile(destination).read().to_pylist()
    assert [(row["step"], row["rank"], row["token_index"], row["values"]) for row in rows] == [
        (1, 0, token, [0.0] * _DUMP_HIDDEN) for _ in _DUMP_ENTRIES for token in range(_DUMP_TOKENS)
    ]


def test_write_ready_dumps_writes_a_file_for_every_ready_rank_and_leaves_the_others_alone(
    dump_buffers, attempt_dir,
) -> None:
    buffers = {rank: dump_buffers(_DUMP_ENTRIES, _DUMP_TOKENS, _DUMP_HIDDEN) for rank in (0, 1, 2)}
    zeros = np.zeros((_DUMP_TOKENS, _DUMP_HIDDEN), dtype=np.float32)
    for rank in (0, 2):
        fill_dump_buffer(
            buffers[rank], step=rank + 1, token_ids=[0, 1, 2, 3], loss=[0.0] * 4, stream_ids=[0] * 4,
            entries_values=[(2, zeros), (2, zeros)],
        )

    DumpWriter(attempt_dir, buffers).write_ready_dumps()

    assert sorted(path.relative_to(attempt_dir.path).as_posix() for path in attempt_dir.path.rglob("*.parquet")) == [
        "dumps/step=00000001/rank=00000.parquet", "dumps/step=00000003/rank=00002.parquet",
    ]
    assert [buffer.is_ready for buffer in buffers.values()] == [False, False, False]
