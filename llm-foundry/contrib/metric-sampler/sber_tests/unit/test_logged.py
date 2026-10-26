from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import metric_sampler.logged as logged_module
from metric_sampler.logged import LoggedWriter

pytestmark = pytest.mark.unit

# A per-parameter monitor logs all the scalars of a step at once.
_PER_PARAMETER_SCALARS = 20000


def _read_window(window_dir: Path) -> tuple[dict, list[tuple[int, str, float]]]:
    manifest = json.loads((window_dir / "manifest.json").read_text())
    table = pq.read_table(window_dir / "logged.parquet")
    rows = list(zip(table["step"].to_pylist(), table["name"].to_pylist(), table["value"].to_pylist()))
    return manifest, rows


def test_close_with_no_data_writes_nothing(tmp_path: Path) -> None:
    writer = LoggedWriter(tmp_path, window_steps=10)

    writer.close()

    assert not (tmp_path / "logged").exists()


def test_add_accumulates_a_window_and_close_flushes_the_remainder(tmp_path: Path) -> None:
    writer = LoggedWriter(tmp_path, window_steps=2)

    writer.add(0, "lr", 0.1)
    writer.add(0, "grad_norm", 1.5)
    writer.add(1, "lr", 0.1)
    writer.add(2, "lr", 0.1)
    writer.close()

    first_dir = tmp_path / "logged" / "w00000000-00000001"
    second_dir = tmp_path / "logged" / "w00000002-00000002"
    assert first_dir.is_dir()
    assert second_dir.is_dir()

    manifest, rows = _read_window(first_dir)
    assert rows == [(0, "lr", 0.1), (0, "grad_norm", 1.5), (1, "lr", 0.1)]
    assert manifest["schema_version"] == 2
    assert manifest["first_step"] == 0
    assert manifest["last_step"] == 1
    assert manifest["rows"] == 3
    assert manifest["dropped_rows"] == 0
    datetime.fromisoformat(manifest["written_at"])

    manifest2, rows2 = _read_window(second_dir)
    assert rows2 == [(2, "lr", 0.1)]
    assert manifest2["first_step"] == 2
    assert manifest2["last_step"] == 2
    assert manifest2["rows"] == 1


def test_window_parquet_schema(tmp_path: Path) -> None:
    writer = LoggedWriter(tmp_path, window_steps=1)
    writer.add(5, "lr", 0.25)
    writer.close()

    table = pq.read_table(tmp_path / "logged" / "w00000005-00000005" / "logged.parquet")
    assert [str(field.type) for field in table.schema] == ["uint32", "string", "double", "uint32"]
    assert table["source_step"].to_pylist() == [None]


def test_preserves_original_steps_when_metrics_share_a_training_step(tmp_path: Path) -> None:
    writer = LoggedWriter(tmp_path, window_steps=10)
    writer.add(7, "loss", 2.5, source_step=6)
    writer.add(7, "throughput", 1200.0, source_step=7)
    writer.close()

    table = pq.read_table(tmp_path / "logged" / "w00000007-00000007" / "logged.parquet")
    assert table.to_pylist() == [
        {"step": 7, "name": "loss", "value": 2.5, "source_step": 6},
        {"step": 7, "name": "throughput", "value": 1200.0, "source_step": 7},
    ]


def test_single_step_windows_preserve_all_metrics_before_advancing(tmp_path: Path) -> None:
    writer = LoggedWriter(tmp_path, window_steps=1)
    expected = [(step, name, float(step + index)) for step in range(3)
                for index, name in enumerate(("loss", "throughput", "grad_norm"))]
    for step, name, value in expected:
        writer.add(step, name, value)
    writer.close()

    rows = []
    for directory in sorted((tmp_path / "logged").iterdir()):
        manifest, window_rows = _read_window(directory)
        assert manifest["rows"] == 3
        rows.extend(window_rows)
    assert rows == expected


@pytest.mark.parametrize("window_steps", [2, 1000])
def test_preserves_a_large_per_parameter_monitor_burst(tmp_path: Path, window_steps: int) -> None:
    writer = LoggedWriter(tmp_path, window_steps=window_steps)
    expected = [(0, "loss", 2.0)]
    expected.extend((1, f"weights/parameter_{index:05d}", index / 100.0) for index in range(_PER_PARAMETER_SCALARS))
    expected.append((2, "loss", 1.5))

    for step, name, value in expected:
        writer.add(step, name, value)
    writer.close()

    rows = []
    dropped = 0
    for directory in sorted((tmp_path / "logged").iterdir()):
        manifest, window_rows = _read_window(directory)
        rows.extend(window_rows)
        dropped += manifest["dropped_rows"]
        assert manifest["rows"] == len(window_rows)
    assert rows == expected
    assert dropped == 0


def test_multiple_windows_are_written_while_running(tmp_path: Path) -> None:
    writer = LoggedWriter(tmp_path, window_steps=1)

    for step in range(3):
        writer.add(step, "lr", float(step))
    writer.close()

    written = sorted(p.name for p in (tmp_path / "logged").iterdir())
    assert written == ["w00000000-00000000", "w00000001-00000001", "w00000002-00000002"]


def test_add_rejects_a_full_queue_explicitly_and_close_preserves_accepted_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = threading.Event()
    original_get = queue.Queue.get

    def _blocked_get(self, *args, **kwargs):
        ready.wait()
        return original_get(self, *args, **kwargs)

    monkeypatch.setattr(queue.Queue, "get", _blocked_get)
    monkeypatch.setattr(logged_module, "_QUEUE_SIZE", 2)

    writer = LoggedWriter(tmp_path, window_steps=1000)
    writer.add(0, "a", 1.0)
    writer.add(0, "b", 2.0)
    with pytest.raises(BufferError, match="queue is full; row was not accepted") as raised:
        writer.add(0, "c", 3.0)
    assert isinstance(raised.value.__cause__, queue.Full)

    ready.set()
    writer.close()

    manifest, rows = _read_window(tmp_path / "logged" / "w00000000-00000000")
    assert rows == [(0, "a", 1.0), (0, "b", 2.0)]
    assert manifest["rows"] == 2
    assert manifest["dropped_rows"] == 0


def test_background_thread_keeps_polling_while_idle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logged_module, "_POLL_INTERVAL_S", 0.01)
    writer = LoggedWriter(tmp_path, window_steps=1)

    time.sleep(0.05)
    writer.add(0, "lr", 0.1)
    writer.close()

    _, rows = _read_window(tmp_path / "logged" / "w00000000-00000000")
    assert rows == [(0, "lr", 0.1)]


def test_background_failure_is_visible_on_add_and_close_can_retry_all_accepted_rows(tmp_path: Path) -> None:
    logged_dir = tmp_path / "logged"
    logged_dir.mkdir()
    blocked_window = logged_dir / "w00000000-00000000"
    blocked_window.write_text("existing file")
    writer = LoggedWriter(tmp_path, window_steps=1)

    writer.add(0, "lr", 0.1)
    writer.add(1, "lr", 0.2)
    writer._thread.join(timeout=2)

    with pytest.raises(BufferError, match="background writer failed") as raised:
        writer.add(2, "lr", 0.3)
    assert isinstance(raised.value.__cause__, FileExistsError)
    with pytest.raises(BufferError, match="could not be persisted") as raised:
        writer.close()
    assert isinstance(raised.value.__cause__, FileExistsError)

    assert blocked_window.read_text() == "existing file"
    assert not (logged_dir / "w00000001-00000001").exists()
    blocked_window.unlink()

    with pytest.raises(BufferError, match="background writer failed") as raised:
        writer.close()
    assert isinstance(raised.value.__cause__, FileExistsError)
    writer.close()

    manifest, rows = _read_window(logged_dir / "w00000000-00000000")
    assert manifest["rows"] == 1
    assert rows == [(0, "lr", 0.1)]
    manifest, rows = _read_window(logged_dir / "w00000001-00000001")
    assert manifest["rows"] == 1
    assert rows == [(1, "lr", 0.2)]


def test_add_after_close_is_rejected_instead_of_queuing_rows_without_a_writer(tmp_path: Path) -> None:
    writer = LoggedWriter(tmp_path, window_steps=10)
    writer.close()

    with pytest.raises(BufferError, match="writer is closed"):
        writer.add(1, "loss", 1.0)

    assert not (tmp_path / "logged").exists()


def test_seventy_five_steps_keep_seven_complete_windows_and_a_five_step_tail(tmp_path: Path) -> None:
    writer = LoggedWriter(tmp_path, window_steps=10)
    expected = [(step, "loss", float(step)) for step in range(1, 76)]
    for row in expected:
        writer.add(*row)

    writer.close()

    directories = sorted((tmp_path / "logged").iterdir())
    assert [path.name for path in directories] == [
        *(f"w{first:08d}-{first + 9:08d}" for first in range(1, 71, 10)),
        "w00000071-00000075",
    ]
    actual = []
    for directory in directories:
        manifest, rows = _read_window(directory)
        assert manifest["rows"] == len(rows)
        assert manifest["dropped_rows"] == 0
        actual.extend(rows)
    assert actual == expected


def test_manifest_failure_keeps_rows_until_retry_and_drains_rows_left_in_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = threading.Event()
    original_get = queue.Queue.get

    def blocked_get(self, *args, **kwargs):
        ready.wait()
        return original_get(self, *args, **kwargs)

    monkeypatch.setattr(queue.Queue, "get", blocked_get)
    directory = tmp_path / "logged" / "w00000000-00000000"
    blocked_manifest = directory / "manifest.json"
    blocked_manifest.mkdir(parents=True)
    (blocked_manifest / "obstruction").write_bytes(b"manifest commit cannot replace this directory")
    writer = LoggedWriter(tmp_path, window_steps=1)
    expected = [(0, "loss", 2.0), (0, "loss", 1.5), (1, "loss", 1.0), (1, "norm", 0.7), (2, "loss", 0.5)]
    for row in expected:
        writer.add(*row)
    ready.set()
    writer._thread.join(timeout=2)
    assert pq.read_table(directory / "logged.parquet").to_pylist() == [
        {"step": 0, "name": "loss", "value": 2.0, "source_step": None},
        {"step": 0, "name": "loss", "value": 1.5, "source_step": None},
    ]
    (blocked_manifest / "obstruction").unlink()
    blocked_manifest.rmdir()

    with pytest.raises(BufferError, match="background writer failed"):
        writer.close()

    actual = []
    for window in sorted((tmp_path / "logged").iterdir()):
        manifest, rows = _read_window(window)
        assert manifest["rows"] == len(rows)
        actual.extend(rows)
    assert actual == expected
