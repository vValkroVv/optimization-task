import json
from pathlib import Path

import pytest

from node_runner.src.trace.events.model import TraceEvent
from node_runner.src.trace.events.writer import EventWriter

pytestmark = pytest.mark.unit


class _StuckThread:
    """A drain thread that never sees the sentinel."""

    def join(self, timeout=None) -> None:
        return None

    def is_alive(self) -> bool:
        return True


class _BrokenFile:
    """An NFS handle that fails one write and then behaves."""

    def __init__(self):
        self.lines: list[str] = []
        self.failed = False

    def write(self, line: str) -> None:
        if not self.failed:
            self.failed = True
            raise OSError("nfs hiccup")
        self.lines.append(line)

    def close(self) -> None:
        return None


class TestEventWriterTracePath:
    def test_trace_path_does_not_require_an_open_writer(self, tmp_path: Path):
        path = EventWriter.trace_path(tmp_path, "test", "job-a")
        assert path == tmp_path / "traces" / "test" / "events-job-a.jsonl"


class TestEventWriterAppendAcrossRestart:
    """The bug this guards: compression used to rename events.jsonl away on
    cleanup, so a restarted process reopening the same path got an empty
    file and silently lost the previous attempt's lines. Now it's plain
    append, forever."""

    def test_second_writer_appends_after_first_without_truncating(self, tmp_path: Path):
        events_path = tmp_path / "traces" / "mpimaster-0" / "events.jsonl"

        first = EventWriter(events_path)
        first.write(
            TraceEvent(
                kind="barrier",
                event="start",
                trace_id="job-a",
                name="REGISTRATION",
            )
        )
        first.close()

        second = EventWriter(events_path)
        second.write(
            TraceEvent(
                kind="barrier",
                event="start",
                trace_id="job-a",
                name="INFERENCE_START",
            )
        )
        second.close()

        lines = events_path.read_text().splitlines()
        assert len(lines) == 2
        first_row = json.loads(lines[0])
        second_row = json.loads(lines[1])
        assert first_row["name"] == "REGISTRATION"
        assert second_row["name"] == "INFERENCE_START"


class TestEventWriterDegradesQuietly:
    """Tracing is observability: a broken event file must never stop the job."""

    def test_a_writer_that_cannot_open_its_file_accepts_and_drops_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.trace.events.writer.Logger.warning", warnings.append
        )
        blocked = tmp_path / "events.jsonl"
        blocked.mkdir()

        writer = EventWriter(blocked)
        writer.write(
            TraceEvent(kind="barrier", event="start", trace_id="job-a", name="X")
        )
        writer.close()

        assert len(warnings) == 1
        assert "event writer disabled" in warnings[0]

    def test_a_failed_write_is_reported_without_killing_the_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.trace.events.writer.Logger.warning", warnings.append
        )
        writer = EventWriter(tmp_path / "events.jsonl", trace_id="job-a")
        broken = _BrokenFile()
        writer._file = broken

        writer.write(
            TraceEvent(kind="barrier", event="start", trace_id="job-a", name="X")
        )
        writer.write(
            TraceEvent(
                kind="barrier", event="end", trace_id="job-a", name="X", status="ok"
            )
        )
        writer.close()

        assert len(warnings) == 1
        assert [json.loads(line)["event"] for line in broken.lines] == ["end"]

    def test_a_drain_that_outlives_close_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.trace.events.writer.Logger.warning", warnings.append
        )
        writer = EventWriter(tmp_path / "events.jsonl", trace_id="job-a")
        writer._thread = _StuckThread()

        writer.close()

        assert "did not drain" in warnings[0]
