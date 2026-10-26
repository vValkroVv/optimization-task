import json
from pathlib import Path

import pytest

from node_runner.src.trace.events.stream import EventsStream
from node_runner.src.trace.events.writer import EventWriter

pytestmark = pytest.mark.unit


class TestEventsStreamEmit:
    """A trace event must never propagate. These calls sit inside the barrier
    handler, where nothing catches — a raise there parks every worker on a
    barrier until its 7-day retry runs out."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        yield
        EventsStream._writer = None

    def test_noop_without_writer(self):
        EventsStream._writer = None
        EventsStream.barrier_start("X")  # must not raise

    def test_barrier_event_is_written(self, tmp_path: Path):
        EventsStream._writer = EventWriter(tmp_path / "events.jsonl", trace_id="job-a")
        EventsStream.barrier_start("X")
        EventsStream.barrier_end("X")
        EventsStream._writer.close()

        rows = [
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text().splitlines()
        ]
        assert [(r["kind"], r["name"], r["event"]) for r in rows] == [
            ("barrier", "X", "start"),
            ("barrier", "X", "end"),
        ]
        assert rows[0]["trace_id"] == "job-a"

    @pytest.mark.parametrize("status", ["ok", "error"])
    def test_close_ends_the_job_event_with_the_run_status(
        self, tmp_path: Path, status: str
    ):
        EventsStream._writer = EventWriter(tmp_path / "events.jsonl", trace_id="job-a")

        EventsStream.close(status)

        rows = [
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text().splitlines()
        ]
        assert [(r["kind"], r["event"], r["status"]) for r in rows] == [
            ("job", "end", status)
        ]
        assert EventsStream._writer is None

    def test_invalid_event_is_dropped_not_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        warnings = []
        EventsStream._writer = EventWriter(tmp_path / "events.jsonl", trace_id="job-a")
        monkeypatch.setattr(
            "node_runner.src.trace.events.writer.Logger.warning", warnings.append
        )

        EventsStream._emit(kind="phase", event="start", name="single")  # phase missing
        EventsStream._writer.close()

        assert (tmp_path / "events.jsonl").read_text() == ""
        assert len(warnings) == 1
        assert "phase" in warnings[0]

    def test_shooter_event_records_error_and_reraises(self, tmp_path: Path):
        EventsStream._writer = EventWriter(tmp_path / "events.jsonl", trace_id="job-a")

        with pytest.raises(RuntimeError):
            with EventsStream.shooter("mmlu", phase="single"):
                raise RuntimeError("boom")
        EventsStream._writer.close()

        rows = [
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text().splitlines()
        ]
        assert rows[-1]["status"] == "error"
        assert rows[-1]["parent_event_id"] == "job"
        assert "boom" in rows[-1]["error"]
