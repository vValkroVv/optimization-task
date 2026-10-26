from __future__ import annotations

import importlib
import signal
from functools import partial
from pathlib import Path

import pytest

from metric_sampler.cloud.summaries import cli
from metric_sampler.cloud.summaries.pressure import IoPressure
from metric_sampler.cloud.summaries.slots import InsertSlots
from sber_tests.unit.cloud.summaries.fakes import START_TIMESTAMP, FakeClickHouse, FakeClock, failure

pytestmark = pytest.mark.unit

ROWS = r"^SELECT table, disk_name, count\(\)"
PARTS = r"^SELECT name FROM system\.parts"


class Fixture:
    """The command line on a fake ClickHouse, with its slot locks and the pressure file under `tmp_path`."""

    def __init__(self, tmp_path: Path, monkeypatch, rules: dict | None = None) -> None:
        self.timeouts: list[int] = []
        self.client = FakeClickHouse(rules)
        (tmp_path / "pressure").write_text("full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n", encoding="utf-8")
        monkeypatch.setattr(cli, "ClickHouse", self._client)
        monkeypatch.setattr(cli, "InsertSlots", partial(InsertSlots, directory=tmp_path))
        monkeypatch.setattr(cli, "IoPressure", lambda: IoPressure(tmp_path / "pressure"))
        monkeypatch.setattr(cli, "Clock", self._clock)

    def _client(self, timeout_seconds: int) -> FakeClickHouse:
        self.timeouts.append(timeout_seconds)
        return self.client

    def _clock(self) -> FakeClock:
        self.clock = FakeClock()
        return self.clock


def test_recompute_replaces_the_rows_of_a_range_of_one_summary_with_the_source_of_its_loop(
    tmp_path, monkeypatch, capsys,
) -> None:
    fixture = Fixture(tmp_path, monkeypatch)

    assert cli.main(["recompute", "layer_load", "e", "a", "1", "12"]) == 0

    inserts = fixture.client.statements
    assert len(inserts) == 2
    assert "FROM metric_sampler.layer_load_selected_range(exp_filter = 'e', run_filter = 'a'" in inserts[0]
    assert "step BETWEEN 1 AND 10) SETTINGS max_threads = 6" in inserts[0]
    assert "log_comment = 'recompute-layer_load'" in inserts[1]
    assert capsys.readouterr().out.splitlines() == [
        f"{START_TIMESTAMP} layer_load 1-10",
        f"{START_TIMESTAMP} layer_load 11-12",
        f"{START_TIMESTAMP} layer_load done 1-12",
    ]


def test_recompute_takes_the_chunk_size_and_the_number_of_slots_as_options(tmp_path, monkeypatch) -> None:
    fixture = Fixture(tmp_path, monkeypatch)

    assert cli.main(["recompute", "loss_rank", "e", "a", "1", "10", "--chunk-steps", "4", "--slots", "2"]) == 0

    assert [statement.split("BETWEEN ")[1].split(")")[0] for statement in fixture.client.statements] == [
        "1 AND 4", "5 AND 8", "9 AND 10",
    ]
    assert sorted(path.name for path in tmp_path.glob("summaries-slot-*.lock")) == ["summaries-slot-1.lock"]


def test_recompute_exits_with_one_when_a_chunk_fails(tmp_path, monkeypatch, capsys) -> None:
    Fixture(tmp_path, monkeypatch, {"^INSERT": [failure()]})

    assert cli.main(["recompute", "loss_rank", "e", "a", "1", "10"]) == 1

    assert capsys.readouterr().err.endswith(f"{START_TIMESTAMP} loss_rank 1-10 failed\n")


def test_every_command_takes_the_statement_timeout_that_curl_waits_for(tmp_path, monkeypatch) -> None:
    fixture = Fixture(tmp_path, monkeypatch)

    cli.main(["recompute", "loss_rank", "e", "a", "1", "1"])
    cli.main(["--timeout", "3000", "recompute", "loss_rank", "e", "a", "1", "1"])

    assert fixture.timeouts == [1900, 3000]


def test_recompute_names_the_summaries_of_the_catalog_and_nothing_else(tmp_path, monkeypatch, capsys) -> None:
    Fixture(tmp_path, monkeypatch)

    with pytest.raises(SystemExit) as raised:
        cli.main(["recompute", "no_such_summary", "e", "a", "1", "10"])

    assert raised.value.code == 2
    assert "invalid choice: 'no_such_summary'" in capsys.readouterr().err


def test_move_cold_moves_the_parts_of_a_run_of_the_tables_with_the_options_it_was_given(
    tmp_path, monkeypatch, capsys,
) -> None:
    fixture = Fixture(tmp_path, monkeypatch, {PARTS: ["p1\n"]})

    assert cli.main([
        "move-cold", "e", "r", "scalars", "expert_counts", "--parallel", "2", "--min-age-hours", "24",
        "--min-bytes", "100", "--max-io-full", "40",
    ]) == 0

    listing = fixture.client.matching(PARTS)[0]
    assert "toIntervalHour(24)" in listing
    assert "bytes_on_disk >= 100" in listing
    assert [line for line in capsys.readouterr().out.splitlines() if "before:" in line] == [
        f"{START_TIMESTAMP} scalars before:", f"{START_TIMESTAMP} expert_counts before:",
    ]


def test_move_cold_moves_everything_six_parts_at_a_time_by_default(tmp_path, monkeypatch) -> None:
    fixture = Fixture(tmp_path, monkeypatch, {PARTS: ["p1\n"]})

    assert cli.main(["move-cold", "e", "r", "scalars"]) == 0

    assert "toIntervalHour(0)\n    AND bytes_on_disk >= 0" in fixture.client.matching(PARTS)[0]
    assert fixture.client.matching(r"^ALTER TABLE") == [
        "ALTER TABLE metric_sampler.scalars MOVE PART 'p1' TO VOLUME 'cold'"
    ]


def test_run_starts_the_service_in_the_log_directory_and_stops_it_on_sigterm_and_sigint(
    tmp_path, monkeypatch,
) -> None:
    fixture = Fixture(tmp_path, monkeypatch)
    handlers = {}
    monkeypatch.setattr(cli.signal, "signal", lambda number, handler: handlers.setdefault(number, handler))
    created = []

    class RecordingService:
        def __init__(self, client, clock, log, log_directory) -> None:
            created.append((client, clock, log_directory))

        def run(self) -> None:
            created.append("run")

    monkeypatch.setattr(cli, "Service", RecordingService)

    assert cli.main(["run"]) == 0

    assert created == [(fixture.client, fixture.clock, Path("logs")), "run"]
    assert sorted(handlers) == sorted([signal.SIGTERM, signal.SIGINT])
    assert not fixture.clock.stopped.is_set()
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert fixture.clock.stopped.is_set()


def test_a_command_is_required(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main([])

    assert raised.value.code == 2
    assert "the following arguments are required: command" in capsys.readouterr().err


def test_dunder_main_module_imports_without_running_the_command_line() -> None:
    importlib.import_module("metric_sampler.cloud.summaries.__main__")
