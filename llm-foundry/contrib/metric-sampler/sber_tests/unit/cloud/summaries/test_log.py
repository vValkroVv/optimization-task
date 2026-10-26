from __future__ import annotations

import pytest

from metric_sampler.cloud.summaries.log import FileLog, Log
from sber_tests.unit.cloud.summaries.fakes import CURL_ERROR, MEMORY_ERROR, START_TIMESTAMP, FakeClock, failure

pytestmark = pytest.mark.unit


def test_log_writes_lines_with_the_time_to_the_output_and_error_lines_to_the_error_stream(capsys) -> None:
    log = Log(FakeClock())

    log.line("test e a 1-2")
    log.error_line("test e a 1-2 failed")

    captured = capsys.readouterr()
    assert captured.out == f"{START_TIMESTAMP} test e a 1-2\n"
    assert captured.err == f"{START_TIMESTAMP} test e a 1-2 failed\n"


def test_log_passes_raw_text_to_each_stream_as_it_is(capsys) -> None:
    log = Log(FakeClock())

    log.output("scalars\tdefault\t3\n")
    log.error_output(CURL_ERROR)

    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("scalars\tdefault\t3\n", CURL_ERROR)


def test_log_reports_what_a_failed_statement_answered_body_first_and_then_error(capsys) -> None:
    Log(FakeClock()).report(failure(MEMORY_ERROR))

    captured = capsys.readouterr()
    assert (captured.out, captured.err) == (MEMORY_ERROR, CURL_ERROR)


def test_file_log_appends_both_streams_to_one_file_in_the_order_they_were_written(tmp_path) -> None:
    path = tmp_path / "loop.log"
    log = FileLog(path, FakeClock())

    log.line("started")
    log.report(failure(MEMORY_ERROR))
    log.error_line("failed")
    log.output("rows\n")
    log.error_output("error\n")

    assert path.read_text(encoding="utf-8").splitlines() == [
        f"{START_TIMESTAMP} started",
        MEMORY_ERROR.strip(),
        CURL_ERROR.strip(),
        f"{START_TIMESTAMP} failed",
        "rows",
        "error",
    ]


def test_file_log_continues_a_file_that_already_has_lines_and_recreates_one_that_was_rotated(tmp_path) -> None:
    path = tmp_path / "loop.log"
    path.write_text("older\n", encoding="utf-8")
    log = FileLog(path, FakeClock())

    log.output("first\n")
    path.unlink()
    log.output("second\n")

    assert path.read_text(encoding="utf-8") == "second\n"


def test_file_log_leaves_the_standard_streams_alone(tmp_path, capsys) -> None:
    FileLog(tmp_path / "loop.log", FakeClock()).line("quiet")

    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
