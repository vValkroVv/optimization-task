"""Tests for the per-measurement file tracking used by the metrics cache."""

import os
from pathlib import Path

import pytest

from node_runner.src.job_dispatcher.measurement import MeasurementFiles

pytestmark = pytest.mark.unit
SECOND_NS = 1_000_000_000
RECORDED_MTIME_NS = 1_700_000_000 * SECOND_NS


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_recorded_file(
    path: Path, text: str, mtime_ns: int = RECORDED_MTIME_NS
) -> None:
    write_file(path, text)
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_changed_lists_the_files_written_after_the_baseline(tmp_path: Path) -> None:
    write_recorded_file(tmp_path / "untouched.json", "of a recorded measurement")
    write_recorded_file(tmp_path / "rewritten.json", "of a recorded measurement")

    measurement = MeasurementFiles(tmp_path, RECORDED_MTIME_NS)
    write_file(tmp_path / "rewritten.json", "rewritten by this run, with more bytes")
    write_file(tmp_path / "cache" / "alpha.jsonl", '{"uid": "1"}\n')

    assert measurement.changed() == [
        Path("cache/alpha.jsonl"),
        Path("rewritten.json"),
    ]


def test_changed_is_empty_when_the_run_wrote_nothing(tmp_path: Path) -> None:
    write_recorded_file(tmp_path / "cache" / "alpha.jsonl", '{"uid": "1"}\n')

    assert MeasurementFiles(tmp_path, RECORDED_MTIME_NS).changed() == []


def test_changed_is_empty_for_a_metrics_dir_that_does_not_exist_yet(
    tmp_path: Path,
) -> None:
    assert MeasurementFiles(tmp_path / "missing").changed() == []


def test_paths_resolve_the_changed_files_under_the_metrics_dir(tmp_path: Path) -> None:
    measurement = MeasurementFiles(tmp_path)
    write_file(tmp_path / "cache" / "alpha.jsonl", '{"uid": "1"}\n')

    assert measurement.paths() == {tmp_path / "cache" / "alpha.jsonl"}


def test_changed_keeps_the_first_scan_for_later_callers(tmp_path: Path) -> None:
    measurement = MeasurementFiles(tmp_path)
    write_file(tmp_path / "cache" / "alpha.jsonl", '{"uid": "1"}\n')
    first = measurement.changed()

    write_file(tmp_path / "cache" / "beta.jsonl", '{"uid": "2"}\n')

    assert first == [Path("cache/alpha.jsonl")]
    assert measurement.changed() == first
    assert measurement.paths() == {tmp_path / "cache" / "alpha.jsonl"}


def test_changed_skips_a_file_that_cannot_be_read(tmp_path: Path) -> None:
    measurement = MeasurementFiles(tmp_path)
    (tmp_path / "metrics.jsonl").symlink_to(tmp_path / "removed_by_the_shooter.jsonl")

    assert measurement.changed() == []
    assert measurement.paths() == set()


def test_changed_includes_the_files_left_unrecorded_by_a_died_job(
    tmp_path: Path,
) -> None:
    write_recorded_file(tmp_path / "recorded.json", "of a recorded measurement")
    write_recorded_file(
        tmp_path / "unrecorded.json",
        "computed after the last recorded measurement",
        RECORDED_MTIME_NS + SECOND_NS,
    )

    measurement = MeasurementFiles(tmp_path, RECORDED_MTIME_NS)

    assert measurement.changed() == [Path("unrecorded.json")]
    assert measurement.paths() == {tmp_path / "unrecorded.json"}


def test_changed_includes_everything_when_no_measurement_was_recorded_yet(
    tmp_path: Path,
) -> None:
    write_recorded_file(tmp_path / "left_by_the_died_job.json", "never recorded")

    assert MeasurementFiles(tmp_path).changed() == [Path("left_by_the_died_job.json")]


def test_mtime_ns_is_the_newest_mtime_of_the_metrics_dir(tmp_path: Path) -> None:
    write_recorded_file(tmp_path / "older.json", "older")
    write_recorded_file(
        tmp_path / "cache" / "newer.jsonl", "newer", RECORDED_MTIME_NS + SECOND_NS
    )

    assert MeasurementFiles(tmp_path).mtime_ns() == RECORDED_MTIME_NS + SECOND_NS


def test_mtime_ns_is_zero_for_an_empty_metrics_dir(tmp_path: Path) -> None:
    assert MeasurementFiles(tmp_path).mtime_ns() == 0
