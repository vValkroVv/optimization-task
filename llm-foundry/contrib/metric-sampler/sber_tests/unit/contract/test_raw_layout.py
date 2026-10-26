from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from metric_sampler.contract.raw_layout import (
    AttemptDir,
    Dump,
    LoggedWindow,
    RawRoot,
    Window,
    attempt_dir_name,
    dump_relative_path,
    family_file_name,
    node_dir,
    node_dir_name,
    window_name,
    write_atomically,
)

pytestmark = pytest.mark.unit


def test_atomic_write_failure_removes_temporary_file_and_allows_retry(tmp_path):
    target = tmp_path / "payload"
    target.mkdir()
    with pytest.raises(OSError):
        write_atomically(target, b"durable data")
    assert list(tmp_path.iterdir()) == [target]
    target.rmdir()
    write_atomically(target, b"durable data")
    assert target.read_bytes() == b"durable data"


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_attempt_dir_complete_windows_skips_incomplete_and_foreign_entries(tmp_path: Path) -> None:
    attempt_path = tmp_path / "expA" / "run1" / "attempt=0"
    _touch(attempt_path / "node=node1" / "w00000000-00000049" / "manifest.json", "{}")
    _touch(attempt_path / "node=node1" / "w00000000-00000049" / "scalars.parquet")
    _touch(attempt_path / "node=node1" / "w00000050-00000099" / "scalars.parquet")
    _touch(attempt_path / "node=node1" / "aggregator.log", "")
    _touch(attempt_path / "node=node2" / "w00000000-00000049" / "manifest.json", "{}")
    _touch(attempt_path / "metrics.json", "[]")
    (attempt_path / "not_a_node_dir").mkdir()

    attempt = AttemptDir(path=attempt_path, exp="expA", run="run1", attempt="0")
    windows = attempt.complete_windows()

    assert [(w.node, w.first_step, w.last_step) for w in windows] == [
        ("node1", 0, 49),
        ("node2", 0, 49),
    ]
    assert windows[0].name == "w00000000-00000049"
    assert windows[0].relative_dir == "node=node1/w00000000-00000049"
    assert windows[0].manifest_path == attempt_path / "node=node1" / "w00000000-00000049" / "manifest.json"
    assert windows[0].family_path("scalars") == (
        attempt_path / "node=node1" / "w00000000-00000049" / "scalars.parquet"
    )
    assert attempt.metrics_path == attempt_path / "metrics.json"
    assert attempt.streams_path == attempt_path / "streams.json"
    assert attempt.monitor_parameters_path == attempt_path / "monitor_parameters.json"
    assert attempt.monitor_expert_ownership_path == attempt_path / "monitor_expert_ownership.json"


def test_attempt_dir_ignores_non_directory_window_and_node_entries(tmp_path: Path) -> None:
    attempt_path = tmp_path / "expA" / "run1" / "attempt=0"
    (attempt_path / "node=node1").mkdir(parents=True)
    (attempt_path / "node=node1" / "w00000000-00000049").write_text("not a directory")
    (attempt_path / "node=node2").write_text("not a directory either")

    attempt = AttemptDir(path=attempt_path, exp="expA", run="run1", attempt="0")

    assert attempt.complete_windows() == []


def test_attempt_dir_collected_path(tmp_path: Path) -> None:
    raw_root = RawRoot(tmp_path / "raw")
    attempt = AttemptDir(path=tmp_path / "raw" / "expA" / "run1" / "attempt=0", exp="expA", run="run1", attempt="0")
    window = Window(path=tmp_path / "irrelevant", node="node1", first_step=0, last_step=49)

    assert attempt.collected_path(raw_root, window) == (
        tmp_path / "raw" / "_collected" / "expA" / "run1" / "attempt=0" / "node=node1" / "w00000000-00000049"
    )


def test_attempt_dir_collected_path_for_logged_window(tmp_path: Path) -> None:
    raw_root = RawRoot(tmp_path / "raw")
    attempt = AttemptDir(path=tmp_path / "raw" / "expA" / "run1" / "attempt=0", exp="expA", run="run1", attempt="0")
    window = LoggedWindow(path=tmp_path / "irrelevant", first_step=0, last_step=49)

    assert attempt.collected_path(raw_root, window) == (
        tmp_path / "raw" / "_collected" / "expA" / "run1" / "attempt=0" / "logged" / "w00000000-00000049"
    )


def test_logged_window_properties(tmp_path: Path) -> None:
    window = LoggedWindow(path=tmp_path / "w00000000-00000049", first_step=0, last_step=49)

    assert window.name == "w00000000-00000049"
    assert window.relative_dir == "logged/w00000000-00000049"
    assert window.manifest_path == tmp_path / "w00000000-00000049" / "manifest.json"
    assert window.data_path == tmp_path / "w00000000-00000049" / "logged.parquet"


def test_attempt_dir_complete_logged_windows_without_logged_dir(tmp_path: Path) -> None:
    attempt_path = tmp_path / "expA" / "run1" / "attempt=0"
    _touch(attempt_path / "metrics.json", "[]")

    attempt = AttemptDir(path=attempt_path, exp="expA", run="run1", attempt="0")

    assert attempt.complete_logged_windows() == []


def test_attempt_dir_complete_logged_windows_skips_incomplete_and_foreign_entries(tmp_path: Path) -> None:
    attempt_path = tmp_path / "expA" / "run1" / "attempt=0"
    _touch(attempt_path / "logged" / "w00000000-00000049" / "manifest.json", "{}")
    _touch(attempt_path / "logged" / "w00000000-00000049" / "logged.parquet")
    _touch(attempt_path / "logged" / "w00000050-00000099" / "logged.parquet")
    (attempt_path / "logged" / "not_a_window_dir").mkdir()
    (attempt_path / "logged" / "w00000100-00000149").write_text("not a directory")

    attempt = AttemptDir(path=attempt_path, exp="expA", run="run1", attempt="0")
    windows = attempt.complete_logged_windows()

    assert [(w.first_step, w.last_step) for w in windows] == [(0, 49)]
    assert windows[0].relative_dir == "logged/w00000000-00000049"


def test_attempt_dir_dump_path_round_trips_through_complete_dumps(tmp_path: Path) -> None:
    attempt = AttemptDir(path=tmp_path / "expA" / "run1" / "attempt=0", exp="expA", run="run1", attempt="0")
    path = attempt.dump_path(step=1200, rank=17)
    _touch(path)

    assert path == tmp_path / "expA" / "run1" / "attempt=0" / "dumps" / "step=00001200" / "rank=00017.parquet"
    assert attempt.complete_dumps() == [Dump(path=path, step=1200, rank=17)]
    assert attempt.complete_dumps()[0].relative_path == "dumps/step=00001200/rank=00017.parquet"


def test_attempt_dir_complete_dumps_without_dumps_dir(tmp_path: Path) -> None:
    attempt = AttemptDir(path=tmp_path / "attempt=0", exp="expA", run="run1", attempt="0")

    assert attempt.complete_dumps() == []


def test_attempt_dir_complete_dumps_skips_temporary_and_foreign_entries(tmp_path: Path) -> None:
    attempt_path = tmp_path / "expA" / "run1" / "attempt=0"
    _touch(attempt_path / "dumps" / "step=00000100" / "rank=00000.parquet")
    _touch(attempt_path / "dumps" / "step=00000100" / "rank=00001.parquet.x1y2.tmp")
    (attempt_path / "dumps" / "step=00000100" / "rank=00002.parquet").mkdir()
    _touch(attempt_path / "dumps" / "step=00000200" / "rank=00003.parquet")
    _touch(attempt_path / "dumps" / "not_a_step" / "rank=00004.parquet")
    _touch(attempt_path / "dumps" / "step=00000300")

    attempt = AttemptDir(path=attempt_path, exp="expA", run="run1", attempt="0")

    assert [(d.step, d.rank) for d in attempt.complete_dumps()] == [(100, 0), (200, 3)]


def test_raw_root_attempt_dirs_skips_collected_and_foreign_entries(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _touch(root / "expA" / "run1" / "attempt=0" / "metrics.json", "[]")
    _touch(root / "expA" / "run1" / "not_an_attempt_dir" / "metrics.json", "[]")
    _touch(root / "_collected" / "expA" / "run1" / "attempt=0" / "node=node1" / "w00000000-00000049" / "manifest.json",
           "{}")
    (root / "README").write_text("not a directory")
    (root / "expA" / "notes.txt").write_text("")

    attempt_dirs = RawRoot(root).attempt_dirs()

    assert [(a.exp, a.run, a.attempt) for a in attempt_dirs] == [("expA", "run1", "0")]


@pytest.mark.parametrize(
    ("written", "dumps", "expected"),
    [
        ([], True, False),
        (["node=node1/w00000050-00000099/scalars.parquet"], True, False),
        (["node=node1/w00000000-00000049/manifest.json"], False, True),
        (["logged/w00000000-00000049/manifest.json"], False, True),
        (["dumps/step=00000100/rank=00000.parquet"], True, True),
        (["dumps/step=00000100/rank=00000.parquet"], False, False),
    ],
)
def test_raw_root_has_uncollected_complete_windows_and_dumps_when_asked(
    tmp_path: Path, written: list[str], dumps: bool, expected: bool
) -> None:
    root = tmp_path / "raw"
    attempt_path = root / "expA" / "run1" / "attempt=0"
    _touch(attempt_path / "metrics.json", "[]")
    _touch(root / "_collected" / "expA" / "run1" / "attempt=0" / "node=node1" / "w00000000-00000009" / "manifest.json",
           "{}")
    for relative in written:
        _touch(attempt_path / relative)

    assert RawRoot(root).has_uncollected(dumps=dumps) is expected


def test_raw_root_attempt_dirs_skips_non_directory_run_and_attempt_entries(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    exp_dir = root / "expA"
    exp_dir.mkdir(parents=True)
    (exp_dir / "run1.txt").write_text("")
    run_dir = exp_dir / "run1"
    _touch(run_dir / "attempt=0" / "metrics.json", "[]")
    (run_dir / "attempt=1").write_text("not a directory")

    attempt_dirs = RawRoot(root).attempt_dirs()

    assert [(a.exp, a.run, a.attempt) for a in attempt_dirs] == [("expA", "run1", "0")]


def test_write_atomically_uses_a_distinct_temporary_file_per_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    renamed_from = []
    real_replace = os.replace

    def recording_replace(src, dst):
        renamed_from.append(Path(src))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", recording_replace)
    target = tmp_path / "metrics.json"

    write_atomically(target, b"first")
    write_atomically(target, b"second")

    assert target.read_bytes() == b"second"
    assert len(set(renamed_from)) == 2
    assert all(src.parent == tmp_path and src.name.startswith("metrics.json.") for src in renamed_from)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["metrics.json"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX group permissions are not available")
@pytest.mark.parametrize("existing", [False, True])
def test_write_atomically_allows_group_reads_under_restrictive_umask(tmp_path: Path, existing: bool) -> None:
    target = tmp_path / "metrics.json"
    if existing:
        target.write_bytes(b"previous")
        target.chmod(0o600)

    previous_umask = os.umask(0o077)
    try:
        write_atomically(target, b"complete metrics")
    finally:
        os.umask(previous_umask)

    assert target.read_bytes() == b"complete metrics"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert list(tmp_path.iterdir()) == [target]


def test_layout_names_are_the_ones_every_stage_reads_and_writes(tmp_path) -> None:
    assert attempt_dir_name("123-0") == "attempt=123-0"
    assert node_dir_name("gpu7") == "node=gpu7"
    assert node_dir(tmp_path, "123-0", "gpu7") == tmp_path / "attempt=123-0" / "node=gpu7"
    assert window_name(1, 40) == "w00000001-00000040"
    assert family_file_name("router_vectors") == "router_vectors.parquet"
    assert dump_relative_path(1200, 17) == "dumps/step=00001200/rank=00017.parquet"
