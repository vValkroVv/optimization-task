from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from metric_sampler.contract.metrics import Family
from metric_sampler.contract.raw_layout import RawRoot
from metric_sampler.island import cli

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _stub_flock(monkeypatch: pytest.MonkeyPatch) -> None:
    # fcntl (used by _flock_exclusive_nonblocking) exists on POSIX only; the call succeeds here, and the lock
    # tests below replace it.
    monkeypatch.setattr(cli, "_flock_exclusive_nonblocking", lambda file: None)


@pytest.fixture(autouse=True)
def _stub_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    # faulthandler writes to a file descriptor, which pytest's captured stderr does not have.
    monkeypatch.setattr(cli.faulthandler, "dump_traceback_later", lambda timeout, exit: None)
    monkeypatch.setattr(cli.faulthandler, "cancel_dump_traceback_later", lambda: None)


@pytest.fixture
def rclone_conf(tmp_path: Path) -> Path:
    path = tmp_path / "rclone.conf"
    path.write_text(
        "[metadata-s3]\n"
        "access_key_id = test-key\n"
        "secret_access_key = test-secret\n"
        "[dumps-s3]\n"
        "access_key_id = dump-key\n"
        "secret_access_key = dump-secret\n"
    )
    return path


@pytest.fixture
def use_session(monkeypatch: pytest.MonkeyPatch):
    """Makes `cli.main` collect with the given session instead of a chdb one."""

    def use(session):
        monkeypatch.setattr(cli, "_build_session", lambda chdb_config: session)
        return session

    return use


@pytest.fixture
def collector_arguments(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """What `cli.main` creates its collector from, recorded when the collector is created."""
    arguments: dict[str, Any] = {}
    original_init = cli.Collector.__init__

    def recording_init(self, session, target, collected_retention_s, dump_target=None, *, shard, scratch_dir):
        arguments.update(
            target=target, collected_retention_s=collected_retention_s, dump_target=dump_target, shard=shard,
            scratch_dir=scratch_dir,
        )
        original_init(self, session, target, collected_retention_s, dump_target, shard, scratch_dir)

    monkeypatch.setattr(cli.Collector, "__init__", recording_init)
    return arguments


def _base_args(raw_root: RawRoot, rclone_conf: Path) -> list[str]:
    return [
        "--raw-root", str(raw_root.path),
        "--rclone-conf", str(rclone_conf),
        "--remote-section", "metadata-s3",
        "--endpoint", "https://s3.example.invalid",
        "--bucket", "metrics-bucket",
    ]


def _raise_already_locked(file) -> None:
    raise OSError("already locked")


def test_main_runs_once_and_prints_receipts(
    raw_root: RawRoot, rclone_conf: Path, capsys: pytest.CaptureFixture[str], fake_session_factory, use_session,
    write_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    write_window(raw_root.path, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    session = use_session(fake_session_factory(stats={"scalars": (4, 0, 49)}))

    cli.main(_base_args(raw_root, rclone_conf))

    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    family, key, rows = out[0].split(" ")
    assert family == "scalars"
    assert key.startswith("metric-sampler/v3/family=scalars/exp=expA/run=run1/attempt=0/part-00000000-00000049-")
    assert rows == "4"
    assert [sql for sql, _ in session.queries if sql.startswith("SET ")] == [
        "SET max_threads=4, max_parsing_threads=4, max_memory_usage=12884901888, max_query_size=67108864, "
        "input_format_parquet_max_block_size=8192, "
        "output_format_parquet_row_group_size=1048576, "
        "output_format_parquet_row_group_size_bytes=67108864, "
        "output_format_parquet_parallel_encoding=1"
    ]


def test_main_keeps_a_watchdog_armed_only_while_a_cycle_runs(
    raw_root: RawRoot, rclone_conf: Path, monkeypatch: pytest.MonkeyPatch, fake_session_factory, use_session,
    write_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    write_window(raw_root.path, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    watchdog: list[tuple[int, bool] | None] = [None]
    armed_during_queries = set()

    class WatchedSession(fake_session_factory):
        def query(self, sql, fmt=None):
            if not sql.startswith("SET "):
                armed_during_queries.add(watchdog[0])
            return super().query(sql, fmt)

    use_session(WatchedSession(stats={"scalars": (4, 0, 49)}))
    monkeypatch.setattr(
        cli.faulthandler, "dump_traceback_later", lambda timeout, exit: watchdog.__setitem__(0, (timeout, exit))
    )
    monkeypatch.setattr(cli.faulthandler, "cancel_dump_traceback_later", lambda: watchdog.__setitem__(0, None))

    cli.main(_base_args(raw_root, rclone_conf))

    assert armed_during_queries == {(30 * 60, True)}
    assert watchdog == [None]


def test_main_uses_default_prefix_and_retention(
    raw_root: RawRoot, rclone_conf: Path, fake_session_factory, use_session, collector_arguments,
) -> None:
    raw_root.path.mkdir()
    use_session(fake_session_factory())

    cli.main(_base_args(raw_root, rclone_conf))

    assert collector_arguments["target"].prefix == "metric-sampler/v3"
    assert collector_arguments["collected_retention_s"] == 24 * 3600
    assert collector_arguments["dump_target"] is None
    assert collector_arguments["shard"] == cli.Shard(0, 1)
    assert collector_arguments["scratch_dir"] is None
    assert (raw_root.path / ".collector.lock").exists()


def test_main_passes_the_shard_and_takes_that_shards_lock(
    raw_root: RawRoot, rclone_conf: Path, fake_session_factory, use_session, collector_arguments,
) -> None:
    raw_root.path.mkdir()
    use_session(fake_session_factory())

    cli.main([*_base_args(raw_root, rclone_conf), "--shard-index", "1", "--shard-count", "2"])

    assert collector_arguments["shard"] == cli.Shard(1, 2)
    assert (raw_root.path / ".collector.1.lock").exists()
    assert not (raw_root.path / ".collector.lock").exists()


@pytest.mark.parametrize(("index", "count"), [("2", "2"), ("-1", "2"), ("0", "0")])
def test_main_rejects_a_shard_index_outside_the_shard_count(
    raw_root: RawRoot, rclone_conf: Path, index: str, count: str
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([*_base_args(raw_root, rclone_conf), "--shard-index", index, "--shard-count", count])

    assert exc_info.value.code == 2


def test_main_passes_custom_prefix_and_retention_hours(
    raw_root: RawRoot, rclone_conf: Path, fake_session_factory, use_session, collector_arguments,
) -> None:
    raw_root.path.mkdir()
    use_session(fake_session_factory())

    cli.main([*_base_args(raw_root, rclone_conf), "--prefix", "custom/v9", "--collected-retention-hours", "1"])

    assert collector_arguments["target"].prefix == "custom/v9"
    assert collector_arguments["collected_retention_s"] == 3600
    assert collector_arguments["dump_target"] is None


def test_main_builds_a_dump_target_from_the_dump_specific_args(
    raw_root: RawRoot, rclone_conf: Path, fake_session_factory, use_session, collector_arguments,
) -> None:
    raw_root.path.mkdir()
    use_session(fake_session_factory())

    cli.main(
        [
            *_base_args(raw_root, rclone_conf),
            "--dump-remote-section", "dumps-s3",
            "--dump-bucket", "dumps-bucket",
            "--dump-prefix", "metric-sampler-dumps",
        ]
    )

    dump_target = collector_arguments["dump_target"]
    assert dump_target is not None
    assert dump_target.endpoint == "https://s3.example.invalid"
    assert dump_target.bucket == "dumps-bucket"
    assert dump_target.prefix == "metric-sampler-dumps"
    assert dump_target.credentials.access_key == "dump-key"
    assert dump_target.credentials.secret_key == "dump-secret"


def test_main_leaves_the_dump_target_unset_when_only_the_bucket_is_given(
    raw_root: RawRoot, rclone_conf: Path, fake_session_factory, use_session, collector_arguments,
) -> None:
    raw_root.path.mkdir()
    use_session(fake_session_factory())

    cli.main([*_base_args(raw_root, rclone_conf), "--dump-bucket", "dumps-bucket"])

    assert collector_arguments["dump_target"] is None


class _StopLoop(Exception):
    pass


def test_main_loops_and_sleeps_between_passes(
    raw_root: RawRoot, rclone_conf: Path, monkeypatch: pytest.MonkeyPatch, fake_session_factory, use_session
) -> None:
    raw_root.path.mkdir()
    use_session(fake_session_factory())

    sleep_calls: list[int] = []

    def fake_sleep(seconds: int) -> None:
        sleep_calls.append(seconds)
        raise _StopLoop

    monkeypatch.setattr(cli.time, "sleep", fake_sleep)

    with pytest.raises(_StopLoop):
        cli.main([*_base_args(raw_root, rclone_conf), "--loop-seconds", "30"])

    assert sleep_calls == [30]


def test_main_drains_bounded_batches_for_each_attempt_before_sleeping(
    raw_root: RawRoot, rclone_conf: Path, monkeypatch: pytest.MonkeyPatch, fake_session_factory, use_session,
    write_window, write_metrics_registry,
) -> None:
    originals = {}
    for attempt in ["0", "1"]:
        write_metrics_registry(raw_root.path, "expA", "run1", attempt)
        for step in range(129):
            window = write_window(
                raw_root.path, "expA", "run1", attempt, "host1", step, step, families=(Family.SCALARS,),
            )
            originals[window.relative_to(raw_root.path)] = {
                path.name: path.read_bytes() for path in window.iterdir()
            }
    session = use_session(fake_session_factory(stats={"scalars": (1, 0, 0)}))

    def stop_when_idle(seconds: int) -> None:
        assert seconds == 60
        for relative, files in originals.items():
            assert not (raw_root.path / relative).exists()
            assert {
                path.name: path.read_bytes() for path in (raw_root.path / "_collected" / relative).iterdir()
            } == files
        assert list(raw_root.path.glob("*/*/attempt=*/_collector_batch.json")) == []
        raise _StopLoop

    monkeypatch.setattr(cli.time, "sleep", stop_when_idle)

    with pytest.raises(_StopLoop):
        cli.main([*_base_args(raw_root, rclone_conf), "--loop-seconds", "60"])

    manifests = [
        json.loads(sql.split("SELECT '", 1)[1].rsplit("' SETTINGS", 1)[0])
        for sql, _ in session.queries if "/batches/" in sql
    ]
    for attempt in ["0", "1"]:
        assert [manifest["source_windows"] for manifest in manifests if manifest["attempt"] == attempt] == [
            [f"node=host1/w{step:08d}-{step:08d}" for step in range(128)],
            ["node=host1/w00000128-00000128"],
        ]


def test_main_sleeps_when_the_only_pending_attempt_fails(
    raw_root: RawRoot, rclone_conf: Path, monkeypatch: pytest.MonkeyPatch, fake_session_factory, use_session,
    write_window,
) -> None:
    window = write_window(raw_root.path, "expA", "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    original_files = {path.name: path.read_bytes() for path in window.iterdir()}
    session = use_session(fake_session_factory())

    def stop_on_backoff(seconds: int) -> None:
        assert seconds == 60
        assert {path.name: path.read_bytes() for path in window.iterdir()} == original_files
        assert json.loads((window.parent.parent / "_collector_batch.json").read_bytes()) == {
            "source_windows": ["node=host1/w00000000-00000009"], "receipts": None,
        }
        raise _StopLoop

    monkeypatch.setattr(cli.time, "sleep", stop_on_backoff)

    with pytest.raises(_StopLoop):
        cli.main([*_base_args(raw_root, rclone_conf), "--loop-seconds", "60"])

    assert not any(sql.startswith("INSERT") for sql, _ in session.queries)


class _Clock:
    """time.monotonic and time.sleep of the collector loop: sleeping advances the clock, and the test stops the loop
    after `limit` sleeps."""

    def __init__(self, limit: int) -> None:
        self.now = 1000.0
        self.sleeps: list[int] = []
        self.limit = limit

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: int) -> None:
        if len(self.sleeps) == self.limit:
            raise _StopLoop
        self.sleeps.append(seconds)
        self.now += seconds


def test_main_exits_once_nothing_was_left_to_collect_for_the_idle_minutes(
    raw_root: RawRoot, rclone_conf: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    fake_session_factory, use_session, write_dump,
) -> None:
    # Without a dump target the collector leaves dumps where they are, so they are nothing it waits for.
    write_dump(raw_root.path, "expA", "run1", "0", step=100, rank=0)
    use_session(fake_session_factory())
    clock = _Clock(limit=10)
    monkeypatch.setattr(cli.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(cli.time, "sleep", clock.sleep)

    cli.main([*_base_args(raw_root, rclone_conf), "--loop-seconds", "60", "--idle-exit-minutes", "3"])

    assert clock.sleeps == [60, 60, 60]
    assert capsys.readouterr().err.splitlines()[-1] == "nothing left to collect for 3 minutes, exiting"


def test_main_keeps_collecting_while_a_window_waits_even_if_its_batch_fails(
    raw_root: RawRoot, rclone_conf: Path, monkeypatch: pytest.MonkeyPatch, fake_session_factory, use_session,
    write_window,
) -> None:
    write_window(raw_root.path, "expA", "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    use_session(fake_session_factory())
    clock = _Clock(limit=10)
    monkeypatch.setattr(cli.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(cli.time, "sleep", clock.sleep)

    with pytest.raises(_StopLoop):
        cli.main([*_base_args(raw_root, rclone_conf), "--loop-seconds", "60", "--idle-exit-minutes", "3"])

    assert clock.sleeps == [60] * 10


def test_main_passes_chdb_config_path_to_session_builder(
    raw_root: RawRoot, rclone_conf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_session_factory
) -> None:
    raw_root.path.mkdir()
    session = fake_session_factory()
    captured: dict[str, Path | None] = {}

    def fake_build_session(chdb_config: Path | None):
        captured["chdb_config"] = chdb_config
        return session

    monkeypatch.setattr(cli, "_build_session", fake_build_session)
    chdb_config_path = tmp_path / "chdb-config.xml"

    cli.main([*_base_args(raw_root, rclone_conf), "--chdb-config", str(chdb_config_path)])

    assert captured["chdb_config"] == chdb_config_path


def test_main_reads_credentials_from_rclone_conf(
    raw_root: RawRoot, rclone_conf: Path, fake_session_factory, use_session, write_window, write_metrics_registry,
) -> None:
    write_metrics_registry(raw_root.path, "expA", "run1", "0")
    write_window(raw_root.path, "expA", "run1", "0", "host1", 0, 49, families=(Family.SCALARS,))
    session = use_session(fake_session_factory(stats={"scalars": (1, 0, 49)}))

    cli.main(_base_args(raw_root, rclone_conf))

    insert_sql = next(sql for sql, fmt in session.queries if fmt is None and ") SELECT * FROM file(" in sql)
    assert "'test-key'" in insert_sql
    assert "'test-secret'" in insert_sql


def test_acquire_collector_lock_returns_the_open_file_when_the_lock_succeeds(tmp_path: Path) -> None:
    lock_file = cli._acquire_collector_lock(tmp_path / ".collector.lock")

    assert lock_file is not None
    assert (tmp_path / ".collector.lock").exists()
    lock_file.close()


def test_acquire_collector_lock_returns_none_when_flock_raises_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_flock_exclusive_nonblocking", _raise_already_locked)

    assert cli._acquire_collector_lock(tmp_path / ".collector.lock") is None


def test_main_exits_with_a_clear_message_and_nonzero_code_when_another_collector_holds_the_lock(
    raw_root: RawRoot, rclone_conf: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    fake_session_factory, use_session,
) -> None:
    raw_root.path.mkdir()
    use_session(fake_session_factory())
    monkeypatch.setattr(cli, "_flock_exclusive_nonblocking", _raise_already_locked)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(_base_args(raw_root, rclone_conf))

    assert exc_info.value.code != 0
    err = capsys.readouterr().err
    assert "already holds the lock" in err
    assert str(raw_root.path / ".collector.lock") in err


def test_main_passes_the_scratch_directory_of_the_merged_files(
    raw_root: RawRoot, rclone_conf: Path, tmp_path: Path, fake_session_factory, use_session, collector_arguments,
) -> None:
    raw_root.path.mkdir()
    use_session(fake_session_factory())

    cli.main([*_base_args(raw_root, rclone_conf), "--scratch-dir", str(tmp_path / "scratch")])

    assert collector_arguments["scratch_dir"] == tmp_path / "scratch"


def test_a_dump_target_needs_its_prefix(raw_root: RawRoot, rclone_conf: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main([
            *_base_args(raw_root, rclone_conf), "--dump-remote-section", "dumps-s3", "--dump-bucket", "dumps-bucket",
        ])
