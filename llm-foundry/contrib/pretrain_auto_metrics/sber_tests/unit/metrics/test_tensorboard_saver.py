import re
import shutil
from pathlib import Path

import pytest
from node_runner.src.metrics.tensorboard_saver import (
    EmptyDirectoryError,
    TensorBoardSaver,
)

pytestmark = pytest.mark.unit


class _FakeWriter:
    instances = []

    def __init__(self, log_dir):
        self.log_dir = str(log_dir)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.closed = False
        self.flushed = False
        _FakeWriter.instances.append(self)

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))

    def flush(self):
        self.flushed = True

    def close(self):
        event_file = Path(self.log_dir) / f"events.out.tfevents.{len(self.calls)}.tmp"
        event_file.write_text("fake", encoding="utf-8")
        self.closed = True


class _FakeEvent:
    def __init__(self, step, value):
        self.step = step
        self.value = value


class _FakeEA:
    events_by_dir: dict[str, dict[str, list[_FakeEvent]]] = {}
    paths = []

    def __init__(self, path, size_guidance):
        self.path = str(path)
        self.size_guidance = size_guidance
        _FakeEA.paths.append(Path(path).name)

    def Reload(self):
        return None

    def Tags(self):
        return {"scalars": list(self.events_by_dir.get(self.path, {}).keys())}

    def Scalars(self, tag):
        return self.events_by_dir[self.path][tag]


class TestTensorboardSaver:
    def test_init_without_tensorboard_path_does_not_create_metrics_dir(
        self, tmp_path: Path
    ):
        saver = TensorBoardSaver(tmp_path / "ep0_ba1", None, "run")
        assert saver.metrics_tensorboard is None

    def test_get_tensorboard_step_parsing(self):
        assert TensorBoardSaver._get_tensorboard_step(Path("ep0_ba2576")) == 2576
        assert TensorBoardSaver._get_tensorboard_step(Path("hf_checkpoint")) == 0

    def test_save_to_tensorboard_uses_prefixes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _FakeWriter.instances = []
        monkeypatch.setattr(
            "node_runner.src.metrics.tensorboard_saver.SummaryWriter",
            _FakeWriter,
        )
        saver = TensorBoardSaver(tmp_path / "ep1_ba42", tmp_path / "tb", "run")
        run_dir = saver._create_new_run_dir()
        saver._save_to_tensorboard(
            {
                "mmlu_pro/math/5-shot/exact_match.custom-extract": 0.9,
                "gsm8k/4-shot/exact_match.flexible-extract": 0.95,
            },
            run_dir,
            step=42,
        )

        writer = _FakeWriter.instances[-1]
        assert (
            "mmlu_pro/math/5-shot/exact_match.custom-extract",
            0.9,
            42,
        ) in writer.calls
        assert (
            "autometrics/gsm8k/4-shot/exact_match.flexible-extract",
            0.95,
            42,
        ) in writer.calls
        assert writer.closed is True

    def test_create_new_run_dir_uses_timestamp_under_metrics(self, tmp_path: Path):
        saver = TensorBoardSaver(tmp_path / "ep0_ba1", tmp_path / "tb", "run")

        run_dir = saver._create_new_run_dir()

        assert run_dir.parent == saver.metrics_tensorboard
        assert re.fullmatch(r"\d{8}_\d{6}_\d{6}", run_dir.name)

    def test_tensorboard_write_lock_serializes_body_and_releases(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        calls = []

        def _fake_flock(fd, operation):
            calls.append((fd, operation))

        monkeypatch.setattr(
            "node_runner.src.metrics.tensorboard_saver.fcntl.flock", _fake_flock
        )
        saver = TensorBoardSaver(tmp_path / "ep0_ba1", tmp_path / "tb", "run")
        lock_path = saver.metrics_tensorboard / ".tensorboard_saver.lock"

        with saver._tensorboard_write_lock():
            assert lock_path.exists()
            assert len(calls) == 1

        assert len(calls) == 2
        assert calls[0][0] == calls[1][0]
        assert calls[0][1] != calls[1][1]

    def test_iter_tensorboard_run_dirs_returns_empty_when_metrics_dir_is_missing(
        self, tmp_path: Path
    ):
        saver = TensorBoardSaver(tmp_path / "ep0_ba1", tmp_path / "tb", "run")
        shutil.rmtree(saver.metrics_tensorboard)

        assert saver._iter_tensorboard_run_dirs() == []

    def test_collect_latest_scalars_reads_all_run_dirs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr(
            "node_runner.src.metrics.tensorboard_saver.event_accumulator.EventAccumulator",
            _FakeEA,
        )
        saver = TensorBoardSaver(tmp_path / "ep0_ba1", tmp_path / "tb", "run")
        old_dir = saver.metrics_tensorboard / "20260430_120000_000001"
        new_dir = saver.metrics_tensorboard / "20260430_120000_000002"
        tmp_dir = saver.metrics_tensorboard / ".sorted_tmp"
        old_dir.mkdir()
        new_dir.mkdir()
        tmp_dir.mkdir()
        _FakeEA.paths = []
        _FakeEA.events_by_dir = {
            str(old_dir): {
                "a/tag": [_FakeEvent(2, 0.2), _FakeEvent(1, 0.1)],
                "b/tag": [_FakeEvent(3, 3.0)],
            },
            str(new_dir): {
                "a/tag": [_FakeEvent(2, 0.3)],
            },
        }

        scalars = saver._collect_latest_scalars()

        assert scalars == {"a/tag": {1: 0.1, 2: 0.3}, "b/tag": {3: 3.0}}
        assert _FakeEA.paths == [old_dir.name, new_dir.name]

    def test_add_metrics_to_scalars_formats_tags_at_checkpoint_step(
        self, tmp_path: Path
    ):
        saver = TensorBoardSaver(tmp_path / "ep0_ba42", tmp_path / "tb", "run")
        scalars = {"autometrics/existing": {1: 0.1}}

        saver._add_metrics_to_scalars(
            scalars,
            {
                "existing": 0.2,
                "mmlu_pro/math/exact_match": 0.9,
            },
        )

        assert scalars == {
            "autometrics/existing": {1: 0.1, 42: 0.2},
            "mmlu_pro/math/exact_match": {42: 0.9},
        }

    def test_write_sorted_scalars_writes_sorted_current_and_cleans_old_dirs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _FakeWriter.instances = []
        monkeypatch.setattr(
            "node_runner.src.metrics.tensorboard_saver.SummaryWriter",
            _FakeWriter,
        )
        saver = TensorBoardSaver(tmp_path / "ep0_ba1", tmp_path / "tb", "run")
        old_dir = saver.metrics_tensorboard / "20260430_120000_000001"
        current_dir = saver.metrics_tensorboard / "20260430_120000_000002"
        old_dir.mkdir()
        current_dir.mkdir()
        scalars_by_tag = {"a/tag": {2: 0.3, 1: 0.1}, "b/tag": {3: 3.0}}

        saver._write_sorted_scalars(current_dir, scalars_by_tag)
        saver._cleanup_previous_timestamp_dirs(current_dir)

        writer = _FakeWriter.instances[-1]
        assert writer.flushed is True
        assert writer.closed is True
        assert writer.calls == [
            ("a/tag", 0.1, 1),
            ("a/tag", 0.3, 2),
            ("b/tag", 3.0, 3),
        ]
        assert current_dir.exists()
        assert (current_dir / TensorBoardSaver._FIXED_EVENT_FILENAME).exists()
        assert not old_dir.exists()
        assert [path.name for path in saver.metrics_tensorboard.iterdir()] == [
            current_dir.name
        ]

    def test_write_sorted_scalars_returns_when_event_file_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        class _NoEventWriter(_FakeWriter):
            def close(self):
                self.closed = True

        monkeypatch.setattr(
            "node_runner.src.metrics.tensorboard_saver.SummaryWriter",
            _NoEventWriter,
        )

        saver = TensorBoardSaver(tmp_path / "ep0_ba1", tmp_path / "tb", "run")
        tmp_dir = saver.metrics_tensorboard / "20260430_120000_000001"
        tmp_dir.mkdir()
        saver._write_sorted_scalars(tmp_dir, {"a/tag": {1: 0.1}})

        assert not (tmp_dir / TensorBoardSaver._FIXED_EVENT_FILENAME).exists()

    def test_write_sorted_scalars_removes_extra_event_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        class _MultiEventWriter(_FakeWriter):
            def close(self):
                log_dir = Path(self.log_dir)
                (log_dir / "events.out.tfevents.first").write_text(
                    "first", encoding="utf-8"
                )
                (log_dir / "events.out.tfevents.second").write_text(
                    "second", encoding="utf-8"
                )
                self.closed = True

        monkeypatch.setattr(
            "node_runner.src.metrics.tensorboard_saver.SummaryWriter",
            _MultiEventWriter,
        )

        saver = TensorBoardSaver(tmp_path / "ep0_ba1", tmp_path / "tb", "run")
        tmp_dir = saver.metrics_tensorboard / "20260430_120000_000001"
        tmp_dir.mkdir()
        saver._write_sorted_scalars(tmp_dir, {"a/tag": {1: 0.1}})

        event_files = list(tmp_dir.glob("events.out.tfevents.*"))
        assert event_files == [tmp_dir / TensorBoardSaver._FIXED_EVENT_FILENAME]

    def test_wait_for_dirs_skips_deleted_dirs_and_fails_on_empty_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        saver = TensorBoardSaver(tmp_path / "ep0_ba1", tmp_path / "tb", "run")
        deleted_dir = saver.metrics_tensorboard / "20260430_120000_000001"
        deleted_dir.mkdir()
        shutil.rmtree(deleted_dir)
        written_dir = saver.metrics_tensorboard / "20260430_120000_000002"
        written_dir.mkdir()
        (written_dir / "events.out.tfevents.fake").write_text("fake", encoding="utf-8")
        empty_dir = saver.metrics_tensorboard / "20260430_120000_000003"
        empty_dir.mkdir()

        assert saver._dir_is_empty(written_dir) is False
        assert saver._dir_is_empty(empty_dir) is True

        monkeypatch.setattr(
            saver, "_iter_tensorboard_run_dirs", lambda: [deleted_dir, written_dir]
        )
        TensorBoardSaver._wait_for_all_dirs_to_be_written.__wrapped__(saver)

        monkeypatch.setattr(saver, "_iter_tensorboard_run_dirs", lambda: [empty_dir])
        with pytest.raises(EmptyDirectoryError):
            TensorBoardSaver._wait_for_all_dirs_to_be_written.__wrapped__(saver)

    def test_save_metrics_to_tensorboard_branches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        infos = []
        monkeypatch.setattr(
            "node_runner.src.metrics.tensorboard_saver.Logger.info", infos.append
        )

        saver = TensorBoardSaver(tmp_path / "ep0_ba10", tmp_path / "tb", "run")
        called = {
            "lock_enter": 0,
            "lock_exit": 0,
            "wait": 0,
            "collect": 0,
            "add": 0,
            "write": 0,
            "cleanup": 0,
        }
        created_run_dir = saver.metrics_tensorboard / "20260430_120000_000001"

        def _create_run_dir():
            created_run_dir.mkdir()
            return created_run_dir

        class _Lock:
            def __enter__(self):
                called["lock_enter"] += 1

            def __exit__(self, exc_type, exc_value, traceback):
                called["lock_exit"] += 1

        monkeypatch.setattr(saver, "_tensorboard_write_lock", lambda: _Lock())
        monkeypatch.setattr(
            saver,
            "_wait_for_all_dirs_to_be_written",
            lambda: called.__setitem__("wait", called["wait"] + 1),
        )
        monkeypatch.setattr(saver, "_create_new_run_dir", _create_run_dir)
        monkeypatch.setattr(
            saver,
            "_collect_latest_scalars",
            lambda: called.__setitem__("collect", called["collect"] + 1) or {},
        )
        monkeypatch.setattr(
            saver,
            "_add_metrics_to_scalars",
            lambda scalars_by_tag, metrics_results: called.__setitem__(
                "add", called["add"] + 1
            ),
        )
        monkeypatch.setattr(
            saver,
            "_write_sorted_scalars",
            lambda current_run_dir, scalars_by_tag: called.__setitem__(
                "write", called["write"] + 1
            ),
        )
        monkeypatch.setattr(
            saver,
            "_cleanup_previous_timestamp_dirs",
            lambda current_run_dir: called.__setitem__(
                "cleanup", called["cleanup"] + 1
            ),
        )

        saver.save_metrics_to_tensorboard({"a/c": "failed"})
        assert called == {
            "lock_enter": 0,
            "lock_exit": 0,
            "wait": 0,
            "collect": 0,
            "add": 0,
            "write": 0,
            "cleanup": 0,
        }

        saver.save_metrics_to_tensorboard({"a/b": 1.0})
        assert called == {
            "lock_enter": 1,
            "lock_exit": 1,
            "wait": 1,
            "collect": 1,
            "add": 1,
            "write": 1,
            "cleanup": 1,
        }
        assert any(str(created_run_dir) in msg for msg in infos)
