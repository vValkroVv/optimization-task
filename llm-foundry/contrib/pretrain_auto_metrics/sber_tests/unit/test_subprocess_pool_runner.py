import sys
from pathlib import Path

import pytest
from node_runner.src.subprocess_pool_runner import SubprocessPoolRunner

pytestmark = pytest.mark.unit


class _FakeSM:
    def __init__(self, cmd, log, cwd, run_name, env=None):
        self.cmd = cmd
        self.log = log
        self.cwd = cwd
        self.run_name = run_name
        self.env = env
        self.started = False
        self.logged = False
        self.waited = False
        self.process = object()
        self.exit_code = None
        self._alive = True

    def start(self):
        self.started = True
        self._alive = False

    def is_alive(self):
        return self._alive

    def wait_for_completion(self):
        self.waited = True
        self.exit_code = 0

    def log_error(self):
        self.logged = True


class TestSubprocessPoolInitialization:
    def test_init_validates_max_parallel(self, tmp_path):
        with pytest.raises(ValueError, match="max_parallel must be >= 1"):
            SubprocessPoolRunner({}, 0, tmp_path / "log.txt")


class TestSubprocessPoolQueueLifecycle:
    def test_start_next_respects_max_parallel(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        created = []

        def _factory(cmd, log, cwd, run_name, env=None):
            sm = _FakeSM(cmd, log, cwd, run_name, env=env)
            created.append(sm)
            return sm

        monkeypatch.setattr(
            "node_runner.src.subprocess_pool_runner.SubprocessManager", _factory
        )
        runner = SubprocessPoolRunner(
            {"a": ["cmd1"], "b": ["cmd2"], "c": ["cmd3"]},
            max_parallel=2,
            autometrics_log_file=tmp_path / "log.txt",
        )

        runner._start_next_if_possible()

        assert len(runner._active) == 2
        assert runner._pending_idx == 2
        assert all(sm.started for sm in created)

    def test_finalize_finished_waits_logs_and_removes(self, tmp_path):
        runner = SubprocessPoolRunner({"a": ["cmd"]}, 1, tmp_path / "log.txt")
        sm = _FakeSM(["cmd"], tmp_path / "log.txt", None, "a")
        sm._alive = False
        sm.process = object()
        sm.exit_code = None
        runner._active = {"a": sm}

        runner._finalize_finished()

        assert sm.waited is True
        assert sm.logged is True
        assert runner._active == {}

    def test_finalize_finished_keeps_alive_process_in_active(self, tmp_path):
        runner = SubprocessPoolRunner({"a": ["cmd"]}, 1, tmp_path / "log.txt")
        sm = _FakeSM(["cmd"], tmp_path / "log.txt", None, "a")
        sm._alive = True
        runner._active = {"a": sm}

        runner._finalize_finished()

        assert sm.waited is False
        assert sm.logged is False
        assert "a" in runner._active

    def test_finalize_finished_skips_wait_when_process_none_or_exit_known(
        self, tmp_path
    ):
        runner = SubprocessPoolRunner(
            {"a": ["cmd"], "b": ["cmd"]}, 2, tmp_path / "log.txt"
        )
        sm_no_process = _FakeSM(["cmd"], tmp_path / "log.txt", None, "a")
        sm_no_process._alive = False
        sm_no_process.process = None
        sm_no_process.exit_code = None

        sm_exit_known = _FakeSM(["cmd"], tmp_path / "log.txt", None, "b")
        sm_exit_known._alive = False
        sm_exit_known.process = object()
        sm_exit_known.exit_code = 1

        runner._active = {"a": sm_no_process, "b": sm_exit_known}
        runner._finalize_finished()

        assert sm_no_process.waited is False
        assert sm_exit_known.waited is False
        assert sm_no_process.logged is True
        assert sm_exit_known.logged is True
        assert runner._active == {}

    def test_has_work_left(self, tmp_path):
        runner = SubprocessPoolRunner({"a": ["cmd"]}, 1, tmp_path / "log.txt")
        assert runner._has_work_left() is True
        runner._pending_idx = 1
        assert runner._has_work_left() is False
        runner._active = {"a": _FakeSM(["cmd"], tmp_path / "log.txt", None, "a")}
        assert runner._has_work_left() is True


class TestSubprocessPoolEnv:
    def test_env_is_passed_to_subprocess_manager(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        created = []
        custom_env = {"PYTHONPATH": "/a:/b", "MY_VAR": "val"}

        def _factory(cmd, log, cwd, run_name, env=None):
            sm = _FakeSM(cmd, log, cwd, run_name, env=env)
            created.append(sm)
            return sm

        monkeypatch.setattr(
            "node_runner.src.subprocess_pool_runner.SubprocessManager", _factory
        )
        runner = SubprocessPoolRunner(
            {"a": ["cmd1"]},
            max_parallel=1,
            autometrics_log_file=tmp_path / "log.txt",
            env=custom_env,
        )
        runner._start_next_if_possible()

        assert len(created) == 1
        assert created[0].env == custom_env

    def test_default_env_is_none(self, tmp_path):
        runner = SubprocessPoolRunner(
            {"a": ["cmd"]}, max_parallel=1, autometrics_log_file=tmp_path / "log.txt"
        )
        assert runner.env is None

    def test_none_env_is_passed_to_subprocess_manager(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        created = []

        def _factory(cmd, log, cwd, run_name, env=None):
            sm = _FakeSM(cmd, log, cwd, run_name, env=env)
            created.append(sm)
            return sm

        monkeypatch.setattr(
            "node_runner.src.subprocess_pool_runner.SubprocessManager", _factory
        )
        runner = SubprocessPoolRunner(
            {"a": ["cmd1"]},
            max_parallel=1,
            autometrics_log_file=tmp_path / "log.txt",
        )
        runner._start_next_if_possible()

        assert created[0].env is None


class TestSubprocessPoolRunAll:
    def test_run_all_handles_empty_queue(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        infos = []
        monkeypatch.setattr(
            "node_runner.src.subprocess_pool_runner.Logger.info", infos.append
        )
        runner = SubprocessPoolRunner({}, 2, tmp_path / "log.txt")
        runner.run_all(poll_interval=0)
        assert any("No PoolRunner tasks to execute" in m for m in infos)

    def test_run_all_completes_all_work(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        infos = []
        monkeypatch.setattr(
            "node_runner.src.subprocess_pool_runner.Logger.info", infos.append
        )
        monkeypatch.setattr(
            "node_runner.src.subprocess_pool_runner.time.sleep", lambda _: None
        )
        monkeypatch.setattr(
            "node_runner.src.subprocess_pool_runner.SubprocessManager",
            lambda cmd, log, cwd, run_name, env=None: _FakeSM(
                cmd, log, cwd, run_name, env=env
            ),
        )

        runner = SubprocessPoolRunner(
            {"a": ["cmd1"], "b": ["cmd2"]},
            max_parallel=1,
            autometrics_log_file=tmp_path / "log.txt",
        )
        runner.run_all(poll_interval=0)

        assert runner._pending_idx == 2
        assert runner._active == {}
        assert any("All PoolRunner runs completed" in m for m in infos)

    def test_subprocess_pool_runner_executes_real_commands(self, tmp_path: Path):
        log_file = tmp_path / "autometrics.log"
        cmds = {
            "run_a": [sys.executable, "-c", "print('run_a_ok')"],
            "run_b": [
                sys.executable,
                "-c",
                "import time; time.sleep(0.05); print('run_b_ok')",
            ],
            "run_c": [sys.executable, "-c", "print('run_c_ok')"],
        }

        runner = SubprocessPoolRunner(
            cmds_to_run=cmds,
            max_parallel=2,
            autometrics_log_file=log_file,
        )
        runner.run_all(poll_interval=0.01)

        assert runner._pending_idx == len(cmds)
        assert runner._active == {}
        content = log_file.read_text(encoding="utf-8")
        assert "run_a_ok" in content
        assert "run_b_ok" in content
        assert "run_c_ok" in content

    def test_subprocess_pool_runner_logs_failed_command_output(self, tmp_path: Path):
        log_file = tmp_path / "autometrics.log"
        cmds = {
            "ok": [sys.executable, "-c", "print('ok_run')"],
            "bad": [sys.executable, "-c", "import sys; print('bad_run'); sys.exit(3)"],
        }

        runner = SubprocessPoolRunner(
            cmds_to_run=cmds,
            max_parallel=2,
            autometrics_log_file=log_file,
        )
        runner.run_all(poll_interval=0.01)

        assert runner._active == {}
        content = log_file.read_text(encoding="utf-8")
        assert "ok_run" in content
        assert "bad_run" in content
