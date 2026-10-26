"""Tests for the job_dispatcher offline sync fallback (tar/symlink to JD_SYNC_DIR)."""

import sys
import types
from pathlib import Path
from typing import Dict

import pytest
from node_runner.src.job_dispatcher import JD
from node_runner.src.job_dispatcher import sync
from node_runner.src.logger import Logger

pytestmark = pytest.mark.unit

EVALUATOR_NAME = "DummyEvaluator"
MOCK_TAR_DATA = "mock data"
_TRANSFER2CLOUD_MODULE = (
    "contrib.job_submitter.job_submitter.jobs.transfers.transfer2cloud"
)


@pytest.fixture
def transfer2cloud_job(monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Stub the vendored job-submitter module and record created jobs.

    The import chain is provided through sys.modules because unit CI does not
    check out contrib/job_submitter. Set ``error`` on the returned class to
    make submit raise.
    """
    created: list = []

    class FakeTransfer2CloudJob:
        error: Exception | None = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        def submit(self) -> str:
            if type(self).error is not None:
                raise type(self).error
            return "12345"

    parts = _TRANSFER2CLOUD_MODULE.split(".")
    for index in range(1, len(parts)):
        name = ".".join(parts[:index])
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType(_TRANSFER2CLOUD_MODULE)
    module.Transfer2CloudJob = FakeTransfer2CloudJob
    monkeypatch.setitem(sys.modules, _TRANSFER2CLOUD_MODULE, module)
    return FakeTransfer2CloudJob, created


@pytest.fixture
def mock_logger(monkeypatch: pytest.MonkeyPatch) -> Dict[str, list]:
    logs: Dict[str, list] = {"info": [], "error": [], "warning": []}

    monkeypatch.setattr(Logger, "info", logs["info"].append)
    monkeypatch.setattr(Logger, "error", logs["error"].append)
    monkeypatch.setattr(Logger, "warning", logs["warning"].append)

    return logs


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(JD, "internet_is_available", lambda: False)


@pytest.fixture
def online(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(JD, "internet_is_available", lambda: True)


@pytest.fixture
def hf_checkpoint_path(tmp_path: Path) -> Path:
    return tmp_path / "model" / "ep0_ba1" / "artifacts" / "hf_checkpoint"


@pytest.fixture
def hf_checkpoint_path_global_step(tmp_path: Path) -> Path:
    return tmp_path / "model" / "global_step_2" / "actor" / "huggingface"


@pytest.fixture
def jd_sync_dir(tmp_path: Path) -> Path:
    return tmp_path / "jd_sync"


@pytest.fixture
def tar_archive_path(tmp_path: Path) -> Path:
    return tmp_path / "artifacts" / "m.tar.zst"


def _setup_tar_archive(tar_path: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tar_path.write_text(MOCK_TAR_DATA)


class TestInternetIsAvailable:
    def test_true_when_connect_succeeds(self, monkeypatch: pytest.MonkeyPatch):
        class MockSocket:
            def connect(self, _addr):
                return None

        monkeypatch.setattr(sync.socket, "setdefaulttimeout", lambda _t: None)
        monkeypatch.setattr(sync.socket, "socket", lambda *_a, **_kw: MockSocket())

        assert JD.internet_is_available(timeout=1) is True

    def test_false_when_connect_fails(self, monkeypatch: pytest.MonkeyPatch):
        class MockSocket:
            def connect(self, _addr):
                raise OSError("no net")

        monkeypatch.setattr(sync.socket, "setdefaulttimeout", lambda _t: None)
        monkeypatch.setattr(sync.socket, "socket", lambda *_a, **_kw: MockSocket())

        assert JD.internet_is_available(timeout=1) is False


class TestTarMetricsDir:
    def test_internet_returns_none(self, online: None, tmp_path: Path):
        result = JD.tar_metrics_dir(tmp_path / "metrics", tmp_path, EVALUATOR_NAME)
        assert result is None

    def test_without_jd_sync_dir_returns_none(
        self,
        offline: None,
        tmp_path: Path,
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("JD_SYNC_DIR", raising=False)

        result = JD.tar_metrics_dir(tmp_path / "metrics", tmp_path, EVALUATOR_NAME)

        assert result is None
        assert any("JD_SYNC_DIR is not set" in msg for msg in mock_logger["info"])

    def test_creates_archive(
        self,
        offline: None,
        tmp_path: Path,
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        metrics_dir = tmp_path / "metrics" / "obm"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("JD_SYNC_DIR", str(tmp_path / "jd-sync"))
        (artifacts / "obm_artifacts.tar.zst").write_text("old")

        call_tracker = {"start": 0, "wait": 0, "raise": 0}

        class MockSubprocessManager:
            def __init__(self, cmd, log_file, cwd, run_name):
                assert cmd == [
                    "tar",
                    "--zstd",
                    "--ignore-failed-read",
                    "-cf",
                    str(artifacts / "obm_artifacts.tar.zst"),
                    "obm",
                ]
                assert cwd == metrics_dir.parent
                assert run_name == "Tar obm"

            def start(self):
                call_tracker["start"] += 1

            def wait_for_completion(self):
                call_tracker["wait"] += 1

            def raise_for_exit_code(self):
                call_tracker["raise"] += 1

        monkeypatch.setattr(sync, "SubprocessManager", MockSubprocessManager)

        result = JD.tar_metrics_dir(metrics_dir, artifacts, EVALUATOR_NAME)

        assert result is not None
        assert result.name == "obm_artifacts.tar.zst"
        assert call_tracker == {"start": 1, "wait": 1, "raise": 1}

    def test_without_existing_archive(
        self,
        offline: None,
        tmp_path: Path,
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        metrics_dir = tmp_path / "metrics" / "lm_eval"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("JD_SYNC_DIR", str(tmp_path / "jd-sync"))

        class MockSubprocessManager:
            def __init__(self, cmd, log_file, cwd, run_name):
                assert run_name == "Tar lm_eval"

            def start(self):
                return None

            def wait_for_completion(self):
                return None

            def raise_for_exit_code(self):
                return None

        monkeypatch.setattr(sync, "SubprocessManager", MockSubprocessManager)

        result = JD.tar_metrics_dir(metrics_dir, artifacts, EVALUATOR_NAME)

        assert result is not None
        assert len(mock_logger["info"]) > 0


class TestSymlinkMetricsForTransfer:
    @pytest.mark.parametrize(
        ("tar_exists", "jd_sync_dir_exists"),
        [
            (False, False),
            (True, False),
            (False, True),
        ],
    )
    def test_skip_cases_warn(
        self,
        offline: None,
        hf_checkpoint_path: Path,
        tmp_path: Path,
        mock_logger: Dict[str, list],
        tar_exists: bool,
        jd_sync_dir_exists: bool,
        monkeypatch: pytest.MonkeyPatch,
    ):
        tar_path = tmp_path / "test.tar.zst"
        if tar_exists:
            tar_path.write_text("data")

        if jd_sync_dir_exists:
            monkeypatch.setenv("JD_SYNC_DIR", str(tmp_path / "jd_sync"))
        else:
            monkeypatch.delenv("JD_SYNC_DIR", raising=False)

        JD.symlink_metrics_for_transfer(tar_path, hf_checkpoint_path, EVALUATOR_NAME)

        assert len(mock_logger["warning"]) > 0

    def test_skips_when_internet_available(
        self,
        online: None,
        tar_archive_path: Path,
        hf_checkpoint_path: Path,
        jd_sync_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _setup_tar_archive(tar_archive_path)
        monkeypatch.setenv("JD_SYNC_DIR", str(jd_sync_dir))

        JD.symlink_metrics_for_transfer(
            tar_archive_path, hf_checkpoint_path, EVALUATOR_NAME
        )

        assert not jd_sync_dir.exists()

    def test_creates_new_link(
        self,
        offline: None,
        tar_archive_path: Path,
        hf_checkpoint_path: Path,
        jd_sync_dir: Path,
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        _setup_tar_archive(tar_archive_path)
        monkeypatch.setenv("JD_SYNC_DIR", str(jd_sync_dir))

        JD.symlink_metrics_for_transfer(
            tar_archive_path, hf_checkpoint_path, EVALUATOR_NAME
        )

        link_path = jd_sync_dir / "model" / "ep0_ba1" / "m.tar.zst"
        assert link_path.is_symlink()
        assert link_path.resolve() == tar_archive_path.resolve()
        assert len(mock_logger["info"]) > 0

    def test_huggingface_layout_uses_global_step_path(
        self,
        offline: None,
        tar_archive_path: Path,
        hf_checkpoint_path_global_step: Path,
        jd_sync_dir: Path,
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        _setup_tar_archive(tar_archive_path)
        monkeypatch.setenv("JD_SYNC_DIR", str(jd_sync_dir))

        JD.symlink_metrics_for_transfer(
            tar_archive_path, hf_checkpoint_path_global_step, EVALUATOR_NAME
        )

        link_path = jd_sync_dir / "model" / "global_step_2" / "m.tar.zst"
        assert link_path.is_symlink()
        assert link_path.resolve() == tar_archive_path.resolve()

    @staticmethod
    def _prepare_link(
        tar_archive_path: Path,
        jd_sync_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Windows runners lack the symlink privilege, and the link itself is
        # not what these tests assert.
        monkeypatch.setattr(
            Path, "symlink_to", lambda self, target: self.write_text("")
        )
        _setup_tar_archive(tar_archive_path)
        monkeypatch.setenv("JD_SYNC_DIR", str(jd_sync_dir))

    def test_submits_transfer_after_symlinking(
        self,
        offline: None,
        tar_archive_path: Path,
        hf_checkpoint_path: Path,
        jd_sync_dir: Path,
        transfer2cloud_job: tuple,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _, created = transfer2cloud_job
        self._prepare_link(tar_archive_path, jd_sync_dir, monkeypatch)

        JD.symlink_metrics_for_transfer(
            tar_archive_path, hf_checkpoint_path, EVALUATOR_NAME
        )

        assert len(created) == 1
        assert created[0].kwargs == {
            "source_path": jd_sync_dir / "model" / "ep0_ba1" / "m.tar.zst",
            "staging_path": jd_sync_dir,
            "exp_name": "pam-dummyevaluator-transfer",
        }

    def test_keeps_the_run_when_job_submitter_is_unavailable(
        self,
        offline: None,
        tar_archive_path: Path,
        hf_checkpoint_path: Path,
        jd_sync_dir: Path,
        transfer2cloud_job: tuple,
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        _, created = transfer2cloud_job
        monkeypatch.setitem(
            sys.modules, _TRANSFER2CLOUD_MODULE, types.ModuleType("outdated")
        )
        self._prepare_link(tar_archive_path, jd_sync_dir, monkeypatch)

        JD.symlink_metrics_for_transfer(
            tar_archive_path, hf_checkpoint_path, EVALUATOR_NAME
        )

        assert created == []
        assert any(
            "stays in JD_SYNC_DIR" in message for message in mock_logger["warning"]
        )

    def test_keeps_the_run_when_the_submit_fails(
        self,
        offline: None,
        tar_archive_path: Path,
        hf_checkpoint_path: Path,
        jd_sync_dir: Path,
        transfer2cloud_job: tuple,
        mock_logger: Dict[str, list],
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_job_class, created = transfer2cloud_job
        fake_job_class.error = RuntimeError("slurmrestd unavailable")
        self._prepare_link(tar_archive_path, jd_sync_dir, monkeypatch)

        JD.symlink_metrics_for_transfer(
            tar_archive_path, hf_checkpoint_path, EVALUATOR_NAME
        )

        assert len(created) == 1
        assert any(
            "slurmrestd unavailable" in message for message in mock_logger["warning"]
        )

    def test_replaces_broken_symlink(
        self,
        offline: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        artifacts_path = tmp_path / "artifacts"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        tar_archive_path = artifacts_path / "obm_artifacts.tar.zst"
        tar_archive_path.write_text("data")

        hf_checkpoint_path = (
            tmp_path / "model" / "ep0_ba2500" / "artifacts" / "hf_checkpoint"
        )
        jd_sync_dir = tmp_path / "jd_sync"
        monkeypatch.setenv("JD_SYNC_DIR", str(jd_sync_dir))
        transfer_directory = jd_sync_dir / "model" / "ep0_ba2500"
        transfer_directory.mkdir(parents=True, exist_ok=True)

        link_path = transfer_directory / "obm_artifacts.tar.zst"
        link_path.symlink_to(tmp_path / "does_not_exist.tar.zst")

        JD.symlink_metrics_for_transfer(
            tar_archive_path, hf_checkpoint_path, EVALUATOR_NAME
        )

        assert link_path.is_symlink()
        assert link_path.resolve() == tar_archive_path.resolve()
