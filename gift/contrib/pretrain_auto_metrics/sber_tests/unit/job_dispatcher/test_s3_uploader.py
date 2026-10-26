import subprocess
from pathlib import Path
from typing import Optional

import pytest

from node_runner.src.job_dispatcher import s3_uploader
from node_runner.src.job_dispatcher.s3_uploader import (
    RCLONE_ATTEMPTS,
    S3ArtifactUploader,
)

pytestmark = pytest.mark.unit


class FakeProcess:
    def __init__(self, times_out: bool) -> None:
        self.times_out = times_out

    def wait(self, timeout: Optional[float] = None) -> None:
        if self.times_out:
            raise subprocess.TimeoutExpired(cmd="rclone", timeout=timeout)


class FakeSubprocessManager:
    """Stand-in for SubprocessManager replaying a prepared exit code."""

    def __init__(
        self,
        cmd: list[str],
        log_file: Optional[Path],
        cwd: Optional[str],
        run_name: str,
        exit_code: Optional[int],
        stdout: bytes,
    ) -> None:
        self.cmd = cmd
        self.log_file = log_file
        self.cwd = cwd
        self.run_name = run_name
        self.exit_code = exit_code
        self.process = FakeProcess(exit_code is None)
        self.graceful_stops = 0
        if log_file:
            log_file.write_bytes(stdout)

    def start(self) -> None:
        pass

    def wait_for_completion(self) -> None:
        pass

    def graceful_stop(self) -> None:
        self.graceful_stops += 1

    def is_successful(self) -> bool:
        return self.exit_code == 0

    def log_error(self) -> None:
        pass


def install_fake_manager(
    monkeypatch: pytest.MonkeyPatch,
    exit_codes: list[Optional[int]],
    stdout: bytes = b"",
) -> list[FakeSubprocessManager]:
    """Replace SubprocessManager with a fake returning ``exit_codes`` in order."""
    started: list[FakeSubprocessManager] = []

    def factory(
        cmd: list[str], log_file: Optional[Path], cwd: Optional[str], run_name: str
    ) -> FakeSubprocessManager:
        manager = FakeSubprocessManager(
            cmd, log_file, cwd, run_name, exit_codes[len(started)], stdout
        )
        started.append(manager)
        return manager

    monkeypatch.setattr(s3_uploader, "SubprocessManager", factory)
    return started


@pytest.fixture
def uploader(tmp_path: Path) -> S3ArtifactUploader:
    config_path = tmp_path / "rclone.conf"
    config_path.write_text("[remote]\n", encoding="utf-8")
    instance = S3ArtifactUploader()
    instance.rclone_binary = "/usr/bin/rclone"
    instance.config_path = str(config_path)
    instance.remote = "remote"
    instance.bucket = "bucket"
    return instance


def test_is_configured_requires_binary_and_config(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader
) -> None:
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.s3_uploader.shutil.which", lambda _: None
    )
    assert uploader.is_configured() is False

    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.s3_uploader.shutil.which", lambda path: path
    )
    uploader.config_path = "/missing/rclone.conf"
    assert uploader.is_configured() is False

    uploader.config_path = ""
    assert uploader.is_configured() is True


def test_base_cmd_includes_config_only_when_set(uploader: S3ArtifactUploader) -> None:
    assert uploader._base_cmd() == [
        uploader.rclone_binary,
        "--config",
        uploader.config_path,
    ]

    uploader.config_path = ""
    assert uploader._base_cmd() == [uploader.rclone_binary]


def test_run_retries_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader
) -> None:
    started = install_fake_manager(monkeypatch, [1, 0])

    assert uploader._run(["copy", "src", "dst"]) is True
    assert len(started) == 2
    assert started[0].cmd[-3:] == ["copy", "src", "dst"]


def test_run_stops_process_that_exceeds_timeout(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader
) -> None:
    started = install_fake_manager(monkeypatch, [None] * RCLONE_ATTEMPTS)

    assert uploader._run(["copy"]) is False
    assert len(started) == RCLONE_ATTEMPTS
    assert all(manager.graceful_stops == 1 for manager in started)


@pytest.mark.parametrize("stdout", [b"", b"data"])
def test_get_object_bytes_returns_rclone_output(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader, stdout: bytes
) -> None:
    monkeypatch.setattr(uploader, "is_configured", lambda: True)
    started = install_fake_manager(monkeypatch, [0], stdout)

    assert uploader.get_object_bytes("/prefix/cache.zst") == stdout
    assert started[0].cmd[-4:] == [
        "cat",
        "-q",
        "--s3-no-check-bucket",
        "remote:bucket/prefix/cache.zst",
    ]


def test_get_object_bytes_returns_none_when_read_fails(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader
) -> None:
    monkeypatch.setattr(uploader, "is_configured", lambda: True)
    install_fake_manager(monkeypatch, [1] * RCLONE_ATTEMPTS)

    assert uploader.get_object_bytes("/prefix/cache.zst") is None


def test_get_object_bytes_skips_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader
) -> None:
    monkeypatch.setattr(uploader, "is_configured", lambda: False)

    assert uploader.get_object_bytes("prefix/cache.zst") is None


def test_upload_dir_merge_runs_copy_to_remote_prefix(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader, tmp_path: Path
) -> None:
    source = tmp_path / "metrics"
    source.mkdir()
    monkeypatch.setattr(uploader, "is_configured", lambda: True)
    started = install_fake_manager(monkeypatch, [0])

    assert uploader.upload_dir_merge(source, "/prefix/") is True
    assert started[0].cmd[3] == "copy"
    assert started[0].cmd[-2:] == [str(source), "remote:bucket/prefix"]
    assert started[0].log_file is None


def test_upload_dir_merge_reports_failure(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader, tmp_path: Path
) -> None:
    source = tmp_path / "metrics"
    source.mkdir()
    monkeypatch.setattr(uploader, "is_configured", lambda: True)
    install_fake_manager(monkeypatch, [1] * RCLONE_ATTEMPTS)

    assert uploader.upload_dir_merge(source, "prefix") is False


def test_upload_dir_merge_skips_invalid_source(
    uploader: S3ArtifactUploader, tmp_path: Path
) -> None:
    assert uploader.upload_dir_merge(tmp_path / "missing", "prefix") is False


def test_upload_dir_merge_skips_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader, tmp_path: Path
) -> None:
    source = tmp_path / "metrics"
    source.mkdir()
    monkeypatch.setattr(uploader, "is_configured", lambda: False)

    assert uploader.upload_dir_merge(source, "prefix") is False


@pytest.mark.parametrize(
    "source_name",
    [
        pytest.param("missing.yaml", id="missing"),
        pytest.param("directory", id="directory"),
    ],
)
def test_upload_file_skips_non_file_source(
    monkeypatch: pytest.MonkeyPatch,
    uploader: S3ArtifactUploader,
    tmp_path: Path,
    source_name: str,
) -> None:
    (tmp_path / "directory").mkdir()
    monkeypatch.setattr(uploader, "is_configured", lambda: True)
    started = install_fake_manager(monkeypatch, [])

    assert uploader.upload_file(tmp_path / source_name, "prefix") is False
    assert started == []


def test_upload_file_copies_file_to_remote_prefix(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader, tmp_path: Path
) -> None:
    source = tmp_path / "pam_config.yaml"
    source.write_text("mode: autometrics\n", encoding="utf-8")
    monkeypatch.setattr(uploader, "is_configured", lambda: True)
    started = install_fake_manager(monkeypatch, [0])

    assert uploader.upload_file(source, "/prefix/") is True
    assert started[0].cmd[3] == "copy"
    assert started[0].cmd[-2:] == [str(source), "remote:bucket/prefix"]


def capture_copy_run(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader, result: bool
) -> dict:
    """Replace ``_run`` with a stub recording the rclone args and the file list."""
    captured: dict = {}

    def fake_run(args: list[str]) -> bool:
        listing = Path(args[args.index("--files-from") + 1])
        captured["args"] = args
        captured["listing"] = listing.read_text(encoding="utf-8")
        return result

    monkeypatch.setattr(uploader, "_run", fake_run)
    return captured


def test_upload_files_merge_copies_only_the_listed_files(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader, tmp_path: Path
) -> None:
    source = tmp_path / "metrics"
    source.mkdir()
    monkeypatch.setattr(uploader, "is_configured", lambda: True)
    captured = capture_copy_run(monkeypatch, uploader, True)

    assert (
        uploader.upload_files_merge(
            source, [Path("cache/alpha.jsonl"), Path("cache/beta.jsonl")], "/prefix/"
        )
        is True
    )
    assert captured["args"][0] == "copy"
    assert captured["args"][-2:] == [str(source), "remote:bucket/prefix"]
    assert captured["listing"] == "cache/alpha.jsonl\ncache/beta.jsonl\n"


def test_files_merge_skips_a_copy_it_cannot_run(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader, tmp_path: Path
) -> None:
    source = tmp_path / "metrics"
    source.mkdir()
    rel_paths = [Path("cache/alpha.jsonl")]
    started = install_fake_manager(monkeypatch, [])

    monkeypatch.setattr(uploader, "is_configured", lambda: True)
    assert uploader.upload_files_merge(tmp_path / "gone", rel_paths, "prefix") is False

    monkeypatch.setattr(uploader, "is_configured", lambda: False)
    assert uploader.upload_files_merge(source, rel_paths, "prefix") is False
    assert uploader.copy_files_merge("record-prefix", rel_paths, "prefix") is False
    assert started == []


def test_copy_files_merge_copies_between_remote_prefixes(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader
) -> None:
    monkeypatch.setattr(uploader, "is_configured", lambda: True)
    captured = capture_copy_run(monkeypatch, uploader, True)

    assert (
        uploader.copy_files_merge(
            "record-prefix", [Path("cache/alpha.jsonl")], "/prefix/"
        )
        is True
    )
    assert captured["args"][-2:] == [
        "remote:bucket/record-prefix",
        "remote:bucket/prefix",
    ]
    assert captured["listing"] == "cache/alpha.jsonl\n"


def test_copy_files_merge_reports_a_failed_copy(
    monkeypatch: pytest.MonkeyPatch, uploader: S3ArtifactUploader
) -> None:
    monkeypatch.setattr(uploader, "is_configured", lambda: True)
    capture_copy_run(monkeypatch, uploader, False)

    assert (
        uploader.copy_files_merge(
            "record-prefix", [Path("cache/alpha.jsonl")], "prefix"
        )
        is False
    )
