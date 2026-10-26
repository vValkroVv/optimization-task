"""Tests for the job_dispatcher S3 artifacts helpers."""

import sys
import types
from pathlib import Path
from typing import Dict, List

import pytest
from node_runner.src.job_dispatcher import JD
from node_runner.src.job_dispatcher import artifacts
from node_runner.src.logger import Logger

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_logger(monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[str]]:
    logs: Dict[str, List[str]] = {"warning": []}

    monkeypatch.setattr(Logger, "warning", logs["warning"].append)

    return logs


class TestArtifactKeyPrefix:
    @pytest.mark.parametrize(
        ("checkpoint_suffix", "expected_checkpoint_name"),
        [
            (("ep3_ba42", "artifacts", "hf_checkpoint"), "ep3_ba42"),
            (("global_step_42", "actor", "huggingface"), "global_step_42"),
            (
                (
                    "global_step_42",
                    "actor",
                    "huggingface",
                    "artifacts",
                    "hf_checkpoint_fp8",
                ),
                "global_step_42",
            ),
            (("ep3_ba42", "artifacts", "hf_checkpoint_fp8"), "ep3_ba42"),
        ],
    )
    def test_prefix_keys_metrics_by_checkpoint_and_shooter(
        self,
        tmp_path: Path,
        checkpoint_suffix: tuple,
        expected_checkpoint_name: str,
    ):
        hf_checkpoint_path = tmp_path / "model"
        for path_part in checkpoint_suffix:
            hf_checkpoint_path /= path_part

        prefix = JD.artifact_key_prefix(
            hf_checkpoint_path, "sft", "42", "obm", "abc123"
        )

        assert prefix == (
            f"pretrain-auto-metrics-artifacts/model/{expected_checkpoint_name}"
            "/artifacts/metrics/sft/42/obm/abc123"
        )

    def test_gigaeye_url_joins_bucket_prefix_and_suffix(self):
        assert JD.gigaeye_url("prefix", "cache") == (
            "gigaeye-s3:gigaeye-pretrain-checkpoints/prefix/cache"
        )


class TestReadArtifact:
    def test_returns_uploader_bytes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            artifacts.S3ArtifactUploader,
            "get_object_bytes",
            lambda _, key: key.encode(),
        )

        assert JD.read_artifact("prefix/cache.zst") == b"prefix/cache.zst"


class TestUploadArtifacts:
    def test_no_internet_skips_upload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_logger: Dict[str, List[str]],
        tmp_path: Path,
    ):
        monkeypatch.setattr(JD, "internet_is_available", lambda: False)

        assert JD.upload_artifacts(tmp_path, "prefix") is False
        assert mock_logger["warning"] == ["No internet available; skipping S3 upload"]

    @pytest.mark.parametrize("upload_result", [True, False])
    def test_returns_uploader_result(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, upload_result: bool
    ):
        calls: Dict[str, object] = {}

        class FakeUploader:
            def upload_dir_merge(self, local_dir: Path, prefix: str) -> bool:
                calls.update({"local_dir": local_dir, "prefix": prefix})
                return upload_result

        monkeypatch.setattr(JD, "internet_is_available", lambda: True)
        monkeypatch.setattr(artifacts, "S3ArtifactUploader", FakeUploader)

        assert JD.upload_artifacts(tmp_path, "prefix") is upload_result
        assert calls == {"local_dir": tmp_path, "prefix": "prefix"}


class TestUploadPamConfig:
    def test_no_internet_skips_upload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_logger: Dict[str, List[str]],
        tmp_path: Path,
    ):
        monkeypatch.setattr(JD, "internet_is_available", lambda: False)

        result = JD.upload_pam_config(
            tmp_path / "pam_config.yaml",
            tmp_path / "model" / "ep3_ba42",
            "hf",
            "42",
            "record-1",
        )

        assert result is None
        assert mock_logger["warning"] == ["No internet available; skipping S3 upload"]

    @pytest.mark.parametrize(
        ("upload_result", "expected_url"),
        [
            pytest.param(
                True,
                "gigaeye-s3:gigaeye-pretrain-checkpoints/pretrain-auto-metrics-artifacts"
                "/model/ep3_ba42/artifacts/metrics/hf/42/record-1/pam_config.yaml",
                id="uploaded",
            ),
            pytest.param(False, None, id="not-uploaded"),
        ],
    )
    def test_uploads_config_under_autometrics_record_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        upload_result: bool,
        expected_url: str | None,
    ):
        calls: Dict[str, object] = {}

        class FakeUploader:
            def upload_file(self, local_path: Path, prefix: str) -> bool:
                calls.update({"local_path": local_path, "prefix": prefix})
                return upload_result

        monkeypatch.setattr(JD, "internet_is_available", lambda: True)
        monkeypatch.setattr(artifacts, "S3ArtifactUploader", FakeUploader)
        config_path = tmp_path / "pam_config.yaml"

        result = JD.upload_pam_config(
            config_path,
            tmp_path / "model" / "ep3_ba42" / "hf_checkpoint",
            "hf",
            "42",
            "record-1",
        )

        assert result == expected_url
        assert calls == {
            "local_path": config_path,
            "prefix": (
                "pretrain-auto-metrics-artifacts/model/ep3_ba42/artifacts/metrics"
                "/hf/42/record-1"
            ),
        }


class TestArtifactFileTransfers:
    """``upload_artifact_files`` and ``copy_artifact_files`` move a file list."""

    def test_no_internet_skips_the_transfers(
        self, monkeypatch: pytest.MonkeyPatch, mock_logger: Dict[str, List[str]]
    ):
        rel_paths = [Path("cache/alpha.jsonl")]
        monkeypatch.setattr(JD, "internet_is_available", lambda: False)

        assert JD.upload_artifact_files(Path("/metrics"), rel_paths, "prefix") is False
        assert JD.copy_artifact_files("record-prefix", rel_paths, "prefix") is False
        assert mock_logger["warning"] == [
            "No internet available; skipping S3 upload",
            "No internet available; skipping S3 copy",
        ]

    def test_hands_the_file_list_to_the_uploader(self, monkeypatch: pytest.MonkeyPatch):
        rel_paths = [Path("cache/alpha.jsonl")]
        calls: List[tuple] = []

        class FakeUploader:
            def upload_files_merge(
                self, local_dir: Path, rel_paths: List[Path], prefix: str
            ) -> bool:
                calls.append(("upload", local_dir, list(rel_paths), prefix))
                return True

            def copy_files_merge(
                self, src_key_prefix: str, rel_paths: List[Path], dst_key_prefix: str
            ) -> bool:
                calls.append(("copy", src_key_prefix, list(rel_paths), dst_key_prefix))
                return False

        monkeypatch.setattr(JD, "internet_is_available", lambda: True)
        monkeypatch.setattr(artifacts, "S3ArtifactUploader", FakeUploader)

        assert JD.upload_artifact_files(Path("/metrics"), rel_paths, "prefix") is True
        assert JD.copy_artifact_files("record-prefix", rel_paths, "prefix") is False
        assert calls == [
            ("upload", Path("/metrics"), rel_paths, "prefix"),
            ("copy", "record-prefix", rel_paths, "prefix"),
        ]


_TRANSFER2CLOUD_MODULE = (
    "contrib.job_submitter.job_submitter.jobs.transfers.transfer2cloud"
)


@pytest.fixture
def transfer2cloud_jobs(monkeypatch: pytest.MonkeyPatch) -> tuple:
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


class TestOfflineTransferToCloud:
    @pytest.fixture(autouse=True)
    def _offline(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setattr(JD, "internet_is_available", lambda: False)
        monkeypatch.setenv("JD_SYNC_DIR", str(tmp_path / "transfer2cloud/o1/jd_sync"))
        metrics_dir = tmp_path / "metrics"
        (metrics_dir / "cache").mkdir(parents=True)
        (metrics_dir / "cache" / "a.jsonl").write_text("a")
        (metrics_dir / "scores.zst").write_text("s")
        (tmp_path / "ci").mkdir()
        (tmp_path / "ci" / "ci.zst").write_text("c")

    @pytest.mark.parametrize(
        ("upload", "expected_staged"),
        [
            pytest.param(
                lambda d: JD.upload_artifacts(d, "prefix"),
                ["cache/a.jsonl", "scores.zst"],
                id="dir",
            ),
            pytest.param(
                lambda d: JD.upload_artifact_files(
                    d, [Path("cache/a.jsonl")], "prefix"
                ),
                ["cache/a.jsonl"],
                id="files",
            ),
            pytest.param(
                lambda d: JD.upload_artifact_files(
                    d, [Path("cache/a.jsonl")], "prefix", [d.parent / "ci" / "ci.zst"]
                ),
                ["cache/a.jsonl", "ci.zst"],
                id="files-with-an-outside-file",
            ),
            pytest.param(
                lambda d: JD.upload_artifact_file(d / "scores.zst", "prefix"),
                ["scores.zst"],
                id="file",
            ),
        ],
    )
    def test_upload_stages_files_and_transfers_them_like_archives(
        self,
        tmp_path: Path,
        transfer2cloud_jobs: tuple,
        upload,
        expected_staged: list,
    ):
        _, created = transfer2cloud_jobs
        jd_sync_dir = tmp_path / "transfer2cloud/o1/jd_sync"

        assert upload(tmp_path / "metrics") is True

        assert [job.kwargs for job in created] == [
            {
                "source_path": jd_sync_dir / "prefix",
                "staging_path": jd_sync_dir,
                "exp_name": "pam-s3-transfer",
            }
        ]
        staged = sorted(
            path.relative_to(jd_sync_dir / "prefix").as_posix()
            for path in (jd_sync_dir / "prefix").rglob("*")
            if path.is_file()
        )
        assert staged == expected_staged

    def test_links_point_where_the_transfer_puts_files(self):
        assert JD.gigaeye_url("prefix", "cache") == (
            "papayaeye-papaya:transfer2cloud/o1/jd_sync/prefix/cache"
        )

    def test_failed_submission_reports_no_upload(
        self,
        tmp_path: Path,
        transfer2cloud_jobs: tuple,
        mock_logger: Dict[str, List[str]],
    ):
        job_class, _ = transfer2cloud_jobs
        job_class.error = RuntimeError("Slurm REST is not configured")

        assert JD.upload_artifacts(tmp_path / "metrics", "prefix") is False

        assert mock_logger["warning"] == [
            f"{tmp_path}/transfer2cloud/o1/jd_sync/prefix stays in JD_SYNC_DIR: "
            "Slurm REST is not configured"
        ]

    def test_unreadable_file_is_not_transferred(
        self,
        tmp_path: Path,
        transfer2cloud_jobs: tuple,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _, created = transfer2cloud_jobs
        errors: List[str] = []
        monkeypatch.setattr(Logger, "error", errors.append)

        assert (
            JD.upload_artifact_files(tmp_path / "metrics", [Path("missing")], "prefix")
            is False
        )

        assert created == []
        assert len(errors) == 1
        assert errors[0].startswith("Failed to stage prefix in JD_SYNC_DIR:")
