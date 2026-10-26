"""Tests for job_submitter.core.tracking."""

import builtins
import importlib.util
import os
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from unittest.mock import patch

from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.llmf.train import TrainJobLLMF
from job_submitter.core.tracking import (
    _build_auto_track_errors_command_data,
    _resolve_tracking_config,
    _sanitize_extra_metadata,
    auto_track,
    extract_logs_path,
    extract_logs_path_from_file,
    _get_nested,
    _strip_cluster_prefix,
)
from job_submitter.core.tracking_storage import (
    _build_s3_storage_config,
    _insert_auto_track_errors_command,
    _parse_env_file,
    _resolve_auto_track_local_dir,
    _resolve_s3_env,
    _resolve_secret_env_path,
    _should_skip_auto_track_due_missing_secret_file,
)


@dataclass(slots=True, kw_only=True)
class _GenericTrackingJob(BaseJob):
    path_to_config: str | Path | None = None

    def _script(self) -> str:
        return "echo test"

    def _before_submit(self) -> None:
        pass


@dataclass(slots=True, kw_only=True)
class _ExplicitTrackingJob(_GenericTrackingJob):
    def _tracking_config_path(self, _cfg) -> str | None:
        if not self.path_to_config:
            return None
        path = str(self.path_to_config).strip()
        return path or None


@dataclass(slots=True, kw_only=True)
class _MultiSourceTrackingJob(BaseJob):
    path_to_config: str | Path | None = None
    pam_config_path: str | Path | None = None
    runtime_env_config_path: str | Path | None = None

    def _script(self) -> str:
        return "echo test"

    def _before_submit(self) -> None:
        pass


class _S3NotFoundError(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class _FakeS3Client:
    def __init__(self) -> None:
        self.remote_bytes: bytes | None = None
        self.uploads: list[tuple[str, str, str]] = []
        self.downloads: list[tuple[str, str, str]] = []

    def head_object(self, *, Bucket: str, Key: str):
        if self.remote_bytes is None:
            raise _S3NotFoundError()
        return {"ETag": '"fake-etag"'}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        if self.remote_bytes is None:
            raise _S3NotFoundError()
        Path(filename).write_bytes(self.remote_bytes)
        self.downloads.append((bucket, key, filename))

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.remote_bytes = Path(filename).read_bytes()
        self.uploads.append((filename, bucket, key))


def _patch_inbox_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[_FakeS3Client, list, Path]:
    fake_client = _FakeS3Client()
    captured_configs = []
    local_dir = tmp_path / "autotrack-sqlite"
    monkeypatch.setenv("TRACKING_AUTO_TRACK_DB_LOCAL_DIR", str(local_dir))

    def _build_client(config):
        captured_configs.append(config)
        return fake_client

    monkeypatch.setattr(
        "job_submitter.core.tracking_storage.build_s3_client", _build_client
    )
    return fake_client, captured_configs, local_dir


def _local_db_path(local_dir: Path, s3_key: str) -> Path:
    return local_dir / s3_key.replace("/", "_").replace("\\", "_")


def _fetch_rows(local_dir: Path, s3_key: str, query: str) -> list[tuple]:
    conn = sqlite3.connect(_local_db_path(local_dir, s3_key))
    try:
        return list(conn.execute(query))
    finally:
        conn.close()


@pytest.mark.unit
class TestGetNested:
    def test_deep_path(self) -> None:
        assert _get_nested({"a": {"b": {"c": 42}}}, "a.b.c") == 42

    def test_missing(self) -> None:
        assert _get_nested({"a": 1}, "b.c", "fallback") == "fallback"

    def test_none_in_chain(self) -> None:
        assert _get_nested({"a": None}, "a.b") is None


@pytest.mark.unit
class TestResolveTrackingConfig:
    def test_resolves(self) -> None:
        cfg = {"base": "/mnt", "path": "${base}/logs"}
        assert _resolve_tracking_config(cfg)["path"] == "/mnt/logs"

    def test_no_refs(self) -> None:
        assert _resolve_tracking_config({"a": 1}) == {"a": 1}


@pytest.mark.unit
class TestExtractLogsPath:
    def test_basic(self) -> None:
        cfg = {"run_name": "run1", "loggers": {"tensorboard": {"log_dir": "/tb"}}}
        assert extract_logs_path(cfg) == "/tb/run1"

    def test_custom_run_name(self) -> None:
        cfg = {"loggers": {"tensorboard": {"log_dir": "/tb"}}}
        assert extract_logs_path(cfg, run_name="custom") == "/tb/custom"

    def test_default_run_name(self) -> None:
        cfg = {"loggers": {"tensorboard": {"log_dir": "/tb"}}}
        assert extract_logs_path(cfg) == "/tb/llm"

    def test_with_variables(self) -> None:
        cfg = {
            "save_folder": "/data",
            "loggers": {"tensorboard": {"log_dir": "${save_folder}/tb"}},
            "run_name": "exp",
        }
        assert extract_logs_path(cfg) == "/data/tb/exp"

    def test_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="tensorboard"):
            extract_logs_path({"a": 1})

    def test_tensorboard_path_only(self) -> None:
        cfg = {"tensorboard_path": "/tb/root/experiment"}
        assert extract_logs_path(cfg) == "/tb/root/experiment"

    def test_tensorboard_path_with_interpolation(self) -> None:
        cfg = {
            "user": "alice",
            "exp_name": "e1",
            "tensorboard_path": "/home/${user}/tb/${exp_name}",
        }
        assert extract_logs_path(cfg) == "/home/alice/tb/e1"

    def test_loggers_log_dir_precedes_tensorboard_path(self) -> None:
        cfg = {
            "tensorboard_path": "/ignored",
            "loggers": {"tensorboard": {"log_dir": "/tb"}},
            "run_name": "r",
        }
        assert extract_logs_path(cfg) == "/tb/r"

    def test_extract_logs_path_from_sh_script(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text('CKPTS_DIR="/ckpts/run"\n', encoding="utf-8")
        assert extract_logs_path_from_file(str(script)) == "/ckpts/run"


@pytest.mark.unit
class TestStripClusterPrefix:
    def test_own_prefix(self) -> None:
        assert (
            _strip_cluster_prefix("pretrain:sr004:/mnt/logs", "pretrain", "sr004")
            == "/mnt/logs"
        )

    def test_foreign_prefix(self) -> None:
        assert (
            _strip_cluster_prefix("other:cl:/mnt/logs", "pretrain", "sr004")
            == "/mnt/logs"
        )

    def test_clean(self) -> None:
        assert _strip_cluster_prefix("/mnt/logs", "pretrain", "sr004") == "/mnt/logs"


@pytest.mark.unit
class TestAutoTrack:
    def test_full_flow(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake_s3, captured_configs, local_dir = _patch_inbox_writer(
            monkeypatch, tmp_path
        )
        s3_key = "tracker_daemons_v2/pretrain_sr004_inbox.sqlite"
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"SR004": "pretrain"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"SR004": s3_key},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {}},
        )

        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "run_name: test_run\nloggers:\n  tensorboard:\n    log_dir: /tb\n"
        )

        assert (
            auto_track(
                "SR004",
                "gdedov",
                str(cfg),
                auto_track_users=["@infidelis", "gdedov", "INFIDELIS", "", "goida"],
                enable_auto_share=True,
            )
            is True
        )

        assert len(captured_configs) == 1
        assert captured_configs[0].key == f"tracker/{s3_key}"
        assert fake_s3.uploads
        rows = _fetch_rows(
            local_dir,
            s3_key,
            "SELECT users, logs_path, yaml_path, script_path, timestamp, metadata "
            "FROM auto_track_requests",
        )
        assert len(rows) == 1
        params = rows[0]
        assert json.loads(params[0]) == ["gdedov", "infidelis", "goida"]
        assert params[1] == "/tb/test_run"
        assert params[2] == os.path.abspath(str(cfg))
        assert params[3] is None
        assert datetime.fromisoformat(params[4]).tzinfo is not None
        assert json.loads(params[5]) == {"auto_share": True}

    def test_extra_metadata_merges_with_auto_share(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, _, local_dir = _patch_inbox_writer(monkeypatch, tmp_path)
        s3_key = "tracker_daemons_v2/pretrain_sr004_inbox.sqlite"
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"SR004": "pretrain"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"SR004": s3_key},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {}},
        )

        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "run_name: test_run\nloggers:\n  tensorboard:\n    log_dir: /tb\n"
        )

        assert (
            auto_track(
                "SR004",
                "gdedov",
                str(cfg),
                enable_auto_share=True,
                extra_metadata={
                    "job_kind": "rclone_transfer",
                    "rclone_json_log": "/tmp/rclone-copy-attempt-1.json",
                },
            )
            is True
        )

        params = _fetch_rows(
            local_dir,
            s3_key,
            "SELECT metadata FROM auto_track_requests",
        )[0]
        assert json.loads(params[0]) == {
            "job_kind": "rclone_transfer",
            "rclone_json_log": "/tmp/rclone-copy-attempt-1.json",
            "auto_share": True,
        }

    def test_extra_metadata_drops_non_serializable_value_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _, _, local_dir = _patch_inbox_writer(monkeypatch, tmp_path)
        s3_key = "tracker_daemons_v2/pretrain_sr004_inbox.sqlite"
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"SR004": "pretrain"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"SR004": s3_key},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {}},
        )

        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "run_name: test_run\nloggers:\n  tensorboard:\n    log_dir: /tb\n"
        )

        assert (
            auto_track(
                "SR004",
                "gdedov",
                str(cfg),
                extra_metadata={
                    "job_kind": "rclone_transfer",
                    "bad_value": object(),
                },
            )
            is True
        )

        params = _fetch_rows(
            local_dir,
            s3_key,
            "SELECT metadata FROM auto_track_requests",
        )[0]
        assert json.loads(params[0]) == {
            "job_kind": "rclone_transfer",
            "auto_share": False,
        }
        assert "non-JSON-serializable" in caplog.text
        assert "bad_value" in caplog.text

    def test_auto_track_from_sh_config_sets_script_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, _, local_dir = _patch_inbox_writer(monkeypatch, tmp_path)
        s3_key = "tracker_daemons_v2/pretrain_sr004_inbox.sqlite"
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"SR004": "pretrain"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"SR004": s3_key},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {}},
        )

        script = tmp_path / "train.sh"
        script.write_text('CKPTS_DIR="/ckpts/run"\n', encoding="utf-8")

        assert auto_track("SR004", "u", str(script)) is True

        params = _fetch_rows(
            local_dir,
            s3_key,
            "SELECT users, logs_path, yaml_path, script_path, timestamp "
            "FROM auto_track_requests",
        )[0]
        assert params[1] == "/ckpts/run"
        assert params[3] == os.path.abspath(str(script))

    @pytest.mark.parametrize("region", ["SR004", "SR008"])
    def test_tts_workspace_uses_gigatts_inbox(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, region: str
    ) -> None:
        _, captured_configs, _ = _patch_inbox_writer(monkeypatch, tmp_path)
        secret_file = tmp_path / "super_secret" / "s3_db.env"
        secret_file.parent.mkdir()
        secret_file.write_text(
            "AWS_ACCESS_KEY_ID=tts_key\nAWS_SECRET_ACCESS_KEY=tts_secret\n"
            "S3_BUCKET=tracker-test-bucket\n"
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {region: {"ENV_FILEPATH_S3_DB": str(secret_file)}},
        )
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "run_name: test_run\nloggers:\n  tensorboard:\n    log_dir: /tb\n"
        )

        assert (
            auto_track(
                region,
                "gdedov",
                str(cfg),
                env={"MLS_NTB_WORKSPACE_NAME": "tts"},
            )
            is True
        )

        assert captured_configs[0].key == (
            f"tracker/tracker_daemons_v2/gigatts_{region.lower()}_inbox.sqlite"
        )
        assert captured_configs[0].aws_access_key_id == "tts_key"
        assert captured_configs[0].aws_secret_access_key == "tts_secret"
        assert captured_configs[0].bucket == "tracker-test-bucket"

    def test_reads_s3_env_from_region_secret_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _, captured_configs, _ = _patch_inbox_writer(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"SR004": "pretrain"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"SR004": "tracker_daemons_v2/pretrain_sr004_inbox.sqlite"},
        )

        secret_file = tmp_path / "s3_db.env"
        secret_file.write_text(
            "\n".join(
                [
                    "AWS_ACCESS_KEY_ID=from_file_key",
                    "AWS_SECRET_ACCESS_KEY=from_file_secret",
                    "S3_BUCKET=tracker-test-bucket",
                ]
            )
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {"ENV_FILEPATH_S3_DB": str(secret_file)}},
        )

        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "run_name: test_run\nloggers:\n  tensorboard:\n    log_dir: /tb\n"
        )

        assert auto_track("SR004", "gdedov", str(cfg)) is True
        storage_config = captured_configs[0]
        assert storage_config.aws_access_key_id == "from_file_key"
        assert storage_config.aws_secret_access_key == "from_file_secret"
        assert storage_config.bucket == "tracker-test-bucket"

    def test_merged_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, captured_configs, local_dir = _patch_inbox_writer(monkeypatch, tmp_path)
        s3_key = "tracker_daemons_v2/papaya_o1_inbox.sqlite"
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"o1": "papaya"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"o1": s3_key},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS", {"o1": {}}
        )

        merged = {"run_name": "merged", "loggers": {"tensorboard": {"log_dir": "/tb"}}}
        cfg = tmp_path / "dummy.yaml"
        cfg.write_text("")

        assert auto_track("o1", "user", str(cfg), merged_config=merged) is True

        assert captured_configs[0].key == f"tracker/{s3_key}"
        params = _fetch_rows(
            local_dir,
            s3_key,
            "SELECT users, logs_path, yaml_path, script_path, timestamp "
            "FROM auto_track_requests",
        )[0]
        assert params[1] == "/tb/merged"

    def test_direct_logs_path_does_not_require_config_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _, _, local_dir = _patch_inbox_writer(monkeypatch, tmp_path)
        s3_key = "tracker_daemons_v2/papaya_o1_inbox.sqlite"
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"o1": "papaya"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"o1": s3_key},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"o1": {}},
        )

        assert (
            auto_track(
                "o1",
                "user",
                None,
                submission_type="slurm",
                job_id="123",
                logs_path=" papaya:o1:/slurm/logs/slurm-123.out ",
            )
            is True
        )

        request = _fetch_rows(
            local_dir,
            s3_key,
            "SELECT logs_path, yaml_path FROM auto_track_requests",
        )[0]
        assert request == ("/slurm/logs/slurm-123.out", None)
        command = _fetch_rows(
            local_dir,
            s3_key,
            "SELECT data FROM commands",
        )[0][0]
        assert json.loads(command)["logs_info"] == {
            "submission_type": "slurm",
            "last_checked_retry": 0,
            "job_id": "123",
            "slurm_log_path": "/slurm/logs/slurm-123.out",
        }

    def test_auto_track_requires_config_or_direct_logs_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"o1": {}},
        )

        assert auto_track("o1", "user", None) is False

    def test_unknown_region(self) -> None:
        assert auto_track("NOPE", "u", "/cfg") is False

    def test_missing_inbox_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_s3, _, _ = _patch_inbox_writer(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"SR004": "pretrain"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {}},
        )

        cfg = tmp_path / "tmp_auto_track_cfg.yaml"
        cfg.write_text("run_name: r\nloggers:\n  tensorboard:\n    log_dir: /tb\n")
        assert auto_track("SR004", "u", str(cfg)) is False
        assert fake_s3.uploads == []

    def test_bad_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake_s3, _, _ = _patch_inbox_writer(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"SR004": "pretrain"},
        )
        assert auto_track("SR004", "u", "/nonexistent.yaml") is False
        assert fake_s3.uploads == []

    def test_skips_when_super_secret_env_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_s3, _, _ = _patch_inbox_writer(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"SR004": "pretrain"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {
                "SR004": {
                    "ENV_FILEPATH_S3_DB": str(
                        tmp_path / "super_secret" / "missing_s3_db.env"
                    )
                }
            },
        )

        cfg = tmp_path / "config.yaml"
        cfg.write_text("run_name: run1\nloggers:\n  tensorboard:\n    log_dir: /tb\n")

        assert auto_track("SR004", "u", str(cfg)) is False
        assert fake_s3.uploads == []

    def test_inserts_auto_track_errors_command_with_logs_info(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, _, local_dir = _patch_inbox_writer(monkeypatch, tmp_path)
        s3_key = "tracker_daemons_v2/pretrain_sr004_inbox.sqlite"
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"SR004": "pretrain"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"SR004": s3_key},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {}},
        )

        cfg = tmp_path / "config.yaml"
        cfg.write_text("run_name: run1\nloggers:\n  tensorboard:\n    log_dir: /tb\n")

        assert (
            auto_track(
                "SR004",
                "u",
                str(cfg),
                submission_type="cloud",
                job_id="cloud-job-123",
            )
            is True
        )

        command_params = _fetch_rows(
            local_dir,
            s3_key,
            "SELECT command_type, experiment_path, hash, timestamp, data FROM commands",
        )[0]
        assert command_params[0] == "auto_track_errors"
        assert command_params[1] == "/tb/run1"
        command_data = json.loads(command_params[4])
        assert command_data["enabled"] is True
        assert command_data["logs_info"]["submission_type"] == "cloud"
        assert command_data["logs_info"]["job_id"] == "cloud-job-123"
        assert command_data["logs_info"]["mlspace_logs_path"] == "/tb/run1"

    def test_request_and_command_carry_the_job_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, _, local_dir = _patch_inbox_writer(monkeypatch, tmp_path)
        s3_key = "tracker_daemons_v2/papaya_o1_inbox.sqlite"
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"o1": "papaya"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"o1": s3_key},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"o1": {}},
        )

        assert auto_track(
            "o1",
            "u",
            None,
            submission_type=" Slurm ",
            job_id="665538",
            job_title=" u | 3b run ",
            logs_path="/tb/run1",
            extra_metadata={"job_kind": "train"},
        )

        metadata = json.loads(
            _fetch_rows(local_dir, s3_key, "SELECT metadata FROM auto_track_requests")[
                0
            ][0]
        )
        assert metadata == {
            "submission_type": "slurm",
            "job_id": "665538",
            "job_title": "u | 3b run",
            "job_kind": "train",
            "auto_share": False,
        }
        command = _fetch_rows(local_dir, s3_key, "SELECT data FROM commands")[0][0]
        assert json.loads(command)["logs_info"]["job_title"] == "u | 3b run"

    def test_uses_resolved_local_dir_for_inbox_sqlite(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_inbox_writer(monkeypatch, tmp_path)
        s3_key = "tracker_daemons_v2/pretrain_sr004_inbox.sqlite"
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"SR004": "pretrain"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"SR004": s3_key},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {}},
        )
        resolved_local_dir = tmp_path / "resolved_autotrack_local_dir"
        resolved_local_dir.mkdir()
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage._resolve_auto_track_local_dir",
            lambda: resolved_local_dir,
        )

        cfg = tmp_path / "config.yaml"
        cfg.write_text("run_name: run1\nloggers:\n  tensorboard:\n    log_dir: /tb\n")

        assert auto_track("SR004", "u", str(cfg)) is True
        assert _local_db_path(resolved_local_dir, s3_key).exists()


@pytest.mark.unit
class TestTrainAutoTrack:
    def test_base_tracking_config_path_resolves_path_to_config(self) -> None:
        job = _GenericTrackingJob(
            user_login="tester",
            exp_name="exp",
            path_to_config=Path("/tmp/config.yaml"),
            submission_type="cloud",
            region="SR004",
        )
        cfg = SimpleNamespace(region="SR004")
        assert job._tracking_config_path(cfg).replace("\\", "/") == "/tmp/config.yaml"

    def test_base_tracking_config_path_resolves_pam_config_path(self) -> None:
        job = _MultiSourceTrackingJob(
            user_login="tester",
            exp_name="exp",
            path_to_config=None,
            pam_config_path="/tmp/pam.yaml",
            submission_type="cloud",
            region="SR004",
        )
        cfg = SimpleNamespace(region="SR004")
        assert job._tracking_config_path(cfg) == "/tmp/pam.yaml"

    def test_base_tracking_config_path_resolves_runtime_env_config_path(self) -> None:
        job = _MultiSourceTrackingJob(
            user_login="tester",
            exp_name="exp",
            path_to_config=None,
            pam_config_path=None,
            runtime_env_config_path="/tmp/runtime.yaml",
            submission_type="cloud",
            region="SR004",
        )
        cfg = SimpleNamespace(region="SR004")
        assert job._tracking_config_path(cfg) == "/tmp/runtime.yaml"

    def test_base_tracking_config_path_returns_none_for_blank(self) -> None:
        job = _GenericTrackingJob(
            user_login="tester",
            exp_name="exp",
            path_to_config="   ",
            submission_type="cloud",
            region="SR004",
        )
        cfg = SimpleNamespace(region="SR004")
        assert job._tracking_config_path(cfg) is None

    @patch("job_submitter.jobs.base.auto_track", side_effect=RuntimeError("boom"))
    def test_auto_track_after_submit_swallows_unexpected_errors(
        self, _mock_track
    ) -> None:
        job = _ExplicitTrackingJob(
            user_login="tester",
            exp_name="exp",
            path_to_config="/tmp/config.yaml",
            submission_type="cloud",
            region="SR004",
            enable_auto_track=True,
        )
        cfg = SimpleNamespace(
            submission_type="cloud",
            region="SR004",
            env_variables={},
            job_desc="tester | exp",
        )
        job._auto_track_after_submit(cfg)

    def test_explicit_tracking_config_path_hook_strips_blank_values(self) -> None:
        job = _ExplicitTrackingJob(
            user_login="tester",
            exp_name="exp",
            path_to_config="   ",
            submission_type="cloud",
            region="SR004",
        )
        cfg = SimpleNamespace(region="SR004")
        assert job._tracking_config_path(cfg) is None

    def test_llmf_tracking_config_path_is_explicit(self) -> None:
        job = TrainJobLLMF(
            user_login="tester",
            exp_name="exp",
            path_to_config=Path("/tmp/config.yaml"),
            submission_type="cloud",
            region="SR004",
        )
        cfg = SimpleNamespace(region="SR004")
        assert job._tracking_config_path(cfg).replace("\\", "/") == "/tmp/config.yaml"

    @patch("job_submitter.jobs.base.auto_track")
    def test_auto_track_after_submit_skips_when_config_missing(
        self, mock_track
    ) -> None:
        job = _GenericTrackingJob(
            user_login="tester",
            exp_name="exp",
            path_to_config=None,
            submission_type="cloud",
            region="SR004",
            enable_auto_track=True,
            auto_track_users=["infidelis", "goida"],
            enable_auto_share=True,
        )
        cfg = SimpleNamespace(region="SR004", env_variables={})
        job._auto_track_after_submit(cfg)
        mock_track.assert_not_called()

    @patch("job_submitter.jobs.base.auto_track")
    def test_auto_track_after_submit_skips_when_region_missing(
        self, mock_track
    ) -> None:
        job = _ExplicitTrackingJob(
            user_login="tester",
            exp_name="exp",
            path_to_config="/tmp/config.yaml",
            submission_type="cloud",
            region=None,
            enable_auto_track=True,
        )
        cfg = SimpleNamespace(region=None, env_variables={})
        job._auto_track_after_submit(cfg)
        mock_track.assert_not_called()

    @patch("job_submitter.jobs.base.auto_track")
    def test_submit_config_calls_auto_track_after_cloud_submit(
        self, mock_track, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_track.return_value = True
        job = TrainJobLLMF(
            user_login="tester",
            exp_name="exp",
            path_to_config="/some/config.yaml",
            enable_auto_track=True,
            auto_track_users=["infidelis", "goida"],
            enable_auto_share=True,
            submission_type="cloud",
            region="SR004",
        )

        class _SubmittedJob:
            job_name = "cloud-job-123"

            def status(self):
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                return ""

        def _submit_cloud(self, _cfg):
            self.last_submitted_job = _SubmittedJob()

        monkeypatch.setattr(
            "job_submitter.jobs.base.CloudBackend.submit",
            _submit_cloud,
        )

        cfg = SimpleNamespace(
            submission_type="cloud",
            region="SR004",
            env_variables={},
            job_desc="tester | exp",
        )
        assert job._submit_config(cfg) is job

        mock_track.assert_called_once_with(
            region="SR004",
            user_login="tester",
            auto_track_users=["infidelis", "goida"],
            enable_auto_share=True,
            config_path="/some/config.yaml",
            merged_config=None,
            env={},
            submission_type="cloud",
            job_id="cloud-job-123",
            script_path=None,
            job_title="tester | exp",
        )

    @patch("job_submitter.jobs.base.auto_track")
    def test_submit_config_skips_auto_track_when_disabled(
        self,
        mock_track,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        job = TrainJobLLMF(
            user_login="tester",
            exp_name="exp",
            path_to_config="/some/config.yaml",
            enable_auto_track=False,
            enable_auto_share=True,
            submission_type="cloud",
            region="SR004",
        )

        class _SubmittedJob:
            job_name = "cloud-job-456"

            def status(self):
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                return ""

        def _submit_cloud(self, _cfg):
            self.last_submitted_job = _SubmittedJob()

        monkeypatch.setattr(
            "job_submitter.jobs.base.CloudBackend.submit",
            _submit_cloud,
        )

        cfg = SimpleNamespace(
            submission_type="cloud",
            region="SR004",
            env_variables={},
            job_desc="tester | exp",
        )
        assert job._submit_config(cfg) is job

        mock_track.assert_not_called()
        assert "enable_auto_track=False" in caplog.text

    @patch("job_submitter.jobs.base.auto_track")
    def test_submit_config_calls_auto_track_after_slurm_submit(
        self, mock_track, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_track.return_value = True
        job = TrainJobLLMF(
            user_login="tester",
            exp_name="exp",
            path_to_config="/some/config.yaml",
            enable_auto_track=True,
            submission_type="slurm",
            region="o1",
        )

        def _submit_slurm(self, _cfg):
            self.last_submitted_job = SimpleNamespace(job_name="slurm-777")

        monkeypatch.setattr(
            "job_submitter.jobs.base.SlurmBackend.submit",
            _submit_slurm,
        )

        cfg = SimpleNamespace(
            submission_type="slurm",
            region="o1",
            env_variables={},
            job_desc="tester | exp",
        )
        assert job._submit_config(cfg) is job

        mock_track.assert_called_once_with(
            region="o1",
            user_login="tester",
            auto_track_users=[],
            enable_auto_share=False,
            config_path="/some/config.yaml",
            merged_config=None,
            env={},
            submission_type="slurm",
            job_id="slurm-777",
            script_path=None,
            job_title="tester | exp",
        )

    @patch("job_submitter.jobs.base.auto_track")
    def test_auto_track_keeps_explicit_slurm_region_when_env_differs(
        self, mock_track, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_track.return_value = True
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "o5")
        monkeypatch.setenv("SLURM_JWT_TOKEN", "jwt")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820")

        job = _GenericTrackingJob(
            user_login="tester",
            exp_name="exp",
            path_to_config="/some/config.yaml",
            enable_auto_track=True,
            submission_type="slurm",
            region="o1",
            queue_name="pretrain",
        )

        def _submit_slurm(self, _cfg):
            self.last_submitted_job = SimpleNamespace(job_name="slurm-888")

        monkeypatch.setattr(
            "job_submitter.jobs.base.SlurmBackend.submit",
            _submit_slurm,
        )

        cfg = job.to_config()
        assert cfg.region == "o1"
        assert job._submit_config(cfg) is job

        mock_track.assert_called_once_with(
            region="o1",
            user_login="tester",
            auto_track_users=[],
            enable_auto_share=False,
            config_path="/some/config.yaml",
            merged_config=None,
            env=cfg.env_variables,
            submission_type="slurm",
            job_id="slurm-888",
            script_path=None,
            job_title="tester | exp",
        )

    @patch("job_submitter.jobs.base.auto_track")
    def test_submit_config_passes_cfg_env_variables_to_auto_track(
        self, mock_track, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_track.return_value = True
        job = TrainJobLLMF(
            user_login="tester",
            exp_name="exp",
            path_to_config="/some/config.yaml",
            enable_auto_track=True,
            submission_type="cloud",
            region="SR004",
        )

        class _SubmittedJob:
            job_name = "cloud-job-999"

            def status(self):
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                return ""

        def _submit_cloud(self, _cfg):
            self.last_submitted_job = _SubmittedJob()

        monkeypatch.setattr(
            "job_submitter.jobs.base.CloudBackend.submit",
            _submit_cloud,
        )

        cfg_env = {
            "ENV_FILEPATH_S3_DB": "/tmp/from-build-env.env",
            "AWS_ACCESS_KEY_ID": "from-build-env",
        }
        cfg = SimpleNamespace(
            submission_type="cloud",
            region="SR004",
            env_variables=cfg_env,
            job_desc="tester | exp",
        )
        assert job._submit_config(cfg) is job

        mock_track.assert_called_once_with(
            region="SR004",
            user_login="tester",
            auto_track_users=[],
            enable_auto_share=False,
            config_path="/some/config.yaml",
            merged_config=None,
            env=cfg_env,
            submission_type="cloud",
            job_id="cloud-job-999",
            script_path=None,
            job_title="tester | exp",
        )

    @patch("job_submitter.jobs.base.logger.warning")
    @patch("job_submitter.jobs.base.auto_track")
    def test_submit_config_warns_when_auto_track_failed(
        self, mock_track, mock_warn, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_track.return_value = False
        job = TrainJobLLMF(
            user_login="tester",
            exp_name="exp",
            path_to_config="/some/config.yaml",
            enable_auto_track=True,
            submission_type="cloud",
            region="SR004",
        )

        class _SubmittedJob:
            job_name = "cloud-job-123"

            def status(self):
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                return ""

        def _submit_cloud(self, _cfg):
            self.last_submitted_job = _SubmittedJob()

        monkeypatch.setattr(
            "job_submitter.jobs.base.CloudBackend.submit",
            _submit_cloud,
        )

        cfg = SimpleNamespace(
            submission_type="cloud",
            region="SR004",
            env_variables={},
            job_desc="tester | exp",
        )
        assert job._submit_config(cfg) is job
        assert mock_warn.call_count == 1


@pytest.mark.unit
class TestResolveTrackingConfigEdgeCases:
    def test_nested_list(self) -> None:
        cfg = {"base": "/mnt", "items": [{"path": "${base}/a"}, "${base}/b"]}
        resolved = _resolve_tracking_config(cfg)
        assert resolved["items"][0]["path"] == "/mnt/a"
        assert resolved["items"][1] == "/mnt/b"


@pytest.mark.unit
class TestExtractLogsPathEdgeCases:
    def test_strips_windows_drive(self) -> None:
        cfg = {"loggers": {"tensorboard": {"log_dir": "C:\\data\\tb"}}, "run_name": "r"}
        assert extract_logs_path(cfg) == "/data/tb/r"

    def test_strip_foreign_prefix_via_auto_track(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _, _, local_dir = _patch_inbox_writer(monkeypatch, tmp_path)
        s3_key = "tracker_daemons_v2/pretrain_sr004_inbox.sqlite"
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP", {"SR004": "pretrain"}
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"SR004": s3_key},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {}},
        )

        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "run_name: r\nloggers:\n  tensorboard:\n    log_dir: other:cl:/mnt/tb\n"
        )

        auto_track("SR004", "u", str(cfg))

        params = _fetch_rows(
            local_dir,
            s3_key,
            "SELECT users, logs_path, yaml_path, script_path, timestamp "
            "FROM auto_track_requests",
        )[0]
        assert params[1] == "/mnt/tb/r"


@pytest.mark.unit
class TestResolveTrackingConfigMoreBranches:
    def test_recursive_and_multiple_placeholders(self) -> None:
        cfg = {
            "root": "/mnt",
            "base": "${root}/tb",
            "name": "run1",
            "path": "${base}/${name}",
        }

        resolved = _resolve_tracking_config(cfg)

        assert resolved["base"] == "/mnt/tb"
        assert resolved["path"] == "/mnt/tb/run1"


@pytest.mark.unit
class TestResolveTrackingConfigInterpolationErrors:
    def test_unknown_interpolation_raises(self) -> None:
        cfg = {
            "name": "run1",
            "path": "${missing}/${name}",
        }

        with pytest.raises(ValueError, match="interpolation"):
            _resolve_tracking_config(cfg)


@pytest.mark.unit
class TestStripClusterPrefixMoreBranches:
    def test_malformed_prefix_is_not_stripped(self) -> None:
        # Contains ":" so it enters the foreign-prefix branch,
        # but second segment is empty, so it should fall through unchanged.
        assert (
            _strip_cluster_prefix("other::/mnt/logs", "pretrain", "sr004")
            == "other::/mnt/logs"
        )


@pytest.mark.unit
class TestS3EnvHelpers:
    def test_resolve_auto_track_local_dir_prefers_env_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        override = tmp_path / "custom_autotrack_dir"
        monkeypatch.setenv("TRACKING_AUTO_TRACK_DB_LOCAL_DIR", str(override))

        resolved = _resolve_auto_track_local_dir()

        assert resolved == override
        assert resolved.exists()
        assert resolved.is_dir()

    def test_resolve_auto_track_local_dir_fails_fast_for_bad_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bad_override = tmp_path / "not_a_directory"
        bad_override.write_text("x", encoding="utf-8")
        monkeypatch.setenv("TRACKING_AUTO_TRACK_DB_LOCAL_DIR", str(bad_override))
        monkeypatch.setenv("USER", "tracking_test_user")

        with pytest.raises(FileExistsError):
            _resolve_auto_track_local_dir()

    def test_resolve_auto_track_local_dir_rejects_non_directory_after_mkdir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bad_override = tmp_path / "not_a_directory"
        bad_override.write_text("x", encoding="utf-8")
        monkeypatch.setenv("TRACKING_AUTO_TRACK_DB_LOCAL_DIR", str(bad_override))
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.Path.mkdir",
            lambda *_args, **_kwargs: None,
        )

        with pytest.raises(NotADirectoryError, match="Auto-track local path"):
            _resolve_auto_track_local_dir()

    def test_parse_env_file_key_value_lines_only(self, tmp_path: Path) -> None:
        env_file = tmp_path / "s3_db.env"
        env_file.write_text(
            "\n".join(
                [
                    "",
                    "AWS_ACCESS_KEY_ID=ak",
                    "AWS_SECRET_ACCESS_KEY=sk",
                    "S3_BUCKET=bucket-name",
                    "NO_EQUALS",
                    "=no_key",
                ]
            ),
            encoding="utf-8",
        )

        parsed = _parse_env_file(env_file)

        assert parsed["AWS_ACCESS_KEY_ID"] == "ak"
        assert parsed["AWS_SECRET_ACCESS_KEY"] == "sk"
        assert parsed["S3_BUCKET"] == "bucket-name"
        assert "NO_EQUALS" not in parsed
        assert "" not in parsed

    def test_parse_env_file_empty_file_returns_empty(self, tmp_path: Path) -> None:
        env_file = tmp_path / "empty.env"
        env_file.write_text("", encoding="utf-8")
        assert _parse_env_file(env_file) == {}

    def test_parse_env_file_only_skipped_lines_returns_empty(
        self, tmp_path: Path
    ) -> None:
        env_file = tmp_path / "skipped.env"
        env_file.write_text(
            "\n".join(["", "   ", "NO_EQUALS", "=no_key"]),
            encoding="utf-8",
        )
        assert _parse_env_file(env_file) == {}

    def test_resolve_s3_env_without_secret_file_is_job_env_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRACKING_TEST_ENV", "present")
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS", {}
        )

        assert _resolve_s3_env("SR404") == {}
        assert _resolve_s3_env("SR404", {"FOO": "bar"}) == {"FOO": "bar"}

    def test_resolve_s3_env_uses_passed_job_env_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS", {}
        )
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)

        resolved = _resolve_s3_env("SR404", {"AWS_ACCESS_KEY_ID": "from_job_env"})

        assert resolved["AWS_ACCESS_KEY_ID"] == "from_job_env"

    def test_resolve_secret_env_path_from_default_region_when_no_job_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_file = tmp_path / "s3.env"
        env_file.write_text("S3_BUCKET=bucket", encoding="utf-8")
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {"ENV_FILEPATH_S3_DB": str(env_file)}},
        )

        assert _resolve_secret_env_path("SR004") == env_file

    def test_resolve_secret_env_path_prefers_passed_job_env_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from_build_env = tmp_path / "from_build.env"
        from_build_env.write_text("S3_BUCKET=build", encoding="utf-8")
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {"ENV_FILEPATH_S3_DB": "/not/used"}},
        )

        resolved = _resolve_secret_env_path(
            "SR004", {"ENV_FILEPATH_S3_DB": str(from_build_env)}
        )

        assert resolved == from_build_env

    def test_resolve_s3_env_parse_error_is_non_fatal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_file = tmp_path / "broken.env"
        env_file.write_text("AWS_ACCESS_KEY_ID=ak", encoding="utf-8")
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {"ENV_FILEPATH_S3_DB": str(env_file)}},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage._parse_env_file",
            lambda _path: (_ for _ in ()).throw(ValueError("broken")),
        )

        captured = {"called": False}

        def _capture_exception(*_args, **_kwargs) -> None:
            captured["called"] = True

        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.logger.exception", _capture_exception
        )

        resolved = _resolve_s3_env("SR004")

        assert captured["called"] is True
        assert isinstance(resolved, dict)

    def test_build_s3_storage_config_does_not_mutate_process_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "old-ak")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "old-sk")
        monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

        config = _build_s3_storage_config(
            {
                "AWS_ACCESS_KEY_ID": "new-ak",
                "AWS_SECRET_ACCESS_KEY": "new-sk",
                "AWS_SESSION_TOKEN": "new-st",
                "S3_KEY_PREFIX": "",
            },
            "inbox.sqlite",
        )

        assert config.aws_access_key_id == "new-ak"
        assert config.aws_secret_access_key == "new-sk"
        assert config.aws_session_token == "new-st"
        assert config.key == "inbox.sqlite"
        assert os.environ["AWS_ACCESS_KEY_ID"] == "old-ak"
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == "old-sk"
        assert "AWS_SESSION_TOKEN" not in os.environ

    def test_build_s3_storage_config_defaults_to_secure_virtual_addressing(
        self,
    ) -> None:
        config = _build_s3_storage_config({"S3_KEY_PREFIX": ""}, "inbox.sqlite")

        assert config.verify_ssl is True
        assert config.addressing_style == "virtual"

    def test_build_s3_storage_config_honours_on_prem_gateway_flags(self) -> None:
        config = _build_s3_storage_config(
            {
                "S3_KEY_PREFIX": "",
                "S3_VERIFY_SSL": "false",
                "S3_FORCE_PATH_STYLE": "true",
            },
            "inbox.sqlite",
        )

        assert config.verify_ssl is False
        assert config.addressing_style == "path"

    def test_should_skip_returns_false_for_non_super_secret_missing_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        missing = tmp_path / "plain" / "missing_s3_db.env"
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {"ENV_FILEPATH_S3_DB": str(missing)}},
        )

        assert _should_skip_auto_track_due_missing_secret_file("SR004") is False

    def test_resolve_s3_env_warns_when_secret_file_path_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        missing = tmp_path / "missing_s3_db.env"
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {"ENV_FILEPATH_S3_DB": str(missing)}},
        )

        resolved = _resolve_s3_env("SR004", {"FOO": "bar"})

        assert resolved == {"FOO": "bar"}
        assert "S3 credentials file not found" in caplog.text

    def test_resolve_auto_track_local_dir_uses_per_user_temp_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("TRACKING_AUTO_TRACK_DB_LOCAL_DIR", raising=False)
        monkeypatch.setenv("USER", "coverage_user")

        resolved = _resolve_auto_track_local_dir()

        assert resolved.name == "tracker_db_autotrack_coverage_user"
        assert resolved.exists()

    def test_resolve_auto_track_local_dir_raises_when_user_dir_cannot_be_created(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TRACKING_AUTO_TRACK_DB_LOCAL_DIR", raising=False)
        monkeypatch.setenv("USER", "coverage_user")

        original_mkdir = Path.mkdir

        def deny_autotrack_dirs(self, *args, **kwargs):
            if self.name == "tracker_db_autotrack_coverage_user":
                raise PermissionError("denied")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", deny_autotrack_dirs)

        with pytest.raises(PermissionError, match="denied"):
            _resolve_auto_track_local_dir()


@pytest.mark.unit
class TestBuildAutoTrackErrorsCommandData:
    def test_returns_none_without_submission_type(self) -> None:
        assert (
            _build_auto_track_errors_command_data(
                submission_type=None,
                job_id="job-1",
                logs_path="/logs",
            )
            is None
        )

    def test_returns_none_for_unsupported_submission_type(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert (
            _build_auto_track_errors_command_data(
                submission_type="kubernetes",
                job_id="job-1",
                logs_path="/logs",
            )
            is None
        )
        assert "unsupported submission_type" in caplog.text

    def test_build_auto_track_errors_command_slurm_branch_uses_slurm_log_path(
        self,
    ) -> None:
        payload = _build_auto_track_errors_command_data(
            submission_type="slurm",
            job_id="slurm-42",
            logs_path="/slurm/logs",
        )

        assert payload is not None
        assert payload["logs_info"]["submission_type"] == "slurm"
        assert payload["logs_info"]["slurm_log_path"] == "/slurm/logs"
        assert payload["logs_info"]["job_id"] == "slurm-42"

    def test_build_auto_track_errors_command_omits_empty_job_id(self) -> None:
        payload = _build_auto_track_errors_command_data(
            submission_type="cloud",
            job_id="   ",
            logs_path="/logs",
        )

        assert payload is not None
        assert "job_id" not in payload["logs_info"]


@pytest.mark.unit
class TestSanitizeExtraMetadata:
    def test_none_returns_empty_dict(self) -> None:
        assert _sanitize_extra_metadata(None) == {}

    def test_empty_dict_returns_empty_dict(self) -> None:
        assert _sanitize_extra_metadata({}) == {}

    def test_keeps_json_serializable_values(self) -> None:
        metadata = {
            "job_kind": "rclone_transfer",
            "rclone_json_log": "/tmp/rclone.json",
            "retry_count": 2,
            "ratio": 0.5,
            "enabled": True,
        }
        assert _sanitize_extra_metadata(metadata) == metadata

    def test_drops_non_serializable_value_and_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        sanitized = _sanitize_extra_metadata(
            {"job_kind": "rclone_transfer", "bad_value": object()}
        )

        assert sanitized == {"job_kind": "rclone_transfer"}
        assert "non-JSON-serializable" in caplog.text
        assert "bad_value" in caplog.text


@pytest.mark.unit
class TestTrackingModuleImport:
    def test_raises_helpful_error_when_sqlite_s3_query_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracking_path = (
            Path(__file__).resolve().parents[3]
            / "job_submitter"
            / "core"
            / "tracking_storage.py"
        )
        real_import = builtins.__import__
        module_name = "job_submitter.core.tracking_storage_import_error_test"

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sqlite_s3_query":
                raise ModuleNotFoundError("sqlite_s3_query")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        spec = importlib.util.spec_from_file_location(module_name, tracking_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader

        with pytest.raises(ModuleNotFoundError) as exc_info:
            spec.loader.exec_module(module)
        assert "sqlite_s3_query is not available" in str(exc_info.value)
        assert "contrib/job_submitter/contrib/sqlite_s3_query" in str(exc_info.value)


@pytest.mark.unit
class TestAutoTrackCommandInsertFailure:
    def test_command_insert_failure_is_non_fatal(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class _FailingConnection:
            def execute(self, *_args, **_kwargs):
                raise RuntimeError("insert failed")

        _insert_auto_track_errors_command(
            conn=_FailingConnection(),
            region="SR004",
            logs_path="/tb/run1",
            timestamp="2026-01-01T00:00:00+00:00",
            command_data={"enabled": True},
        )
        assert "Failed to enqueue 'auto_track_errors' command" in caplog.text

    def test_generic_exception_returns_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _patch_inbox_writer(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_WORKSPACE_MAP",
            {"SR004": "pretrain"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.TRACKING_DAEMON_INBOX_S3_KEY_MAP",
            {"SR004": "tracker_daemons_v2/pretrain_sr004_inbox.sqlite"},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking_storage.DEFAULT_REGION_ENVS",
            {"SR004": {}},
        )
        monkeypatch.setattr(
            "job_submitter.core.tracking.extract_logs_path",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        cfg = tmp_path / "config.yaml"
        cfg.write_text("tensorboard_path: /tb\n")

        assert (
            auto_track(
                "SR004",
                "u",
                str(cfg),
                merged_config={"tensorboard_path": "/tb"},
            )
            is False
        )
        assert "Auto-tracking failed (non-blocking)" in caplog.text


@pytest.mark.unit
class TestAutoTrackWriteRetry:
    @pytest.mark.parametrize("failures", [0, 1, 2])
    @pytest.mark.parametrize("result", [False, True])
    def test_retries_on_concurrent_write_and_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, failures: int, result: bool
    ) -> None:
        from job_submitter.core import tracking_storage
        from job_submitter.infrastructure.s3.exceptions import ConcurrentWriteError

        calls = {"n": 0}

        class _Writer:
            def __init__(self, **_kwargs):
                pass

            def write(self):
                calls["n"] += 1
                if calls["n"] <= failures:
                    raise ConcurrentWriteError("conflict")
                return result

        monkeypatch.setattr(tracking_storage, "AutoTrackInboxWriter", _Writer)
        sleeps: list[float] = []
        monkeypatch.setattr(
            "job_submitter.infrastructure.retry.time.sleep", sleeps.append
        )
        assert (
            tracking_storage._write_auto_track_request(
                region="o1", s3_key="k", payload={}, s3_env={}, command_data=None
            )
            is result
        )
        assert calls["n"] == failures + 1
        assert sleeps == [0.5, 1.0][:failures]

    def test_raises_after_exhausted_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from job_submitter.core import tracking_storage
        from job_submitter.infrastructure.s3.exceptions import ConcurrentWriteError

        calls = []

        class _Writer:
            def __init__(self, **_kwargs):
                pass

            def write(self):
                calls.append(None)
                raise ConcurrentWriteError("conflict")

        monkeypatch.setattr(tracking_storage, "AutoTrackInboxWriter", _Writer)
        sleeps = []
        monkeypatch.setattr(
            "job_submitter.infrastructure.retry.time.sleep", sleeps.append
        )
        with pytest.raises(ConcurrentWriteError):
            tracking_storage._write_auto_track_request(
                region="o1", s3_key="k", payload={}, s3_env={}
            )
        assert len(calls) == 3
        assert sleeps == [0.5, 1.0]

    def test_parse_env_file_strips_matching_quotes(self, tmp_path: Path) -> None:
        env_file = tmp_path / "quoted.env"
        env_file.write_text("A=\"one two\"\nB='x'\n", encoding="utf-8")
        assert _parse_env_file(env_file) == {"A": "one two", "B": "x"}
