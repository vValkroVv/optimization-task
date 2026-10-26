import logging

import pytest

from job_submitter.core.backends import SlurmBackend, SlurmSubmittedJob
from job_submitter.jobs.any_script import AnyScriptJob
from job_submitter.jobs.infinite import InfiniteJob
from job_submitter.jobs.slurm_cpu import SlurmCpuJob


@pytest.mark.unit
class TestSimpleJobs:
    def test_any_script_job_returns_script_verbatim(self) -> None:
        job = AnyScriptJob(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            script="python train.py",
        )
        assert job._script() == "python train.py"

    def test_infinite_job_has_default_infinite_script(self) -> None:
        job = InfiniteJob(
            user_login="u", exp_name="e", submission_type="cloud", region="SR008"
        )
        assert job._script() == "sleep infinity"

    def test_slurm_cpu_job_submits_transfer_partition_without_gpu(
        self,
        monkeypatch,
    ) -> None:
        submitted = {}
        tracked = {}

        def submit(backend, config):
            submitted["config"] = config
            backend.last_submitted_job = SlurmSubmittedJob("12345")

        monkeypatch.setattr(SlurmBackend, "submit", submit)
        monkeypatch.setattr(
            "job_submitter.jobs.slurm_cpu.auto_track",
            lambda **kwargs: tracked.update(kwargs) or True,
        )
        job = SlurmCpuJob(
            user_login="auto-transfer",
            exp_name="xfer-request",
            region="o1",
            script="rclone copy /source remote:path",
            cpus_per_task=16,
            memory="32G",
            job_logdir="/tmp/jobs",
            slurm_logdir="/tmp/slurm",
        )

        assert job.submitted_job_id is None
        job.submit()

        config = submitted["config"]
        assert job.submitted_job_id == "12345"
        assert config.job_desc == "xfer-request"
        assert config.partition == "transfer"
        assert config.slurm_submit_type == "sbatch"
        assert config.cpu_only is True
        assert config.gpu_num == 0
        assert config.base_image is None
        assert config.cpus_per_task == 16
        assert config.memory == "32G"
        assert tracked == {
            "region": "o1",
            "user_login": "auto-transfer",
            "config_path": None,
            "env": {},
            "submission_type": "slurm",
            "job_id": "12345",
            "auto_track_users": [],
            "enable_auto_share": False,
            "logs_path": "/tmp/slurm/slurm-12345.out",
            "extra_metadata": {},
            "job_title": "xfer-request",
        }

    def test_slurm_cpu_job_from_spec_loads_mapping(self, tmp_path) -> None:
        spec = tmp_path / "spec.yaml"
        spec.write_text(
            "user_login: auto-transfer\n"
            "exp_name: xfer-request\n"
            "region: o1\n"
            "script: echo hi\n"
            "extra_env:\n"
            "  FOO: bar\n",
            encoding="utf-8",
        )

        job = SlurmCpuJob.from_spec(spec)

        assert job.user_login == "auto-transfer"
        assert job.exp_name == "xfer-request"
        assert job.region == "o1"
        assert job.script == "echo hi"
        assert job.extra_env == {"FOO": "bar"}

    def test_slurm_cpu_job_from_spec_rejects_non_mapping(self, tmp_path) -> None:
        spec = tmp_path / "spec.yaml"
        spec.write_text("- not a mapping\n", encoding="utf-8")

        with pytest.raises(ValueError, match="mapping"):
            SlurmCpuJob.from_spec(spec)

    def test_slurm_cpu_job_from_spec_empty_file_uses_empty_mapping(
        self, tmp_path
    ) -> None:
        spec = tmp_path / "spec.yaml"
        spec.write_text("", encoding="utf-8")

        with pytest.raises(TypeError):
            SlurmCpuJob.from_spec(spec)

    def test_slurm_cpu_job_to_config_reads_slurmrestd_urls_from_env(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820||http://b:6820")
        monkeypatch.setenv("SLURM_JWT_TOKEN", "tok")
        job = SlurmCpuJob(
            user_login="auto-transfer",
            exp_name="xfer-request",
            region="o1",
            script="echo hi",
        )

        config = job.to_config()

        assert config.slurmrestd_urls == ["http://a:6820", "http://b:6820"]
        assert config.slurm_jwt_token == "tok"
        assert config.job_logdir == "/mnt/nfs/01-home/infidelis/jobs_logs"
        assert config.slurm_logdir == "/mnt/nfs/01-home/infidelis/slurm_logs"

    def test_slurm_cpu_job_to_config_forwards_long_running_options(self) -> None:
        job = SlurmCpuJob(
            user_login="auto-transfer",
            exp_name="tb-hot-o1",
            region="o1",
            script="python -m auto_transfer.hot.runner",
            extra_env={"TB_LOG_DIR": "/data/tensorboard/run"},
            time_limit="12:00:00",
            requeue=True,
        )

        config = job.to_config()

        assert config.time_limit == "12:00:00"
        assert config.requeue is True
        assert config.comment == "/data/tensorboard/run"

    def test_slurm_cpu_job_to_config_supports_rest_env_aliases(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("SLURMRESTD_URLS", raising=False)
        monkeypatch.delenv("SLURM_JWT_TOKEN", raising=False)
        monkeypatch.setenv(
            "TRACKER_BOT_SLURM_API_BASE",
            "http://slurm:6820",
        )
        monkeypatch.setenv("SLURM_JWT", "alias-token")
        job = SlurmCpuJob(
            user_login="auto-transfer",
            exp_name="xfer-request",
            region="o5",
            script="echo hi",
            slurm_submit_type="api",
        )

        config = job.to_config()

        assert config.slurmrestd_urls == ["http://slurm:6820"]
        assert config.slurm_jwt_token == "alias-token"
        assert config.slurm_submit_type == "api"

    def test_slurm_cpu_job_to_config_keeps_explicit_slurmrestd_urls(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820||http://b:6820")
        job = SlurmCpuJob(
            user_login="auto-transfer",
            exp_name="xfer-request",
            region="o1",
            script="echo hi",
            slurmrestd_urls=["http://explicit:6820"],
            slurm_jwt_token="explicit-tok",
        )

        config = job.to_config()

        assert config.slurmrestd_urls == ["http://explicit:6820"]
        assert config.slurm_jwt_token == "explicit-tok"

    def test_slurm_cpu_job_dry_run_skips_backend(self, monkeypatch, capsys) -> None:
        called = {"count": 0}

        def submit(backend, config):
            called["count"] += 1
            backend.last_submitted_job = SlurmSubmittedJob("should-not-run")

        monkeypatch.setattr(SlurmBackend, "submit", submit)
        monkeypatch.setattr(
            "job_submitter.jobs.slurm_cpu.auto_track",
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("dry-run must not auto-track")
            ),
        )
        job = SlurmCpuJob(
            user_login="auto-transfer",
            exp_name="xfer-request",
            region="o1",
            script="echo hi",
            dry_run=True,
        )

        result = job.submit()

        assert result is job
        assert called["count"] == 0
        assert job.submitted_job_id is None
        assert "Dry run:" in capsys.readouterr().out

    def test_slurm_cpu_job_uses_explicit_tracking_configuration(
        self,
        monkeypatch,
    ) -> None:
        tracked = {}

        def submit(backend, config):
            backend.last_submitted_job = SlurmSubmittedJob("321")

        monkeypatch.setattr(SlurmBackend, "submit", submit)
        monkeypatch.setattr(
            "job_submitter.jobs.slurm_cpu.auto_track",
            lambda **kwargs: tracked.update(kwargs) or True,
        )
        job = SlurmCpuJob(
            user_login="alice",
            exp_name="cpu-job",
            region="o5",
            script="echo hi",
            extra_env={"TRACKING_S3_ENV_FILE": "/secrets/tracker.env"},
            tracking_config_path="/configs/experiment.yaml",
            tracking_logs_path="/logs/cpu-job",
            auto_track_users=["bob"],
            enable_auto_share=True,
        )

        job.submit()

        assert tracked["config_path"] == "/configs/experiment.yaml"
        assert tracked["logs_path"] == "/logs/cpu-job"
        assert tracked["env"] == {"TRACKING_S3_ENV_FILE": "/secrets/tracker.env"}
        assert tracked["auto_track_users"] == ["bob"]
        assert tracked["enable_auto_share"] is True

    def test_slurm_cpu_job_forwards_tracking_metadata(
        self,
        monkeypatch,
    ) -> None:
        tracked = {}

        def submit(backend, config):
            backend.last_submitted_job = SlurmSubmittedJob("555")

        monkeypatch.setattr(SlurmBackend, "submit", submit)
        monkeypatch.setattr(
            "job_submitter.jobs.slurm_cpu.auto_track",
            lambda **kwargs: tracked.update(kwargs) or True,
        )
        job = SlurmCpuJob(
            user_login="auto-transfer",
            exp_name="xfer-request",
            region="o1",
            script="rclone copy /source remote:path",
            tracking_metadata={
                "job_kind": "rclone_transfer",
                "rclone_json_log": "/tmp/slurm/rclone-copy-attempt-1.json",
            },
        )

        job.submit()

        assert tracked["extra_metadata"] == {
            "job_kind": "rclone_transfer",
            "rclone_json_log": "/tmp/slurm/rclone-copy-attempt-1.json",
        }

    @pytest.mark.parametrize("tracking_result", [False, RuntimeError("offline")])
    def test_slurm_cpu_job_keeps_submission_when_tracking_fails(
        self,
        monkeypatch,
        tracking_result,
    ) -> None:
        def submit(backend, config):
            backend.last_submitted_job = SlurmSubmittedJob("777")

        def track(**kwargs):
            if isinstance(tracking_result, Exception):
                raise tracking_result
            return tracking_result

        monkeypatch.setattr(SlurmBackend, "submit", submit)
        monkeypatch.setattr("job_submitter.jobs.slurm_cpu.auto_track", track)
        job = SlurmCpuJob(
            user_login="alice",
            exp_name="cpu-job",
            region="o2",
            script="echo hi",
        )

        assert job.submit() is job
        assert job.submitted_job_id == "777"
        assert job.tracking_registered is False

    def test_slurm_cpu_job_skips_tracking_without_submitted_job(
        self,
        monkeypatch,
    ) -> None:
        def submit(backend, config):
            backend.last_submitted_job = None

        monkeypatch.setattr(SlurmBackend, "submit", submit)
        monkeypatch.setattr(
            "job_submitter.jobs.slurm_cpu.auto_track",
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("missing job must not auto-track")
            ),
        )
        job = SlurmCpuJob(
            user_login="alice",
            exp_name="cpu-job",
            region="o1",
            script="echo hi",
        )

        assert job.submit() is job
        assert job.submitted_job_id is None

    def test_slurm_cpu_job_reports_missing_tracking_path(
        self,
        monkeypatch,
        caplog,
    ) -> None:
        tracked = {}

        def submit(backend, config):
            backend.last_submitted_job = SlurmSubmittedJob("888")

        monkeypatch.setattr(SlurmBackend, "submit", submit)
        monkeypatch.setattr(
            "job_submitter.jobs.slurm_cpu.auto_track",
            lambda **kwargs: tracked.update(kwargs) or False,
        )
        job = SlurmCpuJob(
            user_login="alice",
            exp_name="cpu-job",
            region="custom",
            script="echo hi",
        )

        with caplog.at_level(logging.WARNING):
            job.submit()

        assert tracked["config_path"] is None
        assert tracked["logs_path"] is None
        assert job.tracking_registered is False
        assert "888" in caplog.text
