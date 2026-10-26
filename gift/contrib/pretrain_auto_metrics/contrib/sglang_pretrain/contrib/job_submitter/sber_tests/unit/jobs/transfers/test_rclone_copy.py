import pytest

from job_submitter.core.backends import SlurmBackend, SlurmSubmittedJob
from job_submitter.jobs.transfers.rclone import ISLAND_RCLONE_CONFIG
from job_submitter.jobs.transfers.rclone_copy import RcloneCopyJob

pytestmark = pytest.mark.unit


def make_job(**overrides) -> RcloneCopyJob:
    fields = {
        "user_login": "optimusprime",
        "exp_name": "pam-archive-transfer",
        "region": "o1",
        "source_path": "/mnt/nfs/transfer2cloud/o1/artifacts.tar",
        "destination_path": "papayaeye-papaya:transfer2cloud/o1/artifacts.tar",
    }
    fields.update(overrides)
    return RcloneCopyJob(**fields)


def test_to_config_builds_transfer_partition_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RCLONE_CONFIG", raising=False)
    job = make_job()

    config = job.to_config()

    assert config.partition == "transfer"
    assert config.cpu_only is True
    assert config.gpu_num == 0
    assert config.cpus_per_task == 4
    assert config.memory == "8G"
    assert config.slurm_submit_type == "api"
    assert config.env_variables == {"HOME": "/home/optimusprime"}
    assert config.script == job.script
    assert f"--config {ISLAND_RCLONE_CONFIG}" in config.script
    assert "SOURCE=/mnt/nfs/transfer2cloud/o1/artifacts.tar" in config.script
    assert (
        "DESTINATION=papayaeye-papaya:transfer2cloud/o1/artifacts.tar" in config.script
    )


def test_to_config_keeps_explicit_script_and_home() -> None:
    job = make_job(
        script="echo custom",
        extra_env={"HOME": "/home/elsewhere"},
    )

    config = job.to_config()

    assert config.script == "echo custom"
    assert config.env_variables == {"HOME": "/home/elsewhere"}


def test_from_spec_builds_copy_job(monkeypatch: pytest.MonkeyPatch, tmp_yaml) -> None:
    monkeypatch.delenv("RCLONE_CONFIG", raising=False)
    spec = tmp_yaml(
        "spec.yaml",
        "user_login: optimusprime\n"
        "exp_name: pam-archive-transfer\n"
        "region: o5\n"
        "source_path: /mnt/nfs/artifacts.tar\n"
        "destination_path: remote:bucket/artifacts.tar\n",
    )

    job = RcloneCopyJob.from_spec(spec)

    assert job.region == "o5"
    assert "rclone copyto" in job.to_config().script


def test_submit_runs_built_script_in_transfer_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RCLONE_CONFIG", raising=False)
    submitted = {}

    def submit(backend, config):
        submitted["config"] = config
        backend.last_submitted_job = SlurmSubmittedJob("12345")

    monkeypatch.setattr(SlurmBackend, "submit", submit)
    monkeypatch.setattr(
        "job_submitter.jobs.slurm_cpu.auto_track",
        lambda **kwargs: True,
    )
    job = make_job()

    job.submit()

    assert job.submitted_job_id == "12345"
    assert "rclone copyto" in submitted["config"].script
