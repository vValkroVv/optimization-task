from pathlib import Path

import pytest

from job_submitter.core.backends import SlurmBackend, SlurmSubmittedJob
from job_submitter.core.slurm_rest import SlurmRESTError
from job_submitter.jobs.transfers.rclone import ISLAND_RCLONE_CONFIG
from job_submitter.jobs.transfers.transfer2cloud import Transfer2CloudJob

pytestmark = pytest.mark.unit

STAGING_PATH = Path("/mnt/nfs/07-share/transfer2cloud/o1/jd_sync")


def make_job(staging_path: Path = STAGING_PATH) -> Transfer2CloudJob:
    return Transfer2CloudJob(
        source_path=staging_path / "model" / "artifacts.tar.zst",
        staging_path=staging_path,
        exp_name="pam-obm-transfer",
    )


@pytest.fixture(autouse=True)
def closed_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the process-wide retry state that a failed submit leaves behind."""
    monkeypatch.setattr(Transfer2CloudJob, "_retries_exhausted", False)


@pytest.fixture(autouse=True)
def sleep_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record retry backoff instead of really waiting ten minutes."""
    calls: list = []
    monkeypatch.setattr("job_submitter.infrastructure.retry.time.sleep", calls.append)
    return calls


@pytest.fixture
def slurm_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JWT_TOKEN", "jwt")
    monkeypatch.setenv("SLURMRESTD_URLS", "http://slurm:6820")
    monkeypatch.setenv("SLURM_JOB_USER", "optimusprime")
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("RCLONE_CONFIG", raising=False)


@pytest.fixture
def slurm_backend(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Record submitted configs; queue ``errors`` to make submits raise."""
    state: dict = {"configs": [], "errors": [], "job_id": "12345"}

    def submit(backend, config):
        state["configs"].append(config)
        if state["errors"]:
            raise state["errors"].pop(0)
        if state["job_id"] is not None:
            backend.last_submitted_job = SlurmSubmittedJob(state["job_id"])

    monkeypatch.setattr(SlurmBackend, "submit", submit)
    monkeypatch.setattr(
        "job_submitter.jobs.slurm_cpu.auto_track",
        lambda **kwargs: True,
    )
    return state


@pytest.mark.parametrize("region", ["o1", "o2", "o5"])
def test_submit_copies_staged_artifact_to_papaya(
    slurm_rest: None,
    slurm_backend: dict,
    sleep_calls: list,
    region: str,
) -> None:
    staging_path = Path(f"/mnt/nfs/07-share/transfer2cloud/{region}/jd_sync")

    job_id = make_job(staging_path).submit()

    assert job_id == "12345"
    assert sleep_calls == []
    config = slurm_backend["configs"][0]
    assert config.region == region
    assert config.partition == "transfer"
    assert config.user_login == "optimusprime"
    assert config.env_variables == {"HOME": "/home/optimusprime"}
    assert f"--config {ISLAND_RCLONE_CONFIG}" in config.script
    assert (
        f"SOURCE=/mnt/nfs/07-share/transfer2cloud/{region}/jd_sync"
        f"/model/artifacts.tar.zst" in config.script
    )
    assert (
        f"DESTINATION=papayaeye-papaya:transfer2cloud/{region}/jd_sync"
        f"/model/artifacts.tar.zst" in config.script
    )


def test_submit_requires_slurm_rest_configuration(
    monkeypatch: pytest.MonkeyPatch,
    slurm_backend: dict,
) -> None:
    monkeypatch.delenv("SLURM_JWT_TOKEN", raising=False)
    monkeypatch.delenv("SLURM_JWT", raising=False)
    monkeypatch.delenv("SLURMRESTD_URLS", raising=False)
    monkeypatch.delenv("TRACKER_BOT_SLURM_API_BASE", raising=False)

    with pytest.raises(SlurmRESTError, match="not configured"):
        make_job().submit()
    assert slurm_backend["configs"] == []


def test_submit_falls_back_to_user_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
    slurm_rest: None,
    slurm_backend: dict,
) -> None:
    monkeypatch.delenv("SLURM_JOB_USER", raising=False)
    monkeypatch.setenv("USER", "fallback-user")

    make_job().submit()

    assert slurm_backend["configs"][0].user_login == "fallback-user"


def test_submit_requires_a_resolvable_user(
    monkeypatch: pytest.MonkeyPatch,
    slurm_rest: None,
    slurm_backend: dict,
) -> None:
    monkeypatch.delenv("SLURM_JOB_USER", raising=False)
    monkeypatch.delenv("USER", raising=False)

    with pytest.raises(RuntimeError, match="Cannot determine Slurm user"):
        make_job().submit()
    assert slurm_backend["configs"] == []


def test_submit_requires_a_job_id(
    slurm_rest: None,
    slurm_backend: dict,
) -> None:
    slurm_backend["job_id"] = None

    with pytest.raises(RuntimeError, match="did not return a job ID"):
        make_job().submit()


@pytest.mark.parametrize(
    "staging_path",
    [
        Path("/mnt/nfs/07-share/elsewhere/o1/jd_sync"),
        Path("/mnt/nfs/07-share/transfer2cloud/o9/jd_sync"),
    ],
)
def test_submit_rejects_unexpected_staging_path(
    slurm_rest: None,
    slurm_backend: dict,
    staging_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Unexpected transfer staging path"):
        make_job(staging_path).submit()
    assert slurm_backend["configs"] == []


@pytest.mark.parametrize("failures", [1, 2])
def test_submit_retries_slurm_rest_failures(
    slurm_rest: None,
    slurm_backend: dict,
    sleep_calls: list,
    failures: int,
) -> None:
    slurm_backend["errors"] = [SlurmRESTError("slurmrestd down")] * failures

    assert make_job().submit() == "12345"
    assert len(slurm_backend["configs"]) == failures + 1
    assert sleep_calls == [600, 1200][:failures]


@pytest.mark.parametrize("error", [ValueError, RuntimeError])
def test_submit_does_not_retry_unrelated_errors(
    slurm_rest: None,
    slurm_backend: dict,
    sleep_calls: list,
    error: type[Exception],
) -> None:
    slurm_backend["errors"] = [error("invalid job")]

    with pytest.raises(error, match="invalid job"):
        make_job().submit()

    assert len(slurm_backend["configs"]) == 1
    assert sleep_calls == []


def test_submit_gives_up_and_stops_waiting_for_later_transfers(
    slurm_rest: None,
    slurm_backend: dict,
    sleep_calls: list,
) -> None:
    slurm_backend["errors"] = [SlurmRESTError("slurmrestd down")] * 4

    with pytest.raises(SlurmRESTError, match="slurmrestd down"):
        make_job().submit()

    assert len(slurm_backend["configs"]) == 3
    assert sleep_calls == [600, 1200]

    with pytest.raises(SlurmRESTError, match="slurmrestd down"):
        make_job().submit()

    assert len(slurm_backend["configs"]) == 4
    assert sleep_calls == [600, 1200]


def test_successful_submit_restores_the_retry_waits(
    monkeypatch: pytest.MonkeyPatch,
    slurm_rest: None,
    slurm_backend: dict,
    sleep_calls: list,
) -> None:
    monkeypatch.setattr(Transfer2CloudJob, "_retries_exhausted", True)

    make_job().submit()
    slurm_backend["errors"] = [SlurmRESTError("slurmrestd down")]
    make_job().submit()

    assert sleep_calls == [600]
