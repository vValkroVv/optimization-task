from types import SimpleNamespace

import pytest

from job_submitter.core.job_kill import (
    CloudKillBackend,
    KillBackend,
    SlurmKillBackend,
    get_kill_backend,
    kill_job,
)
from job_submitter.core.runtime_context import (
    RuntimeContext,
    _RuntimeContextProbeJob,
    detect_runtime_context,
)


pytestmark = pytest.mark.unit


def test_detect_runtime_context_prefers_slurm_env(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "o5")
    monkeypatch.setenv("CLUSTER_KEY", "SR008")

    context = detect_runtime_context()

    assert context == RuntimeContext(region="o5", submission_type="slurm")


def test_cloud_backend_calls_client_lib_kill(monkeypatch) -> None:
    called = {}

    fake_client_lib = SimpleNamespace(
        kill=lambda job_name, region=None: (
            called.update({"job_name": job_name, "region": region}) or "ok"
        )
    )
    monkeypatch.setattr(
        "job_submitter.core.job_kill.import_module", lambda _: fake_client_lib
    )

    result = CloudKillBackend().kill("job-1", "SR008")

    assert result == "ok"
    assert called == {"job_name": "job-1", "region": "SR008"}


def test_slurm_backend_runs_scancel(monkeypatch) -> None:
    called = {}

    def fake_run(cmd, text, capture_output):
        called["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("job_submitter.core.job_kill.subprocess.run", fake_run)

    result = SlurmKillBackend().kill("12345", "o1")

    assert called["cmd"] == ["scancel", "12345"]
    assert result == 'Slurm job "12345" cancelled in region "o1"'


def test_slurm_backend_uses_rest_when_configured(monkeypatch) -> None:
    cancelled = []

    class FakeRestClient:
        def cancel(self, job_id):
            cancelled.append(job_id)

    monkeypatch.setattr(
        "job_submitter.core.job_kill.has_slurm_rest_config",
        lambda: True,
    )
    monkeypatch.setattr(
        "job_submitter.core.job_kill.SlurmRestClient",
        FakeRestClient,
    )

    result = SlurmKillBackend().kill("12345", "o1")

    assert cancelled == ["12345"]
    assert result == 'Slurm job "12345" cancelled in region "o1"'


def test_slurm_backend_raises_on_scancel_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "job_submitter.core.job_kill.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="permission denied"
        ),
    )

    with pytest.raises(RuntimeError, match="permission denied"):
        SlurmKillBackend().kill("12345", "o1")


def test_slurm_backend_raises_when_scancel_is_missing(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise FileNotFoundError("scancel")

    monkeypatch.setattr("job_submitter.core.job_kill.subprocess.run", _raise)

    with pytest.raises(RuntimeError, match="scancel` not found"):
        SlurmKillBackend().kill("12345", "o1")


def test_get_kill_backend_returns_cloud_and_slurm_instances() -> None:
    assert isinstance(get_kill_backend("cloud"), CloudKillBackend)
    assert isinstance(get_kill_backend("slurm"), SlurmKillBackend)


def test_get_kill_backend_raises_for_unknown_submission_type() -> None:
    with pytest.raises(ValueError, match="Unsupported submission_type"):
        get_kill_backend("unknown")


def test_kill_job_uses_detected_context(monkeypatch) -> None:
    calls = {}

    class FakeBackend:
        def kill(self, job_id: str, region: str) -> str:
            calls["job_id"] = job_id
            calls["region"] = region
            return "done"

    monkeypatch.setattr(
        "job_submitter.core.job_kill.detect_runtime_context",
        lambda: RuntimeContext(region="SR008", submission_type="cloud"),
    )
    monkeypatch.setattr(
        "job_submitter.core.job_kill.get_kill_backend",
        lambda submission_type: FakeBackend(),
    )

    result = kill_job("job-2")

    assert result == "done"
    assert calls == {"job_id": "job-2", "region": "SR008"}


def test_kill_job_rejects_blank_job_id() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        kill_job("   ")


def test_kill_backend_base_super_method_returns_none() -> None:
    class _BackendWithSuperKill(KillBackend):
        def kill(self, job_id: str, region: str) -> str:
            return super().kill(job_id, region)

    backend = _BackendWithSuperKill()
    assert backend.kill("job", "SR008") is None


def test_runtime_probe_job_has_stub_script_and_before_submit() -> None:
    probe = _RuntimeContextProbeJob(
        user_login="job_submitter_cli",
        exp_name="runtime_context_probe",
    )
    assert probe._script() == "echo runtime-context-probe"
    assert probe._before_submit() is None


def test_detect_runtime_context_raises_on_probe_exception(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(
        _RuntimeContextProbeJob,
        "_get_current_region",
        _raise,
    )

    with pytest.raises(RuntimeError, match="Failed to detect runtime context"):
        detect_runtime_context()


def test_detect_runtime_context_raises_on_empty_region(monkeypatch) -> None:
    monkeypatch.setattr(
        _RuntimeContextProbeJob, "_get_current_region", lambda self: (None, "slurm")
    )

    with pytest.raises(RuntimeError, match="region/submission_type are empty"):
        detect_runtime_context()
