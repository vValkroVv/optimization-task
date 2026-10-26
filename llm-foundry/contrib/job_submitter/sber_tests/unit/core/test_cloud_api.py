import pytest
import requests

from job_submitter.core.backends import CloudBackend
from job_submitter.core.cloud_api import CLOUD_API_ENV, CloudApi
from job_submitter.core.job_config import JobTemplate
from job_submitter.core.job_kill import CloudKillBackend
from sber_tests.test_support.mlspace import API_URL, FakeHttp, set_mlspace_env

pytestmark = pytest.mark.unit


@pytest.fixture
def public_http(monkeypatch):
    set_mlspace_env(monkeypatch)
    monkeypatch.setenv(CLOUD_API_ENV, "public")
    monkeypatch.delenv("NB_PREFIX", raising=False)
    fake = FakeHttp()
    monkeypatch.setattr(requests, "Session", lambda: fake)
    return fake


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, CloudApi.GATEWAY),
        ("public", CloudApi.PUBLIC),
        (" Public ", CloudApi.PUBLIC),
    ],
)
def test_cloud_api_is_read_from_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(CLOUD_API_ENV, raising=False)
    else:
        monkeypatch.setenv(CLOUD_API_ENV, value)

    assert CloudApi.from_env() is expected


def test_unknown_cloud_api_is_rejected(monkeypatch):
    monkeypatch.setenv(CLOUD_API_ENV, "nope")

    with pytest.raises(ValueError, match=CLOUD_API_ENV):
        CloudApi.from_env()


def test_gateway_backends_keep_using_client_lib(monkeypatch):
    monkeypatch.delenv(CLOUD_API_ENV, raising=False)
    monkeypatch.setattr(
        "job_submitter.core.backends.import_module", lambda _: "client_lib"
    )
    monkeypatch.setattr(
        "job_submitter.core.job_kill.import_module", lambda _: "client_lib"
    )

    assert CloudBackend().client_lib == "client_lib"
    assert CloudKillBackend().client_lib == "client_lib"


def test_public_submit_creates_job_and_announces_id(public_http, capsys):
    cfg = JobTemplate(
        script="/nfs/x/run.sh",
        base_image="cr.ai.cloud.ru/x/img:1",
        instance_type="a100",
        region="SR008",
        allocation_name="alloc",
        queue_name="q",
        env_variables={"A": "1"},
    )
    backend = CloudBackend()

    backend.submit(cfg)

    method, url, headers, kwargs = public_http.calls[0]
    assert backend.client_lib is None
    assert backend.last_submitted_job.job_name == "lm-mpi-job-1"
    assert capsys.readouterr().out.splitlines()[-1] == "Cloud job id: lm-mpi-job-1"
    assert (method, url, headers["x-region"]) == ("POST", f"{API_URL}/jobs", "SR008")
    assert kwargs["json"]["script"] == "/nfs/x/run.sh"
    assert kwargs["json"]["env_variables"] == {"A": "1"}


def test_public_kill_deletes_job(public_http):
    assert (
        CloudKillBackend().kill("lm-mpi-job-1", "SR008") == 'Job "lm-mpi-job-1" deleted'
    )
    assert public_http.calls[-1][:2] == ("DELETE", f"{API_URL}/jobs/lm-mpi-job-1")
