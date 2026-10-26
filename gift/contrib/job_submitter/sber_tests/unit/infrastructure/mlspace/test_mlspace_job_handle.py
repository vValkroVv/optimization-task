import pytest

from job_submitter.infrastructure.mlspace import MlSpaceClient
from job_submitter.infrastructure.mlspace.job_handle import MlSpaceJobHandle
from sber_tests.test_support.mlspace import FakeHttp, set_mlspace_env

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "statuses",
    [
        ("Inqueue", "Starting", "Running", "Completing", "Completed"),
        ("inqueue", "pending", "running", "completing", "completed"),
    ],
)
def test_status_is_reported_in_monitor_terms(monkeypatch, statuses):
    set_mlspace_env(monkeypatch)
    handle = MlSpaceJobHandle(
        MlSpaceClient.from_env("SR008", FakeHttp(statuses)), "lm-mpi-job-1"
    )

    seen = [handle.status() for _ in statuses]

    assert seen == [
        "Job status=Pending",
        "Job status=Pending",
        "Job status=Running",
        "Job status=Running",
        "Job status=Completed",
    ]


def test_unknown_status_is_passed_through(monkeypatch):
    set_mlspace_env(monkeypatch)
    handle = MlSpaceJobHandle(
        MlSpaceClient.from_env("SR008", FakeHttp(("job not found",))), "x"
    )

    assert handle.status() == "Job status=job not found"


def test_logs_are_printed_and_returned_like_client_lib(monkeypatch, capsys):
    set_mlspace_env(monkeypatch)
    http = FakeHttp()
    handle = MlSpaceJobHandle(MlSpaceClient.from_env("SR008", http), "lm-mpi-job-1")

    logs = handle.logs(tail=2, verbose=False)

    assert logs == "line1\nline2\n"
    assert capsys.readouterr().out == "line1\nline2\n"
    assert http.calls[-1][3]["params"] == {
        "region": "SR008",
        "tail": 2,
        "verbose": False,
    }
