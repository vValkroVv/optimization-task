from unittest.mock import DEFAULT, Mock, call

import pytest
import requests

from job_submitter.infrastructure.mlspace import (
    MlSpaceApiError,
    MlSpaceClient,
    MlSpaceJobNotFoundError,
)
from sber_tests.test_support.mlspace import (
    API_URL,
    FakeHttp,
    FakeResponse,
    set_mlspace_env,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def sleep(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("job_submitter.infrastructure.retry.time.sleep", sleep)
    monkeypatch.setattr(
        "job_submitter.infrastructure.retry.random.uniform", lambda a, b: 0
    )
    return sleep


@pytest.fixture
def http(monkeypatch):
    set_mlspace_env(monkeypatch)
    return FakeHttp()


@pytest.fixture
def client(http):
    return MlSpaceClient.from_env("SR008", http)


@pytest.fixture
def http_request(http, monkeypatch):
    request = Mock(wraps=http.request)
    monkeypatch.setattr(http, "request", request)
    return request


def test_submit_job_posts_params_without_nones(client, http):
    job = client.submit_job(
        {"script": "/nfs/x/run.sh", "queue_name": "q", "max_retry": None}
    )

    method, url, headers, kwargs = http.calls[0]
    assert job.job_name == "lm-mpi-job-1"
    assert (method, url) == ("POST", f"{API_URL}/jobs")
    assert headers["x-region"] == "SR008"
    assert kwargs["json"] == {"script": "/nfs/x/run.sh", "queue_name": "q"}
    assert kwargs["timeout"] == 60


def test_create_job_requires_job_name_in_response(client, http):
    http.response = FakeResponse(200, {})

    with pytest.raises(MlSpaceApiError, match="job creation returned no job_name"):
        client.create_job({"script": "x"})


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "running"}, "running"),
        ({"job_status": "Completed"}, "Completed"),
        ({"other": 1}, "{'other': 1}"),
    ],
)
def test_job_status_reads_known_fields(client, http, payload, expected):
    http.response = FakeResponse(200, payload)

    assert client.job_status("lm-mpi-job-1") == expected


def test_job_status_raises_on_http_error(client, http):
    http.response = FakeResponse(404, text="not found")

    with pytest.raises(MlSpaceApiError, match="job status failed: HTTP 404 not found"):
        client.job_status("lm-mpi-job-1")


def test_job_logs_passes_tail_and_verbose(client, http):
    assert client.job_logs("lm-mpi-job-1", 5, True) == "line1\nline2\n"
    assert http.calls[-1][3]["params"] == {
        "region": "SR008",
        "tail": 5,
        "verbose": True,
    }


def test_job_logs_returns_placeholder_on_http_error(client, http, http_request, sleep):
    http.response = FakeResponse(503, text="down")

    assert (
        client.job_logs("lm-mpi-job-1", 0, True)
        == "<logs unavailable: HTTP 503 down>\n"
    )
    assert http_request.call_count == 3
    assert sleep.call_args_list == [call(5), call(10)]


def test_delete_job_targets_job_in_region(client, http):
    assert client.delete_job("lm-mpi-job-1") == 'Job "lm-mpi-job-1" deleted'
    method, url, _, kwargs = http.calls[-1]
    assert (method, url) == ("DELETE", f"{API_URL}/jobs/lm-mpi-job-1")
    assert kwargs["params"] == {"region": "SR008"}


def test_delete_job_reports_missing_job(client, http):
    http.response = FakeResponse(
        200, {"job_name": "lm-mpi-job-1", "status": "job not found"}
    )

    with pytest.raises(MlSpaceJobNotFoundError, match='job "lm-mpi-job-1" not found'):
        client.delete_job("lm-mpi-job-1")


def test_delete_job_rejects_unexpected_response(client, http):
    http.response = FakeResponse(200, {"job_name": "lm-mpi-job-1", "status": "weird"})

    with pytest.raises(MlSpaceApiError, match="unexpected response"):
        client.delete_job("lm-mpi-job-1")


def test_request_is_retried_once_with_new_token_after_401(client, http):
    http.reject_next = True

    client.job_status("lm-mpi-job-1")

    assert http.logins == 2
    assert http.calls[-1][2]["Authorization"] == "Bearer tok2"


@pytest.mark.parametrize("logs", [False, True], ids=["status", "logs"])
@pytest.mark.parametrize(
    "failure",
    [
        FakeResponse(502),
        FakeResponse(503),
        FakeResponse(504),
        requests.ConnectTimeout("connect timeout"),
        requests.ReadTimeout("read timeout"),
        requests.ConnectionError("connection reset"),
    ],
    ids=["502", "503", "504", "connect-timeout", "read-timeout", "connection-error"],
)
def test_get_recovers_after_two_transient_failures(
    client, http_request, sleep, failure, logs
):
    http_request.side_effect = [
        failure,
        failure,
        FakeResponse(200, {"status": "running"}, text="training log\n"),
    ]

    if logs:
        assert client.job_logs("job-1", 5, True) == "training log\n"
    else:
        assert client.job_status("job-1") == "running"

    assert http_request.call_count == 3
    assert sleep.call_args_list == [call(5), call(10)]
    for request in http_request.call_args_list:
        assert request.args == (
            "GET",
            f"{API_URL}/jobs/job-1" + ("/logs" if logs else ""),
        )
        assert request.kwargs["timeout"] == (300 if logs else 30)
        assert request.kwargs["params"]["region"] == "SR008"


@pytest.mark.parametrize("status", [502, 503, 504])
def test_status_stops_after_three_http_failures(client, http_request, sleep, status):
    http_request.return_value = FakeResponse(status, text="down")

    with pytest.raises(MlSpaceApiError, match=f"HTTP {status} down"):
        client.job_status("job-1")

    assert http_request.call_count == 3
    assert sleep.call_args_list == [call(5), call(10)]


@pytest.mark.parametrize(
    "error", [requests.ConnectTimeout, requests.ReadTimeout, requests.ConnectionError]
)
def test_status_stops_after_three_network_failures(client, http_request, sleep, error):
    http_request.side_effect = error("unavailable")

    with pytest.raises(error, match="unavailable"):
        client.job_status("job-1")

    assert http_request.call_count == 3
    assert sleep.call_args_list == [call(5), call(10)]


@pytest.mark.parametrize("status", [400, 403, 404, 422, 429, 500])
def test_status_does_not_retry_other_http_errors(client, http_request, sleep, status):
    http_request.return_value = FakeResponse(status)

    with pytest.raises(MlSpaceApiError, match=f"HTTP {status}"):
        client.job_status("job-1")

    http_request.assert_called_once()
    sleep.assert_not_called()


def test_status_does_not_retry_ssl_errors(client, http_request, sleep):
    http_request.side_effect = requests.exceptions.SSLError("invalid certificate")

    with pytest.raises(requests.exceptions.SSLError):
        client.job_status("job-1")

    http_request.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize("delete", [False, True], ids=["submit", "delete"])
@pytest.mark.parametrize("status", [502, 503, 504])
def test_writes_do_not_retry_http_failures(client, http_request, sleep, delete, status):
    http_request.return_value = FakeResponse(status)

    with pytest.raises(MlSpaceApiError, match=f"HTTP {status}"):
        if delete:
            client.delete_job("job-1")
        else:
            client.create_job({"script": "train.py"})

    http_request.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize("delete", [False, True], ids=["submit", "delete"])
@pytest.mark.parametrize(
    "error", [requests.ConnectTimeout, requests.ReadTimeout, requests.ConnectionError]
)
def test_writes_do_not_retry_network_failures(
    client, http_request, sleep, delete, error
):
    http_request.side_effect = error("unavailable")

    with pytest.raises(error):
        if delete:
            client.delete_job("job-1")
        else:
            client.create_job({"script": "train.py"})

    http_request.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("statuses", "logins", "delays"),
    [
        ([401, 200], 2, []),
        ([401, 401], 2, []),
        ([503, 401, 200], 2, [5]),
        ([401, 503, 200], 2, [10]),
        ([503, 503, 401], 1, [5, 10]),
        ([401, 503, 401], 2, [10]),
    ],
)
def test_auth_refresh_shares_get_attempt_limit(
    client, http, http_request, sleep, statuses, logins, delays
):
    http_request.side_effect = [
        FakeResponse(status, {"status": "running"}) for status in statuses
    ]

    if statuses[-1] == 200:
        assert client.job_status("job-1") == "running"
    else:
        with pytest.raises(MlSpaceApiError, match="HTTP 401"):
            client.job_status("job-1")

    assert http_request.call_count == len(statuses)
    assert http.logins == logins
    assert sleep.call_args_list == [call(delay) for delay in delays]
    assert (
        http_request.call_args.kwargs["headers"]["Authorization"]
        == f"Bearer tok{logins}"
    )


@pytest.mark.parametrize("status", [200, 401, 503])
def test_submit_only_repeats_after_auth_refresh(
    client, http, http_request, sleep, status
):
    http_request.side_effect = [
        FakeResponse(401),
        FakeResponse(status, {"job_name": "job-1"}),
    ]

    if status == 200:
        assert client.create_job({"script": "train.py"}) == "job-1"
    else:
        with pytest.raises(MlSpaceApiError, match=f"HTTP {status}"):
            client.create_job({"script": "train.py"})

    assert http_request.call_count == 2
    assert http.logins == 2
    assert http_request.call_args.kwargs["headers"]["Authorization"] == "Bearer tok2"
    sleep.assert_not_called()


def test_submit_does_not_retry_timeout_after_auth_refresh(client, http_request, sleep):
    http_request.side_effect = [FakeResponse(401), requests.ReadTimeout("unavailable")]

    with pytest.raises(requests.ReadTimeout, match="unavailable"):
        client.create_job({"script": "train.py"})

    assert http_request.call_count == 2
    sleep.assert_not_called()


def test_get_adds_jitter_to_retry_delays(client, http_request, sleep, monkeypatch):
    monkeypatch.setattr(
        "job_submitter.infrastructure.retry.random.uniform", lambda a, b: 0.25
    )
    http_request.side_effect = [
        FakeResponse(503),
        requests.ReadTimeout("unavailable"),
        FakeResponse(200, {"status": "running"}),
    ]

    assert client.job_status("job-1") == "running"
    assert sleep.call_args_list == [call(5.25), call(10.25)]


def test_status_stops_retrying_when_time_budget_is_spent(
    client, http_request, sleep, monkeypatch
):
    monkeypatch.setattr(
        "job_submitter.infrastructure.retry.time.monotonic",
        Mock(side_effect=[0, 90]),
    )
    http_request.return_value = FakeResponse(503, text="down")

    with pytest.raises(MlSpaceApiError, match="HTTP 503 down"):
        client.job_status("job-1")

    http_request.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize("failures", [1, 2, 3])
def test_auth_network_failures_share_get_attempt_budget(
    client, http, http_request, sleep, monkeypatch, failures
):
    login = Mock(
        wraps=http.post,
        side_effect=[requests.ReadTimeout("auth timeout")] * failures + [DEFAULT],
    )
    monkeypatch.setattr(http, "post", login)

    if failures == 3:
        with pytest.raises(requests.ReadTimeout, match="auth timeout"):
            client.job_status("job-1")
        http_request.assert_not_called()
    else:
        assert client.job_status("job-1") == "Pending"
        http_request.assert_called_once()

    assert login.call_count == min(failures + 1, 3)
    assert sleep.call_args_list == [call(5), call(10)][:failures]


def test_get_recovers_when_token_refresh_times_out(
    client, http, http_request, sleep, monkeypatch
):
    login = Mock(
        wraps=http.post,
        side_effect=[DEFAULT, requests.ReadTimeout("auth timeout"), DEFAULT],
    )
    monkeypatch.setattr(http, "post", login)
    http.reject_next = True

    assert client.job_status("job-1") == "Pending"
    assert login.call_count == 3
    assert http_request.call_count == 2
    assert http_request.call_args.kwargs["headers"]["Authorization"] == "Bearer tok2"
    assert sleep.call_args_list == [call(10)]


def test_submit_does_not_retry_when_token_refresh_times_out(
    client, http, http_request, sleep, monkeypatch
):
    login = Mock(
        wraps=http.post,
        side_effect=[DEFAULT, requests.ReadTimeout("auth timeout")],
    )
    monkeypatch.setattr(http, "post", login)
    http.reject_next = True

    with pytest.raises(requests.ReadTimeout, match="auth timeout"):
        client.create_job({"script": "train.py"})

    assert login.call_count == 2
    http_request.assert_called_once()
    sleep.assert_not_called()
