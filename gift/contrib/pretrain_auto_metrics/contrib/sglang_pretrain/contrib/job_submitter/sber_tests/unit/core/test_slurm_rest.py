from __future__ import annotations

from dataclasses import dataclass

import pytest
import requests

from job_submitter.core import slurm_rest
from job_submitter.core.slurm_rest import (
    SlurmRestClient,
    SlurmRESTError,
    has_slurm_rest_config,
    resolve_slurm_jwt_token,
    resolve_slurmrestd_api_version,
    resolve_slurmrestd_urls,
)

pytestmark = pytest.mark.unit


@dataclass
class FakeResponse:
    status_code: int = 200
    data: object = None
    text: str = "ok"

    def json(self) -> object:
        if isinstance(self.data, Exception):
            raise self.data
        return self.data if self.data is not None else {}


@pytest.fixture(autouse=True)
def clear_slurm_rest_environment(monkeypatch) -> None:
    for name in (
        "SLURM_JWT_TOKEN",
        "SLURM_JWT",
        "SLURMRESTD_URLS",
        "TRACKER_BOT_SLURM_API_BASE",
        "SLURMRESTD_API_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)


def test_configuration_resolves_aliases_and_supported_versions(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JWT", "alias-token")
    monkeypatch.setenv("TRACKER_BOT_SLURM_API_BASE", "http://alias:6820/")
    monkeypatch.setenv("SLURMRESTD_API_VERSION", "0.0.39")

    client = SlurmRestClient()

    assert resolve_slurm_jwt_token() == "alias-token"
    assert resolve_slurmrestd_urls() == ["http://alias:6820"]
    assert resolve_slurmrestd_api_version() == "v0.0.39"
    assert has_slurm_rest_config() is True
    assert client.urls == ("http://alias:6820",)
    assert client.jwt_token == "alias-token"
    assert client.api_version == "v0.0.39"


def test_explicit_configuration_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JWT_TOKEN", "env-token")
    monkeypatch.setenv("SLURMRESTD_URLS", "http://env:6820")

    client = SlurmRestClient(
        urls=["http://first:6820/", "", "http://second:6820"],
        jwt_token="explicit-token",
        api_version="v0.0.40",
        timeout_seconds=5,
    )

    assert resolve_slurm_jwt_token("explicit-token") == "explicit-token"
    assert resolve_slurmrestd_urls(["http://first:6820/"]) == ["http://first:6820"]
    assert client.urls == ("http://first:6820", "http://second:6820")
    assert client.jwt_token == "explicit-token"
    assert client.api_version == "v0.0.40"
    assert client.timeout_seconds == 5


def test_configuration_reports_missing_urls_and_token(monkeypatch) -> None:
    assert resolve_slurmrestd_urls() == []
    assert has_slurm_rest_config() is False

    with pytest.raises(SlurmRESTError, match="URLs"):
        SlurmRestClient(jwt_token="token")

    monkeypatch.setenv("SLURMRESTD_URLS", "http://slurm:6820")
    with pytest.raises(SlurmRESTError, match="JWT"):
        SlurmRestClient()


def test_configuration_rejects_unsupported_version() -> None:
    with pytest.raises(SlurmRESTError, match="supported: v0.0.39, v0.0.40"):
        resolve_slurmrestd_api_version("v0.0.42")


def test_submit_uses_auth_payload_version_and_url_failover(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    responses = iter(
        [
            FakeResponse(status_code=503, text="offline"),
            FakeResponse(data={"job_id": 123}),
        ]
    )

    def post(url: str, **kwargs: object) -> FakeResponse:
        calls.append((url, kwargs))
        return next(responses)

    monkeypatch.setattr(slurm_rest.requests, "post", post)
    client = SlurmRestClient(
        urls=["http://first:6820", "http://second:6820"],
        jwt_token="secret",
        api_version="v0.0.39",
    )

    job_id = client.submit({"script": "echo ok"})

    assert job_id == "123"
    assert [call[0] for call in calls] == [
        "http://first:6820/slurm/v0.0.39/job/submit",
        "http://second:6820/slurm/v0.0.39/job/submit",
    ]
    assert calls[1][1]["headers"] == {
        "Content-Type": "application/json",
        "X-SLURM-USER-TOKEN": "secret",
    }
    assert calls[1][1]["json"] == {"script": "echo ok"}
    assert calls[1][1]["timeout"] == 30.0


def test_submit_rejects_response_without_job_id(monkeypatch) -> None:
    monkeypatch.setattr(
        slurm_rest.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(data={}),
    )

    with pytest.raises(SlurmRESTError, match="Unexpected Slurm submit response"):
        SlurmRestClient(
            urls=["http://slurm:6820"],
            jwt_token="token",
        ).submit({"script": "echo ok"})


def test_get_job_returns_exact_job_and_reports_absence(monkeypatch) -> None:
    responses = iter(
        [
            FakeResponse(
                data={
                    "jobs": [
                        "invalid",
                        {"job_id": 122, "name": "other"},
                        {"job_id": 123, "name": "target"},
                    ]
                }
            ),
            FakeResponse(data={"jobs": [{"job_id": 124, "name": "other"}]}),
        ]
    )
    monkeypatch.setattr(
        slurm_rest.requests,
        "get",
        lambda *args, **kwargs: next(responses),
    )
    client = SlurmRestClient(urls=["http://slurm:6820"], jwt_token="token")

    assert client.get_job("123") == {"job_id": 123, "name": "target"}
    assert client.get_job("123") is None

    with pytest.raises(ValueError, match="Invalid Slurm job ID"):
        client.get_job("../unsafe")


def test_find_job_returns_latest_valid_matching_job(monkeypatch) -> None:
    responses = iter(
        [
            FakeResponse(
                data={
                    "jobs": [
                        {"job_id": "bad", "name": "target"},
                        {"job_id": 123, "name": "other"},
                        {"job_id": "123_1", "name": "target"},
                        {"job_id": 124, "name": "target"},
                    ]
                }
            ),
            FakeResponse(data={}),
        ]
    )
    monkeypatch.setattr(
        slurm_rest.requests,
        "get",
        lambda *args, **kwargs: next(responses),
    )
    client = SlurmRestClient(urls=["http://slurm:6820"], jwt_token="token")

    assert client.find_job("target") == "124"
    assert client.find_job("target") is None


def test_find_job_prefers_active_job_over_newer_finished_one(monkeypatch) -> None:
    monkeypatch.setattr(
        slurm_rest.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            data={
                "jobs": [
                    {"job_id": 120, "name": "target", "job_state": ["RUNNING"]},
                    {"job_id": 121, "name": "target", "job_state": ["PENDING"]},
                    {"job_id": 124, "name": "target", "job_state": ["COMPLETED"]},
                    {"job_id": 125, "name": "target", "job_state": ["FAILED"]},
                ]
            }
        ),
    )
    client = SlurmRestClient(urls=["http://slurm:6820"], jwt_token="token")

    assert client.find_job("target") == "121"


def test_find_job_prefers_active_job_with_dict_shaped_job_state(monkeypatch) -> None:
    monkeypatch.setattr(
        slurm_rest.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            data={
                "jobs": [
                    {"job_id": 120, "name": "target", "job_state": {"name": "RUNNING"}},
                    {
                        "job_id": 121,
                        "name": "target",
                        "job_state": {"state": "PENDING"},
                    },
                    {"job_id": 124, "name": "target", "job_state": {}},
                    {"job_id": 125, "name": "target", "job_state": {"state": "FAILED"}},
                ]
            }
        ),
    )
    client = SlurmRestClient(urls=["http://slurm:6820"], jwt_token="token")

    assert client.find_job("target") == "121"


def test_list_jobs_returns_jobs(monkeypatch) -> None:
    jobs = [{"job_id": 123, "name": "target"}, "invalid"]
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: FakeResponse(data={"jobs": jobs}),
    )
    client = SlurmRestClient(urls=["http://slurm:6820"], jwt_token="token")

    assert client.list_jobs() == [jobs[0]]


def test_cancel_accepts_empty_success_response(monkeypatch) -> None:
    calls: list[str] = []

    def delete(url: str, **kwargs: object) -> FakeResponse:
        calls.append(url)
        return FakeResponse(status_code=204, text="")

    monkeypatch.setattr(slurm_rest.requests, "delete", delete)
    client = SlurmRestClient(urls=["http://slurm:6820"], jwt_token="token")

    assert client.cancel("123_1") is None
    assert calls == ["http://slurm:6820/slurm/v0.0.40/job/123_1"]

    with pytest.raises(ValueError, match="Invalid Slurm job ID"):
        client.cancel("unsafe")


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(data=[]),
        FakeResponse(data={"errors": ["permission denied"]}),
        FakeResponse(data=ValueError("invalid json")),
        requests.ConnectionError("offline"),
    ],
)
def test_request_errors_include_failed_url(monkeypatch, response) -> None:
    def get(*args, **kwargs):
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(slurm_rest.requests, "get", get)
    client = SlurmRestClient(urls=["http://slurm:6820"], jwt_token="token")

    with pytest.raises(
        SlurmRESTError,
        match=r"http://slurm:6820/slurm/v0.0.40/jobs",
    ):
        client.find_job("target")


def test_find_job_rejects_malformed_jobs_collection(monkeypatch) -> None:
    monkeypatch.setattr(
        slurm_rest.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(data={"jobs": "invalid"}),
    )

    with pytest.raises(SlurmRESTError, match="Unexpected Slurm jobs response"):
        SlurmRestClient(
            urls=["http://slurm:6820"],
            jwt_token="token",
        ).find_job("target")


def test_request_rejects_unsupported_http_method() -> None:
    client = SlurmRestClient(urls=["http://slurm:6820"], jwt_token="token")

    with pytest.raises(ValueError, match="Unsupported HTTP method"):
        client._request("PATCH", "jobs")


def test_failover_success_logs_a_visible_warning(monkeypatch, caplog) -> None:
    import logging

    responses = iter(
        [
            FakeResponse(status_code=503, text="offline"),
            FakeResponse(data={"job_id": 7}),
        ]
    )
    monkeypatch.setattr(
        slurm_rest.requests, "post", lambda url, **kwargs: next(responses)
    )
    client = SlurmRestClient(
        urls=["http://primary:6820", "http://backup:6820"],
        jwt_token="secret",
        api_version="v0.0.40",
    )

    with caplog.at_level(logging.WARNING, logger="job_submitter.core.slurm_rest"):
        assert client.submit({"script": "echo ok"}) == "7"

    failover_records = [
        record for record in caplog.records if "slurmrestd failover" in record.message
    ]
    assert len(failover_records) == 1
    message = failover_records[0].getMessage()
    assert "http://backup:6820" in message
    assert "http://primary:6820" in message


def test_primary_success_logs_no_failover_warning(monkeypatch, caplog) -> None:
    import logging

    monkeypatch.setattr(
        slurm_rest.requests,
        "post",
        lambda url, **kwargs: FakeResponse(data={"job_id": 8}),
    )
    client = SlurmRestClient(
        urls=["http://primary:6820", "http://backup:6820"],
        jwt_token="secret",
        api_version="v0.0.40",
    )

    with caplog.at_level(logging.WARNING, logger="job_submitter.core.slurm_rest"):
        assert client.submit({"script": "echo ok"}) == "8"

    assert not [r for r in caplog.records if "slurmrestd failover" in r.message]
