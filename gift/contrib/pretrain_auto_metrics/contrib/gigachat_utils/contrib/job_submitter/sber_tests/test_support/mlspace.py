from __future__ import annotations

from typing import Any

import pytest

API_URL = "https://api.test/public/v2"
CREDENTIALS = {
    "MLS_CLIENT_ID": "id",
    "MLS_CLIENT_SECRET": "secret",
    "MLS_API_KEY": "key",
    "MLS_WORKSPACE_ID": "ws",
    "MLS_API_URL": API_URL,
}


def set_mlspace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in CREDENTIALS.items():
        monkeypatch.setenv(name, value)


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or ("" if payload is None else str(payload))

    def json(self) -> Any:
        return self._payload


class FakeHttp:
    def __init__(self, statuses: tuple[str, ...] = ("Pending",)) -> None:
        self.statuses = list(statuses)
        self.calls: list[tuple[str, str, dict[str, str], dict[str, Any]]] = []
        self.logins = 0
        self.login_response: FakeResponse | None = None
        self.reject_next = False
        self.response: FakeResponse | None = None

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.logins += 1
        assert url == f"{API_URL}/service_auth"
        assert kwargs["json"] == {"client_id": "id", "client_secret": "secret"}
        if self.login_response is not None:
            return self.login_response
        return FakeResponse(
            200, {"token": {"access_token": f"tok{self.logins}", "expires_in": 3600}}
        )

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        headers = kwargs.pop("headers")
        self.calls.append((method, url, headers, kwargs))
        if self.reject_next:
            self.reject_next = False
            return FakeResponse(401, {"error_code": 22})
        if self.response is not None:
            return self.response
        if method == "POST" and url.endswith("/jobs"):
            return FakeResponse(200, {"job_name": "lm-mpi-job-1"})
        if method == "GET" and url.endswith("/logs"):
            return FakeResponse(200, text="line1\nline2\n")
        if method == "GET":
            status = (
                self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            )
            return FakeResponse(200, {"job_name": "lm-mpi-job-1", "status": status})
        if method == "DELETE":
            return FakeResponse(200, {"job_name": "lm-mpi-job-1", "status": "deleted"})
        raise AssertionError(f"unexpected {method} {url}")
