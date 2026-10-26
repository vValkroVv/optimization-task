from __future__ import annotations

from typing import Any, Optional

import requests

from job_submitter.infrastructure.mlspace.auth import ServiceAuth
from job_submitter.infrastructure.mlspace.config import MlSpaceConfig
from job_submitter.infrastructure.mlspace.exceptions import (
    MlSpaceApiError,
    MlSpaceJobNotFoundError,
)
from job_submitter.infrastructure.mlspace.job_handle import MlSpaceJobHandle
from job_submitter.infrastructure.retry import RetryPolicy

REQUEST_TIMEOUT_SECONDS = 60
STATUS_TIMEOUT_SECONDS = 30
LOGS_TIMEOUT_SECONDS = 300
MAX_REQUEST_ATTEMPTS = 3
FIRST_RETRY_DELAY_SECONDS = 5
RETRY_JITTER_SECONDS = 1
RETRYABLE_STATUS_CODES = {502, 503, 504}
JOB_DELETED = "deleted"
JOB_NOT_FOUND = "job not found"


class _RetryableResponseError(Exception):
    def __init__(self, response: requests.Response) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def _retry_delay(attempt: int, error: Exception) -> float:
    if isinstance(error, _RetryableResponseError) and error.response.status_code == 401:
        return 0
    return FIRST_RETRY_DELAY_SECONDS * 2 ** (attempt - 1)


class MlSpaceClient:
    def __init__(
        self,
        config: MlSpaceConfig,
        auth: ServiceAuth,
        region: str,
        session: requests.Session,
    ) -> None:
        self._config = config
        self._auth = auth
        self._region = region
        self._session = session

    @classmethod
    def from_env(
        cls, region: str, session: Optional[requests.Session] = None
    ) -> MlSpaceClient:
        config = MlSpaceConfig.from_env()
        session = session or requests.Session()
        return cls(config, ServiceAuth(config, session), region, session)

    def submit_job(self, params: dict[str, Any]) -> MlSpaceJobHandle:
        return MlSpaceJobHandle(self, self.create_job(params))

    def create_job(self, params: dict[str, Any]) -> str:
        body = {key: value for key, value in params.items() if value is not None}
        payload = self._ok(
            self._request("POST", "/jobs", json=body), "job creation"
        ).json()
        if not payload.get("job_name"):
            raise MlSpaceApiError(f"job creation returned no job_name: {payload}")
        return payload["job_name"]

    def job_status(self, job_name: str) -> str:
        response = self._request(
            "GET",
            f"/jobs/{job_name}",
            params={"region": self._region},
            timeout=STATUS_TIMEOUT_SECONDS,
        )
        payload = self._ok(response, "job status").json()
        return payload.get("status") or payload.get("job_status") or str(payload)

    def job_logs(self, job_name: str, tail: int, verbose: bool) -> str:
        params = {"region": self._region, "tail": tail, "verbose": verbose}
        response = self._request(
            "GET", f"/jobs/{job_name}/logs", params=params, timeout=LOGS_TIMEOUT_SECONDS
        )
        if response.status_code != 200:
            return f"<logs unavailable: HTTP {response.status_code} {response.text[:200]}>\n"
        return response.text

    def delete_job(self, job_name: str) -> str:
        response = self._request(
            "DELETE",
            f"/jobs/{job_name}",
            params={"region": self._region},
            timeout=STATUS_TIMEOUT_SECONDS,
        )
        payload = self._ok(response, "job deletion").json()
        status = payload.get("status")
        if status == JOB_DELETED:
            return f'Job "{job_name}" deleted'
        if status == JOB_NOT_FOUND:
            raise MlSpaceJobNotFoundError(
                f'job "{job_name}" not found in region {self._region}'
            )
        raise MlSpaceApiError(f"job deletion returned unexpected response: {payload}")

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
        url = f"{self._config.base_url}{path}"
        auth_refreshed = False
        retry_exceptions: tuple[type[Exception], ...] = (_RetryableResponseError,)
        if method == "GET":
            retry_exceptions = (
                _RetryableResponseError,
                requests.ConnectionError,
                requests.Timeout,
            )

        policy = RetryPolicy(
            exceptions=retry_exceptions,
            attempts=MAX_REQUEST_ATTEMPTS if method == "GET" else 2,
            delay=_retry_delay,
            should_retry=lambda error: not isinstance(
                error, requests.exceptions.SSLError
            ),
            jitter_seconds=RETRY_JITTER_SECONDS,
            timeout_seconds=MAX_REQUEST_ATTEMPTS * kwargs["timeout"],
        )

        def send() -> requests.Response:
            nonlocal auth_refreshed
            response = self._session.request(
                method, url, headers=self._auth.headers(self._region), **kwargs
            )
            retryable_response = (
                method == "GET" and response.status_code in RETRYABLE_STATUS_CODES
            )
            if response.status_code == 401 and not auth_refreshed:
                self._auth.reset()
                auth_refreshed = True
                retryable_response = True
            if retryable_response:
                raise _RetryableResponseError(response)
            return response

        try:
            return policy.run(send)
        except _RetryableResponseError as error:
            return error.response

    @staticmethod
    def _ok(response: requests.Response, what: str) -> requests.Response:
        if response.status_code >= 300:
            raise MlSpaceApiError(
                f"{what} failed: HTTP {response.status_code} {response.text[:300]}"
            )
        return response
