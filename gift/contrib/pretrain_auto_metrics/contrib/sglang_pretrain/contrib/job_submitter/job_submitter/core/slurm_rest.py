from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable

import requests

logger = logging.getLogger(__name__)

DEFAULT_SLURMRESTD_API_VERSION = "v0.0.40"
SUPPORTED_SLURMRESTD_API_VERSIONS = {"v0.0.39", "v0.0.40"}
SLURMRESTD_API_VERSION_ENV = "SLURMRESTD_API_VERSION"
_JOB_ID_PATTERN = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
_ACTIVE_JOB_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}


class SlurmRESTError(RuntimeError):
    pass


def resolve_slurm_jwt_token(value: str | None = None) -> str | None:
    return value or os.getenv("SLURM_JWT_TOKEN") or os.getenv("SLURM_JWT")


def resolve_slurmrestd_urls(values: list[str] | None = None) -> list[str]:
    if values:
        return [value.rstrip("/") for value in values if value.strip()]
    raw_urls = os.getenv("SLURMRESTD_URLS") or os.getenv("TRACKER_BOT_SLURM_API_BASE")
    if not raw_urls:
        return []
    return [
        value.strip().rstrip("/") for value in raw_urls.split("||") if value.strip()
    ]


def resolve_slurmrestd_api_version(value: str | None = None) -> str:
    version = value or os.getenv(
        SLURMRESTD_API_VERSION_ENV,
        DEFAULT_SLURMRESTD_API_VERSION,
    )
    if not version.startswith("v"):
        version = f"v{version}"
    if version not in SUPPORTED_SLURMRESTD_API_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_SLURMRESTD_API_VERSIONS))
        raise SlurmRESTError(
            f"Unsupported Slurm REST API version {version!r}; supported: {supported}"
        )
    return version


def has_slurm_rest_config() -> bool:
    return bool(resolve_slurm_jwt_token() and resolve_slurmrestd_urls())


class SlurmRestClient:
    def __init__(
        self,
        *,
        urls: list[str] | None = None,
        jwt_token: str | None = None,
        api_version: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        resolved_urls = resolve_slurmrestd_urls(urls)
        resolved_token = resolve_slurm_jwt_token(jwt_token)
        if not resolved_urls:
            raise SlurmRESTError("No slurmrestd URLs provided")
        if not resolved_token:
            raise SlurmRESTError("No Slurm JWT token provided")
        self.urls = tuple(resolved_urls)
        self.jwt_token = resolved_token
        self.api_version = resolve_slurmrestd_api_version(api_version)
        self.timeout_seconds = timeout_seconds

    def submit(self, payload: dict[str, object]) -> str:
        data = self._request("POST", "job/submit", payload=payload)
        job_id = data.get("job_id")
        if job_id is None:
            raise SlurmRESTError(f"Unexpected Slurm submit response: {data}")
        return str(job_id)

    def get_job(self, job_id: str) -> dict[str, object] | None:
        self._validate_job_id(job_id)
        data = self._request("GET", f"job/{job_id}")
        for job in self._jobs(data):
            if str(job.get("job_id")) == job_id:
                return job
        return None

    def find_job(self, job_name: str) -> str | None:
        data = self._request("GET", "jobs")
        matching = [
            job
            for job in self._jobs(data)
            if job.get("name") == job_name
            and job.get("job_id") is not None
            and _JOB_ID_PATTERN.fullmatch(str(job["job_id"])) is not None
        ]
        if not matching:
            return None
        # A completed attempt may carry a newer job ID than the attempt that
        # is still running, so active jobs win over recency.
        active_ids = [
            str(job["job_id"])
            for job in matching
            if self._job_state(job) in _ACTIVE_JOB_STATES
        ]
        candidate_ids = active_ids or [str(job["job_id"]) for job in matching]
        return max(candidate_ids, key=self._job_id_sort_key)

    def list_jobs(self) -> list[dict[str, object]]:
        return self._jobs(self._request("GET", "jobs"))

    def cancel(self, job_id: str) -> None:
        self._validate_job_id(job_id)
        self._request("DELETE", f"job/{job_id}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        request: Callable[..., requests.Response]
        if method == "POST":
            request = requests.post
        elif method == "GET":
            request = requests.get
        elif method == "DELETE":
            request = requests.delete
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        errors: list[str] = []
        headers = {
            "Content-Type": "application/json",
            "X-SLURM-USER-TOKEN": self.jwt_token,
        }
        for base_url in self.urls:
            url = f"{base_url}/slurm/{self.api_version}/{path}"
            try:
                kwargs: dict[str, object] = {
                    "headers": headers,
                    "timeout": self.timeout_seconds,
                }
                if payload is not None:
                    kwargs["json"] = payload
                response = request(url, **kwargs)
                if not 200 <= response.status_code < 300:
                    raise SlurmRESTError(f"{response.status_code}: {response.text}")
                if errors:
                    # Observability: a request served by a fallback URL means
                    # the primary slurmrestd is down — make the switch loud
                    # instead of silently degrading.
                    logger.warning(
                        "slurmrestd failover: request served by %s after %s",
                        base_url,
                        "; ".join(errors),
                    )
                if response.status_code == 204 or not response.text.strip():
                    return {}
                data = response.json()
                if not isinstance(data, dict):
                    raise SlurmRESTError(f"Unexpected response from {url}: {data!r}")
                response_errors = data.get("errors")
                if response_errors:
                    raise SlurmRESTError(f"Slurm REST errors: {response_errors}")
                return data
            except (requests.RequestException, SlurmRESTError, ValueError) as exc:
                errors.append(f"{url}: {exc}")
        raise SlurmRESTError(
            "Slurm REST request failed for all configured URLs: " + "; ".join(errors)
        )

    @staticmethod
    def _job_state(job: dict[str, object]) -> str:
        value = job.get("job_state") if "job_state" in job else job.get("state")
        if isinstance(value, list):
            value = value[0] if value else ""
        if isinstance(value, dict):
            value = value.get("state") or value.get("name") or ""
        parts = str(value or "").strip().split(maxsplit=1)
        return parts[0].rstrip("+").upper() if parts else ""

    @staticmethod
    def _jobs(data: dict[str, object]) -> list[dict[str, object]]:
        jobs = data.get("jobs")
        if jobs is None:
            return []
        if not isinstance(jobs, list):
            raise SlurmRESTError(f"Unexpected Slurm jobs response: {data}")
        return [job for job in jobs if isinstance(job, dict)]

    @staticmethod
    def _job_id_sort_key(job_id: str) -> tuple[int, ...]:
        return tuple(int(part) for part in job_id.split("_"))

    @staticmethod
    def _validate_job_id(job_id: str) -> None:
        if _JOB_ID_PATTERN.fullmatch(job_id) is None:
            raise ValueError(f"Invalid Slurm job ID: {job_id}")
