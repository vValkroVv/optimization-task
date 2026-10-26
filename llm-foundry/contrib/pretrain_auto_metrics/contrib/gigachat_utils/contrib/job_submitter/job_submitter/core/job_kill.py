from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from importlib import import_module

from job_submitter.core.cloud_api import CloudApi
from job_submitter.core.runtime_context import detect_runtime_context
from job_submitter.core.slurm_rest import SlurmRestClient, has_slurm_rest_config
from job_submitter.infrastructure.mlspace import MlSpaceClient


class KillBackend(ABC):
    @abstractmethod
    def kill(self, job_id: str, region: str) -> str: ...


class CloudKillBackend(KillBackend):
    def __init__(self) -> None:
        self.cloud_api = CloudApi.from_env()
        self.client_lib = (
            None
            if self.cloud_api is CloudApi.PUBLIC
            else import_module("job_submitter.c_lib.client_lib")
        )

    def kill(self, job_id: str, region: str) -> str:
        if self.cloud_api is CloudApi.PUBLIC:
            return MlSpaceClient.from_env(region).delete_job(job_id)
        return str(self.client_lib.kill(job_id, region=region))


class SlurmKillBackend(KillBackend):
    def kill(self, job_id: str, region: str) -> str:
        if has_slurm_rest_config():
            SlurmRestClient().cancel(job_id)
            return f'Slurm job "{job_id}" cancelled in region "{region}"'
        try:
            result = subprocess.run(
                ["scancel", job_id],
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Failed to kill slurm job: `scancel` not found") from exc

        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(
                f'Failed to kill slurm job "{job_id}" in region "{region}": {details}'
            )

        return f'Slurm job "{job_id}" cancelled in region "{region}"'


def get_kill_backend(submission_type: str) -> KillBackend:
    match submission_type:
        case "cloud":
            return CloudKillBackend()
        case "slurm":
            return SlurmKillBackend()
        case _:
            raise ValueError(f"Unsupported submission_type={submission_type!r}")


def kill_job(job_id: str) -> str:
    job_id = job_id.strip()
    if not job_id:
        raise ValueError("Job id must not be empty")

    resolved_context = detect_runtime_context()
    backend = get_kill_backend(resolved_context.submission_type)
    return backend.kill(job_id=job_id, region=resolved_context.region)
