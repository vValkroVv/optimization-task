import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, ClassVar, Optional

import requests

from job_submitter.core.job_config import JobTemplate
from job_submitter.core.job_templates.region_configs import (
    DEFAULT_JOB_DISPATCHER_BASE_URL as REGION_DEFAULT_JOB_DISPATCHER_BASE_URL,
)
from job_submitter.core.job_templates.region_configs import (
    DEFAULT_JOB_DISPATCHER_TIMEOUT_SECONDS as REGION_DEFAULT_JOB_DISPATCHER_TIMEOUT_SECONDS,
)
from job_submitter.core.job_templates.region_configs import (
    JOB_DISPATCHER_BASE_URL_ENV,
    JOB_DISPATCHER_TIMEOUT_SECONDS_ENV,
)
from job_submitter.core.cloud_api import CloudApi
from job_submitter.core.job_templates.slurm_client_lib import submit_slurm_job
from job_submitter.core.slurm_rest import SlurmRestClient
from job_submitter.infrastructure.mlspace import MlSpaceClient

logger = logging.getLogger(__name__)


class JobBackend(ABC):
    @abstractmethod
    def submit(self, cfg: JobTemplate) -> None:
        pass


@dataclass(slots=True)
class SlurmSubmittedJob:
    job_id: str

    @property
    def job_name(self) -> str:
        return self.job_id

    def status(self) -> str:
        return "Unknown"

    def logs(self, tail: int = 0, verbose: bool = True) -> str:
        return ""


@dataclass(slots=True, frozen=True)
class SlurmJobStatus:
    state: str
    exit_code: str | None = None


@dataclass(slots=True)
class SlurmCpuSubmittedJob(SlurmSubmittedJob):
    _JOB_ID_PATTERN = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
    _COMMAND_TIMEOUT_SECONDS: ClassVar[float] = 30.0
    slurm_submit_type: str = "sbatch"
    slurm_jwt_token: str | None = field(default=None, repr=False)
    slurmrestd_urls: list[str] = field(default_factory=list, repr=False)

    def status(self) -> str:
        return self.status_details().state

    def status_details(self) -> SlurmJobStatus:
        if self._JOB_ID_PATTERN.fullmatch(self.job_id) is None:
            raise ValueError(f"Invalid Slurm job ID: {self.job_id}")
        if self.slurm_submit_type.lower() == "api":
            return self._status_details_rest()
        queued = self._run(
            ["squeue", "--noheader", f"--jobs={self.job_id}", "--format=%T"]
        )
        queue_state = self._normalize_state(queued.stdout)
        if queued.returncode == 0 and queue_state:
            return SlurmJobStatus(queue_state)

        accounted = self._run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                f"--jobs={self.job_id}",
                "--format=JobIDRaw,State,ExitCode",
            ]
        )
        if accounted.returncode != 0:
            details = (
                accounted.stderr.strip() or accounted.stdout.strip() or "unknown error"
            )
            raise RuntimeError(f"Failed to inspect Slurm job {self.job_id}: {details}")
        for line in accounted.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 3 or parts[0] != self.job_id:
                continue
            state = self._normalize_state(parts[1])
            exit_code = parts[2].strip() or None
            if state == "COMPLETED" and exit_code not in {None, "0:0"}:
                state = "FAILED"
            return SlurmJobStatus(state or "UNKNOWN", exit_code)
        return SlurmJobStatus("UNKNOWN")

    @classmethod
    def find_active_job(
        cls,
        job_name: str,
        *,
        slurm_submit_type: str = "sbatch",
        slurm_jwt_token: str | None = None,
        slurmrestd_urls: list[str] | None = None,
    ) -> str | None:
        if slurm_submit_type.lower() == "api":
            return SlurmRestClient(
                urls=slurmrestd_urls,
                jwt_token=slurm_jwt_token,
            ).find_job(job_name)
        result = cls._run(
            ["squeue", "--noheader", f"--name={job_name}", "--format=%A|%j"]
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"Failed to inspect active Slurm jobs: {details}")
        for line in result.stdout.splitlines():
            job_id, separator, current_name = line.strip().partition("|")
            if (
                separator
                and current_name == job_name
                and cls._JOB_ID_PATTERN.fullmatch(job_id) is not None
            ):
                return job_id
        return None

    @classmethod
    def find_job(
        cls,
        job_name: str,
        *,
        slurm_submit_type: str = "sbatch",
        slurm_jwt_token: str | None = None,
        slurmrestd_urls: list[str] | None = None,
    ) -> str | None:
        if slurm_submit_type.lower() == "api":
            return SlurmRestClient(
                urls=slurmrestd_urls,
                jwt_token=slurm_jwt_token,
            ).find_job(job_name)
        active_job_id = cls.find_active_job(job_name)
        if active_job_id is not None:
            return active_job_id
        result = cls._run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                f"--name={job_name}",
                "--starttime=now-7days",
                "--format=JobIDRaw,JobName",
            ]
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"Failed to inspect Slurm job history: {details}")
        matching_job_ids = []
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if (
                len(parts) >= 2
                and parts[1] == job_name
                and cls._JOB_ID_PATTERN.fullmatch(parts[0]) is not None
            ):
                matching_job_ids.append(parts[0])
        return matching_job_ids[-1] if matching_job_ids else None

    def cancel(self) -> None:
        if self._JOB_ID_PATTERN.fullmatch(self.job_id) is None:
            raise ValueError(f"Invalid Slurm job ID: {self.job_id}")
        if self.slurm_submit_type.lower() != "api":
            raise RuntimeError("REST cancellation requires slurm_submit_type='api'")
        self._rest_client().cancel(self.job_id)

    def _status_details_rest(self) -> SlurmJobStatus:
        job = self._rest_client().get_job(self.job_id)
        if job is None:
            return SlurmJobStatus("UNKNOWN")
        state = self._normalize_rest_state(
            job.get("job_state") if "job_state" in job else job.get("state")
        )
        exit_code = self._normalize_rest_exit_code(job.get("exit_code"))
        if state == "COMPLETED" and exit_code not in {None, "0:0"}:
            state = "FAILED"
        return SlurmJobStatus(state or "UNKNOWN", exit_code)

    def _rest_client(self) -> SlurmRestClient:
        return SlurmRestClient(
            urls=self.slurmrestd_urls,
            jwt_token=self.slurm_jwt_token,
        )

    @staticmethod
    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=SlurmCpuSubmittedJob._COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Cannot run {command[0]}: {exc}") from exc

    @staticmethod
    def _normalize_state(value: str) -> str:
        parts = value.strip().split(maxsplit=1)
        return parts[0].rstrip("+").upper() if parts else ""

    @classmethod
    def _normalize_rest_state(cls, value: object) -> str:
        if isinstance(value, list):
            value = value[0] if value else ""
        if isinstance(value, dict):
            value = value.get("state") or value.get("name") or ""
        return cls._normalize_state(str(value or ""))

    @staticmethod
    def _normalize_rest_exit_code(value: object) -> str | None:
        if value is None or value == "":
            return None
        if isinstance(value, int):
            return f"{value}:0"
        if isinstance(value, dict):
            # slurmrestd v0.0.40 (Slurm 23.11) exit_code shape:
            #   {"status": ["SUCCESS"],
            #    "return_code": {"set": true, "infinite": false, "number": 0},
            #    "signal": {"id": {"set": false, ...}, "name": ""}}
            # Older payloads used plain ints. Normalize both to "rc:signal" so
            # the COMPLETED-with-nonzero-exit demotion only fires on real
            # failures.

            def _number(raw: object) -> int | None:
                if isinstance(raw, dict):
                    if "id" in raw:
                        return _number(raw["id"])
                    if "set" in raw:
                        if not raw.get("set"):
                            return None
                        raw = raw.get("number")
                if isinstance(raw, bool):
                    return None
                if isinstance(raw, (int, float)):
                    return int(raw)
                return None

            status = value.get("status")
            if isinstance(status, str):
                statuses: list[object] = [status]
            elif isinstance(status, list):
                statuses = status
            else:
                statuses = []
            status_names = {str(item).upper() for item in statuses if item is not None}
            return_code = _number(value.get("return_code"))
            if return_code is None:
                return_code = _number(status)
            signal = _number(value.get("signal")) or 0
            if return_code is None and "SUCCESS" in status_names:
                return_code = 0
            if return_code is None:
                if status_names:
                    return f"{','.join(sorted(status_names))}:{signal}"
                return None
            return f"{return_code}:{signal}"
        return str(value)


@dataclass(slots=True, frozen=True)
class DispatcherSettings:
    base_url: str
    timeout_seconds: float


class CloudBackend(JobBackend):
    DEFAULT_JOB_DISPATCHER_URL = REGION_DEFAULT_JOB_DISPATCHER_BASE_URL
    DEFAULT_JOB_DISPATCHER_TIMEOUT_SECONDS = (
        REGION_DEFAULT_JOB_DISPATCHER_TIMEOUT_SECONDS
    )

    def __init__(self) -> None:
        self.cloud_api = CloudApi.from_env()
        self.client_lib = (
            None
            if self.cloud_api is CloudApi.PUBLIC
            else import_module("job_submitter.c_lib.client_lib")
        )
        self.last_submitted_job: Optional[Any] = None

    @staticmethod
    def _resolve_dispatcher_timeout_seconds(env_variables: dict[str, str]) -> float:
        raw_timeout = (
            env_variables.get(JOB_DISPATCHER_TIMEOUT_SECONDS_ENV)
            or os.getenv(JOB_DISPATCHER_TIMEOUT_SECONDS_ENV)
            or str(CloudBackend.DEFAULT_JOB_DISPATCHER_TIMEOUT_SECONDS)
        )
        try:
            timeout = float(raw_timeout)
            if timeout <= 0:
                raise ValueError("timeout must be positive")
            return timeout
        except (TypeError, ValueError):
            return CloudBackend.DEFAULT_JOB_DISPATCHER_TIMEOUT_SECONDS

    def submit(self, cfg: JobTemplate) -> None:
        print(f"Submitting cloud job: {cfg}")

        env_variables = cfg.env_variables or {}
        job_params = self._build_job_params(cfg, env_variables)
        if self.cloud_api is CloudApi.PUBLIC:
            self._submit_via_public_api(cfg.region, job_params)
            return
        job = self.client_lib.Job(**job_params)

        dispatched_job_name = None
        if env_variables.get("MLS_NTB_WORKSPACE_NAME", "").strip().lower() != "tts":
            dispatched_job_name = self._submit_via_dispatcher(job_params, env_variables)
        if dispatched_job_name:
            job.job_name = dispatched_job_name
            self.last_submitted_job = job
            print(f"Cloud job submitted: {job.job_name}")
            return

        submit_message = job.submit()
        if not getattr(job, "job_name", None):
            print(f"Cloud job submission failed: {submit_message}")
            self.last_submitted_job = None
            return
        print(f"Cloud job submitted: {submit_message}")
        print(f"Cloud job id: {job.job_name}", flush=True)
        self.last_submitted_job = job

    def _submit_via_public_api(self, region: str, job_params: dict[str, Any]) -> None:
        job = MlSpaceClient.from_env(region).submit_job(job_params)
        print(f"Cloud job id: {job.job_name}", flush=True)
        self.last_submitted_job = job

    def _build_job_params(
        self,
        cfg: JobTemplate,
        env_variables: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "job_desc": cfg.job_desc,
            "base_image": cfg.base_image,
            "script": cfg.script,
            "n_workers": cfg.n_workers,
            "processes_per_worker": cfg.processes_per_worker,
            "instance_type": cfg.instance_type,
            "type": cfg.script_type,
            "env_variables": env_variables,
            "region": cfg.region,
            "allocation_name": cfg.allocation_name,
            "max_retry": cfg.max_retry,
            "priority_class": cfg.priority_class,
            "flags": cfg.flags or {},
            "pytorch_use_env": True,
            "internet": cfg.internet,
            "queue_name": cfg.queue_name,
        }

    @staticmethod
    def _dispatcher_payload(job_params: dict[str, Any]) -> dict[str, Any]:
        # Nullable fields are still required by the job-dispatcher schema.
        return dict(job_params)

    def _resolve_dispatcher_settings(
        self,
        env_variables: dict[str, str],
    ) -> DispatcherSettings:
        dispatcher_base_url = (
            env_variables.get(JOB_DISPATCHER_BASE_URL_ENV)
            or os.getenv(JOB_DISPATCHER_BASE_URL_ENV)
            or self.DEFAULT_JOB_DISPATCHER_URL
        ).strip()
        return DispatcherSettings(
            base_url=dispatcher_base_url.rstrip("/"),
            timeout_seconds=self._resolve_dispatcher_timeout_seconds(env_variables),
        )

    def _submit_via_dispatcher(
        self,
        job_params: dict[str, Any],
        env_variables: dict[str, str],
    ) -> Optional[str]:
        settings = self._resolve_dispatcher_settings(env_variables)
        try:
            resp = requests.post(
                f"{settings.base_url}/jobs/create-job",
                json=self._dispatcher_payload(job_params),
                timeout=(5, settings.timeout_seconds),
            )
            if not 200 <= resp.status_code < 300:
                logger.warning(
                    "job-dispatcher returned status=%s, falling back to client_lib: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return None

            try:
                return str(resp.json()["job_name"])
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "job-dispatcher returned invalid payload, "
                    "falling back to client_lib: %s",
                    exc,
                )
                return None

        except (requests.RequestException, ConnectionError) as exc:
            logger.warning(
                "job-dispatcher unavailable, falling back to client_lib: %s",
                exc,
            )
            return None


class SlurmBackend(JobBackend):
    def __init__(self) -> None:
        self.last_submitted_job: Optional[SlurmSubmittedJob] = None

    def submit(self, cfg: JobTemplate) -> None:
        print(f"Submitting slurm job: {cfg.job_desc}, nodes={cfg.n_workers}")
        submitted_job_id = submit_slurm_job(
            slurm_submit_type=cfg.slurm_submit_type,
            job_desc=cfg.job_desc,
            base_image=cfg.base_image,
            n_workers=cfg.n_workers,
            gpu_num=cfg.gpu_num,
            cpu_only=cfg.cpu_only,
            gpu_type=cfg.gpu_type,
            type=cfg.script_type,
            processes_per_worker=cfg.processes_per_worker,
            region=cfg.region,
            script=cfg.script,
            env_variables=cfg.env_variables,
            job_logdir=cfg.job_logdir,
            slurm_logdir=cfg.slurm_logdir,
            slurmrestd_urls=cfg.slurmrestd_urls,
            slurm_jwt_token=cfg.slurm_jwt_token,
            max_restarts=cfg.max_retry,
            nodelist=cfg.nodelist,
            exclude_nodes=cfg.exclude_nodes,
            nice=cfg.nice,
            partition=cfg.partition,
            cpus_per_task=cfg.cpus_per_task,
            memory=cfg.memory,
            time_limit=cfg.time_limit,
            requeue=cfg.requeue,
            comment=cfg.comment,
            extra_mounts=cfg.extra_mounts,
            turn_off_first_srun=cfg.turn_off_first_srun,
            qos=cfg.qos,
            account=cfg.account,
        )
        if submitted_job_id:
            if cfg.cpu_only:
                self.last_submitted_job = SlurmCpuSubmittedJob(
                    job_id=str(submitted_job_id),
                    slurm_submit_type=cfg.slurm_submit_type,
                    slurm_jwt_token=cfg.slurm_jwt_token,
                    slurmrestd_urls=cfg.slurmrestd_urls,
                )
            else:
                self.last_submitted_job = SlurmSubmittedJob(
                    job_id=str(submitted_job_id)
                )
        else:
            self.last_submitted_job = None


def submit_job_cfg(cfg: JobTemplate) -> None:
    if cfg.submission_type == "cloud":
        backend: JobBackend = CloudBackend()
    elif cfg.submission_type == "slurm":
        backend = SlurmBackend()
    else:
        raise ValueError(f"Unknown submission_type={cfg.submission_type!r}")
    backend.submit(cfg)
