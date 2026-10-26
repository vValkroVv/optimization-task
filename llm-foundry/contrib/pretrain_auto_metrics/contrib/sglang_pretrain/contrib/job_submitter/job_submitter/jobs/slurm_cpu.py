from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from job_submitter.core.backends import SlurmBackend, SlurmSubmittedJob
from job_submitter.core.job_config import JobTemplate
from job_submitter.core.job_templates.region_configs import (
    DEFAULT_SLURM_REGION_PARAMS,
)
from job_submitter.core.slurm_rest import (
    resolve_slurm_jwt_token,
    resolve_slurmrestd_urls,
)
from job_submitter.core.tracking import auto_track

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class SlurmCpuJob:
    """Run a host-side script in the dedicated Slurm CPU transfer partition."""

    user_login: str
    exp_name: str
    region: str
    script: str
    extra_env: dict[str, str] = field(default_factory=dict)
    cpus_per_task: int = 16
    memory: str | None = "32G"
    time_limit: str | None = None
    requeue: bool = False
    job_logdir: str | None = None
    slurm_logdir: str | None = None
    slurm_submit_type: str = "sbatch"
    slurm_jwt_token: str | None = field(default=None, repr=False)
    slurmrestd_urls: list[str] = field(default_factory=list, repr=False)
    tracking_config_path: str | None = None
    tracking_logs_path: str | None = None
    enable_auto_track: bool = True
    auto_track_users: list[str] = field(default_factory=list)
    enable_auto_share: bool = False
    tracking_metadata: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    tracking_registered: bool = field(default=False, init=False)
    _last_submitted_job: SlurmSubmittedJob | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @classmethod
    def from_spec(cls, spec_path: str | Path) -> SlurmCpuJob:
        import yaml

        with Path(spec_path).open(encoding="utf-8") as spec_file:
            values = yaml.safe_load(spec_file) or {}
        if not isinstance(values, dict):
            raise ValueError("Slurm CPU job spec must contain a mapping")
        return cls(**values)

    @property
    def submitted_job_id(self) -> str | None:
        submitted_job = self._last_submitted_job
        if not isinstance(submitted_job, SlurmSubmittedJob):
            return None
        return submitted_job.job_id

    def to_config(self) -> JobTemplate:
        region_defaults = DEFAULT_SLURM_REGION_PARAMS.get(self.region, {})

        return JobTemplate(
            submission_type="slurm",
            user_login=self.user_login,
            exp_name=self.exp_name,
            job_desc=self.exp_name,
            base_image=None,
            script=self.script,
            env_variables=dict(self.extra_env),
            region=self.region,
            gpu_num=0,
            cpu_only=True,
            priority_class=None,
            slurm_submit_type=self.slurm_submit_type,
            slurm_jwt_token=resolve_slurm_jwt_token(self.slurm_jwt_token),
            slurmrestd_urls=resolve_slurmrestd_urls(self.slurmrestd_urls),
            job_logdir=self.job_logdir or region_defaults.get("job_logdir"),
            slurm_logdir=self.slurm_logdir or region_defaults.get("slurm_logdir"),
            partition="transfer",
            cpus_per_task=self.cpus_per_task,
            memory=self.memory,
            time_limit=self.time_limit,
            requeue=self.requeue,
            comment=self.extra_env.get("TB_LOG_DIR"),
        )

    def submit(self) -> SlurmCpuJob:
        config = self.to_config()
        if self.dry_run:
            print(f"Dry run: {config}")
            self._last_submitted_job = None
            return self

        backend = SlurmBackend()
        backend.submit(config)
        self._last_submitted_job = backend.last_submitted_job
        if self.enable_auto_track and self._last_submitted_job is not None:
            self._track_submitted_job(config, self._last_submitted_job)
        return self

    def _track_submitted_job(
        self,
        config: JobTemplate,
        submitted_job: SlurmSubmittedJob,
    ) -> None:
        job_id = submitted_job.job_id
        direct_logs_path = self.tracking_logs_path
        if not self.tracking_config_path and not direct_logs_path:
            slurm_logdir = str(config.slurm_logdir or "").rstrip("/")
            if slurm_logdir:
                direct_logs_path = f"{slurm_logdir}/slurm-{job_id}.out"

        # Auto-tracking must never break a successful submission: the job already
        # runs in Slurm by the time we get here, so swallow any unexpected failure.
        try:
            self.tracking_registered = auto_track(
                region=self.region,
                user_login=self.user_login,
                config_path=self.tracking_config_path,
                env=config.env_variables,
                submission_type="slurm",
                job_id=job_id,
                auto_track_users=self.auto_track_users,
                enable_auto_share=self.enable_auto_share,
                logs_path=direct_logs_path,
                extra_metadata=self.tracking_metadata,
                job_title=config.job_desc,
            )
        except Exception:
            self.tracking_registered = False
            logger.exception(
                "Auto-tracking raised unexpectedly for Slurm job %s (non-blocking; "
                "job submission already succeeded).",
                job_id,
            )
        if not self.tracking_registered:
            logger.warning(
                "Auto-tracking failed or skipped for Slurm job %s region='%s' "
                "logs='%s'. Job submission succeeded.",
                job_id,
                self.region,
                direct_logs_path or self.tracking_config_path,
            )
