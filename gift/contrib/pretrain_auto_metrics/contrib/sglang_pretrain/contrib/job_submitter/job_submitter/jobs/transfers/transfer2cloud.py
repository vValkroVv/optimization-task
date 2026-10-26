from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from job_submitter.core.slurm_rest import SlurmRESTError, has_slurm_rest_config
from job_submitter.infrastructure.retry import RetryPolicy
from job_submitter.jobs.transfers.rclone import ISLAND_REGIONS
from job_submitter.jobs.transfers.rclone_copy import RcloneCopyJob

logger = logging.getLogger(__name__)

_CLOUD_REMOTE_ROOT = "papayaeye-papaya:transfer2cloud"
_SUBMIT_ATTEMPTS = 3
_FIRST_RETRY_DELAY_SECONDS = 10 * 60


@dataclass(slots=True, kw_only=True)
class Transfer2CloudJob:
    """Ship one staged artifact from an island to the external papaya S3.

    ``staging_path`` must live under ``<root>/transfer2cloud/<region>/``; the
    source keeps its path relative to ``<root>`` on the remote side. A
    slurmrestd outage is retried with a growing delay.
    """

    source_path: Path
    staging_path: Path
    exp_name: str

    # Callers ship one archive per producer from a single blocked process, so
    # once waiting has proven useless every later transfer gets one attempt
    # until one succeeds again.
    _retries_exhausted: ClassVar[bool] = False

    def submit(self) -> str:
        if not has_slurm_rest_config():
            raise SlurmRESTError("Slurm REST is not configured")

        transfer_root, region = self._location()
        relative_path = self.source_path.relative_to(transfer_root).as_posix()
        destination_path = f"{_CLOUD_REMOTE_ROOT}/{relative_path}"
        user_login = os.getenv("SLURM_JOB_USER") or os.getenv("USER")
        if not user_login:
            raise RuntimeError("Cannot determine Slurm user")

        job = RcloneCopyJob(
            user_login=user_login,
            exp_name=self.exp_name,
            region=region,
            source_path=self.source_path.as_posix(),
            destination_path=destination_path,
        )
        job_id = self._submit_with_retries(job)
        logger.info(
            "Submitted transfer job %s: %s -> %s",
            job_id,
            self.source_path,
            destination_path,
        )
        return job_id

    def _submit_with_retries(self, job: RcloneCopyJob) -> str:
        policy = RetryPolicy(
            exceptions=(SlurmRESTError,),
            attempts=1 if Transfer2CloudJob._retries_exhausted else _SUBMIT_ATTEMPTS,
            delay=lambda attempt, error: _FIRST_RETRY_DELAY_SECONDS
            * 2 ** (attempt - 1),
        )
        try:
            policy.run(job.submit)
        except SlurmRESTError:
            Transfer2CloudJob._retries_exhausted = True
            raise
        Transfer2CloudJob._retries_exhausted = False
        job_id = job.submitted_job_id
        if job_id is None:
            raise RuntimeError("Slurm did not return a job ID for the transfer")
        return job_id

    def _location(self) -> tuple[Path, str]:
        transfer_root = self.staging_path.parent.parent
        region = self.staging_path.parent.name
        if transfer_root.name != "transfer2cloud" or region not in ISLAND_REGIONS:
            raise ValueError(f"Unexpected transfer staging path: {self.staging_path}")
        return transfer_root, region
