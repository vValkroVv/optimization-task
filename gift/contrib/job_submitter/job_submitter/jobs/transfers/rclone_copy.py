from __future__ import annotations

from dataclasses import dataclass

from job_submitter.core.job_config import JobTemplate
from job_submitter.jobs.slurm_cpu import SlurmCpuJob
from job_submitter.jobs.transfers.rclone import build_copy_script, resolve_rclone_config


@dataclass(slots=True, kw_only=True)
class RcloneCopyJob(SlurmCpuJob):
    """Copy one file or directory between an island filesystem and S3.

    Runs ``rclone copyto`` in the Slurm CPU transfer partition. The rclone
    script and config resolution live here, so callers only describe the
    transfer itself: source, destination, and region. ``delete_on_success``
    deletes the source once it is uploaded; ``on_success_callback`` is a shell
    command this transfer job runs right after a successful upload.
    """

    source_path: str
    destination_path: str
    delete_on_success: bool = False
    on_success_callback: str | None = None
    script: str = ""
    cpus_per_task: int = 4
    memory: str | None = "8G"
    slurm_submit_type: str = "api"

    def to_config(self) -> JobTemplate:
        if not self.script:
            self.script = build_copy_script(
                self.source_path,
                self.destination_path,
                resolve_rclone_config(self.region),
                delete_on_success=self.delete_on_success,
            )
            if self.on_success_callback:
                self.script += f" && {self.on_success_callback}"
        # REST-submitted jobs have no HOME, which rclone needs for cache paths.
        self.extra_env.setdefault("HOME", f"/home/{self.user_login}")
        # Zero-arg super() breaks under @dataclass(slots=True), which rebuilds
        # the class and leaves the original in the method's __class__ cell.
        return super(RcloneCopyJob, self).to_config()
