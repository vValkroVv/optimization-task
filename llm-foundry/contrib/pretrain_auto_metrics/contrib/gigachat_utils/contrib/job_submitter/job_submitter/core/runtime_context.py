from __future__ import annotations

from dataclasses import dataclass

from job_submitter.jobs.base import BaseJob


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    region: str
    submission_type: str


@dataclass(slots=True, kw_only=True)
class _RuntimeContextProbeJob(BaseJob):
    def _script(self) -> str:
        return "echo runtime-context-probe"


def detect_runtime_context() -> RuntimeContext:
    probe = _RuntimeContextProbeJob(
        user_login="job_submitter_cli",
        exp_name="runtime_context_probe",
    )
    try:
        region, submission_type = probe._get_current_region()
    except Exception as exc:
        raise RuntimeError(
            "Failed to detect runtime context from BaseJob logic. "
            "Set `SLURM_CLUSTER_NAME` for slurm or `CLUSTER_KEY` for cloud."
        ) from exc
    if not region or not submission_type:
        raise RuntimeError(
            "Failed to detect runtime context: region/submission_type are empty"
        )
    return RuntimeContext(region=region, submission_type=submission_type)
