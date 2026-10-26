from dataclasses import dataclass

from job_submitter.jobs.base import BaseJob


@dataclass(slots=True, kw_only=True)
class InfiniteJob(BaseJob):
    script: str = "sleep infinity"

    def _script(self) -> str:
        return self.script

    def _before_submit(self) -> None:
        pass
