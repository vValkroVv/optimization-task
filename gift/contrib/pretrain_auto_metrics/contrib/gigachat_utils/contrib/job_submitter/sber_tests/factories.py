from dataclasses import dataclass

from job_submitter.jobs.base import BaseJob


@dataclass(slots=True, kw_only=True)
class DummyJob(BaseJob):
    script_value: str = "echo test"

    def _script(self) -> str:
        return self.script_value

    def _before_submit(self) -> None:
        pass
