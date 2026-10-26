from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from job_submitter.infrastructure.mlspace.client import MlSpaceClient

STATUS_NAMES = {
    "inqueue": "Pending",
    "starting": "Pending",
    "pending": "Pending",
    "running": "Running",
    "completing": "Running",
    "completed": "Completed",
    "failed": "Failed",
    "stopped": "Stopped",
}


class MlSpaceJobHandle:
    def __init__(self, client: MlSpaceClient, job_name: str) -> None:
        self._client = client
        self._job_name = job_name

    @property
    def job_name(self) -> str:
        return self._job_name

    def status(self) -> str:
        raw = self._client.job_status(self._job_name)
        return f"Job status={STATUS_NAMES.get(raw.lower(), raw)}"

    def logs(self, tail: int = 0, verbose: bool = True) -> str:
        text = self._client.job_logs(self._job_name, tail, verbose)
        sys.stdout.write(text)
        return text
