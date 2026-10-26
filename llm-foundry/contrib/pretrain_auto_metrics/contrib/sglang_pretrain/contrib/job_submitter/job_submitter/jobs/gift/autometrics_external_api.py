from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from job_submitter.jobs.pam.autometrics_external_api import (
    AutometricsJobPAMExternalAPI,
)


@dataclass(slots=True, kw_only=True)
class AutometricsJobGIFTExternalAPI(AutometricsJobPAMExternalAPI):
    gift_root_path: str | Path | None = None

    @property
    def gift_root(self) -> Path:
        if self.gift_root_path is not None:
            return Path(self.gift_root_path).expanduser().resolve()
        if gift_root := os.getenv("GIFT_ROOT_PATH"):
            return Path(gift_root).expanduser().resolve()
        return Path(__file__).resolve().parents[5]

    @property
    def pam_root(self) -> Path:
        return self.gift_root / "contrib" / "pretrain_auto_metrics"

    def _before_submit(self) -> None:
        node_runner_path = self.pam_root / "node_runner" / "node_runner.py"
        if not node_runner_path.is_file():
            raise ValueError(
                "PAM node_runner was not found. Initialize "
                "contrib/pretrain_auto_metrics recursively."
            )
