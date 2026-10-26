from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from job_submitter.asserts.gift_warmup import GIFTWarmupJobAssertor
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import GIFT_IMAGES_PATH, load_gift_base_images


@dataclass(slots=True, kw_only=True)
class WarmupJobTorchTitan(BaseJob, GIFTWarmupJobAssertor):
    """Runs one kernel-cache warmup command on a single GPU.

    The command and the label are built by ``torchtitan/ops/warmup/plan.py`` and
    the JIT environment is set by the warmup entrypoint itself; this wrapper only
    adds the GIFT base image, the code libraries and the single-GPU resource
    shape. One pass per job lets the cluster run them in parallel.

    ``gift_root_path`` must be the same checkout the command runs from, since it
    is what lands on ``PYTHONPATH``.
    """

    script: str
    job_label: str = "warmup"

    # Warmup is always single-GPU, single-worker.
    n_gpus: int = 1
    n_workers: int = 1
    processes_per_worker: int = 1
    script_type: str = "binary"
    gift_root_path: str | Path | None = None

    @property
    def gift_root(self) -> Path:
        if self.gift_root_path is not None:
            return Path(self.gift_root_path).expanduser().resolve()
        if gift_root := os.getenv("GIFT_ROOT_PATH"):
            return Path(gift_root).expanduser().resolve()
        return Path(__file__).resolve().parents[5]

    def _resolve_job_desc(self) -> str:
        return f"{self.user_login} | {self.exp_name} | {self.job_label}"

    def _script(self) -> str:
        return self.script

    def _external_code_libraries(self) -> list[str]:
        return [
            str(self.gift_root),
            str(self.gift_root / "contrib" / "streaming"),
        ]

    def _resolve_base_image(self) -> str:
        if self.base_image:
            return self.base_image
        images = load_gift_base_images(self.gift_root)
        if self.region not in images:
            raise ValueError(
                f"No torchtitan base image pinned for region '{self.region}' in "
                f"{self.gift_root / GIFT_IMAGES_PATH}; available: {sorted(images)}"
            )
        return images[self.region]
