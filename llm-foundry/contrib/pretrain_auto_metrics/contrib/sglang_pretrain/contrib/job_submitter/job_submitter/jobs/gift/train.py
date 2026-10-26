from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from job_submitter.asserts.gift_train import GIFTTrainJobAssertor
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import GIFT_IMAGES_PATH, load_gift_base_images


@dataclass(slots=True, kw_only=True)
class TrainJobGIFT(BaseJob, GIFTTrainJobAssertor):
    config_file: str | Path
    script_type: str = "pytorch2"
    script_args: str = ""
    processes_per_worker: int | None = None
    gift_root_path: str | Path | None = None

    @property
    def gift_root(self) -> Path:
        if self.gift_root_path is not None:
            return Path(self.gift_root_path).expanduser().resolve()
        if gift_root := os.getenv("GIFT_ROOT_PATH"):
            return Path(gift_root).expanduser().resolve()
        return Path(__file__).resolve().parents[5]

    @property
    def entrypoint_path(self) -> Path:
        return self.gift_root / "gift_scripts" / "train.py"

    @property
    def config_file_path(self) -> Path:
        path = Path(str(self.config_file).strip()).expanduser()
        return path if path.is_absolute() else self.gift_root / path

    def _script(self) -> str:
        config_file = Path(str(self.config_file).strip()).expanduser()
        # The entrypoint chdirs to the checkout root, so a relative YAML path
        # resolves there on the cluster as it does here.
        command = (
            f"{shlex.quote(str(self.entrypoint_path))} {shlex.quote(str(config_file))}"
        )
        if script_args := self.script_args.strip():
            command = f"{command} {script_args}"
        return command

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

    def _resolve_processes_per_worker(self) -> int:
        if self.submission_type == "slurm":
            return 1
        if self.processes_per_worker is None:
            return self.n_gpus
        return self.processes_per_worker

    def _build_env(self) -> dict[str, str]:
        env = BaseJob._build_env(self)
        env.setdefault("LOG_RANK", "0")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        return env
