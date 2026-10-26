from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from job_submitter.jobs.autometrics import AutometricsJob


@dataclass(slots=True, kw_only=True)
class AutometricsJobGIFT(AutometricsJob):
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

    @property
    def node_runner_path(self) -> Path:
        return self.pam_root / "node_runner" / "node_runner.py"

    def _script(self) -> str:
        node_runner_root = shlex.quote(str(self.node_runner_path.parent))
        checkpoint_dir = shlex.quote(self.checkpoint_dir)
        mode = shlex.quote(self.mode)
        command = f"python3 node_runner.py --checkpoint {checkpoint_dir} --mode {mode}"
        if self.submission_type == "slurm":
            return f"cd {node_runner_root} && {command}"
        return f"--wdir {node_runner_root} {command}"

    def _external_code_libraries(self) -> list[str]:
        libraries = [
            str(self.gift_root),
            str(self.pam_root / "contrib" / "gigafsdp"),
            str(self.pam_root / "contrib" / "gigachat_utils" / "convertation"),
            str(self.pam_root / "contrib" / "lm_evaluation_harness"),
        ]
        if self.sglang_tag == "custom":
            libraries.append(
                str(self.pam_root / "contrib" / "sglang_pretrain" / "python")
            )
        return libraries

    def _before_submit(self) -> None:
        if not self.node_runner_path.is_file():
            raise ValueError(
                "PAM node_runner was not found. Initialize "
                "contrib/pretrain_auto_metrics recursively."
            )

        checkpoint_dir = Path(self.checkpoint_dir)
        # Raw DCP checkpoints are converted to HF inside the PAM job itself.
        if not (checkpoint_dir / ".metadata").is_file():
            if not (checkpoint_dir / "config.json").is_file():
                raise ValueError(
                    "GIFT autometrics requires a complete Hugging Face checkpoint "
                    "containing config.json."
                )
            if not any(checkpoint_dir.glob("*.safetensors")):
                raise ValueError(
                    "GIFT autometrics requires a Hugging Face checkpoint containing "
                    "safetensors weights."
                )

        AutometricsJob._before_submit(self)
