from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from job_submitter.asserts.base import BaseJobAssertor
from job_submitter.core.job_config import JobTemplate

if TYPE_CHECKING:
    from job_submitter.jobs.gift.train import TrainJobGIFT

# Torch-free pre-submit check shipped with the checkout; CI lint runs the same one.
CONFIG_CHECK_SCRIPT = Path("scripts") / "config_check.py"
CONFIG_CHECK_TIMEOUT_SECONDS = 300

logger = logging.getLogger(__name__)


class GIFTTrainJobAssertor(BaseJobAssertor):
    @staticmethod
    def _run_config_check(
        job: TrainJobGIFT, script_args: list[str]
    ) -> subprocess.CompletedProcess[str]:
        if job.script_type == "pytorch2":
            script_args = [*script_args, f"--world-size={job.n_workers * job.n_gpus}"]
        return subprocess.run(
            [
                sys.executable,
                str(job.gift_root / CONFIG_CHECK_SCRIPT),
                str(job.config_file_path),
                *script_args,
            ],
            cwd=job.gift_root,
            capture_output=True,
            text=True,
            timeout=CONFIG_CHECK_TIMEOUT_SECONDS,
            check=False,
        )

    def assert_job(self, job: TrainJobGIFT, job_template: JobTemplate) -> None:
        super().assert_job(job, job_template)
        assert str(job.config_file).strip(), self._pretty_params_print(
            {"config_file": ["configs/experiments/gift_moe_debug.yaml"]},
            "TrainJobGIFT",
        )
        try:
            script_args = shlex.split(job.script_args)
        except ValueError as exc:
            raise AssertionError(f"Invalid script_args: {exc}") from exc
        assert job.gift_root.is_dir(), self._pretty_params_print(
            {"gift_root_path": ["/path/to/torchtitan"]}
        )
        assert job.entrypoint_path.is_file(), self._pretty_params_print(
            {"gift_root_path": ["/path/to/torchtitan containing gift_scripts/train.py"]}
        )
        if job_template.submission_type == "cloud":
            assert job_template.processes_per_worker == job.n_gpus, (
                self._pretty_params_print(
                    {"processes_per_worker": [job.n_gpus]},
                    "TrainJobGIFT",
                )
            )
        assert job.config_file_path.is_file(), self._pretty_params_print(
            {"config_file": ["configs/experiments/gift_moe_debug.yaml"]}
        )
        assert (job.gift_root / "torchtitan" / "train.py").is_file(), (
            self._pretty_params_print(
                {
                    "gift_root_path": [
                        "/path/to/torchtitan containing torchtitan/train.py"
                    ]
                }
            )
        )
        assert (job.gift_root / CONFIG_CHECK_SCRIPT).is_file(), (
            self._pretty_params_print(
                {
                    "gift_root_path": [
                        f"/path/to/torchtitan containing {CONFIG_CHECK_SCRIPT}"
                    ]
                }
            )
        )
        result = self._run_config_check(job, script_args)
        assert result.returncode == 0, (
            f"config-check rejected {job.config_file} with script_args "
            f"{job.script_args!r}:\n{result.stdout}{result.stderr}"
        )
        if result.stderr.strip():
            logger.warning("%s", result.stderr.rstrip())
