from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from job_submitter.asserts.base import BaseJobAssertor
from job_submitter.core.job_config import JobTemplate
from job_submitter.jobs.verl.autometrics import AutometricsJobVERL

if TYPE_CHECKING:
    from job_submitter.jobs.verl.train import RayTrainJobVERL


class VerlTrainJobAssertor(BaseJobAssertor):
    def _assert_pam_config_path(self, job: RayTrainJobVERL) -> None:
        if not job.pam_config_path:
            return
        path_to_config = Path(job.pam_config_path)
        assert path_to_config.exists(), self._pretty_params_print(
            {
                "path_to_config": ["/some/valid/existing/pam/config.yaml"],
            }
        )
        job = AutometricsJobVERL(
            user_login=job.user_login,
            exp_name="mock_exp",
            n_workers=job.pam_n_workers,
            slurm_submit_type="api",
            path_to_config=str(job.pam_config_path),
            checkpoint_dir=".",
        )
        job_config = job.to_config()
        job.assert_job(job, job_config, source_job="RayTrainJobVERL")

    def assert_job(self, job: RayTrainJobVERL, job_template: JobTemplate) -> None:
        super().assert_job(job, job_template)
        script_path = Path(job.script_path)
        assert script_path.exists(), self._pretty_params_print(
            {
                "script_path": ["/some/valid/existing/script.sh"],
            }
        )
        runtime_env_config_path = Path(job.runtime_env_config_path)
        assert runtime_env_config_path.exists(), self._pretty_params_print(
            {
                "runtime_env_config_path": [
                    "/some/valid/existing/yaml/runtime/env/config.yaml"
                ],
            }
        )
        self._assert_pam_config_path(job)
