from __future__ import annotations

from typing import TYPE_CHECKING

from job_submitter.asserts.base import BaseJobAssertor
from job_submitter.core.job_config import JobTemplate

if TYPE_CHECKING:
    from job_submitter.jobs.gift.warmup import WarmupJobTorchTitan


class GIFTWarmupJobAssertor(BaseJobAssertor):
    def assert_job(self, job: WarmupJobTorchTitan, job_template: JobTemplate) -> None:
        super().assert_job(job, job_template)
        assert job.script.strip(), self._pretty_params_print(
            {"script": ["python .../deep_gemm_warmup.py --config ... --gemm ..."]}
        )
        assert job.n_gpus == 1, self._pretty_params_print({"n_gpus": [1]})
        assert job.n_workers == 1, self._pretty_params_print({"n_workers": [1]})
        assert job.processes_per_worker == 1, self._pretty_params_print(
            {"processes_per_worker": [1]}
        )
        assert job.gift_root.is_dir(), self._pretty_params_print(
            {"gift_root_path": ["/path/to/torchtitan"]}
        )
        assert str(job.gift_root) in job.script, (
            f"script must run the warmup entrypoint from gift_root "
            f"({job.gift_root}), which is the checkout this job puts on "
            f"PYTHONPATH; got: {job.script}"
        )
