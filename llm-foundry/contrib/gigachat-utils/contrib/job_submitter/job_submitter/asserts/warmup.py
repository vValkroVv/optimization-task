from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from job_submitter.asserts.base import BaseJobAssertor
from job_submitter.core.job_config import JobTemplate

if TYPE_CHECKING:
    from job_submitter.jobs.llmf.warmup import WarmupJobLLMF


class WarmupJobAssertor(BaseJobAssertor):
    def _assert_warmup_type(self, warmup_type: str) -> None:
        assert warmup_type in ("triton", "deep_gemm"), self._pretty_params_print(
            {"warmup_type": ["triton", "deep_gemm"]}
        )

    def _assert_warmup_config(self, warmup_config_path: str) -> None:
        path = Path(warmup_config_path)
        assert path.exists() and path.is_file(), self._pretty_params_print(
            {"warmup_config_path": ["/valid/path/to/warmup_config.yaml"]}
        )

    def _assert_cache_dir(self, cache_dir: str) -> None:
        assert cache_dir, self._pretty_params_print(
            {"cache_dir": ["<non-empty cache directory path>"]}
        )

    def _assert_single_gpu(self, job: WarmupJobLLMF) -> None:
        assert job.n_gpus == 1, self._pretty_params_print({"n_gpus": [1]})
        assert job.n_workers == 1, self._pretty_params_print({"n_workers": [1]})
        assert job.processes_per_worker == 1, self._pretty_params_print(
            {"processes_per_worker": [1]}
        )

    def assert_job(self, job: WarmupJobLLMF, job_template: JobTemplate) -> None:
        super().assert_job(job, job_template)
        self._assert_warmup_type(job.warmup_type)
        self._assert_warmup_config(str(job.warmup_config_path))
        self._assert_cache_dir(job.cache_dir)
        self._assert_single_gpu(job)
