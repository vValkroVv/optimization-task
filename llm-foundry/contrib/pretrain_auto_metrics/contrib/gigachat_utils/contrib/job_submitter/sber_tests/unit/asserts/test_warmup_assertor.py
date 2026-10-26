from types import SimpleNamespace

import pytest
from job_submitter.asserts.warmup import WarmupJobAssertor


def _make_warmup_job(**overrides):
    defaults = dict(
        n_gpus=1,
        n_workers=1,
        processes_per_worker=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.unit
class TestWarmupJobAssertorSingleGpu:
    def test_valid_single_gpu(self) -> None:
        job = _make_warmup_job()
        WarmupJobAssertor()._assert_single_gpu(job)

    def test_rejects_multiple_gpus(self) -> None:
        job = _make_warmup_job(n_gpus=2)
        with pytest.raises(AssertionError):
            WarmupJobAssertor()._assert_single_gpu(job)

    def test_rejects_multiple_workers(self) -> None:
        job = _make_warmup_job(n_workers=2)
        with pytest.raises(AssertionError):
            WarmupJobAssertor()._assert_single_gpu(job)

    def test_rejects_multiple_processes_per_worker(self) -> None:
        job = _make_warmup_job(processes_per_worker=2)
        with pytest.raises(AssertionError):
            WarmupJobAssertor()._assert_single_gpu(job)
