from pathlib import Path
from types import SimpleNamespace

import pytest

from job_submitter.asserts.gift_warmup import GIFTWarmupJobAssertor
from job_submitter.core.job_config import JobTemplate


def _make_warmup_job(tmp_path: Path, **overrides):
    defaults = dict(
        user_login="u",
        exp_name="e",
        queue_name="shared",
        script=(
            f"python {tmp_path}/torchtitan/ops/warmup/deep_gemm_warmup.py"
            " --config w.yaml --gemm w1_m_grouped"
        ),
        n_gpus=1,
        n_workers=1,
        processes_per_worker=1,
        gift_root=tmp_path,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_template() -> JobTemplate:
    return JobTemplate(
        submission_type="cloud",
        user_login="u",
        exp_name="e",
        job_desc="u | e | deep_gemm_w1_m_grouped",
        base_image="cr.ai.cloud.ru/x/y:z",
        script="python deep_gemm_warmup.py",
        region="SR008",
        n_workers=1,
    )


@pytest.mark.unit
class TestGIFTWarmupJobAssertor:
    def test_accepts_a_single_gpu_warmup_job(self, tmp_path: Path) -> None:
        GIFTWarmupJobAssertor().assert_job(_make_warmup_job(tmp_path), _make_template())

    @pytest.mark.parametrize(
        "overrides, expected",
        [
            ({"script": "   "}, "script"),
            ({"n_gpus": 2}, "n_gpus"),
            ({"n_workers": 2}, "n_workers"),
            ({"processes_per_worker": 2}, "processes_per_worker"),
        ],
    )
    def test_rejects_invalid_params(
        self, tmp_path: Path, overrides: dict, expected: str
    ) -> None:
        job = _make_warmup_job(tmp_path, **overrides)
        with pytest.raises(AssertionError, match=expected):
            GIFTWarmupJobAssertor().assert_job(job, _make_template())

    def test_rejects_a_gift_root_that_is_not_a_directory(self, tmp_path: Path) -> None:
        job = _make_warmup_job(tmp_path, gift_root=tmp_path / "missing")
        with pytest.raises(AssertionError, match="gift_root_path"):
            GIFTWarmupJobAssertor().assert_job(job, _make_template())

    def test_rejects_a_script_from_another_checkout(self, tmp_path: Path) -> None:
        # The job would run this tree with tmp_path on PYTHONPATH.
        job = _make_warmup_job(
            tmp_path,
            script="python /other/checkout/torchtitan/ops/warmup/deep_gemm_warmup.py",
        )
        with pytest.raises(AssertionError, match="must run the warmup entrypoint"):
            GIFTWarmupJobAssertor().assert_job(job, _make_template())
