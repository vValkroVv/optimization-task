from pathlib import Path

import pytest

from job_submitter.jobs.image_registry import PAM_BASE_IMAGES
from job_submitter.jobs.sglang_pretrain.test import TestSglangPretrainJob


def _make_job(**overrides) -> TestSglangPretrainJob:
    base = dict(
        user_login="u",
        exp_name="e",
        submission_type="cloud",
        region="SR008",
        script="pytest test/unit/",
    )
    base.update(overrides)
    return TestSglangPretrainJob(**base)


@pytest.mark.unit
class TestSglangPretrainJobShape:
    def test_base_images_is_custom_pam(self) -> None:
        assert TestSglangPretrainJob.BASE_IMAGES is PAM_BASE_IMAGES["custom"]

    def test_sglang_pretrain_root_resolves_relative_to_module(self) -> None:
        # The property points 5 parents up from the source file. Verify the
        # formula, not a particular filesystem layout -- the standalone
        # `job_submitter` CI checks out the repo without the sglang_pretrain
        # tree above it, so any assertion about siblings would be runtime-
        # specific.
        import job_submitter.jobs.sglang_pretrain.test as mod

        expected = Path(mod.__file__).resolve().parents[5]
        assert _make_job().sglang_pretrain_root == expected

    def test_external_code_libraries_returns_only_sglang_python(self) -> None:
        job = _make_job()
        libs = job._external_code_libraries()
        assert libs == [str(job.sglang_pretrain_root / "python")]

    def test_script_passthrough(self) -> None:
        job = _make_job(script="echo hi")
        assert job._script() == "echo hi"

    def test_before_submit_is_noop(self) -> None:
        assert _make_job()._before_submit() is None


@pytest.mark.unit
class TestRunNameSuffixAndJobDesc:
    @staticmethod
    def _patch_git(monkeypatch, sha: str = "abcdef0", author: str = "Tester") -> None:
        def _fake_check_output(cmd, *args, **kwargs):
            if "rev-parse" in cmd:
                return f"{sha}\n"
            if "log" in cmd:
                return f"{author}\n"
            raise AssertionError(f"unexpected git invocation: {cmd!r}")

        monkeypatch.setattr(
            "job_submitter.jobs.sglang_pretrain.test.subprocess.check_output",
            _fake_check_output,
        )

    def test_make_run_name_suffix_formats_sha_and_author(self, monkeypatch) -> None:
        self._patch_git(monkeypatch, sha="0123abc", author="Alice")
        suffix = TestSglangPretrainJob._make_run_name_suffix(Path("/irrelevant"))
        assert suffix == "0123abc by Alice"

    def test_resolve_job_desc_concatenates_exp_name_with_suffix(
        self, monkeypatch
    ) -> None:
        self._patch_git(monkeypatch, sha="deadbee", author="Bob")
        job = _make_job(exp_name="my-exp")
        assert job._resolve_job_desc() == "my-exp | deadbee by Bob"
