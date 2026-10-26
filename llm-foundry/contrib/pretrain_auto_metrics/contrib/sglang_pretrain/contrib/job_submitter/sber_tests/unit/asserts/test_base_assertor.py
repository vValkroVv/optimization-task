import pytest
from job_submitter.asserts.base import BaseJobAssertor


@pytest.mark.unit
class TestBaseJobAssertorFormatting:
    def test_pretty_params_print_formats_values(self) -> None:
        assertor = BaseJobAssertor()
        text = assertor._pretty_params_print(
            {"region": ["SR004", "SR008"], "n_workers": [1, 2]}
        )
        assert "region=" in text
        assert '"SR004"/"SR008"' in text
        assert "n_workers=1/2" in text

    def test_is_encoded_in_base64_handles_text_and_bytes(self) -> None:
        assert BaseJobAssertor._is_encoded_in_base64("aGVsbG8=") is True
        assert BaseJobAssertor._is_encoded_in_base64(b"aGVsbG8=") is True
        assert BaseJobAssertor._is_encoded_in_base64("not-base64!") is False


@pytest.mark.unit
class TestBaseJobAssertorValidation:
    def test_assert_job_rejects_invalid_submission_type(
        self, dummy_job, cloud_template
    ) -> None:
        cloud_template.submission_type = "invalid"
        with pytest.raises(AssertionError):
            BaseJobAssertor().assert_job(dummy_job, cloud_template)

    def test_assert_job_valid_cloud_passes(self, dummy_job, cloud_template) -> None:
        BaseJobAssertor().assert_job(dummy_job, cloud_template)
