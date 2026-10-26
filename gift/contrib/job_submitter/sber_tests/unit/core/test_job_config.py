from dataclasses import fields

import pytest

from job_submitter.core.job_config import JobTemplate


@pytest.mark.unit
class TestJobTemplate:
    def test_defaults_are_stable(self) -> None:
        cfg = JobTemplate()
        assert cfg.submission_type == "cloud"
        assert cfg.region == "SR008"
        assert cfg.script_type == "binary"
        assert cfg.env_variables == {}
        assert cfg.flags == {}

    def test_has_expected_field_names(self) -> None:
        field_names = {f.name for f in fields(JobTemplate)}
        assert "submission_type" in field_names
        assert "slurm_submit_type" in field_names
        assert "extra_mounts" in field_names
