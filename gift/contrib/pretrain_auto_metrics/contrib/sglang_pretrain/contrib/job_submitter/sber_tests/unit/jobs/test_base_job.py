from pathlib import Path

import pytest

import job_submitter.jobs.base as base_module
from job_submitter.core.job_config import JobTemplate
from sber_tests.factories import DummyJob


@pytest.mark.unit
class TestBaseJobStatics:
    def test_get_instance_type_by_region(self) -> None:
        assert DummyJob._get_instance_type("SR004", 8) == "a100.8gpu"
        assert DummyJob._get_instance_type("SR006", 2) == "a100.2gpu.16C.486G"
        assert DummyJob._get_instance_type("SR008", 4) == "a100plus.4gpu.80vG.48C.728G"

    def test_get_instance_type_raises_for_unknown_region(self) -> None:
        with pytest.raises(ValueError):
            DummyJob._get_instance_type("UNKNOWN", 8)

    def test_parse_helpers(self) -> None:
        raw = (
            "ClusterName = my_cluster\nSlurmctldHost[0]=host1\nSlurmctldHost[1]=host2\n"
        )
        urls = DummyJob._parse_url(raw)
        assert urls == ["http://host1:6820", "http://host2:6820"]
        assert DummyJob._parse_cluster_name(raw) == "mycluster"

    def test_llm_f_root_returns_parent_workspace(self) -> None:
        job = DummyJob(user_login="u", exp_name="e")
        expected = (
            Path(base_module.__file__).absolute().parent.parent.parent.parent.parent
        )
        assert job.llm_f_root == expected


@pytest.mark.unit
class TestBaseJobResolvers:
    def test_resolve_queue_name_priority(self, monkeypatch) -> None:
        job = DummyJob(
            user_login="u", exp_name="e", region="SR008", submission_type="cloud"
        )
        assert job._resolve_queue_name() == "shared"

        job.queue_name = None
        monkeypatch.setenv("PAM_QUEUE_NAME", "giga-prod")
        assert job._resolve_queue_name() == "giga-prod"

    def test_resolve_priority_class_for_shared_queue(self, monkeypatch) -> None:
        job = DummyJob(
            user_login="u",
            exp_name="e",
            region="SR008",
            submission_type="cloud",
            queue_name="shared",
        )
        monkeypatch.delenv("PAM_PRIORITY_CLASS", raising=False)
        assert job._resolve_priority_class() is None

    def test_to_config_builds_job_template(self, monkeypatch) -> None:
        job = DummyJob(
            user_login="u", exp_name="e", region="SR008", submission_type="cloud"
        )
        monkeypatch.setattr(job, "_get_current_region", lambda: ("SR008", "cloud"))

        cfg = job.to_config()
        assert isinstance(cfg, JobTemplate)
        assert cfg.job_desc == "u | e"
        assert cfg.script == "echo test"
        assert cfg.region == "SR008"
