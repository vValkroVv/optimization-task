import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from omegaconf import OmegaConf

import job_submitter.jobs

from job_submitter.cli import JobType, main, submit_command
from job_submitter.jobs.any_script import AnyScriptJob
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.llmf.train import TrainJobLLMF


pytestmark = pytest.mark.unit


def _write_spec(tmp_path, mapping=None):
    import yaml

    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(mapping or {"user_login": "u", "exp_name": "e"}))
    return str(p)


class _FakeJob:
    def __init__(self, *, job_name=None, raise_on_submit=None):
        self._job_name = job_name
        self._raise = raise_on_submit
        self._last_submitted_job = (
            SimpleNamespace(job_name=job_name) if job_name is not None else None
        )

    def submit(self):
        if self._raise is not None:
            raise self._raise


class TestSubmitCommand:
    def test_submit_emits_pipeline_job_id_marker(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        spec = _write_spec(
            tmp_path, {"user_login": "u", "exp_name": "e", "region": "o1"}
        )
        fake = _FakeJob(job_name="ce1f770d5b")
        monkeypatch.setattr(
            TrainJobLLMF,
            "from_spec",
            SimpleNamespace(from_spec=lambda _path: fake).from_spec,
        )

        rc = main(["submit", spec, "--job-type", "llmf.train"])
        captured = capsys.readouterr()

        assert rc == 0
        assert "PIPELINE_JOB_ID=ce1f770d5b" in captured.out

    def test_dry_run_emits_no_marker(self, tmp_path, monkeypatch, capsys) -> None:
        spec = _write_spec(tmp_path)
        fake = _FakeJob(job_name=None)
        monkeypatch.setattr(
            TrainJobLLMF,
            "from_spec",
            SimpleNamespace(from_spec=lambda _path: fake).from_spec,
        )

        rc = main(["submit", spec, "--job-type", "llmf.train"])
        captured = capsys.readouterr()

        assert rc == 0
        assert "PIPELINE_JOB_ID" not in captured.out

    def test_submit_returns_non_zero_on_error(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        spec = _write_spec(tmp_path)
        fake = _FakeJob(raise_on_submit=RuntimeError("boom"))
        monkeypatch.setattr(
            TrainJobLLMF,
            "from_spec",
            SimpleNamespace(from_spec=lambda _path: fake).from_spec,
        )

        rc = main(["submit", spec, "--job-type", "llmf.train"])
        captured = capsys.readouterr()

        assert rc == 1
        assert "boom" in captured.err

    def test_from_spec_failure_surfaces_as_cli_error(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        spec = _write_spec(tmp_path)

        def _raise(_path):
            raise ValueError("Unknown job spec fields: ['bogus']")

        monkeypatch.setattr(
            TrainJobLLMF, "from_spec", SimpleNamespace(from_spec=_raise).from_spec
        )

        rc = main(["submit", spec, "--job-type", "llmf.train"])
        captured = capsys.readouterr()

        assert rc == 1
        assert "bogus" in captured.err

    def test_missing_spec_argument_is_usage_error(self, capsys) -> None:
        rc = main(["submit"])
        captured = capsys.readouterr()

        assert rc == 2
        assert "Missing" in captured.err


class TestJobType:
    def test_job_type_is_required(self) -> None:
        import inspect

        assert (
            inspect.signature(submit_command).parameters["job_type"].default
            is inspect.Parameter.empty
        )
        assert main(["submit", "spec.yaml"]) == 2

    @pytest.mark.parametrize("job_type", list(JobType))
    def test_every_job_type_is_mapped(self, job_type) -> None:
        job_class = job_type.job_class
        assert callable(job_class.from_spec)
        assert callable(job_class.submit)
        if job_type is not JobType.slurm_cpu:
            assert issubclass(job_class, BaseJob)

    @pytest.mark.parametrize("job_type", list(JobType))
    def test_value_matches_the_class_module(self, job_type) -> None:
        assert job_type.job_class.__module__ == f"job_submitter.jobs.{job_type.value}"

    @pytest.mark.parametrize("job_type", list(JobType))
    def test_class_works_as_a_spec_schema(self, job_type) -> None:
        OmegaConf.structured(job_type.job_class)

    def test_job_type_covers_every_job_class(self) -> None:
        not_offered = {
            "AutometricsJob": "abstract base",
            "RayBaseJobVERL": "abstract base",
            "TestJobLLMF": "takes an argparse.Namespace, not expressible in a YAML spec",
            "TestJobPAM": "smoke-test job, kept out of the CLI",
            "TestSglangPretrainJob": "smoke-test job, kept out of the CLI",
            "TestConvertationJobGigachatUtils": "smoke-test job, kept out of the CLI",
            "WarmupJobTorchTitan": "warmup cache job",
        }
        root = Path(job_submitter.jobs.__file__).parent
        declared = {}
        for path in sorted(root.rglob("*.py")):
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.ClassDef):
                    declared[node.name] = [
                        b.id for b in node.bases if isinstance(b, ast.Name)
                    ]

        job_classes, growing = {"BaseJob"}, True
        while growing:
            growing = False
            for name, bases in declared.items():
                if name not in job_classes and any(b in job_classes for b in bases):
                    job_classes.add(name)
                    growing = True
        job_classes.discard("BaseJob")

        assert job_classes - {t.job_class.__name__ for t in JobType} == set(not_offered)

    def test_unknown_job_type_is_usage_error(self) -> None:
        assert main(["submit", "spec.yaml", "--job-type", "llmf.trian"]) == 2

    def test_job_type_selects_the_class(self, tmp_path, monkeypatch, capsys) -> None:
        spec = _write_spec(
            tmp_path, {"user_login": "u", "exp_name": "e", "script": "echo hi"}
        )
        fake = _FakeJob(job_name="4242")
        monkeypatch.setattr(AnyScriptJob, "from_spec", lambda _path: fake)

        rc = main(["submit", spec, "--job-type", "any_script"])
        captured = capsys.readouterr()

        assert rc == 0
        assert "PIPELINE_JOB_ID=4242" in captured.out
