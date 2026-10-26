from dataclasses import fields as dc_fields

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError
from sber_tests.factories import DummyJob

from job_submitter.jobs.llmf.train import TrainJobLLMF


def write_spec(tmp_path, mapping) -> str:
    path = tmp_path / "spec.yaml"
    OmegaConf.save(OmegaConf.create(mapping), str(path))
    return str(path)


@pytest.mark.unit
class TestFromSpec:
    def test_minimal_spec_applies_defaults(self, tmp_path) -> None:
        job = DummyJob.from_spec(
            write_spec(tmp_path, {"user_login": "alice", "exp_name": "e1"})
        )

        assert job.user_login == "alice"
        assert job.exp_name == "e1"
        assert job.n_gpus == 8
        assert job.region is None
        assert job.priority_class is None
        assert job.script_value == "echo test"

    def test_optional_overrides_applied(self, tmp_path) -> None:
        spec = write_spec(
            tmp_path,
            {
                "user_login": "bob",
                "exp_name": "e2",
                "n_gpus": 4,
                "region": "o1",
                "priority_class": "high",
                "script_value": "python train.py",
            },
        )

        job = DummyJob.from_spec(spec)

        assert job.n_gpus == 4
        assert job.region == "o1"
        assert job.priority_class == "high"
        assert job.script_value == "python train.py"

    def test_returns_correct_subclass(self, tmp_path) -> None:
        spec = write_spec(tmp_path, {"user_login": "a", "exp_name": "e"})

        assert isinstance(DummyJob.from_spec(spec), DummyJob)

    def test_unknown_key_rejected(self, tmp_path) -> None:
        spec = write_spec(tmp_path, {"user_login": "a", "exp_name": "e", "n_gpuz": 4})

        with pytest.raises(ConfigKeyError, match="n_gpuz"):
            DummyJob.from_spec(spec)

    def test_wrong_type_rejected(self, tmp_path) -> None:
        spec = write_spec(
            tmp_path, {"user_login": "a", "exp_name": "e", "n_gpus": "eight"}
        )

        with pytest.raises(ValueError, match="n_gpus"):
            DummyJob.from_spec(spec)

    def test_numeric_string_is_coerced(self, tmp_path) -> None:
        spec = write_spec(tmp_path, {"user_login": "a", "exp_name": "e", "n_gpus": "4"})

        assert DummyJob.from_spec(spec).n_gpus == 4

    def test_init_false_field_rejected(self, tmp_path) -> None:
        spec = write_spec(
            tmp_path, {"user_login": "a", "exp_name": "e", "slurm_jwt_token": "tok"}
        )

        with pytest.raises(TypeError, match="slurm_jwt_token"):
            DummyJob.from_spec(spec)

    def test_missing_required_raises(self, tmp_path) -> None:
        spec = write_spec(tmp_path, {"user_login": "a"})

        with pytest.raises(TypeError, match="exp_name"):
            DummyJob.from_spec(spec)

    def test_empty_spec_file_raises(self, tmp_path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")

        with pytest.raises(TypeError, match="user_login"):
            DummyJob.from_spec(str(path))

    def test_union_typed_field_accepted(self, tmp_path) -> None:
        spec = write_spec(
            tmp_path,
            {
                "user_login": "a",
                "exp_name": "e",
                "region": "o1",
                "path_to_config": "/nfs/train.yaml",
            },
        )

        assert TrainJobLLMF.from_spec(spec).path_to_config == "/nfs/train.yaml"

    @pytest.mark.parametrize("job_cls", [DummyJob, TrainJobLLMF])
    def test_dataclass_is_the_schema(self, job_cls) -> None:
        schema = set(OmegaConf.structured(job_cls).keys())

        assert {f.name for f in dc_fields(job_cls) if f.init} <= schema
