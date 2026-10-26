from pathlib import Path
from types import SimpleNamespace

import pytest

from job_submitter.asserts.train import TrainJobAssertor
from job_submitter.asserts.verl_train import VerlTrainJobAssertor
from job_submitter.core.job_config import JobTemplate


@pytest.mark.unit
class TestTrainJobAssertor:
    def test_assert_autometrics_config_calls_autometrics_job(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        called = {}

        class FakeAutometricsJob:
            def __init__(self, **kwargs):
                called["init"] = kwargs

            def to_config(self):
                return {"ok": True}

            def assert_job(self, job, job_config, source_job="standalone"):
                called["source_job"] = source_job

        monkeypatch.setattr(
            "job_submitter.asserts.train.AutometricsJobLLMF", FakeAutometricsJob
        )

        cfg = {
            "user": "u",
            "autometrics": {},
            "callbacks": {
                "autometrics": {
                    "eval_interval": 1,
                    "stable_save_interval": 2,
                    "nnodes": 1,
                    "user": "u",
                }
            },
        }
        TrainJobAssertor()._assert_autometrics_config(cfg, tmp_path / "train.yaml")

        assert called["init"]["user_login"] == "u"
        assert called["source_job"] == "TrainJobLLMF"

    def test_assert_autometrics_config_requires_autometrics_top_level_when_callbacks_present(
        self,
    ) -> None:
        cfg = {
            "callbacks": {
                "autometrics": {
                    "eval_interval": 1,
                    "stable_save_interval": 2,
                    "nnodes": 1,
                    "user": "u",
                }
            }
        }
        with pytest.raises(AssertionError):
            TrainJobAssertor()._assert_autometrics_config(cfg, Path("/tmp/train.yaml"))

    def test_assert_job_loads_config_and_delegates(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        called = {}
        assertor = TrainJobAssertor()

        monkeypatch.setattr(
            assertor,
            "_load_config",
            lambda path: {
                "autometrics": {},
                "callbacks": {
                    "autometrics": {
                        "eval_interval": 1,
                        "stable_save_interval": 2,
                        "nnodes": 1,
                        "user": "u",
                    }
                },
                "user": "u",
            },
        )
        monkeypatch.setattr(
            assertor,
            "_assert_autometrics_config",
            lambda cfg, path: called.setdefault("ok", True),
        )
        monkeypatch.setattr(
            assertor,
            "_assert_metric_sampler",
            lambda job, cfg: called.setdefault("metric_sampler", job),
        )

        fake_job = SimpleNamespace(
            user_login="u",
            exp_name="e",
            n_gpus=8,
            queue_name="shared",
            path_to_config=str(tmp_path / "train.yaml"),
        )
        Path(fake_job.path_to_config).write_text("x: 1\n", encoding="utf-8")
        template = JobTemplate(
            submission_type="cloud",
            user_login="u",
            exp_name="e",
            job_desc="u | e",
            base_image="cr.ai.cloud.ru/x/y:z",
            script="echo 1",
            region="SR008",
            n_workers=1,
        )
        assertor.assert_job(fake_job, template)
        assert called["ok"] is True
        assert called["metric_sampler"] is fake_job

    def test_assert_autometrics_config_allows_config_without_autometrics_sections(
        self,
    ) -> None:
        # Covers the path where neither "autometrics" nor callbacks autometrics exist.
        TrainJobAssertor()._assert_autometrics_config({}, Path("/tmp/train.yaml"))

    def test_assert_autometrics_config_validates_supported_sglang_tag(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        called = {}

        class FakeAutometricsJob:
            def __init__(self, **kwargs):
                called["init"] = kwargs

            def to_config(self):
                return {"ok": True}

            def assert_job(self, job, job_config, source_job="standalone"):
                called["source_job"] = source_job

        monkeypatch.setattr(
            "job_submitter.asserts.train.AutometricsJobLLMF", FakeAutometricsJob
        )

        cfg = {
            "user": "u",
            "autometrics": {},
            "callbacks": {
                "autometrics": {
                    "eval_interval": 1,
                    "stable_save_interval": 2,
                    "nnodes": 1,
                    "user": "u",
                    "sglang_tag": "opensource",
                }
            },
        }

        TrainJobAssertor()._assert_autometrics_config(cfg, tmp_path / "train.yaml")
        assert called["source_job"] == "TrainJobLLMF"

    def test_assert_autometrics_config_rejects_unknown_callback_key(self) -> None:
        cfg = {
            "user": "u",
            "autometrics": {},
            "callbacks": {
                "autometrics": {
                    "eval_interval": 1,
                    "stable_save_interval": 2,
                    "nnodes": 1,
                    "user": "u",
                    "queue_nmae": "pretrain-ml",
                }
            },
        }
        with pytest.raises(AssertionError, match="queue_nmae"):
            TrainJobAssertor()._assert_autometrics_config(cfg, Path("/tmp/train.yaml"))

    @pytest.mark.parametrize(
        "key", ["checkpoint_dir", "user_login", "exp_name", "n_workers"]
    )
    def test_assert_autometrics_config_rejects_keys_set_by_the_callback(
        self, key: str
    ) -> None:
        cfg = {
            "user": "u",
            "autometrics": {},
            "callbacks": {
                "autometrics": {
                    "eval_interval": 1,
                    "stable_save_interval": 2,
                    "nnodes": 1,
                    "user": "u",
                    key: "x",
                }
            },
        }
        with pytest.raises(AssertionError, match=key):
            TrainJobAssertor()._assert_autometrics_config(cfg, Path("/tmp/train.yaml"))

    @pytest.mark.parametrize(
        "key", ["queue_name", "priority_class", "gpu_type", "extra_env", "max_retry"]
    )
    def test_assert_autometrics_callback_keys_accepts_job_params(
        self, key: str
    ) -> None:
        TrainJobAssertor()._assert_autometrics_callback_keys(
            {"eval_interval": 1, "nnodes": 1, "user": "u", key: "x"}
        )


@pytest.mark.unit
class TestVerlTrainJobAssertor:
    def test_assert_pam_config_path_calls_autometrics_verl(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "pam.yaml"
        config_path.write_text("x: 1\n", encoding="utf-8")
        called = {}

        class FakeAutometricsVerl:
            def __init__(self, **kwargs):
                called["init"] = kwargs

            def to_config(self):
                return {"ok": True}

            def assert_job(self, job, job_config, source_job="standalone"):
                called["source_job"] = source_job

        monkeypatch.setattr(
            "job_submitter.asserts.verl_train.AutometricsJobVERL", FakeAutometricsVerl
        )

        fake_job = SimpleNamespace(
            user_login="u",
            pam_n_workers=2,
            pam_config_path=str(config_path),
        )

        VerlTrainJobAssertor()._assert_pam_config_path(fake_job)
        assert called["init"]["n_workers"] == 2
        assert called["source_job"] == "RayTrainJobVERL"

    def test_assert_job_validates_required_paths(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        env = tmp_path / "runtime.yaml"
        script.write_text("echo 1\n", encoding="utf-8")
        env.write_text("x: 1\n", encoding="utf-8")

        fake_job = SimpleNamespace(
            user_login="u",
            exp_name="e",
            n_gpus=8,
            queue_name="shared",
            script_path=str(script),
            runtime_env_config_path=str(env),
            pam_config_path=None,
        )
        template = JobTemplate(
            submission_type="cloud",
            user_login="u",
            exp_name="e",
            job_desc="u | e",
            base_image="cr.ai.cloud.ru/x/y:z",
            script="echo 1",
            region="SR008",
            n_workers=1,
        )
        VerlTrainJobAssertor().assert_job(fake_job, template)
