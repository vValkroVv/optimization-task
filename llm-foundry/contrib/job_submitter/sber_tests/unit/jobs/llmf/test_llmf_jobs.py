from pathlib import Path

import pytest

from job_submitter.jobs.llmf.autometrics import AutometricsJobLLMF
from job_submitter.jobs.llmf.train import TrainJobLLMF
from sber_tests.dry_run_helpers import assert_dry_run_submit_contract


@pytest.mark.unit
class TestTrainJobLLMF:
    def test_build_env_stamps_resolved_tensorboard_path(self, tmp_path: Path) -> None:
        config_path = tmp_path / "train.yaml"
        config_path.write_text(
            "run_name: run-42\nloggers:\n  tensorboard:\n    log_dir: /tensorboard\n",
            encoding="utf-8",
        )
        job = TrainJobLLMF(
            user_login="u",
            exp_name="submitted-name",
            submission_type="slurm",
            region="o1",
            path_to_config=config_path,
        )

        env = job._build_env()

        assert env["TB_LOG_DIR"] == "/tensorboard/run-42"

    def test_script_builds_expected_command(self) -> None:
        job = TrainJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            path_to_config="/tmp/train.yaml",
            use_clearml=True,
            script_args="foo=bar",
        )
        script = job._script()
        assert "scripts/train/train.py /tmp/train.yaml" in script
        assert "use_clearml=true" in script
        assert "foo=bar" in script

    def test_script_passes_the_metric_sampler_settings_quoted_before_script_args(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "train.yaml"
        config_path.write_text("exp_name: exp\nrun_name: run\n", encoding="utf-8")
        job = TrainJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            path_to_config=config_path,
            metric_sampler="light",
            metric_sampler_overrides={"capture": {"every_n_steps": 5}},
            script_args="foo=bar",
        )

        script = job._script()

        assert "use_clearml=false 'callbacks.metric_sampler.output_dir=" in script
        assert "'callbacks.metric_sampler.capture_interval=5'" in script
        assert script.endswith("'loggers.metric_sampler.window_steps=5' foo=bar")

    def test_submit_dry_run_calls_before_submit_and_skips_backend(
        self, monkeypatch
    ) -> None:
        job = TrainJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            path_to_config="/tmp/train.yaml",
            dry_run=True,
        )
        assert_dry_run_submit_contract(
            monkeypatch=monkeypatch,
            job=job,
            patch_anonimize=True,
        )


@pytest.mark.unit
class TestAutometricsJobLLMF:
    def test_script_switches_by_submission_type(self, tmp_path: Path) -> None:
        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            checkpoint_dir=str(tmp_path),
        )
        assert "cd " in job._script()
        assert "--checkpoint" in job._script()

        job.submission_type = "cloud"
        assert "--wdir " in job._script()

    def test_explicit_queue_and_priority_survive_parent_job_env(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("PAM_QUEUE_NAME", "shared")
        monkeypatch.setenv("PAM_PRIORITY_CLASS", "low")

        cfg = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            workspace_name="pretrain",
            checkpoint_dir=str(tmp_path),
            queue_name="pretrain-ml",
            priority_class="high",
        ).to_config()

        assert cfg.queue_name == "pretrain-ml"
        assert cfg.priority_class == "high"
        assert cfg.env_variables["PAM_QUEUE_NAME"] == "pretrain-ml"
        assert cfg.env_variables["PAM_PRIORITY_CLASS"] == "high"
