from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

from job_submitter.asserts.base import BaseJobAssertor
from job_submitter.core.job_config import JobTemplate
from job_submitter.jobs.llmf.autometrics import AutometricsJobLLMF

if TYPE_CHECKING:
    from job_submitter.jobs.llmf.train import TrainJobLLMF

CALLBACK_OWN_KEYS = frozenset(
    {"eval_interval", "stable_save_interval", "nnodes", "user"}
)
CALLBACK_SET_JOB_KEYS = frozenset(
    {"checkpoint_dir", "user_login", "exp_name", "n_workers"}
)
AUTOMETRICS_CALLBACK_KEYS = (
    frozenset(field.name for field in fields(AutometricsJobLLMF) if field.init)
    - CALLBACK_SET_JOB_KEYS
) | CALLBACK_OWN_KEYS


class TrainJobAssertor(BaseJobAssertor):
    def _assert_autometrics_callback_keys(self, autometrics: dict) -> None:
        unknown = sorted(set(autometrics) - AUTOMETRICS_CALLBACK_KEYS)
        assert not unknown, self._pretty_config_param_print(
            {"callbacks.autometrics": sorted(AUTOMETRICS_CALLBACK_KEYS)},
            f"unknown keys {unknown}, every key must be one of",
        )

    def _assert_autometrics_config(self, train_config: dict, config_path: Path) -> None:
        callback_config = train_config.get("callbacks", {}).get("autometrics")
        if callback_config is not None:
            self._assert_autometrics_callback_keys(callback_config)

        if "autometrics" in train_config:
            assert "callbacks" in train_config, self._pretty_print_missing_key(
                "callbacks"
            )
            callbacks = train_config["callbacks"]
            assert "autometrics" in callbacks, self._pretty_print_missing_key(
                "callbacks.autometrics"
            )
            autometrics = callbacks["autometrics"]
            assert "eval_interval" in autometrics, self._pretty_print_missing_key(
                "autometrics.eval_interval"
            )
            assert "stable_save_interval" in autometrics, (
                self._pretty_print_missing_key("autometrics.stable_save_interval")
            )
            assert "nnodes" in autometrics, self._pretty_print_missing_key(
                "autometrics.nnodes"
            )
            assert "user" in autometrics, self._pretty_print_missing_key(
                "autometrics.user"
            )
        elif "callbacks" in train_config and "autometrics" in train_config["callbacks"]:
            autometrics = train_config["callbacks"]["autometrics"]
            assert "eval_interval" in autometrics, self._pretty_print_missing_key(
                "autometrics.eval_interval"
            )
            assert "stable_save_interval" in autometrics, (
                self._pretty_print_missing_key("autometrics.stable_save_interval")
            )
            assert "nnodes" in autometrics, self._pretty_print_missing_key(
                "autometrics.nnodes"
            )
            assert "user" in autometrics, self._pretty_print_missing_key(
                "autometrics.user"
            )
            assert "autometrics" in train_config, self._pretty_print_missing_key(
                "autometrics"
            )

        if (
            "autometrics" in train_config
            and "callbacks" in train_config
            and "autometrics" in train_config["callbacks"]
        ):
            if "sglang_tag" in train_config["callbacks"]["autometrics"]:
                assert train_config["callbacks"]["autometrics"]["sglang_tag"] in (
                    "opensource",
                    "custom",
                ), self._pretty_config_param_print(
                    {
                        "autometrics.callbacks.autometrics.sglang_tag": [
                            "opensource",
                            "custom",
                        ]
                    }
                )
            job = AutometricsJobLLMF(
                user_login=train_config["user"],
                exp_name="mock_exp",
                n_workers=train_config["callbacks"]["autometrics"]["nnodes"],
                n_gpus=train_config["callbacks"]["autometrics"].get("n_gpus", 8),
                slurm_submit_type="api",
                path_to_config=str(config_path),
                checkpoint_dir=".",
            )
            job_config = job.to_config()
            job.assert_job(job, job_config, source_job="TrainJobLLMF")

    def assert_job(self, job: TrainJobLLMF, job_template: JobTemplate) -> None:
        super().assert_job(job, job_template)
        config = Path(job.path_to_config)
        assert config.exists(), self._pretty_params_print(
            {"path_to_config": ["/some/valid/train/config.yaml"]}
        )
        loaded_config = self._load_config(config)
        self._assert_autometrics_config(loaded_config, config)
