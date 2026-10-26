from __future__ import annotations

import shlex
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from sber_scripts.analytics.experimentation import Experiment, Setup
from sber_scripts.analytics.experimentation.submit import (
    BASELINE_REGISTRY,
    build_script_args,
    load_baseline_patch,
)


class FakeJob:
    instances: list[FakeJob] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.instances.append(self)

    def submit(self) -> str:
        return str(self.kwargs["script_args"])


class ExperimentSubmitTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeJob.instances.clear()

    def test_empty_baseline_patch_has_no_overrides(self) -> None:
        patch_path = (
            BASELINE_REGISTRY.parents[1]
            / "sber_scripts"
            / "analytics"
            / "experimentation"
            / "example_exp"
            / "baseline_patch.yaml"
        )
        self.assertEqual(load_baseline_patch(patch_path), {})

    def test_recipe_patch_replaces_nested_values_without_dropping_other_fields(self) -> None:
        baseline = OmegaConf.create({
            "model": {"config_overrides": {
                "hidden_size": 2048,
                "deepseek_init_std": None,
                "initializer_range": 0.006,
                "zero_init_post_attention_norm": None,
                "norm_type": "ZeroCenteredGatedNorm",
                "lm_head_logit_softcapping": None,
                "linear_gating_type": "gated_rmsnorm_sigmoid_zero_centered",
            }},
            "optimizer": {"lr": 0.001},
        })
        patch = load_baseline_patch(BASELINE_REGISTRY / "diff" / "4_0-3_5.yaml")

        cli_overrides = OmegaConf.from_cli(shlex.split(build_script_args(patch)))
        actual = OmegaConf.merge(baseline, cli_overrides)

        params = actual.model.config_overrides
        self.assertIsNone(params.initializer_range)
        self.assertEqual(params.deepseek_init_std, 0.006)
        self.assertTrue(params.zero_init_post_attention_norm)
        self.assertEqual(params.norm_type, "GatedNorm")
        self.assertEqual(params.lm_head_logit_softcapping, 30.0)
        self.assertEqual(params.linear_gating_type, "gated_rmsnorm_sigmoid")
        self.assertEqual(params.hidden_size, 2048)
        self.assertEqual(actual.optimizer.lr, 0.001)

    def test_patch_and_setup_are_rendered_as_dotlist(self) -> None:
        script_args = build_script_args(
            {"model": {"config_overrides": {"old_feature": None}}},
            {
                "run_name": "feature-seed-2048",
                "global_seed": 2048,
                "optimizer.lr": 0.001,
            },
        )
        self.assertEqual(
            shlex.split(script_args),
            [
                "model.config_overrides.old_feature=null",
                'run_name="feature-seed-2048"',
                "global_seed=2048",
                "optimizer.lr=0.001",
            ],
        )

    def test_setup_overrides_patch_value(self) -> None:
        script_args = build_script_args(
            {"optimizer": {"lr": 0.001}},
            {"optimizer.lr": 0.002},
        )
        self.assertEqual(shlex.split(script_args), ["optimizer.lr=0.002"])

    @patch(
        "sber_scripts.analytics.experimentation.submit.load_baseline_patch",
        return_value={"model": {"config_overrides": {"feature": True}}},
    )
    def test_one_job_is_submitted_per_setup(self, _load_patch: object) -> None:
        baseline = BASELINE_REGISTRY / "mhc" / "3b-3_0b-a-0_6b-4_0-papaya.yaml"
        experiment = Experiment(
            exp_name="feature-test",
            user_login="tester",
            baseline_config=baseline,
            baseline_patch=Path("unused-in-this-test.yaml"),
            setups=[
                Setup(run_name="run-a", config_overrides={"global_seed": 1}),
                Setup(run_name="run-b", config_overrides={"global_seed": 2}),
            ],
            n_workers=1,
            dry_run=True,
        )
        results = experiment.submit(job_factory=FakeJob)

        self.assertEqual(len(results), 2)
        self.assertEqual(len(FakeJob.instances), 2)
        first = FakeJob.instances[0].kwargs
        self.assertEqual(first["path_to_config"], baseline)
        self.assertEqual(first["exp_name"], "feature-test | run-a")
        self.assertEqual(first["user_login"], "tester")
        self.assertEqual(first["n_workers"], 1)
        self.assertEqual(first["n_gpus"], 8)
        self.assertTrue(first["dry_run"])
        self.assertEqual(
            shlex.split(str(first["script_args"])),
            [
                "model.config_overrides.feature=true",
                'run_name="run-a"',
                "global_seed=1",
                'user="tester"',
                'exp_name="feature-test"',
            ],
        )

    def test_duplicate_run_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "run_name values must be unique"):
            Experiment(
                exp_name="feature-test",
                user_login="tester",
                baseline_config=BASELINE_REGISTRY / "mhc" / "3b-3_0b-a-0_6b-4_0-papaya.yaml",
                baseline_patch=Path("unused.yaml"),
                setups=[Setup(run_name="same"), Setup(run_name="same")],
            )

    def test_setup_run_name_is_not_duplicated_in_overrides(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain run_name"):
            Setup(run_name="run-a", config_overrides={"run_name": "run-b"})


if __name__ == "__main__":
    unittest.main()
