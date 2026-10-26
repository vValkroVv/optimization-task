import contextlib
import io
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
MODELS = ("l8-1_5b-a-0_3b", "l16-11_3b-a-1_1b")
MODULE = "contrib.job_submitter.job_submitter.jobs.llmf.warmup"


def script(model):
    return ROOT / f"sber_scripts/jobs/submit_{model}_deep_gemm_warmup.py"


class WarmupSubmissionTests(unittest.TestCase):
    def test_both_clis_submit_sequentially_and_stop_on_failure(self):
        for model in MODELS:
            for fail_at in (None, 1, 2):
                with self.subTest(model=model, fail_at=fail_at):
                    builds, submitted, attempted = [], [], []

                    class Job:
                        def __init__(self, index):
                            self.index = index

                        def submit(self):
                            attempted.append(self.index)
                            print("private-fixture-output", file=sys.stderr)
                            if self.index == fail_at:
                                raise RuntimeError("private-fixture-error")
                            submitted.append(self.index)

                    class Factory:
                        @staticmethod
                        def build_warmup_jobs(**kwargs):
                            builds.append(kwargs)
                            print("private-fixture-build")
                            return [Job(i) for i in (1, 2, 3)]

                    fake = types.ModuleType(MODULE)
                    fake.WarmupJobLLMF = Factory
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with patch.dict(sys.modules, {MODULE: fake}), patch("time.sleep") as sleep, \
                         patch.object(sys, "path", list(sys.path)), \
                         contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        if fail_at:
                            with self.assertRaisesRegex(SystemExit, "reconcile existing jobs") as error:
                                runpy.run_path(str(script(model)), run_name="__main__")
                            self.assertNotIn("private-fixture", str(error.exception))
                        else:
                            runpy.run_path(str(script(model)), run_name="__main__")
                    self.assertEqual(submitted, list(range(1, fail_at or 4)))
                    self.assertEqual(attempted, list(range(1, (fail_at or 3) + 1)))
                    self.assertEqual([c.args for c in sleep.call_args_list], [(1,)] * (len(attempted) - 1))
                    self.assertNotIn("private-fixture", stdout.getvalue() + stderr.getvalue())
                    self.assertEqual(len(builds), 1)
                    self.assertEqual(builds[0]["warmup_config_path"], ROOT / f"sber_configs/warmup/{model}-deep-gemm.yaml")
                    self.assertEqual(builds[0]["warmup_type"], "deep_gemm")
                    self.assertEqual((builds[0]["n_gpus"], builds[0]["n_workers"]), (1, 1))
                    self.assertEqual((builds[0]["region"], builds[0]["priority_class"]), ("o5", "medium"))
                    self.assertEqual(builds[0]["extra_env"], {"DG_CACHE_FORCE_SKIP": "1"})

    def test_import_does_not_construct_native_jobs(self):
        with patch.dict(sys.modules, {MODULE: None}), patch.object(sys, "path", list(sys.path)):
            for model in MODELS:
                runpy.run_path(str(script(model)), run_name="test_warmup")

    def test_warmup_geometry_covers_training_shapes(self):
        with patch.object(sys, "path", list(sys.path)):
            mbs = runpy.run_path(str(ROOT / "sber_scripts/analytics/experimentation/l8-st1-mbs/submit_exp.py"))["build_experiment"]()
        for model in MODELS:
            warm = OmegaConf.load(ROOT / f"sber_configs/warmup/{model}-deep-gemm.yaml").deep_gemm
            train = OmegaConf.load(ROOT / f"sber_configs/baseline_registry/giga/{model}-4_0-papaya.yaml")
            params = train.model.config_overrides
            capacity = (max(s.config_overrides["model.config_overrides.expert_capacity_tokens"] for s in mbs.setups)
                        if model == MODELS[0] else params.expert_capacity_tokens)
            self.assertEqual(warm.hid_dim, params.hidden_size)
            self.assertEqual(warm.int_dim, params.intermediate_size)
            self.assertEqual(warm.num_groups * train.ep_size, params.num_routed_experts)
            self.assertEqual(warm.num_sms, params.float8_deep_gemm_num_sms)
            self.assertEqual(warm.max_size, 32000)
            self.assertGreaterEqual(warm.max_size, capacity)
            self.assertEqual(warm.min_size, 128)
            self.assertEqual(set(warm.gemm_groups), {
                f"w{weight}_{operation}" for weight in (1, 2)
                for operation in ("m_grouped", "m_grouped_nt_for_dgrad_transposed", "k_grouped")
            })


if __name__ == "__main__":
    unittest.main()
