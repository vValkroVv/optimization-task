import ast
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import tempfile
import types
import typing
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from scripts.train.src.config_loader import prepare_train_config
from sber_scripts.analytics.experimentation import run

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "sber_scripts/analytics/experimentation"
SERIES = ("l8-st1-mbs", "l8-st2-gbs-lr", "l8-st3-multi-seed")


def experiment(name):
    with patch.object(sys, "path", list(sys.path)):
        return runpy.run_path(str(EXPERIMENTS / name / "submit_exp.py"))[
            "build_experiment"
        ]()


def native_tracking_helpers():
    # Execute native pure functions without importing credential/SDK integrations.
    source = ROOT / "contrib/job_submitter/job_submitter/core/tracking.py"
    names = {"_get_nested", "_resolve_tracking_config", "extract_logs_path"}
    functions = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in functions} == names
    module = types.ModuleType("job_submitter.core.tracking")
    module.__dict__.update(Any=typing.Any, Optional=typing.Optional)
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        module.__dict__,
    )
    return module


class L8ExperimentTests(unittest.TestCase):
    def test_submitted_arguments_match_prepared_configs(self):
        for name in SERIES:
            exp = experiment(name)
            calls = []

            class Job:
                def __init__(self, **kwargs):
                    calls.append(kwargs)

                def submit(self):
                    return self

            with redirect_stdout(io.StringIO()):
                exp.submit(job_factory=Job)
            expected = run.effective_configs(exp)
            self.assertEqual(len(calls), len(expected))
            for kwargs, config in zip(calls, expected):
                actual = OmegaConf.merge(
                    OmegaConf.load(kwargs["path_to_config"]),
                    OmegaConf.from_cli(shlex.split(kwargs["script_args"])),
                )
                actual = OmegaConf.to_container(
                    prepare_train_config(actual), resolve=True
                )
                self.assertEqual(actual, config)
                self.assertEqual((kwargs["n_workers"], kwargs["n_gpus"]), (1, 8))
                self.assertEqual(
                    (kwargs["region"], kwargs["priority_class"]), ("o5", "medium")
                )
                self.assertIsNone(kwargs["queue_name"])

    def test_gbs_lr_grid_has_equal_token_budgets_and_baseline_metrics(self):
        exp = experiment(SERIES[1])
        self.assertEqual(OmegaConf.to_container(OmegaConf.load(exp.baseline_patch)), {})
        tasks = OmegaConf.to_container(
            OmegaConf.load(exp.baseline_config), resolve=False
        )["autometrics"]
        configs = run.effective_configs(exp)
        self.assertEqual(
            {(c["global_train_batch_size"], c["optimizer"]["lr"]) for c in configs},
            {(gbs, lr) for gbs in (96, 192, 384) for lr in (7e-4, 1e-3, 1.4e-3)},
        )
        for c in configs:
            tokens_per_batch = c["global_train_batch_size"] * c["max_seq_len"]
            self.assertEqual(
                int(c["max_duration"][:-2]) * tokens_per_batch, 62914560000
            )
            self.assertEqual(
                int(c["scheduler"]["t_warmup"][:-2]) * tokens_per_batch, 1572864000
            )
            self.assertEqual(
                c["callbacks"]["ema_weights"]["start_batch"] * tokens_per_batch,
                6291456000,
            )
            self.assertIsInstance(c["optimizer"]["lr"], float)
            self.assertAlmostEqual(
                c["optimizer"]["weight_decay"], 0.1 * c["optimizer"]["lr"]
            )
            self.assertEqual(c["autometrics"]["evaluators"], tasks["evaluators"])
            self.assertEqual(c["seed"], 4096)

    def test_mbs_profile_and_capacity(self):
        configs = run.effective_configs(experiment(SERIES[0]))
        self.assertEqual(
            {c["device_train_microbatch_size"] for c in configs}, {3, 6, 12, 24}
        )
        for c in configs:
            mbs = c["device_train_microbatch_size"]
            self.assertEqual(c["global_train_batch_size"], 192)
            self.assertEqual(c["max_duration"], "100ba")
            self.assertEqual(c["console_log_interval"], "1ba")
            self.assertEqual(c["eval_interval"], 0)
            self.assertEqual(c["save_interval"], "0ba")
            self.assertIsNone(c["stable_save_interval"])
            self.assertEqual(
                int(c["scheduler"]["t_warmup"][:-2]) * 192 * c["max_seq_len"],
                1572864000,
            )
            self.assertEqual(c["optimizer"]["lr"], 1e-3)
            self.assertEqual(c["model"]["config_overrides"]["kv_lora_rank"], 256)
            self.assertEqual(
                c["model"]["config_overrides"]["expert_capacity_tokens"],
                mbs * 4000 // 3,
            )
            self.assertFalse(c["papayafsdp_config"]["activation_checkpointing"])
            self.assertIsNone(c["autometrics"])
            self.assertNotIn("autometrics", c["callbacks"])
            self.assertEqual(c["callbacks"]["ema_weights"]["start_batch"], 20)
            self.assertTrue(
                c["callbacks"]["expert_balancing_monitor"]["activate_activation_norms"]
            )

    def test_multi_seed_changes_only_seed_and_run_identity(self):
        configs = run.effective_configs(experiment(SERIES[2]))
        self.assertEqual({c["seed"] for c in configs}, {1024, 2048, 4096, 8192, 16384})
        normalized = []
        for c in configs:
            self.assertEqual(c["train_loader"]["dataset"]["shuffle_seed"], 4096)
            c.pop("seed")
            normalized.append(
                json.dumps(c, sort_keys=True).replace(c["run_name"], "fixed-run")
            )
        self.assertEqual(len(set(normalized)), 1)

    def test_native_tracking_gets_fresh_effective_run_paths(self):
        tracking = native_tracking_helpers()
        fake = types.ModuleType("contrib.job_submitter.job_submitter.jobs.llmf.train")

        class NativeJob:
            def __init__(self, **kwargs):
                pass

            def _tracking_merged_config(self, cfg):
                return None

        fake.TrainJobLLMF = NativeJob
        paths = set()
        with patch.dict(
            sys.modules, {fake.__name__: fake, tracking.__name__: tracking}
        ):
            for name in SERIES:
                exp = experiment(name)
                raw = OmegaConf.to_container(
                    OmegaConf.load(exp.baseline_config), resolve=False
                )
                with self.assertRaisesRegex(ValueError, "run_name"):
                    tracking.extract_logs_path(raw)
                exp = replace(
                    exp,
                    exp_name="fixture-fresh",
                    user_login="fixture-user",
                    setups=[
                        replace(s, run_name=s.run_name + "-fresh") for s in exp.setups
                    ],
                )
                for c in run.effective_configs(exp):
                    job = run.tracking_job({}, c)
                    merged = job._tracking_merged_config(None)
                    self.assertEqual(merged, c)
                    expected = (
                        c["loggers"]["tensorboard"]["log_dir"] + "/" + c["run_name"]
                    )
                    self.assertEqual(job._resolve_tb_log_dir(), expected)
                    self.assertEqual(tracking.extract_logs_path(merged), expected)
                    self.assertIn("/fixture-fresh/", expected)
                    paths.add(expected)
                    merged["run_name"] = c["run_name"] = "changed"
                    self.assertEqual(job._resolve_tb_log_dir(), expected)
        self.assertEqual(len(paths), 18)

    def test_plan_cli_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as cwd:
            for name, count in zip(SERIES, (4, 9, 5)):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(EXPERIMENTS / name / "submit_exp.py"),
                        "--plan",
                        "--exp-name",
                        "fixture-exp",
                        "--run-suffix",
                        "fresh",
                    ],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"Prepared {count} configurations", result.stdout)
            self.assertEqual(list(Path(cwd).iterdir()), [])

    def test_resume_validates_checkpoints_before_submission(self):
        exp = experiment(SERIES[0])
        configs = run.effective_configs(exp)
        for checkpoint_state, error, message in (
            ("complete", None, None),
            ("missing", FileNotFoundError, "missing"),
            ("mixed", ValueError, "different checkpoint folders"),
        ):
            for load_path in (None, "/checkpoints/previous.pt"):
                with (
                    self.subTest(checkpoints=checkpoint_state, load_path=load_path),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    prepared = []
                    for config in configs:
                        save_folder = Path(directory) / config["run_name"]
                        checkpoint_folder = save_folder / "ep0_ba100"
                        checkpoint_folder.mkdir(parents=True)
                        for rank in range(8):
                            checkpoint = checkpoint_folder / f"part{rank}.pt"
                            checkpoint.touch()
                            (save_folder / f"latest-rank{rank}.pt").symlink_to(
                                checkpoint
                            )
                        prepared.append(
                            {
                                **config,
                                "save_folder": str(save_folder),
                                "load_path": load_path,
                            }
                        )
                    latest = Path(prepared[-1]["save_folder"]) / "latest-rank7.pt"
                    if checkpoint_state == "missing":
                        latest.resolve().unlink()
                    elif checkpoint_state == "mixed":
                        other_checkpoint = latest.parent / "ep0_ba200" / "part7.pt"
                        other_checkpoint.parent.mkdir()
                        other_checkpoint.touch()
                        latest.unlink()
                        latest.symlink_to(other_checkpoint)
                    with (
                        patch.object(run.sys, "platform", "linux"),
                        patch.object(Path, "is_dir", return_value=True),
                        patch.object(run, "tracking_job") as factory,
                        patch.object(run.time, "sleep"),
                        redirect_stdout(io.StringIO()),
                    ):
                        if error is not None:
                            with self.assertRaisesRegex(error, message):
                                run.submit_on_cluster(exp, prepared, resume=True)
                            factory.assert_not_called()
                            continue
                        run.submit_on_cluster(exp, prepared, resume=True)
                    self.assertEqual(factory.call_count, len(prepared))
                    self.assertEqual(
                        [call.args[1] for call in factory.call_args_list], prepared
                    )
                    self.assertEqual(
                        factory.return_value.submit.call_count, len(prepared)
                    )

    def test_resume_requires_autoresume_for_every_setup(self):
        exp = experiment(SERIES[0])
        configs = run.effective_configs(exp)
        for override in ({}, {"autoresume": False}, {"autoresume": None}):
            with self.subTest(override=override):
                prepared = [dict(c) for c in configs]
                prepared[-1].pop("autoresume")
                prepared[-1].update(override)
                with (
                    patch.object(run.sys, "platform", "linux"),
                    patch.object(Path, "is_dir", return_value=True),
                    patch.object(run, "tracking_job") as factory,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "--resume requires autoresume=true"
                    ):
                        run.submit_on_cluster(exp, prepared, resume=True)
                factory.assert_not_called()

    def test_load_path_requires_explicit_resume(self):
        exp = experiment(SERIES[0])
        configs = run.effective_configs(exp)
        configs[-1]["load_path"] = "/checkpoints/previous.pt"
        with (
            patch.object(run.sys, "platform", "linux"),
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "exists", return_value=False),
            patch.object(run, "tracking_job") as factory,
        ):
            with self.assertRaisesRegex(ValueError, "Existing checkpoint"):
                run.submit_on_cluster(exp, configs)
        factory.assert_not_called()

    def test_resume_preserves_submission_guards(self):
        exp = experiment(SERIES[0])
        configs = run.effective_configs(exp)
        for key, value, error in (
            ("run_name", "wrong-run", "Prepared configs do not match"),
            (
                "save_folder",
                configs[0]["save_folder"],
                "Run output paths are not unique",
            ),
        ):
            with self.subTest(key=key):
                prepared = [dict(c) for c in configs]
                prepared[-1][key] = value
                with (
                    patch.object(run.sys, "platform", "linux"),
                    patch.object(Path, "is_dir", return_value=True),
                    patch.object(run, "tracking_job") as factory,
                ):
                    with self.assertRaisesRegex(ValueError, error):
                        run.submit_on_cluster(exp, prepared, resume=True)
                factory.assert_not_called()
        for directories, error in (
            ([False, False], "Submit from the authorized target cluster checkout"),
            ([True, False], "Required kernel cache directory is unavailable"),
        ):
            with self.subTest(directories=directories):
                with (
                    patch.object(run.sys, "platform", "linux"),
                    patch.object(Path, "is_dir", side_effect=directories),
                    patch.object(run, "tracking_job") as factory,
                ):
                    with self.assertRaisesRegex(ValueError, error):
                        run.submit_on_cluster(exp, configs, resume=True)
                factory.assert_not_called()

    def test_cli_identity_overrides_experiment_only_when_provided(self):
        exp = replace(
            experiment(SERIES[0]),
            exp_name="configured-exp",
            user_login="configured-user",
        )
        for cli_exp in (None, "cli-exp"):
            for cli_user in (None, "cli-user"):
                with self.subTest(exp_name=cli_exp, user_login=cli_user):
                    argv = ["submit_exp.py", "--submit", "--run-suffix", "original"]
                    if cli_exp is not None:
                        argv.extend(["--exp-name", cli_exp])
                    if cli_user is not None:
                        argv.extend(["--user-login", cli_user])
                    with (
                        patch.object(sys, "argv", argv),
                        patch.object(run, "submit_on_cluster") as submit,
                        redirect_stdout(io.StringIO()),
                    ):
                        run.main(lambda: exp)
                    submit.assert_called_once()
                    prepared_exp, configs = submit.call_args.args
                    expected_exp = cli_exp if cli_exp is not None else exp.exp_name
                    expected_user = cli_user if cli_user is not None else exp.user_login
                    self.assertEqual(prepared_exp.exp_name, expected_exp)
                    self.assertEqual(prepared_exp.user_login, expected_user)
                    for config in configs:
                        self.assertEqual(config["exp_name"], expected_exp)
                        self.assertEqual(config["user"], expected_user)
                        self.assertIn(f"/{expected_exp}/", config["save_folder"])

    def test_submit_cli_passes_resume_and_preserves_run_identity(self):
        exp = experiment(SERIES[0])
        for resume in (False, True):
            with self.subTest(resume=resume):
                argv = [
                    "submit_exp.py",
                    "--submit",
                    "--exp-name",
                    "fixture-exp",
                    "--run-suffix",
                    "original",
                ] + (["--resume"] if resume else [])
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(run, "submit_on_cluster") as submit,
                    redirect_stdout(io.StringIO()),
                ):
                    run.main(lambda: exp)
                submit.assert_called_once()
                self.assertEqual(submit.call_args.kwargs, {"resume": resume})
                prepared_exp, configs = submit.call_args.args
                self.assertEqual(prepared_exp.exp_name, "fixture-exp")
                self.assertEqual(
                    [c["run_name"] for c in configs],
                    [s.run_name + "-original" for s in exp.setups],
                )
                self.assertTrue(all(c["autoresume"] for c in configs))

    def test_submission_guards_and_partial_failure(self):
        exp = experiment(SERIES[0])
        configs = run.effective_configs(exp)
        with (
            patch.object(run.sys, "platform", "darwin"),
            patch.object(run, "tracking_job") as factory,
        ):
            with self.assertRaises(ValueError):
                run.submit_on_cluster(exp, configs)
            factory.assert_not_called()
        with (
            patch.object(run.sys, "platform", "linux"),
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "exists", return_value=True),
            patch.object(run, "tracking_job") as factory,
        ):
            with self.assertRaisesRegex(ValueError, "Existing checkpoint"):
                run.submit_on_cluster(exp, configs)
            factory.assert_not_called()
        completed = []

        class Job:
            def __init__(self, kwargs, config):
                self.config = config

            def submit(self):
                if len(completed) == 1:
                    raise RuntimeError("fixture failure")
                completed.append(self.config)

        with (
            patch.object(run.sys, "platform", "linux"),
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "exists", return_value=False),
            patch.object(run, "tracking_job", Job),
            patch.object(run.time, "sleep") as sleep,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError):
                run.submit_on_cluster(exp, configs)
        self.assertEqual(completed, configs[:1])
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
