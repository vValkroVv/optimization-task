import json
import sys
import zipfile
from pathlib import Path

import pytest
from node_runner.src.job_dispatcher import JD

from node_runner.src.config_model import (
    ForgeEvaluatorConfig,
    ModelInfoConfig as ModelInfo,
)
from node_runner.src.metrics.evaluators.forge import ForgeEvaluator
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit


# Default fake bundles laid down in the fake forge_root. Each carries a
# meta.json with model temperature/max_tokens so inheritance can be exercised.
_DEFAULT_RECIPES = {
    "camo": {
        "name": "camo",
        "dataset": "adv_bench",
        "model": {
            "provider": "gigachat",
            "model": "x",
            "temperature": 0.7,
            "max_tokens": 512,
        },
        "judge": {
            "provider": "gigachat",
            "model": "x",
            "prompt_template": "builtin:simple",
        },
    },
    "role_playing": {
        "name": "role_playing",
        "dataset": "adv_bench",
        "model": {
            "provider": "gigachat",
            "model": "x",
            "temperature": 0.7,
            "max_tokens": 512,
        },
        "judge": {
            "provider": "gigachat",
            "model": "x",
            "prompt_template": "builtin:simple",
        },
    },
}


def _write_bundle(path: Path, meta: dict) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "attack.py",
            "PARAMETERS = {}\n\n\ndef generate_message(prompt):\n"
            '    return [{"role": "user", "content": prompt}]\n',
        )
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False))


def _build_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    recipes: dict | None = None,
    **cfg_overrides,
) -> ForgeEvaluator:
    recipes = recipes if recipes is not None else _DEFAULT_RECIPES

    forge_root = tmp_path / "forge_lite"
    recipes_dir = forge_root / "recipes"
    data_dir = forge_root / "data"
    recipes_dir.mkdir(parents=True)
    data_dir.mkdir()
    for name, meta in recipes.items():
        _write_bundle(recipes_dir / f"{name}.attack", meta)

    # Force _unzip_forge_lite() to use our fake root as is (no archive).
    monkeypatch.setattr(ForgeEvaluator, "ARCHIVE_PATH", tmp_path / "missing.tar.zst")
    monkeypatch.setattr(ForgeEvaluator, "RUNTIME_ROOT", forge_root)

    cfg_overrides.setdefault("judge", "judge-a")
    cfg = ForgeEvaluatorConfig(**cfg_overrides)
    cfg.judge_endpoints = judge_endpoints(
        {name: "http://127.0.0.1:8000" for name in cfg.secondary_model_names}
    )
    cfg.apply_defaults(
        inference_instances=1,
        base_url="http://localhost:29999",
    )

    evaluator = ForgeEvaluator()
    evaluator.setup(
        autometric_runs_config=cfg,
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=tmp_path / "hf_ckpt",
        model_info=ModelInfo(model_class="sft", size="lite", username="u"),
    )
    return evaluator


class _FakeSubprocessManager:
    """Stand-in for SubprocessManager that emulates a forge-lite run by writing
    a multi-recipe results.json for every bundle found in the input directory."""

    instances: list["_FakeSubprocessManager"] = []

    def __init__(self, cmd, log_file=None, cwd=None, run_name="", env=None):
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        _FakeSubprocessManager.instances.append(self)

    def start(self):
        out = Path(self.cmd[self.cmd.index("-o") + 1])
        names = self.cmd[self.cmd.index("-a") + 1].split(",")
        attacks = {
            name: {
                "metrics": {
                    "asr": 0.5,
                    "blacklist_rate": 0.25,
                    "total": 10,
                    "evaluated": 10,
                }
            }
            for name in names
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"attacks": attacks}))

    def wait_for_completion(self):
        pass

    def is_successful(self):
        return True

    def log_error(self):
        pass

    def raise_for_exit_code(self):
        pass


@pytest.fixture(autouse=True)
def _reset_fake_sm():
    _FakeSubprocessManager.instances = []
    yield
    _FakeSubprocessManager.instances = []


class TestForgeEvaluatorSetup:
    def test_setup_creates_metrics_dir_and_resolves_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        assert evaluator.metrics_dir.is_dir()
        assert evaluator.forge_root.is_absolute()
        assert evaluator.recipes_dir == evaluator.forge_root / "recipes"
        assert evaluator.data_dir == evaluator.forge_root / "data"
        assert evaluator.recipes_dir.is_dir()


class TestForgeJudgeRuns:
    # secondary_runs() was folded into setup: recipes are grouped per judge in
    # _recipes_by_judge, and the judge models come from cfg.secondary_model_names.
    def test_all_recipes_under_default_judge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch, attacks="all")
        assert evaluator.autometric_runs_config.secondary_model_names == ["judge-a"]
        assert sorted(evaluator._recipes_by_judge["judge-a"]) == [
            "camo",
            "role_playing",
        ]

    def test_explicit_attacks_subset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch, attacks=["camo"])
        assert evaluator._recipes_by_judge == {"judge-a": ["camo"]}

    def test_missing_recipe_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        with pytest.raises(ValueError, match="not found"):
            _build_evaluator(tmp_path, monkeypatch, attacks=["camo", "nope"])

    def test_groups_split_judges(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        evaluator = _build_evaluator(
            tmp_path,
            monkeypatch,
            attacks="all",
            groups=[{"recipes": ["role_playing"], "judge": "judge-b"}],
        )
        by_model = {
            judge: sorted(recipes)
            for judge, recipes in evaluator._recipes_by_judge.items()
        }
        assert by_model == {"judge-a": ["camo"], "judge-b": ["role_playing"]}
        assert evaluator.autometric_runs_config.secondary_model_names == [
            "judge-a",
            "judge-b",
        ]


class TestForgeMetaOverride:
    def test_build_meta_override_omits_unset_params(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch, attack_model_name="m1")
        override = evaluator.autometric_runs_config.build_meta_override("camo")

        assert override["model"] == {
            "provider": "openai",
            "model": "m1",
            "base_url": "http://localhost:29999/v1",
            "api_key_env": "EMPTY",
            "timeout": 21600,
        }
        # Unset in config → omitted so the bundle's own values survive the merge.
        assert "temperature" not in override["model"]
        assert "max_tokens" not in override["model"]

        assert override["judge"] == {
            "provider": "openai",
            "model": "judge-a",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key_env": "EMPTY",
            "timeout": 21600,
        }
        assert "prompt_template" not in override["judge"]

    def test_build_meta_override_includes_config_params(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(
            tmp_path,
            monkeypatch,
            temperature=1.0,
            max_tokens=128,
            judge_prompt_template="builtin:reasoning",
        )
        override = evaluator.autometric_runs_config.build_meta_override("camo")
        assert override["model"]["temperature"] == 1.0
        assert override["model"]["max_tokens"] == 128
        assert override["judge"]["prompt_template"] == "builtin:reasoning"

    def test_build_meta_override_group_params_take_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(
            tmp_path,
            monkeypatch,
            temperature=1.0,
            groups=[
                {
                    "recipes": ["camo"],
                    "judge": "judge-b",
                    "temperature": 0.2,
                    "max_tokens": 64,
                    "judge_prompt_template": "builtin:strict",
                }
            ],
        )
        override = evaluator.autometric_runs_config.build_meta_override("camo")
        assert override["model"]["temperature"] == 0.2
        assert override["model"]["max_tokens"] == 64
        assert override["judge"]["model"] == "judge-b"
        assert override["judge"]["prompt_template"] == "builtin:strict"


class TestForgeCmdAndEnv:
    def test_build_cmd_includes_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(
            tmp_path, monkeypatch, workers=8, limit=10, combinations=3
        )
        override = tmp_path / "meta_override.json"
        results = tmp_path / "results.json"
        cmd = evaluator._build_cmd(["camo", "role_playing"], override, results)
        assert cmd[1:5] == ["-m", "forge_lite", "run", str(evaluator.recipes_dir)]
        assert cmd[5:7] == ["-o", str(results)]
        assert cmd[cmd.index("-a") + 1] == "camo,role_playing"
        assert cmd[cmd.index("--meta-override") + 1] == str(override)
        assert cmd[cmd.index("-w") + 1] == "8"
        assert cmd[cmd.index("-n") + 1] == "10"
        assert cmd[cmd.index("-c") + 1] == "3"
        # fail_on_runtime_error defaults True → flag present.
        assert "--fail-on-runtime-error" in cmd

    def test_build_cmd_omits_optional_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch, fail_on_runtime_error=False)
        cmd = evaluator._build_cmd(["camo"], tmp_path / "ov.json", tmp_path / "r.json")
        assert "--fail-on-runtime-error" not in cmd
        assert "-n" not in cmd
        assert "-c" not in cmd

    def test_build_env_sets_forge_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        env = evaluator._build_env()
        assert env["FORGE_LITE_DATA_DIR"] == str(evaluator.data_dir)
        assert env["FORGE_LITE_RECIPES_DIR"] == str(evaluator.recipes_dir)


class TestForgeResultsParsing:
    def test_parses_multi_recipe_attacks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps(
                {
                    "attacks": {
                        "camo": {
                            "metrics": {
                                "asr": 0.4,
                                "blacklist_rate": 0.1,
                                "total": 10,
                                "evaluated": 10,
                            }
                        },
                        "role_playing": {
                            "metrics": {
                                "asr": 0.6,
                                "blacklist_rate": 0.2,
                                "total": 5,
                                "evaluated": 5,
                            }
                        },
                    }
                }
            )
        )
        out = evaluator._parse_results(results, ["camo", "role_playing"])
        assert out == {
            "forge/camo/asr": pytest.approx(0.4),
            "forge/camo/br": pytest.approx(0.1),
            "forge/role_playing/asr": pytest.approx(0.6),
            "forge/role_playing/br": pytest.approx(0.2),
        }

    def test_parses_single_recipe_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps(
                {
                    "metadata": {"attack_name": "camo"},
                    "metrics": {
                        "asr": 0.3,
                        "blacklist_rate": 0.05,
                        "total": 10,
                        "evaluated": 10,
                    },
                }
            )
        )
        out = evaluator._parse_results(results, ["camo"])
        assert out == {
            "forge/camo/asr": pytest.approx(0.3),
            "forge/camo/br": pytest.approx(0.05),
        }

    def test_returns_none_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        assert evaluator._parse_results(tmp_path / "nope.json", ["camo"]) is None

    def test_returns_none_when_endpoint_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Dead attack/judge endpoint: FORGE_LITE reports asr=0 with 0 evaluated.
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps(
                {
                    "attacks": {
                        "camo": {
                            "metrics": {
                                "asr": 0.0,
                                "blacklist_rate": 0.0,
                                "total": 10,
                                "evaluated": 0,
                            }
                        }
                    }
                }
            )
        )
        assert evaluator._parse_results(results, ["camo"]) is None

    def test_returns_none_on_partial_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Partial failure (some runs unevaluated) is all-or-nothing → None.
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps(
                {
                    "attacks": {
                        "camo": {
                            "metrics": {
                                "asr": 0.4,
                                "blacklist_rate": 0.1,
                                "total": 10,
                                "evaluated": 7,
                            }
                        }
                    }
                }
            )
        )
        assert evaluator._parse_results(results, ["camo"]) is None


class TestForgeRunJudge:
    def test_run_judge_without_recipes_is_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # A judge that owns no recipes (e.g. handed a foreign model name)
        # short-circuits before validation and spawns nothing.
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        evaluator.run_on_model_name("unknown-judge")
        assert evaluator.all_metrics == {}
        assert _FakeSubprocessManager.instances == []

    def test_run_judge_runs_and_collects_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        monkeypatch.setattr(evaluator, "_validate_runtime", lambda: None)
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.forge.SubprocessManager",
            _FakeSubprocessManager,
        )

        evaluator.run_on_model_name("judge-a")

        assert len(_FakeSubprocessManager.instances) == 1
        assert _FakeSubprocessManager.instances[0].cwd == str(evaluator.forge_root)
        assert evaluator.get_metrics() == {
            "forge/camo/asr": pytest.approx(0.5),
            "forge/camo/br": pytest.approx(0.25),
            "forge/role_playing/asr": pytest.approx(0.5),
            "forge/role_playing/br": pytest.approx(0.25),
        }

    def test_restart_reuses_results_of_a_finished_judge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.forge.SubprocessManager",
            _FakeSubprocessManager,
        )
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        monkeypatch.setattr(evaluator, "_validate_runtime", lambda: None)
        evaluator.run_on_model_name("judge-a")
        computed = evaluator.get_metrics()

        evaluator.all_metrics = {}
        evaluator.run_on_model_name("judge-a")

        assert len(_FakeSubprocessManager.instances) == 1
        assert evaluator.get_metrics() == computed != {}

    def test_run_judge_aborts_on_nonzero_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        monkeypatch.setattr(evaluator, "_validate_runtime", lambda: None)

        class _CrashingSM(_FakeSubprocessManager):
            def start(self):
                pass  # produce no results.json

            def is_successful(self):
                return False

        _CrashingSM.exit_code = 1
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.forge.SubprocessManager", _CrashingSM
        )

        evaluator.log_file.write_text("log")

        evaluator.run_on_model_name("judge-a")

        assert evaluator._failed is True
        assert evaluator.get_metrics() == {}
        # save_cache defaults False → per-judge cache dirs wiped,
        # top-level files (autometrics.log) kept
        assert list(evaluator.metrics_dir.iterdir()) == [evaluator.log_file]

    def test_run_judge_aborts_on_dead_endpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        monkeypatch.setattr(evaluator, "_validate_runtime", lambda: None)

        class _DeadEndpointSM(_FakeSubprocessManager):
            def start(self):
                out = Path(self.cmd[self.cmd.index("-o") + 1])
                names = self.cmd[self.cmd.index("-a") + 1].split(",")
                attacks = {
                    name: {
                        "metrics": {
                            "asr": 0.0,
                            "blacklist_rate": 0.0,
                            "total": 10,
                            "evaluated": 0,
                        }
                    }
                    for name in names
                }
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"attacks": attacks}))

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.forge.SubprocessManager",
            _DeadEndpointSM,
        )

        evaluator.run_on_model_name("judge-a")

        assert evaluator._failed is True
        assert evaluator.get_metrics() == {}

    def test_run_judge_keeps_cache_on_failure_when_save_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch, save_cache=True)
        monkeypatch.setattr(evaluator, "_validate_runtime", lambda: None)

        class _DeadEndpointSM(_FakeSubprocessManager):
            def start(self):
                out = Path(self.cmd[self.cmd.index("-o") + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    json.dumps(
                        {
                            "attacks": {
                                "camo": {
                                    "metrics": {
                                        "asr": 0.0,
                                        "blacklist_rate": 0.0,
                                        "total": 10,
                                        "evaluated": 0,
                                    }
                                }
                            }
                        }
                    )
                )

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.forge.SubprocessManager",
            _DeadEndpointSM,
        )

        evaluator.run_on_model_name("judge-a")

        assert evaluator._failed is True
        assert evaluator.get_metrics() == {}
        group_dir = evaluator.metrics_dir / evaluator._safe_name("judge-a")
        assert (group_dir / "results.json").exists()

    def test_run_judge_short_circuits_after_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        monkeypatch.setattr(evaluator, "_validate_runtime", lambda: None)
        evaluator._failed = True

        called = {"validate": False}
        monkeypatch.setattr(
            evaluator, "_validate_runtime", lambda: called.__setitem__("validate", True)
        )

        evaluator.run_on_model_name("judge-b")
        assert called["validate"] is False
        assert _FakeSubprocessManager.instances == []


class TestForgeAfterRunAndJd:
    def test_after_run_invokes_jd_tar_and_symlink(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        evaluator.hf_checkpoint_path = (
            tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint"
        )

        called: dict[str, object] = {}

        def _tar(metrics_dir, artifacts_path, _name):
            called["tar"] = (metrics_dir, artifacts_path)
            return tmp_path / "tar.zst"

        def _symlink(tar_path, hf_path, _name):
            called["symlink"] = (tar_path, hf_path)

        monkeypatch.setattr(
            evaluator,
            "_send_metrics_to_jd",
            lambda metrics: called.setdefault("send", metrics),
        )
        monkeypatch.setattr(JD, "tar_metrics_dir", _tar)
        monkeypatch.setattr(JD, "symlink_metrics_for_transfer", _symlink)

        evaluator.after_run({"forge/camo/asr": 0.5})

        assert called["send"] == {"forge/camo/asr": 0.5}
        assert called["tar"] == (
            evaluator.metrics_dir,
            evaluator.artifacts_path,
        )
        assert called["symlink"] == (
            tmp_path / "tar.zst",
            evaluator.hf_checkpoint_path,
        )

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_send_metrics_to_jd_skips_without_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        evaluator.hf_checkpoint_path = None
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.send_metrics",
            lambda body, artifacts_path: pytest.fail(
                "must not send without a checkpoint"
            ),
        )

        evaluator._send_metrics_to_jd({"forge/camo/asr": 0.5})

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_send_metrics_to_jd_builds_payload(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        evaluator.hf_checkpoint_path = (
            tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint"
        )

        sent: dict[str, object] = {}
        monkeypatch.setattr(
            JD, "save_metrics_json", lambda body, step, metrics_dir: None
        )
        monkeypatch.setattr(
            JD,
            "send_metrics",
            lambda body, _artifacts: sent.update(body=body),
        )

        evaluator._send_metrics_to_jd({"forge/camo/asr": 0.5})

        body = sent["body"]
        assert body["shooter_name"] == "forge"
        assert body["model_class"] == "sft"
        assert body["name"] == "model"
        assert body["nfs_path"] == str(tmp_path / "model")
        assert body["metrics"] == {2: {"FORGE_CAMO_ASR": 0.5}}

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_send_metrics_to_jd_skips_when_no_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.send_metrics",
            lambda body, artifacts_path: pytest.fail("must not send empty metrics"),
        )

        evaluator._send_metrics_to_jd({})


class TestForgeValidateRuntime:
    def test_raises_when_runtime_python_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        # Default runtime_python points at an interpreter that does not exist.
        with pytest.raises(FileNotFoundError, match="runtime_python not found"):
            evaluator._validate_runtime()

    def test_raises_when_recipes_dir_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        evaluator.autometric_runs_config.runtime_python = sys.executable
        evaluator.recipes_dir = tmp_path / "no_recipes"
        with pytest.raises(FileNotFoundError, match="recipes dir not found"):
            evaluator._validate_runtime()

    def test_passes_when_runtime_and_recipes_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        evaluator.autometric_runs_config.runtime_python = sys.executable
        # recipes_dir exists (created by _build_evaluator) → validation passes.
        assert evaluator._validate_runtime() is None


class TestForgeUnzipAndFallback:
    def test_no_recipes_selected_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        with pytest.raises(ValueError, match="No FORGE recipes selected"):
            _build_evaluator(tmp_path, monkeypatch, recipes={})

    def test_parse_results_returns_none_on_unexpected_shape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        results = tmp_path / "results.json"
        results.write_text(json.dumps({"unexpected": "shape"}))
        assert evaluator._parse_results(results, ["camo"]) is None

    def test_unzip_forge_lite_extracts_archive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        archive = tmp_path / "forge_lite.tar.zst"
        archive.write_bytes(b"fake-archive")
        runtime_root = tmp_path / "forge_lite_runtime"
        runtime_root.mkdir()  # exists -> exercise the rmtree branch
        log_path = tmp_path / "unzip.log"
        log_path.write_text("stale")  # exists -> exercise the pre-run unlink branch

        monkeypatch.setattr(ForgeEvaluator, "ARCHIVE_PATH", archive)
        monkeypatch.setattr(ForgeEvaluator, "RUNTIME_ROOT", runtime_root)
        monkeypatch.setattr(ForgeEvaluator, "UNZIP_LOG_PATH", log_path)

        captured: dict[str, object] = {}

        class _UnzipSM:
            def __init__(self, cmd, log_file=None, cwd=None, run_name=""):
                captured["cmd"] = cmd

            def start(self):
                # Re-create the log so the post-run unlink branch runs too.
                log_path.write_text("run output")

            def wait_for_completion(self):
                pass

            def raise_for_exit_code(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.forge.SubprocessManager", _UnzipSM
        )

        evaluator = ForgeEvaluator()
        result = evaluator._unzip_forge_lite()

        assert result == runtime_root
        assert captured["cmd"][:3] == ["tar", "--zstd", "-xf"]
        assert archive.as_posix() in captured["cmd"]
        assert not log_path.exists()

    def test_unzip_forge_lite_clean_without_existing_target_or_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # No pre-existing target dir / log file: exercises the "skip" branches
        # (no rmtree, no pre/post unlink).
        archive = tmp_path / "forge_lite.tar.zst"
        archive.write_bytes(b"fake-archive")
        runtime_root = tmp_path / "forge_lite_clean"
        log_path = tmp_path / "missing.log"

        monkeypatch.setattr(ForgeEvaluator, "ARCHIVE_PATH", archive)
        monkeypatch.setattr(ForgeEvaluator, "RUNTIME_ROOT", runtime_root)
        monkeypatch.setattr(ForgeEvaluator, "UNZIP_LOG_PATH", log_path)

        class _UnzipSM:
            def __init__(self, cmd, log_file=None, cwd=None, run_name=""):
                pass

            def start(self):
                pass  # produce no log file

            def wait_for_completion(self):
                pass

            def raise_for_exit_code(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.forge.SubprocessManager", _UnzipSM
        )

        evaluator = ForgeEvaluator()
        assert evaluator._unzip_forge_lite() == runtime_root


@pytest.mark.usefixtures("scalar_ci_upload")
@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "single_recipe",
        "zero_total",
        "missing_result",
        "unevaluated",
        "error",
        "invalid_verdict",
        "duplicate",
        "different_parameters",
        "malformed",
    ],
)
def test_ci_groups_attack_parameters_by_original_prompt(tmp_path, monkeypatch, case):
    evaluator = _build_evaluator(
        tmp_path, monkeypatch, recipes={"camo": _DEFAULT_RECIPES["camo"]}
    )
    rows = [
        {
            "prompt": prompt,
            "parameters": {"variant": variant},
            "messages_sent": [{"role": "user", "content": prompt}],
            "judge_result": "unsafe" if prompt == "a" else "safe",
        }
        for prompt in ("a", "b")
        for variant in (0, 1)
    ]
    metrics = {"total": 4, "evaluated": 4}
    if case == "zero_total":
        metrics["total"] = 0
    elif case == "missing_result":
        rows.pop()
    elif case == "unevaluated":
        metrics["evaluated"] = 3
    elif case == "error":
        rows[0]["error"] = "judge failed"
    elif case == "invalid_verdict":
        rows[0]["judge_result"] = "ungraded"
    elif case == "duplicate":
        rows[1] = rows[0]
    elif case == "different_parameters":
        rows[-1]["parameters"] = {"variant": 2}
    elif case == "malformed":
        rows = [{}] * 4
    payload = {"metadata": {"attack_name": "camo"}, "results": rows, "metrics": metrics}
    data = payload if case == "single_recipe" else {"attacks": {"camo": payload}}
    directory = evaluator.metrics_dir / "judge-a"
    directory.mkdir()
    (directory / "results.json").write_text(json.dumps(data))
    monkeypatch.setattr(JD, "can_send_metrics", lambda **kwargs: True)
    monkeypatch.setattr(JD, "send_evaluator_metrics", lambda **kwargs: None)
    evaluator._send_metrics_to_jd({"forge/camo/asr": 0.5, "forge/camo/errors": 0})
    rows = [
        json.loads(line)
        for line in (evaluator.metrics_dir / "ci_samples.jsonl")
        .read_text()
        .splitlines()
    ]
    assert sorted(row["score"] for row in rows) == (
        [0, 1] if case in {"complete", "single_recipe"} else []
    )
    assert all(row["benchmark"] == "FORGE_CAMO_ASR" for row in rows)
