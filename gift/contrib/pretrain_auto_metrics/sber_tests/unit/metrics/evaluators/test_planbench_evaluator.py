import json
from pathlib import Path

import pytest
from node_runner.src.job_dispatcher import JD
from pydantic import ValidationError

from node_runner.src.config_model import (
    ModelInfoConfig as ModelInfo,
    PlanbenchEvaluatorConfig,
)
from node_runner.src.metrics.evaluators.planbench import PlanbenchEvaluator

pytestmark = pytest.mark.unit


def _build_evaluator(tmp_path: Path, **cfg_overrides) -> PlanbenchEvaluator:
    cfg = PlanbenchEvaluatorConfig(
        domains=cfg_overrides.pop("domains", ["blocksworld"]),
        planbench_root=cfg_overrides.pop(
            "planbench_root", str(tmp_path / "planbench-pam")
        ),
        **cfg_overrides,
    )
    cfg.apply_defaults(
        inference_instances=1,
        base_url="http://127.0.0.1:29999",
    )
    evaluator = PlanbenchEvaluator()
    evaluator.setup(
        autometric_runs_config=cfg,
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=tmp_path / "hf_ckpt",
        model_info=ModelInfo(model_class="sft", size="lite", username="u"),
    )
    return evaluator


def _seed_results(
    evaluator: PlanbenchEvaluator,
    *,
    domain: str,
    correct: int,
    total: int,
) -> Path:
    path = evaluator._result_path(domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    instances = [{"instance_id": i, "llm_correct": i < correct} for i in range(total)]
    path.write_text(json.dumps({"instances": instances}))
    return path


class TestPlanbenchEvaluatorConfig:
    def test_engine_must_end_with_chat(self):
        with pytest.raises(ValidationError):
            PlanbenchEvaluatorConfig(engine="my-model")

    def test_engine_chat_suffix_accepted(self):
        cfg = PlanbenchEvaluatorConfig(engine="my-model_chat")
        assert cfg.engine == "my-model_chat"

    def test_only_t1_task_is_allowed(self):
        with pytest.raises(ValidationError):
            PlanbenchEvaluatorConfig(task="t2")

    def test_domains_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            PlanbenchEvaluatorConfig(domains=[])

    def test_apply_defaults_sets_api_port(self):
        cfg = PlanbenchEvaluatorConfig()
        cfg.apply_defaults(inference_instances=1, base_url="http://127.0.0.1:12345")
        assert cfg.base_url == "http://127.0.0.1:12345/v1"

    def test_apply_defaults_caps_parallel_by_domain_count(self):
        cfg = PlanbenchEvaluatorConfig(domains=["blocksworld"])
        cfg.apply_defaults(inference_instances=8, base_url="http://127.0.0.1:29999")
        assert cfg.max_parallel_domains == 1

    def test_apply_defaults_scales_parallel_with_inference_instances(self):
        cfg = PlanbenchEvaluatorConfig(
            domains=["blocksworld", "logistics", "mystery_blocksworld"]
        )
        cfg.apply_defaults(inference_instances=2, base_url="http://127.0.0.1:29999")
        assert cfg.max_parallel_domains == 2

    def test_build_cli_args(self, tmp_path: Path):
        cfg = PlanbenchEvaluatorConfig(engine="m_chat", domains=["blocksworld"])
        work_dir = tmp_path / "run"
        assert cfg.build_cli_args(domain="blocksworld", work_dir=work_dir) == [
            "--work-dir",
            str(work_dir.resolve()),
            "--task",
            "t1",
            "--config",
            "blocksworld",
            "--engine",
            "m_chat",
        ]


class TestPlanbenchEvaluatorSetup:
    def test_setup_creates_metrics_dir_and_resolves_root(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator.metrics_dir.is_dir()
        assert evaluator.planbench_root.is_absolute()
        assert (
            evaluator._work_dir
            == evaluator.metrics_dir / PlanbenchEvaluator.WORK_SUBDIR
        )
        assert evaluator._runner == evaluator.planbench_root / "run_planbench.py"
        assert evaluator._val_dir == evaluator.planbench_root.joinpath(
            *PlanbenchEvaluator.VAL_SUBPATH
        )

    def test_resolve_bench_root_keeps_absolute(self, tmp_path: Path):
        absolute = tmp_path / "explicit_root"
        absolute.mkdir()
        assert PlanbenchEvaluator._resolve_bench_root(str(absolute)) == absolute

    def test_run_requires_base_url(self, tmp_path: Path):
        cfg = PlanbenchEvaluatorConfig(
            domains=["blocksworld"],
            planbench_root=str(tmp_path / "planbench-pam"),
        )

        with pytest.raises(ValueError, match="base_url must be set"):
            cfg.require_base_url()

    def test_resolve_bench_root_resolves_relative_against_repo_root(self):
        # Just check the resolution lands on an absolute path; the precise
        # repo root depends on the checkout layout.
        resolved = PlanbenchEvaluator._resolve_bench_root("contrib/plan-bench-pam")
        assert resolved.is_absolute()
        assert resolved.name == "plan-bench-pam"


class TestPlanbenchEvaluatorEnv:
    def test_build_env_sets_openai_and_val(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        env = evaluator._build_env("http://127.0.0.1:4242/v1")
        assert env["OPENAI_API_BASE"] == "http://127.0.0.1:4242/v1"
        assert env["OPENAI_API_KEY"] == "empty"
        assert env["VAL"].endswith("planner_tools/VAL")


class TestPlanbenchEvaluatorMetrics:
    def test_collect_metrics_computes_accuracy(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path, domains=["blocksworld"])
        _seed_results(evaluator, domain="blocksworld", correct=3, total=10)
        evaluator._collect_metrics()
        assert evaluator.get_metrics() == {
            "planbench/blocksworld/t1/accuracy": pytest.approx(0.3)
        }
        assert (evaluator.metrics_dir / "planbench_metrics.json").is_file()

    def test_collect_metrics_skips_unscored_domains(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path, domains=["blocksworld"])
        warnings: list[str] = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.planbench.Logger.warning",
            lambda message: warnings.append(message),
        )
        evaluator._collect_metrics()
        assert evaluator.get_metrics() == {}
        assert any("no results file" in w for w in warnings)

    def test_collect_metrics_skips_when_no_scored_instances(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path, domains=["blocksworld"])
        path = evaluator._result_path("blocksworld")
        path.parent.mkdir(parents=True, exist_ok=True)
        # No `llm_correct` keys → no scored instances after filtering.
        path.write_text(json.dumps({"instances": [{"instance_id": 1}]}))
        warnings: list[str] = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.planbench.Logger.warning",
            lambda message: warnings.append(message),
        )
        evaluator._collect_metrics()
        assert evaluator.get_metrics() == {}
        assert any("no scored instances" in w for w in warnings)

    def test_is_already_calculated_true_when_all_results_exist(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path, domains=["blocksworld", "logistics"])
        _seed_results(evaluator, domain="blocksworld", correct=1, total=2)
        _seed_results(evaluator, domain="logistics", correct=2, total=2)
        assert evaluator._is_already_calculated() is True

    def test_is_already_calculated_false_when_missing(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path, domains=["blocksworld", "logistics"])
        _seed_results(evaluator, domain="blocksworld", correct=1, total=2)
        assert evaluator._is_already_calculated() is False


class TestPlanbenchEvaluatorRun:
    def test_run_uses_configured_base_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)
        captured: dict[str, object] = {}

        class _FakePool:
            def __init__(
                self, *, cmds_to_run, max_parallel, autometrics_log_file, cwd, env
            ):
                captured["env"] = env

            def run_all(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.planbench.SubprocessPoolRunner",
            _FakePool,
        )

        evaluator.run()

        assert (
            captured["env"]["OPENAI_API_BASE"]
            == evaluator.autometric_runs_config.base_url
        )

    def test_run_skips_subprocess_when_results_are_cached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path, domains=["blocksworld"])
        _seed_results(evaluator, domain="blocksworld", correct=5, total=10)

        def _explode(*args, **kwargs):
            raise AssertionError("SubprocessPoolRunner should not be constructed")

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.planbench.SubprocessPoolRunner",
            _explode,
        )
        evaluator.run()
        assert evaluator.get_metrics() == {
            "planbench/blocksworld/t1/accuracy": pytest.approx(0.5)
        }

    def test_run_spawns_one_subprocess_per_domain(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path, domains=["blocksworld", "logistics"])
        captured: dict[str, object] = {}

        class _FakePool:
            def __init__(
                self, *, cmds_to_run, max_parallel, autometrics_log_file, cwd, env
            ):
                captured["cmds"] = cmds_to_run
                captured["max_parallel"] = max_parallel
                captured["cwd"] = cwd

            def run_all(self):
                for domain in captured["cmds"]:
                    _seed_results(evaluator, domain=domain, correct=1, total=2)

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.planbench.SubprocessPoolRunner",
            _FakePool,
        )
        evaluator.run()
        assert set(captured["cmds"]) == {"blocksworld", "logistics"}
        for cmd in captured["cmds"].values():
            assert "--config" in cmd
        assert evaluator.get_metrics() == {
            "planbench/blocksworld/t1/accuracy": pytest.approx(0.5),
            "planbench/logistics/t1/accuracy": pytest.approx(0.5),
        }

    def test_after_run_invokes_jd_tar_and_symlink(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)
        evaluator.hf_checkpoint_path = (
            tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint"
        )
        called: dict[str, object] = {}

        def _tar(metrics_dir, original_artifacts_path, _name):
            called["tar"] = (metrics_dir, original_artifacts_path)
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

        evaluator.after_run({"planbench/blocksworld/t1/accuracy": 0.5})

        assert called["send"] == {"planbench/blocksworld/t1/accuracy": 0.5}
        assert called["tar"] == (
            evaluator.metrics_dir,
            evaluator.artifacts_path,
        )
        assert called["symlink"] == (
            tmp_path / "tar.zst",
            evaluator.hf_checkpoint_path,
        )

    def test_run_skips_collection_when_every_domain_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path, domains=["blocksworld"])
        errors: list[str] = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.planbench.Logger.error",
            lambda message: errors.append(message),
        )

        class _FailingPool:
            def __init__(self, **kwargs):
                pass

            def run_all(self):
                # No result files written — simulates every domain failing.
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.planbench.SubprocessPoolRunner",
            _FailingPool,
        )
        evaluator.run()
        assert any("every domain failed" in e for e in errors)
        assert evaluator.get_metrics() == {}


class TestPlanbenchEvaluatorJD:
    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_send_metrics_to_jd_skips_without_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)
        evaluator.hf_checkpoint_path = None
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.send_metrics",
            lambda body, artifacts_path: pytest.fail(
                "must not send without a checkpoint"
            ),
        )

        evaluator._send_metrics_to_jd({"planbench/blocksworld/t1/accuracy": 0.5})

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_send_metrics_to_jd_builds_payload(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)
        evaluator.hf_checkpoint_path = (
            tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint"
        )
        saved: dict[str, object] = {}
        sent: dict[str, object] = {}
        monkeypatch.setattr(
            JD,
            "save_metrics_json",
            lambda body, step, metrics_dir: saved.update(
                body=body, step=step, metrics_dir=metrics_dir
            ),
        )
        monkeypatch.setattr(
            JD,
            "send_metrics",
            lambda body, _artifacts: sent.update(body=body),
        )

        evaluator._send_metrics_to_jd({"planbench/blocksworld/t1/accuracy": 0.5})

        body = sent["body"]
        assert body["shooter_name"] == "planbench"
        assert body["model_class"] == "sft"
        assert body["name"] == "model"
        assert body["metrics"] == {2: {"PLANBENCH_BLOCKSWORLD_T1_ACCURACY": 0.5}}
        assert saved["step"] == 2
        assert saved["metrics_dir"] == evaluator.metrics_dir


@pytest.mark.usefixtures("scalar_ci_upload")
@pytest.mark.parametrize("valid", [True, False])
def test_ci_only_exports_scored_planning_instances(tmp_path, monkeypatch, valid):
    evaluator = _build_evaluator(tmp_path, domains=["blocksworld"])
    path = evaluator._result_path("blocksworld")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "instances": [
                    {"instance_id": "a", "llm_correct": True},
                    {"instance_id": "b", "llm_correct": False if valid else None},
                ]
            }
        )
    )
    monkeypatch.setattr(JD, "can_send_metrics", lambda **kwargs: True)
    monkeypatch.setattr(JD, "send_evaluator_metrics", lambda **kwargs: None)
    key = f"planbench/blocksworld/{evaluator.autometric_runs_config.task}/accuracy"
    evaluator._send_metrics_to_jd({key: 0.5, "errors": 0})
    rows = [
        json.loads(line)
        for line in (evaluator.metrics_dir / "ci_samples.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [(row["benchmark"], row["score"]) for row in rows] == (
        [(key.replace("/", "_").upper(), 1), (key.replace("/", "_").upper(), 0)]
        if valid
        else []
    )
