import json
from pathlib import Path

import pytest

from node_runner.src.config_model import ModelInfoConfig as ModelInfo
from node_runner.src.config_model import Tau2EvaluatorConfig
from node_runner.src.job_dispatcher import JD
from node_runner.src.metrics.evaluators.tau2 import Tau2Evaluator
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit


def _build_evaluator(tmp_path: Path, **cfg_overrides) -> Tau2Evaluator:
    cfg = Tau2EvaluatorConfig(
        domains=cfg_overrides.pop("domains", ["airline"]),
        user_model_name="gpt-oss-120b",
        **cfg_overrides,
    )
    cfg.judge_endpoints = judge_endpoints({"gpt-oss-120b": "http://127.0.0.1:8000"})
    cfg.apply_defaults(
        inference_instances=1,
        base_url="http://localhost:29999",
    )
    evaluator = Tau2Evaluator()
    evaluator.setup(
        autometric_runs_config=cfg,
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=tmp_path / "hf_ckpt",
        model_info=ModelInfo(model_class="sft", size="lite", username="u"),
    )
    return evaluator


class TestTau2EvaluatorSetup:
    def test_setup_creates_metrics_dir_and_resolves_paths(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator.metrics_dir.is_dir()
        assert evaluator.tau2_root.is_absolute()
        assert evaluator.tau2_src == evaluator.tau2_root / "src"

    def test_resolve_bench_root_keeps_absolute(self, tmp_path: Path):
        absolute = tmp_path / "explicit_tau2_root"
        absolute.mkdir()
        assert Tau2Evaluator._resolve_bench_root(str(absolute)) == absolute

    def test_resolve_bench_root_resolves_relative_against_repo_root(self):
        relative = Path("custom/tau2-bench")
        assert (
            Tau2Evaluator._resolve_bench_root(str(relative))
            == Tau2Evaluator.PRETRAIN_ROOT / relative
        )


class TestTau2MeanRewardParsing:
    def test_returns_empty_when_no_simulations(self, tmp_path: Path):
        path = tmp_path / "results.json"
        path.write_text(json.dumps({"simulations": []}))
        assert Tau2Evaluator._parse_mean_reward_from_results(path, "airline") == {}

    def test_averages_reward_info_values(self, tmp_path: Path):
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                {
                    "simulations": [
                        {"reward_info": {"reward": 1.0}},
                        {"reward_info": {"reward": 0.0}},
                        {"reward_info": {"reward": 0.5}},
                    ]
                }
            )
        )
        out = Tau2Evaluator._parse_mean_reward_from_results(path, "retail")
        assert out == {"tau/retail/mean": pytest.approx(0.5)}

    def test_treats_missing_reward_info_as_zero(self, tmp_path: Path):
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                {
                    "simulations": [
                        {"reward_info": {"reward": 1.0}},
                        {},
                    ]
                }
            )
        )
        out = Tau2Evaluator._parse_mean_reward_from_results(path, "telecom")
        assert out == {"tau/telecom/mean": pytest.approx(0.5)}


class TestTau2EvaluatorRun:
    def test_run_collects_metrics_and_keeps_simulations_in_artifacts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path, domains=["airline", "retail"])

        captured: dict[str, dict[str, list[str]]] = {}

        class _FakePool:
            def __init__(self, *, cmds_to_run, max_parallel, autometrics_log_file, cwd):
                captured["cmds"] = cmds_to_run
                captured["max_parallel"] = max_parallel
                captured["cwd"] = cwd

            def run_all(self):
                for save_to in evaluator._save_to_by_domain.values():
                    sim_dir = Path(save_to)
                    sim_dir.mkdir(parents=True, exist_ok=True)
                    (sim_dir / "results.json").write_text(
                        json.dumps(
                            {
                                "simulations": [
                                    {"reward_info": {"reward": 1.0}},
                                    {"reward_info": {"reward": 0.0}},
                                ]
                            }
                        )
                    )

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.tau2.SubprocessPoolRunner", _FakePool
        )

        evaluator.run_on_model_name("gpt-oss-120b")

        assert set(captured["cmds"]) == {"airline", "retail"}
        # cmd shape is sane: python -m tau2.cli run ...
        for cmd in captured["cmds"].values():
            assert cmd[1:4] == ["-m", "tau2.cli", "run"]
        assert evaluator.get_metrics() == {
            "tau/airline/mean": pytest.approx(0.5),
            "tau/retail/mean": pytest.approx(0.5),
        }
        # simulation dirs live in artifacts metrics dir and survive the run
        for save_to in evaluator._save_to_by_domain.values():
            assert Path(save_to).is_relative_to(evaluator.metrics_dir)
            assert Path(save_to).is_dir()

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

        evaluator.after_run({"tau/airline/mean": 0.5})

        assert called["send"] == {"tau/airline/mean": 0.5}
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
        evaluator = _build_evaluator(tmp_path)
        evaluator.hf_checkpoint_path = None
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.send_metrics",
            lambda body, artifacts_path: pytest.fail(
                "must not send without a checkpoint"
            ),
        )

        evaluator._send_metrics_to_jd({"tau/airline/mean": 0.5})

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_send_metrics_to_jd_builds_payload(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path, num_trials=3)
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

        evaluator._send_metrics_to_jd({"tau/airline/mean": 0.5})

        body = sent["body"]
        assert body["shooter_name"] == "tau2"
        assert body["model_class"] == "sft"
        assert body["name"] == "model"
        assert body["nfs_path"] == str(tmp_path / "model")
        assert body["metrics"] == {
            2: {
                "TAU2_MEAN@3": {
                    "value": 0.5,
                    "submetrics": {"TAU_AIRLINE_MEAN": 0.5},
                }
            }
        }
        assert saved["step"] == 2
        assert saved["metrics_dir"] == evaluator.metrics_dir

    def test_run_warns_when_results_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)
        warnings: list[str] = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.tau2.Logger.warning",
            lambda msg: warnings.append(msg),
        )

        class _FakePool:
            def __init__(self, **kwargs):
                pass

            def run_all(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.tau2.SubprocessPoolRunner", _FakePool
        )

        evaluator.run_on_model_name("gpt-oss-120b")
        assert evaluator.all_metrics == {}
        assert any("results not found" in m for m in warnings)


class TestTau2EvaluatorJudges:
    def test_secondary_model_names_returns_user_model(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator.autometric_runs_config.secondary_model_names == [
            "gpt-oss-120b"
        ]

    def test_secondary_model_names_rejects_blank_user_model(self, tmp_path: Path):
        cfg = Tau2EvaluatorConfig(domains=["airline"], user_model_name="  ")
        with pytest.raises(ValueError, match="must not be empty"):
            _ = cfg.secondary_model_names

    def test_run_on_model_name_ignores_model_and_runs_all_domains(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # tau2 has a single user model, so run_on_model_name always fires the
        # full domain pool regardless of the name it is handed.
        evaluator = _build_evaluator(tmp_path)
        pools: list[dict] = []

        class _FakePool:
            def __init__(self, **kwargs):
                pools.append(kwargs)

            def run_all(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.tau2.SubprocessPoolRunner", _FakePool
        )
        evaluator.run_on_model_name("whatever-model")
        assert len(pools) == 1
        assert pools[0]["cwd"] == str(evaluator.tau2_src)
        assert set(pools[0]["cmds_to_run"]) == {"airline"}


class TestTau2Headline:
    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_domains_are_averaged_into_tau2_mean(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(
            tmp_path, domains=["airline", "retail", "telecom"], num_trials=3
        )
        evaluator.hf_checkpoint_path = (
            tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint"
        )
        sent: dict[str, object] = {}
        monkeypatch.setattr(
            JD, "save_metrics_json", lambda body, step, metrics_dir: None
        )
        monkeypatch.setattr(
            JD, "send_metrics", lambda body, _artifacts: sent.update(body=body)
        )

        evaluator._send_metrics_to_jd(
            {"tau/airline/mean": 0.5, "tau/retail/mean": 0.7, "tau/telecom/mean": 0.6}
        )

        assert sent["body"]["metrics"] == {
            2: {
                "TAU2_MEAN@3": {
                    "value": pytest.approx(0.6),
                    "submetrics": {
                        "TAU_AIRLINE_MEAN": 0.5,
                        "TAU_RETAIL_MEAN": 0.7,
                        "TAU_TELECOM_MEAN": 0.6,
                    },
                },
            }
        }


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "duplicate_task",
        "duplicate_trial",
        "no_trial",
        "missing_trial",
        "different_trials",
        "empty",
        "unknown_task",
    ],
)
def test_ci_requires_complete_task_trial_population(tmp_path, case):
    evaluator = _build_evaluator(tmp_path, num_trials=2)
    tasks = [{"id": "a"}, {"id": "b"}]
    rows = [
        {"task_id": uid, "trial": trial, "reward_info": {"reward": float(uid == "a")}}
        for uid in ("a", "b")
        for trial in (0, 1)
    ]
    if case == "duplicate_task":
        tasks.append(tasks[0])
    elif case == "duplicate_trial":
        rows.append(rows[0])
    elif case == "no_trial":
        rows[0]["trial"] = None
    elif case == "missing_trial":
        rows.pop()
    elif case == "different_trials":
        rows[-1]["trial"] = 2
    elif case == "empty":
        tasks, rows = [], []
    elif case == "unknown_task":
        rows[0]["task_id"] = "unknown"
    evaluator._save_to_by_domain = {"airline": str(tmp_path)}
    (tmp_path / "results.json").write_text(
        json.dumps({"tasks": tasks, "simulations": rows, "info": {"seed": 42}})
    )
    scores = evaluator._cache_scores(
        {"AIRLINE": 0.5, "MEAN": 0.5}, {"airline": "AIRLINE"}, "MEAN"
    )
    assert set(scores.records) == ({"AIRLINE", "MEAN"} if case == "complete" else set())
    if case == "complete":
        assert [row["score"] for row in scores.records["MEAN"]] == [1, 0]


def test_ci_missing_domain_path_withholds_evidence(tmp_path):
    evaluator = _build_evaluator(tmp_path)
    assert (
        evaluator._cache_scores(
            {"AIRLINE": 0.5}, {"airline": "AIRLINE"}, "MEAN"
        ).records
        == {}
    )
