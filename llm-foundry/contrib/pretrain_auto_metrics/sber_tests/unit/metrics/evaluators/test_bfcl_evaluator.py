import csv
import json
import shutil
from pathlib import Path

import pytest
from node_runner.src.job_dispatcher import JD

from node_runner.src.config_model import (
    BfclEvaluatorConfig,
    ModelInfoConfig as ModelInfo,
)
from node_runner.src.metrics.evaluators.bfcl import BfclEvaluator

pytestmark = pytest.mark.unit


def _build_evaluator(tmp_path: Path, **cfg_overrides) -> BfclEvaluator:
    cfg = BfclEvaluatorConfig.model_validate(
        {"model": "gigachat-FC", "test_category": "web_search", **cfg_overrides}
    )
    cfg.apply_defaults(
        inference_instances=1,
        base_url="http://127.0.0.1:29999",
    )
    evaluator = BfclEvaluator()
    evaluator.setup(
        autometric_runs_config=cfg,
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=tmp_path / "hf_ckpt",
        model_info=ModelInfo(model_class="sft", size="lite", username="u"),
    )
    return evaluator


class TestBfclEvaluatorSetup:
    def test_setup_creates_metrics_dir_and_resolves_root(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator.metrics_dir.is_dir()
        assert evaluator.bfcl_root.is_absolute()
        assert evaluator._result_dir.name == BfclEvaluator.RESULT_SUBDIR
        assert evaluator._score_dir.name == BfclEvaluator.SCORE_SUBDIR

    def test_resolve_bfcl_root_keeps_absolute(self, tmp_path: Path):
        absolute = tmp_path / "explicit_bfcl_root"
        absolute.mkdir()
        assert BfclEvaluator._resolve_bfcl_root(str(absolute)) == absolute


class TestBfclMetricParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("", None),
            ("N/A", None),
            ("85%", 0.85),
            ("$1,234", 1234.0),
            ("not-a-number", None),
            ("0.42", 0.42),
        ],
    )
    def test_parse_numeric_cell(self, raw: str, expected):
        assert BfclEvaluator._parse_numeric_cell(raw) == expected

    def test_metric_slug_normalizes(self):
        assert BfclEvaluator._metric_slug("Web Search (live)") == "web_search_live"
        assert BfclEvaluator._metric_slug("###") == "unknown"

    def test_row_to_bfcl_metrics_skips_excluded_columns(self):
        row = {
            "Rank": "1",
            "Model": "gigachat",
            "Overall Acc": "70.5%",
            "Total Cost ($)": "$1.50",
            "Live Simple AST": "0.42",
        }
        metrics = BfclEvaluator._row_to_bfcl_metrics(row)
        assert metrics == {
            "BFCL_OVERALL_ACC": 0.705,
            "BFCL_LIVE_SIMPLE_AST": 0.42,
        }

    def test_load_metrics_from_csv_picks_first_row(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        score_dir = tmp_path / "score"
        score_dir.mkdir()
        with open(score_dir / "data_overall.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Overall Acc"])
            writer.writeheader()
            writer.writerow({"Model": "gigachat", "Overall Acc": "0.8"})
            writer.writerow({"Model": "other", "Overall Acc": "0.5"})
        metrics = evaluator._load_metrics_from_csv(score_dir)
        assert metrics == {"BFCL_OVERALL_ACC": 0.8}

    def test_load_metrics_from_csv_missing_returns_empty(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator._load_metrics_from_csv(tmp_path / "missing") == {}


class TestBfclEnvAndRun:
    def test_run_requires_base_url(self):
        cfg = BfclEvaluatorConfig.model_validate(
            {"model": "gigachat-FC", "test_category": "web_search"}
        )
        with pytest.raises(ValueError, match="base_url must be set"):
            cfg.require_base_url()

    def test_build_bfcl_env_decodes_search_secrets(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        env = evaluator._build_bfcl_env("http://127.0.0.1:31234/v1")
        assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:31234/v1"
        assert env["OPENAI_API_KEY"] == "empty"
        assert env["OPENAI_TIMEOUT"] == "21600"
        assert env["SBER_SEARCH_URL"].startswith("https://")
        assert env["SBER_SEARCH_API_KEY"]  # decoded non-empty
        assert env["SBER_RLI_GATEWAY_URL"].startswith("https://")

    @pytest.mark.parametrize("encoder_exists", [True, False])
    def test_build_bfcl_env_sets_only_existing_memory_encoder(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        encoder_exists: bool,
    ):
        encoder_path = tmp_path / "sentence-transformers--all-MiniLM-L6-v2"
        if encoder_exists:
            encoder_path.mkdir()
        monkeypatch.setenv("PAM_MODELS_PATH", str(tmp_path))

        env = _build_evaluator(tmp_path)._build_bfcl_env("http://localhost/v1")

        assert env.get("BFCL_MEMORY_ENCODER_PATH") == (
            str(encoder_path) if encoder_exists else None
        )

    def test_run_uses_configured_base_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)
        captured_env: dict[str, str] = {}

        class _FakeSP:
            def __init__(self, cmd, log_file, cwd, run_name, env):
                captured_env.update(env)
                self.exit_code = 1

            def start(self):
                pass

            def wait_for_completion(self):
                pass

            def log_error(self):
                pass

            def is_successful(self):
                return False

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.SubprocessManager", _FakeSP
        )

        evaluator.run()

        assert (
            captured_env["OPENAI_BASE_URL"] == evaluator.autometric_runs_config.base_url
        )

    def test_run_invokes_generate_then_evaluate_then_loads_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)
        score_dir = evaluator._score_dir
        score_dir.mkdir(parents=True, exist_ok=True)

        calls: list[str] = []

        class _FakeSP:
            def __init__(self, cmd, log_file, cwd, run_name, env):
                calls.append(run_name)
                self.exit_code = 0

            def start(self):
                pass

            def wait_for_completion(self):
                pass

            def log_error(self):
                pass

            def is_successful(self):
                return True

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.SubprocessManager", _FakeSP
        )

        with open(score_dir / "data_overall.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Overall Acc"])
            writer.writeheader()
            writer.writerow({"Model": "gigachat", "Overall Acc": "0.42"})

        evaluator.run()

        assert calls == ["bfcl_generate", "bfcl_evaluate"]
        assert evaluator.get_metrics() == {"BFCL_OVERALL_ACC": 0.42}
        saved = json.loads((evaluator.metrics_dir / "bfcl_metrics.json").read_text())
        assert saved == {"BFCL_OVERALL_ACC": 0.42}

    def test_run_aborts_on_failed_generate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)

        calls: list[str] = []

        class _FakeSP:
            def __init__(self, cmd, log_file, cwd, run_name, env):
                calls.append(run_name)
                self.exit_code = 1

            def start(self):
                pass

            def wait_for_completion(self):
                pass

            def log_error(self):
                pass

            def is_successful(self):
                return False

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.SubprocessManager", _FakeSP
        )
        evaluator.run()
        assert calls == ["bfcl_generate"]
        assert evaluator.all_metrics == {}

    def test_run_aborts_on_failed_evaluate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)

        calls: list[str] = []

        class _FakeSP:
            def __init__(self, cmd, log_file, cwd, run_name, env):
                calls.append(run_name)
                # generate succeeds, evaluate fails
                self._successful = run_name == "bfcl_generate"
                self.exit_code = 0 if self._successful else 3

            def start(self):
                pass

            def wait_for_completion(self):
                pass

            def log_error(self):
                pass

            def is_successful(self):
                return self._successful

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.SubprocessManager", _FakeSP
        )
        evaluator.run()
        assert calls == ["bfcl_generate", "bfcl_evaluate"]
        assert evaluator.all_metrics == {}

    def test_run_warns_when_csv_yields_no_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)
        score_dir = evaluator._score_dir
        score_dir.mkdir(parents=True, exist_ok=True)

        class _FakeSP:
            def __init__(self, *args, **kwargs):
                self.exit_code = 0

            def start(self):
                pass

            def wait_for_completion(self):
                pass

            def log_error(self):
                pass

            def is_successful(self):
                return True

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.SubprocessManager", _FakeSP
        )

        # CSV with a row whose Model is empty -> filtered out -> no rows -> empty.
        with open(score_dir / "data_overall.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Overall Acc"])
            writer.writeheader()
            writer.writerow({"Model": "", "Overall Acc": "0.8"})

        warnings: list[str] = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.Logger.warning",
            lambda message: warnings.append(message),
        )

        evaluator.run()
        assert evaluator.all_metrics == {}
        assert any("No numeric metrics parsed" in m for m in warnings)


class TestBfclRowToBfclMetrics:
    def test_skips_columns_with_unparseable_values(self):
        out = BfclEvaluator._row_to_bfcl_metrics(
            {"Model": "x", "Overall Acc": "0.5", "Bad Col": "N/A"}
        )
        # "Model" is excluded; "Bad Col" yields None (N/A) so it's skipped
        assert out == {"BFCL_OVERALL_ACC": 0.5}


class TestBfclLoadMetricsFromCsv:
    def test_returns_empty_when_all_rows_lack_model(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        score_dir = tmp_path / "score"
        score_dir.mkdir()
        with open(score_dir / "data_overall.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Overall Acc"])
            writer.writeheader()
            writer.writerow({"Model": "", "Overall Acc": "0.8"})
        assert evaluator._load_metrics_from_csv(score_dir) == {}


class TestBfclAfterRun:
    def test_after_run_invokes_jd_and_tar_and_symlink(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)
        evaluator.all_metrics = {"BFCL_OVERALL_ACC": 0.55}
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
            evaluator, "_send_metrics_to_jd", lambda m: called.setdefault("send", m)
        )
        monkeypatch.setattr(JD, "tar_metrics_dir", _tar)
        monkeypatch.setattr(JD, "symlink_metrics_for_transfer", _symlink)

        evaluator.after_run(evaluator.all_metrics)

        assert called["send"] == {"BFCL_OVERALL_ACC": 0.55}
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

        evaluator._send_metrics_to_jd({"BFCL_OVERALL_ACC": 0.42})

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

        evaluator._send_metrics_to_jd(
            {
                "BFCL_OVERALL_ACC": 0.55,
                "BFCL_WEB_SEARCH_ACC": 0.5,
                # Mixed-case key with slash exercises the upper+replace branch.
                "bfcl/live/simple_ast": 0.42,
            }
        )

        body = sent["body"]
        assert body["shooter_name"] == "bfcl"
        assert body["model_class"] == "sft"
        assert body["name"] == "model"
        assert body["nfs_path"] == str(tmp_path / "model")
        # Confirm both key shapes were normalized to upper case keys for JD.
        assert body["metrics"] == {
            2: {
                "BFCL_OVERALL_ACC": {
                    "value": 0.55,
                    "submetrics": {
                        "BFCL_WEB_SEARCH_ACC_SBER": 0.5,
                        "BFCL_LIVE_SIMPLE_AST": 0.42,
                    },
                },
            }
        }
        assert saved["step"] == 2
        assert saved["metrics_dir"] == evaluator.metrics_dir


class TestBfclFlatMetrics:
    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_overall_and_categories_have_separate_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)
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
            {
                "BFCL_OVERALL_ACC": 0.6,
                "BFCL_LIVE_ACC": 0.7,
                "BFCL_WEB_SEARCH_ACC": 0.5,
                "BFCL_FORMAT_SENSITIVITY_MAX_DELTA": 0.03,
            }
        )

        assert sent["body"]["metrics"] == {
            2: {
                "BFCL_OVERALL_ACC": {
                    "value": 0.6,
                    "submetrics": {
                        "BFCL_LIVE_ACC": 0.7,
                        "BFCL_WEB_SEARCH_ACC_SBER": 0.5,
                        "BFCL_FORMAT_SENSITIVITY_MAX_DELTA": 0.03,
                    },
                },
            }
        }


class TestBfclPassAtKFormula:
    @pytest.mark.parametrize(
        ("n", "c", "k", "expected"),
        [
            (4, 4, 1, 1.0),  # all passed
            (4, 0, 2, 0.0),  # none passed
            (4, 1, 2, 0.5),  # 1 - C(3,2)/C(4,2) = 1 - 3/6
            (3, 1, 3, 1.0),  # k=n, c>=1 → n-c < k
            (4, 2, 1, 0.5),  # 1 - C(2,1)/C(4,1) = 1 - 2/4
            (32, 0, 32, 0.0),  # never passed, k=n
            (32, 32, 1, 1.0),  # always passed
        ],
    )
    def test_unbiased_pass_at_k(self, n, c, k, expected):
        assert BfclEvaluator._unbiased_pass_at_k(n, c, k) == pytest.approx(expected)


class TestBfclCollectIds:
    def test_collect_task_ids_from_result_files(self, tmp_path: Path):
        subdir = tmp_path / "model"
        subdir.mkdir()
        result_file = subdir / "BFCL_v4_simple_ast_result.json"
        result_file.write_text(
            '{"id": "simple_ast_0"}\n\n  \n{"metadata": true}\n{"id": "simple_ast_1"}\n'
        )
        (subdir / "unrelated_result.json").write_text("not JSON")
        out = BfclEvaluator._collect_task_ids(tmp_path)
        assert out == {"simple_ast": {"simple_ast_0", "simple_ast_1"}}

    def test_collect_failed_ids_skips_header(self, tmp_path: Path):
        subdir = tmp_path / "model"
        subdir.mkdir()
        score_file = subdir / "BFCL_v4_simple_ast_score.json"
        score_file.write_text(
            '{"accuracy": 0.5, "correct_count": 1, "total_count": 2}\n'
            '\n{"metadata": true}\n'
            '{"id": "simple_ast_1", "valid": false}\n'
        )
        (subdir / "unrelated_score.json").write_text("not JSON")
        out = BfclEvaluator._collect_failed_ids(tmp_path)
        assert out == {"simple_ast": {"simple_ast_1"}}


class TestBfclMultiRunConfig:
    def test_pass_at_k_rejects_k_greater_than_num_runs(self):
        with pytest.raises(Exception, match="pass_at_k"):
            BfclEvaluatorConfig.model_validate(
                {"model": "m", "test_category": "c", "num-runs": 4, "pass-at-k": [8]}
            )

    def test_defaults_preserve_single_run(self):
        cfg = BfclEvaluatorConfig.model_validate({"model": "m", "test_category": "c"})
        assert cfg.num_runs == 1
        assert cfg.max_parallel_runs == 1
        assert cfg.pass_at_k == []

    def test_num_runs_excluded_from_bash_list(self):
        cfg = BfclEvaluatorConfig.model_validate(
            {"model": "m", "test_category": "c", "num-runs": 4}
        )
        bash = cfg.to_bash_list()
        assert "--num-runs" not in bash
        assert "--max-parallel-runs" not in bash
        assert "--pass-at-k" not in bash


def _write_synthetic_run(
    result_dir: Path,
    score_dir: Path,
    category: str,
    all_task_ids: list[str],
    failed_ids: list[str],
) -> None:
    """Write synthetic BFCL result/score/CSV files for one run."""
    rd = result_dir / "model"
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / f"BFCL_v4_{category}_result.json", "w") as f:
        for tid in all_task_ids:
            f.write(json.dumps({"id": tid}) + "\n")

    sd = score_dir / "model"
    sd.mkdir(parents=True, exist_ok=True)
    correct = len(all_task_ids) - len(failed_ids)
    with open(sd / f"BFCL_v4_{category}_score.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "accuracy": correct / len(all_task_ids),
                    "correct_count": correct,
                    "total_count": len(all_task_ids),
                }
            )
            + "\n"
        )
        for fid in failed_ids:
            f.write(json.dumps({"id": fid, "valid": False}) + "\n")

    col_name = category.replace("_", " ").title()
    with open(score_dir / "data_overall.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", col_name])
        writer.writeheader()
        writer.writerow(
            {
                "Model": "test_model",
                col_name: str(correct / len(all_task_ids)),
            }
        )


class TestBfclMultiRun:
    @pytest.mark.parametrize("empty_result_file", [False, True])
    def test_run_multi_warns_without_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, empty_result_file: bool
    ):
        evaluator = _build_evaluator(tmp_path, **{"num-runs": 2})
        if empty_result_file:
            result_dir = evaluator._result_dir / "run_0"
            result_dir.mkdir(parents=True)
            (result_dir / "BFCL_v4_simple_python_result.json").write_text("")

        class _FakePool:
            def __init__(self, **kwargs):
                pass

            def run_all(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.SubprocessPoolRunner", _FakePool
        )
        warnings: list[str] = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.Logger.warning", warnings.append
        )

        evaluator.run()

        assert evaluator.get_metrics() == {}
        assert "No metrics aggregated from BFCL multi-run" in warnings
        assert not (evaluator.metrics_dir / "bfcl_metrics.json").exists()

    def test_run_multi_aggregates_mean_std_and_pass_at_k(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        num_runs = 4
        evaluator = _build_evaluator(
            tmp_path, **{"num-runs": num_runs, "pass-at-k": [1]}
        )
        tasks = ["task_0", "task_1", "task_2", "task_3"]
        # Run 0: task_2 fails → 3/4
        # Run 1: task_1, task_2 fail → 2/4
        # Run 2: task_0, task_1, task_2 fail → 1/4
        # Run 3: task_0, task_1, task_2 fail → 1/4
        failures = [
            ["task_2"],
            ["task_1", "task_2"],
            ["task_0", "task_1", "task_2"],
            ["task_0", "task_1", "task_2"],
        ]

        for i in range(num_runs):
            _write_synthetic_run(
                evaluator._result_dir / f"run_{i}",
                evaluator._score_dir / f"run_{i}",
                "simple_python",
                tasks,
                failures[i],
            )

        class _FakePool:
            def __init__(self, **kwargs):
                pass

            def run_all(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.SubprocessPoolRunner", _FakePool
        )

        evaluator.run()
        metrics = evaluator.get_metrics()

        # mean = (0.75 + 0.5 + 0.25 + 0.25) / 4 = 0.4375
        assert metrics["bfcl/simple_python/mean"] == pytest.approx(0.4375)
        # pstdev of [0.75, 0.5, 0.25, 0.25]
        assert metrics["bfcl/simple_python/std"] == pytest.approx(0.2074, abs=1e-3)

        # pass@1: task_0 c=2 → 0.5, task_1 c=1 → 0.25, task_2 c=0 → 0.0, task_3 c=4 → 1.0
        # avg = 0.4375
        assert metrics["bfcl/simple_python/pass@1"] == pytest.approx(0.4375)

        # pass@4: task_0 → 1.0, task_1 → 1.0, task_2 → 0.0, task_3 → 1.0
        # avg = 0.75
        assert metrics["bfcl/simple_python/pass@4"] == pytest.approx(0.75)

        # bfcl_metrics.json written
        assert (evaluator.metrics_dir / "bfcl_metrics.json").is_file()

    def test_non_live_simple_ast_averages_per_language_pass_at_k(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        num_runs = 2
        evaluator = _build_evaluator(
            tmp_path, **{"num-runs": num_runs, "pass-at-k": [1]}
        )
        # simple_python: 4 tasks, all pass both runs      -> pass@1 = 1.0
        # simple_java:   1 task,  fails both runs          -> pass@1 = 0.0
        # simple_javascript: 1 task, passes run 0 only     -> pass@1 = 0.5
        per_cat = {
            "simple_python": (
                ["py_0", "py_1", "py_2", "py_3"],
                [[], []],
            ),
            "simple_java": (["java_0"], [["java_0"], ["java_0"]]),
            "simple_javascript": (["js_0"], [[], ["js_0"]]),
        }
        for i in range(num_runs):
            for cat, (tasks, failures) in per_cat.items():
                _write_synthetic_run(
                    evaluator._result_dir / f"run_{i}",
                    evaluator._score_dir / f"run_{i}",
                    cat,
                    tasks,
                    failures[i],
                )

        class _FakePool:
            def __init__(self, **kwargs):
                pass

            def run_all(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.SubprocessPoolRunner", _FakePool
        )

        evaluator.run()
        metrics = evaluator.get_metrics()

        # Per-language pass@1: python=1.0, java=0.0, javascript=0.5.
        # Macro average = (1.0 + 0.0 + 0.5) / 3 = 0.5.
        assert metrics["bfcl/non_live_simple_ast/pass@1"] == pytest.approx(0.5)
        # pass@2 (=n): passes if c>=1. python=1.0, java=0.0, javascript=1.0.
        # Macro average = 2 / 3.
        assert metrics["bfcl/non_live_simple_ast/pass@2"] == pytest.approx(2 / 3)

    def test_live_acc_pools_tasks_across_live_ast_categories(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        num_runs = 2
        evaluator = _build_evaluator(
            tmp_path, **{"num-runs": num_runs, "pass-at-k": [1]}
        )
        per_cat = {
            "live_simple": (["ls_0", "ls_1", "ls_2"], [[], []]),
            "live_multiple": (["lm_0"], [["lm_0"], ["lm_0"]]),
            "live_parallel": (["lp_0"], [[], []]),
            "live_parallel_multiple": (["lpm_0"], [[], []]),
        }
        for i in range(num_runs):
            for cat, (tasks, failures) in per_cat.items():
                _write_synthetic_run(
                    evaluator._result_dir / f"run_{i}",
                    evaluator._score_dir / f"run_{i}",
                    cat,
                    tasks,
                    failures[i],
                )

        class _FakePool:
            def __init__(self, **kwargs):
                pass

            def run_all(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.SubprocessPoolRunner", _FakePool
        )

        evaluator.run()
        metrics = evaluator.get_metrics()

        # Pooled counts over 6 tasks: c = [2,2,2, 0, 2, 2]. n=2.
        # pass@1: five tasks with c=2 -> 1.0, one with c=0 -> 0.0. avg = 5/6.
        assert metrics["bfcl/live_acc/pass@1"] == pytest.approx(5 / 6)

    def test_memory_acc_averages_per_backend_pass_at_k(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # memory_acc is a macro slug over the three memory backends.
        num_runs = 2
        evaluator = _build_evaluator(
            tmp_path, **{"num-runs": num_runs, "pass-at-k": [1]}
        )
        # kv: 1 task passes both -> pass@1=1.0
        # vector: 1 task fails both -> pass@1=0.0
        # rec_sum: 1 task passes run 0 only -> pass@1=0.5
        per_cat = {
            "memory_kv": (["kv_0"], [[], []]),
            "memory_vector": (["vec_0"], [["vec_0"], ["vec_0"]]),
            "memory_rec_sum": (["rs_0"], [[], ["rs_0"]]),
        }
        for i in range(num_runs):
            for cat, (tasks, failures) in per_cat.items():
                _write_synthetic_run(
                    evaluator._result_dir / f"run_{i}",
                    evaluator._score_dir / f"run_{i}",
                    cat,
                    tasks,
                    failures[i],
                )

        class _FakePool:
            def __init__(self, **kwargs):
                pass

            def run_all(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.SubprocessPoolRunner", _FakePool
        )

        evaluator.run()
        metrics = evaluator.get_metrics()

        # Macro average of (1.0, 0.0, 0.5) = 0.5.
        assert metrics["bfcl/memory_acc/pass@1"] == pytest.approx(0.5)

    def test_single_run_does_not_use_pool(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(tmp_path)

        calls: list[str] = []

        class _FakeSP:
            def __init__(self, cmd, log_file, cwd, run_name, env):
                calls.append(run_name)
                self.exit_code = 0

            def start(self):
                pass

            def wait_for_completion(self):
                pass

            def log_error(self):
                pass

            def is_successful(self):
                return True

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.bfcl.SubprocessManager", _FakeSP
        )

        score_dir = evaluator._score_dir
        score_dir.mkdir(parents=True, exist_ok=True)
        with open(score_dir / "data_overall.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Overall Acc"])
            writer.writeheader()
            writer.writerow({"Model": "m", "Overall Acc": "0.42"})

        evaluator.run()

        assert calls == ["bfcl_generate", "bfcl_evaluate"]
        assert evaluator.get_metrics() == {"BFCL_OVERALL_ACC": 0.42}


def test_ci_reconstructs_weighted_web_search_and_leaf_accuracy(tmp_path):
    evaluator = _build_evaluator(tmp_path)
    model = evaluator.autometric_runs_config.model.replace("/", "_")
    scores_dir, results_dir = (
        evaluator._score_dir / model,
        evaluator._result_dir / model,
    )
    scores_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    for category, ids, failed in [
        ("web_search_base", ["a", "b"], ["b"]),
        ("web_search_no_snippet", ["c", "d", "e", "f"], []),
        ("multiple", ["a", "b"], ["b"]),
    ]:
        rows = [
            {"total_count": len(ids), "correct_count": len(ids) - len(failed)},
            *({"id": uid} for uid in failed),
        ]
        (scores_dir / f"BFCL_v4_{category}_score.json").write_text(
            "\n".join(map(json.dumps, rows))
        )
        (results_dir / f"BFCL_v4_{category}_result.json").write_text(
            "\n".join(json.dumps({"id": uid}) for uid in ids)
        )
    web_key = (
        f"BFCL_WEB_SEARCH_ACC_{evaluator.autometric_runs_config.web_search.upper()}"
    )
    scores = evaluator._cache_scores(
        {web_key: 0.75, "BFCL_NON_LIVE_MULTIPLE_AST": 0.5, "BFCL_OVERALL_ACC": 0.75}
    )
    assert {
        key: [row["score"] for row in rows] for key, rows in scores.records.items()
    } == {
        web_key: [1, 0, 1, 1, 1, 1],
        "BFCL_NON_LIVE_MULTIPLE_AST": [1, 0],
    }
    assert {row["category"]: row["weight"] for row in scores.records[web_key]} == {
        "web_search_base": 0.5,
        "web_search_no_snippet": 0.5,
    }


@pytest.mark.parametrize(
    ("ids", "failures", "total", "correct", "valid"),
    [
        (["a", "b"], ["b"], 2, 1, True),
        ([], [], 0, 0, False),
        (["a", "a"], [], 2, 2, False),
        (["a", "b"], ["b", "b"], 2, 0, False),
        (["a", "b"], ["c"], 2, 1, False),
        (["a", "b"], [], 3, 2, False),
        (["a", "b"], [], 2, 1, False),
    ],
)
def test_category_samples_reconstructs_validated_accuracy(
    tmp_path, ids, failures, total, correct, valid
):
    scores = tmp_path / "scores" / "model"
    results = tmp_path / "results" / "model"
    scores.mkdir(parents=True)
    results.mkdir(parents=True)
    rows = [
        {"total_count": total, "correct_count": correct},
        *({"id": uid} for uid in failures),
    ]
    (scores / "BFCL_v4_multiple_score.json").write_text(
        "\n\n".join(map(json.dumps, rows))
    )
    (results / "BFCL_v4_multiple_result.json").write_text(
        "\n\n".join(json.dumps({"id": uid}) for uid in ids)
    )
    (scores / "BFCL_v4_format_sensitivity_score.json").write_text("ignored")
    (scores / "BFCL_v4_parallel_score.json").write_text("invalid json")
    assert BfclEvaluator._category_samples(scores.parent, results.parent) == (
        {"multiple": [("a", 1.0), ("b", 0.0)]} if valid else {}
    )


def test_collect_task_ids_ignores_prerequisites(tmp_path):
    for category in ("multiple", "multiple_prereq"):
        (tmp_path / f"BFCL_v4_{category}_result.json").write_text('{"id": "a"}\n')
    assert BfclEvaluator._collect_task_ids(tmp_path) == {"multiple": {"a"}}


def test_weights_follow_bfcl_tree_and_require_live_population():
    assert "Overall Acc" not in BfclEvaluator._metric_weights({})
    categories = {
        "live_simple": [1] * 4,
        "live_multiple": [1] * 2,
        "live_parallel": [1],
        "live_parallel_multiple": [1],
    }
    weights = BfclEvaluator._metric_weights(categories)
    assert weights["Live Acc"] == {
        "live_simple": 0.5,
        "live_multiple": 0.25,
        "live_parallel": 0.125,
        "live_parallel_multiple": 0.125,
    }
    assert sum(weights["Overall Acc"].values()) == pytest.approx(1)
    assert weights["Overall Acc"]["live_simple"] == pytest.approx(0.05)
    assert weights["Overall Acc"]["simple_python"] == pytest.approx(1 / 120)
    assert weights["Overall Acc"]["multi_turn_base"] == pytest.approx(0.075)
    assert "Format Sensitivity" not in weights


@pytest.fixture
def multi_run_ci_evaluator(tmp_path):
    evaluator = _build_evaluator(
        tmp_path, model="model", **{"num-runs": 2, "pass-at-k": [1]}
    )
    categories = {
        "simple_python": (["py0", "py1", "py2", "py3"], [[], []]),
        "simple_java": (["java0", "java1"], [["java0", "java1"], ["java0", "java1"]]),
        "simple_javascript": (["js0", "js1"], [[], ["js0", "js1"]]),
        "live_simple": ([f"ls{i}" for i in range(6)], [[], []]),
        "live_multiple": (["lm0", "lm1"], [["lm0", "lm1"], ["lm0", "lm1"]]),
        "live_parallel": (["lp0", "lp1"], [[], []]),
        "live_parallel_multiple": (["lpm0", "lpm1"], [[], []]),
        "web_search_base": (["wb0", "wb1"], [["wb1"], ["wb0"]]),
        "web_search_no_snippet": (["wn0", "wn1", "wn2", "wn3"], [[], []]),
    }
    for index in range(2):
        score_dir = evaluator._score_dir / f"run_{index}"
        for category, (ids, failures) in categories.items():
            _write_synthetic_run(
                evaluator._result_dir / f"run_{index}",
                score_dir,
                category,
                ids,
                failures[index],
            )
        with (score_dir / "data_overall.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "Model",
                    "Non-Live Simple AST",
                    "Live Acc",
                    "Web Search Acc",
                    "Format Sensitivity Max Delta",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "Model": "model",
                    "Non-Live Simple AST": (2 - index) / 3,
                    "Live Acc": 5 / 6,
                    "Web Search Acc": 0.75,
                    "Format Sensitivity Max Delta": 0.2,
                }
            )
    return evaluator


def _reconstructed_score(rows):
    if "category" not in rows[0]:
        return sum(row["score"] for row in rows) / len(rows)
    categories = {row["category"] for row in rows}
    return sum(
        next(row["weight"] for row in rows if row["category"] == category)
        * sum(row["score"] for row in rows if row["category"] == category)
        / sum(row["category"] == category for row in rows)
        for category in categories
    )


def test_multi_run_ci_reconstructs_means_pass_at_k_and_weighted_composites(
    multi_run_ci_evaluator, monkeypatch
):
    evaluator = multi_run_ci_evaluator
    evaluator.hf_checkpoint_path = (
        evaluator.artifacts_path / "model/ep1_ba2/artifacts/hf_checkpoint"
    )
    metrics = evaluator._aggregate_multi_run(
        [evaluator._result_dir / f"run_{i}" for i in range(2)],
        [evaluator._score_dir / f"run_{i}" for i in range(2)],
    )
    sent = {}
    monkeypatch.setattr(JD, "can_send_metrics", lambda **kwargs: True)
    monkeypatch.setattr(JD, "get_shooter_version", lambda path: "bfcl-test")
    monkeypatch.setattr(
        JD, "send_evaluator_metrics_with_scores", lambda **kwargs: sent.update(kwargs)
    )
    evaluator._send_metrics_to_jd(metrics)
    scores = sent["scores"]
    values = {key: _reconstructed_score(rows) for key, rows in scores.records.items()}
    assert set(values) == {
        key
        for key in scores.metrics
        if "_STD" not in key and "FORMAT_SENSITIVITY" not in key
    }
    assert values == pytest.approx({key: scores.metrics[key] for key in values})
    assert values["BFCL_NON_LIVE_SIMPLE_AST_MEAN"] == pytest.approx(0.5)
    assert values["BFCL_NON_LIVE_SIMPLE_AST_PASS@2"] == pytest.approx(2 / 3)
    assert values["BFCL_LIVE_ACC_PASS@2"] == pytest.approx(5 / 6)
    web = evaluator.autometric_runs_config.web_search.upper()
    assert values[f"BFCL_WEB_SEARCH_ACC_MEAN_{web}"] == pytest.approx(0.75)
    assert values[f"BFCL_WEB_SEARCH_ACC_PASS@2_{web}"] == 1
    assert values["BFCL_OVERALL_ACC_PASS@2"] == pytest.approx(
        0.1 * (2 / 3) + 0.1 * (5 / 6) + 0.4
    )
    assert len(scores.records[f"BFCL_WEB_SEARCH_ACC_PASS@2_{web}"]) == 6


@pytest.mark.parametrize(
    "case",
    ["missing_score", "missing_run", "different_ids", "duplicate_id", "bad_json"],
)
def test_multi_run_ci_rejects_incomplete_or_different_populations(
    multi_run_ci_evaluator, case
):
    evaluator = multi_run_ci_evaluator
    path = evaluator._result_dir / "run_1/model/BFCL_v4_simple_python_result.json"
    if case == "missing_score":
        (evaluator._score_dir / "run_1/model/BFCL_v4_simple_python_score.json").unlink()
    elif case == "missing_run":
        evaluator.autometric_runs_config.num_runs = 3
    elif case == "different_ids":
        path.write_text(path.read_text().replace("py0", "other"))
    elif case == "duplicate_id":
        path.write_text(path.read_text().replace("py0", "py1"))
    else:
        path.write_text("bad json")
    scores = evaluator._cache_scores({"BFCL_NON_LIVE_SIMPLE_AST_PASS@2": 2 / 3})
    assert scores.records == {}


def test_multi_run_ci_does_not_reconstruct_mean_from_missing_csv(
    multi_run_ci_evaluator,
):
    evaluator = multi_run_ci_evaluator
    (evaluator._score_dir / "run_1/data_overall.csv").unlink()
    scores = evaluator._cache_scores(
        {
            "BFCL_NON_LIVE_SIMPLE_AST_MEAN": 0.5,
            "BFCL_NON_LIVE_SIMPLE_AST_PASS@2": 2 / 3,
        }
    )
    assert set(scores.records) == {"BFCL_NON_LIVE_SIMPLE_AST_PASS@2"}


@pytest.mark.parametrize("include_live", [True, False])
def test_multi_run_ci_does_not_pair_different_run_counts(
    multi_run_ci_evaluator, include_live
):
    evaluator = multi_run_ci_evaluator
    if not include_live:
        for directory in (evaluator._result_dir, evaluator._score_dir):
            for path in directory.rglob("BFCL_v4_live_*.json"):
                path.unlink()
    evaluator.autometric_runs_config.pass_at_k = [1, 2]
    key = "BFCL_SIMPLE_PYTHON_PASS@1"
    web_key = f"BFCL_WEB_SEARCH_ACC_PASS@2_{evaluator.autometric_runs_config.web_search.upper()}"
    two_runs = evaluator._cache_scores({key: 1, web_key: 1}).records
    for directory in (evaluator._result_dir, evaluator._score_dir):
        for index in (0, 1):
            shutil.copytree(directory / f"run_{index}", directory / f"run_{index + 2}")
    evaluator.autometric_runs_config.num_runs = 4
    four_runs = evaluator._cache_scores({key: 1, web_key: 11 / 12}).records
    assert {row["uid"] for row in two_runs[key]}.isdisjoint(
        row["uid"] for row in four_runs[key]
    )
    assert len(four_runs[web_key]) == 6
    assert _reconstructed_score(four_runs[web_key]) == pytest.approx(11 / 12)
