import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest
from node_runner.src.job_dispatcher import JD
from node_runner.src.job_dispatcher.cache_scores import encoded_path
from node_runner.src.job_dispatcher.measurement import (
    MeasurementFiles,
    MeasurementUpload,
)
from node_runner.src.config_model import (
    DevicesObmEvaluatorConfig,
    ObmAllConfig,
)
from node_runner.src.config_model import (
    ModelInfoConfig as ModelInfo,
)
from node_runner.src.metrics.evaluators.obm import ObmEvaluator

pytestmark = pytest.mark.unit

# ============================================================================
# Constants for test configuration
# ============================================================================

DATASET_DS1 = "ds1"
DATASET_DS2 = "ds2"

METRIC_ACCURACY = "accuracy"
METRIC_UNKNOWN = "unknown"
METRIC_OTHER = "other"

METRICS_DIR_NAME = "metrics"
METRICS_JSONL_FILENAME = "all_metrics.jsonl"

RECORDED_MTIME_NS = 1_700_000_000 * 1_000_000_000

DEFAULT_METRIC_VALUE = 0.0
TEST_METRIC_VALUE_1 = 0.5
TEST_METRIC_VALUE_2 = 0.2
TEST_METRIC_VALUE_3 = 1.0
TEST_METRIC_VALUE_4 = 1.5
TEST_METRIC_VALUE_5 = 0.7
TEST_METRIC_VALUE_6 = 0.3

MOCK_EXIT_CODE_SUCCESS = 0
MOCK_EXIT_CODE_FAILURE = 1

# ============================================================================
# Helper functions
# ============================================================================


def create_mock_subprocess_manager(
    exit_code: int = MOCK_EXIT_CODE_SUCCESS, is_successful: bool = True
) -> type:
    class MockSubprocessManager:
        def __init__(self, cmd: Any, log_file: Any, cwd: Any) -> None:
            self.exit_code = exit_code

        def start(self) -> None:
            pass

        def wait_for_completion(self) -> None:
            pass

        def is_successful(self) -> bool:
            return is_successful

    return MockSubprocessManager


def create_test_obm_config(
    groups: list[str] | None = None,
    tasks: list[str] | None = None,
    ignore_tasks: list[str] | None = None,
    n_threads: int = 8,
    restart: bool = False,
    context: str | None = None,
    n_workers: int = 5,
    disable_few_shots: bool = False,
    disable_suggested_params: bool = False,
) -> DevicesObmEvaluatorConfig:
    """Create a DevicesObmEvaluatorConfig for tests."""
    all_kwargs: Dict[str, Any] = {}
    if groups is not None:
        all_kwargs["groups"] = groups
    if tasks is not None:
        all_kwargs["tasks"] = tasks
    if ignore_tasks is not None:
        all_kwargs["ignore_tasks"] = ignore_tasks
    if not groups and not tasks:
        all_kwargs["tasks"] = ["default_task"]
    all_kwargs["restart"] = restart
    all_kwargs["n_workers"] = n_workers
    all_kwargs["disable_few_shots"] = disable_few_shots
    all_kwargs["disable_suggested_params"] = disable_suggested_params
    if context is not None:
        all_kwargs["context"] = context
    return DevicesObmEvaluatorConfig(
        obm_config="devices",
        all=ObmAllConfig(**all_kwargs),
        n_threads=n_threads,
    )


def create_metrics_jsonl_content(
    datasets: list[tuple[str, dict, bool]],
) -> str:
    """Create JSONL content in new OBM format (DatasetResult)."""
    lines = []
    for name, metric, error in datasets:
        entry: Dict[str, Any] = {"name": name}
        if error:
            entry["error"] = True
        if metric:
            entry["metric"] = metric
        lines.append(json.dumps(entry))
    return "\n".join(lines)


# ============================================================================
# Test classes
# ============================================================================


class TestObmEvaluatorConfigNormalization:
    def test_replace_group_no_groups_returns_unchanged(self) -> None:
        evaluator = ObmEvaluator()
        cfg = create_test_obm_config(tasks=["mmlu"])
        out = evaluator._replace_papaya_group_with_b3r_group(cfg)
        assert out is cfg

    def test_replace_group_offline_raises_on_sber(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator = ObmEvaluator()
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: False,
        )
        cfg = create_test_obm_config(groups=["sber"])
        with pytest.raises(ValueError):
            evaluator._replace_papaya_group_with_b3r_group(cfg)

    def test_replace_group_offline_replaces_papaya(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator = ObmEvaluator()
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: False,
        )
        cfg = create_test_obm_config(groups=["papaya", "open"])
        out = evaluator._replace_papaya_group_with_b3r_group(cfg)
        assert out.all.groups == ["sber", "open"]

    def test_replace_group_offline_raises_on_sber_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator = ObmEvaluator()
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: False,
        )
        for variant in ["SBER", "Sber", "sber", "sBeR"]:
            cfg = create_test_obm_config(groups=[variant])
            with pytest.raises(ValueError):
                evaluator._replace_papaya_group_with_b3r_group(cfg)

    def test_replace_group_offline_no_sber_in_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator = ObmEvaluator()
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: False,
        )
        cfg = create_test_obm_config(groups=["papaya", "open", "math"])
        out = evaluator._replace_papaya_group_with_b3r_group(cfg)
        result_groups = out.all.groups
        result_lower = [g.lower() for g in result_groups]
        assert "papaya" not in result_lower
        assert "PAPAYA" not in result_groups

    def test_replace_group_offline_multiple_papaya(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator = ObmEvaluator()
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: False,
        )
        cfg = create_test_obm_config(groups=["papaya", "open", "papaya"])
        out = evaluator._replace_papaya_group_with_b3r_group(cfg)
        assert out.all.groups == ["sber", "open", "sber"]

    def test_replace_group_offline_preserves_non_papaya_groups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator = ObmEvaluator()
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: False,
        )
        cfg = create_test_obm_config(groups=["open", "math", "function"])
        out = evaluator._replace_papaya_group_with_b3r_group(cfg)
        assert out.all.groups == ["open", "math", "function"]

    def test_replace_group_online_keeps_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator = ObmEvaluator()
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: True,
        )
        cfg = create_test_obm_config(groups=["open", "math"])
        out = evaluator._replace_papaya_group_with_b3r_group(cfg)
        assert out.all.groups == ["open", "math"]

    def test_replace_group_online_keeps_papaya_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator = ObmEvaluator()
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: True,
        )
        cfg = create_test_obm_config(groups=["PAPAYA", "open"])
        out = evaluator._replace_papaya_group_with_b3r_group(cfg)
        assert out.all.groups == ["PAPAYA", "open"]

    def test_replace_group_returns_same_object_reference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator = ObmEvaluator()
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: False,
        )
        cfg = create_test_obm_config(groups=["PAPAYA"])
        out = evaluator._replace_papaya_group_with_b3r_group(cfg)
        assert out is cfg

    def test_obm_runs_to_bash_list_includes_groups_and_tasks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.artifacts_path = tmp_path
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: False,
        )

        config = create_test_obm_config(
            groups=["open", "math"],
            tasks=["mmlu", "gsm8k"],
            n_workers=4,
            restart=True,
            context="/path/to/ctx.yaml",
        )
        cmds = evaluator._obm_runs_to_bash_list(config, tmp_path / METRICS_DIR_NAME)
        cmd = cmds["all"]

        assert "run" in cmd
        groups_idx = cmd.index("--groups")
        assert cmd[groups_idx + 1 : groups_idx + 3] == ["open", "math"]

        tasks_idx = cmd.index("--tasks")
        assert cmd[tasks_idx + 1 : tasks_idx + 3] == ["mmlu", "gsm8k"]

        w_idx = cmd.index("-w")
        assert cmd[w_idx + 1] == "4"

        assert "-r" in cmd

        ctx_idx = cmd.index("-ctx")
        assert cmd[ctx_idx + 1] == "/path/to/ctx.yaml"

    def test_obm_runs_to_bash_list_includes_ignore_tasks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.artifacts_path = tmp_path
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: True,
        )

        config = create_test_obm_config(
            groups=["open"],
            ignore_tasks=["mmlu", "gsm8k"],
        )
        cmds = evaluator._obm_runs_to_bash_list(config, tmp_path / METRICS_DIR_NAME)
        cmd = cmds["all"]

        ignore_idx = cmd.index("--ignore-tasks")
        assert cmd[ignore_idx + 1 : ignore_idx + 3] == ["mmlu", "gsm8k"]

    def test_obm_runs_generates_context_for_disable_few_shots(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.artifacts_path = tmp_path
        evaluator.obm_repo_path = tmp_path / "obm_repo"
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: True,
        )

        config = create_test_obm_config(
            groups=["open"],
            disable_few_shots=True,
        )
        cmds = evaluator._obm_runs_to_bash_list(config, tmp_path / METRICS_DIR_NAME)
        cmd = cmds["all"]

        assert "-ctx" in cmd
        ctx_idx = cmd.index("-ctx")
        ctx_path = cmd[ctx_idx + 1]
        assert Path(ctx_path).exists()

        import yaml

        with open(ctx_path, "r") as f:
            ctx_data = yaml.safe_load(f)

        assert ctx_data["disable_few_shots"] is True
        assert "disable_suggested_params" not in ctx_data

    def test_obm_runs_generates_context_for_both_flags(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.artifacts_path = tmp_path
        evaluator.obm_repo_path = tmp_path / "obm_repo"
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: True,
        )

        config = create_test_obm_config(
            groups=["open"],
            disable_few_shots=True,
            disable_suggested_params=True,
        )
        cmds = evaluator._obm_runs_to_bash_list(config, tmp_path / METRICS_DIR_NAME)
        cmd = cmds["all"]

        assert "-ctx" in cmd
        ctx_idx = cmd.index("-ctx")
        ctx_path = cmd[ctx_idx + 1]

        import yaml

        with open(ctx_path, "r") as f:
            ctx_data = yaml.safe_load(f)

        assert ctx_data["disable_few_shots"] is True
        assert ctx_data["disable_suggested_params"] is True

    def test_obm_runs_generates_context_for_disable_suggested_params_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.artifacts_path = tmp_path
        evaluator.obm_repo_path = tmp_path / "obm_repo"
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: True,
        )

        config = create_test_obm_config(
            groups=["open"],
            disable_suggested_params=True,
        )
        cmds = evaluator._obm_runs_to_bash_list(config, tmp_path / METRICS_DIR_NAME)
        cmd = cmds["all"]

        assert "-ctx" in cmd
        ctx_idx = cmd.index("-ctx")
        ctx_path = cmd[ctx_idx + 1]
        assert Path(ctx_path).exists()

        import yaml

        with open(ctx_path, "r") as f:
            ctx_data = yaml.safe_load(f)

        assert "disable_few_shots" not in ctx_data
        assert ctx_data["disable_suggested_params"] is True

    def test_obm_runs_explicit_context_takes_priority(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.artifacts_path = tmp_path
        evaluator.obm_repo_path = tmp_path / "obm_repo"
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: True,
        )

        config = create_test_obm_config(
            groups=["open"],
            disable_few_shots=True,
            context="/custom/ctx.yaml",
        )
        cmds = evaluator._obm_runs_to_bash_list(config, tmp_path / METRICS_DIR_NAME)
        cmd = cmds["all"]

        ctx_idx = cmd.index("-ctx")
        assert cmd[ctx_idx + 1] == "/custom/ctx.yaml"
        assert cmd.count("-ctx") == 1

    def test_obm_runs_no_context_when_flags_are_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.artifacts_path = tmp_path
        evaluator.obm_repo_path = tmp_path / "obm_repo"
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: True,
        )

        config = create_test_obm_config(
            groups=["open"],
            disable_few_shots=False,
            disable_suggested_params=False,
        )
        cmds = evaluator._obm_runs_to_bash_list(config, tmp_path / METRICS_DIR_NAME)
        cmd = cmds["all"]
        assert "-ctx" not in cmd

    def test_obm_runs_to_bash_list_minimal_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.artifacts_path = tmp_path
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: True,
        )

        config = create_test_obm_config(tasks=["mmlu"])
        cmds = evaluator._obm_runs_to_bash_list(config, tmp_path / METRICS_DIR_NAME)
        cmd = cmds["all"]

        assert "run" in cmd
        assert "-cfg" in cmd
        assert "-d" in cmd
        assert "--groups" not in cmd
        assert "-r" not in cmd
        assert "-ctx" not in cmd
        assert "--ignore-tasks" not in cmd

    def test_obm_runs_to_bash_list_skips_workers_flag_when_n_workers_is_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.artifacts_path = tmp_path
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: True,
        )

        config = create_test_obm_config(tasks=["mmlu"], n_workers=0, restart=True)
        cmd = evaluator._obm_runs_to_bash_list(config, tmp_path / METRICS_DIR_NAME)[
            "all"
        ]
        assert "-w" not in cmd
        assert "-r" in cmd


class TestObmEvaluatorMetricsCollection:
    def test_parse_metrics_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()

        metrics_file = tmp_path / METRICS_JSONL_FILENAME
        metrics_file.write_text(
            create_metrics_jsonl_content(
                [
                    (
                        DATASET_DS1,
                        {"name": "accuracy", "value": TEST_METRIC_VALUE_1},
                        False,
                    ),
                    (
                        DATASET_DS2,
                        {"name": "score", "value": TEST_METRIC_VALUE_2},
                        False,
                    ),
                    ("failed_ds", None, True),
                ]
            ),
            encoding="utf-8",
        )

        out = evaluator._parse_metrics(str(metrics_file))
        assert out == {
            f"{DATASET_DS1}/accuracy": TEST_METRIC_VALUE_1,
            f"{DATASET_DS2}/score": TEST_METRIC_VALUE_2,
            "failed_ds": "failed",
        }

    def test_parse_metrics_empty_file(self, tmp_path: Path) -> None:
        evaluator = ObmEvaluator()
        metrics_file = tmp_path / METRICS_JSONL_FILENAME
        metrics_file.write_text("", encoding="utf-8")

        out = evaluator._parse_metrics(str(metrics_file))
        assert out == {}

    def test_parse_metrics_reports_cancelled_and_skips_missing_name_or_metric(
        self, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        metrics_file = tmp_path / METRICS_JSONL_FILENAME
        lines = [
            json.dumps({"metric": {"value": 0.5}}),
            json.dumps({"name": "cancelled_ds", "cancelled": True}),
            json.dumps({"name": "no_metric_ds"}),
            json.dumps(
                {
                    "name": "good",
                    "metric": {"name": "accuracy", "value": TEST_METRIC_VALUE_6},
                }
            ),
        ]
        metrics_file.write_text("\n".join(lines), encoding="utf-8")

        out = evaluator._parse_metrics(str(metrics_file))
        assert out == {
            "good/accuracy": TEST_METRIC_VALUE_6,
            "cancelled_ds": "failed",
        }

    def test_parse_metrics_non_numeric_value(self, tmp_path: Path) -> None:
        evaluator = ObmEvaluator()
        metrics_file = tmp_path / METRICS_JSONL_FILENAME
        entry = {
            "name": DATASET_DS1,
            "metric": {
                "name": "accuracy",
                "value": "not_a_number",
            },
        }
        metrics_file.write_text(json.dumps(entry), encoding="utf-8")
        out = evaluator._parse_metrics(str(metrics_file))
        assert out == {}

    def test_parse_metrics_missing_metric_name(self, tmp_path: Path) -> None:
        evaluator = ObmEvaluator()
        metrics_file = tmp_path / METRICS_JSONL_FILENAME
        entry = {
            "name": DATASET_DS1,
            "metric": {
                "value": 0.5,
            },
        }
        metrics_file.write_text(json.dumps(entry), encoding="utf-8")
        out = evaluator._parse_metrics(str(metrics_file))
        assert out == {}

    def test_parse_metrics_parse_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        metrics_file = tmp_path / METRICS_JSONL_FILENAME
        metrics_file.write_text("{bad json}\n", encoding="utf-8")

        errors = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.Logger.error", errors.append
        )
        out = evaluator._parse_metrics(str(metrics_file))
        assert out == {}
        assert errors

    def test_parse_metrics_encoded_file(self, tmp_path: Path) -> None:
        evaluator = ObmEvaluator()
        metrics_file = encoded_path(tmp_path / METRICS_JSONL_FILENAME)
        content = create_metrics_jsonl_content(
            [
                (
                    DATASET_DS1,
                    {"name": METRIC_ACCURACY, "value": TEST_METRIC_VALUE_1},
                    False,
                )
            ]
        )
        evaluator._write_obm_encoded_file(metrics_file, content)

        out = evaluator._parse_metrics(str(metrics_file))

        assert out == {f"{DATASET_DS1}/{METRIC_ACCURACY}": TEST_METRIC_VALUE_1}
        assert evaluator._read_obm_encoded_file(metrics_file) == content

    @pytest.mark.parametrize("failed_count", [2, 0.02, "bad"])
    def test_parse_metrics_reports_failed_or_invalid_request_statistics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failed_count
    ) -> None:
        evaluator = ObmEvaluator()
        metrics_file = tmp_path / METRICS_JSONL_FILENAME
        entry = {
            "name": DATASET_DS1,
            "metric": {"name": METRIC_ACCURACY, "value": TEST_METRIC_VALUE_1},
            "statistics": {"failed": {"request": {"value": failed_count}}},
        }
        metrics_file.write_text(json.dumps(entry), encoding="utf-8")
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.Logger.warning", warnings.append
        )

        out = evaluator._parse_metrics(str(metrics_file))

        assert out == {f"{DATASET_DS1}/{METRIC_ACCURACY}": "failed"}
        assert warnings

    def test_parse_metrics_keeps_zero_failed_statistics(self, tmp_path: Path) -> None:
        evaluator = ObmEvaluator()
        metrics_file = tmp_path / METRICS_JSONL_FILENAME
        entry = {
            "name": DATASET_DS1,
            "metric": {"name": METRIC_ACCURACY, "value": TEST_METRIC_VALUE_1},
            "statistics": {"failed": {"request": {"value": 0}}},
        }
        metrics_file.write_text(json.dumps(entry), encoding="utf-8")

        out = evaluator._parse_metrics(str(metrics_file))

        assert out == {f"{DATASET_DS1}/{METRIC_ACCURACY}": TEST_METRIC_VALUE_1}

    def test_get_metrics_branches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.Logger.warning", warnings.append
        )
        evaluator = ObmEvaluator()
        assert evaluator.get_metrics() == {}

        evaluator.metrics_dir = tmp_path / "metrics"
        evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
        assert evaluator.get_metrics() == {}

        jsonl_path = evaluator.metrics_dir / "all_metrics.jsonl"
        jsonl_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(evaluator, "_parse_metrics", lambda p: {"x": 1.0})
        assert evaluator.get_metrics() == {"x": 1.0}

    def test_find_obm_metrics_file_prefers_encoded_offline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        jsonl_path = metrics_dir / METRICS_JSONL_FILENAME
        encoded = encoded_path(metrics_dir / METRICS_JSONL_FILENAME)
        jsonl_path.write_text("{}", encoding="utf-8")
        evaluator._write_obm_encoded_file(encoded, "{}")
        monkeypatch.setattr(JD, "internet_is_available", lambda: False)

        assert evaluator._find_obm_metrics_file(metrics_dir) == encoded

    def test_find_obm_metrics_file_falls_back_to_encoded_online(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        encoded = encoded_path(metrics_dir / METRICS_JSONL_FILENAME)
        evaluator._write_obm_encoded_file(encoded, "{}")
        monkeypatch.setattr(JD, "internet_is_available", lambda: True)

        assert evaluator._find_obm_metrics_file(metrics_dir) == encoded

    def test_measurement_metrics_keeps_only_the_datasets_of_this_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        metrics_dir = tmp_path / METRICS_DIR_NAME
        cache_dir = metrics_dir / "cache"
        cache_dir.mkdir(parents=True)
        evaluator.metrics_dir = metrics_dir
        monkeypatch.setattr(JD, "internet_is_available", lambda: True)
        recorded_cache_file = cache_dir / f"{DATASET_DS1}.jsonl"
        recorded_cache_file.write_text("{}", encoding="utf-8")
        os.utime(recorded_cache_file, ns=(RECORDED_MTIME_NS, RECORDED_MTIME_NS))

        evaluator.measurement_files = MeasurementFiles(metrics_dir, RECORDED_MTIME_NS)

        (cache_dir / f"{DATASET_DS2}.jsonl").write_text("{}", encoding="utf-8")
        (metrics_dir / METRICS_JSONL_FILENAME).write_text(
            create_metrics_jsonl_content(
                [
                    (
                        DATASET_DS1,
                        {"name": METRIC_ACCURACY, "value": TEST_METRIC_VALUE_1},
                        False,
                    ),
                    (
                        DATASET_DS2,
                        {"name": METRIC_ACCURACY, "value": TEST_METRIC_VALUE_2},
                        False,
                    ),
                ]
            ),
            encoding="utf-8",
        )

        measured = evaluator.measurement_metrics(
            {
                f"{DATASET_DS1}/{METRIC_ACCURACY}": TEST_METRIC_VALUE_1,
                f"{DATASET_DS2}/{METRIC_ACCURACY}": TEST_METRIC_VALUE_2,
            }
        )

        assert measured == {f"{DATASET_DS2}/{METRIC_ACCURACY}": TEST_METRIC_VALUE_2}

    def test_measurement_metrics_without_a_metrics_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.metrics_dir = tmp_path / METRICS_DIR_NAME
        evaluator.metrics_dir.mkdir(parents=True)
        monkeypatch.setattr(JD, "internet_is_available", lambda: True)

        assert (
            evaluator.measurement_metrics({f"{DATASET_DS1}/{METRIC_ACCURACY}": 1.0})
            == {}
        )

    def test_run_executes_all_commands(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = ObmEvaluator()
        evaluator.metrics_dir = tmp_path
        evaluator.cmds_to_run = {"run1": ["a"], "run2": ["b"]}
        evaluator.obm_repo_path = tmp_path

        called = {"start": 0, "wait": 0, "log": 0}

        class _SP:
            def __init__(self, cmd, log_file, cwd, env):
                self.cmd = cmd

            def start(self):
                called["start"] += 1

            def wait_for_completion(self):
                called["wait"] += 1

            def log_error(self):
                called["log"] += 1

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.SubprocessManager", _SP
        )
        evaluator.run()
        assert called == {"start": 2, "wait": 2, "log": 2}


class TestObmEvaluatorJdAndLifecycle:
    def test_convert_raw_config_to_bash_uses_obm_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = ObmEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics"
        expected = {"all": ["python", "-m", "obm.text", "run"]}
        monkeypatch.setattr(
            evaluator,
            "_obm_runs_to_bash_list",
            lambda cfg, metrics_dir: expected,
        )
        config = create_test_obm_config(groups=["open"])
        out = evaluator._convert_raw_config_to_bash(config)
        assert out == expected

    def test_send_metrics_to_jd_skips_without_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.hf_checkpoint_path = None
        evaluator.model_info = None
        monkeypatch.setattr(
            JD,
            "send_metrics",
            lambda body, _artifacts: pytest.fail("must not send without a checkpoint"),
        )

        evaluator._send_metrics_to_jd({"acc": 0.9})

    @pytest.mark.parametrize(
        "cache_s3_paths",
        [
            {},
            {
                "cache_s3_path": "s3:cache_scores.jsonl.zst",
                "cache_dir_s3_path": "s3:cache",
            },
        ],
    )
    def test_send_metrics_to_jd_builds_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache_s3_paths: dict[str, str],
    ):
        evaluator = ObmEvaluator()
        evaluator.model_info = ModelInfo(model_class="sft", size="7b", username="usr")
        evaluator.hf_checkpoint_path = (
            tmp_path / "ep0_ba100" / "artifacts" / "hf_checkpoint"
        )
        evaluator.metrics_dir = tmp_path / "metrics"
        evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
        evaluator.cache_s3_paths = cache_s3_paths

        saved = {}
        requested = {}
        monkeypatch.setattr(
            JD,
            "parse_checkpoint_path",
            lambda p: ("/nfs/path", "model-name", 0, 100),
        )
        monkeypatch.setattr(
            JD,
            "get_custom_model_name_if_exist",
            lambda model_name: model_name,
        )
        monkeypatch.setattr(JD, "generate_metrics_json", lambda x: x)
        monkeypatch.setattr(JD, "get_shooter_version", lambda _: "1.2.3")
        monkeypatch.setattr(
            JD,
            "set_autocompare_id_in_body",
            lambda body: body.update({}),
        )
        monkeypatch.setattr(
            evaluator,
            "_save_metrics_json",
            lambda body, step: saved.update({"body": body, "step": step}),
        )
        monkeypatch.setattr(
            JD,
            "send_metrics",
            lambda body, _artifacts: requested.update({"body": body}),
        )

        evaluator._send_metrics_to_jd({"acc": 0.9})
        assert saved["step"] == 100
        assert saved["body"]["shooter_name"] == "obm"
        assert saved["body"]["metrics"] == {100: {"acc": 0.9}}
        for key, url in cache_s3_paths.items():
            assert saved["body"][key] == url
        assert ("cache_s3_path" in saved["body"]) is bool(cache_s3_paths)
        assert requested["body"]["name"] == "model-name"

    def test_after_run_calls_lifecycle_hooks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = ObmEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics"
        evaluator.original_artifacts_path = tmp_path / "artifacts"
        evaluator.hf_checkpoint_path = tmp_path / "hf"
        calls = {}

        monkeypatch.setattr(
            JD,
            "upload_metrics_and_cache_scores",
            lambda **kwargs: (
                calls.update({"publish": kwargs}) or {"cache_s3_path": "s3:scores"}
            ),
        )
        monkeypatch.setattr(
            evaluator,
            "_send_metrics_to_jd",
            lambda metrics: calls.update({"send": metrics}),
        )
        monkeypatch.setattr(
            JD,
            "tar_metrics_dir",
            lambda metrics_dir, artifacts_path, _name: tmp_path / "m.tar.zst",
        )
        monkeypatch.setattr(
            JD,
            "symlink_metrics_for_transfer",
            lambda tar_path, hf_checkpoint_path, _name: calls.update(
                {"symlink": tar_path}
            ),
        )
        monkeypatch.setattr(
            evaluator,
            "_copy_obm_log_to_metrics_dir",
            lambda: calls.update({"copy_log": True}),
        )

        uploaded = MeasurementUpload("prefix/record-id/obm", [Path("cache/ds1.jsonl")])

        evaluator.after_run({"acc": 0.1}, uploaded)

        assert calls["publish"]["shooter_name"] == "obm"
        assert calls["publish"]["uploaded"] is uploaded
        assert calls["publish"]["make_builder"]().cache_subdir == "cache"
        assert evaluator.cache_s3_paths == {"cache_s3_path": "s3:scores"}
        assert calls["send"] == {"acc": 0.1}
        assert calls["copy_log"] is True
        assert "symlink" in calls

    def test_copy_obm_log_to_metrics_dir_anonymizes_offline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        obm_path = tmp_path / "obm"
        metrics_dir = tmp_path / "metrics"
        obm_path.mkdir()
        metrics_dir.mkdir()
        (obm_path / "obm.log").write_text("sber log", encoding="utf-8")
        evaluator.metrics_dir = metrics_dir
        monkeypatch.setattr(evaluator, "_get_obm_path", lambda: obm_path)
        monkeypatch.setattr(JD, "internet_is_available", lambda: False)

        evaluator._copy_obm_log_to_metrics_dir()

        assert (metrics_dir / "obm.log").read_text(encoding="utf-8") == "papaya log"

    def test_copy_obm_log_to_metrics_dir_copies_online(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        obm_path = tmp_path / "obm"
        metrics_dir = tmp_path / "metrics"
        obm_path.mkdir()
        metrics_dir.mkdir()
        (obm_path / "obm.log").write_text("sber log", encoding="utf-8")
        evaluator.metrics_dir = metrics_dir
        monkeypatch.setattr(evaluator, "_get_obm_path", lambda: obm_path)
        monkeypatch.setattr(JD, "internet_is_available", lambda: True)

        evaluator._copy_obm_log_to_metrics_dir()

        assert (metrics_dir / "obm.log").read_text(encoding="utf-8") == "sber log"

    def test_setup_initializes_state_and_cmds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = ObmEvaluator()
        artifacts_path = tmp_path / "artifacts"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        cfg = create_test_obm_config(groups=["open", "math"])
        created = {}
        monkeypatch.setattr(evaluator, "_unzip_obm", lambda: tmp_path / "obm_repo")
        monkeypatch.setattr(
            evaluator,
            "_create_obm_configs",
            lambda artifacts_path, obm_config: created.update(
                {"cfg_path": artifacts_path, "obm_config": obm_config}
            ),
        )
        monkeypatch.setattr(
            evaluator,
            "_create_metrics_dir",
            lambda artifacts_path, name: setattr(
                evaluator, "metrics_dir", artifacts_path / "metrics" / name
            ),
        )
        monkeypatch.setattr(
            evaluator, "_replace_papaya_group_with_b3r_group", lambda c: c
        )
        monkeypatch.setattr(
            evaluator,
            "_convert_raw_config_to_bash",
            lambda c: {"all": ["python", "-m", "obm.text", "run"]},
        )

        evaluator.setup(
            autometric_runs_config=cfg,
            artifacts_path=artifacts_path,
            hf_checkpoint_path=tmp_path / "hf",
            model_info=ModelInfo(model_class="sft", size="7b", username="u"),
        )
        assert evaluator.obm_repo_path == tmp_path / "obm_repo"
        assert evaluator.obm_config is cfg
        assert evaluator.cmds_to_run == {"all": ["python", "-m", "obm.text", "run"]}
        assert created["cfg_path"] == artifacts_path

    def test_create_obm_configs_writes_yaml(self, tmp_path: Path):
        evaluator = ObmEvaluator()
        artifacts_path = tmp_path / "artifacts"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        config = create_test_obm_config(groups=["open"])
        evaluator._create_obm_configs(artifacts_path, config)
        config_path = artifacts_path / "obm_config.yaml"
        assert config_path.exists()
        content = config_path.read_text(encoding="utf-8")
        assert "benchmark:" in content
        assert "devices:" in content

    def test_unzip_obm_removes_old_target_and_log(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        target = tmp_path / "one-button-metrics"
        target.mkdir(parents=True, exist_ok=True)
        unzip_log = tmp_path / "unzip.log"
        unzip_log.write_text("old", encoding="utf-8")
        calls = {"start": 0, "wait": 0, "raise": 0}

        class _SP:
            def __init__(self, cmd, log_file, cwd, run_name):
                assert cmd[:3] == ["tar", "--zstd", "-xf"]
                assert log_file == unzip_log
                assert run_name == "Unzip OBM"

            def start(self):
                calls["start"] += 1

            def wait_for_completion(self):
                calls["wait"] += 1
                unzip_log.write_text("fresh", encoding="utf-8")

            def raise_for_exit_code(self):
                calls["raise"] += 1

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.SubprocessManager", _SP
        )
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.ObmEvaluator._get_obm_path",
            staticmethod(lambda: target),
        )
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.Path",
            lambda p: unzip_log if p == "/home/user/unzip_obm.log" else Path(p),
        )

        out = ObmEvaluator._unzip_obm()
        assert out == target
        assert calls == {"start": 1, "wait": 1, "raise": 1}
        assert not unzip_log.exists()

    def test_unzip_obm_when_target_and_log_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        target = tmp_path / "one-button-metrics"
        unzip_log = tmp_path / "unzip.log"
        calls = {"start": 0, "wait": 0, "raise": 0}

        class _SP:
            def __init__(self, cmd, log_file, cwd, run_name):
                assert log_file == unzip_log

            def start(self):
                calls["start"] += 1

            def wait_for_completion(self):
                calls["wait"] += 1
                unzip_log.write_text("created", encoding="utf-8")

            def raise_for_exit_code(self):
                calls["raise"] += 1

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.SubprocessManager", _SP
        )
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.ObmEvaluator._get_obm_path",
            staticmethod(lambda: target),
        )
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.Path",
            lambda p: unzip_log if p == "/home/user/unzip_obm.log" else Path(p),
        )

        out = ObmEvaluator._unzip_obm()
        assert out == target
        assert calls == {"start": 1, "wait": 1, "raise": 1}
        assert not unzip_log.exists()

    def test_unzip_obm_skips_unzip_when_path_env_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv(ObmEvaluator.OBM_PATH_ENV, str(tmp_path))

        out = ObmEvaluator._unzip_obm()

        assert out == tmp_path

    def test_save_metrics_json_success_and_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = ObmEvaluator()
        evaluator.metrics_dir = str(tmp_path)
        evaluator._obm_metrics_encrypted = False
        infos = []
        errors = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.Logger.info", infos.append
        )
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.Logger.error", errors.append
        )
        monkeypatch.setattr(JD, "internet_is_available", lambda: True)

        evaluator._save_metrics_json({"x": 1}, step=7)
        assert (tmp_path / "metrics_step_7.json").exists()
        assert infos

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.json.dump",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dump error")),
        )
        evaluator._save_metrics_json({"x": 1}, step=8)
        assert errors

    def test_save_metrics_json_encoded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        evaluator = ObmEvaluator()
        evaluator.metrics_dir = str(tmp_path)
        evaluator._obm_metrics_encrypted = True
        monkeypatch.setattr(JD, "internet_is_available", lambda: True)

        evaluator._save_metrics_json({"x": 1}, step=7)

        save_path = tmp_path / "metrics_step_7.json.enc"
        assert save_path.exists()
        assert json.loads(evaluator._read_obm_encoded_file(save_path)) == {"x": 1}
