import base64
import json
import os
from pathlib import Path
from typing import Dict

import pytest
from node_runner.src.job_dispatcher import JD
from node_runner.src.job_dispatcher.cache_scores import iter_compressed_jsonl_bytes
from node_runner.src.job_dispatcher.measurement import (
    MeasurementFiles,
    MeasurementUpload,
)
import node_runner.src.metrics.evaluators.lm_eval as lm_eval_module
from node_runner.src.config_model import (
    LmEvalEvaluatorConfig,
    LmEvalInstructTaskConfig,
    LmEvalPretrainTaskConfig,
    ModelInfoConfig as ModelInfo,
)
from node_runner.src.metrics.evaluators.lm_eval import LmEvalEvaluator

pytestmark = pytest.mark.unit

# ============================================================================
# Constants for test configuration
# ============================================================================

# Paths and directories
MOCK_METRICS_DIR = Path(__file__).resolve().parents[3] / "test_data" / "mock_metrics"
TEST_DATA_DIR_NAME = "test_data"
MOCK_METRICS_SUBDIR = "mock_metrics"
METRICS_DIR_NAME = "metrics"
LM_EVAL_SUBDIR = "lm_eval"

# Task names
TASK_HUMANEVAL = "humaneval"
TASK_BBH = "bbh"
TASK_NON_EXISTING = "non_existing_task"
TASK_RULER = "ruler"
TASK_MMLU = "mmlu"

CACHE_SCORES_TIMESTAMP = "2026-08-29T17-23-21.150946"

SECOND_NS = 1_000_000_000
RECORDED_MTIME_NS = 1_700_000_000 * SECOND_NS

# Run names (task name + requested shots)
RUN_HUMANEVAL_0SHOT = "humaneval_0shot"
RUN_BBH_3SHOT = "bbh_3shot"
RUN_MMLU_0SHOT = "mmlu_0shot"
RUN_MMLU_5SHOT = "mmlu_5shot"

# File names
HUMANEVAL_JSON_FILENAME = "humaneval_0shot_2025-01-01T00-00-00.000000.json"
LEGACY_MMLU_JSON_FILENAME = "mmlu_2026-01-01T00-00-00.000000.json"
EMPTY_JSON_CONTENT = "{}"

# Model configuration
MODEL_CLASS_SFT = "sft"
MODEL_SIZE_7B = "7b"
MODEL_SIZE_LITE = "lite"
MODEL_USERNAME_TEST = "test"
MODEL_USERNAME_U = "u"
MODEL_USERNAME_DEFAULT = "usr"

# Task-specific constants
BATCH_SIZE = 16
NUM_CONCURRENT = 8
MAX_LENGTH = 8192
NUM_FEWSHOT_0 = 0
NUM_FEWSHOT_3 = 3
NUM_FEWSHOT_5 = 5
BASE_URL = "http://127.0.0.1:30000/generate"
PYTHON_BIN_PATH = "/home/user/lm_eval/bin/python"

# Metric keys and values
METRIC_BBH_KEY = "bbh/3-shot/exact_match.get-answer"
METRIC_BBH_VALUE = 0.8637690062970358
METRIC_BBH_ACC_KEY = "bbh/3-shot/acc"
METRIC_HUMANEVAL_ACC_KEY = "humaneval/0-shot/acc"
METRIC_MMLU_KEY = "mmlu/5-shot/acc"
METRIC_MMLU_VALUE = 0.8753738783649053
METRIC_MMLU_PRO_KEY = "mmlu_pro/5-shot/exact_match.custom-extract"
METRIC_MMLU_PRO_VALUE = 0.7957945478723404
METRIC_BOUQUET_KEY = "bouquet/5-shot/chrf"
METRIC_BOUQUET_VALUE = 0.30167144422143483
METRIC_SPELLCHECK_KEY = "spellcheck_punctuation_ru/5-shot/F1.scoring"
METRIC_SPELLCHECK_VALUE = 0.5966548967147834

# Ruler metric patterns
RULER_METRIC_PREFIX = "ruler/"
METRIC_SUFFIX_MAX = "_max"
METRIC_SUFFIX_DIFF = "_diff"
CONTEXT_4096 = "4096"
CONTEXT_NOT_DIGIT = "not_digit"

# Test metric values
METRIC_VALUE_0_5 = 0.5
METRIC_VALUE_0_7 = 0.7
METRIC_VALUE_0_9 = 0.9
METRIC_VALUE_0_2 = 0.2
METRIC_VALUE_0_6 = 0.6
METRIC_VALUE_0_8 = 0.8
METRIC_VALUE_1_0 = 1.0
METRIC_VALUE_2_0 = 2.0
METRIC_VALUE_3_0 = 3.0
METRIC_VALUE_4_0 = 4.0

# Default values
DEFAULT_APPROX_TOLERANCE = 0.0001

# MultiPL-E executor environment
EXPLICIT_TASKPROXY_URL = "http://explicit-executor:8888"
ONLINE_TASKPROXY_URL = "http://yt.eye.salute.ru:8888"
CACHE_ONLY_ENABLED = "1"
PLAINTEXT_SECRET = "s3cr3t"


# ============================================================================
# Helper functions
# ============================================================================


def return_mock_metrics() -> Dict[str, float]:
    """Return mock metrics for testing."""
    return {
        METRIC_BBH_KEY: METRIC_BBH_VALUE,
        "mgsm_direct_en/8-shot/exact_match.remove_whitespace": 0.0,
        "mgsm_direct_ru/8-shot/exact_match.remove_whitespace": 0.0,
        "mgsm_direct_zh/8-shot/exact_match.remove_whitespace": 0.0,
        "mgsm_en_cot_ru/8-shot/exact_match.strict-match": 0.0,
        "mgsm_en_cot_zh/8-shot/exact_match.strict-match": 0.0,
        METRIC_MMLU_KEY: METRIC_MMLU_VALUE,
    }


def create_test_autometric_runs_config() -> LmEvalEvaluatorConfig:
    """Create test autometric runs configuration."""
    return LmEvalEvaluatorConfig(
        pretrain={
            TASK_HUMANEVAL: LmEvalPretrainTaskConfig(
                batch_size=BATCH_SIZE,
                model_args={
                    "base_url": BASE_URL,
                    "max_length": MAX_LENGTH,
                    "num_concurrent": NUM_CONCURRENT,
                    "tokenized_requests": True,
                },
                num_fewshot=NUM_FEWSHOT_0,
            ),
            TASK_BBH: LmEvalPretrainTaskConfig(
                batch_size=BATCH_SIZE,
                model_args={
                    "base_url": BASE_URL,
                    "max_length": MAX_LENGTH,
                    "num_concurrent": NUM_CONCURRENT,
                    "tokenized_requests": True,
                },
                num_fewshot=NUM_FEWSHOT_3,
            ),
        }
    )


# ============================================================================
# Test classes
# ============================================================================


class TestLmEvalEvaluator:
    """Tests for LM Eval evaluator."""

    @staticmethod
    def _prepare_evaluator_for_send_metrics(tmp_path: Path) -> LmEvalEvaluator:
        """Helper to prepare evaluator for send_metrics tests."""
        evaluator = LmEvalEvaluator()
        evaluator.hf_checkpoint_path = (
            tmp_path / "model-7b" / "ep0_ba10" / "artifacts" / "hf_checkpoint"
        )
        evaluator.model_info = ModelInfo(
            model_class=MODEL_CLASS_SFT, size=MODEL_SIZE_7B, username=MODEL_USERNAME_U
        )
        evaluator.metrics_dir = tmp_path / METRICS_DIR_NAME
        evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
        evaluator.cache_s3_paths = {}
        return evaluator

    def test_setup_builds_cmds_and_skips_existing(self, tmp_path: Path) -> None:
        """Test that setup builds commands and skips already calculated tasks."""
        artifacts_path = tmp_path / "artifacts"
        lm_eval_metrics_path = artifacts_path / METRICS_DIR_NAME / LM_EVAL_SUBDIR
        lm_eval_metrics_path.mkdir(parents=True, exist_ok=True)

        # Create existing humaneval results
        (lm_eval_metrics_path / HUMANEVAL_JSON_FILENAME).write_text(EMPTY_JSON_CONTENT)

        evaluator = LmEvalEvaluator()
        evaluator.setup(
            autometric_runs_config=create_test_autometric_runs_config(),
            artifacts_path=artifacts_path,
            hf_checkpoint_path=tmp_path / "tokenizer",
            model_info=ModelInfo(
                model_class=MODEL_CLASS_SFT,
                size=MODEL_SIZE_LITE,
                username=MODEL_USERNAME_TEST,
            ),
        )

        assert evaluator.metrics_dir == lm_eval_metrics_path
        assert RUN_HUMANEVAL_0SHOT not in evaluator.cmds_to_run  # Already calculated
        assert RUN_BBH_3SHOT in evaluator.cmds_to_run  # Should be calculated
        assert evaluator.cmds_to_run[RUN_BBH_3SHOT][0] == PYTHON_BIN_PATH
        assert evaluator.cmds_to_run[RUN_BBH_3SHOT][3:5] == ["--tasks", TASK_BBH]

    def test_lm_eval_runs_to_bash_list_builds_instruct_tasks(self, tmp_path: Path):
        evaluator = LmEvalEvaluator()
        cfg = LmEvalEvaluatorConfig(
            instruct={
                "chat_task": LmEvalInstructTaskConfig(
                    model_args={
                        "base_url": "http://127.0.0.1:29999/v1/chat/completions"
                    },
                    batch_size=1,
                )
            }
        )
        commands = evaluator._lm_eval_runs_to_bash_list(cfg, tmp_path / "metrics")
        cmd = commands["chat_task"]
        assert cmd[:5] == [PYTHON_BIN_PATH, "-m", "lm_eval", "--tasks", "chat_task"]
        assert "--output_path" in cmd
        assert "--model_args" in cmd
        assert (
            cmd[cmd.index("--model_args") + 1]
            == "base_url=http://127.0.0.1:29999/v1/chat/completions"
        )

    def test_lm_eval_runs_to_bash_list_fans_out_num_fewshot_list(
        self, tmp_path: Path
    ) -> None:
        metrics_dir = tmp_path / METRICS_DIR_NAME
        cfg = LmEvalEvaluatorConfig(
            pretrain={
                TASK_MMLU: LmEvalPretrainTaskConfig(
                    num_fewshot=[NUM_FEWSHOT_0, NUM_FEWSHOT_5]
                )
            }
        )

        commands = LmEvalEvaluator()._lm_eval_runs_to_bash_list(cfg, metrics_dir)

        assert sorted(commands) == [RUN_MMLU_0SHOT, RUN_MMLU_5SHOT]
        for run_name, num_fewshot in (
            (RUN_MMLU_0SHOT, NUM_FEWSHOT_0),
            (RUN_MMLU_5SHOT, NUM_FEWSHOT_5),
        ):
            cmd = commands[run_name]
            assert cmd[3:5] == ["--tasks", TASK_MMLU]
            assert cmd[cmd.index("--output_path") + 1] == str(
                metrics_dir / f"{run_name}.json"
            )
            assert cmd.count("--num_fewshot") == 1
            assert cmd[cmd.index("--num_fewshot") + 1] == str(num_fewshot)

    def test_lm_eval_runs_to_bash_list_keeps_task_name_without_num_fewshot(
        self, tmp_path: Path
    ) -> None:
        metrics_dir = tmp_path / METRICS_DIR_NAME
        cfg = LmEvalEvaluatorConfig(pretrain={TASK_RULER: LmEvalPretrainTaskConfig()})

        commands = LmEvalEvaluator()._lm_eval_runs_to_bash_list(cfg, metrics_dir)

        assert list(commands) == [TASK_RULER]
        cmd = commands[TASK_RULER]
        assert cmd[cmd.index("--output_path") + 1] == str(
            metrics_dir / f"{TASK_RULER}.json"
        )
        assert "--num_fewshot" not in cmd

    def test_lm_eval_runs_to_bash_list_rejects_duplicate_run_names(
        self, tmp_path: Path
    ) -> None:
        cfg = LmEvalEvaluatorConfig(
            pretrain={
                TASK_MMLU: LmEvalPretrainTaskConfig(num_fewshot=NUM_FEWSHOT_5),
                RUN_MMLU_5SHOT: LmEvalPretrainTaskConfig(),
            }
        )

        with pytest.raises(ValueError, match="Duplicate lm_eval run name"):
            LmEvalEvaluator()._lm_eval_runs_to_bash_list(cfg, tmp_path)

    @staticmethod
    def _write_run(
        path: Path, task: str, num_fewshot: int, acc: float, date: object
    ) -> Path:
        payload: dict = {
            "n-shot": {task: num_fewshot},
            "group_subtasks": {task: []},
            "results": {task: {"acc,none": acc}},
        }
        if date is not None:
            payload["date"] = date
        path.write_text(json.dumps(payload))
        return path

    @pytest.mark.parametrize("num_fewshot", (NUM_FEWSHOT_0, NUM_FEWSHOT_5))
    def test_collect_metrics_from_dir_prefers_the_latest_run(
        self, tmp_path: Path, num_fewshot: int
    ) -> None:
        # The pre-rename file sorts before the shots-suffixed one for shots >= 2 and
        # after it for 0 and 1, so file names cannot decide which run is the latest.
        stale = self._write_run(
            tmp_path / f"{TASK_MMLU}_2026-01-01T00-00-00.000000.json",
            TASK_MMLU,
            num_fewshot,
            METRIC_VALUE_0_2,
            date=1.0,
        )
        fresh = self._write_run(
            tmp_path / f"{TASK_MMLU}_{num_fewshot}shot_2026-08-18T00-00-00.000000.json",
            TASK_MMLU,
            num_fewshot,
            METRIC_VALUE_0_9,
            date=2.0,
        )
        # a copy between nodes can leave the older run with the newer mtime
        os.utime(stale, (9, 9))
        os.utime(fresh, (1, 1))

        converted = LmEvalEvaluator().collect_metrics_from_dir(tmp_path)

        assert converted == {f"{TASK_MMLU}/{num_fewshot}-shot/acc": METRIC_VALUE_0_9}

    @pytest.mark.parametrize("date", (None, "2026-08-18T00-00-00"))
    def test_collect_metrics_from_dir_falls_back_to_mtime_without_a_usable_date(
        self, tmp_path: Path, date: object
    ) -> None:
        stale = self._write_run(
            tmp_path / f"{RUN_MMLU_5SHOT}_2026-01-01T00-00-00.000000.json",
            TASK_MMLU,
            NUM_FEWSHOT_5,
            METRIC_VALUE_0_2,
            date=date,
        )
        fresh = self._write_run(
            tmp_path / f"{RUN_MMLU_5SHOT}_2026-08-18T00-00-00.000000.json",
            TASK_MMLU,
            NUM_FEWSHOT_5,
            METRIC_VALUE_0_9,
            date=date,
        )
        os.utime(stale, (1, 1))
        os.utime(fresh, (2, 2))
        (tmp_path / "not_a_result.txt").write_text(EMPTY_JSON_CONTENT)

        converted = LmEvalEvaluator().collect_metrics_from_dir(tmp_path)

        assert converted == {METRIC_MMLU_KEY: METRIC_VALUE_0_9}

    def test_measurement_metrics_keeps_only_the_tasks_of_this_run(
        self, tmp_path: Path
    ) -> None:
        evaluator = LmEvalEvaluator()
        evaluator.metrics_dir = tmp_path
        recorded = self._write_run(
            tmp_path / f"{RUN_BBH_3SHOT}_2026-01-01T00-00-00.000000.json",
            TASK_BBH,
            NUM_FEWSHOT_3,
            METRIC_VALUE_0_2,
            date=1.0,
        )
        os.utime(recorded, ns=(RECORDED_MTIME_NS, RECORDED_MTIME_NS))
        evaluator.measurement_files = MeasurementFiles(tmp_path, RECORDED_MTIME_NS)
        self._write_run(
            tmp_path / f"{RUN_MMLU_5SHOT}_2026-08-18T00-00-00.000000.json",
            TASK_MMLU,
            NUM_FEWSHOT_5,
            METRIC_VALUE_0_9,
            date=2.0,
        )

        measured = evaluator.measurement_metrics(
            {
                METRIC_MMLU_KEY: METRIC_VALUE_0_9,
                METRIC_BBH_ACC_KEY: METRIC_VALUE_0_2,
            }
        )

        assert measured == {METRIC_MMLU_KEY: METRIC_VALUE_0_9}

    @staticmethod
    def _write_jd_data(hf_checkpoint_path: Path, metrics_files_mtime_ns: dict) -> None:
        jd_data_path = hf_checkpoint_path / "artifacts" / "jd_data.json"
        jd_data_path.parent.mkdir(parents=True, exist_ok=True)
        jd_data_path.write_text(
            json.dumps(
                {
                    "checkpoint_info": {"id": "checkpoint-id", "checkpoint_type": "hf"},
                    "history": [
                        {
                            "id": "record-id",
                            "action": "autometrics",
                            "data": {
                                "run_uuid": "run-of-a-recorded-measurement",
                                "metrics_files_mtime_ns": metrics_files_mtime_ns,
                            },
                        }
                    ],
                }
            )
        )

    @pytest.mark.parametrize(
        ("recorded_mtime_ns", "expected_measured", "expected_files"),
        [
            pytest.param(
                {LM_EVAL_SUBDIR: RECORDED_MTIME_NS},
                {METRIC_BBH_ACC_KEY: METRIC_VALUE_0_9},
                [f"{RUN_BBH_3SHOT}_2026-01-02T00-00-00.000000.json"],
                id="only-the-run-finished-after-the-recorded-measurement",
            ),
            pytest.param(
                {},
                {
                    METRIC_HUMANEVAL_ACC_KEY: METRIC_VALUE_0_2,
                    METRIC_BBH_ACC_KEY: METRIC_VALUE_0_9,
                },
                [
                    f"{RUN_BBH_3SHOT}_2026-01-02T00-00-00.000000.json",
                    f"{RUN_HUMANEVAL_0SHOT}_2026-01-01T00-00-00.000000.json",
                ],
                id="every-run-when-no-measurement-was-recorded-yet",
            ),
        ],
    )
    def test_measurement_keeps_the_runs_of_a_job_that_died_mid_evaluator(
        self,
        tmp_path: Path,
        recorded_mtime_ns: dict,
        expected_measured: dict,
        expected_files: list,
    ) -> None:
        artifacts_path = tmp_path / "artifacts"
        hf_checkpoint_path = tmp_path / "hf_checkpoint"
        metrics_dir = artifacts_path / METRICS_DIR_NAME / LM_EVAL_SUBDIR
        metrics_dir.mkdir(parents=True)
        earlier_run = self._write_run(
            metrics_dir / f"{RUN_HUMANEVAL_0SHOT}_2026-01-01T00-00-00.000000.json",
            TASK_HUMANEVAL,
            NUM_FEWSHOT_0,
            METRIC_VALUE_0_2,
            date=1.0,
        )
        later_run = self._write_run(
            metrics_dir / f"{RUN_BBH_3SHOT}_2026-01-02T00-00-00.000000.json",
            TASK_BBH,
            NUM_FEWSHOT_3,
            METRIC_VALUE_0_9,
            date=2.0,
        )
        os.utime(earlier_run, ns=(RECORDED_MTIME_NS, RECORDED_MTIME_NS))
        os.utime(
            later_run,
            ns=(RECORDED_MTIME_NS + SECOND_NS, RECORDED_MTIME_NS + SECOND_NS),
        )
        self._write_jd_data(hf_checkpoint_path, recorded_mtime_ns)

        evaluator = LmEvalEvaluator()
        evaluator.setup(
            autometric_runs_config=create_test_autometric_runs_config(),
            artifacts_path=artifacts_path,
            hf_checkpoint_path=hf_checkpoint_path,
            model_info=ModelInfo(
                model_class=MODEL_CLASS_SFT,
                size=MODEL_SIZE_LITE,
                username=MODEL_USERNAME_TEST,
            ),
        )
        metrics = evaluator.get_metrics()

        assert evaluator.cmds_to_run == {}
        assert metrics == {
            METRIC_HUMANEVAL_ACC_KEY: METRIC_VALUE_0_2,
            METRIC_BBH_ACC_KEY: METRIC_VALUE_0_9,
        }
        assert evaluator.measurement_metrics(metrics) == expected_measured
        assert evaluator.measurement_files.paths() == {
            metrics_dir / file_name for file_name in expected_files
        }
        assert evaluator.measurement_files.mtime_ns() == RECORDED_MTIME_NS + SECOND_NS

    def test_get_metrics_conversion(self) -> None:
        """Test conversion of metrics from LM Eval format."""
        evaluator = LmEvalEvaluator()
        evaluator.metrics_dir = MOCK_METRICS_DIR
        converted = evaluator.get_metrics()
        assert converted[METRIC_BBH_KEY] == METRIC_BBH_VALUE

    def test_get_metrics_parses_bouquet_group_metric(self) -> None:
        evaluator = LmEvalEvaluator()
        evaluator.metrics_dir = MOCK_METRICS_DIR
        converted = evaluator.get_metrics()
        assert converted[METRIC_BOUQUET_KEY] == pytest.approx(METRIC_BOUQUET_VALUE)

    def test_get_metrics_matches_selected_hardcoded_values(self) -> None:
        evaluator = LmEvalEvaluator()
        evaluator.metrics_dir = MOCK_METRICS_DIR
        converted = evaluator.get_metrics()

        assert converted[METRIC_MMLU_KEY] == pytest.approx(METRIC_MMLU_VALUE)
        assert converted[METRIC_MMLU_PRO_KEY] == pytest.approx(METRIC_MMLU_PRO_VALUE)
        assert converted[METRIC_BOUQUET_KEY] == pytest.approx(METRIC_BOUQUET_VALUE)
        assert converted[METRIC_SPELLCHECK_KEY] == pytest.approx(
            METRIC_SPELLCHECK_VALUE
        )

    def test_generate_jd_metrics_json(self) -> None:
        metrics = return_mock_metrics()
        jd_metrics = JD.generate_metrics_json({1: metrics})
        assert jd_metrics == {1: metrics}

    @pytest.mark.parametrize(
        ("run_name", "expected"),
        [
            (TASK_HUMANEVAL, True),
            (TASK_BBH, True),
            (TASK_NON_EXISTING, False),
            (RUN_MMLU_5SHOT, False),
        ],
    )
    def test_already_calculated(self, run_name: str, expected: bool) -> None:
        """Test detection of already calculated runs."""
        evaluator = LmEvalEvaluator()
        evaluator.metrics_dir = MOCK_METRICS_DIR
        result = evaluator._is_already_calculated(run_name, evaluator.metrics_dir)
        assert result is expected

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("agieval_2026-06-04T13-48-57.987979.json", True),
            ("agieval.json", True),
            ("agieval_2026-06-04T13-48-57.987979.json.bak", False),
            ("samples_agieval_aqua_rat_2026-06-04T13-48-57.987979.jsonl", False),
            ("agieval_aqua_rat_2026-06-04T13-48-57.987979.json", False),
            ("metrics_step_0.json", False),
        ],
    )
    def test_already_calculated_only_matches_result_json(
        self, tmp_path: Path, filename: str, expected: bool
    ) -> None:
        (tmp_path / filename).write_text(EMPTY_JSON_CONTENT)

        assert LmEvalEvaluator._is_already_calculated("agieval", tmp_path) is expected

    def test_filter_keeps_drop_f1_metric(self) -> None:
        evaluator = LmEvalEvaluator()
        filtered = evaluator.filter_metrics({"drop/5-shot/f1": 0.6744599412751677})
        assert filtered["drop/5-shot/f1"] == pytest.approx(0.6744599412751677)

    def test_filter_excludes_suffix_metrics(self) -> None:
        """Test filtering of metrics with specific suffixes."""
        evaluator = LmEvalEvaluator()
        raw = {
            f"{RULER_METRIC_PREFIX}0-shot/task_a{METRIC_SUFFIX_MAX}": METRIC_VALUE_1_0,
            f"{RULER_METRIC_PREFIX}0-shot/task_b{METRIC_SUFFIX_DIFF}": METRIC_VALUE_2_0,
            f"{RULER_METRIC_PREFIX}0-shot/task_c": METRIC_VALUE_3_0,
            "plain_metric": METRIC_VALUE_4_0,
        }
        filtered = evaluator.filter_metrics(raw)

        # Metrics with _max and _diff suffixes should be excluded
        assert f"{RULER_METRIC_PREFIX}0-shot/task_a{METRIC_SUFFIX_MAX}" not in filtered
        assert f"{RULER_METRIC_PREFIX}0-shot/task_b{METRIC_SUFFIX_DIFF}" not in filtered

        # Other metrics should be preserved
        assert filtered[f"{RULER_METRIC_PREFIX}0-shot/task_c"] == METRIC_VALUE_3_0
        assert filtered["plain_metric"] == METRIC_VALUE_4_0

    def test_aggregate_ruler_skips_non_numeric_context(self) -> None:
        """Test aggregation of Ruler metrics with non-numeric context skipping."""
        evaluator = LmEvalEvaluator()
        metrics = {
            f"{RULER_METRIC_PREFIX}task1/0-shot/{CONTEXT_4096}": METRIC_VALUE_0_5,
            f"{RULER_METRIC_PREFIX}task2/0-shot/{CONTEXT_4096}": METRIC_VALUE_0_7,
            f"{RULER_METRIC_PREFIX}task3/0-shot/{CONTEXT_NOT_DIGIT}": METRIC_VALUE_0_9,
            "bbh/3-shot/acc": METRIC_VALUE_0_2,
        }
        out = evaluator.aggregate_metrics(metrics)

        # Ruler metrics with numeric context should be aggregated
        assert out[f"{RULER_METRIC_PREFIX}0-shot/{CONTEXT_4096}"] == pytest.approx(
            METRIC_VALUE_0_6, rel=DEFAULT_APPROX_TOLERANCE
        )

        # Ruler metric with non-numeric context should be excluded
        assert f"{RULER_METRIC_PREFIX}task3/0-shot/{CONTEXT_NOT_DIGIT}" not in out

        # Non-ruler metrics should be preserved unchanged
        assert out["bbh/3-shot/acc"] == METRIC_VALUE_0_2

    def test_aggregate_ruler_keeps_num_fewshot_in_metric_name(self) -> None:
        evaluator = LmEvalEvaluator()
        metrics = {
            f"{RULER_METRIC_PREFIX}task1/0-shot/{CONTEXT_4096}": METRIC_VALUE_0_5,
            f"{RULER_METRIC_PREFIX}task1/5-shot/{CONTEXT_4096}": METRIC_VALUE_0_9,
            f"{RULER_METRIC_PREFIX}task2/5-shot/{CONTEXT_4096}": METRIC_VALUE_0_7,
        }

        out = evaluator.aggregate_metrics(metrics)

        assert out == {
            f"{RULER_METRIC_PREFIX}0-shot/{CONTEXT_4096}": pytest.approx(
                METRIC_VALUE_0_5
            ),
            f"{RULER_METRIC_PREFIX}5-shot/{CONTEXT_4096}": pytest.approx(
                METRIC_VALUE_0_8
            ),
        }

    def test_aggregate_ruler_averages_subtasks_of_one_context(self):
        evaluator = LmEvalEvaluator()
        out = evaluator._aggregate_ruler_metrics(
            {"ruler/task1/0-shot/4096": 0.5, "ruler/task2/0-shot/4096": 0.7}
        )
        assert out == {"ruler/0-shot/4096": pytest.approx(0.6)}

    def test_aggregate_single_run_without_results_returns_empty(self):
        evaluator = LmEvalEvaluator()
        assert evaluator.aggregate_single_run({"results": None}) == {}

    def test_aggregate_single_run_mmlu_pro_includes_group_and_subtasks(self):
        evaluator = LmEvalEvaluator()
        run = {
            "n-shot": {"mmlu_pro": 5},
            "group_subtasks": {"mmlu_pro": ["mmlu_pro_math", "mmlu_pro_physics"]},
            "results": {
                "mmlu_pro_math": {
                    "acc,none": 0.3,
                    "acc_stderr,none": 0.1,
                    "alias": "ignore",
                },
                "mmlu_pro_physics": {"acc,none": 0.4},
                "mmlu_pro": {"acc,none": 0.35},
            },
        }

        out = evaluator.aggregate_single_run(run)
        assert out["mmlu_pro/math/5-shot/acc"] == 0.3
        assert out["mmlu_pro/physics/5-shot/acc"] == 0.4
        assert out["mmlu_pro/5-shot/acc"] == 0.35
        assert not any("stderr" in k for k in out)

    def test_aggregate_single_run_ruler_includes_group_and_subtasks(self):
        evaluator = LmEvalEvaluator()
        run = {
            "n-shot": {"ruler": 0},
            "group_subtasks": {"ruler": ["ruler_4096", "ruler_8192"]},
            "results": {
                "ruler_4096": {"acc,none": 0.8},
                "ruler_8192": {"acc,none": 0.6},
                "ruler": {"acc,none": 0.7},
            },
        }

        out = evaluator.aggregate_single_run(run)
        assert out["ruler/4096/0-shot/acc"] == 0.8
        assert out["ruler/8192/0-shot/acc"] == 0.6
        assert out["ruler/0-shot/acc"] == 0.7

    def test_run_uses_current_no_arg_contract(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmEvalEvaluator()
        evaluator.autometric_runs_config = LmEvalEvaluatorConfig()
        evaluator.metrics_dir = tmp_path
        evaluator.cmds_to_run = {"a": ["cmd"]}
        called = {"run_all": 0}

        class _Pool:
            def __init__(
                self, cmds_to_run, max_parallel, autometrics_log_file, cwd, env
            ):
                assert cmds_to_run == {"a": ["cmd"]}

            def run_all(self):
                called["run_all"] += 1

        monkeypatch.setattr(lm_eval_module, "SubprocessPoolRunner", _Pool)
        evaluator.run()
        assert called["run_all"] == 1

    def test_run_master_executes_pool(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmEvalEvaluator()
        evaluator.autometric_runs_config = LmEvalEvaluatorConfig(pool_max_parallel=5)
        evaluator.metrics_dir = tmp_path
        evaluator.cmds_to_run = {"a": ["cmd"]}
        called = {"run_all": 0}

        class _Pool:
            def __init__(
                self, cmds_to_run, max_parallel, autometrics_log_file, cwd, env
            ):
                assert cmds_to_run == {"a": ["cmd"]}
                assert max_parallel == 5
                assert cwd is None
                assert env is not None

            def run_all(self):
                called["run_all"] += 1

        monkeypatch.setattr(lm_eval_module, "SubprocessPoolRunner", _Pool)
        evaluator.run()
        assert called["run_all"] == 1

    def test_get_metrics_without_metrics_dir(self, monkeypatch: pytest.MonkeyPatch):
        warnings = []
        monkeypatch.setattr(lm_eval_module.Logger, "warning", warnings.append)
        evaluator = LmEvalEvaluator()
        evaluator.metrics_dir = None
        assert evaluator.get_metrics() == {}
        assert warnings

    def test_get_metrics_pipeline_filters_and_aggregates(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = LmEvalEvaluator()
        evaluator.metrics_dir = MOCK_METRICS_DIR
        monkeypatch.setattr(
            evaluator,
            "collect_metrics_from_dir",
            lambda _metrics_dir, _files: {
                "ruler/task1/0-shot/4096": 0.5,
                "ruler/task2/0-shot/4096": 0.7,
                "ruler/task1/0-shot/task_a_max": 1.0,
                "spellcheck_punctuation_ru/0-shot/acc": 0.55,
                "bbh/3-shot/acc": 0.2,
            },
        )
        out = evaluator.get_metrics()
        assert out["ruler/0-shot/4096"] == pytest.approx(0.6)
        assert "ruler/task1/0-shot/task_a_max" not in out
        assert out["spellcheck_punctuation_ru/0-shot/acc"] == 0.55
        assert out["bbh/3-shot/acc"] == 0.2

    def test_after_run_calls_send_tar_and_symlink(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmEvalEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics"
        evaluator.artifacts_path = tmp_path / "artifacts"
        evaluator.hf_checkpoint_path = tmp_path / "hf"
        called = {}
        monkeypatch.setattr(
            evaluator, "_send_metrics_to_jd", lambda m: called.update({"send": m})
        )
        monkeypatch.setattr(
            JD,
            "upload_metrics_and_cache_scores",
            lambda **kwargs: (
                called.update({"publish": kwargs}) or {"cache_s3_path": "s3:scores"}
            ),
        )
        monkeypatch.setattr(
            JD,
            "tar_metrics_dir",
            lambda m, a, _name: tmp_path / "m.tar.zst",
        )
        monkeypatch.setattr(
            JD,
            "symlink_metrics_for_transfer",
            lambda tar, hf, _name: called.update({"symlink": tar}),
        )
        uploaded = MeasurementUpload("prefix/record-id/lm_eval", [Path("cache.jsonl")])

        evaluator.after_run({"acc": 0.9}, uploaded)

        assert called["publish"]["shooter_name"] == "lm_eval"
        assert called["publish"]["uploaded"] is uploaded
        assert called["publish"]["make_builder"]().cache_subdir is None
        assert evaluator.cache_s3_paths == {"cache_s3_path": "s3:scores"}
        assert called["send"] == {"acc": 0.9}
        assert called["symlink"] == tmp_path / "m.tar.zst"

    def test_after_run_hands_jd_a_builder_over_the_metrics_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        (metrics_dir / f"gsm8k_{CACHE_SCORES_TIMESTAMP}.json").write_text(
            json.dumps(
                {
                    "results": {"gsm8k": {"exact_match,none": 1.0}},
                    "group_subtasks": {"gsm8k": []},
                    "n-shot": {"gsm8k": 8},
                }
            ),
            encoding="utf-8",
        )
        (metrics_dir / f"samples_gsm8k_{CACHE_SCORES_TIMESTAMP}.jsonl").write_text(
            json.dumps(
                {
                    "doc_id": 0,
                    "doc_hash": "aabbccdd",
                    "filter": "none",
                    "exact_match": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        evaluator = LmEvalEvaluator()
        evaluator.metrics_dir = metrics_dir
        evaluator.artifacts_path = tmp_path / "artifacts"
        evaluator.hf_checkpoint_path = tmp_path / "hf"
        published = {}
        monkeypatch.setattr(evaluator, "_send_metrics_to_jd", lambda metrics: None)
        monkeypatch.setattr(
            JD, "tar_metrics_dir", lambda m, a, _name: tmp_path / "m.tar.zst"
        )
        monkeypatch.setattr(
            JD, "symlink_metrics_for_transfer", lambda tar, hf, _name: None
        )
        monkeypatch.setattr(
            JD,
            "upload_metrics_and_cache_scores",
            lambda **kwargs: published.update(kwargs) or {},
        )

        evaluator.after_run({"acc": 0.9})
        scores_path = published["make_builder"]().build(metrics_dir)

        rows = list(iter_compressed_jsonl_bytes(scores_path.read_bytes()))
        assert [row["benchmark"] for row in rows] == ["gsm8k/8-shot/exact_match"]

    def test_convert_raw_config_to_bash_skips_precalculated_tasks(self, monkeypatch):
        evaluator = LmEvalEvaluator()
        evaluator.metrics_dir = MOCK_METRICS_DIR
        monkeypatch.setattr(
            evaluator,
            "_lm_eval_runs_to_bash_list",
            lambda runs, metrics_dir: {
                "agieval": ["cmd-agieval"],
                "bbh": ["cmd-bbh"],
                "mmlu": ["cmd-mmlu"],
            },
        )
        monkeypatch.setattr(
            evaluator,
            "_is_already_calculated",
            lambda task_name, _metrics_dir: task_name in {"agieval", "mmlu"},
        )

        config = LmEvalEvaluatorConfig(
            pretrain={
                "agieval": LmEvalPretrainTaskConfig(),
                "bbh": LmEvalPretrainTaskConfig(),
                "mmlu": LmEvalPretrainTaskConfig(),
            }
        )
        out = evaluator._convert_raw_config_to_bash(config, Path("metrics"))
        assert out == {"bbh": ["cmd-bbh"]}

    def test_convert_raw_config_to_bash_restart_keeps_precalculated_tasks(
        self, monkeypatch
    ):
        evaluator = LmEvalEvaluator()
        evaluator.metrics_dir = MOCK_METRICS_DIR
        monkeypatch.setattr(
            evaluator,
            "_lm_eval_runs_to_bash_list",
            lambda runs, metrics_dir: {
                "agieval": ["cmd-agieval"],
                "bbh": ["cmd-bbh"],
            },
        )

        config = LmEvalEvaluatorConfig(
            restart=True,
            pretrain={
                "agieval": LmEvalPretrainTaskConfig(),
                "bbh": LmEvalPretrainTaskConfig(),
            },
        )
        out = evaluator._convert_raw_config_to_bash(config, MOCK_METRICS_DIR)

        assert out == {"agieval": ["cmd-agieval"], "bbh": ["cmd-bbh"]}

    def test_send_metrics_to_jd_branches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        warnings = []
        monkeypatch.setattr(lm_eval_module.Logger, "warning", warnings.append)

        evaluator = LmEvalEvaluator()
        evaluator.hf_checkpoint_path = None
        evaluator.model_info = None
        evaluator._send_metrics_to_jd({"x": 1.0})
        assert warnings

        evaluator = self._prepare_evaluator_for_send_metrics(tmp_path)

        captured = {}
        monkeypatch.setattr(
            JD,
            "parse_checkpoint_path",
            lambda p: ("/nfs", "model", 0, 10),
        )
        monkeypatch.setattr(
            JD,
            "get_custom_model_name_if_exist",
            lambda model_name: model_name,
        )
        monkeypatch.setattr(JD, "generate_metrics_json", lambda x: x)
        monkeypatch.setattr(JD, "get_shooter_version", lambda _: "1.0")
        monkeypatch.setattr(JD, "set_autocompare_id_in_body", lambda body: None)
        monkeypatch.setattr(
            JD,
            "save_metrics_json",
            lambda body, step, metrics_dir: captured.update(
                {"body": body, "step": step}
            ),
        )
        monkeypatch.setattr(
            JD,
            "send_metrics",
            lambda body, _artifacts: captured.update({"request": body}),
        )

        evaluator._send_metrics_to_jd({"acc": 0.9})
        assert captured["step"] == 10
        assert captured["body"]["shooter_name"] == "lm_eval"
        assert captured["body"]["metrics"] == {10: {"acc": 0.9}}
        assert "cache_s3_path" not in captured["body"]

    def test_send_metrics_to_jd_carries_cache_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = self._prepare_evaluator_for_send_metrics(tmp_path)
        evaluator.cache_s3_paths = {
            "cache_s3_path": "gigaeye-s3:b/k/scores.zst",
            "cache_dir_s3_path": "gigaeye-s3:b/k",
        }
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("/nfs", "m", 0, 10))
        monkeypatch.setattr(
            JD, "get_custom_model_name_if_exist", lambda model_name: model_name
        )
        monkeypatch.setattr(JD, "generate_metrics_json", lambda x: x)
        monkeypatch.setattr(JD, "get_shooter_version", lambda _: "1.0")
        monkeypatch.setattr(JD, "set_autocompare_id_in_body", lambda body: None)
        monkeypatch.setattr(JD, "save_metrics_json", lambda body, step, d: None)
        captured = {}
        monkeypatch.setattr(
            JD, "send_metrics", lambda body, _a: captured.update({"body": body})
        )

        evaluator._send_metrics_to_jd({"acc": 0.9})

        assert captured["body"]["cache_s3_path"] == "gigaeye-s3:b/k/scores.zst"
        assert captured["body"]["cache_dir_s3_path"] == "gigaeye-s3:b/k"

    @pytest.mark.parametrize(
        "metrics, expected",
        [
            (
                {
                    "ifbench/0-shot/prompt_level_loose_acc": 0.75,
                    "ifbench/0-shot/prompt_level_strict_acc": 0.5,
                    "ifbench/0-shot/inst_level_loose_acc": 0.875,
                    "ifbench/0-shot/inst_level_strict_acc": 0.625,
                    "other/0-shot/acc": 0.25,
                },
                {
                    "ifbench/0-shot/prompt_level_loose_acc": {
                        "value": 0.75,
                        "submetrics": {
                            "ifbench/0-shot/prompt_level_strict_acc": 0.5,
                            "ifbench/0-shot/inst_level_loose_acc": 0.875,
                            "ifbench/0-shot/inst_level_strict_acc": 0.625,
                        },
                    },
                    "other/0-shot/acc": 0.25,
                },
            ),
            (
                {
                    "ifbench_32/0-shot/pass@8_prompt_level_loose_acc.all_responses": 0.75,
                    "ifbench_32/0-shot/pass@8_inst_level_loose_acc.all_responses": 0.5,
                    "ifbench_32/0-shot/pass@32_prompt_level_loose_acc.all_responses": 1.0,
                    "ifbench_32/1-shot/pass@8_prompt_level_loose_acc.all_responses": 0.5,
                    "ifbench_32/1-shot/pass@1_prompt_level_loose_acc.all_responses": 0.25,
                },
                {
                    "ifbench_32/0-shot/mean_8": {
                        "value": 0.75,
                        "submetrics": {
                            "ifbench_32/0-shot/pass@8_inst_level_loose_acc.all_responses": 0.5,
                            "ifbench_32/0-shot/pass@32_prompt_level_loose_acc.all_responses": 1.0,
                        },
                    },
                    "ifbench_32/1-shot/mean_8": {
                        "value": 0.5,
                        "submetrics": {
                            "ifbench_32/1-shot/pass@1_prompt_level_loose_acc.all_responses": 0.25,
                        },
                    },
                },
            ),
            ({"other/0-shot/acc": 0.5}, {"other/0-shot/acc": 0.5}),
        ],
    )
    def test_send_metrics_to_jd_groups_ifbench(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, metrics, expected
    ):
        evaluator = self._prepare_evaluator_for_send_metrics(tmp_path)
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("/nfs", "m", 0, 10))
        monkeypatch.setattr(
            JD, "get_custom_model_name_if_exist", lambda model_name: model_name
        )
        monkeypatch.setattr(JD, "get_shooter_version", lambda _: "1.0")
        monkeypatch.setattr(JD, "set_autocompare_id_in_body", lambda body: None)
        monkeypatch.setattr(JD, "save_metrics_json", lambda body, step, d: None)
        captured = {}
        monkeypatch.setattr(JD, "send_metrics", lambda body, _a: captured.update(body))

        evaluator._send_metrics_to_jd(metrics)

        assert captured["metrics"] == {10: expected}

    @pytest.mark.parametrize("suffix", ["", "_32"])
    def test_jsonschema_submetrics_average_all_levels_and_keep_shots_separate(
        self, suffix
    ):
        metrics = {
            f"jsonschema_bench_easy{suffix}/2-shot/json_validity": 1.0,
            f"jsonschema_bench_easy{suffix}/2-shot/schema_compliance": 0.75,
            f"jsonschema_bench_medium{suffix}/2-shot/json_validity": 0.75,
            f"jsonschema_bench_medium{suffix}/2-shot/schema_compliance": 0.5,
            f"jsonschema_bench_hard{suffix}/2-shot/json_validity": 0.5,
            f"jsonschema_bench_hard{suffix}/2-shot/schema_compliance": 0.25,
            f"jsonschema_bench_easy{suffix}/0-shot/schema_compliance": 0.125,
        }

        nested = JD.nest_submetrics(metrics, LmEvalEvaluator._submetric_groups(metrics))

        assert nested == {
            f"jsonschema_bench{suffix}/2-shot/mean": {
                "value": 0.625,
                "submetrics": {
                    key: value for key, value in metrics.items() if "/2-shot/" in key
                },
            },
            f"jsonschema_bench{suffix}/0-shot/mean": {
                "value": 0.125,
                "submetrics": {
                    f"jsonschema_bench_easy{suffix}/0-shot/schema_compliance": 0.125,
                },
            },
        }

    def test_submetric_groups_without_metrics(self):
        assert JD.nest_submetrics({}, LmEvalEvaluator._submetric_groups({})) == {}

    def test_decode_base64_round_trips(self):
        encoded = base64.b64encode(PLAINTEXT_SECRET.encode()).decode()

        assert lm_eval_module.decode_base64(encoded) == PLAINTEXT_SECRET

    def test_build_lm_eval_env_carries_token_and_executor_url(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("YT_TASKPROXY_BASE_URL", raising=False)
        monkeypatch.delenv("MULTIPL_E_CACHE_ONLY", raising=False)
        monkeypatch.setattr(JD, "internet_is_available", lambda: True)

        env = LmEvalEvaluator()._build_lm_eval_env()

        assert env["HF_ALLOW_CODE_EVAL"] == "1"
        assert env["HF_DATASETS_TRUST_REMOTE_CODE"] == "1"
        assert env["YT_TOKEN"] == lm_eval_module.decode_base64(
            lm_eval_module.ROBOT_SWE_TOKEN_BASE64
        )
        assert env["YT_TASKPROXY_BASE_URL"] == ONLINE_TASKPROXY_URL

    def test_build_lm_eval_env_keeps_explicit_executor_url(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("YT_TASKPROXY_BASE_URL", EXPLICIT_TASKPROXY_URL)
        monkeypatch.delenv("MULTIPL_E_CACHE_ONLY", raising=False)

        env = LmEvalEvaluator()._build_lm_eval_env()

        assert env["YT_TASKPROXY_BASE_URL"] == EXPLICIT_TASKPROXY_URL

    def test_build_lm_eval_env_omits_executor_url_for_cache_only(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("YT_TASKPROXY_BASE_URL", raising=False)
        monkeypatch.setenv("MULTIPL_E_CACHE_ONLY", CACHE_ONLY_ENABLED)

        env = LmEvalEvaluator()._build_lm_eval_env()

        assert "YT_TASKPROXY_BASE_URL" not in env


@pytest.mark.parametrize(
    ("group_name", "task_name", "submetric", "expected"),
    [
        ("gsm8k", "gsm8k", "exact_match,none", "gsm8k/8-shot/exact_match"),
        (
            "gsm8k",
            "gsm8k",
            "exact_match,flexible-extract",
            "gsm8k/8-shot/exact_match.flexible-extract",
        ),
        (
            "mmlu_pro",
            "mmlu_pro_biology",
            "exact_match,custom-extract",
            "mmlu_pro/biology/8-shot/exact_match.custom-extract",
        ),
    ],
)
def test_metric_key(group_name, task_name, submetric, expected) -> None:
    assert LmEvalEvaluator.metric_key(group_name, task_name, 8, submetric) == expected


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("ruler/vt/0-shot/32768", "ruler/0-shot/32768"),
        ("ruler/0-shot/32768", "ruler/0-shot/32768"),
        ("ruler/vt/0-shot/score", None),
    ],
)
def test_ruler_bucket_key(metric_name, expected) -> None:
    assert LmEvalEvaluator.ruler_bucket_key(metric_name) == expected
