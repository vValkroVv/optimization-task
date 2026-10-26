"""Tests for the MetricsEvaluator interface."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from node_runner.src.job_dispatcher import JD
from node_runner.src.job_dispatcher.measurement import MeasurementFiles
from node_runner.src.metrics.evaluators.interface import EVALUATED, MetricsEvaluator
from node_runner.src.metrics.metrics_manager import EVALUATORS

pytestmark = pytest.mark.unit


class _DummyEvaluator(MetricsEvaluator):
    """Dummy implementation of MetricsEvaluator for testing."""

    def setup(
        self,
        autometric_runs_config: Dict[str, Any],
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: Any,
    ) -> None:
        pass

    def run(self, is_worker: bool, autometrics_log_file: Optional[Path]) -> None:
        pass

    def get_metrics(self) -> Dict[str, float]:
        return {}

    def after_run(self, metrics: Dict[int, Dict[str, float]]) -> None:
        pass


@pytest.fixture
def dummy_evaluator() -> _DummyEvaluator:
    return _DummyEvaluator()


class TestRliUrl:
    @pytest.mark.parametrize(
        ("internet_available", "expected"),
        [
            (True, "https://rli.sberdevices.ru"),
            (False, "https://gym.static.space"),
        ],
    )
    def test_get_rli_url_depends_on_internet(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dummy_evaluator: _DummyEvaluator,
        internet_available: bool,
        expected: str,
    ):
        monkeypatch.setattr(
            JD,
            "internet_is_available",
            lambda: internet_available,
        )

        assert dummy_evaluator.get_rli_url() == expected


class TestYtUrl:
    @pytest.mark.parametrize(
        ("internet_available", "expected"),
        [
            (True, "http://yt.eye.salute.ru:8888"),
            (False, "http://saurus.static.space:8888"),
        ],
    )
    def test_get_yt_url_depends_on_internet(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dummy_evaluator: _DummyEvaluator,
        internet_available: bool,
        expected: str,
    ):
        monkeypatch.setattr(
            JD,
            "internet_is_available",
            lambda: internet_available,
        )

        assert dummy_evaluator.get_yt_url() == expected


class TestModelSlug:
    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("minimax-m2.7", "minimax_m2_7"),
            ("chatwm_openrouter:minimax/minimax-m2", "minimax_m2"),
            ("GigaChat/Pro-2.0", "pro_2_0"),
            ("--weird--", "weird"),
        ],
    )
    def test_model_slug_normalizes_name(self, model_name: str, expected: str):
        assert MetricsEvaluator._model_slug(model_name) == expected


class TestMetricsDir:
    def test_create_metrics_dir_builds_path_and_log_file(
        self, dummy_evaluator: _DummyEvaluator, tmp_path: Path
    ):
        dummy_evaluator._create_metrics_dir(tmp_path, "dummy")

        assert dummy_evaluator.metrics_dir == tmp_path / "metrics" / "dummy"
        assert dummy_evaluator.metrics_dir.is_dir()
        assert dummy_evaluator.log_file == (
            tmp_path / "metrics" / "dummy" / "autometrics.log"
        )

    def test_load_json_file_reads_payload(
        self, dummy_evaluator: _DummyEvaluator, tmp_path: Path
    ):
        payload = {"accuracy": 0.5, "nested": {"k": 1}}
        json_path = tmp_path / "jd_metrics.json"
        json_path.write_text(json.dumps(payload), encoding="utf-8")

        assert dummy_evaluator._load_json_file(json_path) == payload


class TestMeasurementDefaults:
    def test_evaluator_caches_no_scores_by_default(
        self, dummy_evaluator: _DummyEvaluator
    ):
        assert dummy_evaluator.make_cache_scores_builder() is None

    def test_every_metric_belongs_to_the_measurement_by_default(
        self, dummy_evaluator: _DummyEvaluator
    ):
        metrics = {"accuracy": 0.5}

        assert dummy_evaluator.measurement_metrics(metrics) == metrics

    def test_no_measured_files_mtime_without_measurement_files(
        self, dummy_evaluator: _DummyEvaluator
    ):
        assert dummy_evaluator.measured_files_mtime_ns() is None

    def test_measured_files_mtime_is_the_newest_metrics_file_mtime(
        self, dummy_evaluator: _DummyEvaluator, tmp_path: Path
    ):
        metrics_file = tmp_path / "jd_metrics.json"
        metrics_file.write_text("{}", encoding="utf-8")
        os.utime(metrics_file, ns=(500, 500))
        dummy_evaluator.measurement_files = MeasurementFiles(tmp_path)

        assert dummy_evaluator.measured_files_mtime_ns() == 500


class TestStages:
    @pytest.mark.parametrize(
        "name",
        [
            "tau2",
            "tau3",
            "forge",
            "obm_judged",
            "new_obm_judged",
            "gigametric_judged",
            "safety_unified",
        ],
    )
    def test_parallel_evaluator_runs_one_stage_per_judge_beside_the_evaluated_model(
        self, name: str
    ):
        evaluator = object.__new__(EVALUATORS[name])
        evaluator.autometric_runs_config = SimpleNamespace(
            secondary_model_names=["ja", "jb"]
        )
        judged: list[str] = []
        evaluator.run_on_model_name = judged.append

        stages = evaluator.stages()
        for stage in stages:
            stage.run()

        assert [stage.engine_keys for stage in stages] == [
            (EVALUATED, "ja"),
            (EVALUATED, "jb"),
        ]
        assert judged == ["ja", "jb"]

    @pytest.mark.parametrize(
        ("name", "scoring_phase"),
        [("dredd", "judgement"), ("aa_lcr", "evaluation"), ("claweval", "evaluation")],
    )
    def test_scoring_evaluator_generates_on_the_evaluated_model_then_scores_on_its_judge(
        self, name: str, scoring_phase: str
    ):
        evaluator = object.__new__(EVALUATORS[name])
        evaluator.autometric_runs_config = SimpleNamespace(secondary_model_names=["j"])
        phases: list[str] = []
        evaluator.run_phase = phases.append

        stages = evaluator.stages()
        for stage in stages:
            stage.run()

        assert [stage.engine_keys for stage in stages] == [(EVALUATED,), ("j",)]
        assert phases == ["generation", scoring_phase]

    @pytest.mark.parametrize(
        ("name", "pins_workers"), [("acc_len", True), ("lm_eval", False)]
    )
    def test_single_evaluator_runs_once_on_the_evaluated_model(
        self, name: str, pins_workers: bool
    ):
        evaluator = object.__new__(EVALUATORS[name])
        runs: list[str] = []
        evaluator.run = lambda: runs.append(name)

        stages = evaluator.stages()
        for stage in stages:
            stage.run()

        assert [(stage.engine_keys, stage.pins_workers) for stage in stages] == [
            ((EVALUATED,), pins_workers)
        ]
        assert runs == [name]
