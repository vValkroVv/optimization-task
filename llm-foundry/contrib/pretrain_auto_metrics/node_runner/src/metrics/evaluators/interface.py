from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores import CacheScoresBuilder
from ...job_dispatcher.measurement import MeasurementFiles, MeasurementUpload
from ...config_model import (
    MetricsEvaluatorConfig,
    ModelInfoConfig,
)
from ...trace.events.model import PhaseName

EVALUATED = "evaluated"


@dataclass(frozen=True)
class Stage:
    engine_keys: tuple[str, ...]
    run: Callable[[], None]
    pins_workers: bool = False

    @property
    def judge(self) -> str | None:
        for key in self.engine_keys:
            if key != EVALUATED:
                return key
        return None

    @property
    def phase(self) -> PhaseName:
        uses_evaluated = EVALUATED in self.engine_keys
        uses_judge = self.judge is not None
        if not uses_evaluated:
            return "scoring"
        if uses_judge:
            return "parallel"
        return "single"


class MetricsEvaluator(ABC):
    LATENCY = False

    def __init__(self):
        self.autometric_runs_config: MetricsEvaluatorConfig | None = None
        self.artifacts_path = None
        self.metrics_dir = None
        self.measurement_files: MeasurementFiles | None = None
        self.cmds_to_run = None

    @staticmethod
    def _load_json_file(file_path: Path) -> dict:
        with open(file_path, "r") as f:
            return json.load(f)

    def _create_metrics_dir(self, artifacts_path: Path, evaluator_name: str) -> None:
        self.metrics_dir = Path(artifacts_path, "metrics", evaluator_name)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def _create_measurement_files(
        self, hf_checkpoint_path: Path, evaluator_name: str
    ) -> None:
        self.measurement_files = MeasurementFiles(
            self.metrics_dir,
            JD.read_metrics_files_mtime_ns(
                JD.checkpoint_artifacts_path(hf_checkpoint_path), evaluator_name
            ),
        )

    def measured_files_mtime_ns(self) -> Optional[int]:
        return self.measurement_files.mtime_ns() if self.measurement_files else None

    @property
    def log_file(self) -> Path:
        return self.metrics_dir / "autometrics.log"

    def get_rli_url(self) -> str:
        if JD.internet_is_available():
            return "https://rli.sberdevices.ru"
        return "https://gym.static.space"

    def get_yt_url(self) -> str:
        if JD.internet_is_available():
            return "http://yt.eye.salute.ru:8888"
        return "http://saurus.static.space:8888"

    @abstractmethod
    def setup(
        self,
        autometric_runs_config: MetricsEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        """
        Method for setting up the evaluator of metrics.

        :param: autometric_runs_config: MetricsEvaluatorConfig
        :param: artifacts_path: Path
        :param: hf_checkpoint_path: Path
        :param: model_info: ModelInfoConfig
        :return: None
        """
        ...

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        """
        Method for parsing/converting metrics to dict for saving in wandb/tensorboard etc.

        Format: key: metric_name, value: metric_value

        Example:
        {
            "mmlu_pro/biology/5-shot/exact_match.custom-extract": 0.6387726638772664,
            "mmlu_pro/business/5-shot/exact_match.custom-extract": 0.44106463878326996,
            "mmlu_pro/health/5-shot/exact_match.custom-extract": 0.4877750611246944,
            "hellaswag/0-shot/acc": 0.5121951219512195,
            "gsm8k/4-shot/exact_match.strict-match": 0.7422289613343442,
            "gsm8k/4-shot/exact_match.flexible-extract": 0.7422289613343442,
            "hendrycks_math/4-shot/exact_match": 0.2272,
            "openbookqa/5-shot/acc": 0.392,
            "openbookqa/5-shot/acc_norm": 0.488,
        }

        :param: None

        :return: dict[str, float]
        """
        ...

    @abstractmethod
    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        """
        Method for cleaning up after the evaluator of metrics.

        :param: metrics: dict[str, float]

        :return: None
        """
        ...

    def stages(self) -> list[Stage]:
        return [Stage((EVALUATED,), self.run)]

    def make_cache_scores_builder(
        self, run_files: Optional[set[Path]] = None
    ) -> Optional[CacheScoresBuilder]:
        return None

    def measurement_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return metrics

    @staticmethod
    def _model_slug(model_name: str) -> str:
        bare_name = model_name.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
        return re.sub(r"[^a-z0-9]+", "_", bare_name.lower()).strip("_")


class ScoringMetricsEvaluator(MetricsEvaluator):
    SCORING_PHASE = "evaluation"

    def stages(self) -> list[Stage]:
        (judge,) = self.autometric_runs_config.secondary_model_names
        generation = partial(self.run_phase, "generation")
        scoring = partial(self.run_phase, self.SCORING_PHASE)
        return [
            Stage((EVALUATED,), generation),
            Stage((judge,), scoring),
        ]

    @abstractmethod
    def run_phase(self, phase: str) -> None: ...


class ParallelInferenceEvaluator(MetricsEvaluator):
    def stages(self) -> list[Stage]:
        stages = []
        for judge in self.autometric_runs_config.secondary_model_names:
            run_with_judge = partial(self.run_on_model_name, judge)
            stages.append(Stage((EVALUATED, judge), run_with_judge))
        return stages

    @abstractmethod
    def run_on_model_name(
        self,
        model_name: str,
    ) -> None: ...
