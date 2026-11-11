import json
import math

from ...config_model import JobBenchEvaluatorConfig
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...logger import Logger
from .harbor_rli_judged import HarborRliJudgedEvaluator


class JobBenchEvaluator(HarborRliJudgedEvaluator):
    BENCHMARK = "jobbench"

    autometric_runs_config: JobBenchEvaluatorConfig

    def _build_shooter_config(self) -> dict:
        return self.autometric_runs_config.to_shooter_config()

    def get_metrics(self) -> dict[str, float]:
        payload = self.read_complete_result()
        if payload is None:
            return {}
        try:
            value = payload["metrics"][
                f"jobbench/{self.autometric_runs_config.split}/score"
            ]
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                Logger.error(f"JobBench returned invalid metrics: {self.metrics_path}")
                return {}
        except (OSError, ValueError, KeyError, TypeError) as error:
            Logger.error(f"Cannot read JobBench metrics {self.metrics_path}: {error}")
            return {}
        config = self.autometric_runs_config
        agent = self._model_slug(config.agent)
        judge = self._model_slug(self.judge_model_name)
        metric_name = (
            f"jobbench/{config.split}/{agent}/"
            f"max_tokens_{config.max_tokens}/timeout_{config.agent_timeout}/"
            f"judge_{judge}/score"
        )
        return {metric_name: float(value)}

    def _cache_scores(self, metrics: dict[str, float]) -> ScalarCacheScoresBuilder:
        scores = super()._cache_scores(metrics)
        payload = json.loads(self.metrics_path.read_text())
        for key in metrics:
            scores.add_metric(
                key,
                [
                    (task_id, 0.0 if reward is None else reward)
                    for task_id, reward in payload["rewards"].items()
                ],
                payload["identity"],
            )
        return scores
