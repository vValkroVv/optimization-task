import json
import math
import os
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import cast

from ...anonymizer import Anonymizer
from ...config_model import (
    HarborRliJudgedEvaluatorConfig,
    MetricsEvaluatorConfig,
    ModelInfoConfig,
)
from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...subprocess_manager import SubprocessManager
from .interface import ScoringMetricsEvaluator


class HarborRliJudgedEvaluator(ScoringMetricsEvaluator):
    SCORING_PHASE = "scoring"
    BENCHMARK: str
    METRICS_DIR_NAME: str | None = None
    autometric_runs_config: HarborRliJudgedEvaluatorConfig

    def setup(
        self,
        autometric_runs_config: MetricsEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = cast(
            HarborRliJudgedEvaluatorConfig, autometric_runs_config
        )
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._create_metrics_dir(
            self.artifacts_path, self.METRICS_DIR_NAME or self.BENCHMARK
        )
        self.metrics_path = self.metrics_dir / "metrics.json"
        self._phase = "generation"
        self._failed = False
        self.shooter_version = ""
        self._version_error: str | None = None
        version_path = (
            Path(self.autometric_runs_config.shooter_root)
            / "harbor_rli/harbor_rli_version.txt"
        )
        try:
            versions = json.loads(version_path.read_text())
            rli_version = versions.get("rli")
            benchmark_version = versions.get("benchmarks", {}).get(self.BENCHMARK)
            if not all(
                isinstance(value, str) and value.strip()
                for value in (rli_version, benchmark_version)
            ):
                raise ValueError("Shooter version components must be nonempty strings")
            self.shooter_version = rli_version + "+" + benchmark_version
        except (OSError, ValueError, AttributeError, TypeError) as error:
            self._version_error = f"{self.BENCHMARK} shooter version unavailable at {version_path}: {error}"
            Logger.error(self._version_error)

    @property
    def judge_model_name(self) -> str:
        return self.autometric_runs_config.judge_model_name

    def run_phase(self, phase: str) -> None:
        if self._failed:
            raise RuntimeError(f"{self.BENCHMARK}: previous phase failed")
        self._phase = phase
        self.run()
        if self._failed:
            raise RuntimeError(f"{self.BENCHMARK} {phase} failed")

    @abstractmethod
    def _build_shooter_config(self) -> dict: ...

    def run(self) -> None:
        self._failed = True
        if self._version_error is not None:
            raise RuntimeError(self._version_error)
        config = self._build_shooter_config()
        root = Path(self.autometric_runs_config.shooter_root)
        env = dict(os.environ)
        env["RLI_API_BASE"] = self.get_rli_url()
        env["PYTHONPATH"] = str(root)
        with tempfile.TemporaryDirectory(prefix=f"{self.BENCHMARK}_") as runtime:
            config_path = Path(runtime) / "config.json"
            config_path.write_text(json.dumps(config))
            config_path.chmod(0o600)
            raw_log = Path(runtime) / "subprocess.log"
            manager = SubprocessManager(
                [
                    self.autometric_runs_config.venv_python,
                    "-m",
                    "harbor_rli.cli",
                    "run",
                    "--benchmark",
                    self.BENCHMARK,
                    "--metrics-dir",
                    str(self.metrics_dir),
                    "--phase",
                    self._phase,
                    "--config",
                    str(config_path),
                ],
                raw_log,
                cwd=str(root),
                run_name=f"{self.BENCHMARK}_{self._phase}",
                env=env,
            )
            try:
                manager.start()
                manager.wait_for_completion()
            finally:
                if raw_log.exists():
                    Anonymizer.anonymize_file(raw_log, self.log_file, append=True)
                manager.log_file = self.log_file
                manager.log_error()
            succeeded = manager.is_successful()
        self._failed = not succeeded

    def read_complete_result(self) -> dict | None:
        if self._phase != self.SCORING_PHASE or self._failed:
            return None
        try:
            payload = json.loads(self.metrics_path.read_text())
            if not isinstance(payload, dict) or payload.get("complete") is not True:
                Logger.error(
                    f"{self.BENCHMARK} result is incomplete: {self.metrics_path}"
                )
                return None
            metrics = payload["metrics"]
            if (
                not isinstance(metrics, dict)
                or not metrics
                or any(
                    not isinstance(key, str)
                    or not key.startswith(f"{self.BENCHMARK}/")
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for key, value in metrics.items()
                )
            ):
                Logger.error(f"{self.BENCHMARK} returned invalid metrics")
                return None
            return payload
        except (OSError, ValueError, KeyError, TypeError) as error:
            Logger.error(f"Cannot read {self.BENCHMARK} metrics: {error}")
            return None

    def _jd_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return {
            name.replace("/", "_").upper(): value for name, value in metrics.items()
        }

    def _cache_scores(self, metrics: dict[str, float]) -> ScalarCacheScoresBuilder:
        return ScalarCacheScoresBuilder(self.metrics_dir, dict(metrics))

    def after_run(
        self, metrics: dict[str, float], uploaded: MeasurementUpload | None = None
    ) -> None:
        if (
            not self._failed
            and self._phase == self.SCORING_PHASE
            and JD.can_send_metrics(
                metrics=metrics,
                hf_checkpoint_path=self.hf_checkpoint_path,
                metrics_dir=self.metrics_dir,
                model_info=self.model_info,
            )
        ):
            jd_metrics = self._jd_metrics(metrics)
            JD.send_evaluator_metrics_with_scores(
                scores=self._cache_scores(jd_metrics),
                metrics_dir=self.metrics_dir,
                hf_checkpoint_path=self.hf_checkpoint_path,
                shooter_name=self.BENCHMARK,
                shooter_version=self.shooter_version,
                model_info=self.model_info,
            )
        archive = JD.tar_metrics_dir(
            self.metrics_dir, self.artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            archive, self.hf_checkpoint_path, type(self).__name__
        )
