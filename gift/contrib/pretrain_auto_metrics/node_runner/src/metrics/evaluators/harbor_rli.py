from __future__ import annotations

import json
import math
import os
from pathlib import Path

from ...config_model import HarborRliEvaluatorConfig, ModelInfoConfig
from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import MetricsEvaluator
from ...subprocess_manager import SubprocessManager
from ...subprocess_pool_runner import SubprocessPoolRunner
from .harbor import task_rewards


class HarborRliEvaluator(MetricsEvaluator):
    LATENCY = True

    NAME = "harbor_rli"
    VERSION_FILE = "/home/user/harbor_rli/harbor_rli_version.txt"
    ENV_ROOT = "/home/user/harbor_rli_env"
    REWARD_KEY: str | None = None
    SHOOTER_NAMES = {
        "cowork-bench": "harbor_rli",
        "harness-bench-lite": "harness_bench",
        "harness-bench-medium": "harness_bench",
        "swe-bench-verified": "swe_bench",
        "swe-bench-multilingual": "swe_bench",
        "terminal-bench-2.0": "terminal_bench",
        "terminal-bench-2.1": "terminal_bench",
    }

    def setup(
        self,
        autometric_runs_config: HarborRliEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
        *,
        _metrics_root: Path | None = None,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self.node_runner_root = Path(__file__).resolve().parents[3]
        self._runs: list[HarborRliEvaluator] = []
        self._version_error = False
        if autometric_runs_config.runs is not None:
            self._create_metrics_dir(self.artifacts_path, self.NAME)
            for config in autometric_runs_config.runs:
                evaluator = type(self)()
                evaluator.setup(
                    config,
                    artifacts_path,
                    hf_checkpoint_path,
                    model_info,
                    _metrics_root=self.metrics_dir,
                )
                self._runs.append(evaluator)
            return
        if _metrics_root is None:
            self._create_metrics_dir(self.artifacts_path, self.NAME)
            _metrics_root = self.metrics_dir
        self.metrics_dir = _metrics_root / autometric_runs_config.benchmark
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.shooter_version = JD.get_shooter_version(self.VERSION_FILE)
        if self.shooter_version.startswith("{"):
            try:
                versions = json.loads(self.shooter_version)
                self.shooter_version = (
                    versions.get("rli")
                    + "+"
                    + versions.get("benchmarks", {}).get(
                        autometric_runs_config.benchmark
                    )
                )
            except (json.JSONDecodeError, AttributeError, TypeError) as error:
                self._version_error = True
                Logger.error(
                    f"Skipping {self.NAME} benchmark={autometric_runs_config.benchmark}: "
                    f"invalid shooter version in {self.VERSION_FILE}: {error}"
                )
        self.metrics_path = self.metrics_dir / "metrics.json"
        self.fingerprint_path = self.metrics_dir / f"{self.NAME}_setup.json"

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["RLI_API_BASE"] = self.get_rli_url()
        python_paths = [self.ENV_ROOT]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(python_paths)
        return env

    def _build_cmd(self) -> list[str]:
        payload = self.autometric_runs_config.to_dict()
        benchmark = payload.pop("benchmark")
        payload.update(payload.pop("benchmark-kwargs"))
        config_json = json.dumps(payload)
        return [
            self.autometric_runs_config.venv_python,
            "-m",
            f"{self.NAME}.cli",
            "run",
            "--benchmark",
            benchmark,
            "--metrics-dir",
            str(self.metrics_dir),
            "--config-json",
            config_json,
        ]

    def _fingerprint(self) -> dict[str, object]:
        return {
            "checkpoint": str(self.hf_checkpoint_path),
            "config": self.autometric_runs_config.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
                exclude={"runs", "job_name", "model_api_base"},
            ),
            "shooter_version": self.shooter_version,
        }

    def _load_metrics_payload(self) -> dict | None:
        if self._version_error:
            return None
        if self.metrics_dir is None or not self.metrics_path.exists():
            Logger.error(f"{self.NAME} metrics file not found at {self.metrics_path}")
            return None
        try:
            payload = self._load_json_file(self.metrics_path)
        except (OSError, TypeError, ValueError) as error:
            Logger.error(
                f"Failed to read {self.NAME} metrics at {self.metrics_path}: {error}"
            )
            return None
        if not isinstance(payload, dict):
            Logger.error(
                f"{self.NAME} metrics at {self.metrics_path} are not an object"
            )
            return None
        return payload

    @staticmethod
    def _numeric_metrics(payload: dict) -> dict[str, float]:
        named_metrics = payload.get("metrics")
        if not isinstance(named_metrics, dict):
            return {}
        return {
            str(name): float(value)
            for name, value in named_metrics.items()
            if type(value) in (int, float) and math.isfinite(value)
        }

    @classmethod
    def _is_publishable_payload(cls, payload: dict) -> bool:
        return payload.get("complete") is True and bool(cls._numeric_metrics(payload))

    def _is_already_calculated(self) -> bool:
        if self.autometric_runs_config.force_recalc:
            return False
        if not self.metrics_path.exists() or not self.fingerprint_path.exists():
            return False
        payload = self._load_metrics_payload()
        if payload is None or not self._is_publishable_payload(payload):
            return False
        try:
            return self._load_json_file(self.fingerprint_path) == self._fingerprint()
        except (OSError, TypeError, ValueError):
            return False

    @property
    def offered_load(self) -> int:
        if self.autometric_runs_config.runs is not None:
            return sum(
                int(config.n_concurrent or 1)
                for config in self.autometric_runs_config.runs
            )
        return int(self.autometric_runs_config.n_concurrent or 1)

    def _run_multiple(self) -> None:
        pending = [
            run
            for run in self._runs
            if not run._version_error and not run._is_already_calculated()
        ]
        if not pending:
            Logger.info(f"No pending {self.NAME} runs")
            return
        commands = {run.metrics_dir.name: run._build_cmd() for run in pending}
        for run in pending:
            run.metrics_path.unlink(missing_ok=True)
            run.fingerprint_path.unlink(missing_ok=True)
        pool = SubprocessPoolRunner(
            cmds_to_run=commands,
            max_parallel=len(pending),
            autometrics_log_file=self.log_file,
            cwd=str(self.node_runner_root),
            env=self._build_env(),
        )
        pool.run_all()
        for run in pending:
            run._save_fingerprint()

    def run(self) -> None:
        if self._version_error:
            return
        if self._runs:
            self._run_multiple()
            return
        if self._is_already_calculated():
            Logger.info(f"{self.NAME} already calculated at {self.metrics_path}")
            return
        self.metrics_path.unlink(missing_ok=True)
        self.fingerprint_path.unlink(missing_ok=True)
        cmd = self._build_cmd()
        Logger.info(f"Starting {self.NAME} run: {cmd}")
        manager = SubprocessManager(
            cmd,
            self.log_file,
            cwd=str(self.node_runner_root),
            run_name=self.NAME,
            env=self._build_env(),
        )
        manager.start()
        manager.wait_for_completion()
        manager.log_error()
        self._save_fingerprint()
        Logger.info(f"{self.NAME} run completed")

    def _save_fingerprint(self) -> None:
        payload = self._load_metrics_payload()
        if payload is not None and self._is_publishable_payload(payload):
            self.fingerprint_path.write_text(
                json.dumps(self._fingerprint(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def get_metrics(self) -> dict[str, float]:
        if self._runs:
            metrics = {}
            for run in self._runs:
                for name, value in run.get_metrics().items():
                    if name in metrics:
                        Logger.error(
                            f"Duplicate {self.NAME} metric name: {name} "
                            f"(benchmark={run.autometric_runs_config.benchmark})"
                        )
                    metrics[name] = value
            return metrics
        payload = self._load_metrics_payload()
        if payload is None or not self._is_publishable_payload(payload):
            return {}
        return self._numeric_metrics(payload)

    def after_run(
        self, metrics: dict[str, float], uploaded: MeasurementUpload | None = None
    ) -> None:
        if self._version_error:
            return
        if self._runs:
            for run in self._runs:
                run.after_run(run.get_metrics())
            return
        payload = self._load_metrics_payload()
        if payload is None or not self._is_publishable_payload(payload):
            Logger.warning(
                f"{self.NAME} result is incomplete, skipping JD metrics send"
            )
        else:
            self._send_metrics_to_jd(metrics)
        tar_path = JD.tar_metrics_dir(
            self.metrics_dir, self.artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            tar_path, self.hf_checkpoint_path, type(self).__name__
        )

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return

        jd_metrics = {
            metric_name.replace("/", "_").upper(): value
            for metric_name, value in processed_metrics.items()
        }
        scores = ScalarCacheScoresBuilder(self.metrics_dir, jd_metrics)
        try:
            payload = json.loads(self.metrics_path.read_text())
            k = self.autometric_runs_config.pass_at_k
            rewards = task_rewards(
                Path(payload["job_result_path"]),
                k,
                int(payload["total"]),
                binary=True,
                reward_key=self.REWARD_KEY,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            Logger.warning(f"{self.NAME} CI unavailable: {error}")
        else:
            for key in jd_metrics:
                if key.endswith((f"_PASS_AT_{k}", f"_PASS@{k}")):
                    values = [
                        (uid, float(any(trials))) for uid, trials in rewards.items()
                    ]
                else:
                    values = [(uid, sum(trials) / k) for uid, trials in rewards.items()]
                scores.add_metric(
                    key,
                    values,
                    self._fingerprint()["config"],
                )
        JD.send_evaluator_metrics_with_scores(
            scores=scores,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            shooter_name=self.SHOOTER_NAMES.get(
                self.autometric_runs_config.benchmark,
                self.autometric_runs_config.benchmark.replace("-", "_"),
            ),
            shooter_version=f"{self.shooter_version}-{self.autometric_runs_config.benchmark}",
            model_info=self.model_info,
        )
