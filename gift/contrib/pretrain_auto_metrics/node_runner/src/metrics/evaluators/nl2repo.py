from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ...config_model import ModelInfoConfig, Nl2RepoEvaluatorConfig
from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import MetricsEvaluator
from ...subprocess_manager import SubprocessManager
from .harbor import task_rewards

NL2REPO_VERSION_FILE = "/home/user/nl2repo/nl2repo_version.txt"


class Nl2RepoEvaluator(MetricsEvaluator):
    LATENCY = True

    def setup(
        self,
        autometric_runs_config: Nl2RepoEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._create_metrics_dir(self.artifacts_path, "nl2repo")
        self.node_runner_root = Path(__file__).resolve().parents[3]
        self.metrics_path = self.metrics_dir / "metrics.json"
        self.fingerprint_path = self.metrics_dir / "nl2repo_setup.json"

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["RLI_API_BASE"] = self.get_rli_url()
        repo_root = self.node_runner_root.parent
        python_paths = [
            "/home/user/terminal_bench_env",
            "/home/user/nl2repo",
            str(repo_root / "contrib"),
        ]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(python_paths)
        return env

    def _build_cmd(self) -> list[str]:
        return [
            self.autometric_runs_config.venv_python,
            "-m",
            "nl2repo.cli",
            "run",
            "--metrics-dir",
            str(self.metrics_dir),
            "--config-json",
            json.dumps(self.autometric_runs_config.to_dict()),
        ]

    def _fingerprint(self) -> dict[str, object]:
        config = self.autometric_runs_config
        methodology = {
            "allow-internet-access": config.allow_internet_access,
            "agent": config.agent,
            "agent-kwargs": config.agent_kwargs,
            "enable-contamination-filter": config.enable_contamination_filter,
            "max-tokens": config.max_tokens,
            "max-input-tokens": config.max_input_tokens,
            "max-total-output-tokens": config.max_total_output_tokens,
            "pass-at-k": config.pass_at_k,
            "n-tasks": config.n_tasks,
            "task-name": config.task_name,
            "exclude-task-name": config.exclude_task_name,
            "env-prefix": config.env_prefix,
            "agent-env-version": config.agent_env_version,
            "verifier-env-version": config.verifier_env_version,
            "verifier-timeout": config.verifier_timeout,
        }
        if config.agent_timeout is not None:
            methodology["agent-timeout"] = config.agent_timeout
        else:
            methodology["n-concurrent"] = config.n_concurrent
        return {
            "checkpoint": str(self.hf_checkpoint_path),
            "config": methodology,
            "shooter_version": JD.get_shooter_version(NL2REPO_VERSION_FILE),
        }

    def _load_metrics_payload(self) -> dict | None:
        if self.metrics_dir is None or not self.metrics_path.exists():
            Logger.error(f"NL2Repo metrics file not found at {self.metrics_path}")
            return None
        try:
            payload = self._load_json_file(self.metrics_path)
        except (OSError, TypeError, ValueError) as error:
            Logger.error(
                f"Failed to read NL2Repo metrics at {self.metrics_path}: {error}"
            )
            return None
        if not isinstance(payload, dict):
            Logger.error(f"NL2Repo metrics at {self.metrics_path} are not an object")
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
            if isinstance(value, (int, float))
        }

    @classmethod
    def _is_publishable_payload(cls, payload: dict) -> bool:
        run_result = payload.get("run")
        if not isinstance(run_result, dict):
            return False
        planned = run_result.get("planned")
        return (
            payload.get("complete") is True
            and isinstance(planned, int)
            and planned > 0
            and run_result.get("infrastructure_errors") == 0
            and run_result.get("api_errors") == 0
            and bool(cls._numeric_metrics(payload))
        )

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
        return int(self.autometric_runs_config.n_concurrent or 1)

    def run(self) -> None:
        if self._is_already_calculated():
            Logger.info(
                f"NL2Repo already calculated at {self.metrics_path}, skipping run"
            )
            return
        self.metrics_path.unlink(missing_ok=True)
        self.fingerprint_path.unlink(missing_ok=True)
        cmd = self._build_cmd()
        Logger.info(f"Starting NL2Repo run: {cmd}")
        manager = SubprocessManager(
            cmd,
            self.log_file,
            cwd=str(self.node_runner_root),
            run_name="nl2repo",
            env=self._build_env(),
        )
        manager.start()
        manager.wait_for_completion()
        manager.log_error()
        payload = self._load_metrics_payload()
        if payload is not None and self._is_publishable_payload(payload):
            self.fingerprint_path.write_text(
                json.dumps(self._fingerprint(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        Logger.info("NL2Repo run completed")

    def get_metrics(self) -> dict[str, float]:
        payload = self._load_metrics_payload()
        if payload is None:
            return {}
        return self._numeric_metrics(payload)

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        payload = self._load_metrics_payload()
        if payload is None:
            return
        config = self.autometric_runs_config
        if payload.get("publish_metrics") is not True:
            Logger.warning("NL2Repo result is incomplete, skipping JD metrics send")
        elif any(
            selector is not None
            for selector in (
                config.n_tasks,
                config.task_name,
                config.exclude_task_name,
            )
        ):
            Logger.warning(
                "NL2Repo filtered run is not comparable with the full benchmark, "
                "skipping JD metrics send"
            )
        else:
            self._send_metrics_to_jd(metrics)
        tar_path = JD.tar_metrics_dir(
            self.metrics_dir, self.artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            tar_path, self.hf_checkpoint_path, type(self).__name__
        )

    def _cache_scores(self, metrics: dict[str, float]) -> ScalarCacheScoresBuilder:
        scores = ScalarCacheScoresBuilder(self.metrics_dir, metrics)
        payload = self._load_metrics_payload()
        if payload is None or not self._is_publishable_payload(payload):
            return scores
        k = self.autometric_runs_config.pass_at_k
        try:
            result = payload["run"]
            run_dir = Path(result["ci_run_dir"])
            rewards = task_rewards(run_dir / "result.json", k, result["planned"])
            tasks = {uid.rsplit("/", 1)[-1]: values for uid, values in rewards.items()}
            test_counts = result["ci_test_counts"]
            if len(tasks) != len(rewards) or set(tasks) != set(test_counts):
                Logger.warning("NL2Repo CI unavailable: task roster mismatch")
                return scores
            identity = [self._fingerprint()["config"], test_counts]
            for key in metrics:
                if key.endswith("_MEAN_SCORE"):
                    scores.add_metric(
                        key,
                        [(uid, sum(values) / k) for uid, values in tasks.items()],
                        identity,
                    )
            passed = {uid: 0 for uid in tasks}
            for path in sorted(run_dir.glob("*/result.json")):
                trial = json.loads(path.read_text())
                uid = trial["task_name"].rsplit("/", 1)[-1]
                summary = json.loads(
                    (path.parent / "verifier" / "pytest_summary.json").read_text()
                )
                if summary["test_case_count"] != test_counts[uid]:
                    Logger.warning(
                        "NL2Repo pass-rate CI unavailable: test roster mismatch"
                    )
                    return scores
                passed[uid] += min(int(summary.get("passed") or 0), test_counts[uid])
            denominator = k * sum(test_counts.values())
            if denominator > 0:
                for key in metrics:
                    if key.endswith("_PASS_RATE"):
                        scores.add_ratio_metric(
                            key,
                            [
                                (uid, value, k * test_counts[uid])
                                for uid, value in passed.items()
                            ],
                            identity,
                        )
        except (OSError, ValueError, KeyError, TypeError) as error:
            Logger.warning(f"NL2Repo CI unavailable: {error}")
        return scores

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
        JD.send_evaluator_metrics_with_scores(
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            scores=self._cache_scores(jd_metrics),
            shooter_name="nl2repo",
            shooter_version=JD.get_shooter_version(NL2REPO_VERSION_FILE),
            model_info=self.model_info,
        )
