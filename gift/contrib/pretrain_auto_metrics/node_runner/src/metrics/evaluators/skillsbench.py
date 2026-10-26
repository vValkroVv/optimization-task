from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ...config_model import ModelInfoConfig, SkillsBenchEvaluatorConfig
from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import MetricsEvaluator
from ...subprocess_manager import SubprocessManager
from .harbor import task_rewards


SKILLSBENCH_VERSION_FILE = "/home/user/skillsbench/skillsbench_version.txt"


class SkillsBenchEvaluator(MetricsEvaluator):
    LATENCY = True

    def setup(
        self,
        autometric_runs_config: SkillsBenchEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._create_metrics_dir(self.artifacts_path, "skillsbench")
        self.node_runner_root = Path(__file__).resolve().parents[3]
        self.metrics_path = self.metrics_dir / "metrics.json"
        self.fingerprint_path = self.metrics_dir / "skillsbench_setup.json"

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["RLI_API_BASE"] = self.get_rli_url()
        repo_root = self.node_runner_root.parent
        python_paths = [
            "/home/user/terminal_bench_env",
            "/home/user/skillsbench",
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
            "skillsbench.cli",
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
            "conditions": config.conditions,
            "text-only": config.text_only,
            "max-tokens": config.max_tokens,
            "max-input-tokens": config.max_input_tokens,
            "max-total-output-tokens": config.max_total_output_tokens,
            "pass-at-k": config.pass_at_k,
            "n-tasks": config.n_tasks,
            "task-name": config.task_name,
            "exclude-task-name": config.exclude_task_name,
            "env-prefix": config.env_prefix,
            "env-version": config.env_version,
        }
        if config.agent_timeout is not None:
            methodology["agent-timeout"] = config.agent_timeout
        else:
            methodology["n-concurrent"] = config.n_concurrent
        return {
            "checkpoint": str(self.hf_checkpoint_path),
            "config": methodology,
            "shooter_version": JD.get_shooter_version(SKILLSBENCH_VERSION_FILE),
        }

    def _load_metrics_payload(self) -> dict | None:
        if self.metrics_dir is None or not self.metrics_path.exists():
            Logger.error(f"SkillsBench metrics file not found at {self.metrics_path}")
            return None
        try:
            payload = self._load_json_file(self.metrics_path)
        except (OSError, TypeError, ValueError) as error:
            Logger.error(
                f"Failed to read SkillsBench metrics at {self.metrics_path}: {error}"
            )
            return None
        if not isinstance(payload, dict):
            Logger.error(
                f"SkillsBench metrics at {self.metrics_path} are not an object"
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
            if isinstance(value, (int, float))
        }

    @classmethod
    def _is_publishable_payload(cls, payload: dict) -> bool:
        conditions = payload.get("conditions")
        return (
            payload.get("complete") is True
            and isinstance(conditions, dict)
            and bool(conditions)
            and all(
                isinstance(condition, dict)
                and condition.get("rli_errors") == 0
                and condition.get("api_errors") == 0
                for condition in conditions.values()
            )
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
            Logger.info(f"SkillsBench already calculated at {self.metrics_path}")
            return
        self.metrics_path.unlink(missing_ok=True)
        self.fingerprint_path.unlink(missing_ok=True)
        cmd = self._build_cmd()
        Logger.info(f"Starting SkillsBench run: {cmd}")
        manager = SubprocessManager(
            cmd,
            self.log_file,
            cwd=str(self.node_runner_root),
            run_name="skillsbench",
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
        Logger.info("SkillsBench run completed")

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
            Logger.warning("SkillsBench result is incomplete, skipping JD metrics send")
        elif any(
            selector is not None
            for selector in (
                config.n_tasks,
                config.task_name,
                config.exclude_task_name,
            )
        ):
            Logger.warning(
                "Filtered SkillsBench run is not comparable with the full benchmark, "
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
        for condition, result in payload["conditions"].items():
            try:
                rewards = task_rewards(
                    Path(result["ci_run_dir"]) / "result.json", k, result["planned"]
                )
                tasks = {
                    uid.rsplit("/", 1)[-1]: values for uid, values in rewards.items()
                }
                domains = result["ci_task_domains"]
                if len(tasks) != len(rewards) or set(tasks) != set(domains):
                    Logger.warning(
                        f"SkillsBench CI unavailable for {condition}: task roster mismatch"
                    )
                    continue
                identity = [self._fingerprint()["config"], condition, domains]
                scopes = {"": tasks}
                scopes.update(
                    {
                        f"DOMAIN_{domain.upper()}_": {
                            uid: values
                            for uid, values in tasks.items()
                            if domains[uid] == domain
                        }
                        for domain in set(domains.values())
                    }
                )
                for scope, selected in scopes.items():
                    prefix = (
                        f"SKILLSBENCH_{condition.replace('-', '_').upper()}_{scope}"
                    )
                    for key in metrics:
                        if not key.startswith(prefix):
                            continue
                        suffix = key[len(prefix) :]
                        if suffix.startswith("SCORE_"):
                            samples = [
                                (uid, len(selected) * sum(values) / k / 100)
                                for uid, values in selected.items()
                            ]
                        elif suffix.startswith("RESOLUTION_RATE_"):
                            samples = [
                                (uid, sum(values) / k)
                                for uid, values in selected.items()
                            ]
                        elif suffix.startswith(f"PASS@{k}_"):
                            samples = [
                                (uid, float(any(value == 1 for value in values)))
                                for uid, values in selected.items()
                            ]
                        else:
                            continue
                        scores.add_metric(key, samples, identity)
            except (OSError, ValueError, KeyError, TypeError) as error:
                Logger.warning(f"SkillsBench CI unavailable for {condition}: {error}")
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
            metric_name: (
                value / 100
                if "_SCORE_" in metric_name or "_RESOLUTION_RATE_" in metric_name
                else value
            )
            for metric_name, value in processed_metrics.items()
            if metric_name.startswith(
                ("SKILLSBENCH_NO_SKILL_", "SKILLSBENCH_WITH_SKILL_")
            )
        }
        JD.send_evaluator_metrics_with_scores(
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            scores=self._cache_scores(jd_metrics),
            shooter_name="skillsbench",
            shooter_version=JD.get_shooter_version(SKILLSBENCH_VERSION_FILE),
            model_info=self.model_info,
        )
