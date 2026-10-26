from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...config_model import ModelInfoConfig, SweBenchEvaluatorConfig
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import MetricsEvaluator
from ...subprocess_manager import SubprocessManager
from .harbor import task_rewards


class SweBenchEvaluator(MetricsEvaluator):
    LATENCY = True

    def setup(
        self,
        autometric_runs_config: SweBenchEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._create_metrics_dir(self.artifacts_path, "swe_bench")
        self.node_runner_root = Path(__file__).resolve().parents[3]
        self.metrics_path = self.metrics_dir / "metrics.json"

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["RLI_API_BASE"] = self.get_rli_url()
        repo_root = self.node_runner_root.parent
        python_paths = [
            str(repo_root / "contrib" / "gigarli" / "src"),
            str(repo_root / "contrib"),
        ]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = ":".join(python_paths)
        return env

    def _build_cmd(self, *, skip_primary_run: bool = False) -> list[str]:
        if self.autometric_runs_config.model_api_base is None:
            raise ValueError("SWE-Bench model_api_base must be set before run")

        config = self.autometric_runs_config.to_dict()
        if skip_primary_run:
            config["skip-primary-run"] = True
        config_json = json.dumps(config)
        return [
            self.autometric_runs_config.venv_python,
            "-m",
            "swe_bench_harbor_rli.cli",
            "run",
            "--metrics-dir",
            str(self.metrics_dir),
            "--config-json",
            config_json,
        ]

    def _should_run_post_reverify_only(self) -> bool:
        return (
            self.metrics_path.exists()
            and self.autometric_runs_config.post_reverify
            and not self.autometric_runs_config.force_recalc
        )

    def _is_already_calculated(self) -> bool:
        if self.autometric_runs_config.force_recalc:
            return False
        if self._should_run_post_reverify_only():
            return False
        return self.metrics_path.exists()

    @property
    def offered_load(self) -> int:
        return self.autometric_runs_config.n_concurrent

    def run(self) -> None:
        post_reverify_only = self._should_run_post_reverify_only()
        if self._is_already_calculated():
            Logger.info(
                f"SWE-bench already calculated at {self.metrics_path}, skipping run"
            )
            return
        cmd = self._build_cmd(skip_primary_run=post_reverify_only)
        if post_reverify_only:
            Logger.info(f"Starting SWE-bench post-reverify for existing metrics: {cmd}")
        else:
            Logger.info(f"Starting SWE-bench run: {cmd}")
        manager = SubprocessManager(
            cmd,
            self.log_file,
            cwd=str(self.node_runner_root),
            run_name="swe_bench",
            env=self._build_env(),
        )
        manager.start()
        manager.wait_for_completion()
        manager.log_error()

    def get_metrics(self) -> dict[str, float]:
        if self.metrics_dir is None or not self.metrics_path.exists():
            Logger.warning(f"SWE-bench metrics file not found at {self.metrics_path}")
            return {}
        with open(self.metrics_path, encoding="utf-8") as metrics_file:
            payload = json.load(metrics_file)
        named_metrics = payload.get("metrics")
        if isinstance(named_metrics, dict):
            return {
                str(name): float(value)
                for name, value in named_metrics.items()
                if isinstance(value, (int, float))
            }
        pass_rate = payload.get("pass_rate")
        if not isinstance(pass_rate, (int, float)):
            Logger.warning(
                f"SWE-bench pass_rate is missing or invalid in {self.metrics_path}"
            )
            return {}
        metrics = {"swe_bench_verified/resolved": float(pass_rate)}
        pass_at_k = payload.get("pass_at_k")
        if isinstance(pass_at_k, dict):
            k = pass_at_k.get("k")
            score = pass_at_k.get("score")
            if isinstance(k, int) and k > 1 and isinstance(score, (int, float)):
                metrics[f"swe_bench_verified/pass@{k}"] = float(score)
        return metrics

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        with open(self.metrics_path, encoding="utf-8") as metrics_file:
            metrics_payload = json.load(metrics_file)
        completed_trials = int(metrics_payload.get("completed") or 0)
        rli_errors = metrics_payload.get("rli_infrastructure_errors") or ()
        pass_at_k = metrics_payload.get("pass_at_k")
        pass_at_k_incomplete = isinstance(pass_at_k, dict) and not pass_at_k.get(
            "complete", False
        )

        if pass_at_k_incomplete:
            Logger.warning(
                "SWE-bench pass@k calculation is incomplete, skipping JD metrics send"
            )
        elif completed_trials < 500:
            Logger.warning(
                "SWE-bench completed "
                f"{completed_trials}/500 trials, skipping JD metrics send"
            )
        elif rli_errors:
            Logger.warning(
                "SWE-bench encountered RLI infrastructure errors "
                f"{sorted(rli_errors)}, skipping JD metrics send"
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
                Path(payload["job_result_path"]), k, int(payload["total"]), binary=True
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            Logger.warning(f"SWE-bench CI unavailable: {error}")
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
                    self.autometric_runs_config.model_dump(
                        mode="json",
                        include={
                            "dataset",
                            "agent",
                            "agent_kwargs",
                            "max_tokens",
                            "max_input_tokens",
                            "max_total_output_tokens",
                            "pass_at_k",
                            "agent_timeout",
                            "env_prefix",
                            "env_version",
                        },
                    ),
                )
        JD.send_evaluator_metrics_with_scores(
            scores=scores,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            shooter_name="swe_bench",
            shooter_version=JD.get_shooter_version("/home/user/swe_bench_version.txt"),
            model_info=self.model_info,
        )
