import json
import re
from pathlib import Path

from ...config_model import ModelInfoConfig, Tau3EvaluatorConfig
from ...job_dispatcher import JD, SubmetricGroup
from ...logger import Logger
from .tau2 import Tau2Evaluator


class Tau3Evaluator(Tau2Evaluator):
    def setup(
        self,
        autometric_runs_config: Tau3EvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self.all_metrics: dict[str, float] = {}
        self._create_metrics_dir(self.artifacts_path, "tau3")

        self.tau2_root = self._resolve_bench_root(
            autometric_runs_config.tau2_bench_root
        )
        self.tau2_src = self.tau2_root / "src"
        self.cmds_to_run = self._build_commands_for_domains(self.autometric_runs_config)

        Logger.info(
            f"Tau3Evaluator setup: domains={autometric_runs_config.domains}, "
            f"user_model={autometric_runs_config.user_model_name}, "
            f"agent_api_base={autometric_runs_config.agent_api_base}, "
            f"user_api_base={autometric_runs_config.user_api_base}, "
            f"max_parallel_domains={autometric_runs_config.max_parallel_domains}"
        )

    def run_on_model_name(self, model_name: str) -> None:
        self._run_domains()

        failed_domains: list[str] = []
        for domain, save_to in self._save_to_by_domain.items():
            results_path = Path(save_to) / "results.json"
            try:
                metrics = self._parse_mean_reward_from_results(results_path, domain)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                Logger.error(f"tau3-bench domain {domain} failed: {exc}")
                metrics = {}
            if not metrics:
                failed_domains.append(domain)
            self.all_metrics.update(metrics)

        if failed_domains:
            raise RuntimeError(
                f"tau3-bench produced no metrics for: {', '.join(failed_domains)}"
            )

    def _parse_mean_reward_from_results(
        self, results_path: Path, domain: str
    ) -> dict[str, float]:
        with open(results_path, "r") as f:
            data = json.load(f)
        simulations = data.get("simulations", [])
        if not simulations:
            Logger.error(f"tau3-bench results at {results_path} are empty")
            return {}

        tasks = data.get("tasks", [])
        expected_runs = {
            (task["id"], trial)
            for task in tasks
            for trial in range(self.autometric_runs_config.num_trials)
        }
        actual_runs = {
            (simulation.get("task_id"), simulation.get("trial"))
            for simulation in simulations
        }
        if (
            not expected_runs
            or actual_runs != expected_runs
            or len(simulations) != len(expected_runs)
        ):
            Logger.error(
                f"tau3-bench {domain}: incomplete or duplicate task/trial coverage "
                f"at {results_path}; expected {len(expected_runs)} simulations, "
                f"found {len(simulations)}"
            )
            return {}

        rewards: list[float] = []
        failed_simulations = 0
        for simulation in simulations:
            reward_info = simulation.get("reward_info")
            if (
                simulation.get("termination_reason") == "infrastructure_error"
                or not reward_info
                or reward_info.get("reward") is None
            ):
                failed_simulations += 1
                continue
            rewards.append(float(reward_info["reward"]))

        if failed_simulations:
            Logger.error(
                f"tau3-bench {domain}: {failed_simulations}/{len(simulations)} "
                f"simulations have infrastructure errors or no reward; "
                f"skipping incomplete domain at {results_path}"
            )
            return {}

        return {
            f"tau3/{domain}/mean@{self.autometric_runs_config.num_trials}": (
                sum(rewards) / len(rewards)
            )
        }

    @staticmethod
    def _normalize_metric_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9@]+", "_", value).strip("_").upper()

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return

        baseline = self._normalize_metric_name(
            self.autometric_runs_config.user_model_name
        )
        jd_metrics: dict[str, float] = {}
        missing_domains = set(self.autometric_runs_config.domains)
        domain_keys: dict[str, str] = {}
        for key, value in processed_metrics.items():
            domain = key.split("/")[1]
            missing_domains.discard(domain)
            jd_key = f"{self._normalize_metric_name(key)}_VS_{baseline}"
            if domain == "banking_knowledge":
                retrieval = self._normalize_metric_name(
                    self.autometric_runs_config.retrieval_config
                )
                jd_key = f"{jd_key}_RETRIEVAL_{retrieval}"
            jd_metrics[jd_key] = value
            domain_keys[domain] = jd_key

        headline = f"TAU3_MEAN@{self.autometric_runs_config.num_trials}_VS_{baseline}"
        if not missing_domains:
            jd_metrics[headline] = sum(jd_metrics.values()) / len(jd_metrics)
        scores = self._cache_scores(
            jd_metrics,
            domain_keys,
            headline,
            extra_identity={"retrieval": self.autometric_runs_config.retrieval_config},
        )
        JD.send_evaluator_metrics_with_scores(
            scores=scores,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            shooter_name="tau3",
            shooter_version=JD.get_shooter_version(
                str(self.tau2_root / "tau2_version.txt")
            ),
            model_info=self.model_info,
            submetric_groups=[]
            if missing_domains
            else [
                SubmetricGroup(
                    key=headline,
                    members=list(domain_keys.values()),
                    value=headline,
                )
            ],
        )
