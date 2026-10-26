import json
import time
from pathlib import Path
from typing import Optional

from ...config_model import ModelInfoConfig, Tau2EvaluatorConfig
from ...job_dispatcher import JD, SubmetricGroup
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import (
    ParallelInferenceEvaluator,
)
from ...subprocess_pool_runner import SubprocessPoolRunner


class Tau2Evaluator(ParallelInferenceEvaluator):
    LATENCY = True

    PRETRAIN_ROOT = Path(__file__).resolve().parents[4]

    @staticmethod
    def _resolve_bench_root(bench_root: str) -> Path:
        root = Path(bench_root)
        if not root.is_absolute():
            root = Tau2Evaluator.PRETRAIN_ROOT / root
        return root

    def _build_commands_for_domains(
        self, cfg: Tau2EvaluatorConfig
    ) -> dict[str, list[str]]:
        save_to_by_domain: dict[str, str] = {}
        commands: dict[str, list[str]] = {}
        run_ts = int(time.time())
        for domain in cfg.domains:
            save_to = str(self.metrics_dir / f"pam_{domain}_{run_ts}")
            save_to_by_domain[domain] = save_to
            commands[domain] = [
                cfg.venv_python,
                "-m",
                "tau2.cli",
                "run",
                *cfg.build_cli_run(domain=domain, save_to=save_to),
            ]
        self._save_to_by_domain = save_to_by_domain
        return commands

    def setup(
        self,
        autometric_runs_config: Tau2EvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self.all_metrics: dict[str, float] = {}
        self._create_metrics_dir(self.artifacts_path, "tau2")

        self.tau2_root = self._resolve_bench_root(
            autometric_runs_config.tau2_bench_root
        )
        self.tau2_src = self.tau2_root / "src"
        self.cmds_to_run = self._build_commands_for_domains(self.autometric_runs_config)

        Logger.info(
            f"Tau2Evaluator setup: domains={autometric_runs_config.domains}, "
            f"user_model={autometric_runs_config.user_model_name}, "
            f"agent_api_base={autometric_runs_config.agent_api_base}, "
            f"user_api_base={autometric_runs_config.user_api_base}, "
            f"max_parallel_domains={autometric_runs_config.max_parallel_domains}"
        )

    def _run_domains(self) -> None:
        Logger.info(f"Starting tau2-bench runs in {self.tau2_src}")
        SubprocessPoolRunner(
            cmds_to_run=self.cmds_to_run,
            max_parallel=self.autometric_runs_config.max_parallel_domains,
            autometrics_log_file=self.log_file,
            cwd=str(self.tau2_src),
        ).run_all()

    def run_on_model_name(self, model_name: str) -> None:
        self._run_domains()
        for domain, save_to in self._save_to_by_domain.items():
            results_path = Path(save_to) / "results.json"
            if results_path.exists():
                self.all_metrics.update(
                    self._parse_mean_reward_from_results(results_path, domain)
                )
            else:
                Logger.warning(f"tau2-bench results not found at {results_path}")

    def get_metrics(self) -> dict[str, float]:
        return self.all_metrics

    @staticmethod
    def _parse_mean_reward_from_results(
        results_path: Path, domain: str
    ) -> dict[str, float]:
        with open(results_path, "r") as f:
            data = json.load(f)

        simulations = data.get("simulations", [])
        if not simulations:
            Logger.warning("No simulations found in results")
            return {}

        rewards: list[float] = []
        for sim in simulations:
            reward = 0.0
            reward_info = sim.get("reward_info")
            if reward_info and reward_info.get("reward") is not None:
                reward = float(reward_info["reward"])
            rewards.append(reward)

        avg_reward = sum(rewards) / len(rewards)
        return {f"tau/{domain}/mean": avg_reward}

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        self._send_metrics_to_jd(metrics)
        tar_path = JD.tar_metrics_dir(
            self.metrics_dir, self.artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            tar_path, self.hf_checkpoint_path, type(self).__name__
        )

    def _cache_scores(
        self,
        metrics: dict[str, float],
        domain_keys: dict[str, str],
        headline: str,
        extra_identity: object = None,
    ) -> ScalarCacheScoresBuilder:
        scores = ScalarCacheScoresBuilder(self.metrics_dir, metrics)
        cfg = self.autometric_runs_config
        samples_by_domain: dict[str, list[tuple[str, float]]] = {}
        identities = {}
        for domain, key in domain_keys.items():
            try:
                path = Path(self._save_to_by_domain[domain]) / "results.json"
                data = json.loads(path.read_text())
                tasks = {str(task["id"]): task for task in data["tasks"]}
                info = data["info"]
                valid = len(tasks) == len(data["tasks"])
                trials: dict[str, dict[int, float]] = {uid: {} for uid in tasks}
                for simulation in data["simulations"]:
                    uid = str(simulation["task_id"])
                    trial = simulation["trial"]
                    if trial in trials[uid] or trial is None:
                        valid = False
                        break
                    reward_info = simulation.get("reward_info") or {}
                    reward = reward_info.get("reward")
                    trials[uid][trial] = float(reward) if reward is not None else 0.0
                if (
                    not valid
                    or not trials
                    or any(len(values) != cfg.num_trials for values in trials.values())
                ):
                    Logger.warning(
                        f"tau CI unavailable for {domain}: incomplete task/trial population"
                    )
                    continue
                trial_ids = {tuple(sorted(values)) for values in trials.values()}
                if len(trial_ids) != 1:
                    Logger.warning(
                        f"tau CI unavailable for {domain}: inconsistent trial identities"
                    )
                    continue
            except (OSError, ValueError, KeyError, TypeError) as error:
                Logger.warning(f"tau CI unavailable for {domain}: {error}")
                continue
            samples = [
                (f"{domain}:{uid}", sum(values.values()) / cfg.num_trials)
                for uid, values in trials.items()
            ]
            identity = {
                "tasks": tasks,
                "trials": sorted(next(iter(trial_ids))),
                "seed": info.get("seed"),
                "max_steps": info.get("max_steps"),
                "max_errors": info.get("max_errors"),
                "user_model": cfg.user_model_name,
                "user_sampling": cfg.user_llm_args.model_dump(),
                "agent_sampling": cfg.agent_llm_args.model_dump(),
                "assertions_model": cfg.nl_assertions_llm,
                "extra": extra_identity,
            }
            scores.add_metric(key, samples, identity)
            samples_by_domain[domain] = samples
            identities[domain] = identity
        if set(samples_by_domain) == set(domain_keys) and samples_by_domain:
            scores.add_weighted_metric(
                headline,
                samples_by_domain,
                {domain: 1 / len(domain_keys) for domain in domain_keys},
                identities,
            )
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
            key.replace("/", "_").upper(): value
            for key, value in processed_metrics.items()
        }
        headline = f"TAU2_MEAN@{self.autometric_runs_config.num_trials}"
        jd_metrics[headline] = sum(jd_metrics.values()) / len(jd_metrics)
        domain_keys = {
            domain: f"TAU_{domain.upper()}_MEAN" for domain in self._save_to_by_domain
        }
        scores = self._cache_scores(
            jd_metrics,
            domain_keys,
            headline,
        )
        JD.send_evaluator_metrics_with_scores(
            scores=scores,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            shooter_name="tau2",
            shooter_version=JD.get_shooter_version(
                str(self.tau2_root / "tau2_version.txt")
            ),
            model_info=self.model_info,
            submetric_groups=[
                SubmetricGroup(
                    key=headline,
                    members=list(domain_keys.values()),
                    value=headline,
                )
            ],
        )
