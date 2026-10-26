import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...config_model import (
    ClawEvalEvaluatorConfig,
    ModelInfoConfig,
)
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import ScoringMetricsEvaluator
from ...subprocess_manager import SubprocessManager

_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_JD_JUDGE_SLUG_LIMIT = 12
_JD_METRIC_RE = re.compile(
    r"(?:^|/)(?:avg_score|pass_hat_\d+|pass_at_\d+"
    r"|n_skipped_web_real|n_skipped_no_user_sim|n_skipped_no_sandbox"
    r"|rollout_error_rate|judge_error_rate)$"
)


class ClawEvalEvaluator(ScoringMetricsEvaluator):
    LATENCY = True

    SHOOTER_NAME = "claweval"
    autometric_runs_config: ClawEvalEvaluatorConfig

    @property
    def checkpoint_model_name(self) -> str:
        _, checkpoint_model_name, _, _ = JD.parse_checkpoint_path(
            self.hf_checkpoint_path
        )
        return checkpoint_model_name

    def setup(
        self,
        autometric_runs_config: ClawEvalEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        if autometric_runs_config.model.name is None:
            autometric_runs_config.model.name = self.checkpoint_model_name or "model"
        self._create_metrics_dir(self.artifacts_path, self.SHOOTER_NAME)
        autometric_runs_config.out_dir = self.metrics_dir
        self._runner_root = Path(autometric_runs_config.runner_root)
        assert self.autometric_runs_config is not None
        assert self.metrics_dir is not None
        Logger.info(
            "ClawEvalEvaluator setup successfully, ready for inference/scoring phases"
        )

    def _build_overrides(self) -> dict[str, Any]:
        return self.autometric_runs_config.model_dump(mode="json", exclude_none=True)

    def run_phase(self, phase: str) -> None:
        runner_script = self._runner_root / "run_claweval.py"
        if not runner_script.exists():
            raise RuntimeError(
                f"claweval: runner not found in the image: {runner_script}"
            )
        config_fd, config_path = tempfile.mkstemp(prefix="claweval_", suffix=".json")
        try:
            os.fchmod(config_fd, 0o600)
            with os.fdopen(config_fd, "w", encoding="utf-8") as config_file:
                json.dump(self._build_overrides(), config_file, default=str)
            cmd = [
                self.autometric_runs_config.python_path,
                str(runner_script),
                "--phase",
                phase,
                "--overrides-file",
                config_path,
            ]
            Logger.info(f"ClawEval run phase={phase}")
            sm = SubprocessManager(
                cmd,
                Logger.autometrics_log_file,
                str(self._runner_root),
                f"ClawEval {phase}",
            )
            sm.start()
            sm.wait_for_completion()
            sm.log_error()
            if phase == "generation" and not sm.is_successful():
                raise RuntimeError(
                    f"claweval: generation failed with exit code {sm.exit_code}"
                )
        finally:
            Path(config_path).unlink(missing_ok=True)
        Logger.info(f"ClawEval phase={phase} completed")

    def get_metrics(self) -> dict[str, float]:
        if self.metrics_dir is None:
            raise RuntimeError("claweval: metrics_dir is not set")
        metrics_path = Path(self.metrics_dir) / "claweval_metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(
                f"claweval: the evaluation phase produced no metrics at {metrics_path}"
            )
        try:
            raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"Failed to read {metrics_path}: {e}") from e
        result = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
        if not result:
            raise RuntimeError(f"claweval: no numeric metrics in {metrics_path}")
        Logger.info(f"Collected {len(result)} ClawEval metrics")
        return result

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

    @staticmethod
    def _normalize_metric_name_for_jd(metric_name: str) -> str:
        return _NON_ALNUM_RE.sub("_", metric_name).strip("_").upper()

    @staticmethod
    def _shooter_version() -> str:
        path = Path("/home/user/claweval/upstream_commit.txt")
        try:
            pin = path.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise RuntimeError(
                f"claweval: cannot read the upstream pin from the image: {path}: {e}"
            ) from e
        if not pin:
            raise RuntimeError(f"claweval: the upstream pin file is empty: {path}")
        return f"vendored-{pin[:12]}"

    def _jd_metric_name(self, metric_key: str) -> str:
        scope = [part for part in metric_key.split("/") if part != self.SHOOTER_NAME]
        judge_slug = self._model_slug(self.autometric_runs_config.judge.model_name)
        judge = judge_slug[:_JD_JUDGE_SLUG_LIMIT].rstrip("_")
        name = "_".join(
            (
                self.SHOOTER_NAME,
                *scope,
                self.autometric_runs_config.search_backend,
                judge,
            )
        )
        return self._normalize_metric_name_for_jd(name)

    def _convert_metrics_to_jd_keys(
        self, processed_metrics: dict[str, float]
    ) -> dict[str, float]:
        return {
            self._jd_metric_name(name): value
            for name, value in processed_metrics.items()
            if _JD_METRIC_RE.search(name)
        }

    def _cache_scores(self, metrics: dict[str, float]) -> ScalarCacheScoresBuilder:
        scores = ScalarCacheScoresBuilder(self.metrics_dir, metrics)
        try:
            manifest = json.loads(
                (self.metrics_dir / "generation_manifest.json").read_text()
            )
            rows = json.loads((self.metrics_dir / "claweval_results.json").read_text())
            trials = manifest["trials"]
            by_task = {uid: {} for uid in manifest["selected"]}
            for row in rows:
                attempts = by_task[row["task_id"]]
                index = row["trial_index"]
                if index in attempts or row["status"] != "graded":
                    Logger.warning(
                        "ClawEval CI unavailable: duplicate or ungraded trial"
                    )
                    return scores
                attempts[index] = row
            if not by_task or any(
                set(items) != set(range(trials)) for items in by_task.values()
            ):
                Logger.warning("ClawEval CI unavailable: incomplete task/trial matrix")
                return scores
            cfg = self.autometric_runs_config
            identity = {
                "content": manifest["content_sha256"],
                "trials": trials,
                "split_of": manifest["split_of"],
                "search_backend": cfg.search_backend,
                "model": cfg.model.model_dump(exclude={"name", "api_base", "api_key"}),
                "judge": cfg.judge.model_dump(
                    exclude={"api_base", "api_key", "concurrency"}
                ),
                "user_model": cfg.user_model.model_dump(
                    exclude={"api_base", "api_key"}
                ),
                "env": cfg.env,
            }
            scopes = {"claweval": list(by_task)}
            for split in set(manifest["split_of"].values()):
                scopes[f"claweval/{split}"] = [
                    uid for uid in by_task if manifest["split_of"][uid] == split
                ]
            for prefix, ids in scopes.items():
                for metric in ("avg_score", f"pass_hat_{trials}", f"pass_at_{trials}"):
                    samples = []
                    for uid in ids:
                        attempts = list(by_task[uid].values())
                        if metric == "avg_score":
                            value = sum(row["task_score"] for row in attempts) / trials
                        elif metric.startswith("pass_hat_"):
                            value = float(all(row["passed"] for row in attempts))
                        else:
                            value = float(any(row["passed"] for row in attempts))
                        samples.append((uid, value))
                    scores.add_metric(
                        self._jd_metric_name(f"{prefix}/{metric}"), samples, identity
                    )
        except (OSError, ValueError, KeyError, TypeError) as error:
            scores.records.clear()
            Logger.warning(f"ClawEval CI unavailable: {error}")
        return scores

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return

        metrics_for_jd = self._convert_metrics_to_jd_keys(processed_metrics)
        JD.send_evaluator_metrics_with_scores(
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            scores=self._cache_scores(metrics_for_jd),
            shooter_name=self.SHOOTER_NAME,
            shooter_version=self._shooter_version(),
            model_info=self.model_info,
        )
